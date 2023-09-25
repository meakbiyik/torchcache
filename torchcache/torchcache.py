"""This module implements the torchcache decorator and the underlying class."""
import atexit
import hashlib
import inspect
import io
import logging
import mmap
import shutil
import tempfile
from pathlib import Path
from typing import Type, Union

import torch
import zstd
from torch import Tensor

logger = logging.getLogger(__name__)


def torchcache(
    *,
    memory_cache_device: str = "cpu",
    subsample_count: int = 10000,
    persistent: bool = False,
    persistent_cache_dir: str = None,
    persistent_module_hash: str = None,
    max_persistent_cache_size: int = int(10e9),
    max_memory_cache_size: int = int(1e9),
    zstd_compression: bool = False,
    zstd_compression_level: int = 3,
    zstd_compression_threads: int = 1,
    cache_dtype: torch.dtype = None,
    use_mmap_on_load: bool = False,
) -> callable:
    r"""Decorate a nn.Module class to cache the output of the forward pass.

    Call this decorator on a nn.Module class to cache the output of the forward
    pass, given the same input and the same module definition.

    Always invoke the decorator with parentheses, even if no arguments are
    passed. For example:

    .. code-block:: python

        @torchcache()
        class CachedModule(nn.Module):
            pass

    You can also override the arguments of the underlying class instance by
    setting class attributes that starts with "torchcache\_". For example,
    to set the cache directory for a module (persistent_cache_dir), you can do:

    .. code-block:: python

        @torchcache(persistent=True)
        class CachedModule(nn.Module):
            def __init__(self, cache_dir: str | Path):
                self.torchcache_persistent_cache_dir = cache_dir

    Parameters
    ----------
    subsample_count : int
        Number of values to subsample from the tensor in hash computation,
        by default 10000. This is used to improve hashing performance,
        at the cost of a higher probability of hash collisions. Current
        default is 10000, which should be enough for most use cases.
    memory_cache_device : str or torch.device, optional
        Device to use for the cache, by default "cpu". If None, then the
        original device of the tensor is used.
    persistent : bool, optional
        Whether to use a file-system-based cache, by default False
    persistent_cache_dir : str or Path, optional
        Directory to use for caching, by default None. If None, then a temporary
        directory is used. Only used if persistent is True.
    persistent_module_hash : str, optional
        Hash of the module definition, args, and kwargs, by default None. If None,
        then the module hash is automatically determined. You can explicitly
        set this if you want to use the same cache for slightly different
        modules. You can find the module hash in the following locations:
        - In the logs, if you set the logging level to INFO or DEBUG
        - In the cached module's self.cache_instance.module_hash attribute
        - As the name of the subdirectory in the persistent cache
    max_persistent_cache_size : int, optional
        Maximum size of the persistent cache in bytes, by default 10e9 (10 GB)
    max_memory_cache_size : int, optional
        Maximum size of the memory cache in bytes, by default 1e9 (1 GB)
    zstd_compression : bool, optional
        Whether to use zstd compression, by default False. See
        https://github.com/sergey-dryabzhinsky/python-zstd for more information
        on the arguments below.
    zstd_compression_level : int, optional
        Compression level to use, by default 3. Must be between -100 and 22,
        where -100 is the fastest compression and 22 is the slowest.
    zstd_compression_threads : int, optional
        Number of threads to use for compression, by default 1. If 0, then the
        number of threads is automatically determined.
    cache_dtype : torch.dtype, optional
        Data type to use for the cache, by default None. If None, then the
        data type of the first tensor that is processed is used.
    use_mmap_on_load : bool, optional
        Whether to use mmap when loading the cached embeddings from file, by
        default False. This option might be useful if each embedding is very
        large, as it might improve the performance for large files.
    """
    # Multiple initialization of the same class shares the same cache
    cache_instance = None
    cache_kwargs = {
        "memory_cache_device": memory_cache_device,
        "subsample_count": subsample_count,
        "persistent": persistent,
        "persistent_cache_dir": persistent_cache_dir,
        "persistent_module_hash": persistent_module_hash,
        "max_persistent_cache_size": max_persistent_cache_size,
        "max_memory_cache_size": max_memory_cache_size,
        "zstd_compression": zstd_compression,
        "zstd_compression_level": zstd_compression_level,
        "zstd_compression_threads": zstd_compression_threads,
        "cache_dtype": cache_dtype,
        "use_mmap_on_load": use_mmap_on_load,
    }
    magic_prefix = "torchcache_"

    def decorator(ModuleClass):
        class WrappedModule(ModuleClass):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                logger.debug("Initializing torchcache")
                nonlocal cache_instance
                if cache_instance is None:
                    logger.debug("Initializing torchcache instance")
                    # Override the cache keyword arguments using the class properties
                    # that starts with torchcache_
                    for key in dir(self):
                        if key.startswith(magic_prefix):
                            logger.info(
                                f"Overriding {key[len(magic_prefix) :]} with "
                                f"{getattr(self, key)}"
                            )
                            cache_kwargs[key[len(magic_prefix) :]] = getattr(self, key)
                    logger.debug(f"Torchcache kwargs: {cache_kwargs}")
                    cache_instance = _TorchCache(**cache_kwargs)
                self.cache_instance = cache_instance
                self.cache_instance.wrap_module(self, ModuleClass, *args, **kwargs)
                __name__ = f"{ModuleClass.__name__}Cached"  # noqa: F841
                __qualname__ = f"{ModuleClass.__qualname__}Cached"  # noqa: F841
                logger.debug("Initialized torchcache")

        return WrappedModule

    return decorator


class _TorchCache:
    """Class that implements the caching logic.

    Do not initialize this class directly, use the torchcache decorator instead.
    """

    def __init__(
        self,
        *,
        memory_cache_device: str,
        subsample_count: int,
        persistent: bool,
        persistent_cache_dir: str,
        persistent_module_hash: str,
        max_persistent_cache_size: int,
        max_memory_cache_size: int,
        zstd_compression: bool,
        zstd_compression_level: int,
        zstd_compression_threads: int,
        cache_dtype: torch.dtype,
        use_mmap_on_load: bool,
    ):
        """Initialize the torchcache."""
        if not persistent and zstd_compression:
            raise ValueError("Cannot use zstd compression without persistent cache")

        # Rolling powers of the hash base, up until 2**15 to fit in float16
        roll_powers = torch.arange(0, subsample_count * 2) % 15
        self.subsample_count = subsample_count
        self.coefficients = (
            torch.pow(torch.tensor([2.0]), roll_powers).float().detach().view(1, -1)
        )
        self.coefficients.requires_grad_(False)

        self.zstd_compression = zstd_compression
        self.zstd_compression_level = zstd_compression_level
        self.zstd_compression_threads = zstd_compression_threads
        self.persistent = persistent
        self.memory_cache_device = memory_cache_device
        self.cache: dict[int, Tensor] = {}
        self.max_memory_cache_size = max_memory_cache_size
        self.memory_cache_size = 0
        self.is_memory_cache_full = False
        self.cache_dtype = cache_dtype
        self.use_mmap_on_load = use_mmap_on_load

        if self.persistent:
            logger.debug("Initializing persistent cache")
            self.cache_parent_dir = (
                Path(persistent_cache_dir) if persistent_cache_dir is not None else None
            )
            logger.info(f"Torchcache parent dir: {self.cache_parent_dir}")
            self.max_persistent_cache_size = max_persistent_cache_size
            self.persistent_cache_size = 0
            self.is_persistent_cache_full = False
            if self.cache_parent_dir is None:
                self.cache_parent_dir = Path(tempfile.mkdtemp())
                atexit.register(self.cache_cleanup)

        logger.debug(f"Params: {self.__dict__}")

        # Runtime-stored variables for the current batch
        self.current_dtype = None
        self.current_embeddings = None
        self.current_hashes = None
        self.current_indices_to_embed = None
        self.current_skip_forward = False

        # Overridden in wrap_module if None
        self.module_hash: str = persistent_module_hash
        if persistent_module_hash is not None:
            logger.warning(
                f"Overriding module hash: {self.module_hash}. "
                "This might cause you quite a bit of headache if you are not "
                "careful, since any changes to the module definition, args, "
                "or kwargs will not be reflected in the cache."
            )

    def cache_cleanup(self):
        logger.info(f"Cleaning up the persistent cache in {self.cache_parent_dir}")
        shutil.rmtree(self.cache_parent_dir, ignore_errors=True)
        logger.info(f"Deleted cache dir: {self.cache_parent_dir}")

    def forward_pre_hook(self, module, inputs):
        """Forward pre-hook to check the cache.

        Takes in arbitrary number of tensors with shape (B, *), hashes
        them separately, and checks if the hash values are in the cache.
        If so, we store them temporarily, and forward the rest of the
        tensors to the module.

        In the post-hook, we will combine the cached embeddings with the
        newly computed embeddings.

        Parameters
        ----------
        module : nn.Module
            Module to hook.
        inputs : Tuple[Tensor]
            Inputs to the module.
        """
        logger.debug(
            f"Forward pre-hook input shapes: {[input.shape for input in inputs]}"
        )
        flattened_inputs = [input.flatten(1) for input in inputs]
        concatenated_inputs = torch.cat(flattened_inputs, dim=1)
        self.current_dtype = concatenated_inputs.dtype
        self.cache_dtype = self.cache_dtype if self.cache_dtype else self.current_dtype

        self.current_hashes = self.hash_tensor(concatenated_inputs)

        (
            self.current_indices_to_embed,
            self.current_embeddings,
        ) = self._fetch_cached_embeddings()

        if self.current_indices_to_embed.shape[0] > 0:
            logger.debug("Forwarding the rest of the inputs")
            inputs_to_embed = [
                input[self.current_indices_to_embed].view(-1, *input.shape[1:])
                for input in inputs
            ]
            return tuple(inputs_to_embed)
        else:
            logger.debug("Skipping forward pass")
            self.current_skip_forward = True
            return None

    def forward_hook(self, module, inputs, outputs):
        """Forward post-hook to replace and cache the embeddings.

        Takes the output tensor, replaces the embeddings for the ones
        we have in cache, and caches the newly computed embeddings.

        Parameters
        ----------
        module : nn.Module
            Module to hook.
        inputs : Tuple[Tensor]
            Inputs to the module.
        outputs : Tensor
            Outputs from the module.
        """
        logger.debug(f"Forward hook outputs shape: {outputs.shape}")
        # If forward pass was skipped, restore the flag and just return
        if self.current_skip_forward:
            logger.debug("Forward pass was skipped, restoring the flag")
            self.current_skip_forward = False
            return

        self._cache_embeddings(
            outputs.to(self.cache_dtype),
            self.current_hashes[self.current_indices_to_embed],
        )

        if self.current_embeddings is None:
            # This is the first forward pass, so we do not
            # need to combine the embeddings
            logger.debug("First forward pass, returning the embeddings")
            return outputs
        else:
            # Add the newly computed embeddings to the rest
            embedding_to_return = self.current_embeddings[
                : self.current_hashes.shape[0]
            ].to(self.current_dtype)
            embedding_to_return[self.current_indices_to_embed] = outputs

            return embedding_to_return

    def wrap_module(
        self,
        module: torch.nn.Module,
        moduleClass: Type[torch.nn.Module],
        *args,
        **kwargs,
    ):
        """Wrap a nn.Module with the pre-hook and the post-hook.

        This method wraps a given nn.Module, adding the
        pre-hook and the post-hook, as well as overriding the
        forward method.
        """
        # Ensure that there are no parameters that require gradients in the module
        for param in module.parameters():
            assert not param.requires_grad, (
                "TorchCache does not support modules with parameters that require "
                "gradients."
            )

        logger.debug("Wrapping the module and registering hooks")
        module.register_forward_pre_hook(self.forward_pre_hook)
        module.register_forward_hook(self.forward_hook)
        if self.module_hash is None:
            # Create a hash of the module definition, args, and kwargs
            # So that we do not mistakenly use the cache for a different module
            logger.debug("Creating module hash")
            try:
                module_definition = inspect.getsource(moduleClass)
                hash_string = module_definition + repr(args) + repr(kwargs)
            except OSError as e:
                logger.error(f"Could not retrieve the module source: {e}")
                # If the module source cannot be retrieved, we use the module name
                hash_string = module.__class__.__name__ + repr(args) + repr(kwargs)
            # Also add the crucial parameters of torchcache
            hash_string += repr(self.subsample_count) + repr(self.zstd_compression)
            logger.debug(f"Module hash string: {hash_string}")
            self.module_hash = hashlib.blake2b(
                hash_string.encode(),
                digest_size=32,
            ).hexdigest()
        logger.info(f"Module hash: {self.module_hash}")
        # If we are using a persistent cache, create a subdirectory for the module
        if self.persistent:
            self.cache_dir = self.cache_parent_dir / str(self.module_hash)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Wrap the forward method so that we can skip it if needed
        current_original_forward = module.forward

        def forward_wrapper(*args, **kwargs):
            if self.current_skip_forward:
                return self.current_embeddings[: self.current_hashes.shape[0]].to(
                    self.current_dtype
                )
            else:
                return current_original_forward(*args, **kwargs)

        module.forward = forward_wrapper

        return module

    def _fetch_cached_embeddings(
        self,
    ) -> Tensor:
        """Fetch cached embeddings for the tensors that have already been processed.

        Returns
        -------
        Tensor
            Indices of the tensors that need to be embedded.
        Tensor
            Embeddings of the tensors that have already been embedded.
        """
        embeddings = self.current_embeddings
        indices_to_embed = []

        if embeddings is not None:
            logger.debug("Detaching the embeddings")
            embeddings = embeddings.detach()

        for i, hash_val in enumerate(self.current_hashes):
            int_hash_val = hash_val.item()
            embedding = self._load_from_memory(int_hash_val)
            if embedding is None and self.persistent:
                logger.debug(
                    f"Hash value: {int_hash_val} not in memory, "
                    "attempting to load from file"
                )
                embedding = self._load_from_file(int_hash_val)

            if embedding is None:
                logger.debug(
                    f"Cache miss for hash value: {int_hash_val}, embedding index {i}"
                )
                indices_to_embed.append(i)
            else:
                logger.debug(f"Cache hit for hash value: {int_hash_val}")
                if (
                    embeddings is None
                    or embeddings.shape[0] < self.current_hashes.shape[0]
                    or embedding.shape != embeddings[i].shape
                ):
                    if embeddings is None:
                        logger.info("Embeddings is None, initializing")
                    elif embedding.shape != embeddings[i].shape:
                        logger.warn(
                            f"Embedding shape mismatch: {embedding.shape} vs "
                            f"{embeddings[i].shape}. Resizing to "
                            f"{self.current_hashes.shape[0]}"
                        )
                    else:
                        logger.info(
                            f"Embeddings is too small ({embeddings.shape[0]}), "
                            f"resizing to {self.current_hashes.shape[0]}"
                        )
                    embeddings = torch.empty(
                        self.current_hashes.shape[0],
                        *embedding.shape,
                        dtype=self.cache_dtype,
                        device=self.current_hashes.device,
                    )
                embeddings[i] = embedding.to(embeddings)

        indices_to_embed = torch.tensor(
            indices_to_embed,
            dtype=torch.long,
            device=self.current_hashes.device,
        )
        logger.debug(f"Indices to embed: {indices_to_embed}")

        return indices_to_embed, embeddings

    def _cache_embeddings(
        self,
        embeddings_to_cache: Tensor,
        embedding_hashes: Tensor,
    ) -> None:
        """Cache the embeddings of the images that have been processed.

        Parameters
        ----------
        embeddings_to_cache : Tensor
            Embeddings of the images that have been processed, with shape (B, D).
        embedding_hashes : Tensor
            Hash values of the images.
        """
        logger.debug(
            f"Caching the embeddings with shape: {embeddings_to_cache.shape} "
            f"for hashes: {embedding_hashes}"
        )
        embedding_hashes = embedding_hashes.detach()
        for i, hash_val in enumerate(embedding_hashes):
            embedding_to_cache = embeddings_to_cache[i].clone()
            int_hash_val = hash_val.item()
            if self.persistent:
                self._cache_to_file(embedding_to_cache, int_hash_val)
            self._cache_to_memory(embedding_to_cache, int_hash_val)

    def hash_tensor(self, tensor: Tensor) -> Tensor:
        """Hashes a tensor.

        Parameters
        ----------
        tensor : Tensor
            Tensor to hash, with shape (B, *).

        Returns
        -------
        Tensor
            Hash values of the tensor.
        """
        logger.debug(f"Hashing the tensor with shape: {tensor.shape}")
        self.coefficients = self.coefficients.to(tensor.device)

        subsample_rate = max(1, tensor.shape[1] // self.subsample_count)
        logger.debug(f"Subsample rate: {subsample_rate}")
        tensor = tensor[:, ::subsample_rate]
        logger.debug(f"Subsampled tensor shape: {tensor.shape}")

        # in mixed precision, the matmul operation returns float16 values,
        # which overflows
        hash_val = torch.sum(
            tensor * self.coefficients[:, : tensor.shape[1]],
            dim=1,
            dtype=torch.long,
        )

        return hash_val

    def _cache_to_memory(self, embedding: Tensor, hash_val: int) -> None:
        """Cache the embedding in memory."""
        logger.debug(
            f"Caching embedding with shape: {embedding.shape} "
            f"to memory with hash value: {hash_val}"
        )
        if self.is_memory_cache_full:
            logger.info("Memory cache is full, skipping caching to memory")
            return

        embedding_size = embedding.element_size() * embedding.nelement()
        if self.memory_cache_size + embedding_size < self.max_memory_cache_size:
            if self.memory_cache_device is not None:
                embedding = embedding.to(self.memory_cache_device)
            self.cache[hash_val] = embedding
            self.memory_cache_size += embedding_size
            logger.debug(f"New memory cache size: {self.memory_cache_size}")
        else:
            logger.warning("Memory cache is full, skipping caching to memory")
            self.is_memory_cache_full = True

    def _load_from_memory(self, hash_val: int) -> Union[Tensor, None]:
        """Load the cached embedding from memory."""
        logger.debug(f"Loading from memory with hash value: {hash_val}")
        if hash_val in self.cache:
            return self.cache[hash_val]
        else:
            logger.debug("Hash value not in memory")
            return None

    def _cache_to_file(self, embedding: Tensor, hash_val: int) -> None:
        """Cache the embedding to a file, optionally using zstd compression."""
        logger.debug(
            f"Caching embedding with shape: {embedding.shape} to file "
            f"with hash value: {hash_val}"
        )
        if self.is_persistent_cache_full:
            logger.info("Persistent cache is full, skipping caching to file")
            return

        file_path: Path = self.cache_dir / f"{hash_val}.pt.br"
        logger.debug(f"File path to cache for hash value {hash_val}: {file_path}")
        if file_path.exists():
            logger.warning(
                f"File {file_path} already exists, skipping caching to file. "
                "This should not happen, please report this as an issue on Github"
            )
            return

        buffer = io.BytesIO()
        torch.save(embedding, buffer)
        raw_data = buffer.getvalue()
        logger.debug(f"Raw data size: {len(raw_data)}")

        if self.zstd_compression:
            raw_data = zstd.compress(
                raw_data,
                self.zstd_compression_level,
                self.zstd_compression_threads,
            )
            logger.debug(f"Compressed data size: {len(raw_data)}")

        if self.persistent_cache_size + len(raw_data) > self.max_persistent_cache_size:
            logger.warning("Persistent cache is full, skipping caching to file")
            self.is_persistent_cache_full = True
            return

        with open(file_path, "wb") as f:
            f.write(raw_data)

        self.persistent_cache_size += len(raw_data)
        logger.debug(f"New persistent cache size: {self.persistent_cache_size}")

    def _load_from_file(self, hash_val: int) -> Union[Tensor, None]:
        """Load the cached embedding from a file, maybe after decompression."""
        file_path: Path = self.cache_dir / f"{hash_val}.pt.br"
        logger.debug(f"Loading from file {file_path} with hash value: {hash_val}")

        if not file_path.exists():
            logger.debug("File does not exist")
            return None

        load_kwargs = {
            "map_location": self.current_hashes.device,
            "weights_only": False,  # True causes a huge performance hit
        }

        try:
            with open(file_path, "rb") as f:
                f = (
                    mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    if self.use_mmap_on_load
                    else f
                )
                raw_data = f.read()

        except Exception as e:
            logger.error(
                f"Could not read file {file_path}, "
                f"skipping loading from file. Error: {e}\n"
                "Removing the file to avoid future errors."
            )
            file_path.unlink(missing_ok=True)
            return None

        if self.zstd_compression:
            try:
                raw_data = zstd.decompress(raw_data)
            except Exception as e:
                logger.error(
                    f"Could not decompress file {file_path}, "
                    f"skipping loading from file. Error: {e}\n"
                    "Removing the file to avoid future errors."
                )
                file_path.unlink(missing_ok=True)
                return None

        buffer = io.BytesIO(raw_data)
        embedding = torch.load(buffer, **load_kwargs)

        logger.debug("Caching to memory before returning")
        self._cache_to_memory(embedding, hash_val)

        return embedding
