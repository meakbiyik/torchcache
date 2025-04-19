"""Tests for the caching module."""

import os

import pytest
import torch
import torch.nn as nn

from torchcache import torchcache
from torchcache.torchcache import _TorchCache


# Test Module
class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()

    def forward(self, x):
        return x * 2


# Test basic functionality of the caching mechanism without persistent storage.
def test_basic_caching():
    @torchcache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # Assert that the cache is empty
    assert len(model.cache_instance.cache) == 0

    # First pass, caching should occur
    output = model(input_tensor)
    assert torch.equal(output, input_tensor * 2)

    # Assert that the cache is not empty
    assert len(model.cache_instance.cache) == input_tensor.shape[0]

    # Check that the hash of the input tensor is in the cache
    hashes = model.cache_instance.hash_tensor(input_tensor)
    for hash in hashes:
        assert hash.item() in model.cache_instance.cache

    # Second pass, should retrieve from cache but result should be the same
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached)

    # Third time is the charm, but let's use a bigger batch size
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached[:2])

    with pytest.raises(ValueError):

        @torchcache(persistent=False, zstd_compression=True)
        class CachedModule(SimpleModule):
            pass

        CachedModule()


# Test caching mechanism with persistent storage.
def test_persistent_caching(tmp_path):
    @torchcache(persistent=True, persistent_cache_dir=tmp_path)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # First pass, caching should occur and save to file
    output = model(input_tensor)
    assert torch.equal(output, input_tensor * 2)

    # Check if cache files were created
    assert len(list((tmp_path / model.cache_instance.module_hash).iterdir())) == 2

    # Second pass, should retrieve from cache from memory
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached)

    # Now create a new instance of the model and check if the cache is loaded from disk
    # We re-define the class to flush the cache
    @torchcache(persistent=True, persistent_cache_dir=tmp_path)
    class CachedModule(SimpleModule):
        pass

    model2 = CachedModule()
    original_load_from_file = model2.cache_instance._load_from_file
    model2.cache_instance.original_load_from_file = original_load_from_file
    load_from_file_called = False

    def _load_from_file(*args, **kwargs):
        nonlocal load_from_file_called
        load_from_file_called = True
        return original_load_from_file(*args, **kwargs)

    model2.cache_instance._load_from_file = _load_from_file
    output_cached = model2(input_tensor)
    assert torch.equal(output, output_cached)
    assert load_from_file_called


# Test temporary cachedir
def test_persistent_caching_temporary_cachedir():
    @torchcache(persistent=True)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    temporary_path = model.cache_instance.cache_dir

    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # First pass, caching should occur and save to file
    output = model(input_tensor)
    assert torch.equal(output, input_tensor * 2)

    # Check if cache files were created
    assert len(list(temporary_path.iterdir())) == 2

    # Second pass, should retrieve from cache but result should be the same
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached)


# Test hashing functionality.
def test_hashing():
    cache = _TorchCache(
        memory_cache_device="cpu",
        subsample_count=10000,
        persistent=False,
        persistent_cache_dir=None,
        persistent_module_hash=None,
        max_persistent_cache_size=int(10e9),
        max_memory_cache_size=int(1e9),
        zstd_compression=False,
        zstd_compression_level=3,
        zstd_compression_threads=1,
        cache_dtype=None,
        use_mmap_on_load=False,
    )
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    hashes = cache.hash_tensor(input_tensor)

    assert hashes.shape[0] == input_tensor.shape[0]


def test_compression(tmp_path):
    @torchcache(persistent=True, persistent_cache_dir=tmp_path, zstd_compression=True)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # First pass, caching should occur and save to file
    output = model(input_tensor)
    assert torch.equal(output, input_tensor * 2)

    # Check if cache files were created
    assert len(list((tmp_path / model.cache_instance.module_hash).iterdir())) == 2

    # Second pass, should retrieve from cache from memory
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached)

    # Now create a new instance of the model and check if the cache is loaded from disk
    # We re-define the class to flush the cache in memory
    @torchcache(persistent=True, persistent_cache_dir=tmp_path, zstd_compression=True)
    class CachedModule(SimpleModule):
        pass

    model2 = CachedModule()
    original_load_from_file = model2.cache_instance._load_from_file
    model2.cache_instance.original_load_from_file = original_load_from_file
    load_from_file_called = False

    def _load_from_file(*args, **kwargs):
        nonlocal load_from_file_called
        load_from_file_called = True
        return original_load_from_file(*args, **kwargs)

    model2.cache_instance._load_from_file = _load_from_file
    output_cached = model2(input_tensor)
    assert torch.equal(output, output_cached)
    assert load_from_file_called


# Test cache size limits
def test_cache_size(tmp_path):
    # Overhead of saving a tensor in disk is around 1200 bytes
    @torchcache(
        persistent=True,
        persistent_cache_dir=tmp_path,
        max_persistent_cache_size=2500,
        max_memory_cache_size=20,
    )
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    input_tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

    # First pass, caching should occur and save to file
    output = model(input_tensor1)
    assert torch.equal(output, input_tensor1 * 2)

    # Check if cache files were created
    assert len(list((tmp_path / model.cache_instance.module_hash).iterdir())) == 2

    # Check that the persistent flag is not set, but the memory flag is
    assert not model.cache_instance.is_persistent_cache_full
    assert model.cache_instance.is_memory_cache_full

    # Now pass a tensor that is bigger than the cache size
    output = model(input_tensor2)
    assert torch.equal(output, input_tensor2 * 2)

    # Check if cache files were not created
    assert len(list((tmp_path / model.cache_instance.module_hash).iterdir())) == 2

    # Check that the flag is set
    assert model.cache_instance.is_persistent_cache_full


# Test for mixed cache hits
def test_mixed_cache_hits():
    @torchcache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    input_tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

    # First pass for caching
    model(input_tensor1)
    # Second pass where one tensor is cached and the other isn't
    combined_output = model(torch.cat([input_tensor1, input_tensor2], dim=0))

    assert torch.equal(combined_output[:2], input_tensor1 * 2)
    assert torch.equal(combined_output[2:], input_tensor2 * 2)


# Test monkey patching
def test_monkey_patching():
    @torchcache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    # First pass for caching
    model(input_tensor)
    # Monkey-patching should be applied here
    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)


def test_multiple_inputs():
    class DoubleInputModule(nn.Module):
        def forward(self, x, y):
            return x * 2 + y * 3

    @torchcache(persistent=False)
    class CachedModule(DoubleInputModule):
        pass

    model = CachedModule()
    input_tensor1 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    input_tensor2 = torch.tensor([[4, 5, 6]], dtype=torch.float32)

    output = model(input_tensor1, input_tensor2)

    assert torch.equal(output, input_tensor1 * 2 + input_tensor2 * 3)

    # Second pass, should retrieve from cache but result should be the same
    output = model(input_tensor1, input_tensor2)

    assert torch.equal(output, input_tensor1 * 2 + input_tensor2 * 3)


# Test different tensor shapes
def test_different_shapes():
    @torchcache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.randn((5, 5, 5, 5))

    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)


# Test overriding torchcache arguments with module attributes that starts with "torchcache_"
def test_override_torchcache_args():
    @torchcache(persistent=False)
    class CachedModule(SimpleModule):
        def __init__(self):
            super().__init__()
            self.torchcache_persistent = True

    model = CachedModule()
    input_tensor = torch.randn((5, 5, 5, 5))

    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)
    assert model.cache_instance.persistent


# Test training with forward and backward to ensure that
# there are no missing detach calls.
def test_training():
    @torchcache(persistent=False)
    class CachedModule(SimpleModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(5, 5)

        def forward(self, x):
            return x @ self.weight

    class TrainedModule(SimpleModule):
        def __init__(self):
            super().__init__()
            self.cached_module = CachedModule()
            self.weight = nn.Parameter(torch.randn(5, 5))

        def forward(self, x):
            return self.cached_module(x) @ self.weight

    model = TrainedModule()
    input_tensor = torch.randn((1, 5))

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    assert torch.equal(output, input_tensor @ model.cached_module.weight @ model.weight)
    assert model.weight.grad is not None

    # run again to ensure that the graph is sane
    input_tensor = torch.randn((1, 5))
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()


# Ensure that wrapping fails if there is a parameter that requires grad
def test_fails_with_parameter():
    @torchcache(persistent=False)
    class CachedModule(SimpleModule):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 5))

        def forward(self, x):
            return x @ self.weight

    with pytest.raises(AssertionError):
        model = CachedModule()


# Check using different dtype for caching
def test_different_dtype():
    @torchcache(persistent=True, cache_dtype=torch.half)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)

    # Second pass, should retrieve from cache but result should be the same
    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)


def test_persistent_module_hash(tmp_path):
    @torchcache(
        persistent=True, persistent_module_hash="test", persistent_cache_dir=tmp_path
    )
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)

    # Create another Module with the same persistent_module_hash but different forward pass
    @torchcache(
        persistent=True, persistent_module_hash="test", persistent_cache_dir=tmp_path
    )
    class CachedModule2(SimpleModule):
        def forward(self, x):
            return x * 3

    model2 = CachedModule2()

    # Second pass, should retrieve from cache but result should be the same
    # as the first module since the persistent_module_hash is the same
    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)


def test_mmap_on_load(tmp_path):
    @torchcache(
        persistent=True,
        persistent_cache_dir=tmp_path,
        use_mmap_on_load=True,
    )
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)

    # Second pass, should retrieve from cache but result should be the same
    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)


# Ensure that if one creates two instances of a module and
# only desires caching for one, they can disable caching via
# magic attribute torchcache_enabled
def test_duplicate_modules():
    @torchcache(persistent=False)
    class MyModule(SimpleModule):
        def __init__(self, cache=True):
            super().__init__()
            self.torchcache_enabled = cache

    cached_module = MyModule()
    noncached_module = MyModule(cache=False)

    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    # Assert that the cache is empty
    assert len(cached_module.cache_instance.cache) == 0
    # First pass, caching should occur
    output = cached_module(input_tensor)
    assert torch.equal(output, input_tensor * 2)
    # Assert that the cache is not empty
    assert len(cached_module.cache_instance.cache) == input_tensor.shape[0]

    # Assert that the other module has no cache
    assert not hasattr(noncached_module, "cache_instance")


# Test pure function decorator basic behavior
def test_pure_function_basic():
    calls = []

    @torchcache(persistent=False)
    def mul_fn(x):
        calls.append(x.clone())
        return x * 3

    input_tensor = torch.tensor([[1], [2]], dtype=torch.float32)
    out1 = mul_fn(input_tensor)
    assert torch.equal(out1, input_tensor * 3)
    # second call same input, should not re-execute function
    out2 = mul_fn(input_tensor)
    assert torch.equal(out2, input_tensor * 3)
    assert len(calls) == 1


# Test pure function with persistence and disk files
def test_pure_function_persistent(tmp_path):
    @torchcache(persistent=True, persistent_cache_dir=tmp_path)
    def add_fn(x):
        return x + 5

    input_tensor = torch.tensor([[3], [4]], dtype=torch.float32)
    out = add_fn(input_tensor)
    assert torch.equal(out, input_tensor + 5)

    # extract module instance from closure
    cache_mod = next(
        cell.cell_contents
        for cell in add_fn.__closure__
        if isinstance(cell.cell_contents, torch.nn.Module)
    )
    cache_dir = cache_mod.cache_instance.cache_dir
    assert cache_dir.exists()
    # should have one file per batch element
    assert len(list(cache_dir.iterdir())) == input_tensor.shape[0]

    # second call should use cache
    out2 = add_fn(input_tensor)
    assert torch.equal(out2, out)


def test_non_tensor_args_kwargs_affect_cache():
    @torchcache(persistent=False)
    class CachedScaleModule(SimpleModule):
        def forward(self, x, scale):
            return x * scale

    model = CachedScaleModule()
    input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

    # initial cache empty
    assert len(model.cache_instance.cache) == 0

    # call with positional non-tensor arg
    out1 = model(input_tensor, 2)
    assert torch.equal(out1, input_tensor * 2)
    assert len(model.cache_instance.cache) == input_tensor.shape[0]

    # call with different pos arg, should add new entries
    out2 = model(input_tensor, 3)
    assert torch.equal(out2, input_tensor * 3)
    assert len(model.cache_instance.cache) == 2 * input_tensor.shape[0]

    # repeat first call, cache size should not change
    out3 = model(input_tensor, 2)
    assert torch.equal(out3, input_tensor * 2)
    assert len(model.cache_instance.cache) == 2 * input_tensor.shape[0]

    # call with kwarg instead of positional
    out4 = model(input_tensor, scale=3)
    assert torch.equal(out4, input_tensor * 3)
    assert len(model.cache_instance.cache) == 2 * input_tensor.shape[0]


def test_pure_function_kwargs_affect_cache():
    calls = []

    @torchcache(persistent=False)
    def add_fn(x, offset):
        calls.append(offset)
        return x + offset

    input_tensor = torch.tensor([[1], [2]], dtype=torch.float32)

    # first call with offset=5
    out1 = add_fn(input_tensor, 5)
    assert torch.equal(out1, input_tensor + 5)

    # call with different offset, should compute and cache new values
    out2 = add_fn(input_tensor, 6)


def test_ensure_unique_function_cache(tmp_path):
    @torchcache(persistent=True, persistent_cache_dir=tmp_path)
    def add_fn(x):
        return x + 1

    @torchcache(persistent=True, persistent_cache_dir=tmp_path)
    def add_fn2(x):
        return x + 2

    input_tensor = torch.tensor([[1], [2]], dtype=torch.float32)
    out1 = add_fn(input_tensor)
    assert torch.equal(out1, input_tensor + 1)
    # second call same input, should not re-execute function
    out2 = add_fn(input_tensor)
    assert torch.equal(out2, input_tensor + 1)
    assert len(list((tmp_path / add_fn.cache_instance.module_hash).iterdir())) == 2

    # check that the two functions have different module hashes
    assert add_fn.cache_instance.module_hash != add_fn2.cache_instance.module_hash

    # check that the second function does not use the first function's cache
    out3 = add_fn2(input_tensor)
    assert torch.equal(out3, input_tensor + 2)
    assert len(list((tmp_path / add_fn2.cache_instance.module_hash).iterdir())) == 2


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Cleanup after all tests run, ensuring no cache files are left behind
    for f in os.listdir():
        if f.endswith(".pt.br"):
            os.remove(f)
