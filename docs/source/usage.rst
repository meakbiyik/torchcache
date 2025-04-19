Usage
=====

Cache the output of your PyTorch module effortlessly with a single decorator:

.. code-block:: python

   from torchcache import torchcache

    @torchcache()
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            # This output will be cached
            return self.linear(x)

Alternatively, cache the output of a torch-heavy function:

.. code-block:: python

   from torchcache import torchcache

    @torchcache()
    def my_function(x):
        # This output will be cached
        return x + 1

Limitations
-----------

`torchcache` is designed to cache the tensor-based batched inputs and outputs efficiently in a batch-agnostic way. However, it has some limitations:

- The module's forward method should return a single tensor.
- All input tensors should be on the same device and have the same dtype, with the same batch dimension.
- Only basic immutable Python types (int, str, float, boolean) are supported as input arguments in addition to tensors.

Persistent caching
------------------

For persistent caching, use the ``persistent`` argument. You have the chance to specify the cache directory with the ``persistent_cache_dir`` argument. Without it, the cache will be stored in a temporary directory and deleted after the program exits.

.. code-block:: python

   from torchcache import torchcache

    @torchcache(persistent=True, persistent_cache_dir="/path/to/cache")
    class MyModule(nn.Module):
        ...

Adjusting cache size
--------------------

Even when you enable persistent caching, `torchcache` will still store some data in memory, up to ``max_memory_cache_size`` bytes, by default 1 GB. In addition, the persistent cache will be limited to ``max_persistent_cache_size`` bytes, by default 10 GB. You can adjust these values with the corresponding arguments.

.. code-block:: python

   from torchcache import torchcache

    @torchcache(
        persistent=True,
        max_memory_cache_size=5e9, # 5 GB
        max_persistent_cache_size=50e9, # 50 GB
    )
    class MyModule(nn.Module):
        ...

Optimizing cache size
---------------------

If you want to optimize the cache size, you can play with compression quality and the dtype of cached tensor. By default, tensors are cached with the same dtype as the original tensor without compression. You can use ztsd compression, and change the ``cache_dtype``. Note that the compression quality is only used for persistent caching, and comes with implications on both the speed of caching and loading. For most cases, the default values should be fine.

.. code-block:: python

   from torchcache import torchcache

    @torchcache(
        persistent=True,
        cache_dtype=torch.half,
        zstd_compression=True,
    )
    class MyModule(nn.Module):
        ...

Setting parameters directly in the module
-----------------------------------------

You can also set the parameters directly inside the decorated module, in case the caching behavior somehow depends on the module's parameters. In this case, you can set attributes with the magic prefix ``torchcache_``. For example, if you want to set the cache dtype to ``torch.half`` and the compression quality to 11, you can do:

.. code-block:: python

   from torchcache import torchcache

    @torchcache(persistent=True)
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.torchcache_cache_dtype = torch.half
            self.torchcache_zstd_compression = True

        def forward(self, x):
            # This output will be cached
            return self.linear(x)

Note that the parameters set in the module take precedence over the parameters set in the decorator.

Selectively enabling `torchcache` for module instances
------------------------------------------------------

Using the magic prefix described above and the `enabled` argument, you can selectively enable or disable `torchcache` for different instances of the same class.

.. code-block:: python

   from torchcache import torchcache

    @torchcache()
    class MyModule(nn.Module):
        def __init__(self, caching_enabled):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.torchcache_enabled = caching_enabled

        def forward(self, x):
            return self.linear(x)

    module_with_caching = MyModule(caching_enabled=True)
    module_without_caching = MyModule(caching_enabled=False)

When is cache invalidated?
--------------------------

The cache is invalidated when:

- The module's code changes
- The module initialization arguments/keywork arguments change
- The torchcache parameters change

The invalidation does not remove the old cached items, it just skips them. They can also be, although highly unlikely, overwritten by a new cache with the same key. If you revert the changes, the cache will be valid again.

Note that updating `torchcache` might, in some cases, invalidate the cache. I will try to avoid this as much as possible, but cannot guarantee full backward compatibility, particularly in the early stages of the project.

Avoiding cache invalidation
---------------------------

If you created a cache with a certain configuration, and you want to avoid accidental cache invalidation and subsequently re-caching, you can set the module hash directly via the ``persistent_module_hash`` argument.

.. code-block:: python

   from torchcache import torchcache

    @torchcache(
        persistent=True,
        persistent_module_hash="my_module_hash",
    )
    class MyModule(nn.Module):
        ...

This will make sure that the cache is not invalidated if you change the module's code, initialization arguments, or keyword arguments. Be careful though, as this will also prevent the cache from being invalidated if you change the `torchcache` parameters, or if you change the module's code in a way that matters. In this case, you will have to delete the cache manually. I recommend using this option only if you are sure that the cache will not be invalidated, otherwise you might end up with a stale cache and a nasty headache.

You can find the current module hash in the following locations:

- In the logs, if you set the logging level to INFO or DEBUG
- In the cached module's self.cache_instance.module_hash attribute
- As the name of the subdirectory in the persistent cache
