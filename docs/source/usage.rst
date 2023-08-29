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

If you want to optimize the cache size, you can play with compression quality and the dtype of cached tensor. By default, tensors are cached with the same dtype as the original tensor, and with a compression quality of 9 (out of 11 in brotli). You can change these values with the ``cache_dtype`` and ``brotli_quality`` arguments. Note that the compression quality is only used for persistent caching, and comes with implications on both the speed of caching and loading. For most cases, the default values should be fine.

.. code-block:: python

   from torchcache import torchcache

    @torchcache(
        persistent=True,
        cache_dtype=torch.half,
        brotli_quality=11,
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
            self.torchcache_brotli_quality = 11

        def forward(self, x):
            # This output will be cached
            return self.linear(x)

Note that the parameters set in the module take precedence over the parameters set in the decorator.

When is cache invalidated?
--------------------------

The cache is invalidated when:

- The module's code changes
- The module initialization arguments/keywork arguments change
- The torchcache parameters change

More specifically, if you change the compression quality, the cache will be invalidated. If you change the cache dtype, the cache will be invalidated. If you change the module's code, the cache will be invalidated. If you change the module's initialization arguments, the cache will be invalidated. If you change the module's keyword arguments, the cache will be invalidated.

The invalidation does not remove the old cached items, it just skips them. They can also be, although highly unlikely, overwritten by a new cache with the same key. If you revert the changes, the cache will be valid again.

Note that updating `torchcache` might, in some cases, invalidate the cache. I will try to avoid this as much as possible, but cannot guarantee full backward compatibility, particularly in the early stages of the project.
