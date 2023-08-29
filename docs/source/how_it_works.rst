How it works
============

`torchcache` emerged from the need to cache the projected output of a large vision backbone, as it was taking the majority of the training time. However, as with any cache, care had to be taken regarding cache size management, memory usage, and cache invalidation.

Automatic cache management
--------------------------

`torchcache` automatically manages the cache by hashing both:

1. The decorated module (including its source code obtained through `inspect.getsource`) and its args/kwargs.
2. The inputs provided to the module's forward method.

This hash serves as the cache key for the forward method's output per item in a batch. When our MRU (most-recently-used) cache fills up for the given session, the system continues running the forward method and dismisses the oldest output. This MRU strategy streamlines cache invalidation, aligning with the iterative nature of neural network training, without requiring any additional record-keeping.

.. warning::

   To avoid having to calculate the directory size on every forward pass, `torchcache` measures and limits the size of the persistent data created only for the given session. To prevent the persistent cache from growing indefinitely, you should periodically clear the cache directory. Note that if you let `torchcache` create a temporary directory, it will be automatically deleted when the session ends.

Tensor hashing
--------------

Creating an effective hashing mechanism for torch tensors involved addressing several criteria:

- **Deterministic Hashing:** Consistent inputs should invariably yield the same hash.
- **Speed:** Given its execution on every forward pass—regardless of caching status—the hashing process needs to be rapid.
- **Size Constraints:** Given the frequent use of mixed precision in backbone models, it was crucial to prevent overflow scenarios.
- **Batch Sensitivity:** The cache shouldn't invalidate with every new batch due to fluctuating batch sizes or sequences.

`torchcache` achieves these via the steps outlined below:

1. **Coefficient Generation:** Begin with a coefficient tensor rolling with powers of 2 (i.e. `[1, 2, 4, 8, ...]`). After reaching 2^15, the sequence starts over to avoid overflow, especially in mixed precision scenarios.
2. **Tensor Flattening & Subsampling:** Flatten the input tensor and subsample `subsample_count` elements, defaulting to 10000. This is to avoid using the whole batch as the hash input. The subsampling is deterministic, taking every `tensor.shape[0] // subsample_count` element.
3. **Hashing Process:** The subsampled tensor multiplies the coefficient tensor. Sum the results along the batch dimension for the final hash.
