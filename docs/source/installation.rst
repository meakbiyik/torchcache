Installation
============

To install `torchcache`, simply run:

.. code-block:: bash

   pip install torchcache

`torchcache` is compatible with Python >= 3.8 and PyTorch >= 1.0.0.

Assumptions
-----------

`torchcache` works seamlessly under a few assumptions:

- Your module is a subclass of `nn.Module`.
- The module's forward method accepts any number of positional or keyword arguments with shapes `(B, *)`, where `B` is the batch size and `*` represents any number of dimensions, or any other basic immutable Python types (int, str, float, boolean). All tensors should be on the same device and have the same dtype.
- The forward method returns a single tensor of shape `(B, *)`.

If your module does not meet these assumptions, you might not be able to use `torchcache` directly. Feel free to open an issue or submit a PR if you think your use case should be supported.
