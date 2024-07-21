# torchcache

[![Lint and Test](https://github.com/meakbiyik/torchcache/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/meakbiyik/torchcache/actions/workflows/ci.yaml) [![Codecov](https://codecov.io/gh/meakbiyik/torchcache/graph/badge.svg?token=Oh6mNp0pc8)](https://codecov.io/gh/meakbiyik/torchcache) [![Documentation Status](https://readthedocs.org/projects/torchcache/badge/?version=latest)](https://torchcache.readthedocs.io/en/latest/?badge=latest)

Effortlessly cache PyTorch module outputs on-the-fly with `torchcache`.

Particularly useful for caching and serving the outputs of computationally expensive large, pre-trained PyTorch modules, such as vision transformers. Note that gradients will not flow through the cached outputs.

- [Features](#features)
- [Installation](#installation)
- [Basic usage](#basic-usage)
- [Assumptions](#assumptions)
- [Contribution](#contribution)

## Features

- Cache PyTorch module outputs either in-memory or persistently to disk.
- Simple decorator-based interface for easy usage.
- Uses an MRU (most-recently-used) cache to limit memory/disk usage

## Installation

```bash
pip install torchcache
```

## Basic usage

Quickly cache the output of your PyTorch module with a single decorator:

```python
from torchcache import torchcache

@torchcache()
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        # This output will be cached
        return self.linear(x)

input_tensor = torch.ones(10, dtype=torch.float32)
# Output is cached during the first call...
output = model(input_tensor)
# ...and is retrieved from the cache for the next one
output_cached = model(input_tensor)

```

See documentation at [torchcache.readthedocs.io](https://torchcache.readthedocs.io/en/latest/) for more examples.

## Assumptions

To ensure seamless operation, `torchcache` assumes the following:

- Your module is a subclass of `nn.Module`.
- The module's forward method accepts any number of positional arguments with shapes `(B, *)`, where `B` is the batch size and `*` represents any number of dimensions. All tensors should be on the same device and have the same dtype.
- The forward method returns a single tensor of shape `(B, *)`.

## Contribution

1. Ensure you have Python installed.
2. Install [`poetry`](https://python-poetry.org/docs/#installation).
3. Run `poetry install`  to set up dependencies.
4. Run `poetry run pre-commit install` to install pre-commit hooks.
5. Create a branch, make your changes, and open a pull request.
