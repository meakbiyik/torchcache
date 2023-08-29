# torchcache

[![Lint and Test](https://github.com/meakbiyik/torchcache/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/meakbiyik/torchcache/actions/workflows/ci.yaml) [![Codecov](https://codecov.io/gh/meakbiyik/torchcache/graph/badge.svg?token=Oh6mNp0pc8)](https://codecov.io/gh/meakbiyik/torchcache) [![Documentation Status](https://readthedocs.org/projects/torchcache/badge/?version=latest)](https://torchcache.readthedocs.io/en/latest/?badge=latest)

Effortlessly cache PyTorch module outputs on-the-fly with `torchcache`.

The documentation is available [torchcache.readthedocs.io](https://torchcache.readthedocs.io/en/latest/).

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Assumptions](#assumptions)
- [Use cases](#use-cases)
- [How it works](#how-it-works)
  - [Automatic cache management](#automatic-cache-management)
  - [Tensor hashing](#tensor-hashing)
- [Environment variables](#environment-variables)
- [Contribution](#contribution)

## Features

- Cache PyTorch module outputs either in-memory or persistently to disk.
- Simple decorator-based interface for easy usage.
- Uses an MRU (most-recently-used) cache to limit memory/disk usage

## Installation

```bash
pip install torchcache
```

## Usage

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
```

## Assumptions

To ensure seamless operation, `torchcache` assumes the following:

- Your module is a subclass of `nn.Module`.
- The module's forward method accepts any number of positional arguments with shapes `(B, \*)`, where `B` is the batch size and `\*` represents any number of dimensions. All tensors should be on the same device and have the same dtype.
- The forward method returns a single tensor of shape `(B, \*)`.

## Use cases

A common use case is caching the outputs of frozen, pre-trained model backbones to accelerate training:

```python
import torch
import torch.nn as nn
from torchcache import torchcache

@torchcache(persistent=True)
class MyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.eval()
        self.requires_grad_(False)

    def forward(self, x):
        # Cached to disk
        return self.linear(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MyBackbone()
        self.head = nn.Linear(10, 10)

    def forward(self, x):
        x = self.backbone(x)  # Cached output
        x = self.head(x)      # Not cached
        return x

model = MyModel()
```

## How it works

`torchcache` emerged from the need to cache the projected output of a large vision backbone, as it was taking to majority of the training time. However, as with any cache, I had to be careful with cache size management, memory usage and cache invalidation.

Here's an overview of how `torchcache` addresses these challenges:

### Automatic cache management

torchcache automatically manages the cache by hashing both:

1. The decorated module (including its source code obtained through `inspect.getsource`) and its args/kwargs.
2. The inputs provided to the module's forward method.

This hash serves as the cache key for the forward method's output per item in a batch. When our MRU (most-recently-used) cache fills up for the given session, the system continues running the forward method and dismisses the newest output. This MRU strategy streamlines cache invalidation, aligning with the iterative nature of neural network training, without requiring any auxiliary record-keeping.

> :warning: **Warning**: To avoid having to calculate the directory size on every forward pass, `torchcache` measures and limits the size of the persistent data created only for the given session. To prevent the persistent cache from growing indefinitely, you should periodically clear the cache directory. Note that if you let `torchcache` create a temporary directory, it will be automatically deleted when the session ends.

### Tensor hashing

Creating an effective hashing mechanism for torch tensors involved addressing several criteria:

- **Deterministic Hashing:** Consistent inputs should invariably yield the same hash.
- **Speed:** Given its execution on every forward pass—regardless of caching status—the hashing process needs to be rapid.
- **Size Constraints:** Given the frequent use of mixed precision in backbone models, it was crucial to prevent overflow scenarios.
- **Batch Sensitivity:** The cache shouldn't invalidate with every new batch due to fluctuating batch sizes or sequences.

`torchcache` achieves these via the steps outlined below:

1. **Coefficient Generation:** We initiate a coefficient tensor rolling with powers of 2 (i.e. `[1, 2, 4, 8, ...]`). After reaching 2^15, the sequence starts over to sidestep overflow situations, particularly when using mixed precision.
2. **Tensor Flattening & Subsampling:** Flatten the input tensor and subsample `subsample_count` elements from it, by default 10000. This is done to avoid using the whole batch as input to the hash. The subsampling is done deterministically by taking every `tensor.shape[0] // subsample_count` element.
3. **Hashing Process:** The subsampled tensor is then multiplied by the coefficient tensor. The final hash is obtained by summing the results along the batch dimension.

## Environment variables

Customize `torchcache` logging behavior using the following environment variables:

- `TORCHCACHE_LOG_LEVEL` - logging level, defaults to `WARN`
- `TORCHCACHE_LOG_FMT` - logging format, defaults to `[torchcache] - %(asctime)s - %(name)s - %(levelname)s - %(message)s`
- `TORCHCACHE_LOG_DATEFMT` - logging date format, defaults to `%Y-%m-%d %H:%M:%S`
- `TORCHCACHE_LOG_FILE` - path to the log file, defaults to `None`. Opened in append mode.

## Contribution

1. Ensure you have Python installed.
2. Install [`poetry`](https://python-poetry.org/docs/#installation).
3. Run `poetry install`  to set up dependencies.
4. Run `poetry run pre-commit install` to install pre-commit hooks.
5. Create a branch, make your changes, and open a pull request.
