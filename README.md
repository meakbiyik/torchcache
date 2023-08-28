# torchcache

[![Lint and Test](https://github.com/meakbiyik/torchcache/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/meakbiyik/torchcache/actions/workflows/ci.yaml) [![codecov](https://codecov.io/gh/meakbiyik/torchcache/graph/badge.svg?token=Oh6mNp0pc8)](https://codecov.io/gh/meakbiyik/torchcache)

`torchcache` caches PyTorch module outputs on the fly. It can cache persistent outputs to disk or in-memory, and can be applied with a simple decorator:

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

## Installation

```bash
pip install torchcache
```

## Assumptions

- The module is a subclass of `nn.Module`
- The module forward method is called with any number of positional and arguments with shapes (B, \*) where B is the batch size and \* is any number of dimensions. The tensors must be on the same device and have the same dtype.
- The forward method returns a single tensor with shape (B, \*).

## Use case

Caching the outputs of a pre-trained model backbones for faster training, assuming the backbone is frozen and the outputs are not needed for backpropagation. For example, in the following snippet, the outputs of the backbone are cached, but the outputs of the head are not:

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
        # This output will be cached to disk
        return self.linear(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MyBackbone()
        self.head = nn.Linear(10, 10)

    def forward(self, x):
        # This output will be cached
        x = self.backbone(x)
        # This output will not be cached
        x = self.head(x)
        return x

model = MyModel()
```

## Environment variables

The following environment variables may be useful to set the package behavior:

- `TORCHCACHE_LOG_LEVEL` - logging level, defaults to `WARN`
- `TORCHCACHE_LOG_FMT` - logging format, defaults to `[torchcache] - %(asctime)s - %(name)s - %(levelname)s - %(message)s`
- `TORCHCACHE_LOG_DATEFMT` - logging date format, defaults to `%Y-%m-%d %H:%M:%S`
- `TORCHCACHE_LOG_FILE` - path to the log file, defaults to `None`. Opened in append mode.

## Contribution

1. Install Python.
2. Install [`poetry`](https://python-poetry.org/docs/#installation)
3. Run `poetry install` to install dependencies
4. Run `poetry run pre-commit install` to install pre-commit hooks
