"""torchcache package."""
from .logging import set_logger_config
from .torchcache import torchcache

__all__ = ["torchcache", "set_logger_config"]

# set the logger config per environment variables
set_logger_config()
