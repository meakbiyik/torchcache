"""Tests for the logging module."""
import logging
import os
import sys

from torchcache.logging import set_logger_config


def test_set_logger_config() -> None:
    """Test the set_logger_config function."""
    set_logger_config()
    assert logging.getLogger().level == logging.WARN
    assert len(logging.getLogger().handlers) == 1
    assert isinstance(logging.getLogger().handlers[0], logging.StreamHandler)
    assert logging.getLogger().handlers[0].stream == sys.stdout
    assert (
        logging.getLogger().handlers[0].formatter._fmt
        == "[torchcache] - %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    assert logging.getLogger().handlers[0].formatter.datefmt == "%Y-%m-%d %H:%M:%S"
    # Also test various environment variable configurations
    os.environ["TORCHCACHE_LOG_LEVEL"] = "DEBUG"
    os.environ[
        "TORCHCACHE_LOG_FMT"
    ] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    os.environ["TORCHCACHE_LOG_DATEFMT"] = "%Y-%m-%d %H:%M:%S"
    os.environ["TORCHCACHE_LOG_FILE"] = "test.log"
    set_logger_config()
    assert logging.getLogger().level == logging.DEBUG
    assert len(logging.getLogger().handlers) == 2
    assert isinstance(logging.getLogger().handlers[0], logging.StreamHandler)
    assert logging.getLogger().handlers[0].stream == sys.stdout
    assert (
        logging.getLogger().handlers[0].formatter._fmt
        == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    assert logging.getLogger().handlers[0].formatter.datefmt == "%Y-%m-%d %H:%M:%S"
    assert isinstance(logging.getLogger().handlers[1], logging.FileHandler)
    assert (
        logging.getLogger().handlers[1].formatter._fmt
        == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    assert logging.getLogger().handlers[1].formatter.datefmt == "%Y-%m-%d %H:%M:%S"
    assert logging.getLogger().handlers[1].baseFilename.endswith("test.log")
    # Test invalid log level
    os.environ["TORCHCACHE_LOG_LEVEL"] = "INVALID"
    set_logger_config()
    assert logging.getLogger().level == logging.WARN
