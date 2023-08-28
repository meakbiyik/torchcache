"""Logger utilites of torchcache."""
import logging
import os
import sys
from typing import Dict, Optional


def set_logger_config(
    level: Optional[int] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    stream: Optional[object] = None,
    filename: Optional[str] = None,
) -> None:
    """Set configuration for logger object.

    Parameters
    ----------
    level : int, optional
        Logging level, by default None
    fmt : str, optional
        Logging format, by default None
    datefmt : str, optional
        Logging date format, by default None
    stream : object, optional
        Logging stream, by default None
    filename : str, optional
        Logging filename, by default None

    Returns
    -------
    None
    """
    env_vars = _parse_env_vars()

    level = level or env_vars.get("TORCHCACHE_LOG_LEVEL", logging.WARN)
    fmt = fmt or env_vars.get(
        "TORCHCACHE_LOG_FMT",
        "[torchcache] - %(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    datefmt = datefmt or env_vars.get("TORCHCACHE_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
    stream = stream or sys.stdout

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handlers = []

    stream_handler = logging.StreamHandler(stream=stream)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    filename = filename or env_vars.get("TORCHCACHE_LOG_FILE")
    if filename is not None:
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # set logger
    logging.basicConfig(level=level, handlers=handlers, force=True)


def _parse_env_vars() -> Dict[str, str]:
    """Parse environment variables.

    Returns
    -------
    Dict[str, str]
        Dictionary of environment variables
    """
    env_vars = {}
    for key, value in os.environ.items():
        if key.startswith("TORCHCACHE_"):
            env_vars[key] = value
    # if log level is set, convert to int
    if "TORCHCACHE_LOG_LEVEL" in env_vars:
        candidate_int = getattr(logging, env_vars["TORCHCACHE_LOG_LEVEL"], None)
        if isinstance(candidate_int, int):
            env_vars["TORCHCACHE_LOG_LEVEL"] = candidate_int
        else:
            logging.warning(
                f"Invalid TORCHCACHE_LOG_LEVEL: {env_vars['TORCHCACHE_LOG_LEVEL']}. "
                "Valid values are: DEBUG, INFO, WARNING, ERROR, CRITICAL. "
                "Defaulting to logging.INFO."
            )
            del env_vars["TORCHCACHE_LOG_LEVEL"]
    return env_vars
