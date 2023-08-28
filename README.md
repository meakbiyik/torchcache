# torchcache

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
