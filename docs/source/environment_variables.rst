Environment variables
=====================

You can customize `torchcache` logging behavior through these environment variables:

- **TORCHCACHE_LOG_LEVEL:** Set the logging level. Default is `WARN`.
- **TORCHCACHE_LOG_FMT:** Adjust the logging format. The default is `[torchcache] - %(asctime)s - %(name)s - %(levelname)s - %(message)s`.
- **TORCHCACHE_LOG_DATEFMT:** Define the logging date format. The default is `%Y-%m-%d %H:%M:%S`.
- **TORCHCACHE_LOG_FILE:** Provide a path for the log file. By default, it is `None` and will be opened in append mode.
