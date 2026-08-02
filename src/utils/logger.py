"""Centralized Logging Utility Module.

Provides standard logger factory function for consistent ISO 8601 log formatting.
"""

import sys
import logging

LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"


def get_logger(
    name: str = __name__,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Returns a configured Python logger instance with standard stream handler."""
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger

