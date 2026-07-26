"""Centralized Logging Utility Module.

Provides standard logger factory function for consistent log formatting across modules.
"""

import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT: int = 5
DEFAULT_LOG_FILENAME: str = "feast_pipeline.log"
LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"


def get_logger(
    name: str = __name__,
    level: int | str = logging.INFO,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Returns a configured Python logger instance with standard stream and rotating file handlers."""
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / DEFAULT_LOG_FILENAME
            file_handler = RotatingFileHandler(
                str(log_file),
                maxBytes=DEFAULT_MAX_BYTES,
                backupCount=DEFAULT_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
