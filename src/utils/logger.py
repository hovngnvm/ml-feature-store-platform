"""
Structured Logging Utility.

Provides centralized logger setup with consistent formatting across all modules.
"""

import os
import sys
import logging

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance with standardized formatting.

    Args:
        name: Name of the logger, typically __name__ or module identifier.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    level_name = settings.log_level.upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger
