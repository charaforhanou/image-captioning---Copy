"""Structured logging configuration shared across the project."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def get_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a logger with a consistent format.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_file: Optional path to also write logs to a file.
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).

    Returns:
        A configured ``logging.Logger`` instance. Repeated calls with the
        same ``name`` return the same logger without duplicating handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
