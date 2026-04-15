"""
utils/logger.py - Centralized structured logging configuration.

All modules import `get_logger(__name__)` to obtain a named logger.
Format includes: timestamp, level, module name, and message.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Log levels used throughout the app:
      DEBUG   - verbose internal state (chunking, embedding steps)
      INFO    - normal operations (file uploaded, chunk count, latency)
      WARNING - degraded quality (low similarity scores, small chunk returned)
      ERROR   - failures (parse error, FAISS write failure, LLM error)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    return logger
