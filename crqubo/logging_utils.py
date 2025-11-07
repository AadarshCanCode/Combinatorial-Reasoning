"""
Logging utilities for CRQUBO framework.

Provides structured logging, context managers, and helper functions
for consistent logging across the codebase.
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = DEFAULT_LOG_LEVEL,
    structured: bool = False,
    stream: Optional[object] = None,
) -> None:
    """Configure global logging settings.

    Args:
        level: Logging level
        structured: Whether to use structured (JSON-like) logging
        stream: Optional stream for logging (defaults to sys.stdout)
    """
    if stream is None:
        stream = sys.stdout

    if structured:
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt=DEFAULT_DATE_FORMAT,
        )
    else:
        formatter = logging.Formatter(
            DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger by name.

    Ensures logging is configured with defaults if not already set.
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        configure_logging()
    return logger


@contextmanager
def log_duration(
    logger: logging.Logger,
    message: str,
    level: int = logging.INFO,
    **context,
) -> Iterator[None]:
    """Context manager to log duration of operations.

    Args:
        logger: Logger instance to use
        message: Message describing the operation
        level: Logging level
        **context: Additional context to include in logs
    """
    start_time = time.perf_counter()
    context_suffix = (
        " " + " ".join(f"{k}={v}" for k, v in context.items()) if context else ""
    )
    logger.log(level, f"START: {message}{context_suffix}")
    try:
        yield
    except Exception as exc:
        duration = time.perf_counter() - start_time
        logger.exception(
            f"FAILED: {message}{context_suffix} duration={duration:.3f}s | error={exc}"
        )
        raise
    else:
        duration = time.perf_counter() - start_time
        logger.log(
            level,
            f"END: {message}{context_suffix} duration={duration:.3f}s",
        )


def add_context(logger: logging.Logger, **context) -> logging.LoggerAdapter:
    """Create a logger adapter with additional context."""
    return logging.LoggerAdapter(logger, extra=context)
