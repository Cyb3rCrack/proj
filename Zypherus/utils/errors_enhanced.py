"""Enhanced error handling and logging utilities."""

from __future__ import annotations

import logging
import functools
import traceback
from typing import Any, Callable, Optional, TypeVar
import time

logger = logging.getLogger("ACE")

T = TypeVar('T')


class ACEException(Exception):
    """Base exception for ACE."""
    pass


class LLMException(ACEException):
    """LLM-related errors."""
    pass


class MemoryException(ACEException):
    """Memory-related errors."""
    pass


class EmbeddingException(ACEException):
    """Embedding-related errors."""
    pass


def log_exception(level: int = logging.ERROR, include_traceback: bool = True) -> Callable:
    """Decorator to log exceptions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = f"Exception in {func.__name__}: {type(e).__name__}: {str(e)}"
                if include_traceback:
                    logger.log(level, msg, exc_info=True)
                else:
                    logger.log(level, msg)
                raise
        return wrapper
    return decorator


def safe_call(func: Callable[..., T], *args: Any, default: Optional[T] = None,
              log_errors: bool = True, **kwargs: Any) -> Optional[T]:
    """Safely call a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error calling {func.__name__}: {type(e).__name__}: {str(e)}")
        return default


def timed_operation(operation_name: str, log_level: int = logging.DEBUG) -> Callable:
    """Decorator to log operation timing."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.log(log_level, f"{operation_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"{operation_name} failed after {elapsed:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator


def setup_logging(config: Any) -> None:
    """Setup logging based on config."""
    try:
        from logging.handlers import RotatingFileHandler
        
        level = getattr(logging, config.logging.level.upper(), logging.INFO)
        fmt = logging.Formatter(config.logging.format)
        
        # Root logger
        root_logger = logging.getLogger("ZYPHERUS")
        root_logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)
        
        # File handler if configured
        if config.logging.file_path:
            file_handler = RotatingFileHandler(
                config.logging.file_path,
                maxBytes=config.logging.file_size_mb * 1024 * 1024,
                backupCount=config.logging.backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(fmt)
            root_logger.addHandler(file_handler)
        
        logger.info(f"Logging configured at level {config.logging.level}")
    except Exception as e:
        logging.warning(f"Failed to setup logging: {e}")


__all__ = [
    "ACEException",
    "LLMException",
    "MemoryException",
    "EmbeddingException",
    "log_exception",
    "safe_call",
    "timed_operation",
    "setup_logging",
]
