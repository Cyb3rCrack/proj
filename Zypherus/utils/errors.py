"""Enhanced error handling with custom exceptions and retry logic."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger("ZYPHERUS.Errors")

T = TypeVar("T")


class ACEException(Exception):
    """Base exception for Zypherus system."""
    
    def __init__(self, message: str, context: Optional[dict] = None):
        """Initialize exception with context.
        
        Args:
            message: Error message
            context: Dictionary with additional context (component, details, etc.)
        """
        self.message = message
        self.context = context or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        """String representation with context."""
        result = f"[{self.__class__.__name__}] {self.message}"
        if self.context:
            result += f" (context: {self.context})"
        return result


class ConfigurationError(ACEException):
    """Raised when configuration is invalid."""
    pass


class EmbeddingError(ACEException):
    """Raised when embedding operations fail."""
    pass


class LLMError(ACEException):
    """Raised when LLM operations fail."""
    pass


class StorageError(ACEException):
    """Raised when storage operations fail."""
    pass


class ReasoningError(ACEException):
    """Raised when reasoning operations fail."""
    pass


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_backoff_s: float = 0.5,
        backoff_factor: float = 1.5,
        max_backoff_s: float = 30.0,
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_backoff_s: Initial backoff delay in seconds
            backoff_factor: Exponential backoff multiplier
            max_backoff_s: Maximum backoff delay
        """
        self.max_attempts = max(1, max_attempts)
        self.initial_backoff_s = max(0.0, initial_backoff_s)
        self.backoff_factor = max(1.0, backoff_factor)
        self.max_backoff_s = max(self.initial_backoff_s, max_backoff_s)


def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs,
) -> T:
    """Execute function with exponential backoff retry.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: RetryConfig instance
        on_retry: Callback on retry (attempt_num, exception)
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func execution
        
    Raises:
        The last exception encountered if all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    backoff_s = config.initial_backoff_s
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt < config.max_attempts:
                if on_retry:
                    on_retry(attempt, e)
                
                logger.warning(
                    f"Attempt {attempt} failed: {e}. "
                    f"Retrying in {backoff_s:.1f}s..."
                )
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * config.backoff_factor, config.max_backoff_s)
            else:
                logger.error(f"All {config.max_attempts} attempts failed. Last error: {e}")
    
    raise last_exception


def with_error_context(component: str, operation: str) -> Callable:
    """Decorator to add context to errors.
    
    Args:
        component: Component name (e.g., "embedding", "llm")
        operation: Operation name (e.g., "encode", "query")
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except ACEException:
                # Re-raise ACE exceptions as-is
                raise
            except Exception as e:
                # Wrap other exceptions with context
                context = {
                    "component": component,
                    "operation": operation,
                    "original_error": str(e),
                }
                raise ACEException(
                    f"Error in {component}.{operation}: {e}",
                    context=context,
                ) from e
        return wrapper
    return decorator


__all__ = [
    "ACEException",
    "ConfigurationError",
    "EmbeddingError",
    "LLMError",
    "StorageError",
    "ReasoningError",
    "RetryConfig",
    "retry_with_backoff",
    "with_error_context",
]
