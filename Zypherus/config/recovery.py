"""Error recovery and resilience utilities."""
import time
import logging
import random
from functools import wraps
from typing import Callable, Any, Tuple, Type, Optional, List
import asyncio

logger = logging.getLogger("ZYPHERUS.Recovery")


class RetryPolicy:
	"""Configuration for retry behavior."""
	
	def __init__(
		self,
		max_attempts: int = 3,
		initial_delay: float = 1.0,
		max_delay: float = 60.0,
		backoff_factor: float = 2.0,
		jitter: bool = True,
		retry_on: Optional[Tuple[Type[Exception], ...]] = None
	):
		"""Initialize retry policy.
		
		Args:
			max_attempts: Maximum number of retry attempts
			initial_delay: Initial delay between retries (seconds)
			max_delay: Maximum delay between retries (seconds)
			backoff_factor: Exponential backoff multiplier
			jitter: Add randomness to delay
			retry_on: Tuple of exception types to retry on
		"""
		self.max_attempts = max_attempts
		self.initial_delay = initial_delay
		self.max_delay = max_delay
		self.backoff_factor = backoff_factor
		self.jitter = jitter
		self.retry_on = retry_on or (Exception,)
	
	def get_delay(self, attempt: int) -> float:
		"""Calculate delay for given attempt number.
		
		Args:
			attempt: Attempt number (0-indexed)
			
		Returns:
			Delay in seconds
		"""
		delay = min(
			self.initial_delay * (self.backoff_factor ** attempt),
			self.max_delay
		)
		
		if self.jitter:
			# Add ±10% jitter
			jitter_factor = 1 + random.uniform(-0.1, 0.1)
			delay *= jitter_factor
		
		return delay


class CircuitBreaker:
	"""Circuit breaker pattern for preventing cascading failures."""
	
	def __init__(
		self,
		failure_threshold: int = 5,
		recovery_timeout: float = 60.0,
		name: str = "circuit_breaker"
	):
		"""Initialize circuit breaker.
		
		Args:
			failure_threshold: Number of failures to open circuit
			recovery_timeout: Time before attempting to recover (seconds)
			name: Name for logging
		"""
		self.failure_threshold = failure_threshold
		self.recovery_timeout = recovery_timeout
		self.name = name
		
		self.failure_count = 0
		self.success_count = 0
		self.last_failure_time = None
		self.state = "closed"  # closed, open, half-open
	
	def call(self, func: Callable, *args, **kwargs) -> Any:
		"""Execute function with circuit breaker protection.
		
		Args:
			func: Function to execute
			*args: Function arguments
			**kwargs: Function keyword arguments
			
		Returns:
			Function result
			
		Raises:
			RuntimeError: If circuit is open
		"""
		if self.state == "open":
			if self._should_attempt_reset():
				self.state = "half-open"
				logger.info(f"Circuit breaker '{self.name}' attempting recovery")
			else:
				raise RuntimeError(f"Circuit breaker '{self.name}' is open")
		
		try:
			result = func(*args, **kwargs)
			self._on_success()
			return result
		except Exception as e:
			self._on_failure()
			raise
	
	def _should_attempt_reset(self) -> bool:
		"""Check if enough time has passed to attempt reset."""
		if self.last_failure_time is None:
			return True
		
		return (time.time() - self.last_failure_time) >= self.recovery_timeout
	
	def _on_success(self):
		"""Handle successful call."""
		self.failure_count = 0
		self.success_count += 1
		
		if self.state == "half-open" and self.success_count >= 2:
			self.state = "closed"
			logger.info(f"Circuit breaker '{self.name}' recovered to closed state")
	
	def _on_failure(self):
		"""Handle failed call."""
		self.failure_count += 1
		self.last_failure_time = time.time()
		self.success_count = 0
		
		if self.failure_count >= self.failure_threshold:
			self.state = "open"
			logger.warning(
				f"Circuit breaker '{self.name}' opened after "
				f"{self.failure_count} failures"
			)
	
	def get_state(self) -> dict:
		"""Get circuit breaker state."""
		return {
			"name": self.name,
			"state": self.state,
			"failure_count": self.failure_count,
			"success_count": self.success_count,
			"last_failure_time": self.last_failure_time
		}


def with_retry(
	policy: Optional[RetryPolicy] = None,
	on_retry: Optional[Callable] = None
) -> Callable:
	"""Decorator to add retry logic to a function.
	
	Usage:
		@with_retry(RetryPolicy(max_attempts=3))
		def unstable_function():
			pass
	
	Args:
		policy: RetryPolicy instance
		on_retry: Optional callback function called on each retry
		
	Returns:
		Decorated function
	"""
	if policy is None:
		policy = RetryPolicy()
	
	def decorator(func: Callable) -> Callable:
		@wraps(func)
		def wrapper(*args, **kwargs) -> Any:
			last_exception: Optional[Exception] = None
			
			for attempt in range(policy.max_attempts):
				try:
					result = func(*args, **kwargs)
					if attempt > 0:
						logger.info(
							f"Successfully recovered {func.__name__} "
							f"on attempt {attempt + 1}"
						)
					return result
				
				except Exception as e:
					last_exception = e
					
					# Check if exception type should be retried
					if not isinstance(e, policy.retry_on):
						raise
					
					if attempt < policy.max_attempts - 1:
						delay = policy.get_delay(attempt)
						logger.warning(
							f"Attempt {attempt + 1}/{policy.max_attempts} failed "
							f"for {func.__name__}: {e}. "
							f"Retrying in {delay:.2f}s..."
						)
						
						if on_retry:
							on_retry(attempt, e)
						
						time.sleep(delay)
					else:
						logger.error(
							f"All {policy.max_attempts} attempts failed "
							f"for {func.__name__}"
						)
			
			if last_exception is not None:
				raise last_exception
			raise RuntimeError(f"Failed to execute {func.__name__}")
		
		return wrapper
	
	return decorator
def fallback(default_value: Any = None) -> Callable:
	"""Decorator to provide fallback value on exception.
	
	Usage:
		@fallback(default_value=[])
		def risky_function():
			pass
	"""
	def decorator(func: Callable) -> Callable:
		@wraps(func)
		def wrapper(*args, **kwargs) -> Any:
			try:
				return func(*args, **kwargs)
			except Exception as e:
				logger.warning(
					f"Function {func.__name__} failed with {e}, "
					f"using fallback value"
				)
				return default_value
		
		return wrapper
	
	return decorator


class TransientErrorHandler:
	"""Handles detection and recovery from transient errors."""
	
	# Common transient errors
	TRANSIENT_ERRORS = {
		"ConnectionError",
		"TimeoutError",
		"BrokenPipeError",
		"PoolError",
		"HTTPError"
	}
	
	@staticmethod
	def is_transient(exception: Exception) -> bool:
		"""Check if exception is likely transient.
		
		Args:
			exception: Exception to check
			
		Returns:
			True if exception appears to be transient
		"""
		exc_name = exception.__class__.__name__
		
		# Check explicit transient error types
		if exc_name in TransientErrorHandler.TRANSIENT_ERRORS:
			return True
		
		# Check error message for transient indicators
		message = str(exception).lower()
		transient_keywords = [
			"temporary",
			"timeout",
			"connection",
			"unavailable",
			"busy",
			"try again"
		]
		
		return any(keyword in message for keyword in transient_keywords)
	
	@staticmethod
	def handle_transient_error(
		exception: Exception,
		context: str = ""
	) -> Tuple[bool, Optional[str]]:
		"""Handle a transient error with recovery suggestions.
		
		Args:
			exception: Exception to handle
			context: Optional context about where error occurred
			
		Returns:
			Tuple of (is_transient, recovery_suggestion)
		"""
		is_transient = TransientErrorHandler.is_transient(exception)
		
		if not is_transient:
			return False, None
		
		suggestions = {
			"ConnectionError": "Check network connectivity and retry",
			"TimeoutError": "Increase timeout and retry",
			"BrokenPipeError": "Connection lost, reconnect and retry",
			"PoolError": "Resource pool exhausted, retry after delay",
			"HTTPError": "Server error, retry with exponential backoff"
		}
		
		exc_name = exception.__class__.__name__
		suggestion = suggestions.get(exc_name, "Retry operation")
		
		if context:
			suggestion = f"{suggestion} (in {context})"
		
		return True, suggestion
