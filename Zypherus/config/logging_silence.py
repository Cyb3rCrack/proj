"""
Logging utilities with AI response silence support.
Prevents logging output while AI is generating responses.
"""
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any

# Thread-local storage for silence state
_silence_context = threading.local()


class SilentFilter(logging.Filter):
	"""Filter that prevents logging when in silent mode."""
	
	def filter(self, record: logging.LogRecord) -> bool:
		"""Return False if in silent mode, True otherwise."""
		return not getattr(_silence_context, 'silent', False)


def setup_logging_with_silence(logger_name: str = "ACE") -> logging.Logger:
	"""Setup logger with silence mode support."""
	logger = logging.getLogger(logger_name)
	
	# Add silent filter to all handlers
	for handler in logger.handlers:
		handler.addFilter(SilentFilter())
	
	return logger


@contextmanager
def silent_logging():
	"""Context manager to suppress logging output.
	
	Usage:
		with silent_logging():
			# This code will not produce logging output
			ace.answer(user_input)
	"""
	old_value = getattr(_silence_context, 'silent', False)
	_silence_context.silent = True
	try:
		yield
	finally:
		_silence_context.silent = old_value


def enable_logging():
	"""Enable logging output."""
	_silence_context.silent = False


def disable_logging():
	"""Disable logging output."""
	_silence_context.silent = True


def is_silent() -> bool:
	"""Check if logging is currently silenced."""
	return getattr(_silence_context, 'silent', False)
