"""Enhanced logging system with structured logging support.

Provides JSON logging, multiple handlers, and per-component configuration.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats logs as JSON for easier parsing."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_dict)


class StandardFormatter(logging.Formatter):
    """Standard text formatter with color support."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, "")
            reset = self.RESET
        else:
            color = reset = ""
        
        timestamp = self.formatTime(record)
        level = f"{color}{record.levelname:8s}{reset}"
        name = f"[{record.name}]"
        msg = record.getMessage()
        
        result = f"{timestamp} {level} {name} {msg}"
        
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)
        
        return result


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "standard",
    log_file: Optional[str] = None,
    component_levels: Optional[dict] = None,
) -> None:
    """Set up logging for the entire Zypherus system.
    
    Args:
        log_level: Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format style ("standard" or "json")
        log_file: Optional path to log file
        component_levels: Dict mapping component names to levels, e.g. 
                         {"ace.llm": "DEBUG", "ace.embedding": "INFO"}
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root level
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Choose formatter
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = StandardFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10_000_000,  # 10MB
                backupCount=5,
            )
            file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Failed to set up file logging: {e}")

    # Per-component levels
    if component_levels:
        for component, level in component_levels.items():
            logger = logging.getLogger(component)
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    logging.getLogger("ACE").info(f"Logging initialized: level={log_level}, format={log_format}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger", "JSONFormatter", "StandardFormatter"]
