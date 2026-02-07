"""Logging configuration for Zypherus."""

import logging
import sys
import os
from datetime import datetime


def setup_logging(level: str = "INFO", format_type: str = "json"):
    """Configure logging for production use."""
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # File handler - daily rotation
    timestamp = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(f"{logs_dir}/zypherus_{timestamp}.log")
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Format
    if format_type.lower() == "json":
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore[import]
            formatter = jsonlogger.JsonFormatter(  # type: ignore[attr-defined]
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        except ImportError:
            # Fallback if pythonjsonlogger not available
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for module."""
    return logging.getLogger(name)


# Initialize on import
if os.getenv("LOG_LEVEL"):
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
