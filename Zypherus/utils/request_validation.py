"""Request validation and security middleware."""

from typing import Optional, Tuple
import logging
from flask import Flask, request

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
DEFAULT_MAX_JSON_SIZE = 5 * 1024 * 1024  # 5MB


def setup_request_validation(app: Flask, 
                            max_content_length: Optional[int] = None) -> None:
    """Setup request validation middleware.
    
    Args:
        app: Flask application
        max_content_length: Maximum request size in bytes
    """
    max_size = max_content_length or DEFAULT_MAX_CONTENT_LENGTH
    app.config["MAX_CONTENT_LENGTH"] = max_size
    
    @app.before_request
    def validate_request() -> Optional[Tuple]:
        """Validate incoming requests."""
        # Check content length
        content_length = request.content_length
        if content_length and content_length > max_size:
            logger.warning(f"Request too large: {content_length} bytes")
            return {
                "success": False,
                "error": "PAYLOAD_TOO_LARGE",
                "message": f"Request exceeds {max_size} bytes limit"
            }, 413
        
        # Check for required headers on POST/PUT
        if request.method in ("POST", "PUT"):
            if request.is_json and not request.headers.get("Content-Type", "").startswith("application/json"):
                logger.warning(f"Missing Content-Type header for {request.method}")
                # Continue anyway, Flask will handle it
        
        return None


def setup_rate_limiting(app: Flask, 
                       requests_per_minute: int = 60) -> None:
    """Setup basic rate limiting (production should use Redis).
    
    Args:
        app: Flask application  
        requests_per_minute: Max requests per minute per IP
    """
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    request_times = defaultdict(list)
    
    @app.before_request
    def check_rate_limit() -> Optional[Tuple]:
        """Check rate limit for client IP."""
        client_ip = request.remote_addr
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        # Clean old requests
        request_times[client_ip] = [
            t for t in request_times[client_ip] if t > window_start
        ]
        
        # Check limit
        if len(request_times[client_ip]) >= requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return {
                "success": False,
                "error": "RATE_LIMIT_EXCEEDED",
                "message": f"Exceeded {requests_per_minute} requests per minute"
            }, 429
        
        # Record request
        request_times[client_ip].append(now)
        return None
