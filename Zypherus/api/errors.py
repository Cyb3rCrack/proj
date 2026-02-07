"""Error handling for Zypherus API."""

import logging
from typing import Optional, Tuple, Dict, Any
from flask import jsonify, Flask
from werkzeug.exceptions import HTTPException

logger = logging.getLogger("ACE.Errors")


class APIError(Exception):
    """Base API error."""
    
    def __init__(self, message: str, status_code: int = 400, error_code: str = "API_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class AuthenticationError(APIError):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 401, "AUTH_ERROR")


class AuthorizationError(APIError):
    """Authorization error."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, 403, "AUTHZ_ERROR")


class ValidationError(APIError):
    """Validation error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, "VALIDATION_ERROR")
        self.details = details


class NotFoundError(APIError):
    """Resource not found."""
    
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, 404, "NOT_FOUND")


class ConflictError(APIError):
    """Resource conflict."""
    
    def __init__(self, message: str = "Resource conflict"):
        super().__init__(message, 409, "CONFLICT")


class ServiceUnavailableError(APIError):
    """Service unavailable."""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message, 503, "SERVICE_UNAVAILABLE")


class RateLimitExceededError(APIError):
    """Rate limit exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, 429, "RATE_LIMIT_EXCEEDED")


def format_error_response(error: Exception, request_id: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
    """Format error response."""
    if isinstance(error, APIError):
        return {
            "success": False,
            "error": error.error_code,
            "message": error.message,
            "details": getattr(error, "details", None),
            "request_id": request_id
        }, error.status_code
    
    elif isinstance(error, HTTPException):
        return {
            "success": False,
            "error": "HTTP_ERROR",
            "message": error.description or str(error),
            "request_id": request_id
        }, error.code or 400
    
    else:
        logger.error(f"Unhandled exception: {type(error).__name__}: {str(error)}", exc_info=True)
        return {
            "success": False,
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id
        }, 500


def setup_error_handlers(app: Flask):
    """Register error handlers with Flask app."""
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        from flask import request
        request_id = getattr(request, "request_id", None)
        response, status = format_error_response(error, request_id)
        return jsonify(response), status
    
    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        from flask import request
        request_id = getattr(request, "request_id", None)
        response, status = format_error_response(error, request_id)
        return jsonify(response), status
    
    @app.errorhandler(Exception)
    def handle_generic_error(error):
        from flask import request
        request_id = getattr(request, "request_id", None)
        response, status = format_error_response(error, request_id)
        return jsonify(response), status
    
    @app.errorhandler(404)
    def handle_not_found(error):
        from flask import request
        request_id = getattr(request, "request_id", None)
        response, status = format_error_response(error, request_id)
        return jsonify(response), status
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        from flask import request
        request_id = getattr(request, "request_id", None)
        response, status = format_error_response(error, request_id)
        return jsonify(response), status
