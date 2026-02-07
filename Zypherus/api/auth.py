"""Authentication and authorization for Zypherus API."""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Callable, List

import jwt
from flask import request, jsonify, Request

logger = logging.getLogger("ACE.Auth")


class AuthConfig:
    """Authentication configuration."""
    
    def __init__(self):
        # JWT_SECRET_KEY is REQUIRED in production - no insecure default
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            if os.getenv("FLASK_ENV") == "production":
                raise ValueError("JWT_SECRET_KEY environment variable is required in production")
            else:
                logger.warning("No JWT_SECRET_KEY configured - using development mode only")
                self.secret_key = "dev-only-insecure-key"
        
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.token_expiry_hours = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment."""
        keys_env = os.getenv("API_KEYS", "")
        keys = {}
        if keys_env:
            for key_def in keys_env.split(";"):
                if ":" in key_def:
                    key, role = key_def.split(":", 1)
                    keys[key.strip()] = {
                        "role": role.strip(),
                        "created_at": datetime.utcnow().isoformat()
                    }
        return keys


class TokenManager:
    """JWT token management."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
    
    def generate_token(self, user_id: str, role: str = "user", expires_in_hours: Optional[int] = None) -> str:
        """Generate JWT token."""
        expiry_hours = expires_in_hours or self.config.token_expiry_hours
        payload = {
            "user_id": user_id,
            "role": role,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=expiry_hours)
        }
        token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key."""
        if api_key in self.config.api_keys:
            key_info = self.config.api_keys[api_key]
            return {
                "user_id": f"api_key_{api_key[:8]}",
                "role": key_info["role"],
                "type": "api_key"
            }
        return None


class AuthService:
    """Authentication service for API."""
    
    def __init__(self):
        self.config = AuthConfig()
        self.token_manager = TokenManager(self.config)
    
    def extract_token(self) -> Optional[str]:
        """Extract token from request."""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None
    
    def extract_api_key(self) -> Optional[str]:
        """Extract API key from request."""
        return request.headers.get("X-API-Key")
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user."""
        # Try JWT first
        token = self.extract_token()
        if token:
            user = self.token_manager.verify_token(token)
            if user:
                return user
        
        # Try API key
        api_key = self.extract_api_key()
        if api_key:
            user = self.token_manager.verify_api_key(api_key)
            if user:
                return user
        
        return None
    
    def require_auth(self, required_roles: Optional[List[str]] = None) -> Callable:
        """Decorator for protected endpoints."""
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                user = self.get_current_user()
                
                if not user:
                    return jsonify({
                        "error": "Unauthorized",
                        "message": "Missing or invalid authentication credentials"
                    }), 401
                
                # Check role if specified
                if required_roles and user.get("role") not in required_roles:
                    return jsonify({
                        "error": "Forbidden",
                        "message": f"Insufficient permissions. Required roles: {required_roles}"
                    }), 403
                
                # Store user in request context
                request.user = user  # type: ignore[attr-defined]
                return fn(*args, **kwargs)
            
            return wrapper
        return decorator


def create_auth_service() -> AuthService:
    """Factory function for auth service."""
    return AuthService()
