"""Rate limiting for Zypherus API."""

import time
from collections import defaultdict
from typing import Dict, Tuple
from flask import request


class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, requests_per_period: int = 100, period_seconds: int = 60):
        self.requests_per_period = requests_per_period
        self.period_seconds = period_seconds
        self.buckets: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, time.time()))
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        tokens, last_update = self.buckets.get(key, (self.requests_per_period, now))
        
        # Refill tokens based on time elapsed
        time_elapsed = now - last_update
        refill_tokens = int(time_elapsed * (self.requests_per_period / self.period_seconds))
        tokens = min(self.requests_per_period, tokens + refill_tokens)
        
        if tokens > 0:
            tokens -= 1
            self.buckets[key] = (tokens, now)
            return True
        
        self.buckets[key] = (tokens, now)
        return False
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key."""
        tokens, _ = self.buckets.get(key, (self.requests_per_period, time.time()))
        return max(0, tokens)
    
    def cleanup(self):
        """Remove old entries to prevent memory leak."""
        now = time.time()
        expired_keys = [
            key for key, (_, last_update) in self.buckets.items()
            if now - last_update > self.period_seconds * 2
        ]
        for key in expired_keys:
            del self.buckets[key]


def get_client_id() -> str:
    """Get client identifier for rate limiting."""
    # Try X-API-Key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    
    # Try Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return f"token:{token[:20]}"  # First 20 chars
    
    # Fall back to IP
    return f"ip:{request.remote_addr}"


def rate_limit_middleware(limiter: RateLimiter, enabled: bool = True):
    """Create rate limiting middleware."""
    
    def decorator(f):
        from functools import wraps
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not enabled:
                return f(*args, **kwargs)
            
            client_id = get_client_id()
            
            if not limiter.is_allowed(client_id):
                from flask import jsonify
                return jsonify({
                    "success": False,
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Max {limiter.requests_per_period} requests per {limiter.period_seconds}s"
                }), 429
            
            remaining = limiter.get_remaining(client_id)
            result = f(*args, **kwargs)
            
            # Add rate limit headers
            if isinstance(result, tuple):
                response, status_code = result
                response["rate_limit_remaining"] = remaining
                return response, status_code
            else:
                result["rate_limit_remaining"] = remaining
                return result
        
        return wrapper
    return decorator
