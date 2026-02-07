"""Query result caching with TTL support."""

from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class QueryCache:
    """Thread-safe query result cache with TTL."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_size: Maximum number of cached queries
        """
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
    
    def _key(self, query: str, **kwargs: Any) -> str:
        """Generate cache key from query and parameters."""
        cache_data = {"query": query, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, query: str, **kwargs: Any) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._key(query, **kwargs)
        
        if key not in self._cache:
            return None
        
        result, timestamp = self._cache[key]
        
        if datetime.utcnow() - timestamp > self.ttl:
            del self._cache[key]
            logger.debug(f"Cache entry expired: {key}")
            return None
        
        logger.debug(f"Cache hit: {key}")
        return result
    
    def set(self, result: Any, query: str, **kwargs: Any) -> None:
        """Store result in cache."""
        # Evict oldest entry if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry: {oldest_key}")
        
        key = self._key(query, **kwargs)
        self._cache[key] = (result, datetime.utcnow())
        logger.debug(f"Cache stored: {key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"Cache cleared ({count} entries)")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl.total_seconds()
        }
