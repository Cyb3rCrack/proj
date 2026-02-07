"""LLM fallback strategies and graceful degradation."""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger("ZYPHERUS.LLM")


class LLMStrategy(ABC):
    """Abstract base for LLM strategies."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Generate response."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if strategy is available."""
        pass


class OllamaStrategy(LLMStrategy):
    """Use Ollama local LLM."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "neural-chat"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Ollama client."""
        try:
            import requests
            self.client = requests.Session()
            # Test connection
            self.client.get(f"{self.base_url}/api/tags", timeout=2)
            logger.info(f"Ollama connected at {self.base_url}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check Ollama availability."""
        if self.client is None:
            return False
        try:
            self.client.get(f"{self.base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Generate using Ollama."""
        if self.client is None:
            raise RuntimeError("Ollama client not initialized")
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "num_predict": max_tokens,
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class HeuristicStrategy(LLMStrategy):
    """Fallback heuristic-based strategy."""
    
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Generate using heuristics."""
        # Extract key terms from prompt
        lines = prompt.split("\n")
        
        # Simple pattern matching for common question types
        lower_prompt = prompt.lower()
        
        if "extract" in lower_prompt and "fact" in lower_prompt:
            # Return a heuristic fact extraction
            return "- Unable to extract facts without LLM; please provide a source document"
        
        if "summarize" in lower_prompt:
            # Return first few lines as summary
            content_lines = [l for l in lines if l.strip() and len(l.split()) > 5]
            return " ".join(content_lines[:2])
        
        if "extract" in lower_prompt and "entit" in lower_prompt:
            # Return empty entities
            return "[]"
        
        # Default: return acknowledgment
        return "I cannot generate a response without LLM connectivity. Please ingest documents and ask simpler questions."
    
    def is_available(self) -> bool:
        """Heuristic strategy is always available."""
        return True


class CachedStrategy(LLMStrategy):
    """Cache LLM responses to reduce API calls."""
    
    def __init__(self, strategy: LLMStrategy, ttl_seconds: float = 3600):
        self.strategy = strategy
        self.cache: Dict[str, tuple[str, float]] = {}
        self.ttl = ttl_seconds
        self.max_cache_size = 500  # MEMORY LEAK FIX: Cap cache size
        self.cleanup_interval = 50  # MEMORY LEAK FIX: Clean every N operations
    
    def _cleanup_expired_cache(self):
        """MEMORY LEAK FIX: Remove expired cache entries proactively."""
        now = time.time()
        expired_keys = [k for k, (_, ts) in self.cache.items() if now - ts >= self.ttl]
        for k in expired_keys:
            del self.cache[k]
        
        # If still too large, remove oldest entries
        if len(self.cache) > self.max_cache_size:
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            remove_count = len(self.cache) - self.max_cache_size
            for k, _ in sorted_items[:remove_count]:
                del self.cache[k]
    
    def _cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Generate with caching."""
        key = self._cache_key(prompt)
        
        # MEMORY LEAK FIX: Periodically clean up expired entries
        if len(self.cache) % self.cleanup_interval == 0:
            self._cleanup_expired_cache()
        
        # Check cache
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Cache hit for prompt (key={key[:8]}...)")
                return response
            else:
                del self.cache[key]
        
        # Generate and cache
        response = self.strategy.generate(prompt, max_tokens, **kwargs)
        self.cache[key] = (response, time.time())
        logger.debug(f"Cached response for prompt (key={key[:8]}...)")
        return response
    
    def is_available(self) -> bool:
        """Available if underlying strategy is available."""
        return self.strategy.is_available()


class LLMManager:
    """Manage LLM strategies with fallback."""
    
    def __init__(self, primary: Optional[LLMStrategy] = None,
                 fallback: Optional[LLMStrategy] = None):
        self.primary = primary
        self.fallback = fallback or HeuristicStrategy()
        self.retry_count = 0
        self.max_retries = 3
    
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Generate using available strategy."""
        # Try primary strategy
        if self.primary and self.primary.is_available():
            try:
                return self.primary.generate(prompt, max_tokens, **kwargs)
            except Exception as e:
                logger.warning(f"Primary LLM strategy failed: {e}, falling back to {self.fallback.__class__.__name__}")
        
        # Try fallback
        try:
            return self.fallback.generate(prompt, max_tokens, **kwargs)
        except Exception as e:
            logger.error(f"All LLM strategies failed: {e}")
            return f"[Unable to generate response: {str(e)}]"
    
    def is_primary_available(self) -> bool:
        """Check if primary LLM is available."""
        return self.primary is not None and self.primary.is_available()
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM status."""
        return {
            "primary_available": self.is_primary_available(),
            "primary_type": self.primary.__class__.__name__ if self.primary else None,
            "fallback_type": self.fallback.__class__.__name__,
        }


def create_llm_manager(config: Any) -> LLMManager:
    """Create LLM manager from config."""
    primary = None
    
    # Create primary strategy based on config
    if config.llm.provider == "ollama":
        primary = OllamaStrategy(config.llm.base_url, config.llm.model_name)
        if config.llm.cache_enabled:
            primary = CachedStrategy(primary, config.llm.cache_ttl_s)
    
    # Fallback is always heuristic
    fallback = HeuristicStrategy()
    
    return LLMManager(primary=primary, fallback=fallback)


__all__ = [
    "LLMStrategy",
    "OllamaStrategy",
    "HeuristicStrategy",
    "CachedStrategy",
    "LLMManager",
    "create_llm_manager",
]
