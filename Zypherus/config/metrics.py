"""Metrics collection and reporting for ACE.

Tracks performance metrics like latency, cache hits, errors, etc.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ZYPHERUS.Metrics")


@dataclass
class ComponentMetrics:
    """Metrics for a single component."""
    
    component: str
    requests: int = 0
    errors: int = 0
    total_latency_s: float = 0.0
    min_latency_s: float = float("inf")
    max_latency_s: float = 0.0
    last_latency_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def avg_latency_s(self) -> float:
        """Average latency in seconds."""
        return self.total_latency_s / max(1, self.requests)

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        return (self.errors / max(1, self.requests)) * 100

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(1, total)) * 100

    def record_request(self, latency_s: float, success: bool = True, 
                       cache_hit: bool = False) -> None:
        """Record a request."""
        self.requests += 1
        self.total_latency_s += latency_s
        self.last_latency_s = latency_s
        self.min_latency_s = min(self.min_latency_s, latency_s)
        self.max_latency_s = max(self.max_latency_s, latency_s)
        self.last_updated = time.time()
        
        if not success:
            self.errors += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["avg_latency_s"] = self.avg_latency_s
        data["error_rate"] = self.error_rate
        data["cache_hit_rate"] = self.cache_hit_rate
        return data


class MetricsCollector:
    """Centralized metrics collection."""

    def __init__(self, enabled: bool = True):
        """Initialize metrics collector.
        
        Args:
            enabled: Whether to collect metrics
        """
        self._enabled = enabled
        self._metrics: Dict[str, ComponentMetrics] = {}
        self._lock = threading.RLock()

    def record(self, component: str, latency_s: float = 0.0, 
               success: bool = True, cache_hit: bool = False) -> None:
        """Record a metric for a component.
        
        Args:
            component: Component name (e.g., "llm.renderer", "embedding")
            latency_s: Operation latency in seconds
            success: Whether operation succeeded
            cache_hit: Whether it was a cache hit
        """
        if not self._enabled:
            return

        with self._lock:
            if component not in self._metrics:
                self._metrics[component] = ComponentMetrics(component=component)
            
            self._metrics[component].record_request(latency_s, success, cache_hit)

    def get_metrics(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a component or all components.
        
        Args:
            component: Component name, or None for all
            
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            if component:
                if component in self._metrics:
                    return self._metrics[component].to_dict()
                return {}
            
            return {name: metrics.to_dict() 
                    for name, metrics in self._metrics.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            if not self._metrics:
                return {"message": "No metrics collected"}
            
            total_requests = sum(m.requests for m in self._metrics.values())
            total_errors = sum(m.errors for m in self._metrics.values())
            avg_latency = (sum(m.total_latency_s for m in self._metrics.values()) / 
                          max(1, total_requests))
            
            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate_pct": (total_errors / max(1, total_requests)) * 100,
                "avg_latency_s": avg_latency,
                "components": len(self._metrics),
                "components_detail": {name: metrics.to_dict() 
                                     for name, metrics in self._metrics.items()}
            }

    def reset(self, component: Optional[str] = None) -> None:
        """Reset metrics.
        
        Args:
            component: Component to reset, or None for all
        """
        with self._lock:
            if component:
                if component in self._metrics:
                    del self._metrics[component]
            else:
                self._metrics.clear()

    def export_json(self, path: str) -> None:
        """Export metrics to JSON file."""
        with self._lock:
            with open(path, "w") as f:
                json.dump(self.get_summary(), f, indent=2)
        logger.info(f"Exported metrics to {path}")

    def enable(self) -> None:
        """Enable metrics collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable metrics collection."""
        self._enabled = False

    def __repr__(self) -> str:
        """String representation."""
        return f"MetricsCollector({len(self._metrics)} components)"


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics(enabled: bool = True) -> MetricsCollector:
    """Get or create global metrics collector.
    
    Args:
        enabled: Whether to enable metrics collection
        
    Returns:
        Global MetricsCollector instance
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector(enabled=enabled)
    return _global_metrics


__all__ = ["ComponentMetrics", "MetricsCollector", "get_metrics"]
