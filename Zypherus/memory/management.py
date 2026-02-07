"""Memory management and cleanup system to prevent memory leaks and manage resources efficiently."""

import time
import logging
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger("ZYPHERUS.MemoryManagement")


@dataclass
class MemoryStats:
    """Track memory usage statistics."""
    timestamp: float
    rss_mb: float  # Resident set size
    vms_mb: float  # Virtual memory
    percent: float  # Percentage of total system memory
    num_entries: int
    num_claims: int
    cache_size: int


class MemoryMonitor:
    """Monitor system and ACE memory usage."""
    
    def __init__(self, warning_threshold_mb: float = 2048, critical_threshold_mb: float = 3072):
        """
        Initialize monitor.
        
        Args:
            warning_threshold_mb: Warn when RSS exceeds this (default 2GB)
            critical_threshold_mb: Trigger cleanup at this (default 3GB)
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.stats_history: List[MemoryStats] = []
        self.max_history = 1000  # Keep recent stats
        
    def get_current_stats(self, ace_instance=None) -> MemoryStats:
        """Get current memory statistics."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return MemoryStats(
            timestamp=time.time(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            percent=process.memory_percent(),
            num_entries=len(ace_instance.memory.index) if ace_instance else 0,
            num_claims=len(ace_instance.claim_store.claims) if ace_instance else 0,
            cache_size=len(ace_instance.memory.cache.cache) if ace_instance and hasattr(ace_instance.memory, 'cache') else 0,
        )
    
    def track_stats(self, ace_instance=None) -> Optional[str]:
        """Track memory stats snapshot."""
        stats = self.get_current_stats(ace_instance)
        self.stats_history.append(stats)
        
        # Enforce max history size
        if len(self.stats_history) > self.max_history:
            self.stats_history.pop(0)
        
        # Check thresholds
        if stats.rss_mb > self.critical_threshold_mb:
            logger.warning(f"CRITICAL memory usage: {stats.rss_mb:.1f}MB - triggering cleanup")
            return "critical"
        elif stats.rss_mb > self.warning_threshold_mb:
            logger.warning(f"HIGH memory usage: {stats.rss_mb:.1f}MB")
            return "warning"
        return "ok"
    
    def get_trend(self, lookback_minutes: int = 5) -> Dict[str, float]:
        """Get memory trend over time."""
        if not self.stats_history:
            return {"trend": 0, "current": 0}
        
        now = time.time()
        cutoff = now - (lookback_minutes * 60)
        
        recent = [s for s in self.stats_history if s.timestamp >= cutoff]
        if len(recent) < 2:
            return {"trend": 0, "current": self.stats_history[-1].rss_mb}
        
        first, last = recent[0], recent[-1]
        trend = last.rss_mb - first.rss_mb  # MB increase
        
        return {
            "trend_mb": trend,
            "current_mb": last.rss_mb,
            "avg_mb": sum(s.rss_mb for s in recent) / len(recent),
            "samples": len(recent),
        }


class MemoryCleanupManager:
    """Handle automatic memory cleanup and eviction policies."""
    
    def __init__(self, lru_mode: bool = True):
        """
        Initialize cleanup manager.
        
        Args:
            lru_mode: Use LRU (least recently used) eviction policy
        """
        self.lru_mode = lru_mode
        self.last_cleanup = time.time()
        self.cleanup_interval_s = 300  # 5 minutes
        self.stats: Dict[str, Any] = {
            "cleanups_total": 0,
            "entries_evicted": 0,
            "claims_evicted": 0,
            "cache_clears": 0,
            "bytes_freed": 0,
        }
    
    def should_cleanup(self) -> bool:
        """Check if cleanup interval has elapsed."""
        return (time.time() - self.last_cleanup) > self.cleanup_interval_s
    
    def cleanup_old_entries(self, ace_instance, max_age_days: float = 30,  keep_ratio: float = 0.9) -> int:
        """
        Remove old entries from memory based on age.
        
        Args:
            ace_instance: ACE instance with memory to clean
            max_age_days: Remove entries older than this
            keep_ratio: Keep top (keep_ratio * 100)% of entries by recency
            
        Returns:
            Number of entries evicted
        """
        if not hasattr(ace_instance.memory, 'entries') or not ace_instance.memory.entries:
            return 0
        
        now = time.time()
        max_age_s = max_age_days * 86400
        evicted = 0
        
        try:
            to_remove = []
            for idx, entry in enumerate(ace_instance.memory.entries):
                if not isinstance(entry, dict):
                    continue
                    
                timestamp = entry.get("timestamp", now)
                age = now - timestamp
                
                # Remove if older than threshold
                if age > max_age_s:
                    to_remove.append(idx)
                    evicted += 1
            
            # Also apply keep ratio  
            if len(ace_instance.memory.entries) > 0:
                keep_count = int(len(ace_instance.memory.entries) * keep_ratio)
                if len(to_remove) > (len(ace_instance.memory.entries) - keep_count):
                    # Sort by timestamp and remove oldest
                    entries_with_idx = [(idx, entry.get("timestamp", now)) 
                                       for idx, entry in enumerate(ace_instance.memory.entries)
                                       if isinstance(entry, dict)]
                    entries_with_idx.sort(key=lambda x: x[1])
                    to_remove = [idx for idx, _ in entries_with_idx[:len(entries_with_idx) - keep_count]]
                    evicted = len(to_remove)
            
            # Remove in reverse order to maintain indices
            for idx in sorted(to_remove, reverse=True):
                try:
                    ace_instance.memory.entries.pop(idx)
                except:
                    pass
            
            self.stats["entries_evicted"] += evicted
            logger.info(f"Evicted {evicted} old entries")
            return evicted
            
        except Exception as e:
            logger.error(f"Error during entry cleanup: {e}")
            return 0
    
    def cleanup_low_confidence_claims(self, ace_instance, min_confidence: float = 0.3) -> int:
        """
        Remove low-confidence claims to save memory.
        
        Args:
            ace_instance: ACE instance with claims to clean
            min_confidence: Remove claims below this confidence
            
        Returns:
            Number of claims evicted
        """
        if not hasattr(ace_instance, 'claim_store'):
            return 0
        
        try:
            claims = ace_instance.claim_store.claims
            if not claims:
                return 0
            
            evicted = 0
            to_remove = []
            
            for claim_id, claim in list(claims.items()):
                conf = claim.get("confidence", 0.5)
                if conf < min_confidence:
                    to_remove.append(claim_id)
                    evicted += 1
            
            # Keep some randomness - don't remove all low confidence
            # Remove only if we have lots of them
            if evicted / len(claims) > 0.3:
                to_remove = to_remove[:len(claims) // 20]  # Remove at most 5%
                evicted = len(to_remove)
            
            for claim_id in to_remove:
                try:
                    del claims[claim_id]
                except:
                    pass
            
            self.stats["claims_evicted"] += evicted
            logger.info(f"Evicted {evicted} low-confidence claims")
            return evicted
            
        except Exception as e:
            logger.error(f"Error during claim cleanup: {e}")
            return 0
    
    def clear_cache(self, ace_instance) -> bool:
        """Clear embeddings cache to free memory."""
        try:
            if hasattr(ace_instance.memory, 'cache') and hasattr(ace_instance.memory.cache, 'clear'):
                ace_instance.memory.cache.clear()
                self.stats["cache_clears"] += 1
                logger.info("Cleared embeddings cache")
                return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        return False
    
    def full_cleanup(self, ace_instance) -> Dict[str, int]:
        """
        Perform full cleanup cycle.
        
        Returns:
            Dict with cleanup statistics
        """
        logger.info("Starting full memory cleanup...")
        self.last_cleanup = time.time()
        
        results = {
            "entries_removed": self.cleanup_old_entries(ace_instance, max_age_days=30),
            "claims_removed": self.cleanup_low_confidence_claims(ace_instance, min_confidence=0.3),
            "cache_cleared": self.clear_cache(ace_instance),
        }
        
        self.stats["cleanups_total"] += 1
        logger.info(f"Cleanup complete: {results}")
        
        return results


class LeakDetector:
    """Detect potential memory leaks and resource issues."""
    
    def __init__(self, sample_interval_s: float = 60):
        self.sample_interval_s = sample_interval_s
        self.samples: List[Tuple[float, float]] = []  # (time, rss_mb)
        self.growth_threshold_mb = 100  # Alert if grows >100MB/hour
        
    def record_sample(self) -> None:
        """Record current memory as a sample."""
        try:
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / 1024 / 1024
            self.samples.append((time.time(), rss_mb))
            
            # Keep recent samples
            if len(self.samples) > 3600:  # ~1 hour at 1 sample/sec
                self.samples.pop(0)
        except:
            pass
    
    def detect_leak(self) -> Optional[Dict[str, Any]]:
        """
        Detect if memory is growing suspiciously.
        
        Returns:
            Dict with leak info if detected, None otherwise
        """
        if len(self.samples) < 120:  # Need at least 2 minutes of data
            return None
        
        # Calculate growth rate
        old_sample = self.samples[0]
        new_sample = self.samples[-1]
        
        time_hours = (new_sample[0] - old_sample[0]) / 3600
        memory_growth = new_sample[1] - old_sample[1]
        growth_rate_per_hour = memory_growth / max(time_hours, 0.001)
        
        if growth_rate_per_hour > self.growth_threshold_mb:
            return {
                "leak_detected": True,
                "growth_mb_per_hour": growth_rate_per_hour,
                "total_growth_mb": memory_growth,
                "current_mb": new_sample[1],
                "timespan_hours": time_hours,
            }
        
        return None


class MemoryHealthCheck:
    """Comprehensive memory health and performance check."""
    
    def __init__(self):
        self.monitor = MemoryMonitor()
        self.cleanup = MemoryCleanupManager()
        self.leak_detector = LeakDetector()
    
    def check(self, ace_instance) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dictionary with health status
        """
        # Track stats
        status = self.monitor.track_stats(ace_instance)
        
        # Record sample for leak detection
        self.leak_detector.record_sample()
        
        # Cleanup if needed
        cleanup_results = {}
        if self.cleanup.should_cleanup() and status == "critical":
            cleanup_results = self.cleanup.full_cleanup(ace_instance)
        
        # Detect leaks
        leak_info = self.leak_detector.detect_leak()
        
        # Get trend
        trend = self.monitor.get_trend()
        
        health = {
            "status": status,
            "memory_mb": self.monitor.stats_history[-1].rss_mb if self.monitor.stats_history else 0,
            "trend": trend,
            "cleanup": cleanup_results,
            "leak_detected": leak_info is not None,
            "leak_info": leak_info,
        }
        
        if leak_info:
            logger.warning(f"Potential memory leak detected: {leak_info}")
        
        return health
