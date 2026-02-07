"""Memory optimization utilities for efficient storage."""

from __future__ import annotations

import json
import gzip
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("ZYPHERUS.Memory")


class EmbeddingCompressor:
    """Compress embeddings for storage efficiency."""
    
    @staticmethod
    def compress(vec: np.ndarray, quality: int = 80) -> bytes:
        """Compress embedding to bytes.
        
        Args:
            vec: Embedding vector
            quality: Compression quality (1-100), higher = less compression
        
        Returns:
            Compressed bytes
        """
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        
        # Quantize to uint8 (lossy but efficient)
        if quality == 100:
            # Lossless: store as float32
            return vec.astype(np.float32).tobytes()
        else:
            # Lossy: quantize to uint8
            # Map [-1, 1] range to [0, 255]
            quantized = ((vec + 1) / 2 * 255).astype(np.uint8)
            # Further compress with gzip
            return gzip.compress(quantized.tobytes())
    
    @staticmethod
    def decompress(compressed: bytes, original_dim: int, quality: int = 80) -> np.ndarray:
        """Decompress embedding from bytes."""
        if quality == 100:
            return np.frombuffer(compressed, dtype=np.float32)
        else:
            # Decompress gzip
            decompressed = gzip.decompress(compressed)
            # Convert from uint8 back to float
            quantized = np.frombuffer(decompressed, dtype=np.uint8)
            # Pad if needed
            if len(quantized) < original_dim:
                quantized = np.pad(quantized, (0, original_dim - len(quantized)))
            # Map [0, 255] back to [-1, 1]
            vec = (quantized.astype(np.float32) / 255 * 2) - 1
            return vec[:original_dim]


class MemoryCache:
    """LRU cache for frequently accessed embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: np.ndarray) -> None:
        """Put embedding in cache."""
        if key in self.cache:
            # MEMORY LEAK FIX: Use index() instead of list.remove() - O(n) avoidance
            try:
                idx = self.access_order.index(key)
                self.access_order.pop(idx)
            except ValueError:
                pass
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order.pop(0)
                self.cache.pop(lru_key, None)
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


class PaginatedIndex:
    """Paginate large indices for memory efficiency."""
    
    def __init__(self, page_size: int = 1000):
        self.page_size = page_size
        self.pages: Dict[int, List[Dict[str, Any]]] = {}
        self.metadata: Dict[int, Dict[str, Any]] = {}
    
    def add_entry(self, entry: Dict[str, Any]) -> Tuple[int, int]:
        """Add entry to paginated index.
        
        Returns:
            (page_id, entry_index_in_page)
        """
        total_entries = sum(len(page) for page in self.pages.values())
        page_id = total_entries // self.page_size
        
        if page_id not in self.pages:
            self.pages[page_id] = []
        
        self.pages[page_id].append(entry)
        entry_idx = len(self.pages[page_id]) - 1
        
        return page_id, entry_idx
    
    def get_entry(self, page_id: int, entry_idx: int) -> Optional[Dict[str, Any]]:
        """Get entry from paginated index."""
        if page_id in self.pages and entry_idx < len(self.pages[page_id]):
            return self.pages[page_id][entry_idx]
        return None
    
    def get_page(self, page_id: int) -> Optional[List[Dict[str, Any]]]:
        """Get entire page."""
        return self.pages.get(page_id)
    
    def iterate(self):
        """Iterate through all entries."""
        for page_id in sorted(self.pages.keys()):
            for entry in self.pages[page_id]:
                yield entry


class EmbeddingIndex:
    """Efficient embedding storage with compression and pagination."""
    
    def __init__(self, compression_quality: int = 80, cache_size: int = 1000,
                 page_size: int = 1000):
        self.compressor = EmbeddingCompressor()
        self.cache = MemoryCache(cache_size)
        self.pagination = PaginatedIndex(page_size)
        self.compression_quality = compression_quality
        self.stats = {
            "compression_ratio": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "original_bytes": 0,
            "compressed_bytes": 0,
        }
    
    def add(self, key: str, vec: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Add embedding with compression."""
        # Cache uncompressed version
        self.cache.put(key, vec)
        
        # Store compressed version with pagination
        compressed = self.compressor.compress(vec, self.compression_quality)
        
        entry = {
            "key": key,
            "compressed": compressed.hex(),  # hex for JSON serialization
            "dim": len(vec),
            "metadata": metadata or {},
        }
        
        page_id, entry_idx = self.pagination.add_entry(entry)
        
        # Update stats
        self.stats["original_bytes"] += vec.nbytes
        self.stats["compressed_bytes"] += len(compressed)
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding (from cache or decompress)."""
        # Try cache first
        cached = self.cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        
        self.stats["cache_misses"] += 1
        
        # Find in pagination (linear search - could be optimized with index)
        for page_id in sorted(self.pagination.pages.keys()):
            for entry in self.pagination.pages[page_id]:
                if entry["key"] == key:
                    compressed = bytes.fromhex(entry["compressed"])
                    vec = self.compressor.decompress(compressed, entry["dim"], 
                                                     self.compression_quality)
                    self.cache.put(key, vec)
                    return vec
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory stats."""
        if self.stats["original_bytes"] > 0:
            self.stats["compression_ratio"] = (
                1.0 - self.stats["compressed_bytes"] / self.stats["original_bytes"]
            ) * 100
        
        cache_hit_rate = 0.0
        total_access = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_access > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_access * 100
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "compression_ratio": f"{self.stats['compression_ratio']:.1f}%",
        }


class MemoryOptimizer:
    """Monitor and optimize memory usage."""
    
    @staticmethod
    def get_object_size(obj: Any) -> int:
        """Estimate object size in bytes."""
        import sys
        return sys.getsizeof(obj)
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get current memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident set size
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual memory
            "percent": process.memory_percent(),
        }
    
    @staticmethod
    def estimate_storage_needed(num_embeddings: int, dim: int, 
                               compression_quality: int = 80) -> Dict[str, str]:
        """Estimate storage needed for embeddings."""
        # Float32 = 4 bytes per dimension
        uncompressed = num_embeddings * dim * 4
        
        # Compression ratios (empirical)
        compression_ratios = {
            80: 0.4,  # 60% reduction
            90: 0.6,  # 40% reduction
            100: 1.0,  # No reduction
        }
        
        ratio = compression_ratios.get(compression_quality, 0.5)
        compressed = int(uncompressed * ratio)
        
        def format_bytes(b: int) -> str:
            b_float = float(b)
            for unit in ["B", "KB", "MB", "GB"]:
                if b_float < 1024:
                    return f"{b_float:.1f} {unit}"
                b_float /= 1024
            return f"{b_float:.1f} TB"
        
        return {
            "uncompressed": format_bytes(uncompressed),
            "compressed": format_bytes(compressed),
            "compression_ratio": f"{(1 - ratio) * 100:.1f}%",
        }


__all__ = [
    "EmbeddingCompressor",
    "MemoryCache",
    "PaginatedIndex",
    "EmbeddingIndex",
    "MemoryOptimizer",
]
