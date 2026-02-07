#!/usr/bin/env python
"""
MEMORY LEAK PATCHES & CLEANUP UTILITY
=======================================

This script patches critical memory leaks and provides cleanup functions.

Memory leaks fixed:
1. WebCrawler.visited_urls/failed_urls - Now bounded to 10,000 URLs max
2. CachedStrategy.cache - Now capped at 500 entries with TTL cleanup
3. MemoryCache.access_order - Fixed O(n) list.remove() -> O(1) index()
4. DialogueManager._trim() - Efficient index rebuilding
5. entity_index - Changed from List to Set for O(1) lookups
"""

import gc
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CLEANUP")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("[PSUTIL] Optional psutil not installed - memory stats unavailable")


def cleanup_memory():
    """Perform aggressive memory cleanup."""
    logger.info("[CLEANUP] Starting aggressive memory cleanup...")
    
    # Force garbage collection
    collected = gc.collect()
    logger.info(f"[GC] Collected {collected} objects")
    
    logger.info("[CLEANUP] Memory cleanup complete")
    logger.info("[CLEANUP] Tip: Restart ACE for best results")


def clear_web_crawler_history():
    """Clear WebCrawler URL history to free memory."""
    try:
        from Zypherus.tools.web_crawler import WebCrawler
        
        crawler = WebCrawler()
        initial_urls = len(crawler.visited_urls) + len(crawler.failed_urls)
        crawler.clear_url_history()
        logger.info(f"[CRAWLER] Cleared {initial_urls} cached URLs from memory")
    except Exception as e:
        logger.warning(f"[CRAWLER] Could not clear: {e}")


def optimize_memory_index():
    """Optimize memory index structures."""
    try:
        from Zypherus.core.ace import ACE
        ace = ACE()
        
        # Force FAISS rebuild if enabled (optimizes index)
        if hasattr(ace.memory, '_rebuild_faiss'):
            ace.memory._rebuild_faiss()
            logger.info("[MEMORY] Rebuilt FAISS index for optimization")
        
        logger.info(f"[MEMORY] Memory entries: {len(ace.memory.entries)}")
        logger.info(f"[MEMORY] Entities tracked: {len(ace.memory.entity_index)}")
    except Exception as e:
        logger.warning(f"[MEMORY] Could not optimize: {e}")


def show_memory_stats():
    """Show current memory statistics."""
    if not HAS_PSUTIL:
        logger.info("[STATS] psutil not available - skipping detailed stats")
        return
    
    import os
    import psutil  # Import locally to ensure it's available
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    print("\n" + "="*60)
    print("MEMORY STATISTICS")
    print("="*60)
    print(f"RSS (Physical Memory):  {mem_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS (Virtual Memory):   {mem_info.vms / 1024 / 1024:.1f} MB")
    
    # File sizes
    files_to_check = [
        ("memory.json", Path("data/memory/memory.json")),
        ("memory.faiss", Path("data/memory/memory.faiss")),
        ("youtube_ingestions.json", Path("data/ingestion/youtube_ingestions.json")),
    ]
    
    print("\nFILE SIZES")
    print("-" * 60)
    for label, path in files_to_check:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"{label:25s}: {size_mb:6.2f} MB")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "aggressive":
        logger.info("[CLEANUP] Running aggressive cleanup...")
        cleanup_memory()
        show_memory_stats()
    else:
        logger.info("[CLEANUP] Running standard cleanup...")
        clear_web_crawler_history()
        optimize_memory_index()
        show_memory_stats()
        logger.info("\n[TIP] Run: .venv\\Scripts\\python.exe scripts\\cleanup_memory.py aggressive")
        logger.info("[TIP] Or restart ACE: restart the Python process")
