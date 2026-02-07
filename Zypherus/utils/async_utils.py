"""Asynchronous and concurrent operations for ACE."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional, Dict, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools

logger = logging.getLogger("ZYPHERUS.Async")


class AsyncExecutor:
    """Manage async/concurrent operations."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            if self.loop is None or self.loop.is_closed():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            return self.loop
    
    async def run_in_executor(self, func: Callable, *args: Any) -> Any:
        """Run blocking function in thread executor."""
        loop = self.get_loop()
        return await loop.run_in_executor(self.thread_executor, func, *args)
    
    async def gather_tasks(self, *coros: Coroutine) -> List[Any]:
        """Gather multiple coroutines."""
        return await asyncio.gather(*coros, return_exceptions=True)
    
    async def run_with_timeout(self, coro: Coroutine, timeout_s: float) -> Any:
        """Run coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout_s}s")
            raise
    
    def close(self):
        """Cleanup resources."""
        self.thread_executor.shutdown(wait=True)
        if self.loop and not self.loop.is_closed():
            self.loop.close()


class AsyncBatcher:
    """Batch processing with async support."""
    
    def __init__(self, batch_size: int = 32, timeout_s: float = 30.0):
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.queue: asyncio.Queue = None
        self.executor = AsyncExecutor()
    
    async def process_batch(self, processor: Callable, items: List[Any]) -> List[Any]:
        """Process items in batches concurrently."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch items concurrently
            tasks = [
                self.executor.run_with_timeout(
                    self._process_item(processor, item),
                    self.timeout_s
                )
                for item in batch
            ]
            
            batch_results = await self.executor.gather_tasks(*tasks)
            results.extend(batch_results)
        
        return results
    
    async def _process_item(self, processor: Callable, item: Any) -> Any:
        """Process single item."""
        try:
            if asyncio.iscoroutinefunction(processor):
                return await processor(item)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, processor, item)
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            return None


class AsyncPool:
    """Connection/resource pool for async operations."""
    
    def __init__(self, factory: Callable, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self.available: asyncio.Queue = None
        self.in_use = 0
    
    async def initialize(self):
        """Initialize pool."""
        self.available = asyncio.Queue(maxsize=self.max_size)
        for _ in range(self.max_size):
            resource = await asyncio.to_thread(self.factory)
            await self.available.put(resource)
    
    async def acquire(self) -> Any:
        """Acquire resource from pool."""
        if self.available is None:
            await self.initialize()
        
        resource = await self.available.get()
        self.in_use += 1
        return resource
    
    async def release(self, resource: Any) -> None:
        """Release resource back to pool."""
        if self.available:
            await self.available.put(resource)
            self.in_use -= 1


def async_timer(name: str) -> Callable:
    """Decorator for timing async operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                logger.debug(f"{name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"{name} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator


class AsyncQueue:
    """Priority queue for async task scheduling."""
    
    def __init__(self, max_size: int = 100):
        self.queue: asyncio.PriorityQueue = None
        self.max_size = max_size
    
    async def put(self, item: Any, priority: int = 0) -> None:
        """Put item in queue with priority."""
        if self.queue is None:
            self.queue = asyncio.PriorityQueue(maxsize=self.max_size)
        
        await self.queue.put((priority, item))
    
    async def get(self) -> Any:
        """Get highest priority item."""
        if self.queue is None:
            self.queue = asyncio.PriorityQueue(maxsize=self.max_size)
        
        priority, item = await self.queue.get()
        return item
    
    async def get_many(self, count: int) -> List[Any]:
        """Get multiple items."""
        items = []
        for _ in range(count):
            try:
                item = self.queue.get_nowait()
                items.append(item)
            except asyncio.QueueEmpty:
                break
        return items


class RateLimiter:
    """Rate limiting for async operations."""
    
    def __init__(self, max_per_second: float = 10):
        self.max_per_second = max_per_second
        self.tokens = max_per_second
        self.last_update = None
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1) -> None:
        """Acquire tokens from rate limiter."""
        async with self.lock:
            import time
            now = time.time()
            
            if self.last_update is None:
                self.last_update = now
            
            elapsed = now - self.last_update
            self.tokens += elapsed * self.max_per_second
            self.tokens = min(self.tokens, self.max_per_second)
            self.last_update = now
            
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.max_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= tokens


async def parallel_map(func: Callable, items: List[Any], 
                      max_concurrent: int = 4) -> List[Any]:
    """Map function over items with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_func(item: Any) -> Any:
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, item)
    
    return await asyncio.gather(*[bounded_func(item) for item in items])


__all__ = [
    "AsyncExecutor",
    "AsyncBatcher",
    "AsyncPool",
    "AsyncQueue",
    "RateLimiter",
    "async_timer",
    "parallel_map",
]
