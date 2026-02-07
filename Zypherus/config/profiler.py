"""Performance profiling and monitoring utilities."""
import time
import logging
from functools import wraps
from typing import Callable, Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger("ZYPHERUS.Performance")


@dataclass
class PerformanceMetrics:
	"""Performance metrics for a function or operation."""
	name: str
	call_count: int = 0
	total_time: float = 0.0
	min_time: float = float('inf')
	max_time: float = 0.0
	times: List[float] = field(default_factory=list)
	errors: int = 0
	
	@property
	def avg_time(self) -> float:
		"""Average execution time."""
		return self.total_time / self.call_count if self.call_count > 0 else 0
	
	@property
	def median_time(self) -> float:
		"""Median execution time."""
		return statistics.median(self.times) if self.times else 0
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"name": self.name,
			"call_count": self.call_count,
			"total_time": round(self.total_time, 4),
			"avg_time": round(self.avg_time, 4),
			"min_time": round(self.min_time, 4) if self.min_time != float('inf') else 0,
			"max_time": round(self.max_time, 4),
			"median_time": round(self.median_time, 4),
			"errors": self.errors
		}


class PerformanceProfiler:
	"""Global performance profiler."""
	
	_instance = None
	_metrics: Dict[str, PerformanceMetrics] = {}
	
	def __new__(cls):
		"""Singleton pattern."""
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	
	def reset(self):
		"""Clear all metrics."""
		self._metrics.clear()
	
	def record_time(self, name: str, duration: float, error: bool = False):
		"""Record execution time for a function.
		
		Args:
			name: Function or operation name
			duration: Execution time in seconds
			error: Whether an error occurred
		"""
		if name not in self._metrics:
			self._metrics[name] = PerformanceMetrics(name=name)
		
		metrics = self._metrics[name]
		metrics.call_count += 1
		metrics.total_time += duration
		metrics.min_time = min(metrics.min_time, duration)
		metrics.max_time = max(metrics.max_time, duration)
		metrics.times.append(duration)
		if error:
			metrics.errors += 1
		
		# Keep only last 1000 times to prevent memory growth
		if len(metrics.times) > 1000:
			metrics.times = metrics.times[-1000:]
	
	def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
		"""Get metrics.
		
		Args:
			name: Specific function name, or None for all
			
		Returns:
			Metrics dictionary
		"""
		if name:
			return self._metrics.get(name, PerformanceMetrics(name=name)).to_dict()
		
		return {name: metrics.to_dict() for name, metrics in self._metrics.items()}
	
	def get_slowest_operations(self, top_n: int = 10) -> List[Dict[str, Any]]:
		"""Get slowest operations by average time.
		
		Args:
			top_n: Number of operations to return
			
		Returns:
			List of slowest operations
		"""
		sorted_ops = sorted(
			self._metrics.values(),
			key=lambda m: m.avg_time,
			reverse=True
		)
		return [op.to_dict() for op in sorted_ops[:top_n]]
	
	def get_most_called_operations(self, top_n: int = 10) -> List[Dict[str, Any]]:
		"""Get most frequently called operations.
		
		Args:
			top_n: Number of operations to return
			
		Returns:
			List of most called operations
		"""
		sorted_ops = sorted(
			self._metrics.values(),
			key=lambda m: m.call_count,
			reverse=True
		)
		return [op.to_dict() for op in sorted_ops[:top_n]]
	
	def get_summary(self) -> str:
		"""Get performance summary report."""
		if not self._metrics:
			return "No performance data collected."
		
		lines = ["Performance Summary:", "==================="]
		
		slowest = self.get_slowest_operations(5)
		if slowest:
			lines.append("\nTop 5 Slowest Operations:")
			for op in slowest:
				lines.append(
					f"  {op['name']}: "
					f"avg={op['avg_time']:.4f}s, "
					f"total={op['total_time']:.2f}s, "
					f"calls={op['call_count']}"
				)
		
		most_called = self.get_most_called_operations(5)
		if most_called:
			lines.append("\nTop 5 Most Called Operations:")
			for op in most_called:
				lines.append(
					f"  {op['name']}: "
					f"calls={op['call_count']}, "
					f"avg={op['avg_time']:.4f}s"
				)
		
		return "\n".join(lines)


def profile_function(func: Callable) -> Callable:
	"""Decorator to profile function execution time.
	
	Usage:
		@profile_function
		def my_function():
			pass
	"""
	@wraps(func)
	def wrapper(*args, **kwargs) -> Any:
		profiler = PerformanceProfiler()
		start_time = time.time()
		try:
			result = func(*args, **kwargs)
			return result
		except Exception as e:
			profiler.record_time(func.__name__, time.time() - start_time, error=True)
			raise
		finally:
			duration = time.time() - start_time
			profiler.record_time(func.__name__, duration)
	
	return wrapper


def profile_method(method: Callable) -> Callable:
	"""Decorator to profile method execution time.
	
	Usage:
		class MyClass:
			@profile_method
			def my_method(self):
				pass
	"""
	@wraps(method)
	def wrapper(self, *args, **kwargs) -> Any:
		profiler = PerformanceProfiler()
		func_name = f"{self.__class__.__name__}.{method.__name__}"
		start_time = time.time()
		try:
			result = method(self, *args, **kwargs)
			return result
		except Exception as e:
			profiler.record_time(func_name, time.time() - start_time, error=True)
			raise
		finally:
			duration = time.time() - start_time
			profiler.record_time(func_name, duration)
	
	return wrapper


class PerformanceContext:
	"""Context manager for profiling specific code blocks."""
	
	def __init__(self, operation_name: str):
		"""Initialize performance context.
		
		Args:
			operation_name: Name of the operation being profiled
		"""
		self.operation_name = operation_name
		self.start_time = None
		self.profiler = PerformanceProfiler()
	
	def __enter__(self):
		"""Enter context."""
		self.start_time = time.time()
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Exit context and record time."""
		if self.start_time is not None:
			duration = time.time() - self.start_time
			error = exc_type is not None
			self.profiler.record_time(self.operation_name, duration, error=error)


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
	"""Get global profiler instance."""
	return _profiler
