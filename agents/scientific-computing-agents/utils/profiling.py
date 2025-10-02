"""
Profiling utilities for performance monitoring and optimization.

This module provides decorators, context managers, and utilities for tracking
execution time, memory usage, and identifying performance bottlenecks in
scientific computing agents.
"""

import time
import functools
import tracemalloc
from typing import Any, Callable, Dict, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""

    function_name: str
    execution_time: float  # seconds
    memory_peak: Optional[float] = None  # MB
    memory_current: Optional[float] = None  # MB
    call_count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'function_name': self.function_name,
            'execution_time': self.execution_time,
            'memory_peak_mb': self.memory_peak,
            'memory_current_mb': self.memory_current,
            'call_count': self.call_count,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"Function: {self.function_name}",
            f"Execution Time: {self.execution_time:.4f} s",
        ]
        if self.memory_peak is not None:
            lines.append(f"Peak Memory: {self.memory_peak:.2f} MB")
        if self.memory_current is not None:
            lines.append(f"Current Memory: {self.memory_current:.2f} MB")
        lines.append(f"Call Count: {self.call_count}")
        if self.metadata:
            lines.append(f"Metadata: {self.metadata}")
        return "\n".join(lines)


class PerformanceTracker:
    """Global performance tracking registry."""

    _metrics: Dict[str, List[PerformanceMetrics]] = {}

    @classmethod
    def record(cls, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        if metrics.function_name not in cls._metrics:
            cls._metrics[metrics.function_name] = []
        cls._metrics[metrics.function_name].append(metrics)

    @classmethod
    def get_metrics(cls, function_name: Optional[str] = None) -> Dict[str, List[PerformanceMetrics]]:
        """Get recorded metrics for a function or all functions."""
        if function_name:
            return {function_name: cls._metrics.get(function_name, [])}
        return cls._metrics.copy()

    @classmethod
    def get_summary(cls, function_name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a function."""
        metrics_list = cls._metrics.get(function_name, [])
        if not metrics_list:
            return None

        times = [m.execution_time for m in metrics_list]
        memories = [m.memory_peak for m in metrics_list if m.memory_peak is not None]

        summary = {
            'function_name': function_name,
            'call_count': len(metrics_list),
            'total_time': sum(times),
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
        }

        if memories:
            summary['mean_memory_mb'] = sum(memories) / len(memories)
            summary['peak_memory_mb'] = max(memories)

        return summary

    @classmethod
    def clear(cls, function_name: Optional[str] = None) -> None:
        """Clear recorded metrics."""
        if function_name:
            cls._metrics.pop(function_name, None)
        else:
            cls._metrics.clear()

    @classmethod
    def generate_report(cls) -> str:
        """Generate a comprehensive performance report."""
        if not cls._metrics:
            return "No performance data recorded."

        lines = ["=" * 70, "Performance Report", "=" * 70, ""]

        for func_name in sorted(cls._metrics.keys()):
            summary = cls.get_summary(func_name)
            if summary:
                lines.extend([
                    f"Function: {func_name}",
                    f"  Calls: {summary['call_count']}",
                    f"  Total Time: {summary['total_time']:.4f} s",
                    f"  Mean Time: {summary['mean_time']:.4f} s",
                    f"  Min Time: {summary['min_time']:.4f} s",
                    f"  Max Time: {summary['max_time']:.4f} s",
                ])
                if 'mean_memory_mb' in summary:
                    lines.extend([
                        f"  Mean Memory: {summary['mean_memory_mb']:.2f} MB",
                        f"  Peak Memory: {summary['peak_memory_mb']:.2f} MB",
                    ])
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


@contextmanager
def timer(name: str = "Operation", track: bool = False):
    """
    Context manager for timing code blocks.

    Args:
        name: Name of the operation being timed
        track: If True, record metrics to PerformanceTracker

    Yields:
        Dictionary that will contain 'elapsed' key with execution time

    Example:
        with timer("My operation") as t:
            # ... code ...
            pass
        print(f"Took {t['elapsed']:.4f} seconds")
    """
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result['elapsed'] = elapsed
        print(f"{name}: {elapsed:.4f} s")

        if track:
            metrics = PerformanceMetrics(
                function_name=name,
                execution_time=elapsed
            )
            PerformanceTracker.record(metrics)


@contextmanager
def memory_tracker(name: str = "Operation", track: bool = False):
    """
    Context manager for tracking memory usage.

    Args:
        name: Name of the operation being tracked
        track: If True, record metrics to PerformanceTracker

    Yields:
        Dictionary that will contain 'peak_mb' and 'current_mb' keys

    Example:
        with memory_tracker("My operation") as m:
            # ... code ...
            pass
        print(f"Peak memory: {m['peak_mb']:.2f} MB")
    """
    result = {}
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    try:
        yield result
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        result['current_mb'] = current_mb
        result['peak_mb'] = peak_mb

        print(f"{name} - Memory: Current={current_mb:.2f} MB, Peak={peak_mb:.2f} MB")

        if track:
            metrics = PerformanceMetrics(
                function_name=name,
                execution_time=0.0,  # Not timed in this context
                memory_peak=peak_mb,
                memory_current=current_mb
            )
            PerformanceTracker.record(metrics)


def profile_performance(track_memory: bool = True, track_global: bool = True):
    """
    Decorator for profiling function execution time and memory usage.

    Args:
        track_memory: If True, track memory usage (adds overhead)
        track_global: If True, record metrics to global PerformanceTracker

    Returns:
        Decorated function that prints and optionally tracks performance

    Example:
        @profile_performance(track_memory=True)
        def expensive_operation(n):
            return sum(range(n))
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Start timing
            start_time = time.perf_counter()

            # Start memory tracking if requested
            if track_memory:
                tracemalloc.start()

            try:
                # Execute function
                result = func(*args, **kwargs)
                return result

            finally:
                # Stop timing
                elapsed_time = time.perf_counter() - start_time

                # Stop memory tracking
                peak_mb = None
                current_mb = None
                if track_memory:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    current_mb = current / 1024 / 1024
                    peak_mb = peak / 1024 / 1024

                # Create metrics
                metrics = PerformanceMetrics(
                    function_name=func_name,
                    execution_time=elapsed_time,
                    memory_peak=peak_mb,
                    memory_current=current_mb
                )

                # Print performance info
                print(f"\n{'='*60}")
                print(f"Performance: {func_name}")
                print(f"{'='*60}")
                print(f"Execution Time: {elapsed_time:.4f} s")
                if track_memory and peak_mb is not None:
                    print(f"Peak Memory: {peak_mb:.2f} MB")
                    print(f"Current Memory: {current_mb:.2f} MB")
                print(f"{'='*60}\n")

                # Track globally if requested
                if track_global:
                    PerformanceTracker.record(metrics)

        return wrapper
    return decorator


def compare_performance(funcs: List[Callable], *args, **kwargs) -> Dict[str, PerformanceMetrics]:
    """
    Compare performance of multiple functions with the same inputs.

    Args:
        funcs: List of functions to compare
        *args, **kwargs: Arguments to pass to all functions

    Returns:
        Dictionary mapping function names to their performance metrics

    Example:
        def method_a(n): return sum(range(n))
        def method_b(n): return n * (n-1) // 2

        results = compare_performance([method_a, method_b], 1000000)
    """
    results = {}

    print(f"\n{'='*70}")
    print("Performance Comparison")
    print(f"{'='*70}\n")

    for func in funcs:
        func_name = func.__name__

        # Time execution
        start = time.perf_counter()
        tracemalloc.start()

        try:
            func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            metrics = PerformanceMetrics(
                function_name=func_name,
                execution_time=elapsed,
                memory_peak=peak / 1024 / 1024,
                memory_current=current / 1024 / 1024
            )
            results[func_name] = metrics

            print(f"{func_name}:")
            print(f"  Time: {elapsed:.4f} s")
            print(f"  Memory: {peak / 1024 / 1024:.2f} MB")
            print()

    # Find fastest
    fastest = min(results.values(), key=lambda m: m.execution_time)
    print(f"Fastest: {fastest.function_name} ({fastest.execution_time:.4f} s)")

    # Show speedups
    print(f"\nSpeedups relative to {fastest.function_name}:")
    for name, metrics in results.items():
        if name != fastest.function_name:
            speedup = metrics.execution_time / fastest.execution_time
            print(f"  {name}: {speedup:.2f}x slower")

    print(f"{'='*70}\n")

    return results


def benchmark(n_runs: int = 10, warmup: int = 1):
    """
    Decorator for benchmarking functions with multiple runs.

    Args:
        n_runs: Number of benchmark runs
        warmup: Number of warmup runs (not counted)

    Returns:
        Decorated function that runs benchmarks and reports statistics

    Example:
        @benchmark(n_runs=10)
        def test_algorithm():
            # ... code to benchmark ...
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Warmup runs
            for _ in range(warmup):
                func(*args, **kwargs)

            # Benchmark runs
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            # Compute statistics
            mean_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = (sum((t - mean_time)**2 for t in times) / len(times))**0.5

            # Print results
            print(f"\n{'='*60}")
            print(f"Benchmark: {func_name}")
            print(f"{'='*60}")
            print(f"Runs: {n_runs} (+ {warmup} warmup)")
            print(f"Mean Time: {mean_time:.6f} s")
            print(f"Std Dev: {std_dev:.6f} s")
            print(f"Min Time: {min_time:.6f} s")
            print(f"Max Time: {max_time:.6f} s")
            print(f"{'='*60}\n")

            return mean_time

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    @profile_performance(track_memory=True)
    def example_function(n):
        """Example function to demonstrate profiling."""
        data = list(range(n))
        result = sum(data)
        return result

    # Test the profiling
    result = example_function(1000000)

    # Test timer context manager
    with timer("List comprehension"):
        squares = [x**2 for x in range(1000000)]

    # Generate report
    print(PerformanceTracker.generate_report())
