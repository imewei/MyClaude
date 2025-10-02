"""Performance Profiling and Optimization Tools.

This module provides comprehensive performance analysis, profiling, and
optimization utilities for optimal control computations.

Features:
- Time profiling (function-level, line-level)
- Memory profiling and leak detection
- Benchmarking framework with regression detection
- Optimization strategies (caching, JIT, vectorization)
- Performance reporting and visualization
- Automated optimization recommendations

Author: Nonequilibrium Physics Agents
Week: 27-28 of Phase 4
"""

import time
import functools
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import defaultdict
import numpy as np
import warnings

try:
    import cProfile
    import pstats
    from io import StringIO
    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False


@dataclass
class ProfilingResult:
    """Results from profiling."""
    function_name: str
    total_time: float
    num_calls: int
    time_per_call: float
    memory_peak: Optional[float] = None
    memory_delta: Optional[float] = None
    line_stats: Optional[Dict] = None


@dataclass
class BenchmarkResult:
    """Results from benchmark."""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    num_iterations: int
    memory_used: Optional[float] = None


class Timer:
    """High-precision timer for benchmarking."""

    def __init__(self, name: str = "Timer"):
        """Initialize timer.

        Args:
            name: Timer name for display
        """
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop timing."""
        self.elapsed = time.perf_counter() - self.start_time

    def __str__(self):
        """String representation."""
        if self.elapsed is not None:
            return f"{self.name}: {self.elapsed:.6f} seconds"
        return f"{self.name}: Not started"


class FunctionProfiler:
    """Profile function execution time and memory."""

    def __init__(self):
        """Initialize profiler."""
        self.profile_data = {}

    def profile(
        self,
        func: Callable,
        *args,
        track_memory: bool = True,
        **kwargs
    ) -> Tuple[Any, ProfilingResult]:
        """Profile a function call.

        Args:
            func: Function to profile
            *args: Function arguments
            track_memory: Track memory usage
            **kwargs: Function keyword arguments

        Returns:
            (result, profiling_result)
        """
        # Start memory tracking
        if track_memory:
            tracemalloc.start()
            mem_before = tracemalloc.get_traced_memory()[0]

        # Time execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time

        # Memory tracking
        mem_peak = None
        mem_delta = None
        if track_memory:
            mem_after, mem_peak = tracemalloc.get_traced_memory()
            mem_delta = mem_after - mem_before
            tracemalloc.stop()

        # Create result
        prof_result = ProfilingResult(
            function_name=func.__name__,
            total_time=elapsed,
            num_calls=1,
            time_per_call=elapsed,
            memory_peak=mem_peak / 1024 / 1024 if mem_peak else None,  # MB
            memory_delta=mem_delta / 1024 / 1024 if mem_delta else None  # MB
        )

        return result, prof_result

    def profile_multiple(
        self,
        func: Callable,
        args_list: List[Tuple],
        track_memory: bool = False
    ) -> ProfilingResult:
        """Profile function with multiple inputs.

        Args:
            func: Function to profile
            args_list: List of argument tuples
            track_memory: Track memory usage

        Returns:
            Aggregated profiling result
        """
        times = []
        total_mem = 0.0

        for args in args_list:
            _, prof = self.profile(func, *args, track_memory=track_memory)
            times.append(prof.total_time)
            if prof.memory_delta:
                total_mem += prof.memory_delta

        return ProfilingResult(
            function_name=func.__name__,
            total_time=sum(times),
            num_calls=len(args_list),
            time_per_call=np.mean(times),
            memory_peak=total_mem if track_memory else None
        )


class CProfileWrapper:
    """Wrapper for cProfile for detailed profiling."""

    def __init__(self):
        """Initialize cProfile wrapper."""
        if not CPROFILE_AVAILABLE:
            raise ImportError("cProfile not available")

        self.profiler = None
        self.stats = None

    def profile(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """Profile function with cProfile.

        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            (result, stats_string)
        """
        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()

        # Get stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions

        return result, s.getvalue()


class Benchmarker:
    """Benchmark framework for performance testing."""

    def __init__(self):
        """Initialize benchmarker."""
        self.results = {}

    def benchmark(
        self,
        name: str,
        func: Callable,
        num_iterations: int = 100,
        warmup: int = 10,
        track_memory: bool = False
    ) -> BenchmarkResult:
        """Benchmark a function.

        Args:
            name: Benchmark name
            func: Function to benchmark (no arguments)
            num_iterations: Number of iterations
            warmup: Warmup iterations (not counted)
            track_memory: Track memory usage

        Returns:
            Benchmark result
        """
        # Warmup
        for _ in range(warmup):
            func()

        # Benchmark
        times = []
        mem_total = 0.0

        for _ in range(num_iterations):
            if track_memory:
                tracemalloc.start()

            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if track_memory:
                _, peak = tracemalloc.get_traced_memory()
                mem_total += peak
                tracemalloc.stop()

        times = np.array(times)

        result = BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            num_iterations=num_iterations,
            memory_used=mem_total / num_iterations / 1024 / 1024 if track_memory else None
        )

        self.results[name] = result
        return result

    def compare(
        self,
        benchmarks: Dict[str, Callable],
        num_iterations: int = 100
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple implementations.

        Args:
            benchmarks: Dict of name -> function
            num_iterations: Iterations per benchmark

        Returns:
            Dict of benchmark results
        """
        results = {}

        for name, func in benchmarks.items():
            result = self.benchmark(name, func, num_iterations)
            results[name] = result

        return results

    def detect_regression(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """Detect performance regression.

        Args:
            current: Current benchmark result
            baseline: Baseline benchmark result
            threshold: Regression threshold (10% = 0.1)

        Returns:
            (is_regression, relative_change)
        """
        relative_change = (current.mean_time - baseline.mean_time) / baseline.mean_time

        is_regression = relative_change > threshold

        return is_regression, relative_change


class CacheOptimizer:
    """Implement caching strategies for optimization."""

    def __init__(self, max_cache_size: int = 1000):
        """Initialize cache optimizer.

        Args:
            max_cache_size: Maximum cache entries
        """
        self.max_cache_size = max_cache_size
        self.cache_stats = {'hits': 0, 'misses': 0}

    def memoize(self, func: Callable) -> Callable:
        """Memoization decorator with LRU cache.

        Args:
            func: Function to memoize

        Returns:
            Memoized function
        """
        cache = {}
        access_order = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))

            if key in cache:
                self.cache_stats['hits'] += 1
                # Update access order (move to end)
                access_order.remove(key)
                access_order.append(key)
                return cache[key]

            self.cache_stats['misses'] += 1

            # Compute result
            result = func(*args, **kwargs)

            # Add to cache
            cache[key] = result
            access_order.append(key)

            # Evict if cache full (LRU)
            if len(cache) > self.max_cache_size:
                oldest = access_order.pop(0)
                del cache[oldest]

            return result

        wrapper.cache_info = lambda: self.cache_stats.copy()
        wrapper.cache_clear = lambda: (cache.clear(), access_order.clear())

        return wrapper


class VectorizationOptimizer:
    """Optimize code via vectorization."""

    @staticmethod
    def vectorize_loop(
        func: Callable,
        inputs: np.ndarray,
        axis: int = 0
    ) -> np.ndarray:
        """Vectorize a function over an array.

        Args:
            func: Function to vectorize (scalar input)
            inputs: Array of inputs
            axis: Axis to vectorize over

        Returns:
            Vectorized results
        """
        vfunc = np.vectorize(func)
        return vfunc(inputs)

    @staticmethod
    def replace_loops_with_operations(
        array: np.ndarray,
        operation: str
    ) -> np.ndarray:
        """Replace loops with vectorized operations.

        Args:
            array: Input array
            operation: Operation name ('sum', 'mean', 'cumsum', etc.)

        Returns:
            Result of vectorized operation
        """
        ops = {
            'sum': np.sum,
            'mean': np.mean,
            'cumsum': np.cumsum,
            'max': np.max,
            'min': np.min,
            'std': np.std
        }

        if operation in ops:
            return ops[operation](array)
        else:
            raise ValueError(f"Unknown operation: {operation}")


class MemoryProfiler:
    """Profile memory usage and detect leaks."""

    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots = []

    def take_snapshot(self, label: str = ""):
        """Take memory snapshot.

        Args:
            label: Snapshot label
        """
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))

    def compare_snapshots(
        self,
        idx1: int = 0,
        idx2: int = -1,
        top_n: int = 10
    ) -> List[Tuple[str, int]]:
        """Compare two snapshots.

        Args:
            idx1: First snapshot index
            idx2: Second snapshot index
            top_n: Number of top differences to return

        Returns:
            List of (filename, size_diff) tuples
        """
        if len(self.snapshots) < 2:
            return []

        label1, snap1 = self.snapshots[idx1]
        label2, snap2 = self.snapshots[idx2]

        stats = snap2.compare_to(snap1, 'lineno')

        # Top differences
        top_stats = []
        for stat in stats[:top_n]:
            top_stats.append((str(stat), stat.size_diff))

        return top_stats

    def detect_leaks(self, threshold_mb: float = 10.0) -> List[str]:
        """Detect potential memory leaks.

        Args:
            threshold_mb: Threshold in MB

        Returns:
            List of potential leak locations
        """
        if len(self.snapshots) < 2:
            return []

        leaks = []
        diffs = self.compare_snapshots(top_n=50)

        for stat_str, size_diff in diffs:
            size_mb = size_diff / 1024 / 1024
            if size_mb > threshold_mb:
                leaks.append(f"{stat_str}: +{size_mb:.2f} MB")

        return leaks


class PerformanceReporter:
    """Generate performance reports."""

    def __init__(self):
        """Initialize reporter."""
        self.reports = []

    def generate_report(
        self,
        profiling_results: List[ProfilingResult],
        benchmark_results: List[BenchmarkResult]
    ) -> str:
        """Generate performance report.

        Args:
            profiling_results: List of profiling results
            benchmark_results: List of benchmark results

        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 70)

        # Profiling section
        if profiling_results:
            report.append("\nPROFILING RESULTS:")
            report.append("-" * 70)

            for prof in profiling_results:
                report.append(f"\nFunction: {prof.function_name}")
                report.append(f"  Total time: {prof.total_time:.6f} s")
                report.append(f"  Num calls: {prof.num_calls}")
                report.append(f"  Time/call: {prof.time_per_call:.6f} s")

                if prof.memory_peak is not None:
                    report.append(f"  Peak memory: {prof.memory_peak:.2f} MB")
                if prof.memory_delta is not None:
                    report.append(f"  Memory delta: {prof.memory_delta:.2f} MB")

        # Benchmark section
        if benchmark_results:
            report.append("\n\nBENCHMARK RESULTS:")
            report.append("-" * 70)

            for bench in benchmark_results:
                report.append(f"\nBenchmark: {bench.name}")
                report.append(f"  Mean time: {bench.mean_time*1000:.3f} ms")
                report.append(f"  Std dev: {bench.std_time*1000:.3f} ms")
                report.append(f"  Min time: {bench.min_time*1000:.3f} ms")
                report.append(f"  Max time: {bench.max_time*1000:.3f} ms")
                report.append(f"  Iterations: {bench.num_iterations}")

                if bench.memory_used is not None:
                    report.append(f"  Memory: {bench.memory_used:.2f} MB")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def recommend_optimizations(
        self,
        profiling_results: List[ProfilingResult]
    ) -> List[str]:
        """Generate optimization recommendations.

        Args:
            profiling_results: Profiling results

        Returns:
            List of recommendations
        """
        recommendations = []

        for prof in profiling_results:
            # Check for slow functions
            if prof.time_per_call > 1.0:
                recommendations.append(
                    f"{prof.function_name}: Very slow ({prof.time_per_call:.2f}s). "
                    "Consider caching, vectorization, or JIT compilation."
                )

            # Check for high memory usage
            if prof.memory_peak and prof.memory_peak > 100:
                recommendations.append(
                    f"{prof.function_name}: High memory usage ({prof.memory_peak:.1f} MB). "
                    "Consider using generators or processing in chunks."
                )

            # Check for many calls
            if prof.num_calls > 1000 and prof.time_per_call > 0.001:
                recommendations.append(
                    f"{prof.function_name}: Called {prof.num_calls} times. "
                    "Consider batching or memoization."
                )

        if not recommendations:
            recommendations.append("No major performance issues detected.")

        return recommendations


class PerformanceOptimizer:
    """Automated performance optimization."""

    def __init__(self):
        """Initialize optimizer."""
        self.optimizations_applied = []

    def apply_caching(
        self,
        func: Callable,
        max_cache_size: int = 1000
    ) -> Callable:
        """Apply caching optimization.

        Args:
            func: Function to optimize
            max_cache_size: Maximum cache size

        Returns:
            Optimized function
        """
        cache_opt = CacheOptimizer(max_cache_size)
        optimized = cache_opt.memoize(func)

        self.optimizations_applied.append(f"Caching applied to {func.__name__}")

        return optimized

    def apply_vectorization(
        self,
        loop_func: Callable,
        data: np.ndarray
    ) -> np.ndarray:
        """Apply vectorization optimization.

        Args:
            loop_func: Loop function to vectorize
            data: Data to process

        Returns:
            Vectorized result
        """
        vec_opt = VectorizationOptimizer()
        result = vec_opt.vectorize_loop(loop_func, data)

        self.optimizations_applied.append("Vectorization applied")

        return result

    def get_optimization_summary(self) -> str:
        """Get summary of applied optimizations.

        Returns:
            Summary string
        """
        if not self.optimizations_applied:
            return "No optimizations applied."

        summary = "Optimizations Applied:\n"
        for i, opt in enumerate(self.optimizations_applied, 1):
            summary += f"  {i}. {opt}\n"

        return summary


# Backward compatibility aliases
# ProfilerConfig: alias for configuration
@dataclass
class ProfilerConfig:
    """Configuration for PerformanceProfiler (alias for backward compatibility).

    Attributes
    ----------
    enabled : bool
        Whether profiling is enabled
    track_memory : bool
        Whether to track memory usage
    num_iterations : int
        Number of benchmark iterations
    warmup_iterations : int
        Number of warmup iterations
    """
    enabled: bool = True
    track_memory: bool = True
    num_iterations: int = 100
    warmup_iterations: int = 10


# PerformanceProfiler: unified interface combining FunctionProfiler and Benchmarker
class PerformanceProfiler:
    """Unified performance profiler (combines FunctionProfiler and Benchmarker).

    Provides a unified interface for profiling and benchmarking that wraps
    the existing FunctionProfiler and Benchmarker classes for backward compatibility.

    Parameters
    ----------
    config : ProfilerConfig, optional
        Profiler configuration

    Methods
    -------
    profile(func, *args, **kwargs)
        Profile a function execution
    benchmark(name, func, *args, **kwargs)
        Benchmark a function with multiple iterations
    get_results()
        Get accumulated profiling results
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """Initialize unified profiler.

        Parameters
        ----------
        config : ProfilerConfig, optional
            Configuration. If None, uses defaults.
        """
        self.config = config or ProfilerConfig()
        self.function_profiler = FunctionProfiler()
        self.benchmarker = Benchmarker()
        self.results = []

    def profile(self, func: Callable, *args, **kwargs) -> Any:
        """Profile a function execution.

        Parameters
        ----------
        func : Callable
            Function to profile
        *args
            Positional arguments
        **kwargs
            Keyword arguments

        Returns
        -------
        Any
            Function result
        """
        if not self.config.enabled:
            return func(*args, **kwargs)

        result, prof_result = self.function_profiler.profile(
            func, *args,
            track_memory=self.config.track_memory,
            **kwargs
        )
        self.results.append(prof_result)
        return result

    def benchmark(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a function.

        Parameters
        ----------
        name : str
            Benchmark name
        func : Callable
            Function to benchmark
        *args
            Positional arguments
        **kwargs
            Keyword arguments

        Returns
        -------
        BenchmarkResult
            Benchmark statistics
        """
        if not self.config.enabled:
            return BenchmarkResult(
                name=name,
                mean_time=0.0,
                std_time=0.0,
                min_time=0.0,
                max_time=0.0,
                num_iterations=0
            )

        # Wrap function with args
        def wrapped_func():
            return func(*args, **kwargs)

        return self.benchmarker.benchmark(
            name=name,
            func=wrapped_func,
            num_iterations=self.config.num_iterations,
            warmup=self.config.warmup_iterations,
            track_memory=self.config.track_memory
        )

    def get_results(self) -> List[ProfilingResult]:
        """Get accumulated profiling results.

        Returns
        -------
        List[ProfilingResult]
            List of profiling results
        """
        return self.results


# Convenience decorators and functions for backward compatibility
def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution.

    Parameters
    ----------
    func : Callable
        Function to time

    Returns
    -------
    Callable
        Wrapped function that prints timing

    Examples
    --------
    >>> @timing_decorator
    ... def expensive_function():
    ...     # ... computation ...
    ...     pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"{func.__name__}: {elapsed:.6f}s")
        return result
    return wrapper


def memory_profiler(func: Callable) -> Callable:
    """Decorator to profile memory usage.

    Parameters
    ----------
    func : Callable
        Function to profile

    Returns
    -------
    Callable
        Wrapped function that reports memory usage

    Examples
    --------
    >>> @memory_profiler
    ... def memory_intensive_function():
    ...     # ... computation ...
    ...     pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        result = func(*args, **kwargs)

        mem_after, mem_peak = tracemalloc.get_traced_memory()
        mem_delta = (mem_after - mem_before) / 1024 / 1024
        mem_peak_mb = mem_peak / 1024 / 1024
        tracemalloc.stop()

        print(f"{func.__name__}: Δ{mem_delta:.2f}MB, Peak: {mem_peak_mb:.2f}MB")
        return result
    return wrapper


def benchmark_function(
    func: Callable,
    num_iterations: int = 100,
    warmup: int = 10,
    *args,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function with given arguments.

    Parameters
    ----------
    func : Callable
        Function to benchmark
    num_iterations : int
        Number of iterations
    warmup : int
        Warmup iterations
    *args
        Function arguments
    **kwargs
        Function keyword arguments

    Returns
    -------
    Dict[str, float]
        Benchmark statistics (mean, std, min, max times)

    Examples
    --------
    >>> def compute(x):
    ...     return x ** 2
    >>> stats = benchmark_function(compute, 100, 10, 5)
    >>> print(f"Mean: {stats['mean']:.6f}s")
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_array = np.array(times)

    return {
        'mean': float(np.mean(times_array)),
        'std': float(np.std(times_array)),
        'min': float(np.min(times_array)),
        'max': float(np.max(times_array)),
        'total': float(np.sum(times_array)),
        'iterations': num_iterations
    }


def vectorize_computation(func: Callable, inputs: np.ndarray) -> np.ndarray:
    """Vectorize a computation over an array of inputs.

    Parameters
    ----------
    func : Callable
        Function to vectorize (scalar input)
    inputs : np.ndarray
        Array of inputs

    Returns
    -------
    np.ndarray
        Vectorized results

    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> inputs = np.array([1, 2, 3, 4])
    >>> results = vectorize_computation(square, inputs)
    >>> print(results)  # [1, 4, 9, 16]
    """
    return VectorizationOptimizer.vectorize_loop(func, inputs)
