"""Tests for Performance Profiling and Optimization.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.performance import (
    Timer,
    FunctionProfiler,
    Benchmarker,
    CacheOptimizer,
    VectorizationOptimizer,
    MemoryProfiler,
    PerformanceReporter,
    PerformanceOptimizer,
    ProfilingResult,
    BenchmarkResult
)


class TestTimer:
    """Tests for Timer."""

    def test_timer_context_manager(self):
        """Test: Timer as context manager."""
        with Timer("test") as t:
            time.sleep(0.01)

        assert t.elapsed is not None
        assert t.elapsed >= 0.01
        assert "test" in str(t)

    def test_timer_multiple_uses(self):
        """Test: Timer multiple times."""
        timer = Timer()

        with timer:
            time.sleep(0.005)
        elapsed1 = timer.elapsed

        with timer:
            time.sleep(0.01)
        elapsed2 = timer.elapsed

        assert elapsed2 > elapsed1


class TestFunctionProfiler:
    """Tests for FunctionProfiler."""

    def test_profile_simple_function(self):
        """Test: Profile simple function."""
        profiler = FunctionProfiler()

        def test_func(n):
            return sum(range(n))

        result, prof = profiler.profile(test_func, 1000, track_memory=False)

        assert result == sum(range(1000))
        assert prof.function_name == "test_func"
        assert prof.total_time > 0
        assert prof.num_calls == 1

    def test_profile_with_memory(self):
        """Test: Profile with memory tracking."""
        profiler = FunctionProfiler()

        def allocate_memory():
            return [0] * 1000000

        result, prof = profiler.profile(allocate_memory, track_memory=True)

        assert len(result) == 1000000
        assert prof.memory_peak is not None
        assert prof.memory_delta is not None
        assert prof.memory_delta > 0

    def test_profile_multiple_calls(self):
        """Test: Profile multiple function calls."""
        profiler = FunctionProfiler()

        def square(x):
            return x ** 2

        args_list = [(i,) for i in range(10)]

        prof = profiler.profile_multiple(square, args_list, track_memory=False)

        assert prof.num_calls == 10
        assert prof.total_time > 0


class TestBenchmarker:
    """Tests for Benchmarker."""

    def test_benchmark_simple_function(self):
        """Test: Benchmark simple function."""
        benchmarker = Benchmarker()

        def test_func():
            return sum(range(100))

        result = benchmarker.benchmark("test", test_func, num_iterations=50, warmup=5)

        assert result.name == "test"
        assert result.mean_time > 0
        assert result.std_time >= 0
        assert result.num_iterations == 50

    def test_benchmark_with_memory(self):
        """Test: Benchmark with memory tracking."""
        benchmarker = Benchmarker()

        def allocate():
            return [0] * 10000

        result = benchmarker.benchmark("alloc", allocate, num_iterations=10, track_memory=True)

        assert result.memory_used is not None
        assert result.memory_used > 0

    def test_compare_implementations(self):
        """Test: Compare multiple implementations."""
        benchmarker = Benchmarker()

        def impl_loop(n=100):
            total = 0
            for i in range(n):
                total += i
            return total

        def impl_builtin(n=100):
            return sum(range(n))

        def impl_numpy(n=100):
            return np.sum(np.arange(n))

        benchmarks = {
            "loop": impl_loop,
            "builtin": impl_builtin,
            "numpy": impl_numpy
        }

        results = benchmarker.compare(benchmarks, num_iterations=100)

        assert len(results) == 3
        assert all(r.mean_time > 0 for r in results.values())

    def test_detect_regression(self):
        """Test: Detect performance regression."""
        benchmarker = Benchmarker()

        baseline = BenchmarkResult("test", 0.1, 0.01, 0.09, 0.11, 100)
        current_good = BenchmarkResult("test", 0.105, 0.01, 0.095, 0.115, 100)
        current_bad = BenchmarkResult("test", 0.15, 0.01, 0.14, 0.16, 100)

        is_reg_good, change_good = benchmarker.detect_regression(current_good, baseline, threshold=0.1)
        is_reg_bad, change_bad = benchmarker.detect_regression(current_bad, baseline, threshold=0.1)

        assert not is_reg_good  # 5% increase < 10% threshold
        assert is_reg_bad  # 50% increase > 10% threshold
        assert change_bad > change_good


class TestCacheOptimizer:
    """Tests for CacheOptimizer."""

    def test_memoize_decorator(self):
        """Test: Memoization decorator."""
        cache_opt = CacheOptimizer(max_cache_size=10)

        call_count = [0]

        @cache_opt.memoize
        def expensive_func(x):
            call_count[0] += 1
            time.sleep(0.01)
            return x ** 2

        # First call (miss)
        result1 = expensive_func(5)
        assert result1 == 25
        assert call_count[0] == 1

        # Second call (hit)
        result2 = expensive_func(5)
        assert result2 == 25
        assert call_count[0] == 1  # Not called again

        # Different argument (miss)
        result3 = expensive_func(10)
        assert result3 == 100
        assert call_count[0] == 2

    def test_cache_info(self):
        """Test: Cache statistics."""
        cache_opt = CacheOptimizer()

        @cache_opt.memoize
        def test_func(x):
            return x * 2

        test_func(1)
        test_func(1)
        test_func(2)

        stats = test_func.cache_info()
        assert stats['hits'] == 1
        assert stats['misses'] == 2

    def test_cache_eviction(self):
        """Test: Cache LRU eviction."""
        cache_opt = CacheOptimizer(max_cache_size=3)

        @cache_opt.memoize
        def test_func(x):
            return x ** 2

        # Fill cache
        for i in range(5):
            test_func(i)

        stats = test_func.cache_info()
        # All should be misses initially
        assert stats['misses'] == 5


class TestVectorizationOptimizer:
    """Tests for VectorizationOptimizer."""

    def test_vectorize_loop(self):
        """Test: Vectorize function over array."""
        vec_opt = VectorizationOptimizer()

        def square(x):
            return x ** 2

        inputs = np.array([1, 2, 3, 4, 5])
        result = vec_opt.vectorize_loop(square, inputs)

        expected = np.array([1, 4, 9, 16, 25])
        assert np.allclose(result, expected)

    def test_replace_loops_with_operations(self):
        """Test: Replace loops with vectorized operations."""
        vec_opt = VectorizationOptimizer()

        array = np.array([1, 2, 3, 4, 5])

        result_sum = vec_opt.replace_loops_with_operations(array, 'sum')
        assert result_sum == 15

        result_mean = vec_opt.replace_loops_with_operations(array, 'mean')
        assert result_mean == 3.0

        result_max = vec_opt.replace_loops_with_operations(array, 'max')
        assert result_max == 5


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""

    def test_take_snapshot(self):
        """Test: Take memory snapshot."""
        mem_prof = MemoryProfiler()

        mem_prof.take_snapshot("start")
        data = [0] * 100000
        mem_prof.take_snapshot("after_alloc")

        assert len(mem_prof.snapshots) == 2

    def test_compare_snapshots(self):
        """Test: Compare memory snapshots."""
        mem_prof = MemoryProfiler()

        mem_prof.take_snapshot("start")
        data = [0] * 100000
        mem_prof.take_snapshot("after")

        diffs = mem_prof.compare_snapshots(top_n=5)

        # Should show some memory increase
        assert len(diffs) > 0

    def test_detect_leaks_no_leak(self):
        """Test: Detect leaks when there are none."""
        mem_prof = MemoryProfiler()

        mem_prof.take_snapshot("start")
        x = [0] * 1000
        mem_prof.take_snapshot("end")

        leaks = mem_prof.detect_leaks(threshold_mb=100.0)

        # Small allocation, should not be flagged as leak
        assert len(leaks) == 0


class TestPerformanceReporter:
    """Tests for PerformanceReporter."""

    def test_generate_report(self):
        """Test: Generate performance report."""
        reporter = PerformanceReporter()

        prof_results = [
            ProfilingResult("func1", 0.5, 10, 0.05, 10.0, 5.0),
            ProfilingResult("func2", 0.2, 5, 0.04, 2.0, 1.0)
        ]

        bench_results = [
            BenchmarkResult("bench1", 0.01, 0.001, 0.009, 0.011, 100, 5.0)
        ]

        report = reporter.generate_report(prof_results, bench_results)

        assert "PERFORMANCE REPORT" in report
        assert "func1" in report
        assert "bench1" in report

    def test_recommend_optimizations(self):
        """Test: Generate optimization recommendations."""
        reporter = PerformanceReporter()

        # Slow function
        prof_slow = ProfilingResult("slow_func", 10.0, 1, 10.0)
        recommendations = reporter.recommend_optimizations([prof_slow])

        assert len(recommendations) > 0
        assert "slow_func" in recommendations[0]

        # High memory
        prof_mem = ProfilingResult("mem_func", 0.1, 1, 0.1, 200.0)
        recommendations = reporter.recommend_optimizations([prof_mem])

        assert any("memory" in r.lower() for r in recommendations)

        # Many calls (time_per_call must be > 0.001 to trigger recommendation)
        prof_calls = ProfilingResult("many_calls", 20.0, 10000, 0.002)
        recommendations = reporter.recommend_optimizations([prof_calls])

        assert any("calls" in r.lower() or "batching" in r.lower() or "memoization" in r.lower()
                   for r in recommendations)


class TestPerformanceOptimizer:
    """Tests for PerformanceOptimizer."""

    def test_apply_caching(self):
        """Test: Apply caching optimization."""
        optimizer = PerformanceOptimizer()

        def expensive(x):
            time.sleep(0.01)
            return x ** 2

        optimized = optimizer.apply_caching(expensive, max_cache_size=100)

        # First call (slow)
        start = time.perf_counter()
        result1 = optimized(5)
        time1 = time.perf_counter() - start

        # Second call (cached, fast)
        start = time.perf_counter()
        result2 = optimized(5)
        time2 = time.perf_counter() - start

        assert result1 == result2
        assert time2 < time1

    def test_apply_vectorization(self):
        """Test: Apply vectorization."""
        optimizer = PerformanceOptimizer()

        def square(x):
            return x ** 2

        data = np.arange(10)
        result = optimizer.apply_vectorization(square, data)

        expected = data ** 2
        assert np.allclose(result, expected)

    def test_optimization_summary(self):
        """Test: Get optimization summary."""
        optimizer = PerformanceOptimizer()

        summary = optimizer.get_optimization_summary()
        assert "No optimizations" in summary

        # Apply some optimizations
        optimizer.apply_caching(lambda x: x, max_cache_size=10)
        optimizer.apply_vectorization(lambda x: x**2, np.array([1,2,3]))

        summary = optimizer.get_optimization_summary()
        assert "Caching" in summary
        assert "Vectorization" in summary


class TestIntegration:
    """Integration tests."""

    def test_complete_profiling_workflow(self):
        """Test: Complete profiling workflow."""
        print("\nComplete profiling workflow:")

        # 1. Profile function
        profiler = FunctionProfiler()

        def matrix_multiply(n=100):
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            return A @ B

        result, prof = profiler.profile(matrix_multiply, 50, track_memory=True)

        print(f"  1. Profiled {prof.function_name}")
        print(f"     Time: {prof.total_time:.4f}s")
        print(f"     Memory: {prof.memory_delta:.2f} MB")

        # 2. Benchmark
        benchmarker = Benchmarker()
        bench = benchmarker.benchmark("matrix_mult", lambda: matrix_multiply(50), num_iterations=10)

        print(f"  2. Benchmark: {bench.mean_time*1000:.2f} ms ± {bench.std_time*1000:.2f} ms")

        # 3. Generate report
        reporter = PerformanceReporter()
        report = reporter.generate_report([prof], [bench])

        print("  3. Report generated")

        # 4. Recommendations
        recommendations = reporter.recommend_optimizations([prof])
        print(f"  4. Recommendations: {len(recommendations)}")

        print("  ✓ Workflow completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
