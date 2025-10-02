"""Demonstrations of Performance Profiling and Optimization.

Shows practical applications of profiling tools, benchmarking, caching,
vectorization, memory profiling, and automated optimization strategies.

Author: Nonequilibrium Physics Agents
Week: 27-28 of Phase 4
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
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


def demo_1_timer_profiling():
    """Demo 1: Basic timing with Timer context manager."""
    print("\n" + "="*70)
    print("DEMO 1: Timer and Basic Profiling")
    print("="*70)

    print("\nScenario: Time different matrix operations")

    # Small matrix multiplication
    print("\n1. Small matrix (100x100):")
    with Timer("Small matrix multiply") as t:
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        C = A @ B
    print(f"   {t}")

    # Large matrix multiplication
    print("\n2. Large matrix (500x500):")
    with Timer("Large matrix multiply") as t:
        A = np.random.rand(500, 500)
        B = np.random.rand(500, 500)
        C = A @ B
    print(f"   {t}")

    # FFT operation
    print("\n3. FFT (N=100000):")
    with Timer("FFT computation") as t:
        x = np.random.rand(100000)
        y = np.fft.fft(x)
    print(f"   {t}")

    print("\n→ Timer provides high-precision timing via time.perf_counter()")
    print("→ Context manager ensures proper start/stop")


def demo_2_function_profiling():
    """Demo 2: Detailed function profiling with memory tracking."""
    print("\n" + "="*70)
    print("DEMO 2: Function Profiling with Memory Tracking")
    print("="*70)

    print("\nScenario: Profile control computation with memory tracking")

    profiler = FunctionProfiler()

    # Define expensive control function
    def compute_optimal_control(n_states, n_steps, n_iterations):
        """Simulate iterative optimal control computation."""
        # State trajectory storage
        states = np.zeros((n_steps, n_states))
        controls = np.zeros((n_steps, n_states))

        # Cost-to-go
        V = np.random.rand(n_states, n_states)

        for _ in range(n_iterations):
            # Backward pass
            for t in range(n_steps-1, -1, -1):
                # Compute optimal control
                K = -np.linalg.solve(
                    np.eye(n_states) + V,
                    V
                )
                controls[t] = K @ states[t]

            # Forward pass
            for t in range(n_steps-1):
                states[t+1] = states[t] + 0.1 * controls[t]

        return states, controls

    # Profile with memory tracking
    print("\nProfiling optimal control computation:")
    print("  States: 10, Steps: 50, Iterations: 20")

    result, prof = profiler.profile(
        compute_optimal_control,
        10, 50, 20,
        track_memory=True
    )

    print(f"\nProfiling Results:")
    print(f"  Function: {prof.function_name}")
    print(f"  Total time: {prof.total_time:.4f} seconds")
    print(f"  Time per call: {prof.time_per_call:.4f} seconds")
    print(f"  Peak memory: {prof.memory_peak:.2f} MB")
    print(f"  Memory delta: {prof.memory_delta:.2f} MB")

    print("\n→ FunctionProfiler tracks both time and memory")
    print("→ Uses tracemalloc for memory profiling")
    print("→ Provides detailed per-call statistics")


def demo_3_benchmarking():
    """Demo 3: Benchmark comparison and regression detection."""
    print("\n" + "="*70)
    print("DEMO 3: Benchmarking and Regression Detection")
    print("="*70)

    print("\nScenario: Compare different matrix multiplication approaches")

    benchmarker = Benchmarker()

    n = 200

    # Method 1: Loop-based (slow)
    def matmul_loops():
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i,j] += A[i,k] * B[k,j]
        return C

    # Method 2: NumPy dot (fast)
    def matmul_numpy():
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        return A @ B

    # Method 3: Einsum (medium)
    def matmul_einsum():
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        return np.einsum('ij,jk->ik', A, B)

    print("\nBenchmarking three implementations:")
    print(f"  Matrix size: {n}x{n}")
    print(f"  Iterations: 10 (loops), 100 (numpy/einsum)")

    # Benchmark (loops - fewer iterations because slow)
    print("\n  1. Triple loops (reference)...")
    bench_loops = benchmarker.benchmark(
        "loops",
        matmul_loops,
        num_iterations=10,
        warmup=2
    )

    # Benchmark numpy
    print("  2. NumPy @ operator...")
    bench_numpy = benchmarker.benchmark(
        "numpy",
        matmul_numpy,
        num_iterations=100,
        warmup=10
    )

    # Benchmark einsum
    print("  3. NumPy einsum...")
    bench_einsum = benchmarker.benchmark(
        "einsum",
        matmul_einsum,
        num_iterations=100,
        warmup=10
    )

    print(f"\nResults:")
    print(f"  {'Method':<15} {'Mean Time':<15} {'Speedup':<10}")
    print("-" * 45)

    baseline = bench_loops.mean_time
    for name, bench in [("Loops", bench_loops), ("NumPy", bench_numpy), ("Einsum", bench_einsum)]:
        speedup = baseline / bench.mean_time
        print(f"  {name:<15} {bench.mean_time*1000:>8.2f} ms     {speedup:>6.1f}x")

    # Regression detection
    print("\nRegression Detection:")
    is_reg, change = benchmarker.detect_regression(
        bench_einsum,
        bench_numpy,
        threshold=0.2
    )
    print(f"  Einsum vs NumPy: {change*100:+.1f}% change")
    print(f"  Regression detected: {is_reg} (threshold: 20%)")

    print("\n→ Benchmarker provides statistical analysis (mean, std, min, max)")
    print("→ Regression detection helps catch performance degradation")
    print("→ NumPy operations ~100-1000x faster than pure Python loops")


def demo_4_cache_optimization():
    """Demo 4: Caching optimization with memoization."""
    print("\n" + "="*70)
    print("DEMO 4: Cache Optimization")
    print("="*70)

    print("\nScenario: Cache expensive Fibonacci computation")

    cache_opt = CacheOptimizer(max_cache_size=100)

    # Expensive recursive function (intentionally inefficient)
    @cache_opt.memoize
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    # First computation (cache misses)
    print("\nFirst computation (populating cache):")
    with Timer("First fib(30)") as t1:
        result1 = fibonacci(30)
    print(f"  Result: {result1}")
    print(f"  {t1}")
    stats1 = fibonacci.cache_info()
    print(f"  Cache hits: {stats1['hits']}, misses: {stats1['misses']}")

    # Second computation (cache hits)
    print("\nSecond computation (using cache):")
    with Timer("Second fib(30)") as t2:
        result2 = fibonacci(30)
    print(f"  Result: {result2}")
    print(f"  {t2}")
    stats2 = fibonacci.cache_info()
    print(f"  Cache hits: {stats2['hits']}, misses: {stats2['misses']}")

    speedup = t1.elapsed / t2.elapsed
    print(f"\nSpeedup: {speedup:.0f}x faster with caching")
    print(f"Hit rate: {stats2['hits'] / (stats2['hits'] + stats2['misses']) * 100:.1f}%")

    print("\n→ Memoization caches function results by input")
    print("→ LRU eviction prevents unbounded memory growth")
    print("→ Massive speedup for repeated computations")


def demo_5_vectorization():
    """Demo 5: Vectorization optimization."""
    print("\n" + "="*70)
    print("DEMO 5: Vectorization Optimization")
    print("="*70)

    print("\nScenario: Replace loops with vectorized NumPy operations")

    vec_opt = VectorizationOptimizer()

    # Scalar function
    def nonlinear_control(x):
        """Nonlinear control law."""
        return -np.tanh(x) - 0.1 * x**3

    # Generate state trajectory
    states = np.linspace(-5, 5, 10000)

    # Method 1: Python loop (slow)
    print("\n1. Python loop:")
    controls_loop = np.zeros_like(states)
    with Timer("Loop-based") as t_loop:
        for i, x in enumerate(states):
            controls_loop[i] = nonlinear_control(x)
    print(f"   {t_loop}")

    # Method 2: Vectorized (fast)
    print("\n2. Vectorized NumPy:")
    with Timer("Vectorized") as t_vec:
        controls_vec = vec_opt.vectorize_loop(nonlinear_control, states)
    print(f"   {t_vec}")

    # Verify results match
    error = np.linalg.norm(controls_loop - controls_vec)
    print(f"\nVerification:")
    print(f"  Max difference: {np.max(np.abs(controls_loop - controls_vec)):.2e}")
    print(f"  Speedup: {t_loop.elapsed / t_vec.elapsed:.1f}x")

    # Array operations
    print("\n3. Vectorized array operations:")
    data = np.random.rand(1000000)

    ops = ['sum', 'mean', 'max', 'min', 'std']
    print(f"  Array size: {len(data)}")
    print(f"\n  {'Operation':<10} {'Result':<15} {'Time':<10}")
    print("  " + "-" * 35)

    for op in ops:
        with Timer() as t:
            result = vec_opt.replace_loops_with_operations(data, op)
        print(f"  {op:<10} {result:<15.6f} {t.elapsed*1000:<10.2f} ms")

    print("\n→ Vectorization leverages SIMD and optimized C code")
    print("→ 10-100x speedup over Python loops")
    print("→ NumPy operations work on entire arrays")


def demo_6_memory_profiling():
    """Demo 6: Memory profiling and leak detection."""
    print("\n" + "="*70)
    print("DEMO 6: Memory Profiling and Leak Detection")
    print("="*70)

    print("\nScenario: Detect memory leaks in iterative algorithm")

    mem_prof = MemoryProfiler()

    # Simulate algorithm with potential leak
    def iterative_algorithm(n_iterations, leak=False):
        """Iterative computation with optional memory leak."""
        data = []

        mem_prof.take_snapshot("start")

        for i in range(n_iterations):
            # Allocate memory
            temp = np.random.rand(10000)
            result = np.sum(temp**2)

            # Leak: keep reference (bad!)
            if leak:
                data.append(temp)

            if i == n_iterations // 2:
                mem_prof.take_snapshot("midpoint")

        mem_prof.take_snapshot("end")

        return result

    # Test 1: No leak (good)
    print("\n1. Algorithm WITHOUT leak:")
    result1 = iterative_algorithm(100, leak=False)

    diffs1 = mem_prof.compare_snapshots(0, 2, top_n=3)
    print(f"  Memory change (start → end):")
    for stat_str, size_diff in diffs1[:1]:
        size_mb = size_diff / 1024 / 1024
        print(f"    {size_mb:+.2f} MB")

    leaks1 = mem_prof.detect_leaks(threshold_mb=1.0)
    print(f"  Leaks detected: {len(leaks1)}")

    # Clear snapshots
    mem_prof.snapshots.clear()

    # Test 2: With leak (bad)
    print("\n2. Algorithm WITH leak:")
    result2 = iterative_algorithm(100, leak=True)

    diffs2 = mem_prof.compare_snapshots(0, 2, top_n=3)
    print(f"  Memory change (start → end):")
    for stat_str, size_diff in diffs2[:1]:
        size_mb = size_diff / 1024 / 1024
        print(f"    {size_mb:+.2f} MB")

    leaks2 = mem_prof.detect_leaks(threshold_mb=1.0)
    print(f"  Leaks detected: {len(leaks2)}")
    if leaks2:
        print(f"  Warning: {leaks2[0][:60]}...")

    print("\n→ Memory profiling via tracemalloc snapshots")
    print("→ Leak detection compares snapshots over time")
    print("→ Identifies growing allocations that may indicate leaks")


def demo_7_complete_workflow():
    """Demo 7: Complete profiling and optimization workflow."""
    print("\n" + "="*70)
    print("DEMO 7: Complete Performance Optimization Workflow")
    print("="*70)

    print("\nScenario: End-to-end optimization of optimal control solver")

    # Original (unoptimized) solver
    def lqr_solver_original(A, B, Q, R, N):
        """Discrete-time LQR via dynamic programming (unoptimized)."""
        n = A.shape[0]
        m = B.shape[1]

        # Storage
        P = [None] * (N + 1)
        K = [None] * N

        # Terminal cost
        P[N] = Q.copy()

        # Backward pass
        for t in range(N-1, -1, -1):
            # Riccati recursion
            temp = np.linalg.solve(R + B.T @ P[t+1] @ B, B.T @ P[t+1] @ A)
            K[t] = temp
            P[t] = Q + A.T @ P[t+1] @ A - A.T @ P[t+1] @ B @ temp

        return K, P

    # System matrices
    n_states = 8
    n_controls = 4
    N = 50

    A = np.eye(n_states) + 0.1 * np.random.randn(n_states, n_states)
    B = 0.1 * np.random.randn(n_states, n_controls)
    Q = np.eye(n_states)
    R = np.eye(n_controls)

    print("\nStep 1: PROFILE original implementation")
    print("-" * 70)

    profiler = FunctionProfiler()
    _, prof_orig = profiler.profile(
        lqr_solver_original,
        A, B, Q, R, N,
        track_memory=True
    )

    print(f"  Time: {prof_orig.total_time:.6f} seconds")
    print(f"  Memory: {prof_orig.memory_delta:.2f} MB")

    print("\nStep 2: BENCHMARK for statistical analysis")
    print("-" * 70)

    benchmarker = Benchmarker()
    bench_orig = benchmarker.benchmark(
        "original",
        lambda: lqr_solver_original(A, B, Q, R, N),
        num_iterations=50,
        warmup=5
    )

    print(f"  Mean: {bench_orig.mean_time*1000:.3f} ms")
    print(f"  Std:  {bench_orig.std_time*1000:.3f} ms")

    print("\nStep 3: OPTIMIZE with caching")
    print("-" * 70)

    optimizer = PerformanceOptimizer()

    # Apply caching (for repeated calls with same inputs)
    lqr_solver_cached = optimizer.apply_caching(
        lqr_solver_original,
        max_cache_size=100
    )

    # Benchmark cached version (repeated call)
    lqr_solver_cached(A, B, Q, R, N)  # First call (miss)
    bench_cached = benchmarker.benchmark(
        "cached",
        lambda: lqr_solver_cached(A, B, Q, R, N),
        num_iterations=50,
        warmup=0  # No warmup, want cache hits
    )

    print(f"  Mean: {bench_cached.mean_time*1000:.3f} ms")
    print(f"  Speedup: {bench_orig.mean_time / bench_cached.mean_time:.1f}x")

    print("\nStep 4: GENERATE performance report")
    print("-" * 70)

    reporter = PerformanceReporter()
    report = reporter.generate_report(
        [prof_orig],
        [bench_orig, bench_cached]
    )

    # Print subset of report
    lines = report.split('\n')
    for line in lines[:15]:
        print(f"  {line}")
    print("  ...")

    print("\nStep 5: RECOMMENDATIONS")
    print("-" * 70)

    recommendations = reporter.recommend_optimizations([prof_orig])
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("\nStep 6: OPTIMIZATION summary")
    print("-" * 70)

    summary = optimizer.get_optimization_summary()
    for line in summary.split('\n'):
        print(f"  {line}")

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print("\n→ Profile → Benchmark → Optimize → Report → Iterate")
    print("→ Automated recommendations guide optimization")
    print("→ Track improvements with regression detection")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*15 + "PERFORMANCE PROFILING & OPTIMIZATION DEMOS")
    print("="*70)

    # Run demos
    demo_1_timer_profiling()
    demo_2_function_profiling()
    demo_3_benchmarking()
    demo_4_cache_optimization()
    demo_5_vectorization()
    demo_6_memory_profiling()
    demo_7_complete_workflow()

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
