#!/usr/bin/env python3
"""Complete Performance Optimization Workflow.

This script demonstrates a full performance optimization cycle:
1. Write baseline (slow) code
2. Profile to identify bottlenecks
3. Optimize the bottleneck
4. Verify improvement with benchmarks

Run: python3 profiling-workflow.py
"""

import cProfile
import pstats
import time
from functools import lru_cache
from io import StringIO


# ============================================================================
# STEP 1: Baseline Implementation (Slow)
# ============================================================================

def fibonacci_slow(n: int) -> int:
    """Naive recursive Fibonacci - exponential time complexity O(2^n)."""
    if n <= 1:
        return n
    return fibonacci_slow(n - 1) + fibonacci_slow(n - 2)


def process_data_slow(data: list[int]) -> list[int]:
    """Slow data processing with nested loops and repeated calculations."""
    result = []
    for i in range(len(data)):
        # Inefficient: recalculating sum every iteration
        total = sum(data)
        # Inefficient: quadratic complexity
        for j in range(len(data)):
            if data[i] > data[j]:
                result.append(data[i] + data[j])
    return result


# ============================================================================
# STEP 2: Optimized Implementation
# ============================================================================

@lru_cache(maxsize=None)
def fibonacci_fast(n: int) -> int:
    """Memoized Fibonacci - linear time complexity O(n)."""
    if n <= 1:
        return n
    return fibonacci_fast(n - 1) + fibonacci_fast(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Iterative Fibonacci - linear time, constant space O(n) time, O(1) space."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def process_data_fast(data: list[int]) -> list[int]:
    """Optimized data processing with pre-computed values."""
    result = []
    total = sum(data)  # Calculate once
    sorted_data = sorted(data)  # Sort once for efficient comparisons

    for i, val in enumerate(data):
        # Use binary search approach or sorted data
        for other_val in sorted_data:
            if val > other_val:
                result.append(val + other_val)
            else:
                break  # No need to continue
    return result


# ============================================================================
# STEP 3: Profiling Functions
# ============================================================================

def profile_function(func, *args, **kwargs):
    """Profile a function and return statistics."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Get statistics
    string_io = StringIO()
    stats = pstats.Stats(profiler, stream=string_io)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

    return result, string_io.getvalue()


def benchmark_function(func, *args, iterations=100, **kwargs):
    """Benchmark a function over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        'average': avg_time,
        'min': min_time,
        'max': max_time,
        'total': sum(times)
    }


# ============================================================================
# STEP 4: Demonstration and Analysis
# ============================================================================

def demonstrate_fibonacci():
    """Demonstrate Fibonacci optimization."""
    print("=" * 70)
    print("FIBONACCI PERFORMANCE COMPARISON")
    print("=" * 70)

    n = 30

    print(f"\nCalculating fibonacci({n})...")
    print("\n1. Naive Recursive Implementation:")
    result, profile_output = profile_function(fibonacci_slow, n)
    print(f"   Result: {result}")
    print("   Profile (top functions):")
    for line in profile_output.split('\n')[5:10]:  # Show relevant lines
        if line.strip():
            print(f"   {line}")

    print("\n2. Memoized Implementation:")
    fibonacci_fast.cache_clear()  # Clear cache for fair comparison
    result, profile_output = profile_function(fibonacci_fast, n)
    print(f"   Result: {result}")
    print("   Profile (top functions):")
    for line in profile_output.split('\n')[5:10]:
        if line.strip():
            print(f"   {line}")

    print("\n3. Iterative Implementation:")
    result, profile_output = profile_function(fibonacci_iterative, n)
    print(f"   Result: {result}")
    print("   Profile (top functions):")
    for line in profile_output.split('\n')[5:10]:
        if line.strip():
            print(f"   {line}")

    # Benchmark comparison
    print("\n" + "-" * 70)
    print("BENCHMARK COMPARISON (10 iterations each)")
    print("-" * 70)

    slow_bench = benchmark_function(fibonacci_slow, n, iterations=10)
    print(f"\nNaive Recursive:  {slow_bench['average']*1000:.2f}ms (avg)")

    fibonacci_fast.cache_clear()
    fast_bench = benchmark_function(fibonacci_fast, n, iterations=10)
    print(f"Memoized:         {fast_bench['average']*1000:.2f}ms (avg)")

    iter_bench = benchmark_function(fibonacci_iterative, n, iterations=10)
    print(f"Iterative:        {iter_bench['average']*1000:.2f}ms (avg)")

    speedup_memo = slow_bench['average'] / fast_bench['average']
    speedup_iter = slow_bench['average'] / iter_bench['average']
    print(f"\nSpeedup (memoized): {speedup_memo:.1f}x faster")
    print(f"Speedup (iterative): {speedup_iter:.1f}x faster")


def demonstrate_data_processing():
    """Demonstrate data processing optimization."""
    print("\n\n" + "=" * 70)
    print("DATA PROCESSING PERFORMANCE COMPARISON")
    print("=" * 70)

    # Small dataset for demonstration
    data = list(range(100, 0, -1))  # 100 descending integers

    print(f"\nProcessing {len(data)} integers...")

    print("\n1. Slow Implementation (nested loops):")
    result, profile_output = profile_function(process_data_slow, data)
    print(f"   Result length: {len(result)}")
    print("   Profile (top functions):")
    for line in profile_output.split('\n')[5:10]:
        if line.strip():
            print(f"   {line}")

    print("\n2. Optimized Implementation (pre-computed values):")
    result, profile_output = profile_function(process_data_fast, data)
    print(f"   Result length: {len(result)}")
    print("   Profile (top functions):")
    for line in profile_output.split('\n')[5:10]:
        if line.strip():
            print(f"   {line}")

    # Benchmark comparison
    print("\n" + "-" * 70)
    print("BENCHMARK COMPARISON (10 iterations each)")
    print("-" * 70)

    slow_bench = benchmark_function(process_data_slow, data, iterations=10)
    print(f"\nSlow:       {slow_bench['average']*1000:.2f}ms (avg)")

    fast_bench = benchmark_function(process_data_fast, data, iterations=10)
    print(f"Optimized:  {fast_bench['average']*1000:.2f}ms (avg)")

    speedup = slow_bench['average'] / fast_bench['average']
    print(f"\nSpeedup: {speedup:.1f}x faster")


def demonstrate_memory_usage():
    """Demonstrate memory optimization techniques."""
    print("\n\n" + "=" * 70)
    print("MEMORY OPTIMIZATION TECHNIQUES")
    print("=" * 70)

    print("\n1. List vs Generator (memory efficient):")
    print("   List:      [x**2 for x in range(1000000)]  # Stores all in memory")
    print("   Generator: (x**2 for x in range(1000000))  # Generates on demand")

    print("\n2. Slots for classes (reduce memory per instance):")
    print("""
    class RegularClass:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class SlottedClass:
        __slots__ = ['x', 'y']
        def __init__(self, x, y):
            self.x = x
            self.y = y
    """)

    print("\n3. Use appropriate data structures:")
    print("   - list: Dynamic array for ordered data")
    print("   - set: O(1) lookup for membership testing")
    print("   - dict: O(1) key-value lookups")
    print("   - deque: O(1) append/pop from both ends")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Run complete performance optimization workflow."""
    print("\n" + "=" * 70)
    print("PYTHON PERFORMANCE OPTIMIZATION WORKFLOW")
    print("=" * 70)
    print("\nThis demonstration shows:")
    print("  1. Profiling slow code to identify bottlenecks")
    print("  2. Applying optimizations (memoization, better algorithms)")
    print("  3. Benchmarking to verify improvements")
    print("=" * 70)

    # Run demonstrations
    demonstrate_fibonacci()
    demonstrate_data_processing()
    demonstrate_memory_usage()

    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. PROFILE FIRST: Always measure before optimizing
   - Use cProfile for CPU profiling
   - Use memory_profiler for memory usage
   - Focus on hot spots (most time-consuming functions)

2. COMMON OPTIMIZATIONS:
   - Memoization: Cache expensive function results
   - Better algorithms: O(2^n) → O(n) → O(log n)
   - Vectorization: Use NumPy for numerical operations
   - Pre-compute: Calculate once, use many times

3. DATA STRUCTURES MATTER:
   - list: Good for sequential access
   - set/dict: Excellent for lookups (O(1))
   - deque: Better for queue operations
   - array: Memory-efficient for numeric data

4. AVOID PREMATURE OPTIMIZATION:
   - Write clear code first
   - Profile to find real bottlenecks
   - Optimize hot paths only
   - Maintain code readability

5. TOOLS:
   - cProfile: CPU profiling
   - line_profiler: Line-by-line profiling
   - memory_profiler: Memory usage
   - py-spy: Sampling profiler (production-safe)
   - timeit: Micro-benchmarking
    """)

    print("=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
