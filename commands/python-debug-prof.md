---
description: Profile Python/JAX code for bottlenecks using jax.profiler and cProfile with optimization suggestions
category: python-performance
argument-hint: "[--jax-profiler] [--cprofile] [--memory] [--suggest-opts]"
allowed-tools: "*"
---

# /python-debug-prof

Profile Python/JAX code for bottlenecks using jax.profiler or cProfile. Suggest optimizations like vectorization or JIT placement.

## Description

Comprehensive profiling toolkit for Python/JAX scientific computing code. Identifies performance bottlenecks, memory issues, and provides actionable optimization suggestions including vectorization, JIT compilation placement, and algorithmic improvements.

## Usage

```
/python-debug-prof [--jax-profiler] [--cprofile] [--memory] [--suggest-opts]
```

## What it does

1. Profile JAX code with jax.profiler for device utilization
2. Use cProfile for CPU-bound Python code analysis
3. Identify memory bottlenecks and allocation patterns
4. Suggest specific optimizations for scientific workflows
5. Provide before/after performance comparisons

## Example output

```python
import jax
import jax.numpy as jnp
import jax.profiler
import cProfile
import pstats
import tracemalloc
import time
from typing import Dict, Any, Callable
import numpy as np

# JAX Profiler Setup and Usage
def setup_jax_profiling(log_dir: str = "./jax_profile") -> None:
    """Set up JAX profiler for detailed performance analysis."""
    # Start JAX profiler server
    jax.profiler.start_server(9999)

    # Configure profiling
    print(f"JAX Profiler running at: http://localhost:9999")
    print(f"Profile data will be saved to: {log_dir}")

def profile_jax_function(
    func: Callable,
    *args,
    profile_name: str = "jax_profile",
    **kwargs
) -> Any:
    """Profile a JAX function with detailed trace analysis."""
    # Start profiling
    jax.profiler.start_trace(profile_name)

    # Run function with warm-up
    print("Warming up...")
    for _ in range(3):
        _ = func(*args, **kwargs)

    print("Profiling...")
    result = func(*args, **kwargs)

    # Stop profiling
    jax.profiler.stop_trace()
    print(f"Profile saved. View at: http://localhost:9999")

    return result

# Example: Profiling matrix operations
@jax.jit
def optimized_matrix_chain(A: jax.Array, B: jax.Array, C: jax.Array) -> jax.Array:
    """Optimized matrix chain multiplication."""
    return jnp.dot(jnp.dot(A, B), C)

def unoptimized_matrix_chain(A: jax.Array, B: jax.Array, C: jax.Array) -> jax.Array:
    """Unoptimized matrix chain (for comparison)."""
    temp1 = []
    for i in range(A.shape[0]):
        temp_row = []
        for j in range(B.shape[1]):
            temp_row.append(jnp.sum(A[i, :] * B[:, j]))
        temp1.append(temp_row)

    temp1 = jnp.array(temp1)

    temp2 = []
    for i in range(temp1.shape[0]):
        temp_row = []
        for j in range(C.shape[1]):
            temp_row.append(jnp.sum(temp1[i, :] * C[:, j]))
        temp2.append(temp_row)

    return jnp.array(temp2)

# Performance benchmarking utilities
def benchmark_function(
    func: Callable,
    *args,
    num_runs: int = 100,
    warmup_runs: int = 10,
    **kwargs
) -> Dict[str, float]:
    """Benchmark function performance with statistics."""
    # Warmup
    for _ in range(warmup_runs):
        _ = func(*args, **kwargs)

    # Timing
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        # Ensure computation completes (important for JAX)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times)
    }

# cProfile integration for Python code
def profile_python_code(func: Callable, *args, **kwargs) -> pstats.Stats:
    """Profile Python code using cProfile."""
    profiler = cProfile.Profile()

    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Create stats object
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')

    print("Top 20 functions by cumulative time:")
    stats.print_stats(20)

    return stats

# Memory profiling utilities
def profile_memory_usage(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile memory usage of a function."""
    tracemalloc.start()

    # Get initial memory
    snapshot_before = tracemalloc.take_snapshot()

    # Run function
    result = func(*args, **kwargs)

    # Get final memory
    snapshot_after = tracemalloc.take_snapshot()

    # Calculate difference
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

    total_size = sum(stat.size for stat in top_stats)

    print(f"Memory usage: {total_size / 1024 / 1024:.2f} MB")
    print("Top 10 memory allocations:")
    for index, stat in enumerate(top_stats[:10], 1):
        print(f"{index}. {stat}")

    tracemalloc.stop()

    return {
        'total_memory_mb': total_size / 1024 / 1024,
        'top_allocations': top_stats[:10],
        'result': result
    }

# JAX-specific optimization detection
def analyze_jax_compilation(func: Callable, *args, **kwargs) -> None:
    """Analyze JAX function compilation and suggest optimizations."""
    print("=== JAX Compilation Analysis ===")

    # Check if function is JIT compiled
    if hasattr(func, '__wrapped__'):
        print("✓ Function is JIT compiled")
    else:
        print("⚠ Function is NOT JIT compiled - Consider adding @jax.jit")

    # Analyze function with different transformations
    print("\nTesting different JAX transformations:")

    # Original function
    original_stats = benchmark_function(func, *args, **kwargs)
    print(f"Original: {original_stats['mean_time']:.6f}s")

    # JIT compiled version (if not already)
    if not hasattr(func, '__wrapped__'):
        jit_func = jax.jit(func)
        jit_stats = benchmark_function(jit_func, *args, **kwargs)
        print(f"With JIT: {jit_stats['mean_time']:.6f}s")
        speedup = original_stats['mean_time'] / jit_stats['mean_time']
        print(f"JIT Speedup: {speedup:.2f}x")

    # Vectorized version (if applicable)
    try:
        vmap_func = jax.vmap(func)
        # Test with batch dimension
        batch_args = [jnp.expand_dims(arg, 0) for arg in args if hasattr(arg, 'shape')]
        if batch_args:
            vmap_stats = benchmark_function(vmap_func, *batch_args, **kwargs)
            print(f"With vmap: {vmap_stats['mean_time']:.6f}s")
    except Exception as e:
        print(f"vmap analysis failed: {e}")

# Scientific computing optimization suggestions
class OptimizationSuggester:
    """Suggest optimizations for scientific computing code."""

    @staticmethod
    def analyze_array_operations(func_source: str) -> list[str]:
        """Analyze source code for optimization opportunities."""
        suggestions = []

        # Check for common anti-patterns
        if 'for' in func_source and 'range' in func_source:
            suggestions.append("⚠ Consider vectorizing loops with JAX operations")

        if '.append(' in func_source:
            suggestions.append("⚠ Avoid list.append() in loops - use jnp.concatenate or pre-allocate arrays")

        if 'numpy' in func_source and 'jax' in func_source:
            suggestions.append("⚠ Mixing NumPy and JAX - consider using only JAX for GPU compatibility")

        if '@jax.jit' not in func_source and 'def ' in func_source:
            suggestions.append("✓ Consider adding @jax.jit decorator for compilation")

        if 'jnp.dot' in func_source:
            suggestions.append("✓ Good use of JAX linear algebra operations")

        return suggestions

    @staticmethod
    def suggest_memory_optimizations(memory_stats: Dict[str, Any]) -> list[str]:
        """Suggest memory optimizations based on profiling."""
        suggestions = []

        if memory_stats['total_memory_mb'] > 1000:  # > 1GB
            suggestions.append("⚠ High memory usage - consider processing in chunks")

        # Analyze allocation patterns
        for stat in memory_stats['top_allocations'][:3]:
            if 'concatenate' in str(stat):
                suggestions.append("⚠ Frequent concatenations - consider pre-allocating arrays")
            if 'array' in str(stat) and stat.size > 100 * 1024 * 1024:  # > 100MB
                suggestions.append("⚠ Large array allocations - consider memory mapping for large datasets")

        return suggestions

# Complete profiling workflow
def comprehensive_profile(
    func: Callable,
    *args,
    profile_name: str = "analysis",
    **kwargs
) -> Dict[str, Any]:
    """Run comprehensive profiling analysis."""
    print(f"=== Comprehensive Profile: {profile_name} ===\n")

    # 1. Performance benchmarking
    print("1. Performance Benchmarking:")
    perf_stats = benchmark_function(func, *args, **kwargs)
    print(f"   Mean execution time: {perf_stats['mean_time']:.6f}s ± {perf_stats['std_time']:.6f}s")

    # 2. Memory profiling
    print("\n2. Memory Profiling:")
    memory_stats = profile_memory_usage(func, *args, **kwargs)

    # 3. JAX-specific analysis
    print("\n3. JAX Compilation Analysis:")
    analyze_jax_compilation(func, *args, **kwargs)

    # 4. Optimization suggestions
    print("\n4. Optimization Suggestions:")
    import inspect
    func_source = inspect.getsource(func)

    optimizer = OptimizationSuggester()
    code_suggestions = optimizer.analyze_array_operations(func_source)
    memory_suggestions = optimizer.suggest_memory_optimizations(memory_stats)

    all_suggestions = code_suggestions + memory_suggestions
    for suggestion in all_suggestions:
        print(f"   {suggestion}")

    if not all_suggestions:
        print("   ✓ No obvious optimization opportunities found!")

    return {
        'performance': perf_stats,
        'memory': memory_stats,
        'suggestions': all_suggestions
    }

# Example usage and demonstrations
def example_profiling_workflow():
    """Demonstrate the complete profiling workflow."""
    # Create test data
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (1000, 1000))
    B = jax.random.normal(key, (1000, 1000))
    C = jax.random.normal(key, (1000, 1000))

    print("Profiling optimized vs unoptimized matrix multiplication:")

    # Profile optimized version
    print("\n=== Optimized Version ===")
    opt_results = comprehensive_profile(
        optimized_matrix_chain, A, B, C,
        profile_name="optimized"
    )

    # Profile unoptimized version
    print("\n=== Unoptimized Version ===")
    unopt_results = comprehensive_profile(
        unoptimized_matrix_chain, A, B, C,
        profile_name="unoptimized"
    )

    # Compare results
    speedup = (unopt_results['performance']['mean_time'] /
               opt_results['performance']['mean_time'])
    print(f"\n=== COMPARISON ===")
    print(f"Speedup: {speedup:.2f}x faster with optimizations")

    return opt_results, unopt_results

# Integration with Jupyter notebooks
def jupyter_profile_cell(cell_code: str) -> None:
    """Profile code in Jupyter cells."""
    print("Add to Jupyter cell:")
    print("""
# Load profiling extensions
%load_ext line_profiler
%load_ext memory_profiler

# Line-by-line profiling
%lprun -f your_function your_function(args)

# Memory profiling
%memit your_function(args)

# JAX profiler in Jupyter
import jax.profiler
jax.profiler.start_trace("/tmp/jax-trace")
result = your_function(args)
jax.profiler.stop_trace()
    """)

# Performance optimization checklist
OPTIMIZATION_CHECKLIST = """
JAX/Python Performance Optimization Checklist:

□ Use @jax.jit for frequently called functions
□ Replace Python loops with vectorized JAX operations
□ Use jax.vmap for batch processing
□ Pre-allocate arrays instead of using append()
□ Use appropriate dtypes (float32 vs float64)
□ Avoid mixing NumPy and JAX arrays
□ Use jax.lax operations for control flow inside JIT
□ Profile memory usage for large datasets
□ Consider jax.pmap for multi-device parallelization
□ Use static_argnums for JIT with varying shapes
□ Minimize host-device transfers
□ Use jnp.where instead of Python conditionals
"""

print(OPTIMIZATION_CHECKLIST)
```

## Related Commands

- `/python-type-hint` - Add type hints for better profiling
- `/jax-jit` - Optimize with JIT compilation
- `/jax-debug` - Debug JAX-specific issues
- `/jax-vmap` - Vectorize for better performance