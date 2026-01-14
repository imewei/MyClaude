#!/usr/bin/env python3
"""
JAX Profiling Utilities

Tools for profiling JAX code performance, identifying bottlenecks,
and optimizing computational efficiency.
"""

import jax
import jax.numpy as jnp
import time
from typing import Callable, Any, Dict
from functools import wraps


def profile_function(fn: Callable, *args, n_warmup: int = 3, n_runs: int = 10, **kwargs) -> Dict[str, float]:
    """
    Profile a JAX function with proper warmup and timing.

    Args:
        fn: Function to profile
        *args: Positional arguments for fn
        n_warmup: Number of warmup runs (for JIT compilation)
        n_runs: Number of timed runs
        **kwargs: Keyword arguments for fn

    Returns:
        Dictionary with timing statistics (mean, std, min, max in ms)
    """
    print(f"\nProfiling {fn.__name__}...")

    # Warmup runs (compilation)
    print(f"  Warmup ({n_warmup} runs)...")
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)

    # Timed runs
    print(f"  Timing ({n_runs} runs)...")
    times = []
    for i in range(n_runs):
        start = time.time()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # Convert to ms

    times_array = jnp.array(times)
    stats = {
        'mean': float(jnp.mean(times_array)),
        'std': float(jnp.std(times_array)),
        'min': float(jnp.min(times_array)),
        'max': float(jnp.max(times_array)),
    }

    print(f"  Results: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
    return stats


def compare_implementations(*fns, args=None, kwargs=None, n_runs=10):
    """
    Compare performance of multiple implementations.

    Args:
        *fns: Functions to compare
        args: Tuple of arguments for all functions
        kwargs: Dict of keyword arguments for all functions
        n_runs: Number of runs for timing

    Returns:
        Dictionary mapping function names to timing stats
    """
    args = args or ()
    kwargs = kwargs or {}

    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    results = {}
    for fn in fns:
        stats = profile_function(fn, *args, n_runs=n_runs, **kwargs)
        results[fn.__name__] = stats

    # Print comparison
    print("\nComparison Summary:")
    print("-" * 60)
    baseline = results[fns[0].__name__]['mean']

    for name, stats in results.items():
        speedup = baseline / stats['mean']
        print(f"{name:30s}: {stats['mean']:8.2f}ms (speedup: {speedup:5.2f}x)")

    return results


def memory_profile(fn: Callable, *args, **kwargs):
    """
    Profile memory usage of a JAX function.

    Args:
        fn: Function to profile
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    print(f"\nMemory profiling {fn.__name__}...")

    # Get memory before
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                           '--format=csv,nounits,noheader'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        mem_before = int(result.stdout.strip().split('\n')[0])
        print(f"  Memory before: {mem_before} MB")
    else:
        print("  nvidia-smi not available (CPU mode)")
        mem_before = None

    # Run function
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)

    # Get memory after
    if mem_before is not None:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                               '--format=csv,nounits,noheader'],
                              capture_output=True, text=True)
        mem_after = int(result.stdout.strip().split('\n')[0])
        mem_used = mem_after - mem_before
        print(f"  Memory after: {mem_after} MB")
        print(f"  Memory used: {mem_used} MB")

    return result


def check_recompilation(fn: Callable, test_inputs: list):
    """
    Check if function recompiles for different inputs.

    Args:
        fn: JIT-compiled function
        test_inputs: List of input tuples to test
    """
    print(f"\nChecking recompilation for {fn.__name__}...")
    print(f"Testing {len(test_inputs)} different inputs")

    compilation_count = 0

    for i, inputs in enumerate(test_inputs):
        start = time.time()
        result = fn(*inputs)
        jax.block_until_ready(result)
        elapsed = time.time() - start

        # First call or slow call indicates compilation
        if i == 0 or elapsed > 0.1:
            compilation_count += 1
            print(f"  Input {i}: {elapsed*1000:.2f}ms (COMPILED)")
        else:
            print(f"  Input {i}: {elapsed*1000:.2f}ms (cached)")

    print(f"\nTotal compilations: {compilation_count}/{len(test_inputs)}")
    if compilation_count > 1:
        print("⚠ Warning: Function recompiles for different inputs!")
        print("  Consider using static_argnums or padding to fixed shapes.")


def profile_gradient(loss_fn: Callable, params, *args, **kwargs):
    """
    Profile gradient computation.

    Args:
        loss_fn: Loss function to differentiate
        params: Parameters for loss function
        *args: Additional arguments for loss_fn
        **kwargs: Additional keyword arguments
    """
    print("\nProfiling gradient computation...")

    # Forward pass
    print("  Forward pass...")
    forward_stats = profile_function(loss_fn, params, *args, **kwargs)

    # Gradient computation
    print("  Gradient computation...")
    grad_fn = jax.grad(loss_fn)
    grad_stats = profile_function(grad_fn, params, *args, **kwargs)

    # Value and gradient together
    print("  Value + gradient...")
    value_grad_fn = jax.value_and_grad(loss_fn)
    combined_stats = profile_function(value_grad_fn, params, *args, **kwargs)

    print("\nGradient Profiling Summary:")
    print(f"  Forward only:     {forward_stats['mean']:.2f}ms")
    print(f"  Gradient only:    {grad_stats['mean']:.2f}ms")
    print(f"  Value + gradient: {combined_stats['mean']:.2f}ms")
    print(f"  Overhead:         {(combined_stats['mean'] - forward_stats['mean']):.2f}ms")


def profile_decorator(n_warmup=3, n_runs=10):
    """
    Decorator for automatic profiling of functions.

    Usage:
        @profile_decorator(n_runs=20)
        def my_function(x):
            return x ** 2
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            stats = profile_function(fn, *args, n_warmup=n_warmup, n_runs=n_runs, **kwargs)
            result = fn(*args, **kwargs)
            return result, stats
        return wrapper
    return decorator


# Example usage
if __name__ == '__main__':
    print("JAX Profiling Utilities Demo")
    print("=" * 60)

    # Example 1: Basic profiling
    print("\n1. Basic Function Profiling")

    def naive_matmul(A, B):
        return jnp.dot(A, B)

    @jax.jit
    def jit_matmul(A, B):
        return jnp.dot(A, B)

    A = jnp.ones((1000, 1000))
    B = jnp.ones((1000, 1000))

    compare_implementations(
        naive_matmul,
        jit_matmul,
        args=(A, B),
        n_runs=20
    )

    # Example 2: Recompilation check
    print("\n2. Recompilation Check")

    @jax.jit
    def dynamic_fn(x, n):
        return jnp.sum(x ** n)

    test_inputs = [
        (jnp.ones(100), 2),
        (jnp.ones(100), 3),  # Different n -> recompilation
        (jnp.ones(200), 2),  # Different shape -> recompilation
    ]

    check_recompilation(dynamic_fn, test_inputs)

    # Example 3: Gradient profiling
    print("\n3. Gradient Profiling")

    def loss_fn(params, x):
        return jnp.sum((params @ x) ** 2)

    params = jnp.ones((100, 100))
    x = jnp.ones(100)

    profile_gradient(loss_fn, params, x)

    print("\n" + "=" * 60)
    print("Profiling complete!")
