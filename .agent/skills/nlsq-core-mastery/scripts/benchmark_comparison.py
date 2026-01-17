#!/usr/bin/env python3
"""
Benchmark NLSQ vs SciPy curve_fit across different dataset sizes.

Usage:
    python benchmark_comparison.py

Outputs performance metrics showing speedup of NLSQ over SciPy.
"""

import time
import numpy as np
from scipy.optimize import curve_fit
import jax.numpy as jnp
from nlsq import CurveFit


def exponential_decay_numpy(t, A, lambda_, c):
    """Exponential decay model (NumPy)."""
    return A * np.exp(-lambda_ * t) + c


def exponential_decay_jax(t, params):
    """Exponential decay model (JAX)."""
    A, lambda_, c = params
    return A * jnp.exp(-lambda_ * t) + c


def benchmark_comparison(sizes=[1000, 10_000, 100_000, 1_000_000]):
    """Compare SciPy vs NLSQ across dataset sizes."""

    # True parameters
    true_params = [5.0, 0.5, 1.0]

    # Generate full dataset
    t_max = np.linspace(0, 10, max(sizes))
    y_max = exponential_decay_numpy(t_max, *true_params)
    y_max += np.random.normal(0, 0.1, len(t_max))

    print("=" * 70)
    print("NLSQ vs SciPy Benchmark")
    print("=" * 70)
    print(f"{'Size':>10s} | {'SciPy (s)':>10s} | {'NLSQ (s)':>10s} | {'Speedup':>8s}")
    print("-" * 70)

    results = []

    for n in sizes:
        # Subset data
        t = t_max[:n]
        y = y_max[:n]

        # Initial guess
        p0 = [4.0, 0.4, 0.8]

        # Benchmark SciPy
        start = time.time()
        popt_scipy, _ = curve_fit(exponential_decay_numpy, t, y, p0=p0)
        scipy_time = time.time() - start

        # Benchmark NLSQ (include JIT time)
        t_jax = jnp.array(t)
        y_jax = jnp.array(y)
        p0_jax = jnp.array(p0)

        start = time.time()
        optimizer = CurveFit(exponential_decay_jax, t_jax, y_jax, p0_jax)
        result = optimizer.fit()
        nlsq_time = time.time() - start

        # Calculate speedup
        speedup = scipy_time / nlsq_time

        # Store results
        results.append({
            'size': n,
            'scipy_time': scipy_time,
            'nlsq_time': nlsq_time,
            'speedup': speedup
        })

        # Print results
        print(f"{n:10,d} | {scipy_time:10.3f} | {nlsq_time:10.3f} | {speedup:8.1f}x")

        # Verify parameters match
        param_diff = np.linalg.norm(popt_scipy - np.array(result.x))
        if param_diff > 0.01:
            print(f"  ⚠️  WARNING: Parameter mismatch (diff={param_diff:.3f})")

    print("=" * 70)

    # Summary
    print("\nSummary:")
    avg_speedup = np.mean([r['speedup'] for r in results if r['size'] >= 10_000])
    print(f"  Average speedup (N≥10K): {avg_speedup:.1f}x")

    max_speedup = max(results, key=lambda r: r['speedup'])
    print(f"  Maximum speedup: {max_speedup['speedup']:.1f}x at N={max_speedup['size']:,}")

    print("\nRecommendation:")
    print("  N < 10K:      SciPy or NLSQ (similar performance)")
    print("  N ≥ 10K:      NLSQ (significant speedup)")
    print("  N ≥ 1M:       NLSQ (orders of magnitude faster)")
    print("  N > GPU mem:  StreamingOptimizer required")

    return results


if __name__ == "__main__":
    results = benchmark_comparison()
