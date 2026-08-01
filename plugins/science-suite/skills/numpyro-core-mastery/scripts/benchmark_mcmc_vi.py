#!/usr/bin/env python3
"""Benchmark MCMC vs VI inference."""

import time

from jax import random
from numpyro import optim
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


def benchmark_mcmc_vi(model, x, y, sizes=None):
    """
    Compare MCMC vs VI performance across dataset sizes.

    Args:
        model: NumPyro model function
        x: Input data  (will subsample)
        y: Output data (will subsample)
        sizes: Dataset sizes to test

    Returns:
        dict: Benchmark results
    """
    if sizes is None:
        sizes = [1000, 10000, 100000]
    results = {"mcmc": {}, "vi": {}}

    print("MCMC vs VI Benchmark")
    print("=" * 70)

    for n in sizes:
        if n > len(x):
            print(f"\nSkipping N={n} (exceeds data size)")
            continue

        x_sub = x[:n]
        y_sub = y[:n]

        print(f"\n### Dataset Size: N={n}")

        # MCMC (NUTS)
        try:
            nuts_kernel = NUTS(model)
            mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)

            start = time.time()
            mcmc.run(random.PRNGKey(0), x_sub, y_sub)
            mcmc_time = time.time() - start

            results["mcmc"][n] = mcmc_time
            print(f"MCMC: {mcmc_time:.2f}s")
        except Exception as e:  # noqa: BLE001 - keep benchmarking remaining sizes on any inference failure
            print(f"MCMC failed: {e}")
            results["mcmc"][n] = None

        # VI (SVI)
        try:
            guide = AutoNormal(model)
            optimizer = optim.Adam(0.001)
            svi = SVI(model, guide, optimizer, Trace_ELBO())

            start = time.time()
            _ = svi.run(random.PRNGKey(1), 5000, x_sub, y_sub)
            vi_time = time.time() - start

            results["vi"][n] = vi_time
            print(f"VI:   {vi_time:.2f}s")

            # Speedup
            if results["mcmc"][n]:
                speedup = results["mcmc"][n] / vi_time
                print(f"Speedup: {speedup:.1f}x")
        except Exception as e:  # noqa: BLE001 - keep benchmarking remaining sizes on any inference failure
            print(f"VI failed: {e}")
            results["vi"][n] = None

    print("\n" + "=" * 70)
    print("Recommendation:")
    print("  N < 10K:  MCMC (accurate, fast enough)")
    print("  N > 100K: VI (much faster)")
    print("  10K-100K: Try both, compare accuracy vs speed")
    print("=" * 70)

    return results
