#!/usr/bin/env python3
"""Model comparison using WAIC and LOO."""

import jax.numpy as jnp
from numpyro.diagnostics import waic, loo


def compare_models(models_dict, data_x, data_y):
    """
    Compare multiple models using WAIC and LOO.

    Args:
        models_dict: {name: (model_fn, posterior_samples)}
        data_x: Input data
        data_y: Output data

    Returns:
        dict: Comparison results
    """
    results = {}

    print("Model Comparison")
    print("="*60)

    for name, (model, samples) in models_dict.items():
        # WAIC
        waic_result = waic(model, samples, data_x, data_y)
        # LOO
        loo_result = loo(model, samples, data_x, data_y)

        results[name] = {
            'waic': waic_result.waic,
            'waic_se': waic_result.waic_se,
            'loo': loo_result.loo,
            'loo_se': loo_result.loo_se
        }

        print(f"\n{name}:")
        print(f"  WAIC: {waic_result.waic:.2f} ± {waic_result.waic_se:.2f}")
        print(f"  LOO:  {loo_result.loo:.2f} ± {loo_result.loo_se:.2f}")

    # Best model (lowest LOO)
    best_model = min(results.items(), key=lambda x: x[1]['loo'])
    print(f"\n{'='*60}")
    print(f"Best model: {best_model[0]} (LOO={best_model[1]['loo']:.2f})")
    print("="*60)

    return results
