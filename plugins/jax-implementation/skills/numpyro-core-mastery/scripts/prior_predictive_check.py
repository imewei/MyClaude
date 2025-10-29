#!/usr/bin/env python3
"""Prior predictive checks for model validation."""

import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive


def prior_predictive_check(model, x_data, y_data=None, num_samples=1000, rng_seed=0):
    """
    Generate and visualize prior predictive samples.

    Args:
        model: NumPyro model function
        x_data: Input data
        y_data: Observed data (for comparison)
        num_samples: Number of prior samples
        rng_seed: Random seed

    Returns:
        dict: Prior samples
    """
    rng_key = random.PRNGKey(rng_seed)

    # Generate prior predictive samples
    prior_predictive = Predictive(model, num_samples=num_samples)
    prior_samples = prior_predictive(rng_key, x_data, y=None)

    # Extract predictions
    y_prior = prior_samples['obs']

    print("Prior Predictive Check")
    print("="*60)
    print(f"Prior predictive range: [{y_prior.min():.2f}, {y_prior.max():.2f}]")

    if y_data is not None:
        print(f"Observed data range:    [{y_data.min():.2f}, {y_data.max():.2f}]")

        # Check if observed within reasonable prior range
        within_range = (y_data.min() >= y_prior.min()) and (y_data.max() <= y_prior.max())

        if within_range:
            print("✓ Observed data within prior predictive range")
        else:
            print("⚠ Observed data outside prior predictive range")
            print("  → Consider adjusting priors")

    print("="*60)

    return prior_samples
