#!/usr/bin/env python3
"""
Example: NLSQ Non-linear Least Squares with JAX

Demonstrates fitting a custom model function to data using NLSQ,
leveraging JAX for automatic differentiation and GPU acceleration.
"""

import jax
import jax.numpy as jnp
from nlsq import CurveFit

def decay_model(x, params):
    """Simple exponential decay model: A * exp(-lambda * x) + C"""
    A, lambda_, C = params
    return A * jnp.exp(-lambda_ * x) + C

def main():
    print("Running NLSQ example...")

    # 1. Generate synthetic data
    key = jax.random.PRNGKey(0)
    x = jnp.linspace(0, 10, 100)
    true_params = jnp.array([5.0, 0.5, 1.0]) # A, lambda, C
    y_clean = decay_model(x, true_params)
    y_noisy = y_clean + 0.1 * jax.random.normal(key, x.shape)

    # 2. Define initial guess and bounds
    p0 = [1.0, 0.1, 0.0]
    # Bounds: A > 0, lambda > 0, C unrestricted
    bounds = ([0.0, 0.0, -jnp.inf], [jnp.inf, jnp.inf, jnp.inf])

    # 3. Fit the model using NLSQ
    # workflow="auto" selects appropriate solver based on problem size
    fitter = CurveFit(decay_model, x, y_noisy, p0=p0, bounds=bounds, workflow="auto")
    result = fitter.fit()

    # 4. Report results
    print("\nFit Results:")
    print(f"True Params: {true_params}")
    print(f"Fitted Params: {result.params}")
    print(f"Success: {result.success}")
    print(f"Cost: {result.cost:.4f}")

    if not result.success:
        print("Optimization failed.")

if __name__ == "__main__":
    main()
