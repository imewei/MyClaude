#!/usr/bin/env python3
"""
Example: Bayesian Regression with NumPyro

Demonstrates a simple Bayesian linear regression model using NumPyro's
NUTS sampler. This serves as a template for more complex inference tasks.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def linear_model(x, y=None):
    """
    Standard Bayesian Linear Regression: y = alpha + beta * x + epsilon
    """
    # 1. Define priors
    alpha = numpyro.sample('alpha', dist.Normal(0.0, 10.0))
    beta = numpyro.sample('beta', dist.Normal(0.0, 10.0))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1.0))

    # 2. Define likelihood
    mu = alpha + beta * x

    # "plate" for independent observations
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

def main():
    print("Running NumPyro example...")

    # 1. Generate synthetic data
    key = jax.random.PRNGKey(42)
    key_data, key_infer = jax.random.split(key)

    N = 100
    x = jax.random.normal(key_data, (N,))
    true_alpha, true_beta, true_sigma = 1.0, 2.5, 0.5
    y = true_alpha + true_beta * x + true_sigma * jax.random.normal(key_data, (N,))

    # 2. Setup Inference
    # NUTS is the No-U-Turn Sampler, a state-of-the-art MCMC algorithm
    kernel = NUTS(linear_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)

    # 3. Run Inference
    print("Starting MCMC sampling...")
    mcmc.run(key_infer, x=x, y=y)

    # 4. Report Results
    mcmc.print_summary()
    samples = mcmc.get_samples()

    print("\nPosterior Means:")
    print(f"Alpha: {jnp.mean(samples['alpha']):.3f} (True: {true_alpha})")
    print(f"Beta:  {jnp.mean(samples['beta']):.3f}  (True: {true_beta})")
    print(f"Sigma: {jnp.mean(samples['sigma']):.3f} (True: {true_sigma})")

if __name__ == "__main__":
    main()
