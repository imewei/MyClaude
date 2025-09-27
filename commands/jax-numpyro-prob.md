---
description: Set up probabilistic models with Numpyro including distributions and MCMC sampling
category: jax-probabilistic
argument-hint: "[--model-type] [--inference] [--sampling]"
allowed-tools: "*"
---

# /jax-numpyro-prob

Set up probabilistic models and Bayesian inference with Numpyro.

## Description

Creates probabilistic models using Numpyro, a JAX-based probabilistic programming library. Includes distributions, sampling, variational inference, and MCMC methods.

## Usage

```
/jax-numpyro-prob [--model-type] [--inference] [--sampling]
```

## What it does

1. Define probabilistic models with Numpyro
2. Set up probability distributions and priors
3. Implement variational inference and MCMC sampling
4. Handle Bayesian parameter estimation

## Example output

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
import jax
import jax.numpy as jnp
import jax.random as random

# Basic probabilistic model
def linear_regression_model(X, y=None):
    \"\"\"Bayesian linear regression model.\"\"\"
    # Priors for parameters
    w = numpyro.sample('w', dist.Normal(0, 1).expand([X.shape[1]]))
    b = numpyro.sample('b', dist.Normal(0, 1))
    sigma = numpyro.sample('sigma', dist.Exponential(1))

    # Linear model
    mu = jnp.dot(X, w) + b

    # Likelihood
    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

# Hierarchical model
def hierarchical_model(group_idx, y=None):
    \"\"\"Hierarchical model with group-level effects.\"\"\"
    num_groups = len(jnp.unique(group_idx))

    # Hyperpriors
    mu_prior = numpyro.sample('mu_prior', dist.Normal(0, 10))
    sigma_prior = numpyro.sample('sigma_prior', dist.Exponential(1))

    # Group-level parameters
    with numpyro.plate('groups', num_groups):
        group_means = numpyro.sample('group_means', dist.Normal(mu_prior, sigma_prior))

    # Individual observations
    sigma_obs = numpyro.sample('sigma_obs', dist.Exponential(1))
    with numpyro.plate('data', len(y)):
        means = group_means[group_idx]
        numpyro.sample('obs', dist.Normal(means, sigma_obs), obs=y)

# Neural network as probabilistic model
def bayesian_neural_network(X, y=None, hidden_dim=50):
    \"\"\"Bayesian neural network with weight uncertainty.\"\"\"
    input_dim, output_dim = X.shape[1], 1

    # Weight priors
    w1 = numpyro.sample('w1', dist.Normal(0, 1).expand([input_dim, hidden_dim]))
    b1 = numpyro.sample('b1', dist.Normal(0, 1).expand([hidden_dim]))
    w2 = numpyro.sample('w2', dist.Normal(0, 1).expand([hidden_dim, output_dim]))
    b2 = numpyro.sample('b2', dist.Normal(0, 1).expand([output_dim]))

    # Forward pass
    hidden = jax.nn.tanh(jnp.dot(X, w1) + b1)
    mu = jnp.dot(hidden, w2) + b2

    # Observation noise
    sigma = numpyro.sample('sigma', dist.Exponential(1))

    # Likelihood
    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('obs', dist.Normal(mu.squeeze(), sigma), obs=y)

# MCMC sampling
def run_mcmc(model, X, y, num_samples=2000, num_warmup=1000):
    \"\"\"Run MCMC sampling for Bayesian inference.\"\"\"
    rng_key = random.PRNGKey(0)

    # Set up NUTS sampler
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)

    # Run sampling
    mcmc.run(rng_key, X, y)

    # Get samples
    samples = mcmc.get_samples()
    return samples, mcmc

# Variational inference
def run_svi(model, X, y, num_steps=5000):
    \"\"\"Run stochastic variational inference.\"\"\"
    rng_key = random.PRNGKey(0)

    # Automatic guide (mean-field approximation)
    guide = autoguide.AutoNormal(model)

    # Set up SVI
    optimizer = numpyro.optim.Adam(step_size=0.01)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Run optimization
    svi_result = svi.run(rng_key, num_steps, X, y)

    # Get posterior samples from guide
    guide_params = svi_result.params
    posterior_samples = guide.sample_posterior(
        random.PRNGKey(1), guide_params, sample_shape=(1000,)
    )

    return posterior_samples, svi_result

# Predictive sampling
def make_predictions(samples, model, X_new):
    \"\"\"Generate predictions using posterior samples.\"\"\"
    predictive = numpyro.infer.Predictive(model, samples)
    rng_key = random.PRNGKey(2)
    predictions = predictive(rng_key, X_new)
    return predictions

# Model comparison
def compare_models(models, X, y):
    \"\"\"Compare models using WAIC.\"\"\"
    results = {}

    for name, model in models.items():
        # Run MCMC
        samples, mcmc = run_mcmc(model, X, y)

        # Compute WAIC
        predictive = numpyro.infer.Predictive(model, samples)
        rng_key = random.PRNGKey(3)
        log_likelihood = predictive(rng_key, X, y)['obs']

        waic = numpyro.diagnostics.waic(log_likelihood)
        results[name] = {'waic': waic, 'samples': samples}

    return results

# Time series model
def ar_model(y=None, order=2):
    \"\"\"Autoregressive model for time series.\"\"\"
    # AR coefficients
    phi = numpyro.sample('phi', dist.Normal(0, 0.5).expand([order]))

    # Noise variance
    sigma = numpyro.sample('sigma', dist.Exponential(1))

    # Initial values
    init_values = numpyro.sample('init', dist.Normal(0, 1).expand([order]))

    def transition(carry, t):
        history = carry
        # AR prediction
        pred = jnp.dot(phi, history)
        # Sample next value
        next_val = numpyro.sample(f'y_{t}', dist.Normal(pred, sigma))
        # Update history
        new_history = jnp.concatenate([history[1:], jnp.array([next_val])])
        return new_history, next_val

    if y is not None:
        T = len(y)
        with numpyro.plate('time', T - order):
            _, y_pred = jax.lax.scan(transition, init_values, jnp.arange(order, T))
            numpyro.sample('obs', dist.Normal(y_pred, sigma), obs=y[order:])

# Usage example
def bayesian_analysis_workflow():
    \"\"\"Complete Bayesian analysis workflow.\"\"\"
    # Generate synthetic data
    rng_key = random.PRNGKey(0)
    X = random.normal(rng_key, (100, 3))
    true_w = jnp.array([1.5, -2.0, 0.5])
    y = jnp.dot(X, true_w) + 0.1 * random.normal(rng_key, (100,))

    # Run MCMC
    samples, mcmc = run_mcmc(linear_regression_model, X, y)

    # Print summary
    numpyro.diagnostics.print_summary(samples)

    # Make predictions
    X_new = random.normal(random.PRNGKey(1), (20, 3))
    predictions = make_predictions(samples, linear_regression_model, X_new)

    return samples, predictions
```

## Related Commands

- `/jax-init` - Set up PRNG keys for sampling
- `/jax-grad` - Compute gradients for VI optimization
- `/jax-jit` - JIT compile probabilistic models