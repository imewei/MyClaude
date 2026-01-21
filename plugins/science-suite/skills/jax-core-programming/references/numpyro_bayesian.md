# NumPyro Bayesian Patterns - Quick Reference

## Overview

NumPyro is a probabilistic programming library built on JAX for Bayesian inference. This guide covers integration patterns with JAX Core workflows.

**For comprehensive NumPyro expertise, refer to the `numpyro-core-mastery` skill.**

---

## 1. Basic Model Specification

### Simple Linear Regression

```python
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def linear_model(X, y=None):
    # Priors
    w = numpyro.sample('w', dist.Normal(0, 1).expand([X.shape[1]]))
    b = numpyro.sample('b', dist.Normal(0, 1))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))

    # Likelihood
    mean = X @ w + b
    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('y', dist.Normal(mean, sigma), obs=y)
```

### Hierarchical Model

```python
def hierarchical_model(group_idx, y=None):
    # Hyperpriors
    mu_mu = numpyro.sample('mu_mu', dist.Normal(0, 10))
    sigma_mu = numpyro.sample('sigma_mu', dist.HalfNormal(10))

    # Group-level parameters
    n_groups = len(jnp.unique(group_idx))
    with numpyro.plate('groups', n_groups):
        group_mu = numpyro.sample('group_mu', dist.Normal(mu_mu, sigma_mu))

    # Observation-level
    mu = group_mu[group_idx]
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))

    with numpyro.plate('data', len(y)):
        numpyro.sample('y', dist.Normal(mu, sigma), obs=y)
```

---

## 2. MCMC Inference

### NUTS Sampling

```python
from numpyro.infer import MCMC, NUTS

# Setup MCMC
nuts_kernel = NUTS(linear_model)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4
)

# Run inference (JAX automatically parallelizes chains)
mcmc.run(jax.random.PRNGKey(0), X, y)

# Get samples
samples = mcmc.get_samples()
print(samples['w'].shape)  # (2000, n_features)
```

### With JAX Optimizations

```python
# JIT-compiled MCMC
@jax.jit
def run_inference(rng_key, X, y):
    kernel = NUTS(linear_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
    mcmc.run(rng_key, X, y)
    return mcmc.get_samples()

# Vectorized prediction
@jax.vmap
def predict_sample(w, b, X):
    return X @ w + b

# Use samples for prediction
predictions = predict_sample(samples['w'], samples['b'], X_test)
```

---

## 3. Variational Inference

### SVI with AutoGuide

```python
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

# Automatic variational family
guide = AutoNormal(linear_model)

# SVI optimizer
optimizer = numpyro.optim.Adam(0.01)
svi = SVI(linear_model, guide, optimizer, loss=Trace_ELBO())

# Run SVI (fast approximate inference)
svi_result = svi.run(jax.random.PRNGKey(0), 2000, X, y)

# Get approximate posterior
params = svi_result.params
samples = guide.sample_posterior(
    jax.random.PRNGKey(1),
    params,
    sample_shape=(1000,)
)
```

---

## 4. JAX Integration Patterns

### Batched Inference

```python
# Inference over multiple datasets
@jax.vmap
def batched_inference(rng_key, X, y):
    kernel = NUTS(linear_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
    mcmc.run(rng_key, X, y)
    return mcmc.get_samples()

# Run on multiple datasets
rng_keys = jax.random.split(jax.random.PRNGKey(0), n_datasets)
all_samples = batched_inference(rng_keys, X_batch, y_batch)
```

### Custom Distributions with JAX

```python
from numpyro.distributions import Distribution

class CustomDist(Distribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        super().__init__(batch_shape=loc.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        # JAX random sampling
        eps = jax.random.normal(key, sample_shape + self.batch_shape)
        return self.loc + self.scale * eps

    def log_prob(self, value):
        # JAX log probability computation
        return -0.5 * ((value - self.loc) / self.scale) ** 2
```

---

## 5. Bayesian Neural Networks

### Simple BNN

```python
def bnn(X, y=None, hidden_dim=50):
    n_features = X.shape[-1]

    # Priors for weights
    w1 = numpyro.sample('w1', dist.Normal(0, 1).expand([n_features, hidden_dim]))
    w2 = numpyro.sample('w2', dist.Normal(0, 1).expand([hidden_dim, 1]))

    # Forward pass
    hidden = jax.nn.relu(X @ w1)
    mean = (hidden @ w2).squeeze(-1)

    # Likelihood
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('y', dist.Normal(mean, sigma), obs=y)
```

### With Flax NNX

```python
from flax import nnx

class BayesianFlaxModel(nnx.Module):
    def __init__(self, rngs):
        self.dense1 = nnx.Linear(10, 50, rngs=rngs)
        self.dense2 = nnx.Linear(50, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)

def bayesian_flax_model(X, y=None):
    # Sample network parameters
    model = BayesianFlaxModel(rngs=nnx.Rngs(0))

    # Place priors on all parameters
    for name, param in nnx.state(model).items():
        numpyro.sample(name, dist.Normal(0, 1).expand(param.shape))

    # Forward pass with sampled params
    mean = model(X).squeeze(-1)

    # Likelihood
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('y', dist.Normal(mean, sigma), obs=y)
```

---

## 6. Predictive Distributions

### Posterior Predictive

```python
from numpyro.infer import Predictive

# Get posterior predictive samples
predictive = Predictive(linear_model, samples)
predictions = predictive(jax.random.PRNGKey(1), X_test)

# Uncertainty quantification
y_pred_mean = predictions['y'].mean(axis=0)
y_pred_std = predictions['y'].std(axis=0)

# Credible intervals
lower = jnp.percentile(predictions['y'], 2.5, axis=0)
upper = jnp.percentile(predictions['y'], 97.5, axis=0)
```

---

## 7. Integration with JAX Core

### JIT-Compiled Bayesian Workflow

```python
@jax.jit
def bayesian_training_step(rng_key, X, y):
    """Complete Bayesian inference in one JIT-compiled function"""

    # MCMC sampling
    kernel = NUTS(linear_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
    mcmc.run(rng_key, X, y)
    samples = mcmc.get_samples()

    # Posterior predictive
    predictive = Predictive(linear_model, samples)
    predictions = predictive(rng_key, X)

    return samples, predictions

# Usage
samples, predictions = bayesian_training_step(
    jax.random.PRNGKey(0), X_train, y_train
)
```

---

## 8. Performance Tips

```python
"""
NumPyro + JAX Performance Tips:

1. Use jax.jit for predictive distributions
2. Use jax.vmap for batched inference
3. Set num_chains > 1 for parallel sampling
4. Use SVI for large datasets
5. Pre-compile models before training loops
6. Use predictive caching for repeated predictions
"""

# Example: Cached predictions
@jax.jit
def cached_predict(samples, X):
    predictive = Predictive(model, samples)
    return predictive(jax.random.PRNGKey(0), X)
```

---

## References

- **For comprehensive NumPyro expertise**: See `numpyro-core-mastery` skill
- [NumPyro Documentation](https://num.pyro.ai/)
- [NumPyro Examples](https://num.pyro.ai/en/latest/examples.html)
- [JAX + NumPyro Tutorial](https://num.pyro.ai/en/latest/getting_started.html)
