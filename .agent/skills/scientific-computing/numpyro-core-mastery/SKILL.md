---
name: numpyro-core-mastery
version: "1.0.7"
maturity: "5-Expert"
specialization: Bayesian Inference with JAX
description: Master NumPyro for production Bayesian inference, MCMC sampling (NUTS/HMC), variational inference (SVI), hierarchical models, and uncertainty quantification. Use when building probabilistic models with numpyro.sample(), running MCMC with NUTS/HMC, implementing SVI with AutoGuides, diagnosing convergence (R-hat, ESS, divergences), or deploying production Bayesian pipelines.
---

# NumPyro Core Mastery

Production Bayesian inference with JAX-accelerated probabilistic programming.

---

## Inference Method Selection

| Method | Data Size | Speed | Accuracy | Use Case |
|--------|-----------|-------|----------|----------|
| NUTS | < 100K | Slow | Exact | Research, complex posteriors |
| HMC | < 100K | Medium | Exact | Manual tuning needed |
| SVI (AutoNormal) | > 100K | Fast | Approximate | Exploration, production |
| HMCECS | > 100K | Medium | Approximate | Large-scale MCMC |
| Consensus MC | > 1M | Fast | Approximate | Distributed data |

---

## Model Building

```python
import numpyro
import numpyro.distributions as dist

def bayesian_regression(x, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    mu = alpha + beta * x
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

def hierarchical_model(group_idx, x, y=None):
    n_groups = len(jnp.unique(group_idx))

    # Hyperpriors
    mu_alpha = numpyro.sample('mu_alpha', dist.Normal(0, 10))
    sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(5))

    # Group effects (non-centered for better sampling)
    with numpyro.plate('groups', n_groups):
        alpha_raw = numpyro.sample('alpha_raw', dist.Normal(0, 1))
    alpha = mu_alpha + sigma_alpha * alpha_raw

    mu = alpha[group_idx]
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, 1), obs=y)
```

---

## MCMC Inference

```python
from numpyro.infer import NUTS, MCMC, init_to_median
import jax.random as random

# Standard NUTS
nuts_kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=10,
                   init_strategy=init_to_median())
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(random.PRNGKey(0), x, y)

# Check convergence
mcmc.print_summary(prob=0.95)
posterior_samples = mcmc.get_samples()

# Access diagnostics
extra = mcmc.get_extra_fields()
print(f"Divergences: {extra['diverging'].sum()}")
```

---

## Variational Inference

```python
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoMultivariateNormal
import numpyro.optim as optim

guide = AutoNormal(model)  # Factorized Gaussian
svi = SVI(model, guide, optim.Adam(0.001), Trace_ELBO())
svi_result = svi.run(random.PRNGKey(0), 10000, x, y)

# Sample from approximate posterior
posterior_samples = guide.sample_posterior(random.PRNGKey(1),
                                           svi_result.params,
                                           sample_shape=(2000,))
```

### Guide Selection

| Guide | Correlation | Speed | Use Case |
|-------|-------------|-------|----------|
| AutoNormal | None | Fastest | Independent parameters |
| AutoMultivariateNormal | Full | Slower | Correlated posterior |
| AutoLowRankMVN | Partial | Medium | Large models |
| AutoDelta | None | Fastest | MAP estimation |

---

## ArviZ Integration

```python
import arviz as az

# Convert to InferenceData
idata = az.from_numpyro(mcmc,
    dims={"obs": ["time"]},
    coords={"time": np.arange(len(y))})

# Diagnostics
az.plot_trace(idata, var_names=['alpha', 'beta'])
az.plot_rank(idata)
az.plot_energy(idata)

# Convergence checks
rhat = az.rhat(idata)
ess = az.ess(idata)

# Model comparison
loo = az.loo(idata, pointwise=True)
waic = az.waic(idata)
```

---

## Predictive Distributions

```python
from numpyro.infer import Predictive

# Prior predictive
prior_pred = Predictive(model, num_samples=1000)
prior_samples = prior_pred(random.PRNGKey(0), x, y=None)

# Posterior predictive
posterior_pred = Predictive(model, posterior_samples)
ppc = posterior_pred(random.PRNGKey(1), x, y=None)

# Credible intervals
y_mean = ppc['obs'].mean(axis=0)
y_lower = jnp.percentile(ppc['obs'], 2.5, axis=0)
y_upper = jnp.percentile(ppc['obs'], 97.5, axis=0)
```

---

## Convergence Diagnostics

| Metric | Target | Action if Failed |
|--------|--------|------------------|
| R-hat | < 1.01 | Run longer chains |
| ESS | > 400 | Increase samples |
| Divergences | 0 | Reparameterize or increase target_accept |
| Tree depth | < max | Reduce max_tree_depth or reparameterize |

### Fixing Divergences

```python
# 1. Increase target acceptance
nuts = NUTS(model, target_accept_prob=0.95)

# 2. Non-centered parameterization
def noncentered():
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    theta_raw = numpyro.sample('theta_raw', dist.Normal(0, 1))
    theta = mu + sigma * theta_raw  # Transform

# 3. Use LocScaleReparam
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

with reparam(config={'theta': LocScaleReparam()}):
    mcmc.run(rng_key, x, y)
```

---

## Best Practices

| Area | Practice |
|------|----------|
| **Priors** | Start weakly informative, check prior predictive |
| **MCMC** | Run ≥4 chains, check R-hat < 1.01, ESS > 400 |
| **Divergences** | Fix immediately, don't ignore |
| **VI** | Use for exploration, MCMC for final inference |
| **Validation** | Posterior predictive checks essential |
| **PRNG** | Explicit keys, split properly |
| **Performance** | GPU for N > 10K, JIT custom functions |

---

## Checklist

- [ ] Model specified with sample/plate/deterministic
- [ ] Prior predictive checked (reasonable data range)
- [ ] MCMC: R-hat < 1.01, ESS > 400, zero divergences
- [ ] Trace plots examined for convergence
- [ ] Posterior predictive validates model fit
- [ ] Results saved with params and samples
- [ ] PRNG keys documented for reproducibility

---

**Version**: 1.0.5
