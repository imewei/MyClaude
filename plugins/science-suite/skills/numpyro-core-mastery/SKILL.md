---
name: numpyro-core-mastery
description: Master NumPyro for production Bayesian inference, MCMC sampling (NUTS/HMC), variational inference (SVI), hierarchical models, and uncertainty quantification. Use when building probabilistic models with numpyro.sample(), running MCMC with NUTS/HMC, implementing SVI with AutoGuides, diagnosing convergence (R-hat, ESS, divergences), or deploying production Bayesian pipelines.
---

# NumPyro Core Mastery

Production Bayesian inference with JAX-accelerated probabilistic programming.

## Expert Agents

For complex Bayesian modeling, hierarchical inference, and probabilistic programming tasks, delegate to:

- **`jax-pro`** (primary): JAX-accelerated NumPyro implementation, NUTS/HMC tuning, SVI, AutoGuides, and differentiable physics integration.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
- **`statistical-physicist`** (secondary): Bayesian inference theory, prior elicitation, identifiability analysis, sampler geometry, PSIS-LOO model comparison.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`

For the Julia counterpart (Turing.jl) see `turing-model-design`. For multimodal posteriors that defeat NUTS, see `consensus-mcmc-pigeons`. Convergence checks live in `mcmc-diagnostics`.

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

> **ArviZ v1.0 (2025) is a major-version bump.** Function names (`from_numpyro`, `rhat`, `ess`, `summary`, `loo`, `waic`, `plot_*`) are preserved but internal behavior changed — verify compatibility for pipelines written against `arviz<1.0` before upgrading. See `mcmc-diagnostics` for the full Python-workflow diagnostics block.

---

## Related JAX Bayesian Tools

NumPyro is the default JAX PPL, but several neighboring libraries cover adjacent niches. Reach for them when NumPyro alone is the wrong shape:

| Tool | When to use |
|------|-------------|
| **BlackJAX** | Hand-rolled sampler loops, composable kernels (NUTS, HMC, MALA, MCLMC, tempered SMC, Pathfinder, SVGD), and low-level `init`/`step` interfaces over any JAX `logdensity_fn` — including NumPyro models via `numpyro.infer.util.potential_energy` |
| **GPJax** | Feature-rich JAX Gaussian processes (exact + sparse variational, Laplace / MCMC hyperpriors). Integrates with NumPyro for fully-Bayesian hyperparameters |
| **tinygp** | Minimal JAX GP with celerite2-style quasi-separable O(N) kernels for 1D timeseries. Drop a `GaussianProcess(kernel, X).log_probability(y)` directly inside a `@numpyro.sample` block |
| **emcee** | Goodman-Weare affine-invariant ensemble MCMC (NumPy). Use for legacy / black-box log-probs or when the overhead of NUTS warmup isn't worth it for cheap likelihoods |
| **pocoMC** | Preconditioned tempered SMC with normalizing-flow reparameterization. Handles multimodal and strongly-correlated posteriors, and returns marginal likelihood (Z) alongside samples |
| **corner** | Posterior corner plots from `(nsamples, ndim)` arrays or ArviZ `InferenceData` — use with `emcee` / `numpyro.infer.MCMC` for publication figures |
| **Pigeons.jl** (via `juliacall`) | Non-Reversible Parallel Tempering for multimodal posteriors where NUTS / tempered SMC still fail to mix. See `consensus-mcmc-pigeons` for the Julia-native path |

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

## Parallel Inference

| Strategy | Implementation | Use Case |
|----------|----------------|----------|
| **Parallel Chains** | `MCMC(num_chains=4, chain_method='parallel')` | Standard production use |
| **Vectorized Map** | `numpyro.plate('data', N)` | Independent data points |
| **Pmap (Data)** | `pmap(lambda k: run_mcmc(k, data_shard))` | Distributed data (consensus) |
| **Pmap (Model)** | `jax.sharding` mesh | Large model parameter count |

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

**Version**: 1.0.6
