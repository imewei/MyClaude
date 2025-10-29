---
name: numpyro-core-mastery
description: Master NumPyro probabilistic programming library for Bayesian inference using JAX. This skill should be used when working with Bayesian modeling, MCMC sampling (NUTS, HMC), variational inference (SVI), hierarchical models, uncertainty quantification, or probabilistic machine learning. Covers model specification, inference algorithms, convergence diagnostics, JAX optimization, and production deployment patterns.
---

# NumPyro Core Mastery

Master NumPyro, the JAX-based probabilistic programming library for Bayesian inference and generative modeling.

## What This Skill Provides

This skill equips Claude to become a NumPyro expert capable of:

1. **Probabilistic Model Specification** - Build Bayesian models using NumPyro primitives (sample, plate, deterministic, factor)
2. **MCMC Inference** - Run robust Markov Chain Monte Carlo with NUTS, HMC, and specialized kernels
3. **Variational Inference** - Implement fast approximate inference with SVI and AutoGuides
4. **JAX Integration** - Leverage JIT compilation, vectorization, and GPU/TPU acceleration
5. **Convergence Diagnostics** - Interpret R-hat, ESS, trace plots, and handle divergences
6. **Real-World Applications** - Apply Bayesian methods across domains (regression, time series, mixtures, hierarchical models)
7. **Production Deployment** - Build reliable, scalable, and reproducible inference pipelines

## When to Use This Skill

Invoke this skill when encountering:

- **Bayesian modeling tasks**: Building generative models with priors and likelihoods
- **Uncertainty quantification**: Need credible intervals, posterior predictive distributions
- **Hierarchical/multilevel models**: Partial pooling across groups, random effects
- **MCMC inference**: Implementing or troubleshooting NUTS, HMC, or other samplers
- **Variational inference**: Fast approximate inference with SVI
- **Convergence issues**: Divergences, low ESS, poor R-hat values
- **Probabilistic machine learning**: Bayesian neural networks, Gaussian processes
- **Time series with uncertainty**: State space models, structural time series
- **Model comparison**: WAIC, LOO cross-validation
- **JAX performance optimization**: JIT, vmap, GPU acceleration for probabilistic models

## Core Capabilities

### 1. Probabilistic Model Specification

Build Bayesian models using NumPyro's functional primitives.

**Basic workflow**:

```python
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def bayesian_model(x, y=None):
    """
    Standard pattern for NumPyro models.

    Args:
        x: Independent variable(s)
        y: Dependent variable (None for prior/posterior predictive)
    """
    # 1. Specify priors
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    # 2. Compute expected value
    mu = alpha + beta * x

    # 3. Specify likelihood with plate for independence
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)
```

**Key primitives**:
- `numpyro.sample(name, distribution, obs=data)` - Draw from prior (obs=None) or condition on data (obs=data)
- `numpyro.plate(name, size)` - Declare independence structure for vectorization
- `numpyro.deterministic(name, value)` - Track derived quantities in posterior
- `numpyro.factor(name, log_prob)` - Add arbitrary log-probability terms

**Hierarchical models**:

```python
def hierarchical_model(group_idx, x, y=None):
    """Partial pooling across groups."""
    n_groups = len(jnp.unique(group_idx))

    # Global hyperpriors
    mu_alpha = numpyro.sample('mu_alpha', dist.Normal(0, 10))
    sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(5))

    # Group-level parameters
    with numpyro.plate('groups', n_groups):
        alpha = numpyro.sample('alpha', dist.Normal(mu_alpha, sigma_alpha))

    # Likelihood
    mu = alpha[group_idx] + beta * x
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)
```

For comprehensive distribution catalog and usage patterns, see `references/distribution_catalog.md`.

### 2. MCMC Inference Workflows

Run Markov Chain Monte Carlo inference using gradient-based samplers.

**Standard NUTS workflow**:

```python
from numpyro.infer import NUTS, MCMC
import jax.random as random

# 1. Create NUTS kernel (No-U-Turn Sampler)
nuts_kernel = NUTS(
    bayesian_model,
    target_accept_prob=0.8,      # Increase to 0.9-0.95 if divergences
    max_tree_depth=10,            # Max trajectory length
    init_strategy=init_to_median()
)

# 2. Configure MCMC
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,    # Adaptation phase
    num_samples=2000,   # Sampling phase
    num_chains=4,       # Parallel chains for convergence checks
    progress_bar=True
)

# 3. Run inference
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, x_data, y_data)

# 4. Check diagnostics
mcmc.print_summary(prob=0.95)

# 5. Extract samples
posterior_samples = mcmc.get_samples()
```

**Convergence checks**:

```python
from numpyro.diagnostics import summary

summary_dict = summary(posterior_samples, prob=0.95)

# Check all parameters converged
for param, stats in summary_dict.items():
    assert stats['r_hat'] < 1.01, f"{param} not converged (R-hat={stats['r_hat']:.3f})"
    assert stats['n_eff'] > 400, f"{param} has low ESS ({stats['n_eff']:.0f})"
```

**Specialized kernels**:
- `HMC` - Manual control over step size and trajectory length
- `SA` - Slice sampler for constrained spaces
- `BarkerMH` - Robust alternative to HMC
- `MixedHMC` - Continuous + discrete parameters
- `HMCECS` - Subsampling for large datasets (N > 100K)

For detailed MCMC diagnostics and troubleshooting, see `references/mcmc_diagnostics.md`.

### 3. Variational Inference Workflows

Fast approximate inference using stochastic variational inference.

**Standard SVI workflow**:

```python
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import numpyro.optim as optim

# 1. Choose guide (variational family)
guide = AutoNormal(bayesian_model)  # Fully factorized Gaussian

# 2. Choose optimizer
optimizer = optim.Adam(step_size=0.001)

# 3. Configure SVI
svi = SVI(
    model=bayesian_model,
    guide=guide,
    optim=optimizer,
    loss=Trace_ELBO()
)

# 4. Run optimization
rng_key = random.PRNGKey(0)
svi_result = svi.run(rng_key, num_steps=10000, x_data, y_data)

# 5. Extract results
params = svi_result.params
losses = svi_result.losses

# 6. Sample from posterior approximation
posterior_samples = guide.sample_posterior(
    random.PRNGKey(1),
    params,
    sample_shape=(2000,)
)
```

**Guide options**:
- `AutoNormal` - Fully factorized Gaussian (fast, simple)
- `AutoMultivariateNormal` - Correlated Gaussian (slower, more accurate)
- `AutoLowRankMultivariateNormal` - Low-rank + diagonal (balanced)
- `AutoDelta` - Point estimate (MAP)
- `AutoLaplaceApproximation` - Laplace approximation

**When to use VI vs MCMC**:
- **VI**: Large datasets (N > 100K), need speed, embedded systems
- **MCMC**: Accurate uncertainty, complex posteriors, research/high-stakes

For complete VI guide including guide selection and convergence, see `references/variational_inference_guide.md`.

### 4. JAX Integration & Performance

Leverage JAX for high-performance probabilistic computing.

**Automatic GPU usage**:

```python
import jax
print(jax.devices())  # Check available devices

# NumPyro automatically uses GPU if available
mcmc.run(rng_key, x_data, y_data)  # Runs on GPU
```

**JIT compilation** (automatic in NumPyro):

```python
# Model functions are automatically JIT-compiled
# No manual @jit needed for models

# For custom analysis functions:
from jax import jit

@jit
def compute_statistic(samples):
    return jnp.mean(samples['alpha']) + jnp.std(samples['beta'])

result = compute_statistic(posterior_samples)  # Compiled on first call
```

**Vectorization with vmap**:

```python
from jax import vmap

# Evaluate model at multiple parameter sets
log_probs = vmap(lambda params: model_log_prob(params, x, y))(parameter_grid)
```

**PRNG handling**:

```python
# JAX uses explicit PRNG keys
rng_key = random.PRNGKey(42)

# Split for multiple uses
key1, key2, key3 = random.split(rng_key, 3)

mcmc.run(key1, x, y)
predictions = posterior_predictive(key2, x_new)
```

**Performance tips**:
1. Use vectorized operations (avoid Python loops)
2. Run multiple chains in parallel (`num_chains=4`)
3. GPU benefit increases with data size (optimal for N > 10K)
4. Clear caches if memory issues: `jax.clear_caches()`

### 5. Convergence Diagnostics & Debugging

Ensure reliable inference through comprehensive diagnostics.

**R-hat (convergence)**:
- R-hat < 1.01: Converged ✓
- R-hat > 1.01: Run longer or check model

**Effective Sample Size (efficiency)**:
- ESS > 400: Good ✓
- ESS < 400: High autocorrelation, need more samples

**Divergences (numerical stability)**:

```python
# Check divergences
num_divergences = mcmc.get_extra_fields()['diverging'].sum()
print(f"Divergences: {num_divergences}")

# If divergences occur:
# 1. Increase target_accept_prob to 0.9 or 0.95
# 2. Reparameterize (non-centered parameterization)
# 3. Use more informative priors
```

**Non-centered parameterization** (fixes divergences in hierarchical models):

```python
# CENTERED (may diverge)
def centered():
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    theta = numpyro.sample('theta', dist.Normal(mu, sigma))

# NON-CENTERED (better)
def noncentered():
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    theta_raw = numpyro.sample('theta_raw', dist.Normal(0, 1))
    theta = mu + sigma * theta_raw  # Manual transformation
```

**Trace plots**:

```python
import arviz as az

idata = az.from_numpyro(mcmc)
az.plot_trace(idata, var_names=['alpha', 'beta'])
```

Use `scripts/mcmc_diagnostics.py` for automated comprehensive diagnostics.

### 6. Predictive Distributions

Generate predictions and validate models.

**Prior predictive checks**:

```python
from numpyro.infer import Predictive

# Sample from prior (before seeing data)
prior_predictive = Predictive(model, num_samples=1000)
prior_samples = prior_predictive(random.PRNGKey(0), x_data, y=None)

# Check if prior generates reasonable data
y_prior = prior_samples['obs']
print(f"Prior range: [{y_prior.min():.1f}, {y_prior.max():.1f}]")
```

**Posterior predictive checks**:

```python
# Generate predictions using posterior
posterior_predictive = Predictive(model, posterior_samples)
ppc_samples = posterior_predictive(random.PRNGKey(1), x_data, y=None)
y_ppc = ppc_samples['obs']

# Compare to observed data
import matplotlib.pyplot as plt
plt.hist(y_ppc.flatten(), bins=50, alpha=0.5, label='Posterior predictive')
plt.hist(y_observed, bins=50, alpha=0.5, label='Observed')
plt.legend()
```

**Posterior predictions on new data**:

```python
# Predict at new x values
predictions = posterior_predictive(random.PRNGKey(2), x_new, y=None)
y_pred = predictions['obs']

# Credible intervals
y_mean = y_pred.mean(axis=0)
y_lower = jnp.percentile(y_pred, 2.5, axis=0)
y_upper = jnp.percentile(y_pred, 97.5, axis=0)
```

Use `scripts/prior_predictive_check.py` for automated prior validation.

### 7. Model Comparison & Selection

Compare models using information criteria.

**WAIC (Widely Applicable Information Criterion)**:

```python
from numpyro.diagnostics import waic

waic_result = waic(model, posterior_samples, x_data, y_data)
print(f"WAIC: {waic_result.waic:.2f} ± {waic_result.waic_se:.2f}")
```

**LOO (Leave-One-Out Cross-Validation)**:

```python
from numpyro.diagnostics import loo

loo_result = loo(model, posterior_samples, x_data, y_data)
print(f"LOO: {loo_result.loo:.2f} ± {loo_result.loo_se:.2f}")
```

**Compare models**:

```python
# Lower is better
if model1_loo < model2_loo:
    print("Model 1 has better predictive performance")
```

Use `scripts/model_comparison.py` for systematic model comparison workflows.

## Workflow Patterns

### Pattern 1: Quick Bayesian Inference

For simple models and exploratory analysis.

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
import jax.numpy as jnp
import jax.random as random

# 1. Define model
def model(x, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    mu = alpha + beta * x
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

# 2. Run MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(random.PRNGKey(0), x, y)

# 3. Analyze
mcmc.print_summary()
```

### Pattern 2: Production-Ready Inference

With comprehensive diagnostics and error handling.

```python
def production_inference(x, y, max_retries=3):
    """Robust inference with automatic retry."""

    for attempt in range(max_retries):
        # Increase robustness with each retry
        target_accept = 0.8 + 0.05 * attempt

        nuts_kernel = NUTS(model, target_accept_prob=target_accept)
        mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=4)

        try:
            mcmc.run(random.PRNGKey(attempt), x, y)

            # Check convergence
            summary_dict = summary(mcmc.get_samples())
            converged = all(s['r_hat'] < 1.01 for s in summary_dict.values())

            if converged:
                # Check for divergences
                divergences = mcmc.get_extra_fields()['diverging'].sum()
                if divergences == 0:
                    return mcmc.get_samples()
                else:
                    print(f"Attempt {attempt+1}: {divergences} divergences, retrying...")
            else:
                print(f"Attempt {attempt+1}: Convergence failed, retrying...")

        except Exception as e:
            print(f"Attempt {attempt+1}: Error {e}, retrying...")

    raise RuntimeError("Inference failed after max retries")
```

### Pattern 3: Large-Scale Inference

For datasets with N > 100K observations.

**Option 1: Variational Inference (fast)**:

```python
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

guide = AutoNormal(model)
optimizer = numpyro.optim.Adam(0.001)
svi = SVI(model, guide, optimizer, Trace_ELBO())

# Fast optimization (seconds to minutes)
svi_result = svi.run(random.PRNGKey(0), 10000, x_large, y_large)
```

**Option 2: MCMC with subsampling**:

```python
from numpyro.infer import HMCECS

# Subsample data for each MCMC step
hmcecs = HMCECS(model, subsample_size=1000)
mcmc = MCMC(hmcecs, num_warmup=500, num_samples=1000)
mcmc.run(random.PRNGKey(0), x_large, y_large)
```

Use `scripts/benchmark_mcmc_vi.py` to compare MCMC vs VI performance.

### Pattern 4: Research Implementation

Custom inference and advanced techniques.

**Custom guide for VI**:

```python
def custom_guide(x, y=None):
    # Variational parameters
    alpha_loc = numpyro.param('alpha_loc', 0.0)
    alpha_scale = numpyro.param('alpha_scale', 1.0,
                                constraint=constraints.positive)

    # Variational distributions
    numpyro.sample('alpha', dist.Normal(alpha_loc, alpha_scale))
```

**Effect handlers** for model surgery:

```python
from numpyro.handlers import seed, substitute, reparam
from numpyro.infer.reparam import LocScaleReparam

# Reparameterize for better geometry
with reparam(config={'theta': LocScaleReparam()}):
    mcmc.run(rng_key, x, y)
```

For detailed effect handlers usage, see `references/effect_handlers_guide.md`.

## Utility Scripts

Execute comprehensive diagnostics and comparisons:

### MCMC Diagnostics

Automated convergence analysis:

```bash
python scripts/mcmc_diagnostics.py --samples posterior_samples.pkl
```

Checks:
- R-hat for all parameters
- Effective sample size (ESS)
- Divergences and warnings
- Trace plot generation
- Actionable recommendations

### Model Comparison

Compare multiple models systematically:

```bash
python scripts/model_comparison.py --models model1.pkl model2.pkl model3.pkl --data data.pkl
```

Computes:
- WAIC with standard errors
- LOO with Pareto-k diagnostics
- Model selection recommendations
- Visualization of comparison

### Prior Predictive Checks

Validate prior specifications:

```bash
python scripts/prior_predictive_check.py --model model.py --data data.pkl
```

Generates:
- Prior predictive samples
- Visualization against data range
- Prior sensitivity analysis
- Recommendations for prior tuning

### MCMC vs VI Benchmark

Compare inference methods:

```bash
python scripts/benchmark_mcmc_vi.py --model model.py --data data.pkl --sizes 1000,10000,100000
```

Benchmarks:
- MCMC (NUTS) timing and accuracy
- VI (SVI) timing and accuracy
- Speedup analysis
- Recommendations for dataset size

## Best Practices

**Model building**:
1. Start simple, iterate complexity
2. Check prior predictive before fitting
3. Use weakly informative priors by default
4. Document prior choices and model assumptions

**Inference**:
1. Always run multiple chains (≥4) for MCMC
2. Check R-hat < 1.01 and ESS > 400 for convergence
3. Address divergences immediately (don't ignore)
4. Use VI for exploration, MCMC for final inference

**Diagnostics**:
1. Posterior predictive checks are essential
2. Trace plots reveal convergence issues visually
3. Model comparison guides model selection
4. Save posteriors for reproducibility

**Performance**:
1. Leverage GPU for N > 10K
2. Use vectorized operations (avoid loops)
3. JIT-compile custom analysis functions
4. Consider VI for very large datasets (N > 100K)

**Reproducibility**:
1. Set explicit PRNG keys
2. Document all hyperparameters
3. Save posterior samples
4. Version control model code

## Real-World Applications

**Regression**: Linear, logistic, hierarchical regression with uncertainty quantification

**Time Series**: State space models, structural time series, stochastic volatility

**Hierarchical Models**: Partial pooling, multilevel regression, random effects

**Mixture Models**: Gaussian mixtures, zero-inflated models, robust regression

**Survival Analysis**: Proportional hazards, time-to-event modeling

**Causal Inference**: Treatment effects, instrumental variables, counterfactuals

**Probabilistic ML**: Bayesian neural networks, Gaussian processes, VAEs

## References

Comprehensive guides for deep dives:

- **`references/mcmc_diagnostics.md`** - Complete MCMC convergence diagnostics, troubleshooting divergences, reparameterization strategies, trace analysis
- **`references/variational_inference_guide.md`** - VI theory, guide selection, ELBO optimization, when to use VI vs MCMC
- **`references/distribution_catalog.md`** - Full NumPyro distributions reference with use cases, parameterizations, and conjugacy
- **`references/effect_handlers_guide.md`** - Effect handlers (seed, substitute, trace, reparam), composition patterns, advanced usage

## Examples Collection

**Comprehensive catalog**: See `scripts/examples/README.md` for complete listing of 50+ NumPyro examples.

**Categories**:
- **Basic**: Regression, GLM, statistical testing
- **Hierarchical**: Baseball, funnel, multilevel models
- **Time Series**: AR, HMM, forecasting, state space
- **Advanced**: GP, BNN, VAE, mixture models, neural approaches

**Access**:
- Examples catalog: `scripts/examples/README.md`
- Source repository: `/Users/b80985/Documents/GitHub/numpyro/examples/`
- Online: https://num.pyro.ai/en/latest/examples.html

## Resources

**Official NumPyro**:
- Documentation: https://num.pyro.ai/
- GitHub: https://github.com/pyro-ppl/numpyro
- Examples: https://num.pyro.ai/en/latest/examples.html

**JAX**:
- Documentation: https://jax.readthedocs.io/
- GitHub: https://github.com/google/jax

**Bayesian Modeling**:
- *Statistical Rethinking* by Richard McElreath
- *Bayesian Data Analysis* by Gelman et al.
- ArviZ for visualization: https://arviz-devs.github.io/arviz/

**Community**:
- PyMC Discourse (Bayesian community): https://discourse.pymc.io/
- NumPyro GitHub Discussions: https://github.com/pyro-ppl/numpyro/discussions

---

**NumPyro Core Mastery Skill** - Master Bayesian inference with JAX

Version: 1.0.0
Last Updated: 2025-10-28
Compatible with: NumPyro 0.15+, JAX 0.4+
