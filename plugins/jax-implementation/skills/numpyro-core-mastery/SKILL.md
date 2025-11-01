---
name: numpyro-core-mastery
description: Master NumPyro, the JAX-based probabilistic programming library for production-ready Bayesian inference, MCMC sampling, and variational inference. Use this skill when writing or modifying Python files that import NumPyro (import numpyro, import numpyro.distributions as dist, from numpyro.infer import NUTS, MCMC, SVI, Predictive, from numpyro.infer.autoguide import AutoNormal), when building Bayesian statistical models with prior and likelihood specifications using numpyro.sample() calls, when implementing MCMC inference with NUTS (No-U-Turn Sampler), HMC (Hamiltonian Monte Carlo), or specialized kernels like HMCECS or DiscreteHMCGibbs, when running variational inference (SVI) with AutoGuides for fast approximate Bayesian inference, when designing hierarchical or multilevel models with partial pooling and random effects, when performing uncertainty quantification with posterior distributions and credible intervals, when diagnosing MCMC convergence issues including R-hat statistics, effective sample size (ESS), divergences, and trace plot analysis, when implementing non-centered parameterization or reparameterization strategies to resolve divergences in hierarchical models, when building probabilistic machine learning models including Bayesian neural networks, Gaussian processes, or variational autoencoders, when working with time series models such as state space models, structural time series, or autoregressive processes, when performing Bayesian model comparison using WAIC (Widely Applicable Information Criterion) or LOO (Leave-One-Out) cross-validation, when generating prior predictive distributions for prior sensitivity checks or posterior predictive distributions for model validation, when integrating with ArviZ for comprehensive visualization and diagnostics (trace plots, posterior plots, energy plots, convergence diagnostics), when optimizing Bayesian inference performance with JAX features including JIT compilation, GPU/TPU acceleration, vmap vectorization, or pmap parallelization, when implementing custom variational guides for specialized inference problems, when using NumPyro effect handlers for model surgery (seed, substitute, condition, reparam, trace), when analyzing posterior samples or computing posterior summaries and derived quantities, when implementing Consensus Monte Carlo for large-scale distributed Bayesian inference on datasets with millions of observations, when handling missing data through imputation models or marginalization, when deploying production Bayesian inference pipelines with reproducible PRNG key management, or when working on scientific computing projects requiring rigorous uncertainty quantification and statistical inference.
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

Invoke this skill when encountering any of the following scenarios:

### File Types and Code Patterns
- Writing or modifying Python files (`.py`, `.ipynb`) that import NumPyro libraries
- Working with files containing `import numpyro`, `import numpyro.distributions as dist`, `from numpyro.infer import NUTS, MCMC, SVI`
- Building Bayesian models with `numpyro.sample()`, `numpyro.plate()`, `numpyro.deterministic()` calls
- Implementing probabilistic models with prior and likelihood specifications

### MCMC Inference Tasks
- Implementing MCMC sampling with NUTS (No-U-Turn Sampler) or HMC (Hamiltonian Monte Carlo)
- Running inference with specialized kernels: HMCECS (subsampling), DiscreteHMCGibbs (discrete parameters), BarkerMH, or SA
- Configuring MCMC parameters: `num_warmup`, `num_samples`, `num_chains`, `target_accept_prob`, `max_tree_depth`
- Implementing Consensus Monte Carlo for distributed Bayesian inference on large-scale datasets (N > 1M observations)

### Variational Inference
- Running variational inference (SVI) with AutoGuides for fast approximate Bayesian inference
- Implementing custom variational guides for specialized inference problems
- Optimizing ELBO (Evidence Lower Bound) with Adam, SGD, or other optimizers
- Working with AutoNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal, or AutoDelta guides

### Model Design and Architecture
- Designing hierarchical or multilevel models with partial pooling and random effects
- Building generative models with priors and likelihoods for Bayesian data analysis
- Implementing non-centered parameterization to resolve divergences in hierarchical models
- Creating probabilistic machine learning models: Bayesian neural networks, Gaussian processes, variational autoencoders
- Working with time series models: state space models, structural time series, autoregressive processes
- Handling missing data through Bayesian imputation models or marginalization

### Convergence Diagnostics and Troubleshooting
- Diagnosing MCMC convergence issues: R-hat statistics, effective sample size (ESS), divergences
- Analyzing trace plots and posterior samples for convergence assessment
- Implementing reparameterization strategies (centered vs non-centered parameterization) to improve MCMC geometry
- Debugging divergences with `target_accept_prob` tuning, dense mass matrices, or alternative samplers
- Accessing NUTS diagnostics: tree depth, energy error, acceptance probability

### Model Validation and Comparison
- Performing Bayesian model comparison using WAIC or LOO cross-validation
- Generating prior predictive distributions for prior sensitivity checks
- Generating posterior predictive distributions for model validation and posterior predictive checks
- Computing model selection criteria and posterior predictive accuracy

### Visualization and Diagnostics with ArviZ
- Integrating NumPyro with ArviZ for comprehensive Bayesian workflow visualization
- Converting MCMC results to ArviZ InferenceData format with dimensions, coordinates, and constant data
- Creating diagnostic plots: trace plots, posterior plots, forest plots, energy plots, rank plots, autocorrelation plots
- Performing posterior predictive checks (PPC) and Bayesian p-value analysis
- Computing numerical diagnostics: R-hat, ESS, MCSE (Monte Carlo Standard Error)

### JAX Performance Optimization
- Optimizing Bayesian inference with JAX JIT compilation for faster sampling
- Leveraging GPU/TPU acceleration for large-scale Bayesian models
- Using vmap vectorization for batch operations and parallel evaluation
- Using pmap for multi-device parallelization
- Implementing efficient PRNG key management with `jax.random.PRNGKey` and `jax.random.split`

### Advanced NumPyro Features
- Using effect handlers for model surgery: `seed`, `substitute`, `condition`, `reparam`, `trace`
- Working with custom distributions or transformations
- Implementing LocScaleReparam or other reparameterization strategies
- Computing posterior summaries and derived quantities with `numpyro.deterministic`
- Analyzing posterior samples and extracting parameter estimates

### Production and Deployment
- Deploying production Bayesian inference pipelines with reproducible PRNG handling
- Implementing inference serving for real-time or batch prediction
- Serializing and saving posterior samples for later use
- Building scalable Bayesian systems for scientific computing and decision-making under uncertainty
- Working on scientific research projects requiring rigorous uncertainty quantification

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

**Advanced NUTS configuration**:

```python
from numpyro.infer import NUTS, MCMC, init_to_median

# Fine-tuned NUTS for challenging posteriors
nuts_kernel = NUTS(
    model,
    target_accept_prob=0.9,          # Higher = more robust, slower
    max_tree_depth=12,                # Max trajectory length (2^12 = 4096 steps)
    init_strategy=init_to_median(),   # Initialize at prior median
    dense_mass=False                  # Diagonal mass matrix (faster)
)

# For highly correlated posteriors, use dense mass matrix
nuts_kernel_dense = NUTS(
    model,
    dense_mass=True,                  # Full covariance adaptation
    adapt_step_size=True,             # Dual averaging for step size
    adapt_mass_matrix=True,           # Adapt during warmup
    regularize_mass_matrix=True       # Regularization for stability
)

# Access NUTS diagnostics
mcmc.run(rng_key, x, y)
extra_fields = mcmc.get_extra_fields()

print(f"Divergences: {extra_fields['diverging'].sum()}")
print(f"Tree depth: {extra_fields['tree_depth'].mean():.1f}")
print(f"Energy error: {extra_fields['energy_error'].mean():.3f}")
print(f"Step size: {extra_fields['mean_accept_prob'].mean():.3f}")
```

**Specialized kernels**:
- `HMC` - Manual control over step size and trajectory length
- `SA` - Slice sampler for constrained spaces
- `BarkerMH` - Robust alternative to HMC
- `MixedHMC` - Continuous + discrete parameters
- `HMCECS` - Subsampling for large datasets (N > 100K)
- `DiscreteHMCGibbs` - HMC/NUTS with Gibbs sampling for discrete parameters

**Consensus Monte Carlo for Large-Scale Distributed Inference**:

```python
from numpyro.infer.hmc_util import consensus
import jax.numpy as jnp

# For datasets too large for single-machine MCMC
# Split data across workers, combine posteriors

def run_distributed_mcmc(data_shards, model, num_workers=4):
    """
    Consensus Monte Carlo: Parallel MCMC on data shards,
    combine using weighted averaging.

    Better scalability than full-data MCMC for N > 1M observations.
    """
    subposterior_samples = []

    # Run MCMC on each data shard independently
    for worker_id, data_shard in enumerate(data_shards):
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)

        # Each worker gets different random seed
        mcmc.run(random.PRNGKey(worker_id), **data_shard)

        # Collect subposterior samples
        subposterior_samples.append(mcmc.get_samples())

    # Combine subposteriors using consensus
    # Weighted average based on shard sizes
    combined_posterior = consensus(
        subposterior_samples,
        num_draws=1000  # Final posterior sample size
    )

    return combined_posterior

# Example: 1M observations split across 10 workers
n_total = 1_000_000
n_workers = 10
shard_size = n_total // n_workers

data_shards = [
    {"x": x[i*shard_size:(i+1)*shard_size],
     "y": y[i*shard_size:(i+1)*shard_size]}
    for i in range(n_workers)
]

# Distributed inference
consensus_posterior = run_distributed_mcmc(data_shards, model, n_workers)

# Use consensus posterior for predictions
posterior_mean = {k: v.mean(axis=0) for k, v in consensus_posterior.items()}
```

**When to use Consensus Monte Carlo**:
- **Data size**: N > 1M observations (too large for single GPU)
- **Memory constraints**: Data doesn't fit in memory
- **Computational limits**: Single-machine MCMC takes > 24 hours
- **Distributed systems**: Data already sharded across machines

**Consensus vs alternatives**:
- **Better than**: Subsampling (HMCECS) when data can be naturally partitioned
- **Worse than**: Full-data MCMC for N < 100K (communication overhead)
- **Trade-off**: Slight bias (consensus approximation) vs massive speedup

For detailed MCMC diagnostics and troubleshooting, see `references/mcmc_diagnostics.md`.

### 2.5. ArviZ Integration for Visualization & Diagnostics

NumPyro seamlessly integrates with ArviZ for comprehensive Bayesian workflow visualization and diagnostics.

**Converting NumPyro results to ArviZ InferenceData**:

```python
import arviz as az
from numpyro.infer import NUTS, MCMC, Predictive
import jax.random as random

# After running MCMC
mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(random.PRNGKey(0), x_data, y_data)

# Convert to ArviZ InferenceData with metadata
idata = az.from_numpyro(
    mcmc,
    # Specify dimensions for multi-dimensional variables
    dims={
        "y": ["time"],           # y indexed by time dimension
        "theta": ["groups"]       # theta indexed by groups
    },
    # Add coordinates for interpretable labels
    coords={
        "time": np.arange(len(y_data)),
        "groups": group_names
    },
    # Include constant data for reference
    constant_data={
        "x": x_data,
        "treatment": treatment_indicator
    }
)

# InferenceData contains organized groups
print(idata.groups())  # ['posterior', 'sample_stats', 'constant_data']
```

**Comprehensive diagnostic visualization suite**:

```python
# 1. Convergence diagnostics
az.plot_trace(idata, var_names=['alpha', 'beta', 'sigma'])
az.plot_rank(idata)           # Rank plots (uniform if converged)
az.plot_ess(idata)            # Effective sample size by parameter
az.plot_mcse(idata)           # Monte Carlo standard error
az.plot_autocorr(idata)       # Autocorrelation by lag

# 2. Posterior analysis
az.plot_posterior(idata, var_names=['alpha', 'beta'])  # HDI + mean
az.plot_forest(idata, combined=True)                   # Forest plot
az.plot_density(idata, var_names=['alpha', 'beta'])    # Kernel density
az.plot_violin(idata)                                  # Violin plots
az.plot_pair(idata, var_names=['alpha', 'beta'],       # Scatter matrix
             divergences=True)                          # Highlight divergences

# 3. HMC/NUTS specific diagnostics
az.plot_energy(idata)         # Energy transition distribution
az.summary(idata)             # Comprehensive summary table

# 4. Posterior predictive checks
# First, generate posterior predictive samples
posterior_predictive = Predictive(model, mcmc.get_samples())
ppc_samples = posterior_predictive(random.PRNGKey(1), x_data, y=None)

# Add to InferenceData
idata.add_groups(posterior_predictive={"y": ppc_samples['obs']})

# Visualize
az.plot_ppc(idata, num_pp_samples=100)
az.plot_bpv(idata)            # Bayesian p-value plot
```

**Model comparison with ArviZ**:

```python
# Compare multiple models
idata_model1 = az.from_numpyro(mcmc1, ...)
idata_model2 = az.from_numpyro(mcmc2, ...)
idata_model3 = az.from_numpyro(mcmc3, ...)

# Compute LOO and WAIC
comparison = az.compare({
    "Model 1": idata_model1,
    "Model 2": idata_model2,
    "Model 3": idata_model3
})

print(comparison)  # Ranked by ELPD with standard errors

# Visualize comparison
az.plot_compare(comparison)
az.plot_elpd({"Model 1": idata_model1, "Model 2": idata_model2})
```

**ArviZ diagnostic functions**:

```python
# Numerical diagnostics
rhat_vals = az.rhat(idata)              # Convergence statistic
ess_vals = az.ess(idata)                # Effective sample size
mcse_vals = az.mcse(idata)              # MC standard error

# Check all parameters converged
assert (rhat_vals < 1.01).all(), "Some parameters not converged"
assert (ess_vals > 400).all(), "Low effective sample size"

# LOO cross-validation with diagnostic warnings
loo_result = az.loo(idata, pointwise=True)
print(f"LOO: {loo_result.loo:.2f} ± {loo_result.se:.2f}")

# Check for problematic observations
if loo_result.warning:
    print("LOO warning:", loo_result.warning)
    az.plot_khat(loo_result)  # Visualize Pareto-k diagnostic
```

**Advanced ArviZ workflows**:

```python
# Custom labeling for publication-quality plots
from arviz.labels import MapLabeller

labeller = MapLabeller(var_name_map={
    "alpha": r"$\alpha$ (Intercept)",
    "beta": r"$\beta$ (Slope)",
    "sigma": r"$\sigma$ (Residual SD)"
})

az.plot_forest(idata, labeller=labeller)

# Export to NetCDF for sharing/archiving
idata.to_netcdf("analysis_results.nc")

# Load later
idata_loaded = az.from_netcdf("analysis_results.nc")

# Summary statistics to DataFrame
summary_df = az.summary(idata, var_names=['alpha', 'beta'])
summary_df.to_csv("posterior_summary.csv")
```

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

**Trace plots and comprehensive visualization with ArviZ**:

```python
import arviz as az

# Convert NumPyro MCMC results to ArviZ InferenceData
idata = az.from_numpyro(
    mcmc,
    dims={"obs": ["data_points"]},
    coords={"data_points": np.arange(len(x))},
    constant_data={"x": x}
)

# Convergence diagnostics plots
az.plot_trace(idata, var_names=['alpha', 'beta'])  # Trace and marginal distributions
az.plot_rank(idata)                                # Rank plots for convergence
az.plot_ess(idata)                                 # Effective sample size
az.plot_autocorr(idata)                            # Autocorrelation

# Posterior visualization
az.plot_posterior(idata, var_names=['alpha', 'beta'])  # Posterior distributions
az.plot_forest(idata)                                  # Forest plot with credible intervals
az.plot_pair(idata, var_names=['alpha', 'beta'])       # Pair plots for correlations

# Model diagnostics
az.plot_energy(idata)                              # Energy plot for HMC/NUTS
az.plot_parallel(idata)                            # Parallel coordinates plot

# Posterior predictive checks
az.plot_ppc(idata, num_pp_samples=100)            # Posterior predictive check
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

Version: 1.0.2
Last Updated: 2025-10-31
Compatible with: NumPyro 0.15+, JAX 0.4+, ArviZ 0.20+
