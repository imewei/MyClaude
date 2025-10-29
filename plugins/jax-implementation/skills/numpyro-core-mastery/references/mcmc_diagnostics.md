# MCMC Convergence Diagnostics

Comprehensive guide to diagnosing and troubleshooting MCMC inference in NumPyro.

## Quick Convergence Checklist

Before trusting MCMC results, verify:

1. **R-hat < 1.01** for all parameters (between-chain vs within-chain variance)
2. **ESS > 400** per chain (effective sample size accounts for autocorrelation)
3. **Zero divergences** (numerical stability in HMC)
4. **Trace plots show good mixing** (chains explore same region, no trends)
5. **No warnings from NumPyro** (max tree depth, initialization failures)

```python
from numpyro.diagnostics import summary

summary_dict = summary(posterior_samples, prob=0.95)

for param, stats in summary_dict.items():
    print(f"{param}:")
    print(f"  R-hat: {stats['r_hat']:.4f} {'✓' if stats['r_hat'] < 1.01 else '✗ FAIL'}")
    print(f"  ESS: {stats['n_eff']:.0f} {'✓' if stats['n_eff'] > 400 else '✗ LOW'}")

# Check divergences
divergences = mcmc.get_extra_fields()['diverging'].sum()
print(f"Divergences: {divergences} {'✓' if divergences == 0 else '✗ FAIL'}")
```

## R-hat (Gelman-Rubin Statistic)

**What it measures**: Convergence by comparing between-chain variance to within-chain variance.

**Interpretation**:
- **R-hat < 1.01**: Converged ✓ - Chains have mixed well
- **1.01 < R-hat < 1.1**: Marginal - Run longer or investigate
- **R-hat > 1.1**: Not converged ✗ - Serious issue, don't trust results

**Formul a**: R-hat ≈ sqrt(Var_between / Var_within)

**Common causes of high R-hat**:
1. **Insufficient warmup**: Chains haven't reached stationary distribution
2. **Multimodal posterior**: Different chains stuck in different modes
3. **Poor initialization**: Chains started too far from typical set
4. **Model misspecification**: Posterior doesn't exist or is improper

**Solutions**:
```python
# 1. Increase warmup
mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000)  # Was 1000

# 2. Run more chains
mcmc = MCMC(kernel, num_chains=8)  # Was 4

# 3. Better initialization
from numpyro.infer import init_to_median, init_to_sample
kernel = NUTS(model, init_strategy=init_to_median())

# 4. Check trace plots for multimodality
import arviz as az
az.plot_trace(az.from_numpyro(mcmc))
```

## Effective Sample Size (ESS)

**What it measures**: Number of independent samples accounting for autocorrelation.

**Interpretation**:
- **ESS > 400 per chain**: Good ✓ - Low autocorrelation
- **100 < ESS < 400**: Acceptable but suboptimal
- **ESS < 100**: Poor ✗ - Very high autocorrelation

**Why it matters**: If ESS = 100 from 2000 samples, only 100 are effectively independent. Credible intervals computed from 2000 samples are as reliable as those from 100 independent samples.

**Common causes of low ESS**:
1. **High posterior correlation**: Parameters highly correlated
2. **Stiff ODE or complex model**: Gradient evaluation expensive
3. **Poor posterior geometry**: Narrow valleys, high curvature
4. **Insufficient adaptation**: Step size or mass matrix not optimal

**Solutions**:
```python
# 1. Run more samples
mcmc = MCMC(kernel, num_samples=5000)  # Was 2000

# 2. Thin samples (trade storage for independence)
mcmc = MCMC(kernel, thinning=2)  # Keep every 2nd sample

# 3. Reparameterize (non-centered for hierarchical models)
def noncentered_model():
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    z = numpyro.sample('z', dist.Normal(0, 1))  # Standard normal
    theta = mu + sigma * z  # Transform

# 4. Check for strong correlations
import matplotlib.pyplot as plt
az.plot_pair(az.from_numpyro(mcmc), var_names=['alpha', 'beta'])
```

## Divergences

**What they are**: Numerical instability in HMC trajectory integration.

**Interpretation**:
- **0 divergences**: Excellent ✓
- **< 1% of samples**: Acceptable for exploratory analysis
- **> 1% of samples**: Serious issue ✗ - Results biased

**Why they matter**: Divergences indicate sampler unable to accurately explore regions of high posterior curvature. Samples may be biased away from these regions.

**Common causes**:
1. **Step size too large**: Integration error accumulates
2. **Posterior has funnels or high curvature**: Challenging geometry
3. **Centered parameterization in hierarchical models**: Creates funnel
4. **Weak priors**: Posterior concentrates in narrow region

**Solutions (in order of preference)**:

### Solution 1: Increase `target_accept_prob`

```python
# Default: 0.8, try 0.9 or 0.95
nuts_kernel = NUTS(model, target_accept_prob=0.95)
```

**Effect**: Reduces step size, more careful exploration.
**Trade-off**: Slower sampling (more gradient evaluations per sample).

### Solution 2: Reparameterize (Non-Centered Parameterization)

**Centered (causes divergences)**:
```python
def centered_hierarchical():
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    with numpyro.plate('groups', n_groups):
        theta = numpyro.sample('theta', dist.Normal(mu, sigma))
```

**Non-Centered (fixes divergences)**:
```python
def noncentered_hierarchical():
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    with numpyro.plate('groups', n_groups):
        theta_raw = numpyro.sample('theta_raw', dist.Normal(0, 1))

    theta = mu + sigma * theta_raw  # Manual transformation
```

**Why it works**: Decorrelates mu/sigma from theta, eliminates funnel geometry.

### Solution 3: More Informative Priors

**Weak prior (may cause issues)**:
```python
sigma = numpyro.sample('sigma', dist.HalfNormal(100))  # Too vague
```

**Regularizing prior (helps)**:
```python
sigma = numpyro.sample('sigma', dist.HalfNormal(5))  # Constrains reasonable range
```

### Solution 4: Use Reparameterization Effect Handler

```python
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

# Automatically reparameterize location-scale distributions
with reparam(config={'theta': LocScaleReparam()}):
    mcmc.run(rng_key, x, y)
```

## Trace Plots

**Visual convergence check**: Plot parameter values vs iteration for each chain.

```python
import arviz as az
import matplotlib.pyplot as plt

idata = az.from_numpyro(mcmc)
az.plot_trace(idata, var_names=['alpha', 'beta', 'sigma'])
plt.tight_layout()
plt.show()
```

**What to look for**:

**Good trace (converged)**:
- All chains overlap (same region)
- No trends or drifts
- "Hairy caterpillar" appearance
- Rapid mixing

**Bad trace (not converged)**:
- Chains explore different regions (multimodality or non-convergence)
- Trends or drifts (not stationary)
- One chain stuck (initialization issue)
- Slow mixing (high autocorrelation)

**Example diagnoses**:

**Multimodal posterior**:
```
Chain 1: ――――――― (oscillates around -5)
Chain 2: ―――――― (oscillates around +5)
```
**Solution**: Check if multiple modes are real or model misspecification.

**Non-stationarity**:
```
Chain 1: /‾‾‾‾ (upward drift)
Chain 2: /‾‾‾‾ (upward drift)
```
**Solution**: Increase warmup, check model specification.

**Stuck chain**:
```
Chain 1: ∿∿∿∿ (mixing)
Chain 2: ――― (flat line)
Chain 3: ∿∿∿∿ (mixing)
```
**Solution**: Better initialization, check for numerical issues.

## Advanced Diagnostics

### Rank Plots (Uniform if converged)

```python
az.plot_rank(idata, var_names=['alpha', 'beta'])
```

If chains have converged, ranks should be uniformly distributed across chains.

### Autocorrelation Plots

```python
az.plot_autocorr(idata, var_names=['alpha', 'beta'])
```

Autocorrelation should decay quickly. Slow decay = high correlation = low ESS.

### Pair Plots (Check correlations)

```python
az.plot_pair(idata, var_names=['alpha', 'beta', 'sigma'],
             divergences=True, kind='hexbin')
```

Divergences often cluster in regions of high curvature.

## Troubleshooting Workflow

**Step 1**: Check R-hat and ESS
```python
summary(posterior_samples)
```

**Step 2**: If R-hat > 1.01 or ESS < 400:
- Plot traces
- Check for multimodality, non-stationarity, or stuck chains
- Increase warmup or run longer

**Step 3**: Check divergences
```python
mcmc.get_extra_fields()['diverging'].sum()
```

**Step 4**: If divergences > 0:
- Increase `target_accept_prob` to 0.9-0.95
- Reparameterize (non-centered for hierarchical)
- Use more informative priors

**Step 5**: Verify posterior makes sense
- Prior predictive check
- Posterior predictive check
- Parameter estimates reasonable?

## Common Failure Modes

### "Maximum tree depth reached"

**Meaning**: NUTS trajectory hit max length (2^max_tree_depth steps).

**Causes**:
- Heavy-tailed posterior
- Very large step sizes
- Pathological geometry

**Solutions**:
```python
# Increase max tree depth
nuts_kernel = NUTS(model, max_tree_depth=12)  # Was 10

# Or reduce step size
nuts_kernel = NUTS(model, target_accept_prob=0.9)
```

### "Initial particles out of bounds"

**Meaning**: Initialization sampled invalid parameter values.

**Solutions**:
```python
# Better initialization
nuts_kernel = NUTS(model, init_strategy=init_to_median())

# Or tighter priors to avoid extreme values
```

### "NaN or Inf in gradients"

**Meaning**: Numerical overflow in likelihood or gradient computation.

**Solutions**:
- Use log-scale for positive parameters
- Add numerical stability (jnp.clip, jnp.log1p)
- Check for division by zero

## Performance Monitoring

**Time per sample**:
```python
import time

start = time.time()
mcmc.run(rng_key, x, y)
elapsed = time.time() - start

samples_per_sec = (mcmc.num_samples * mcmc.num_chains) / elapsed
print(f"Sampling rate: {samples_per_sec:.1f} samples/sec")
```

**Gradient evaluations per sample**:
```python
# Lower is better (more efficient)
n_grads = mcmc.get_extra_fields()['num_steps'].mean()
print(f"Average gradient evaluations per sample: {n_grads:.1f}")
```

## When to Trust Results

✓ **Safe to proceed**:
- R-hat < 1.01 for all parameters
- ESS > 400 per chain
- Zero divergences
- Trace plots show good mixing
- Posterior predictive checks look reasonable

⚠ **Proceed with caution**:
- R-hat slightly above 1.01 (1.01-1.05)
- ESS 100-400 (usable but less reliable)
- < 1% divergences

✗ **Do not trust**:
- R-hat > 1.1
- ESS < 100
- > 1% divergences
- Chains explore different modes
- Any "stuck" chains

---

**Use `scripts/mcmc_diagnostics.py` for automated checking of all these criteria.**
