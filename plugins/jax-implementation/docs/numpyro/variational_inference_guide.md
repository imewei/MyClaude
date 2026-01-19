# Variational Inference Guide

Complete guide to Stochastic Variational Inference (SVI) in NumPyro.

## VI vs MCMC Decision Matrix

| Criterion | MCMC (NUTS) | VI (SVI) |
|-----------|-------------|----------|
| **Accuracy** | Gold standard, asymptotically exact | Approximate, may underestimate uncertainty |
| **Speed** | Slower (minutes to hours) | Faster (seconds to minutes) |
| **Scalability** | N < 100K | N > 100K |
| **Convergence** | R-hat, ESS diagnostics | ELBO convergence |
| **Use Case** | Research, high-stakes decisions | Production, large-scale, prototyping |

**Rule of thumb**:
- Use MCMC when you need accurate uncertainty quantification
- Use VI when you need speed or have very large datasets

## Basic SVI Workflow

```python
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import numpyro.optim as optim

# 1. Define model (same as for MCMC)
def model(x, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    mu = alpha + beta * x
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

# 2. Choose guide
guide = AutoNormal(model)

# 3. Choose optimizer
optimizer = optim.Adam(step_size=0.001)

# 4. Create SVI object
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# 5. Run optimization
svi_result = svi.run(jax.random.PRNGKey(0), num_steps=10000, x, y)

# 6. Extract parameters and sample
params = svi_result.params
posterior_samples = guide.sample_posterior(
    jax.random.PRNGKey(1), params, sample_shape=(2000,)
)
```

## Guide Selection

### AutoNormal (Fully Factorized Gaussian)

**Assumption**: All parameters independent, Gaussian marginals.

```python
guide = AutoNormal(model)
```

**Pros**:
- Fast optimization
- Few parameters to learn
- Works well for simple posteriors

**Cons**:
- Ignores posterior correlations
- Poor for strongly correlated parameters

**Use when**: Fast prototyping, simple models, parameters roughly independent.

### AutoMultivariateNormal (Correlated Gaussian)

**Assumption**: Joint Gaussian with full covariance.

```python
guide = AutoMultivariateNormal(model)
```

**Pros**:
- Captures posterior correlations
- More accurate than AutoNormal

**Cons**:
- Slower (O(p²) parameters for p dimensions)
- May overfit with many parameters

**Use when**: Parameters are correlated, accuracy matters, p < 100.

### AutoLowRankMultivariateNormal (Low-Rank + Diagonal)

**Assumption**: Covariance = low-rank + diagonal.

```python
guide = AutoLowRankMultivariateNormal(model, rank=10)
```

**Pros**:
- Balanced speed/accuracy
- Captures main correlations
- Scalable to moderate p

**Cons**:
- Need to choose rank

**Use when**: Large p, want correlations, need efficiency.

### AutoDelta (Point Estimate / MAP)

**Assumption**: Posterior is point mass (no uncertainty).

```python
guide = AutoDelta(model)
```

**Pros**:
- Fastest (no sampling)
- Maximum A Posteriori (MAP) estimate

**Cons**:
- No uncertainty quantification
- Not Bayesian inference

**Use when**: Only need point estimates, not inference.

### Custom Guide

For specialized variational families:

```python
def custom_guide(x, y=None):
    # Variational parameters (optimized by SVI)
    alpha_loc = numpyro.param('alpha_loc', 0.0)
    alpha_scale = numpyro.param('alpha_scale', 1.0,
                                constraint=constraints.positive)

    beta_loc = numpyro.param('beta_loc', 0.0)
    beta_scale = numpyro.param('beta_scale', 1.0,
                              constraint=constraints.positive)

    # Variational distributions
    numpyro.sample('alpha', dist.Normal(alpha_loc, alpha_scale))
    numpyro.sample('beta', dist.Normal(beta_loc, beta_scale))
```

## ELBO Objectives

### Trace_ELBO (Standard)

```python
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
```

**Use**: Default choice, works for most models.

### TraceMeanField_ELBO (Optimized for Mean-Field)

```python
svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
```

**Use**: When using AutoNormal (mean-field guide).

### RenyiELBO (Rényi Divergence)

```python
svi = SVI(model, guide, optimizer, loss=RenyiELBO(alpha=0.5))
```

**Parameters**:
- `alpha=0`: Maximum likelihood (mode-seeking)
- `alpha=1`: Standard KL (default ELBO)
- `alpha=∞`: Zero-forcing

**Use**: Experimenting with different divergences.

## Optimizers

### Adam (Default)

```python
optimizer = optim.Adam(step_size=0.001)
```

**Best for**: Most problems, default choice.

### ClippedAdam (Gradient Clipping)

```python
optimizer = optim.ClippedAdam(step_size=0.001, clip_norm=10.0)
```

**Use when**: Gradients explode, numerical instability.

### Learning Rate Schedules

**Exponential decay**:
```python
schedule = optim.exponential_decay(
    init_step_size=0.01,
    decay_rate=0.1,
    decay_steps=1000
)
optimizer = optim.Adam(step_size=schedule)
```

**Polynomial decay**:
```python
schedule = optim.polynomial_decay(
    init_step_size=0.01,
    final_step_size=0.0001,
    power=1.0,
    decay_steps=5000
)
optimizer = optim.Adam(step_size=schedule)
```

## Convergence Monitoring

**Plot ELBO**:

```python
import matplotlib.pyplot as plt

losses = svi_result.losses

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('ELBO Loss')
plt.title('Convergence')
plt.show()
```

**What to look for**:
- Smooth decrease (good)
- Plateaus → converged
- Oscillations → reduce learning rate
- Increasing → numerical issues

**Early stopping**:

```python
svi_state = svi.init(rng_key, x, y)
losses = []

for step in range(10000):
    svi_state, loss = svi.update(svi_state, x, y)

    if step % 100 == 0:
        losses.append(loss)

    # Check convergence
    if step > 1000 and len(losses) > 10:
        recent_change = abs(losses[-1] - losses[-10])
        if recent_change < 1e-4:
            print(f"Converged at step {step}")
            break

params = svi.get_params(svi_state)
```

## Checking VI Approximation Quality

### 1. Compare to MCMC (Gold Standard)

```python
# Run both
svi_samples = guide.sample_posterior(rng_key, params, (2000,))
mcmc_samples = mcmc.get_samples()

# Compare marginals
import arviz as az

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

az.plot_kde(svi_samples['alpha'], ax=axes[0], label='VI')
az.plot_kde(mcmc_samples['alpha'], ax=axes[0], label='MCMC')
axes[0].legend()
axes[0].set_title('Alpha')

az.plot_kde(svi_samples['beta'], ax=axes[1], label='VI')
az.plot_kde(mcmc_samples['beta'], ax=axes[1], label='MCMC')
axes[1].legend()
axes[1].set_title('Beta')
```

### 2. Posterior Predictive Checks

```python
from numpyro.infer import Predictive

predictive = Predictive(model, posterior_samples=svi_samples)
ppc = predictive(rng_key, x, y=None)

plt.hist(ppc['obs'].flatten(), bins=50, alpha=0.5, label='VI Posterior Predictive')
plt.hist(y, bins=50, alpha=0.5, label='Observed')
plt.legend()
```

### 3. Check ELBO on Held-Out Data

```python
# Split data
x_train, x_test = x[:800], x[800:]
y_train, y_test = y[:800], y[800:]

# Train on train set
svi_result = svi.run(rng_key, 10000, x_train, y_train)

# Evaluate on test set
test_loss = svi.evaluate(svi.get_params(svi_result.state), x_test, y_test)
print(f"Test ELBO: {test_loss:.2f}")
```

## Common Issues

### Issue 1: ELBO Not Decreasing

**Symptoms**: Loss stays constant or increases.

**Solutions**:
1. Reduce learning rate: `step_size=0.0001`
2. Check model specification (NaN/Inf?)
3. Try different initialization
4. Use gradient clipping: `ClippedAdam`

### Issue 2: Posterior Variance Too Small

**Symptoms**: VI underestimates uncertainty compared to MCMC.

**Solutions**:
1. Use `AutoMultivariateNormal` instead of `AutoNormal`
2. Increase `num_steps` (may not have converged)
3. This is inherent to VI (mode-seeking), consider MCMC

### Issue 3: Slow Convergence

**Symptoms**: Takes 100K+ iterations.

**Solutions**:
1. Increase learning rate: `step_size=0.01`
2. Use learning rate schedule
3. Better initialization
4. Simplify model

### Issue 4: Multimodal Posterior

**Symptoms**: VI finds one mode, misses others.

**Solutions**:
- VI is mode-seeking, will converge to one mode
- Run from multiple initializations
- Use MCMC for multimodal posteriors

## Advanced: Amortized Inference

For inference on many datasets with shared structure:

```python
def amortized_guide(x, y=None):
    # Neural network maps data to variational parameters
    hidden = numpyro.module('encoder_hidden', nn.Dense(50), (x,))
    hidden = jax.nn.relu(hidden)

    alpha_loc = numpyro.module('alpha_loc', nn.Dense(1), (hidden,))
    alpha_scale = jax.nn.softplus(
        numpyro.module('alpha_scale', nn.Dense(1), (hidden,))
    )

    numpyro.sample('alpha', dist.Normal(alpha_loc, alpha_scale))
```

**Use when**: Many similar inference problems (e.g., per-user models).

## Summary

**Quick Decision Tree**:

1. **Need accurate uncertainty?**
   - Yes → MCMC
   - No → Continue

2. **Dataset size?**
   - N < 10K → MCMC (fast enough)
   - N > 100K → VI
   - 10K < N < 100K → Try both

3. **Correlated parameters?**
   - Yes → `AutoMultivariateNormal` or MCMC
   - No → `AutoNormal`

4. **Only need point estimate?**
   - Yes → `AutoDelta`
   - No → Use probabilistic guide

**Best Practices**:
- Always compare VI to MCMC on subset
- Check posterior predictive
- Monitor ELBO convergence
- Use learning rate schedules for stability
- VI for exploration, MCMC for final analysis

---

**Use `scripts/benchmark_mcmc_vi.py` to systematically compare methods.**
