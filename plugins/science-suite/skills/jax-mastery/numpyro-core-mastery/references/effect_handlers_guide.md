# Effect Handlers Guide

NumPyro effect handlers for model surgery and advanced inference patterns.

## Core Handlers

### seed (Control Randomness)
```python
from numpyro.handlers import seed

with seed(rng_seed=42):
    samples = model(x, y)  # Reproducible
```
**Use**: Debugging, reproducible sampling

### substitute (Fix Parameter Values)
```python
from numpyro.handlers import substitute

# Fix alpha=5, sample rest
with substitute(data={'alpha': 5.0}):
    trace = numpyro.handlers.trace(model).get_trace(x, y)
```
**Use**: Sensitivity analysis, conditional sampling

### condition (Observe Latent Variables)
```python
from numpyro.handlers import condition

# Condition on latent z
with condition(data={'z': z_observed}):
    posterior_pred = Predictive(model, posterior_samples)(rng_key, x)
```
**Use**: Counterfactual inference, what-if analysis

### trace (Inspect Execution)
```python
from numpyro.handlers import trace

trace_data = trace(model).get_trace(x, y)
for name, site in trace_data.items():
    if site['type'] == 'sample':
        print(f"{name}: {site['fn']}, shape: {site['value'].shape}")
```
**Use**: Debugging, understanding model structure

### reparam (Reparameterization)
```python
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

# Auto-reparameterize location-scale distributions
with reparam(config={'theta': LocScaleReparam()}):
    mcmc.run(rng_key, x, y)
```
**Use**: Fixing divergences, improving posterior geometry

## Reparameterization Strategies

### LocScaleReparam (Non-Centered)
Transforms `X ~ Normal(loc, scale)` to `X = loc + scale * Z, Z ~ Normal(0, 1)`

```python
# BEFORE (centered - may diverge)
theta = numpyro.sample('theta', dist.Normal(mu, sigma))

# AFTER (auto non-centered)
with reparam(config={'theta': LocScaleReparam()}):
    theta = numpyro.sample('theta', dist.Normal(mu, sigma))
```

### Manual Reparameterization
```python
# Non-centered hierarchical model
def noncentered():
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    # Sample standard normal
    z = numpyro.sample('z', dist.Normal(0, 1))

    # Transform (not sampled, computed)
    theta = numpyro.deterministic('theta', mu + sigma * z)
```

## Handler Composition

Handlers can be composed (nested):

```python
from numpyro.handlers import seed, substitute, trace

# Fix alpha, set seed, trace execution
with seed(rng_seed=42):
    with substitute(data={'alpha': 5.0}):
        trace_data = trace(model).get_trace(x, y)
```

## Advanced Patterns

### Predictive with Counterfactuals
```python
# What if we intervene on variable z?
z_counterfactual = jnp.array([...])

with condition(data={'z': z_counterfactual}):
    counterfactual_pred = Predictive(model, posterior_samples)(rng_key, x)

# Compare factual vs counterfactual
y_factual = posterior_pred['obs']
y_counterfactual = counterfactual_pred['obs']
treatment_effect = y_counterfactual.mean() - y_factual.mean()
```

### Model Surgery (Fix Some, Sample Others)
```python
# Fix problematic parameters, sample rest
fixed_params = {'sigma': 1.0, 'beta_0': 0.0}

with substitute(data=fixed_params):
    mcmc.run(rng_key, x, y)

# Only alpha, beta_1, ... are sampled
```

### Debugging Non-Convergence
```python
# Check which parameters cause issues
with trace(model) as tr:
    tr.get_trace(x, y)

for name, site in tr.items():
    if site['type'] == 'sample':
        val = site['value']
        if jnp.any(jnp.isnan(val)) or jnp.any(jnp.isinf(val)):
            print(f"NaN/Inf in {name}: {val}")
```

---

**Common use cases**:
- **Divergences** → `reparam` with `LocScaleReparam`
- **Reproducibility** → `seed`
- **Sensitivity** → `substitute` with different values
- **Counterfactuals** → `condition` on interventions
- **Debugging** → `trace` to inspect execution
