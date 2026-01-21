# NumPyro vs Blackjax: Mastering Both Ecosystems

This reference covers the two dominant JAX-based probabilistic programming approaches and when to use each.

## NumPyro: The High-Level Architect

NumPyro provides a Pyro-style API with effect handlers for model manipulation.

### Effect Handlers Deep Dive

Effect handlers allow intervention in model execution without modifying source code.

#### handlers.trace: Inspect Execution

```python
from numpyro import handlers
import numpyro.distributions as dist

def model(data):
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    with numpyro.plate('data', len(data)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=data)

# Get execution trace
traced = handlers.trace(handlers.seed(model, rng_seed=0))
trace = traced.get_trace(data)

# Inspect sites
for name, site in trace.items():
    print(f"{name}: {site['type']}, value={site['value']}, log_prob={site.get('fn').log_prob(site['value']) if site['type'] == 'sample' else 'N/A'}")
```

#### handlers.condition: Fix Values

```python
# Fix mu to specific value
conditioned_model = handlers.condition(model, {'mu': 2.5})

# Run with fixed mu - only sigma is sampled
trace = handlers.trace(handlers.seed(conditioned_model, rng_seed=0)).get_trace(data)
print(trace['mu']['value'])  # Always 2.5
```

#### handlers.substitute: Replace Samples

```python
# Substitute specific values (for predictive sampling)
substituted = handlers.substitute(model, {'mu': posterior_mu, 'sigma': posterior_sigma})

# Useful for posterior predictive checks
with handlers.seed(rng_seed=42):
    pred_trace = handlers.trace(substituted).get_trace(None)
```

#### handlers.block: Hide Sites

```python
# Hide certain sample sites from outer handlers
def outer_model():
    x = numpyro.sample('x', dist.Normal(0, 1))

    # Inner sites won't appear in outer trace
    with handlers.block():
        z = numpyro.sample('z', dist.Normal(x, 1))

    return z
```

#### handlers.scale: Reweight Log-Prob

```python
# Scale log-probability (for importance weighting, data subsampling)
def subsampled_model(data, subsample_size):
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    # Subsample data
    idx = numpyro.sample('idx', dist.Categorical(jnp.ones(len(data)) / len(data)),
                          sample_shape=(subsample_size,))

    # Scale likelihood to account for subsampling
    scale = len(data) / subsample_size
    with handlers.scale(scale=scale):
        with numpyro.plate('data', subsample_size):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=data[idx])
```

### NumPyro Model Patterns

#### Hierarchical Models

```python
def hierarchical_model(group_idx, observations):
    n_groups = len(jnp.unique(group_idx))

    # Hyperpriors
    mu_0 = numpyro.sample('mu_0', dist.Normal(0, 10))
    sigma_0 = numpyro.sample('sigma_0', dist.HalfNormal(5))

    # Group-level parameters
    with numpyro.plate('groups', n_groups):
        mu_group = numpyro.sample('mu_group', dist.Normal(mu_0, sigma_0))

    # Observation noise
    sigma = numpyro.sample('sigma', dist.HalfNormal(2))

    # Likelihood
    with numpyro.plate('data', len(observations)):
        numpyro.sample('obs', dist.Normal(mu_group[group_idx], sigma), obs=observations)
```

#### Non-Centered Parameterization

```python
def funnel_model_centered(n_samples):
    """Centered - struggles with funnel geometry."""
    tau = numpyro.sample('tau', dist.HalfNormal(3))
    with numpyro.plate('samples', n_samples):
        x = numpyro.sample('x', dist.Normal(0, tau))  # tau in scale!
    return x

def funnel_model_noncentered(n_samples):
    """Non-centered - handles funnel geometry well."""
    tau = numpyro.sample('tau', dist.HalfNormal(3))
    with numpyro.plate('samples', n_samples):
        x_raw = numpyro.sample('x_raw', dist.Normal(0, 1))  # Standard normal
        x = numpyro.deterministic('x', x_raw * tau)  # Transform
    return x
```

## Blackjax: The Low-Level Surgeon

Blackjax provides composable inference algorithms as pure functions.

### Building Custom Inference Loops

#### Basic NUTS with Warmup

```python
import blackjax
import jax

def run_nuts(log_prob_fn, initial_position, num_warmup, num_samples, rng_key):
    """Full NUTS inference with adaptation."""
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

    # Warmup: adapt step size and mass matrix
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
    (state, params), warmup_info = warmup.run(
        warmup_key,
        initial_position,
        num_steps=num_warmup
    )

    print(f"Adapted step_size: {params['step_size']:.4f}")
    print(f"Adapted inverse_mass_matrix shape: {params['inverse_mass_matrix'].shape}")

    # Sampling
    nuts_kernel = blackjax.nuts(log_prob_fn, **params).step

    def step_fn(state, key):
        state, info = nuts_kernel(key, state)
        return state, (state.position, info)

    keys = jax.random.split(sample_key, num_samples)
    final_state, (samples, infos) = jax.lax.scan(step_fn, state, keys)

    return samples, infos, params
```

#### Custom Kernel Composition

```python
def mixed_kernel(log_prob_fn, hmc_params, gibbs_update_fn):
    """Mix HMC with custom Gibbs steps."""
    hmc_kernel = blackjax.hmc(log_prob_fn, **hmc_params).step

    def combined_step(state, key):
        key1, key2 = jax.random.split(key)

        # HMC step for continuous parameters
        state, hmc_info = hmc_kernel(key1, state)

        # Custom Gibbs step for discrete/special parameters
        new_position = gibbs_update_fn(key2, state.position)
        state = state._replace(position=new_position)

        return state, hmc_info

    return combined_step
```

#### Parallel Chains with pmap

```python
def parallel_chains(log_prob_fn, initial_positions, num_samples, rng_keys):
    """Run multiple chains in parallel across devices."""

    @jax.pmap
    def run_single_chain(init_pos, rng_key):
        warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
        (state, params), _ = warmup.run(rng_key, init_pos, num_steps=500)

        kernel = blackjax.nuts(log_prob_fn, **params).step
        keys = jax.random.split(rng_key, num_samples)

        def step(state, key):
            state, info = kernel(key, state)
            return state, state.position

        _, samples = jax.lax.scan(step, state, keys)
        return samples

    # Run on all devices
    all_samples = run_single_chain(initial_positions, rng_keys)
    return all_samples  # Shape: (n_devices, num_samples, param_dim)
```

### Available Kernels

| Kernel | Use Case | Key Parameters |
|--------|----------|----------------|
| `blackjax.nuts` | General-purpose, adaptive | `step_size`, `inverse_mass_matrix` |
| `blackjax.hmc` | Fixed trajectory length | `step_size`, `inverse_mass_matrix`, `num_integration_steps` |
| `blackjax.mala` | Metropolis-adjusted Langevin | `step_size` |
| `blackjax.rmh` | Random-walk Metropolis | `sigma` |
| `blackjax.elliptical_slice` | Gaussian priors | None (uses prior) |
| `blackjax.sgld` | Stochastic gradient | `step_size`, `temperature` |

### Diagnostics

```python
def diagnose_samples(samples, infos):
    """Comprehensive MCMC diagnostics."""
    from blackjax.diagnostics import effective_sample_size, potential_scale_reduction

    # Reshape for diagnostics: (n_chains, n_samples, n_dims)
    n_chains = 4
    samples_reshaped = samples.reshape(n_chains, -1, samples.shape[-1])

    # Effective sample size
    ess = effective_sample_size(samples_reshaped)
    print(f"ESS: {ess}")

    # R-hat (potential scale reduction)
    rhat = potential_scale_reduction(samples_reshaped)
    print(f"R-hat: {rhat}")

    # Divergences (from NUTS info)
    if hasattr(infos, 'is_divergent'):
        n_divergent = jnp.sum(infos.is_divergent)
        print(f"Divergences: {n_divergent} / {len(infos.is_divergent)}")

    # Acceptance rate
    if hasattr(infos, 'acceptance_rate'):
        print(f"Mean acceptance rate: {jnp.mean(infos.acceptance_rate):.3f}")
```

## NumPyro ↔ Blackjax Interop

### Extract Log-Prob from NumPyro Model

```python
from numpyro.infer.util import log_density
from numpyro import handlers

def numpyro_to_blackjax(model, data):
    """Convert NumPyro model to Blackjax-compatible log_prob."""

    def log_prob_fn(params):
        # Compute log-density given parameters
        log_p, _ = log_density(model, (data,), {}, params)
        return log_p

    return log_prob_fn

# Usage
model_fn = lambda data: my_numpyro_model(data)
log_prob = numpyro_to_blackjax(model_fn, observed_data)

# Now use with Blackjax
samples, _ = run_nuts(log_prob, initial_params, 500, 1000, rng_key)
```

### Use Blackjax Samples in NumPyro

```python
from numpyro.infer import Predictive

def posterior_predictive_from_blackjax(model, blackjax_samples, data, rng_key):
    """Generate predictive samples using Blackjax posterior."""
    # Convert samples dict to format expected by Predictive
    posterior_samples = {
        'mu': blackjax_samples['mu'],
        'sigma': blackjax_samples['sigma'],
    }

    predictive = Predictive(model, posterior_samples)
    return predictive(rng_key, data)
```

## Choosing Between NumPyro and Blackjax

| Scenario | Recommendation |
|----------|----------------|
| Standard hierarchical model | NumPyro |
| Rapid prototyping | NumPyro |
| Custom MCMC algorithm | Blackjax |
| Mixed discrete/continuous | Blackjax (custom Gibbs) |
| Maximum performance | Blackjax |
| Teaching/interpretability | NumPyro |
| Research on new samplers | Blackjax |
| Production deployment | Either (depends on model) |
