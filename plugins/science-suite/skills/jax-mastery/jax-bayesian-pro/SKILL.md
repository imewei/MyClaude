---
name: jax-bayesian-pro
version: "1.1.0"
description: This skill should be used when the user asks to "write Bayesian models in JAX", "use NumPyro or Blackjax", "implement custom MCMC samplers", "tune HMC mass matrix", "debug NUTS divergences", "run parallel MCMC chains", "integrate Diffrax with Bayesian inference", "build neural surrogate models for SBI", "implement effect handlers", "write pure log-prob functions", or needs expert guidance on probabilistic programming, simulation-based inference, or Bayesian parameter estimation for physics/soft matter.
---

# JAX Bayesian Pro: Inference as Transformation

A JAX-first Bayesian expert combines probabilistic intuition with compiler-aware engineering. Unlike traditional "PyMC/Stan" workflows—write model, hit "Magic Inference Button"—JAX Bayesian engineering treats inference as a **composable program transformation**.

## Expert Agent

For complex Bayesian modeling, hierarchical inference, and probabilistic programming tasks, delegate to the expert agent:

- **`jax-pro`**: Unified specialist for Bayesian inference (NumPyro), MCMC diagnostics, and differentiable physics integration.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Hierarchical models, NUTS/HMC tuning, SVI, and convergence diagnostics (R-hat, ESS).

## The Inference-as-Transformation Mindset

### Decoupled Architecture

View the *Model* (log-density) and *Inference Kernel* (NUTS/HMC step) as separate, pure functions. Define log-probability in vanilla JAX, pass it into a Blackjax sampler.

```python
import jax.numpy as jnp
from jax.scipy import stats

def log_prob(params, data):
    """Pure log-probability function - no magic contexts."""
    mu, log_sigma = params['mu'], params['log_sigma']
    sigma = jnp.exp(log_sigma)

    # Prior
    log_prior = stats.norm.logpdf(mu, 0, 10) + stats.norm.logpdf(log_sigma, 0, 2)

    # Likelihood
    log_lik = jnp.sum(stats.norm.logpdf(data, mu, sigma))

    return log_prior + log_lik
```

### Differentiable Physics

The simulation is part of the computation graph. Backpropagate *through* the simulation (ODE solver) to inform priors or likelihoods.

```python
import diffrax

def physics_log_prob(params, observations, ts):
    """Log-prob with differentiable ODE solver."""
    # Solve ODE with current parameters
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: ode_fn(y, params)),
        diffrax.Tsit5(), t0=0, t1=ts[-1], dt0=0.01, y0=y0,
        saveat=diffrax.SaveAt(ts=ts)
    )
    predicted = solution.ys

    # Compare to observations
    log_lik = jnp.sum(stats.norm.logpdf(observations, predicted, params['sigma']))
    log_prior = compute_prior(params)

    return log_prior + log_lik
```

### Generative Transparency

Explicitly manage PRNG keys. Split and fold for every stochastic event to ensure determinism.

```python
def sample_predictive(rng_key, params, n_samples):
    """Explicit RNG management for predictive samples."""
    keys = jax.random.split(rng_key, n_samples)

    def sample_one(key, p):
        return p['mu'] + jax.random.normal(key) * jnp.exp(p['log_sigma'])

    return jax.vmap(sample_one)(keys, params)
```

## Ecosystem Duel: NumPyro vs Blackjax

| Aspect | NumPyro | Blackjax |
|--------|---------|----------|
| **Use case** | Rapid prototyping, standard models | Custom algorithms, low-level control |
| **Abstraction** | High (effect handlers) | Low (pure functions) |
| **Flexibility** | Model syntax constraints | Full freedom |
| **Best for** | Hierarchical models, PPL features | Custom kernels, research |

### NumPyro: High-Level Architect

Master effect handlers to intervene in model execution without changing source code.

```python
import numpyro
from numpyro import handlers

def model(data):
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    with numpyro.plate('data', len(data)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=data)

# Trace: inspect all sample sites
trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(data)

# Condition: fix specific values
conditioned = handlers.condition(model, {'mu': 2.5})

# Substitute: replace distributions
substituted = handlers.substitute(model, {'mu': 0.0, 'sigma': 1.0})
```

### Blackjax: Low-Level Surgeon

Build custom inference loops with `jax.lax.scan`. Control the transition kernel directly.

```python
import blackjax

def run_custom_hmc(log_prob, initial_position, num_samples, rng_key):
    """Custom HMC loop with full control."""
    # Initialize
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    rng_key, warmup_key = jax.random.split(rng_key)
    (state, params), _ = warmup.run(warmup_key, initial_position, num_steps=500)

    # Build kernel
    kernel = blackjax.nuts(log_prob, **params).step

    # Run with scan
    def step(state, key):
        state, info = kernel(key, state)
        return state, (state.position, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (samples, infos) = jax.lax.scan(step, state, keys)

    return samples, infos
```

## Skills Matrix: Junior vs Expert

| Category | Junior | Expert/Pro |
|----------|--------|------------|
| **Model Definition** | Monolithic function with hidden global state | Pure `log_prob(params, data)` function |
| **Sampling** | Calls `mcmc.run()` and waits | Writes `jax.lax.scan` loop with Blackjax |
| **Physics Integration** | Treats simulation as external data | Differentiates through simulation (Diffrax) |
| **Geometry** | Struggles with "funnel" geometries | Uses Non-Centered Reparameterization |
| **Parallelism** | Runs 1 chain per GPU | Uses `jax.pmap` for massively parallel chains |

## Log-Prob Engineering

### Vectorization over Data

Ensure log-likelihood is `vmap`-able over data batches:

```python
def single_particle_log_lik(params, particle_data):
    """Log-likelihood for single particle."""
    return stats.norm.logpdf(particle_data, params['mu'], params['sigma'])

# Vectorize over all particles
batch_log_lik = jax.vmap(single_particle_log_lik, in_axes=(None, 0))
total_log_lik = batch_log_lik(params, all_particles).sum()
```

### Masking for Ragged Data

Handle varying particle counts without breaking JIT:

```python
def masked_log_lik(params, padded_data, mask):
    """Log-likelihood with masking for ragged arrays."""
    log_probs = stats.norm.logpdf(padded_data, params['mu'], params['sigma'])
    return jnp.where(mask, log_probs, 0.0).sum()
```

## Additional Resources

### Reference Files

- **`references/numpyro-blackjax.md`** - Effect handlers, custom kernels, NumPyro-Blackjax interop, model surgery techniques
- **`references/advanced-inference.md`** - HMC/NUTS internals, mass matrix tuning, divergence diagnosis, reparameterization, simulation-based inference, Diffrax integration

### Example Files

- **`examples/pure-log-prob-models.py`** - Pure functional Bayesian models without PPL syntax
- **`examples/blackjax-custom-sampler.py`** - Custom MCMC loops with Blackjax
- **`examples/diffrax-bayesian-ode.py`** - Bayesian parameter estimation with differentiable ODE solvers

**Outcome**: Write pure log-prob functions, master effect handlers, build custom samplers, integrate differentiable physics, run massively parallel chains.
