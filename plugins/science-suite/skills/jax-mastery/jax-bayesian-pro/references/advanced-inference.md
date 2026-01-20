# Advanced Inference Skills

This reference covers HMC/NUTS internals, mass matrix tuning, divergence diagnosis, reparameterization, and simulation-based inference with Diffrax.

## HMC & NUTS Internals

### The Leapfrog Integrator

HMC simulates Hamiltonian dynamics using the leapfrog (Störmer-Verlet) integrator:

```python
def leapfrog_step(q, p, grad_log_prob, step_size):
    """Single leapfrog step."""
    # Half-step momentum
    p = p + 0.5 * step_size * grad_log_prob(q)

    # Full-step position
    q = q + step_size * p

    # Half-step momentum
    p = p + 0.5 * step_size * grad_log_prob(q)

    return q, p

def hmc_trajectory(q0, p0, grad_log_prob, step_size, num_steps):
    """Complete HMC trajectory."""
    q, p = q0, p0
    for _ in range(num_steps):
        q, p = leapfrog_step(q, p, grad_log_prob, step_size)
    return q, p
```

### Energy Conservation and Acceptance

```python
def hamiltonian(q, p, log_prob, inverse_mass_matrix):
    """Total energy = potential + kinetic."""
    potential = -log_prob(q)  # U(q) = -log p(q)
    kinetic = 0.5 * p @ inverse_mass_matrix @ p  # K(p) = 0.5 * p^T M^-1 p
    return potential + kinetic

def hmc_accept_reject(q0, p0, q1, p1, log_prob, inverse_mass_matrix, rng_key):
    """Metropolis-Hastings correction for integration error."""
    H0 = hamiltonian(q0, p0, log_prob, inverse_mass_matrix)
    H1 = hamiltonian(q1, p1, log_prob, inverse_mass_matrix)

    log_accept = H0 - H1  # Should be ~0 for perfect integration
    u = jax.random.uniform(rng_key)

    accept = jnp.log(u) < log_accept
    return jnp.where(accept, q1, q0), accept, H1 - H0  # Return energy error
```

## Mass Matrix Tuning

The mass matrix (metric) determines how momentum is distributed across dimensions.

### Why It Matters

```
Posterior with different scales:
  mu ~ N(0, 100)      # Wide prior
  sigma ~ HalfNormal(0.1)  # Narrow prior

Without tuning:
  - Same step size for both
  - mu needs large steps
  - sigma needs small steps
  - Compromise hurts both

With diagonal mass matrix adaptation:
  - M^{-1}_{mm} ≈ Var(mu)
  - M^{-1}_{ss} ≈ Var(sigma)
  - Each dimension gets appropriate scaling
```

### Adaptation Schemes

```python
def estimate_mass_matrix(samples, method='diagonal'):
    """Estimate inverse mass matrix from samples."""
    if method == 'diagonal':
        # Simple but effective
        return jnp.diag(jnp.var(samples, axis=0))
    elif method == 'full':
        # Better for correlated posteriors
        return jnp.cov(samples.T)
    elif method == 'regularized':
        # Shrinkage estimator for stability
        n, p = samples.shape
        S = jnp.cov(samples.T)
        trace_S = jnp.trace(S) / p
        shrinkage = min(1.0, (1 - 2/p) / jnp.sum((S - trace_S * jnp.eye(p))**2))
        return shrinkage * trace_S * jnp.eye(p) + (1 - shrinkage) * S
```

### Blackjax Window Adaptation

```python
import blackjax

# Window adaptation tunes both step_size and mass_matrix
warmup = blackjax.window_adaptation(
    blackjax.nuts,
    log_prob_fn,
    is_mass_matrix_diagonal=True,  # False for full matrix
    initial_step_size=1.0,
    target_acceptance_rate=0.8,  # NUTS optimal: 0.65-0.8
)

(state, params), info = warmup.run(rng_key, initial_position, num_steps=1000)

# Inspect adapted parameters
print(f"Step size: {params['step_size']}")
print(f"Mass matrix diag: {jnp.diag(params['inverse_mass_matrix'])}")
```

## Diagnosing Divergences

Divergences indicate numerical instability—the integrator is failing to conserve energy.

### What Causes Divergences

1. **Step size too large** for local curvature
2. **Funnel geometries** (narrow regions with high curvature)
3. **Numerical precision** issues
4. **Improper priors** creating undefined regions

### Detecting Divergences

```python
def analyze_divergences(infos, samples):
    """Analyze where divergences occur."""
    divergent_mask = infos.is_divergent

    n_divergent = jnp.sum(divergent_mask)
    print(f"Divergent transitions: {n_divergent} / {len(divergent_mask)}")

    if n_divergent > 0:
        # Where in parameter space?
        divergent_samples = samples[divergent_mask]
        normal_samples = samples[~divergent_mask]

        print("\nDivergent samples statistics:")
        print(f"  Mean: {jnp.mean(divergent_samples, axis=0)}")
        print(f"  Std:  {jnp.std(divergent_samples, axis=0)}")

        print("\nNormal samples statistics:")
        print(f"  Mean: {jnp.mean(normal_samples, axis=0)}")
        print(f"  Std:  {jnp.std(normal_samples, axis=0)}")

        # Often divergences cluster near boundaries or in narrow regions
```

### Fixing Divergences

```python
# 1. Reduce step size
params['step_size'] *= 0.5

# 2. Increase precision
jax.config.update("jax_enable_x64", True)

# 3. Reparameterize (see below)

# 4. Improve priors
def better_prior():
    # Bounded instead of unbounded
    sigma = numpyro.sample('sigma', dist.LogNormal(0, 1))
    # Instead of HalfNormal which can be too close to 0
```

## Reparameterization Techniques

### Non-Centered Parameterization

The classic fix for funnel geometries in hierarchical models.

```python
# CENTERED (problematic):
# y_i ~ N(mu, sigma)
# When sigma → 0, y_i is tightly constrained → narrow funnel

def centered_model(data):
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    with numpyro.plate('data', len(data)):
        y = numpyro.sample('y', dist.Normal(mu, sigma), obs=data)

# NON-CENTERED (better):
# z_i ~ N(0, 1)  # Standard normal
# y_i = mu + sigma * z_i  # Transform

def noncentered_model(data):
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    with numpyro.plate('data', len(data)):
        z = numpyro.sample('z', dist.Normal(0, 1))
        y = numpyro.deterministic('y', mu + sigma * z)
        numpyro.sample('obs', dist.Normal(y, 0.1), obs=data)
```

### Log-Scale for Positive Parameters

```python
# Instead of:
sigma = numpyro.sample('sigma', dist.HalfNormal(1))

# Use:
log_sigma = numpyro.sample('log_sigma', dist.Normal(0, 1))
sigma = numpyro.deterministic('sigma', jnp.exp(log_sigma))

# This makes the parameter unbounded, easier for HMC
```

### Cholesky for Covariance Matrices

```python
from numpyro.distributions import LKJCholesky

def multivariate_model(data):
    n_dims = data.shape[1]

    # Sample correlation matrix via Cholesky
    L_omega = numpyro.sample('L_omega', LKJCholesky(n_dims, concentration=2.0))

    # Sample scales
    sigma = numpyro.sample('sigma', dist.HalfNormal(1).expand([n_dims]))

    # Construct covariance Cholesky factor
    L_Sigma = jnp.diag(sigma) @ L_omega

    # Mean
    mu = numpyro.sample('mu', dist.Normal(0, 5).expand([n_dims]))

    # Likelihood
    numpyro.sample('obs', dist.MultivariateNormal(mu, scale_tril=L_Sigma), obs=data)
```

## Simulation-Based Inference (SBI)

For expensive simulations where likelihood is intractable.

### Neural Surrogate Models

```python
import equinox as eqx
import optax

class SurrogateModel(eqx.Module):
    """Neural emulator for expensive simulation."""
    layers: list

    def __init__(self, key, input_dim, hidden_dim, output_dim):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, output_dim, key=keys[2]),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

def train_surrogate(model, params_train, sims_train, n_epochs=1000):
    """Train surrogate on simulation outputs."""
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, params, sims):
        def loss_fn(m):
            pred = jax.vmap(m)(params)
            return jnp.mean((pred - sims) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for epoch in range(n_epochs):
        model, opt_state, loss = step(model, opt_state, params_train, sims_train)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss:.6f}")

    return model

def surrogate_log_prob(surrogate, observations, params):
    """Log-prob using surrogate predictions."""
    predicted = surrogate(params)
    return jnp.sum(jax.scipy.stats.norm.logpdf(observations, predicted, 0.1))
```

### Diffrax Integration

```python
import diffrax

def bayesian_ode_model(observations, ts):
    """Bayesian parameter estimation with differentiable ODE."""

    # Priors on ODE parameters
    k = numpyro.sample('k', dist.LogNormal(0, 1))
    y0 = numpyro.sample('y0', dist.Normal(1, 0.1))
    sigma = numpyro.sample('sigma', dist.HalfNormal(0.1))

    # Define ODE
    def ode_fn(t, y, args):
        return -k * y  # Exponential decay

    # Solve ODE
    term = diffrax.ODETerm(ode_fn)
    solver = diffrax.Tsit5()
    solution = diffrax.diffeqsolve(
        term, solver,
        t0=ts[0], t1=ts[-1], dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        adjoint=diffrax.RecursiveCheckpointAdjoint()  # For backprop
    )

    # Likelihood
    predicted = solution.ys
    with numpyro.plate('data', len(observations)):
        numpyro.sample('obs', dist.Normal(predicted, sigma), obs=observations)
```

## Soft Matter Applications

### Neighbor Lists for Pairwise Interactions

```python
def compute_pairwise_energy(positions, box_size, cutoff, epsilon, sigma):
    """Lennard-Jones energy with neighbor lists."""
    n_particles = positions.shape[0]

    # Compute all pairwise distances (with PBC)
    def pairwise_distance(pos_i, pos_j):
        delta = pos_i - pos_j
        delta = delta - box_size * jnp.round(delta / box_size)  # PBC
        return jnp.sqrt(jnp.sum(delta ** 2))

    # Vectorize over all pairs
    distances = jax.vmap(lambda i: jax.vmap(lambda j: pairwise_distance(positions[i], positions[j]))(jnp.arange(n_particles)))(jnp.arange(n_particles))

    # Mask diagonal and beyond cutoff
    mask = (distances > 0) & (distances < cutoff)

    # LJ potential
    r6 = (sigma / distances) ** 6
    energy = 4 * epsilon * (r6 ** 2 - r6)
    energy = jnp.where(mask, energy, 0.0)

    return 0.5 * jnp.sum(energy)  # Factor of 0.5 for double-counting

def soft_matter_log_prob(params, observed_positions, box_size):
    """Log-prob for soft matter system."""
    epsilon = params['epsilon']
    sigma = params['sigma']

    # Prior
    log_prior = (
        jax.scipy.stats.norm.logpdf(jnp.log(epsilon), 0, 1) +
        jax.scipy.stats.norm.logpdf(jnp.log(sigma), 0, 0.5)
    )

    # Likelihood: Boltzmann distribution
    energy = compute_pairwise_energy(observed_positions, box_size, 2.5 * sigma, epsilon, sigma)
    log_lik = -energy  # At T=1

    return log_prior + log_lik
```

### Parallel Walkers for Rare Events

```python
def parallel_walkers(log_prob_fn, initial_walkers, n_steps, rng_key):
    """Run many walkers in parallel using pmap."""

    @jax.pmap
    def run_walker(init_pos, key):
        warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
        (state, params), _ = warmup.run(key, init_pos, num_steps=100)

        kernel = blackjax.nuts(log_prob_fn, **params).step
        keys = jax.random.split(key, n_steps)

        def step(state, key):
            state, info = kernel(key, state)
            return state, state.position

        _, trajectory = jax.lax.scan(step, state, keys)
        return trajectory

    n_walkers = len(initial_walkers)
    keys = jax.random.split(rng_key, n_walkers)

    # Shape: (n_devices, n_steps, param_dim)
    all_trajectories = run_walker(initial_walkers, keys)

    return all_trajectories
```
