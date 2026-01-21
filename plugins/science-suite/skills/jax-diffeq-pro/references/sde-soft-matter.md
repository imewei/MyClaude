# SDEs and Soft Matter Applications

Stochastic differential equations, VirtualBrownianTree, Itō vs Stratonovich interpretation, neural ODEs, and rheology-specific patterns.

## Stochastic Differential Equations

Soft matter is thermal—fluctuations matter. Diffrax supports SDEs with proper Brownian motion handling.

### SDE Basics

```
General SDE form:
dy = f(t, y) dt + g(t, y) dW

Where:
- f(t, y) = drift term (deterministic)
- g(t, y) = diffusion term (noise coefficient)
- dW = Wiener process increment
```

### Basic SDE in Diffrax

```python
import diffrax
import jax
import jax.numpy as jnp

def drift(t, y, args):
    """Deterministic part: dy/dt when no noise."""
    return -args['k'] * y

def diffusion(t, y, args):
    """Noise coefficient: scales Brownian motion."""
    return args['sigma'] * jnp.ones_like(y)

# Create terms
drift_term = diffrax.ODETerm(drift)
diffusion_term = diffrax.ControlTerm(diffusion, diffrax.VirtualBrownianTree(
    t0=0.0, t1=10.0, tol=1e-3, shape=(1,), key=jax.random.PRNGKey(0)
))

# Combine
terms = diffrax.MultiTerm(drift_term, diffusion_term)

# Solve
solver = diffrax.Euler()  # Simple SDE solver
solution = diffrax.diffeqsolve(
    terms, solver,
    t0=0.0, t1=10.0, dt0=0.01,
    y0=jnp.array([1.0]),
    args={'k': 0.5, 'sigma': 0.1},
)
```

## VirtualBrownianTree

The key to reproducible, adaptive-step SDE solving.

### Why VirtualBrownianTree?

```
Problem with naive noise:
- Fixed-step: noise = sqrt(dt) * random.normal()
- Adaptive-step: Different dt at each step
- Changing dt changes the noise realization!
- Results are non-reproducible

Solution: VirtualBrownianTree
- Defines Brownian motion path upfront
- Queries consistent values at any time
- Adaptive stepping doesn't change the path
```

### Using VirtualBrownianTree

```python
# Create Brownian motion path
brownian = diffrax.VirtualBrownianTree(
    t0=0.0,           # Start time
    t1=10.0,          # End time
    tol=1e-3,         # Tolerance for path reconstruction
    shape=(3,),       # Shape of Brownian motion (3D particle)
    key=jax.random.PRNGKey(42),  # Reproducibility!
)

# Use in ControlTerm
diffusion_term = diffrax.ControlTerm(
    diffusion_fn,
    brownian,
)
```

### Multiple Independent Brownian Motions

```python
# For n particles with independent noise
n_particles = 100
dim = 3

# Shape: (n_particles, dim)
brownian = diffrax.VirtualBrownianTree(
    t0=0.0, t1=10.0, tol=1e-3,
    shape=(n_particles, dim),
    key=jax.random.PRNGKey(0),
)
```

## Itō vs Stratonovich Interpretation

The choice matters for physics!

### When to Use Which

```
Itō (default in most libraries):
- Natural for financial math
- Martingale property preserved
- Used when noise is "external"

Stratonovich (physics convention):
- Preserves chain rule: d(f(y)) = f'(y) dy
- Natural for physical systems with inertia
- Used when noise is "internal" (thermal fluctuations)
- Limit of correlated noise → white noise
```

### Implementing Stratonovich SDEs

```python
# Stratonovich SDE: dy = f(t,y) dt + g(t,y) ∘ dW
# Convert to Itō: dy = [f(t,y) + 0.5 * g(t,y) * g'(t,y)] dt + g(t,y) dW

def stratonovich_to_ito(drift_strat, diffusion, y):
    """Convert Stratonovich drift to Itō drift."""
    # Noise-induced drift correction
    g = diffusion(0.0, y, None)
    # dg/dy (Jacobian of diffusion)
    dg_dy = jax.jacobian(lambda y: diffusion(0.0, y, None))(y)
    correction = 0.5 * jnp.sum(g[:, None] * dg_dy, axis=0)
    return drift_strat(0.0, y, None) + correction

# Or use Diffrax's Stratonovich solver
solver = diffrax.StratonovichMilstein()
```

### Example: Overdamped Langevin Dynamics

```python
def langevin_drift(t, position, args):
    """Force from potential: F = -∇U"""
    def potential(x):
        return 0.5 * args['k'] * jnp.sum(x ** 2)  # Harmonic

    force = -jax.grad(potential)(position)
    return force / args['gamma']  # Overdamped: dx/dt = F/γ

def langevin_diffusion(t, position, args):
    """Thermal noise: sqrt(2 * k_B * T / γ)"""
    return jnp.sqrt(2 * args['kT'] / args['gamma']) * jnp.ones_like(position)

# Solve Langevin equation
terms = diffrax.MultiTerm(
    diffrax.ODETerm(langevin_drift),
    diffrax.ControlTerm(langevin_diffusion, brownian),
)

solution = diffrax.diffeqsolve(
    terms, diffrax.Euler(),
    t0=0.0, t1=100.0, dt0=0.01,
    y0=initial_positions,
    args={'k': 1.0, 'gamma': 1.0, 'kT': 1.0},
)
```

## Neural ODEs / Universal Differential Equations

Hybrid physics-ML models where part of the dynamics is learned.

### Basic Neural ODE

```python
import equinox as eqx

class NeuralODE(eqx.Module):
    """Neural network as vector field."""
    net: eqx.nn.MLP

    def __init__(self, key, hidden_dim=32, state_dim=2):
        self.net = eqx.nn.MLP(
            in_size=state_dim + 1,  # state + time
            out_size=state_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        t_expanded = jnp.array([t])
        input = jnp.concatenate([t_expanded, y])
        return self.net(input)

# Train by backprop through ODE solve
def loss_fn(model, y0, target_trajectory, ts):
    term = diffrax.ODETerm(model)
    solver = diffrax.Tsit5()

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=ts[0], t1=ts[-1], dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    return jnp.mean((sol.ys - target_trajectory) ** 2)
```

### Universal Differential Equations (UDEs)

Combine known physics with learned corrections.

```python
class UDE_Rheology(eqx.Module):
    """Known constitutive model + neural correction."""
    neural_correction: eqx.nn.MLP
    physics_params: dict

    def __call__(self, t, stress, args):
        shear_rate = args['shear_rate']

        # Known physics: Maxwell model
        # dσ/dt = G * γ̇ - σ/λ
        G = self.physics_params['G']  # Modulus
        lam = self.physics_params['lambda']  # Relaxation time

        physics_term = G * shear_rate - stress / lam

        # Learned correction for nonlinear effects
        input = jnp.concatenate([stress, jnp.array([shear_rate])])
        neural_term = self.neural_correction(input)

        return physics_term + neural_term
```

## Rheology-Specific Patterns

### Oldroyd-B Model

```python
def oldroyd_b(t, state, args):
    """Oldroyd-B constitutive equation for viscoelastic fluids.

    State: [σ_xx, σ_xy, σ_yy] - extra stress tensor components
    """
    sigma_xx, sigma_xy, sigma_yy = state
    shear_rate = args['shear_rate']  # γ̇
    lam = args['lambda']  # Relaxation time
    eta_p = args['eta_p']  # Polymer viscosity

    # Upper convected derivative in simple shear
    d_sigma_xx = -sigma_xx / lam + 2 * sigma_xy * shear_rate
    d_sigma_xy = -sigma_xy / lam + sigma_yy * shear_rate + eta_p * shear_rate / lam
    d_sigma_yy = -sigma_yy / lam

    return jnp.array([d_sigma_xx, d_sigma_xy, d_sigma_yy])
```

### Thixotropic Model with Yield Stress

```python
class ThixotropicModel(eqx.Module):
    """Structure-based thixotropic model."""

    def __call__(self, t, state, args):
        stress, structure = state  # structure ∈ [0, 1]
        shear_rate = args['shear_rate']

        # Yield stress depends on structure
        yield_stress = args['tau_y0'] * structure

        # Stress evolution
        if stress < yield_stress:
            # Below yield: elastic
            d_stress = args['G'] * shear_rate
        else:
            # Above yield: viscoplastic
            d_stress = (args['eta'] * shear_rate - (stress - yield_stress)) / args['lambda']

        # Structure evolution: breakdown under shear, rebuild at rest
        d_structure = args['k_b'] * (1 - structure) - args['k_d'] * structure * shear_rate

        return jnp.array([d_stress, d_structure])
```

### Brownian Dynamics of Colloidal Particles

```python
def colloidal_dynamics(t, positions, args):
    """Brownian dynamics with pairwise interactions.

    positions: (n_particles, 3)
    """
    n_particles = positions.shape[0]
    kT = args['kT']
    gamma = args['gamma']

    # Pairwise forces (e.g., Lennard-Jones)
    forces = compute_pairwise_forces(positions, args)

    # Overdamped: dx/dt = F/γ
    return forces / gamma

def colloidal_diffusion(t, positions, args):
    """Thermal diffusion coefficient."""
    kT = args['kT']
    gamma = args['gamma']
    D = kT / gamma  # Einstein relation
    return jnp.sqrt(2 * D) * jnp.ones_like(positions)

def compute_pairwise_forces(positions, args):
    """Compute forces from pairwise potential."""
    n = positions.shape[0]

    def force_on_i(i):
        def pair_force(j):
            r = positions[i] - positions[j]
            dist = jnp.linalg.norm(r) + 1e-10
            # Lennard-Jones derivative
            f_mag = 24 * args['epsilon'] * (
                2 * (args['sigma'] / dist) ** 12 -
                (args['sigma'] / dist) ** 6
            ) / dist
            return f_mag * r / dist

        # Sum over all j ≠ i
        forces = jax.vmap(pair_force)(jnp.arange(n))
        mask = jnp.arange(n) != i
        return jnp.sum(jnp.where(mask[:, None], forces, 0), axis=0)

    return jax.vmap(force_on_i)(jnp.arange(n))
```

## Continuous-Discrete Hybrids

Systems with continuous evolution and discrete jumps.

### Bond Breaking/Formation

```python
def hybrid_dynamics(t, state, args):
    """Polymer with dynamic bonds."""
    positions, bond_state = state[:n_coords], state[n_coords:]

    # Continuous: particle motion
    d_positions = compute_motion(positions, bond_state, args)

    # Bond state evolution (approximate discrete with soft transition)
    bond_stress = compute_bond_stress(positions, bond_state, args)
    breaking_rate = args['k_b'] * jnp.exp(bond_stress / args['f_c'])
    forming_rate = args['k_f'] * (1 - bond_state)

    d_bonds = forming_rate - breaking_rate * bond_state

    return jnp.concatenate([d_positions, d_bonds])
```

### Using lax.cond for Sharp Transitions

```python
def vector_field_with_regime(t, y, args):
    """Different physics in different regimes."""
    stress = y[0]
    yield_stress = args['yield_stress']

    def elastic_regime(y):
        return args['G'] * args['strain_rate']

    def plastic_regime(y):
        return (args['eta'] * args['strain_rate'] - y[0]) / args['lambda']

    d_stress = jax.lax.cond(
        stress < yield_stress,
        elastic_regime,
        plastic_regime,
        y
    )

    return jnp.array([d_stress])
```

## Reproducibility Checklist

For publishable soft matter simulations:

```python
def reproducible_simulation(seed, params):
    """Fully reproducible SDE simulation."""

    # 1. Fixed RNG seed
    key = jax.random.PRNGKey(seed)

    # 2. VirtualBrownianTree for reproducible Brownian path
    brownian = diffrax.VirtualBrownianTree(
        t0=0.0, t1=params['t_final'],
        tol=1e-3,
        shape=params['state_shape'],
        key=key,
    )

    # 3. Explicit solver configuration
    solver = diffrax.Heun()  # Document solver choice
    controller = diffrax.PIDController(
        rtol=params['rtol'],
        atol=params['atol'],
    )

    # 4. Explicit saveat for output times
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, params['t_final'], params['n_save']))

    # 5. Solve
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )

    solution = diffrax.diffeqsolve(
        terms, solver,
        t0=0.0, t1=params['t_final'], dt0=params['dt0'],
        y0=params['y0'],
        args=params['args'],
        stepsize_controller=controller,
        saveat=saveat,
    )

    return solution

# Document everything
params = {
    'seed': 42,
    'rtol': 1e-5,
    'atol': 1e-7,
    'dt0': 0.001,
    't_final': 100.0,
    'n_save': 1000,
    # ... physical parameters
}
```
