---
name: stochastic-dynamics
description: Model stochastic dynamics using master equations, Fokker-Planck, Langevin dynamics, and Green-Kubo transport theory. Use when simulating noise-driven systems, calculating transport coefficients, or modeling rare events.
---

# Stochastic Dynamics & Transport

Stochastic processes and transport properties from microscopic dynamics.

## Expert Agent

For stochastic modeling, Langevin simulations, and rare event sampling, delegate to the expert agent:

- **`statistical-physicist`** or **`simulation-expert`**:
  - **`statistical-physicist`**: For theoretical framework, Fokker-Planck equations, and transport theory.
    - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - **`simulation-expert`**: For numerical implementation of Langevin dynamics and rare event sampling.
    - *Location*: `plugins/science-suite/agents/simulation-expert.md`

## Framework Selection

| Framework | Use Case | Equation |
|-----------|----------|----------|
| Master Equation | Discrete states | dP_n/dt = Σ[W_{mn}P_m - W_{nm}P_n] |
| Fokker-Planck | Probability evolution | ∂P/∂t = -∂(μF·P)/∂x + D·∂²P/∂x² |
| Langevin | Stochastic trajectories | dx/dt = μF(x) + √(2D)ξ(t) |

## Gillespie Algorithm (Master Equation)

```python
def gillespie_step(state, propensities):
    total_rate = sum(propensities)
    tau = np.random.exponential(1/total_rate)
    reaction = np.random.choice(len(propensities), p=propensities/total_rate)
    return tau, reaction
```

## Langevin Dynamics

```python
def langevin_step(x, F, mu, D, dt):
    # Euler-Maruyama integration
    return x + mu*F(x)*dt + np.sqrt(2*D*dt)*np.random.randn()
```

White noise: ⟨ξ(t)ξ(t')⟩ = δ(t-t')

## Green-Kubo Relations

| Property | Formula |
|----------|---------|
| Diffusion | D = ∫⟨v(t)·v(0)⟩ dt |
| Viscosity | η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩ dt |
| Thermal conductivity | κ = (V/kT²)∫⟨J_q(t)·J_q(0)⟩ dt |

**Einstein relations**: D = kT·μ (diffusion-mobility)

## Rare Event Sampling

| Method | Approach |
|--------|----------|
| Forward Flux Sampling | Cross interfaces sequentially |
| Transition Path Sampling | Sample reactive trajectories |
| Metadynamics | History-dependent bias |

## Applications

| System | Method |
|--------|--------|
| Brownian motion | Langevin, overdamped |
| Chemical kinetics | Master equation, Gillespie |
| Polymer dynamics | Langevin (Rouse/Zimm) |
| Transport properties | Green-Kubo, NEMD |
| Glass relaxation | Stress autocorrelation |

## JAX-accelerated Langevin pattern

Euler-Maruyama on a differentiable potential, vectorized across replicas for ensemble statistics. `jax.vmap` over the batch axis + `jax.lax.scan` over time gives a full GPU-utilizing loop in ~20 lines.

```python
import jax, jax.numpy as jnp, jax.random as jr

def langevin_step(x, key, grad_U, dt, kT):
    force = -grad_U(x)
    noise = jr.normal(key, x.shape)
    return x + force * dt + jnp.sqrt(2 * kT * dt) * noise

def simulate_ensemble(U, x0, key, n_steps, dt, kT):
    grad_U = jax.grad(U)
    def body(carry, key_t):
        x = carry
        x = jax.vmap(lambda xi, k: langevin_step(xi, k, grad_U, dt, kT))(
            x, jr.split(key_t, x.shape[0]))
        return x, x
    keys = jr.split(key, n_steps)
    _, traj = jax.lax.scan(body, x0, keys)
    return traj                                   # (n_steps, n_replicas, d)
```

See `jax-physics-applications` for the thermostat/integrator catalog and `advanced-simulations` for the production MD stack.

## Numerical Fokker-Planck

When the state space is 1-2D, solve the Fokker-Planck equation directly rather than sampling its stationary distribution via Langevin:

| Tool | Role |
|------|------|
| **`py-pde`** | Finite-difference / spectral PDE solver — define the Fokker-Planck operator as a `PDE` class; supports periodic, Dirichlet, Neumann boundaries. See `numerical-methods-implementation`. |
| **`fenics` / `firedrake`** | Finite-element — use when the drift/diffusion structure is spatially complex or the geometry is non-rectangular |
| **`DifferentialEquations.jl` + `MethodOfLines`** | Julia path — discretize ∂ₜP = -∇·(μFP) + ∇²(DP) via MOL. See `neural-pde` for the MOL pattern. |

For the **stationary** distribution, bypass time integration entirely: discretize the generator and solve the eigenproblem Lφ = 0 for the zero-eigenvalue eigenvector.

## Stochastic differential equations — library choices

| Library | Strengths | Integrator choices |
|---------|-----------|--------------------|
| **`diffrax`** (JAX) | AD-friendly, composes with NumPyro / Equinox / Optax; solvers implement the `AbstractSolver` protocol | `EulerHeun`, `ItoMilstein`, `StratonovichMilstein`, `SPaRK`, `SRA1`, `SlowRK` |
| **`DifferentialEquations.jl` / `StochasticDiffEq.jl`** | Largest SDE solver catalog in any language; adaptive strong/weak order control | `EM`, `SRIW1`, `SRA1`, `SOSRA`, `ImplicitEM`, `SKenCarp` (stiff) |
| **`sdeint`** (Python) | NumPy reference; no GPU | `itoEuler`, `itoSRI2`, `stratHeun`, `stratKP2iS` |
| **`torchsde`** | PyTorch-native for learned SDEs | `srk`, `reversible-heun`, `midpoint` |

See `sciml-and-diffeq` for the surrounding SciML ecosystem and `jax-diffeq-pro` for Diffrax integration with NumPyro likelihoods.

## Research applications

- **Rare events from Langevin / SDE** — combine the ensemble pattern above with forward-flux sampling or weighted ensemble (see `advanced-simulations` for WESTPA / OPS).
- **Learning the drift from data** — if the potential `U(x)` is unknown, fit it with a Lux/Equinox MLP and train against a trajectory log-likelihood. This overlaps with `bayesian-ude-workflow` when uncertainty on the learned drift is required.
- **Jump-diffusion / PDMP** — SDEs with embedded jump events belong in `catalyst-reactions` alongside `JumpProcesses.jl` and PDMP.

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Match timescales | Timestep << relaxation time |
| Overdamped limit | Use when inertia negligible |
| Convergence check | Integrate correlations sufficiently |
| Validate Einstein | Check D = kT·μ relation |
| Sample sufficiently | Long trajectories for statistics |

## Checklist

- [ ] Appropriate framework selected
- [ ] Noise strength correctly parameterized
- [ ] Timestep small enough for stability
- [ ] Correlation functions converged
- [ ] Transport coefficients validated
- [ ] Einstein relations checked
