---
name: active-matter
description: Model active matter including self-propelled particles, flocking, pattern formation, and collective behavior. Use when simulating ABPs, Vicsek model, MIPS, reaction-diffusion systems, or designing bio-inspired materials.
---

# Active Matter & Complex Systems

Model self-propelled particles, pattern formation, and collective behavior.

## Expert Agent

For simulating active matter systems, analyzing collective motion, and modeling non-equilibrium phase transitions, delegate to the expert agent:

- **`statistical-physicist`**: Unified specialist for Statistical Physics, Soft Matter, and Active Matter.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - *Capabilities*: Active Brownian Particles (ABP) simulation, MIPS analysis, and topological defect tracking.

## Active Matter Models

| Model | Dynamics | Application |
|-------|----------|-------------|
| Active Brownian Particles | dr/dt = v₀n̂ + √(2D)ξ | Colloids, bacteria |
| Vicsek | θ_i(t+1) = ⟨θ⟩ + noise | Flocking, swarms |
| MIPS | Activity → phase separation | Bacterial colonies |
| Toner-Tu | Hydrodynamic active fluids | 2D flocking order |

## Active Brownian Particles

```python
def abp_step(r, theta, v0, Dr, Dt, dt):
    r += v0 * np.array([np.cos(theta), np.sin(theta)]) * dt
    r += np.sqrt(2*Dt*dt) * np.random.randn(2)
    theta += np.sqrt(2*Dr*dt) * np.random.randn()
    return r, theta
```

**Parameters**: v₀ (propulsion), D_r (rotational diffusion), D_t (translational diffusion)

### JAX-accelerated ensemble ABP

For production active-matter simulation (thousands of particles, many replicas), vectorize across the particle axis with `jax.vmap` and scan over time with `jax.lax.scan`:

```python
import jax, jax.numpy as jnp, jax.random as jr

def abp_step(state, key, v0, Dt, Dr, dt):
    r, theta = state                                   # (N, 2), (N,)
    k1, k2 = jr.split(key)
    fwd = v0 * jnp.stack([jnp.cos(theta), jnp.sin(theta)], -1)
    r = r + fwd * dt + jnp.sqrt(2 * Dt * dt) * jr.normal(k1, r.shape)
    theta = theta + jnp.sqrt(2 * Dr * dt) * jr.normal(k2, theta.shape)
    return (r, theta), r                               # carry, emit positions

def simulate_abp(N, T, key, v0, Dt, Dr, dt, L):
    k0, k1 = jr.split(key)
    r0 = jr.uniform(k0, (N, 2), minval=0.0, maxval=L)
    theta0 = jr.uniform(k1, (N,), minval=0.0, maxval=2 * jnp.pi)
    keys = jr.split(key, T)
    _, traj = jax.lax.scan(lambda s, k: abp_step(s, k, v0, Dt, Dr, dt), (r0, theta0), keys)
    return traj                                        # (T, N, 2)
```

`jax.vmap` this across a replica axis to run independent ensembles on one GPU; see `stochastic-dynamics` for the Langevin base pattern.

## Pattern Formation

### Reaction-Diffusion (Turing)

```
∂u/∂t = D_u∇²u + f(u,v)
∂v/∂t = D_v∇²v + g(u,v)
```

| Condition | Pattern |
|-----------|---------|
| D_u ≠ D_v | Instability |
| Specific kinetics | Spots, stripes, labyrinths |

### FitzHugh-Nagumo

```
∂u/∂t = D∇²u + u(u-a)(1-u) - v
∂v/∂t = ε(u - γv)
```

Produces: Traveling waves, spirals, excitable media

## Collective Behavior

### Boids Rules (Swarm Dynamics)

1. **Cohesion**: Move toward center of mass
2. **Alignment**: Match neighbors' velocity
3. **Separation**: Avoid collisions

### Chemotaxis (Keller-Segel)

```
∂ρ/∂t = D∇²ρ - χ∇·(ρ∇c)
```

Applications: Cell aggregation, wound healing

## Computational Methods

| Method | Use Case |
|--------|----------|
| Agent-based | Discrete particles, interactions |
| Continuum PDE | Large-scale patterns |
| Spectral methods | Periodic domains |
| Adaptive mesh | Sharp interfaces |

## Materials Applications

| Application | Principle |
|-------------|-----------|
| Active metamaterials | Self-propelling structures |
| Bio-inspired materials | Artificial cilia, self-healing |
| Microfluidic control | Bacteria-driven mixing |
| Swarm robotics | Collective decision-making |

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Choose appropriate model | ABP for particles, PDE for continuum |
| Validate phase behavior | Check for MIPS, flocking transitions |
| Scale correctly | Match length/time scales to physics |
| Test collective emergence | Verify order parameters |

---

## Python / JAX ↔ Julia ecosystem map

| Role | Python / JAX | Julia |
|---|---|---|
| Active-particle MD (production GPU) | **`HOOMD-blue`** with `hoomd.md.force.Active`; **`ESPResSo`**; **`LAMMPS`** `fix active` | `Molly.jl` (pure Julia, GPU via CUDA.jl); `LAMMPS.jl` wrapper |
| Lattice Vicsek / run-and-tumble models | hand-rolled with **`numba`** or **`jax`** (`jax.lax.scan` over spins) | hand-rolled with `StaticArrays` + `KernelAbstractions.jl` |
| Order parameters & defect tracking | **`freud.order.Nematic`**, **`freud.order.Hexatic`**, **`scikit-image`** topological-defect tracking | via `freud` through `PythonCall`; or hand-rolled nearest-neighbor Q_l |
| MIPS detection / cluster analysis | **`freud.cluster.Cluster`**, **`scipy.spatial.cKDTree`** | `NearestNeighbors.jl` + hand-rolled connected-components |
| Reaction-diffusion / pattern PDEs | **`py-pde`**, **`fipy`** | `MethodOfLines.jl` + `DifferentialEquations.jl` (symbolic PDE pipeline; see `neural-pde`) |
| Bayesian inference of activity / self-propulsion rate | NumPyro on JAX ABP trajectories | Turing.jl + `StochasticDiffEq`; see `bayesian-ude-workflow` for the hybrid approach |

> **Stay in Julia** when the active-matter system is coupled to a symbolic reaction-diffusion PDE (MTK + MethodOfLines pipeline) or when the workflow shares the session with SciML sensitivity analysis for activity-parameter inference. **Drop to HOOMD-blue / ESPResSo / LAMMPS** for production GPU-accelerated particle MD with established force-field libraries, established visualization pipelines (OVITO), and the full `freud` analysis ecosystem downstream. The JAX ABP pattern above is the right call when you need a tight differentiable loop (learning self-propulsion from data, Bayesian activity inference) rather than conventional simulation throughput.

## Checklist

- [ ] Model selection matches physics
- [ ] Parameters in realistic regime
- [ ] Phase behavior characterized
- [ ] Collective order measured
- [ ] Boundary conditions appropriate
- [ ] Time scales resolved
