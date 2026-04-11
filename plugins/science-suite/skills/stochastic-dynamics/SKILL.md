---
name: stochastic-dynamics
description: Model stochastic dynamics using master equations, Fokker-Planck, Langevin dynamics, Green-Kubo transport theory, and jump-diffusion SDEs (dx = f dt + g dW + J dN). Use when simulating noise-driven systems, calculating transport coefficients, modeling rare events, or simulating general physics jump-diffusion SDEs outside the reaction-network context (for biochemical reaction networks use catalyst-reactions instead).
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

When the state space is 1-2D (or 3D for simple geometries), solve the Fokker-Planck PDE directly rather than sampling its stationary distribution via Langevin. Two regimes: **time-dependent** propagation of P(x, t) and **stationary** solution via the zero-eigenvalue eigenvector of the generator.

### Tool selection

| Tool | Geometry | Use when |
|------|----------|----------|
| **`py-pde`** (Python) | Structured grids (Cartesian, cylindrical, spherical, polar) | Rapid prototyping of 1D/2D FP; define the operator as a `PDE` class with periodic / Dirichlet / Neumann / no-flux boundaries in a few lines |
| **`DOLFINx`** (Python, modern FEniCSx) | Unstructured FEM, arbitrary geometry | Irregular domain, spatially-varying drift/diffusion, or when you need mass-conserving flux BCs that a grid can't express |
| **`firedrake`** (Python) | Unstructured FEM, high-order | Same niche as DOLFINx; alternative form-language implementation |
| **`MethodOfLines.jl`** (Julia) | Structured grids | MOL discretization feeding into any DifferentialEquations.jl solver — stiff implicit integration comes free |
| **`Gridap.jl`** / **`Ferrite.jl`** (Julia) | Unstructured FEM | Julia-native FEM; pair with `DifferentialEquations.jl` for time integration |

### Time-dependent FP with `py-pde`

Overdamped FP in 2D: `∂ₜ P = -∇·(μF P) + D ∇²P` with a scalar drift `F = -∇U`.

```python
from pde import PDE, ScalarField, CartesianGrid

grid = CartesianGrid([[-5, 5], [-5, 5]], [128, 128], periodic=False)
P0   = ScalarField.from_expression(grid, "exp(-(x**2+y**2)/0.2) / (0.2*pi)")  # initial delta-ish

# U(x,y) = 0.5*(x^2 + y^2) - double-well in x: -x^2/2 + x^4/4
# F = -grad U; FP = -div(F*P) + D*lap(P)
eq = PDE(
    {"P": "-d_dx((-x + x**3) * P) - d_dy(y * P) + 0.1 * laplace(P)"},
    bc=[{"value": 0}, {"value": 0}],   # Dirichlet=0 on all sides
)
result = eq.solve(P0, t_range=5.0, dt=1e-3, tracker=["progress"])
```

`py-pde` exposes `d_dx`, `d_dy`, `laplace`, `divergence`, `gradient` as operator shorthand. For non-Cartesian geometries use `PolarGrid`, `CylindricalSymGrid`, `SphericalSymGrid`. See `numerical-methods-implementation` for the full `py-pde` interface.

### Irregular domains with `DOLFINx` (FEniCSx)

For geometries that a structured grid cannot express — off-axis drifts, curved boundaries, species-specific reaction zones — use DOLFINx. The canonical pattern for a time-dependent FP on a rectangle, discretized by implicit Euler + Lagrange elements:

```python
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl, numpy as np

msh = mesh.create_rectangle(MPI.COMM_WORLD, ((-5, -5), (5, 5)), (64, 64))
V   = fem.functionspace(msh, ("Lagrange", 1))

P      = ufl.TrialFunction(V)
v      = ufl.TestFunction(V)
P_n    = fem.Function(V)              # previous time step
dt     = fem.Constant(msh, 1e-2)
D      = fem.Constant(msh, 0.1)
x      = ufl.SpatialCoordinate(msh)
drift  = ufl.as_vector((-x[0] + x[0]**3, -x[1]))   # F = -grad U

# Implicit Euler weak form for ∂ₜP = -∇·(F P) + D ΔP
a = (P * v + dt * ufl.inner(D * ufl.grad(P), ufl.grad(v))
         - dt * P * ufl.dot(drift, ufl.grad(v))) * ufl.dx
L = P_n * v * ufl.dx

bcs = []    # supply zero-flux / Dirichlet here as needed
problem = LinearProblem(a, L, bcs=bcs, u=fem.Function(V),
                        petsc_options_prefix="fp_",
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
```

Loop `problem.solve()` then `P_n.x.array[:] = uh.x.array` for each step. For reactive / non-conservative extensions, switch to `NonlinearProblem` + SNES. Full DOLFINx catalog: `numerical-methods-implementation`.

### Stationary distribution via the generator eigenproblem

The stationary Fokker-Planck solution is the zero-eigenvalue eigenvector of the adjoint generator `L†`. Discretize `L†` as a sparse matrix, then solve for the smallest-magnitude eigenvalue directly — no time integration.

```python
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigs

# 1D FP generator on a uniform grid with no-flux BCs
N, dx, D = 256, 10.0 / 256, 0.1
x  = np.linspace(-5, 5, N)
F  = -x + x**3                    # drift component
# Discretize L† u = -∂_x(F u) + D ∂²_x u with centered differences
upper = D / dx**2 - F[:-1] / (2 * dx)
lower = D / dx**2 + F[1:]  / (2 * dx)
main  = -2 * D / dx**2 * np.ones(N)
Ldag  = diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")
# Solve for the smallest-magnitude eigenvalue — the stationary mode
w, v  = eigs(Ldag, k=1, sigma=0.0)
P_ss  = np.real(v[:, 0]); P_ss /= np.trapezoid(P_ss, x)
```

The same trick works in 2D via `kron` of 1D operators for separable drift, or via `DOLFINx` + `slepc4py.SLEPc.EPS` for unstructured meshes. In Julia, use `ArnoldiMethod.jl` or `KrylovKit.jl` `eigsolve` on a `SparseMatrixCSC` operator built the same way — stable and mass-conserving.

### When to use FP-PDE vs Langevin ensemble

| Regime | Prefer |
|--------|--------|
| High dimension (≥ 4) | Langevin ensemble — FP PDE discretization scales as N^d |
| Want stationary distribution on a 1D/2D potential | Eigenproblem of `L†` — no time-to-equilibrium wait |
| Rare-event tails matter | Langevin + forward-flux sampling (see `advanced-simulations`) |
| Committor / reaction-coordinate PDE | FP PDE with Dirichlet BCs on the two basins; DOLFINx is the right tool |
| Drift is learned from data | Langevin ensemble (grad via autodiff); see `bayesian-ude-workflow` for uncertainty

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
- **Jump-diffusion / PDMP** — General physics SDEs with embedded jump events (stick-slip, shot noise, Lévy flights, Markov-switching Langevin) are handled here using `JumpProcesses.jl` or `DiffEqJump` alongside a continuous Langevin drift. For **biochemical reaction networks** specifically (mass-action kinetics, Gillespie SSA on a species vector), route to `catalyst-reactions` — Catalyst.jl builds the jump problem from a symbolic `@reaction_network`. Use this skill when the jumps are physical (phase slips, pinning, regime switches) rather than chemical.

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
