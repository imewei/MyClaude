---
name: neural-pde
maturity: "5-Expert"
specialization: Physics-Informed Neural Networks
description: Solve PDEs with NeuralPDE.jl using physics-informed neural networks (PINNs) and method-of-lines discretization. Use when solving forward or inverse PDE problems with neural networks, enforcing boundary conditions, training neural surrogates for expensive simulations, or applying physics-informed loss functions. Use proactively when the user mentions PINNs, physics-informed learning, neural operators, or wants to solve high-dimensional or irregular-domain PDEs where mesh-based methods fail. For credible intervals on PINN solutions (BPINN / BNNODE), see the sibling `bayesian-pinn` skill.
---

# NeuralPDE.jl — PINNs (deterministic)

## Expert Agent

For physics-informed neural networks and PDE solving with NeuralPDE.jl, delegate to:

- **`julia-pro`**: Julia SciML ecosystem and neural PDE workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

NeuralPDE.jl couples symbolic PDE specification (via ModelingToolkit) with neural-network ansätze, supporting forward solves, inverse problems, and Bayesian inference over network weights.

---

## When PINNs vs MOL vs classical solvers

| Situation | Use |
|-----------|-----|
| Regular grid, low-to-moderate dimension, smooth solution | **MethodOfLines** (MOL): symbolic PDE → ODE system → standard DiffEq solver. Fastest, most accurate for routine problems. |
| High-dimensional PDE (≥ 4 spatial dims), curse of dimensionality | **PINN**: neural ansatz scales without a mesh |
| Irregular / time-varying / unknown domain | **PINN**: collocation points work where meshing is hard |
| Inverse problem (unknown PDE coefficient) | **PINN**: parameters and weights co-train against data + physics residual |
| Need uncertainty over the solution | See `bayesian-pinn` — BPINN / BNNODE with internal HMC sampler |

PINNs are not strictly better than mesh methods. Reach for them when the mesh is the bottleneck.

---

## Symbolic PDE specification

NeuralPDE consumes a `PDESystem` built with ModelingToolkit symbolic primitives:

```julia
using NeuralPDE, Lux, ModelingToolkit, DomainSets, IntervalSets, Random
import ModelingToolkit: Interval

@parameters t x
@variables u(..)
Dt  = Differential(t)
Dxx = Differential(x)^2

# 1D heat equation
eq  = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cos(π * x),
       u(t, 0) ~ 0,
       u(t, 1) ~ 0]

domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
```

`@parameters`, `@variables`, and `Differential` come from ModelingToolkit. See `modeling-toolkit` for the symbolic-numeric pipeline that backs this.

---

## Lux neural ansatz

NeuralPDE uses Lux's explicit-parameter style — parameters are a separate object passed alongside the model, AD-friendly and consistent with the rest of the Julia DL ecosystem:

```julia
chain = Chain(Dense(2, 16, σ),
              Dense(16, 16, σ),
              Dense(16, 1))

rng = Random.default_rng()
ps_init, st = Lux.setup(rng, chain)
```

For systems of PDEs, pass a `Vector` of chains — one per dependent variable.

---

## Discretization and training

The `PhysicsInformedNN` discretizer turns a `PDESystem` into an `OptimizationProblem`. Choose the collocation strategy explicitly:

```julia
using QuasiMonteCarlo, Optimization, OptimizationOptimisers

strategy = QuasiRandomTraining(256;                # 256 collocation points
                               sampling_alg = SobolSample())
discretization = PhysicsInformedNN(chain, strategy)

prob = discretize(pde_system, discretization)
result = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 5000)
```

Strategy options:

| Strategy | When |
|----------|------|
| `GridTraining(dx)` | Low dimensions, regular spacing |
| `StochasticTraining(n)` | Uniform random collocation |
| `QuasiRandomTraining(n; sampling_alg = SobolSample())` | **Default**. Quasi-MC (low-discrepancy) — faster convergence than uniform random |
| `QuadratureTraining()` | Smooth solutions, integral-form residual |

`QuasiMonteCarlo`, `DomainSets`, and `IntervalSets` together give the spatial-domain API: define the domain shape, then sample collocation points without a mesh.

---

## Method of Lines (MethodOfLines.jl) — when not to use a PINN

For classical PDE problems on regular domains, `MethodOfLines` discretizes the same `PDESystem` to an ODE system that any DiffEq solver can integrate:

```julia
using MethodOfLines, OrdinaryDiffEq

dx = 0.05
discretization = MOLFiniteDifference(
    [x => dx], t;
    approx_order    = 2,                # finite-difference order
    advection_scheme = UpwindScheme(),  # or WENOScheme() for shocks
    grid_align       = CenterAlignedGrid(),
)
prob = discretize(pde_system, discretization)
sol  = solve(prob, Tsit5(), saveat = 0.1)
```

This is much faster and more accurate than a PINN for routine 1D/2D problems. Use PINN only when MOL would require an impractical mesh. **For advection-dominated or hyperbolic PDEs**, switch the `advection_scheme` to `WENOScheme()` — vanilla central differences will introduce spurious oscillations near sharp fronts.

---

## Forward vs inverse problems

**Forward** — fit neural ansatz to satisfy a fully specified PDE. The pattern above is forward.

**Inverse** — some PDE coefficients are unknown; co-train them with the network weights against observation data plus physics residual:

```julia
@parameters α                                  # unknown diffusion coefficient
eq = Dt(u(t, x)) ~ α * Dxx(u(t, x))

discretization = PhysicsInformedNN(
    chain,
    strategy;
    additional_loss = (phi, θ, p) -> sum(abs2, phi(t_obs, θ) .- u_obs),  # data term
    param_estim = true,
    init_params = ComponentArray(α = 0.1)
)
```

`additional_loss` adds a data-misfit term; `param_estim = true` exposes the symbolic parameters as trainable.

---

## Bayesian PINN — see sibling skill

NeuralPDE's internal AdvancedHMC-based Bayesian engine (`BNNODE` for ODEs, `BayesianPINN` for PDEs) lives in the dedicated **[bayesian-pinn](../bayesian-pinn/SKILL.md)** skill. Reach for it when you need credible intervals on a PINN solution, an inverse problem with uncertainty on physical parameters, or an HMC escape path through a multimodal weight posterior via Pigeons.

The split exists because BPINN content grew to the point where this parent `neural-pde` skill was approaching its context budget. Keeping deterministic and Bayesian PINN work in separate skills makes the discoverability cleaner for both and leaves room in each to grow.

---

## Python / JAX counterpart stack

NeuralPDE.jl has no drop-in Python equivalent; JAX has no MTK-level symbolic PDE layer. The JAX-first path is **hand-rolled residuals** on Equinox + Optax + Diffrax, supported by:

| Role | Package | Key API |
|------|---------|---------|
| QMC collocation (analog of Julia's `SobolSample`) | **`scipy.stats.qmc`** | `Sobol`, `Halton`, `LatinHypercube`, `PoissonDisk`, `discrepancy`, `scale` |
| Polynomial chaos + Sobol sensitivity | **`chaospy`** | `generate_expansion` (Hermite/Legendre/Laguerre/Jacobi), `generate_quadrature` (Gauss/Clenshaw-Curtis/Leja/Smolyak), `fit_quadrature`, `E`, `Std`, `Sens_m`, `Sens_t` |
| SymPy → JAX bridge | **`sympy2jax`** (Kidger fork) | `SymbolicModule` — wraps SymPy expressions as Equinox modules with trainable numeric leaves. Expression-only, no PDE/DAE machinery |
| Numerical PDE baseline for convergence checks | **`py-pde`** | See `numerical-methods-implementation` — NumPy/Numba-based reference solver |

### Minimal pattern: hand-rolled PINN residual in JAX

```python
import jax, jax.numpy as jnp
from scipy.stats import qmc

# QMC collocation (direct analog to Julia's SobolSample)
X_colloc = jnp.asarray(qmc.Sobol(d=2, scramble=True, seed=0).random_base2(m=8))

# 1D heat residual u_t = u_xx via nested grad
def residual(model, tx):
    u_t  = jax.grad(lambda p: model(jnp.array([p, tx[1]])))(tx[0])
    u_xx = jax.grad(jax.grad(lambda p: model(jnp.array([tx[0], p]))))(tx[1])
    return u_t - u_xx

def loss_fn(model, X):
    return jnp.mean(jax.vmap(lambda tx: residual(model, tx) ** 2)(X))
```

Wrap with an `eqx.Module` MLP + Optax Adam loop; see `jax-diffeq-pro` for the surrounding boilerplate.

### Sobol sensitivity on PINN output via `chaospy`

```python
import chaospy as cp
dist      = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0.5, 2.0))
nodes, w  = cp.generate_quadrature(order=4, dist=dist, rule="G")
expansion = cp.generate_expansion(order=4, dist=dist)
fitted    = cp.fit_quadrature(expansion, nodes, w, pinn_evaluate(nodes))
Sobol_total = cp.Sens_t(fitted, dist)      # total-effect indices
Sobol_first = cp.Sens_m(fitted, dist)      # first-order indices
```

> **Stay in Julia / NeuralPDE.jl** when you want `QuasiRandomTraining` + `SobolSample` in one line and the surrounding stack is already Julia. **Drop to the Python / JAX hand-rolled stack** when the surrounding training loop is already JAX, or when tight integration with Diffrax / Equinox / NumPyro matters more than MTK-level symbolic modeling.

---

## Composition with neighboring skills

- **Bayesian PINN** — credible intervals on PINN solutions via `BNNODE` / `BayesianPINN`. See `bayesian-pinn`.
- **Modeling toolkit** — symbolic PDE primitives (`@parameters`, `@variables`, `Differential`). See `modeling-toolkit`.
- **Differential equations** — solver selection for the MOL path. See `differential-equations`.
- **SciML modern stack** — sensealg and SciMLBase interfaces. See `sciml-modern-stack`.
- **Bayesian UDE workflow** — hybrid physics + NN with Turing. See `bayesian-ude-workflow`.

---

## Checklist

- [ ] Confirmed PINN is the right tool — would MOL work on a reasonable mesh?
- [ ] Defined the PDE symbolically with ModelingToolkit (`@parameters`, `@variables`, `Differential`)
- [ ] Specified all boundary and initial conditions explicitly
- [ ] Built a Lux chain with explicit `(ps, st)` setup, not implicit Flux parameters
- [ ] Picked a collocation strategy — `QuasiRandomTraining` with Sobol is the safe default
- [ ] Validated training loss includes both physics residual and (if inverse) data terms
- [ ] Compared the trained PINN against an MOL reference on a simplified test problem
- [ ] If posterior uncertainty is needed, switched to the `bayesian-pinn` skill
