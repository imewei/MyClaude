---
name: sciml-modern-stack
description: Modern Julia SciML stack with Lux.jl neural networks, SciMLSensitivity.jl adjoint/forward sensitivity, Universal Differential Equations (UDEs), UncertaintyQuantification.jl, and DeepEquilibriumNetworks.jl. Use when building neural ODEs, performing sensitivity analysis, quantifying uncertainty, or combining physics with ML in Julia SciML workflows.
---

# Modern SciML Stack

> **⚠️ FROZEN (2026-04-02):** This skill is at 78% context budget. Do not add new content — create separate skills for new SciML topics instead.

Build scientific machine learning workflows with Lux.jl explicit-parameter neural networks and SciMLSensitivity.jl automatic differentiation through differential equation solvers. This stack replaces Flux-based SciML patterns with composable, AD-friendly components.

---

## Expert Agent

Delegate to **julia-pro** for implementation, debugging, and architecture decisions involving this stack.

---

## Lux.jl vs Flux.jl

| Feature | Lux.jl | Flux.jl |
|---------|--------|---------|
| Parameterization | Explicit `(ps, st)` tuple | Implicit (params in model struct) |
| SciML compatibility | First-class (designed for it) | Legacy support only |
| Gradient support | Zygote, Enzyme, ForwardDiff | Zygote only |
| State management | Explicit stateful layers `(y, st_new) = model(x, ps, st)` | Implicit mutable state |
| Composability | Functional — pure functions | Object-oriented |

> **Rule:** Always use Lux.jl for new SciML work. Flux.jl is legacy in the SciML ecosystem.

---

## Lux.jl Fundamentals

### Model Definition

```julia
using Lux, Random

model = Chain(
    Dense(2 => 32, tanh),
    Dense(32 => 32, tanh),
    Dense(32 => 2)
)

rng = Random.default_rng()
Random.seed!(rng, 42)
ps, st = Lux.setup(rng, model)
```

### Forward Pass

```julia
x = rand(Float32, 2, 64)  # (features, batch)
y, st_new = model(x, ps, st)
```

### Training Loop

```julia
using Optimisers, Zygote

opt_state = Optimisers.setup(Adam(1e-3), ps)

for epoch in 1:1000
    (loss, st), pb = Zygote.pullback(ps) do p
        y_pred, st_ = model(x_train, p, st)
        return sum(abs2, y_pred .- y_target), st_
    end
    gs = pb((one(loss), nothing))[1]
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
end
```

---

## Universal Differential Equations (UDEs)

UDEs combine known physics (conservation laws, established mechanistic terms) with neural networks that learn unknown or poorly understood terms directly from data. The general form:

```
du/dt = f_known(u, p, t) + NN(u, p_nn)
```

The neural network acts as a universal function approximator for the missing physics, while the known terms enforce structure, conservation laws, and dimensional consistency.

### Basic UDE Example: Lotka-Volterra with Unknown Interaction

Replace the unknown predator-prey interaction with a neural network:

```julia
using Lux, DifferentialEquations, SciMLSensitivity, Random, ComponentArrays

# Neural network for unknown interaction term
nn = Chain(Dense(2 => 32, tanh), Dense(32 => 32, tanh), Dense(32 => 2))
rng = Random.default_rng()
Random.seed!(rng, 42)
ps_nn, st_nn = Lux.setup(rng, nn)
p_nn = ComponentArray(ps_nn)

# UDE: known structure + neural network unknown
function ude!(du, u, p, t)
    x, y = u
    # Known: population growth and decay
    du[1] = 1.5 * x    # known growth rate
    du[2] = -3.0 * y   # known decay rate
    # Unknown interaction: learned by NN
    nn_out, _ = nn([x, y], p, st_nn)
    du[1] += nn_out[1]
    du[2] += nn_out[2]
end

u0 = Float32[1.0, 1.0]
tspan = (0.0f0, 10.0f0)
prob = ODEProblem(ude!, u0, tspan, p_nn)
```

### UDE Training Pattern: Two-Phase Optimization

Use ADAM for rough convergence, then switch to BFGS/L-BFGS for fine convergence:

```julia
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Generate or load training data
t_data = 0.0f0:0.5f0:10.0f0
u_data = # observed trajectory data (2 x N_timepoints)

function loss(p, _)
    sol = solve(
        remake(prob, p=p), Tsit5(),
        saveat=t_data,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())
    )
    if sol.retcode !== ReturnCode.Success
        return Inf32
    end
    return sum(abs2, Array(sol) .- u_data)
end

# Phase 1: ADAM (rough convergence)
opt_func = OptimizationFunction(loss, AutoZygote())
opt_prob = OptimizationProblem(opt_func, p_nn)
result1 = solve(opt_prob, OptimizationOptimisers.Adam(1e-3); maxiters=5000)

# Phase 2: BFGS (fine convergence)
opt_prob2 = OptimizationProblem(opt_func, result1.u)
result2 = solve(opt_prob2, OptimizationOptimJL.BFGS(); maxiters=2000)

p_trained = result2.u
```

### UDE for Discovery: UDE + SINDy Pipeline

After training a UDE, extract symbolic equations from the trained neural network using Sparse Identification of Nonlinear Dynamics (SINDy):

```julia
using DataDrivenDiffEq, DataDrivenSparse

# Step 1: Generate NN predictions on a grid
u_grid = # grid of state-space points
nn_predictions = [nn([u...], p_trained, st_nn)[1] for u in eachcol(u_grid)]

# Step 2: Build SINDy problem
@variables x y
basis = Basis(polynomial_basis([x, y], 3), [x, y])  # polynomial library up to degree 3

# Step 3: Sparse regression to find symbolic form
problem = DirectDataDrivenProblem(u_grid, hcat(nn_predictions...))
result = solve(problem, basis, STLSQ(1e-2))  # Sequentially Thresholded Least Squares

# Step 4: Extract discovered equations
println(result.basis)  # e.g., discovers -1.0*x*y and +1.0*x*y
```

This pipeline: (1) train UDE on data, (2) evaluate trained NN on state-space grid, (3) apply SINDy to find symbolic expression that the NN learned.

### UDE Design Patterns

| Pattern | When to Use | Example |
|---------|------------|---------|
| Additive correction | Known dynamics + unknown perturbation | `du/dt = f(u) + NN(u)` |
| Multiplicative correction | Unknown scaling of known physics | `du/dt = NN(u) * f(u)` |
| Missing physics | Known structure, unknown functional form | `du/dt = -k*u + NN(u,v)` |
| Hybrid closure | Coarse-grained model with learned closure | `du/dt = f(u) + NN(u; subgrid)` |

### UDE Practical Tips

- **Network size:** Start small (2-3 hidden layers, 32-64 neurons each). Larger networks overfit noise and slow training.
- **Regularization:** Add weight decay (`L2` penalty on `p_nn`) to prevent overfitting noisy data: `loss += 1e-4 * sum(abs2, p)`.
- **Multiple trajectories:** Train on data from multiple initial conditions to improve generalization.
- **Initial conditions:** Include IC as part of the optimization if uncertain: `u0_learned = p[end-1:end]`.
- **Stiff UDEs:** Use implicit solvers (`TRBDF2`, `Rodas5P`) with `InterpolatingAdjoint` sensitivity. Avoid `BacksolveAdjoint` for stiff systems (numerical instability).
- **Float32:** Use `Float32` throughout for faster training; promote to `Float64` only for final validation.

### UDE Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| NN too large | Loss drops fast then plateaus; poor generalization | Reduce to 2 layers, 32 neurons; add L2 regularization |
| Wrong sensitivity algorithm | `NaN` gradients or excessive memory | Use `InterpolatingAdjoint(autojacvec=ZygoteVJP())` |
| No regularization | Overfits noise; non-physical NN output | Add weight decay `1e-4 * sum(abs2, p)` to loss |
| Single trajectory training | NN memorizes one IC; fails on others | Train on 5-10 trajectories from different ICs |
| Ignoring physical constraints | NN violates conservation laws | Encode constraints in architecture (e.g., skew-symmetric Jacobian) |
| Solver failures during training | `Inf` loss, `retcode != Success` | Add `retcode` check; reduce `dt` or switch solver |

---

## SciMLSensitivity.jl

Compute gradients through differential equation solvers for parameter estimation, neural ODEs, and UDE training.

### Algorithm Selection

| Algorithm | Parameters | Time | Memory | Use Case |
|-----------|-----------|------|--------|----------|
| `ForwardDiffSensitivity()` | < 100 | O(p * T) | Low | Few parameters, need full Jacobian |
| `InterpolatingAdjoint()` | Any | O(T) | O(T) (checkpoints) | Default for neural ODEs / UDEs |
| `BacksolveAdjoint()` | Any | O(T) | O(1) | Non-stiff, memory-constrained |
| `QuadratureAdjoint()` | Any | O(T) | O(T) | High-accuracy gradient required |
| `ForwardSensitivity()` | < 50 | O(p * T) | Low | Forward-mode, small parameter count |

### Usage Examples

```julia
using SciMLSensitivity

# InterpolatingAdjoint (recommended default for UDEs)
sol = solve(prob, Tsit5(),
    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    saveat=0.1)

# ForwardDiffSensitivity (few parameters)
sol = solve(prob, Tsit5(),
    sensealg=ForwardDiffSensitivity(),
    saveat=0.1)

# BacksolveAdjoint (memory-constrained, non-stiff only)
sol = solve(prob, Tsit5(),
    sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()),
    saveat=0.1)

# QuadratureAdjoint (high-accuracy gradients)
sol = solve(prob, Tsit5(),
    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
    saveat=0.1)

# ForwardSensitivity (explicit forward sensitivity equations)
sol = solve(prob, Tsit5(),
    sensealg=ForwardSensitivity(),
    saveat=0.1)
```

---

## UncertaintyQuantification.jl

Quantify parametric uncertainty via polynomial chaos expansions and global sensitivity analysis.

### Polynomial Chaos Expansion

```julia
using UncertaintyQuantification

# Define uncertain parameters
p1 = RandomVariable(Uniform(0.8, 1.2), :k1)
p2 = RandomVariable(Normal(1.0, 0.1), :k2)

# Model function (wraps ODE solve)
function model_response(inputs)
    p = [inputs[:k1], inputs[:k2]]
    sol = solve(remake(prob, p=p), Tsit5(), saveat=1.0)
    return sol[end][1]  # final state
end

# Polynomial chaos expansion
pce = PolynomialChaosExpansion([p1, p2], 4)  # degree 4
sample!(pce, model_response)

# Extract statistics
mean_val = mean(pce)
std_val = std(pce)
```

### Sobol Sensitivity Indices

```julia
# Global sensitivity via Sobol indices
sobol = SobolIndices([p1, p2], model_response, 10_000)

# First-order indices (main effects)
println("S1: ", sobol.first_order)

# Total-order indices (includes interactions)
println("ST: ", sobol.total_order)
```

---

## DeepEquilibriumNetworks.jl

Implicit layers that solve for fixed points instead of stacking explicit layers. Memory-efficient (O(1) in depth) via implicit differentiation.

```julia
using DeepEquilibriumNetworks, Lux, Random

# Define the implicit layer
model = Chain(
    Dense(2 => 32),
    DeepEquilibriumNetwork(
        Chain(Dense(32 => 32, tanh), Dense(32 => 32)),
        NewtonRaphson();           # or Broyden() for Jacobian-free
        maxiters=50,
        abstol=1e-6
    ),
    Dense(32 => 2)
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

x = rand(Float32, 2, 16)
y, st_new = model(x, ps, st)
```

**Solver choice:** Use `NewtonRaphson()` for small layers (exact Jacobian), `Broyden()` for large layers (Jacobian-free, lower memory).

---

## NeuralPDE.jl v5+

Physics-informed neural networks (PINNs) with ModelingToolkit symbolic PDE definitions.

```julia
using NeuralPDE, ModelingToolkit, Lux, DomainSets
import ModelingToolkit: Interval

@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Heat equation: du/dt = α * d²u/dx²
α = 0.1
eq = Dt(u(t, x)) ~ α * Dxx(u(t, x))

# Boundary and initial conditions
bcs = [
    u(0, x) ~ sin(π * x),          # IC
    u(t, 0) ~ 0.0,                  # BC left
    u(t, 1) ~ 0.0                   # BC right
]

domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]

# Neural network discretization
chain = Chain(Dense(2 => 32, tanh), Dense(32 => 32, tanh), Dense(32 => 1))
discretization = PhysicsInformedNN(chain, QuadratureTraining())

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)
sol = solve(prob, OptimizationOptimisers.Adam(1e-3); maxiters=5000)
```

---

## Decision Framework

```
Need neural network inside ODE/SDE?
  → UDE (this skill: UDE section)

Need gradients through a solver?
  → SciMLSensitivity.jl (algorithm selection table above)

Need parameter uncertainty quantification?
  → UncertaintyQuantification.jl (PCE + Sobol)

Need implicit/fixed-point layer?
  → DeepEquilibriumNetworks.jl (DEQ)

Need to solve PDE with neural network?
  → NeuralPDE.jl (PINN with ModelingToolkit)

Have known physics + unknown terms?
  → UDE with additive/multiplicative correction pattern

Want to discover equations from trained NN?
  → UDE + SINDy pipeline (DataDrivenDiffEq.jl)
```

## Checklist

- [ ] Verify Lux.jl is used instead of Flux.jl for all new SciML work
- [ ] Confirm explicit parameterization: `(ps, st)` tuple is passed to every model call
- [ ] Check UDE neural network size is small (2-3 layers, 32-64 neurons) to avoid overfitting
- [ ] Ensure sensitivity algorithm matches problem type: `InterpolatingAdjoint` for UDEs, `ForwardDiffSensitivity` for <100 parameters
- [ ] Validate UDE training on multiple trajectories from different initial conditions
- [ ] Add L2 regularization (`1e-4 * sum(abs2, p)`) to UDE loss to prevent overfitting noise
- [ ] Confirm solver retcode check (`ReturnCode.Success`) in UDE loss function
- [ ] Use two-phase optimization: ADAM for rough convergence, then BFGS for fine convergence
- [ ] Verify SINDy extraction from trained NN produces physically interpretable equations
