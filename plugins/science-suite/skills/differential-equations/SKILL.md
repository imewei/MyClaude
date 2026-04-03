---
name: differential-equations
maturity: "5-Expert"
specialization: Julia DiffEq
description: Solve ODE/SDE/PDE with DifferentialEquations.jl. Use when defining differential equation systems, selecting solvers, implementing callbacks, or creating ensemble simulations. Also use when choosing between Tsit5/Vern7/Rodas5 for stiff vs non-stiff problems, adding event handling, running Monte Carlo parameter sweeps, or integrating with sensitivity analysis. Use proactively when the user mentions solving differential equations in Julia, time integration, or dynamical systems simulation, even without naming DifferentialEquations.jl.
---

# DifferentialEquations.jl

## Expert Agent

For ODE/SDE/PDE solving with DifferentialEquations.jl and the SciML ecosystem, delegate to:

- **`julia-pro`**: Julia SciML ecosystem, solver selection, and differential equation workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

Solve differential equations with the SciML ecosystem.

---

## ODE Template

```julia
using DifferentialEquations

function my_ode!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]

prob = ODEProblem(my_ode!, u0, tspan, p)
sol = solve(prob, Tsit5())
```

---

## Solver Selection

| Solver | Use Case |
|--------|----------|
| Tsit5() | General-purpose (default) |
| Vern7() | High accuracy |
| TRBDF2() | Stiff problems |
| Rodas5() | Stiff + high accuracy |

---

## Callbacks

```julia
condition(u, t, integrator) = u[1] < 0.1
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

cb_periodic = PeriodicCallback(t -> nothing, 1.0)

sol = solve(prob, Tsit5(), callback=CallbackSet(cb, cb_periodic))
```

---

## Ensemble Simulations

```julia
function prob_func(prob, i, repeat)
    remake(prob, u0 = rand(2))
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=100)
```

---

## Checklist

- [ ] Problem type matches equations (ODE/SDE/DAE)
- [ ] Appropriate solver selected
- [ ] Callbacks configured if needed
- [ ] Solution accuracy validated

---

## Continuation Methods

Use [BifurcationKit.jl](https://github.com/bifurcationkit/BifurcationKit.jl) for parameter continuation and bifurcation analysis. See the `bifurcation-analysis` skill for advanced workflows.

```julia
using BifurcationKit

F(u, p) = @. u^3 - p.mu * u  # steady-state equation

u0 = zeros(1)
params = (mu = 0.0,)

prob = BifurcationProblem(F, u0, params, (@optic _.mu))
opts = ContinuationPar(p_min = -1.0, p_max = 2.0, ds = 0.01)
br = continuation(prob, PALC(), opts)
```

---

## Sensitivity Analysis

Choose forward or adjoint sensitivity based on problem characteristics:

| Criterion | ForwardDiffSensitivity | InterpolatingAdjoint |
|-----------|----------------------|---------------------|
| Parameters | < 100 | > 100 |
| Time span | Short | Long |
| Memory | O(N x p) | O(N) |
| Use case | Few-param models | Neural ODEs, UDEs |

**Forward sensitivity** — best for few parameters:

```julia
using SciMLSensitivity

sol = solve(prob, Tsit5(), sensealg=ForwardDiffSensitivity())
```

**Adjoint sensitivity** — best for many parameters (neural ODEs, UDEs):

```julia
sol = solve(prob, Tsit5(),
    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
```

---

## Stiff System Patterns

**Auto stiffness detection** — switches between non-stiff and stiff solvers automatically:

```julia
sol = solve(prob, AutoTsit5(Rosenbrock23()))
```

**Jacobian sparsity** — exploit structure for large stiff systems:

```julia
using SparseDiffTools, SparseArrays

jac_sparsity = Symbolics.jacobian_sparsity(my_ode!, similar(u0), u0, p, 0.0)
f = ODEFunction(my_ode!; jac_prototype=float.(jac_sparsity))
prob_sparse = ODEProblem(f, u0, tspan, p)
sol = solve(prob_sparse, Rosenbrock23())
```

---

## Event Handling

**Continuous callback** — Poincare section (detect zero-crossing of `u[1]`):

```julia
condition(u, t, integrator) = u[1]  # triggers when u[1] crosses zero
affect!(integrator) = nothing        # record state, no modification

cb = ContinuousCallback(condition, affect!)
sol = solve(prob, Tsit5(), callback=cb, save_everystep=false)
```

**Threshold crossing with termination**:

```julia
condition(u, t, integrator) = u[2] - 1e-6  # stop when u[2] hits threshold
affect!(integrator) = terminate!(integrator)

cb = ContinuousCallback(condition, affect!)
sol = solve(prob, Tsit5(), callback=cb)
```

---

**Version**: 1.0.5
