---
name: differential-equations
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia DiffEq
description: Solve ODE/SDE/PDE with DifferentialEquations.jl. Use when defining differential equation systems, selecting solvers, implementing callbacks, or creating ensemble simulations.
---

# DifferentialEquations.jl

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

**Version**: 1.0.5
