---
name: differential-equations
description: Master solving ODE, PDE, SDE, and DAE with DifferentialEquations.jl for scientific simulations and numerical analysis. Use when defining differential equation systems (.jl files with ODEProblem, SDEProblem, PDEProblem, DAEProblem), selecting appropriate solvers (Tsit5, Vern7, Rodas5, TRBDF2 for stiff problems), implementing callbacks (ContinuousCallback, DiscreteCallback, PeriodicCallback for events), creating ensemble simulations (EnsembleProblem with EnsembleThreads, EnsembleGPUArray), performing sensitivity analysis, handling stiff vs non-stiff problems, working with in-place vs out-of-place formulations, or integrating with ModelingToolkit.jl. Foundation for /sciml-setup command templates and essential for all scientific computing with differential equations in Julia.
---

# Differential Equations

Master solving differential equations in Julia with the DifferentialEquations.jl ecosystem.

## When to use this skill

- Defining ODE, PDE, SDE, or DAE systems in Julia (.jl files)
- Creating ODEProblem, SDEProblem, PDEProblem, DAEProblem instances
- Selecting appropriate solvers (Tsit5, Vern7, Rodas5, KenCarp4, TRBDF2)
- Implementing callbacks for event handling (PeriodicCallback, ContinuousCallback)
- Setting up ensemble simulations with EnsembleProblem
- Performing parameter studies and sensitivity analysis
- Handling stiff vs non-stiff differential equations
- Working with in-place (du, u, p, t) vs out-of-place formulations
- Integrating with ModelingToolkit.jl for symbolic equations
- Solving systems from physics, biology, chemistry, or engineering
- Running Monte Carlo simulations with stochastic differential equations

## Core Patterns

### ODE Template
```julia
using DifferentialEquations

function my_ode!(du, u, p, t)
    # Define derivatives
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]

prob = ODEProblem(my_ode!, u0, tspan, p)
sol = solve(prob, Tsit5())  # Recommended solver
```

### Solver Selection
- **Tsit5()**: General-purpose, good default (Runge-Kutta 5/4)
- **Vern7()**: High accuracy requirements
- **TRBDF2()**: Stiff problems
- **Rodas5()**: Stiff with high accuracy
- **KenCarp4()**: Moderately stiff

### Callbacks
```julia
# Terminate when condition met
condition(u, t, integrator) = u[1] < 0.1
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

# Periodic events
affect_periodic!(integrator) = integrator.u[1] *= 0.9
cb_periodic = PeriodicCallback(affect_periodic!, 1.0)

sol = solve(prob, Tsit5(), callback=CallbackSet(cb, cb_periodic))
```

### Ensemble Simulations
```julia
function prob_func(prob, i, repeat)
    remake(prob, u0 = rand(2))
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=100)
```

## Resources
- **DifferentialEquations.jl**: https://docs.sciml.ai/DiffEqDocs/stable/
