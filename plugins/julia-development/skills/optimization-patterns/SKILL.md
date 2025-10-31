---
name: optimization-patterns
description: Master Optimization.jl for SciML parameter estimation, loss minimization, and differential equation parameter fitting. Use when estimating parameters in differential equations (.jl files with OptimizationProblem), minimizing loss functions for SciML workflows, fitting ODE/SDE models to data, working with optimization algorithms (BFGS, Adam, NelderMead, PSO), integrating with DifferentialEquations.jl for parameter estimation, performing inverse problems in scientific computing, using automatic differentiation for gradients (ForwardDiff, ReverseDiff, Zygote), or running optimization with SciMLSensitivity.jl. Distinct from JuMP.jl (mathematical programming) - use Optimization.jl for SciML workflows and JuMP.jl for LP/QP/MIP.
---

# Optimization Patterns (Optimization.jl)

Use Optimization.jl for SciML parameter estimation. For mathematical programming, use JuMP.jl with julia-pro.

## When to use this skill

- Estimating parameters in differential equations (ODE, SDE, PDE)
- Minimizing loss functions for SciML model fitting
- Fitting mathematical models to experimental data
- Working with optimization algorithms (BFGS, Adam, NelderMead, PSO, etc.)
- Integrating optimization with DifferentialEquations.jl
- Performing inverse problems in scientific computing
- Using automatic differentiation for gradient computation
- Running parameter estimation with SciMLSensitivity.jl
- Optimizing neural network parameters in scientific models
- Choosing between gradient-based vs derivative-free optimizers
- Working with constrained optimization in SciML context

## Parameter Estimation Pattern
```julia
using Optimization, OptimizationOptimJL

function loss(p, data)
    prob_p = remake(prob, p=p)
    sol = solve(prob_p, Tsit5(), saveat=0.1)
    return sum(abs2, sol[1, :] .- data)
end

opt_prob = OptimizationProblem(loss, p_init, measured_data)
result = solve(opt_prob, BFGS())
```

## Resources
- **Optimization.jl**: https://docs.sciml.ai/Optimization/stable/
