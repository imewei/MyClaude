---
name: optimization-patterns
description: Optimization.jl usage for SciML parameter estimation and loss minimization. Distinct from JuMP.jl mathematical programming to avoid conflicts.
---

# Optimization Patterns (Optimization.jl)

Use Optimization.jl for SciML parameter estimation. For mathematical programming, use JuMP.jl with julia-pro.

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
