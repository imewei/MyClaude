---
name: optimization-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: SciML Optimization
description: Use Optimization.jl for parameter estimation in differential equations. Use when fitting models to data or solving inverse problems. For LP/QP/MIP, use JuMP.jl instead.
---

# Optimization.jl Patterns

Parameter estimation for SciML workflows.

---

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

---

## Optimization.jl vs JuMP.jl

| Use | Package |
|-----|---------|
| SciML parameter estimation | Optimization.jl |
| Model fitting to data | Optimization.jl |
| LP, QP, MIP, MILP | JuMP.jl |
| Mathematical programming | JuMP.jl |

---

## Algorithms

| Algorithm | Use Case |
|-----------|----------|
| BFGS | Smooth, gradient-based |
| Adam | Neural network training |
| NelderMead | Derivative-free |
| PSO | Global optimization |

---

## Checklist

- [ ] Loss function defined
- [ ] Initial parameters set
- [ ] Appropriate algorithm selected
- [ ] Result validated against data

---

**Version**: 1.0.5
