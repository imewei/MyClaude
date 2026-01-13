---
name: jump-optimization
version: "1.0.7"
maturity: "5-Expert"
specialization: Mathematical Programming
description: Master JuMP.jl for LP, QP, NLP, and MIP with HiGHS, Ipopt, and commercial solvers. Use for production planning, portfolio optimization, scheduling, and constrained optimization. Note that JuMP.jl is separate from Optimization.jl (sciml-pro).
---

# JuMP Optimization

Mathematical programming with JuMP.jl for LP, QP, NLP, and MIP problems.

---

## Problem Type Selection

| Type | Solver | Use Case |
|------|--------|----------|
| LP | HiGHS, GLPK | Production planning, resource allocation |
| QP | Ipopt, COSMO | Portfolio optimization, least squares |
| NLP | Ipopt, KNITRO | General constrained optimization |
| MIP | HiGHS, Gurobi | Scheduling, routing, assignment |
| Conic | COSMO, Mosek | SDP, SOCP problems |

**NOT for**: SciML parameter estimation (use Optimization.jl), neural network training (use Flux.jl)

---

## Linear Programming

```julia
using JuMP, HiGHS

function production_optimization()
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x₁ >= 0)
    @variable(model, x₂ >= 0)
    @objective(model, Max, 40x₁ + 30x₂)

    @constraint(model, labor, 2x₁ + x₂ <= 100)
    @constraint(model, material, x₁ + 2x₂ <= 80)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return (x₁=value(x₁), x₂=value(x₂), profit=objective_value(model),
                shadow_prices=(labor=dual(labor), material=dual(material)))
    end
end
```

---

## Quadratic Programming

```julia
using JuMP, Ipopt

function portfolio_optimization(μ, Σ, target_return)
    n = length(μ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, 0 <= w[1:n] <= 1)
    @objective(model, Min, w' * Σ * w)       # Minimize variance
    @constraint(model, sum(w) == 1)           # Weights sum to 1
    @constraint(model, w' * μ >= target_return)

    optimize!(model)
    return value.(w)
end
```

---

## Nonlinear Programming

```julia
using JuMP, Ipopt

function constrained_nlp()
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @objective(model, Min, (x - 3)^2 + (y - 2)^2)
    @constraint(model, x^2 + y^2 >= 1)

    set_start_value(x, 1.0)  # Warm start
    set_start_value(y, 1.0)

    optimize!(model)
    return (x=value(x), y=value(y))
end
```

---

## Mixed-Integer Programming

```julia
using JuMP, HiGHS

function knapsack(values, weights, capacity)
    n = length(values)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:n], Bin)
    @objective(model, Max, sum(values[i] * x[i] for i in 1:n))
    @constraint(model, sum(weights[i] * x[i] for i in 1:n) <= capacity)

    optimize!(model)
    return [i for i in 1:n if value(x[i]) > 0.5]
end
```

---

## Advanced Patterns

### Warm Starting

```julia
function solve_sequence(problems)
    model = Model(Ipopt.Optimizer)
    @variable(model, x >= 0)
    @variable(model, y >= 0)

    results = []
    for (i, p) in enumerate(problems)
        @objective(model, Min, (x - p.tx)^2 + (y - p.ty)^2)
        if i > 1
            set_start_value(x, results[end].x)
            set_start_value(y, results[end].y)
        end
        optimize!(model)
        push!(results, (x=value(x), y=value(y)))
    end
    return results
end
```

### Fixing Variables

```julia
model = Model(HiGHS.Optimizer)
@variable(model, x)
@variable(model, y)

fix(x, 2.0)          # Fix x to 2.0
is_fixed(x)          # true
optimize!(model)

unfix(x)             # Release constraint
optimize!(model)
```

---

## Solution Checking

```julia
optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    result = value(x)
elseif termination_status(model) == MOI.INFEASIBLE
    println("Problem is infeasible")
elseif termination_status(model) == MOI.DUAL_INFEASIBLE
    println("Problem is unbounded")
else
    println("Status: ", termination_status(model))
end
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Check status | Always verify termination_status before using results |
| Name constraints | For shadow prices and debugging |
| Warm starts | Provide initial values for NLP |
| Reuse models | empty!(model) for repeated solves |
| Vectorize | @variable(model, x[1:n]) for arrays |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Wrong solver for type | LP→HiGHS, NLP→Ipopt, MIP→HiGHS |
| Assuming solution exists | Check status before value() |
| Creating model in loop | Reuse model with empty!() |
| Slow NLP convergence | Provide good starting points |

---

## Checklist

- [ ] Problem type identified (LP/QP/NLP/MIP)
- [ ] Appropriate solver selected
- [ ] Variables bounded appropriately
- [ ] Constraints named for debugging
- [ ] Solution status checked
- [ ] Warm starts for iterative problems

---

**Version**: 1.0.5
