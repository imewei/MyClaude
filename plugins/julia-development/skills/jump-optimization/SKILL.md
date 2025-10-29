---
name: jump-optimization
description: Master JuMP.jl for mathematical programming and optimization modeling. This skill should be used for linear programming (LP), quadratic programming (QP), nonlinear programming (NLP), and mixed-integer programming (MIP). Note that JuMP.jl is separate from Optimization.jl (used in sciml-pro) to avoid conflicts.
---

# JuMP Optimization

Master JuMP.jl (Julia for Mathematical Programming), a domain-specific modeling language for mathematical optimization embedded in Julia. JuMP provides a high-level, algebraic modeling interface for linear, quadratic, nonlinear, and mixed-integer optimization.

**Important**: JuMP.jl is with julia-pro for mathematical programming. For SciML optimization workflows (parameter estimation, loss minimization), use sciml-pro's Optimization.jl skill.

## What This Skill Provides

1. **Linear Programming (LP)** - Model and solve LP problems with various solvers
2. **Quadratic Programming (QP)** - Handle quadratic objectives and constraints
3. **Nonlinear Programming (NLP)** - Solve nonlinear optimization with automatic differentiation
4. **Mixed-Integer Programming (MIP)** - Discrete optimization with binary and integer variables
5. **Solver Management** - Interface with multiple solvers (HiGHS, Ipopt, GLPK, COSMO)
6. **Modeling Patterns** - Common optimization patterns and formulations

## When to Use This Skill

Use JuMP when encountering:
- Linear programming problems (production planning, resource allocation)
- Quadratic programming (portfolio optimization, least squares)
- Nonlinear optimization (general constrained optimization)
- Mixed-integer programming (scheduling, routing, assignment)
- Mathematical modeling of optimization problems
- Solver selection and configuration
- Sensitivity analysis and dual values

**DO NOT use for**:
- SciML parameter estimation → use sciml-pro's Optimization.jl
- Neural network training → use Flux.jl or MLJ.jl
- Unconstrained numerical optimization → consider Optim.jl or Optimization.jl

## Core Concepts

### Linear Programming Example

```julia
using JuMP
using HiGHS  # Fast open-source LP/MIP solver

# Production planning problem
function production_optimization()
    model = Model(HiGHS.Optimizer)
    set_silent(model)  # Suppress solver output

    # Decision variables
    @variable(model, x₁ >= 0)  # Product 1 quantity
    @variable(model, x₂ >= 0)  # Product 2 quantity

    # Objective: Maximize profit
    @objective(model, Max, 40x₁ + 30x₂)

    # Constraints: Resource limits
    @constraint(model, labor, 2x₁ + x₂ <= 100)      # Labor hours
    @constraint(model, material, x₁ + 2x₂ <= 80)    # Material units
    @constraint(model, machine, x₁ + x₂ <= 50)      # Machine time

    # Solve
    optimize!(model)

    # Check solution status
    if termination_status(model) == MOI.OPTIMAL
        println("Optimal solution found!")
        println("  x₁ = ", value(x₁))
        println("  x₂ = ", value(x₂))
        println("  Profit = \$", objective_value(model))

        # Shadow prices (dual values)
        println("\nShadow prices:")
        println("  Labor: ", dual(labor))
        println("  Material: ", dual(material))
        println("  Machine: ", dual(machine))
    else
        println("No optimal solution: ", termination_status(model))
    end

    return (x₁=value(x₁), x₂=value(x₂), profit=objective_value(model))
end
```

### Quadratic Programming

```julia
using Ipopt  # Nonlinear solver (handles QP)

# Portfolio optimization: Minimize risk subject to return constraint
function portfolio_optimization(μ, Σ, target_return)
    n = length(μ)  # Number of assets
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Decision variables: portfolio weights
    @variable(model, 0 <= w[1:n] <= 1)

    # Objective: Minimize variance (risk)
    @objective(model, Min, w' * Σ * w)

    # Constraints
    @constraint(model, sum(w) == 1)              # Weights sum to 1
    @constraint(model, w' * μ >= target_return)  # Minimum return

    optimize!(model)

    return value.(w)
end

# Example usage
μ = [0.10, 0.15, 0.12]  # Expected returns
Σ = [0.04 0.01 0.02;     # Covariance matrix
     0.01 0.09 0.03;
     0.02 0.03 0.06]
weights = portfolio_optimization(μ, Σ, 0.11)
```

### Nonlinear Programming

```julia
using Ipopt

# Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
function optimize_rosenbrock()
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, x)
    @variable(model, y)

    # Nonlinear objective
    @objective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)

    # Optional starting point
    set_start_value(x, 0.0)
    set_start_value(y, 0.0)

    optimize!(model)

    println("Solution: x = ", value(x), ", y = ", value(y))
    println("Minimum = ", objective_value(model))

    return (x=value(x), y=value(y))
end

# Constrained nonlinear optimization
function constrained_nlp()
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, y >= 0)

    # Nonlinear objective and constraint
    @objective(model, Min, (x - 3)^2 + (y - 2)^2)
    @constraint(model, nlcon, x^2 + y^2 >= 1)  # Nonlinear constraint

    optimize!(model)

    return (x=value(x), y=value(y), obj=objective_value(model))
end
```

### Mixed-Integer Programming

```julia
using HiGHS  # Supports MIP

# Knapsack problem
function knapsack(values, weights, capacity)
    n = length(values)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Binary variables: include item or not
    @variable(model, x[1:n], Bin)

    # Maximize total value
    @objective(model, Max, sum(values[i] * x[i] for i in 1:n))

    # Capacity constraint
    @constraint(model, sum(weights[i] * x[i] for i in 1:n) <= capacity)

    optimize!(model)

    selected = [i for i in 1:n if value(x[i]) > 0.5]
    return (items=selected, value=objective_value(model))
end

# Facility location problem
function facility_location(n_facilities, n_customers, costs, demands, capacities)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Binary: open facility i?
    @variable(model, y[1:n_facilities], Bin)

    # Continuous: fraction of demand j served by facility i
    @variable(model, 0 <= x[1:n_facilities, 1:n_customers] <= 1)

    # Minimize total cost
    @objective(model, Min,
        sum(costs[i,j] * demands[j] * x[i,j]
            for i in 1:n_facilities, j in 1:n_customers))

    # Each customer's demand must be met
    @constraint(model, [j in 1:n_customers],
        sum(x[i,j] for i in 1:n_facilities) == 1)

    # Facility capacity constraints
    @constraint(model, [i in 1:n_facilities],
        sum(demands[j] * x[i,j] for j in 1:n_customers) <= capacities[i] * y[i])

    optimize!(model)

    return (facilities=value.(y), assignments=value.(x))
end
```

## Best Practices

### Modeling
- Define variables with appropriate bounds and types
- Use @expression for reusable sub-expressions
- Name constraints for easy reference and dual value access
- Use vectorized notation for conciseness: `@variable(model, x[1:n])`
- Provide good starting points for nonlinear problems

### Solver Selection
- **Linear (LP)**: HiGHS (fast, open-source), GLPK (free), Gurobi (commercial, powerful)
- **Quadratic (QP)**: Ipopt, COSMO, Gurobi
- **Nonlinear (NLP)**: Ipopt (general-purpose), KNITRO (commercial)
- **Mixed-Integer (MIP)**: HiGHS, GLPK, Gurobi, CPLEX
- **Conic**: COSMO, Mosek, SCS

### Performance
- Use set_silent(model) to suppress solver output
- Set solver-specific options for performance tuning
- Provide warm starts for iterative solving
- Use @expression to avoid duplicate computations
- Consider problem reformulations for better performance

### Robustness
- Check termination_status(model) before using results
- Handle infeasible and unbounded cases
- Validate bounds and constraints
- Use has_values(model) to check if solution exists

## Common Pitfalls

### Incorrect Solver for Problem Type
```julia
# BAD: Using LP solver for nonlinear problem
using GLPK  # Only supports LP/MIP
model = Model(GLPK.Optimizer)
@variable(model, x)
@objective(model, Min, x^2)  # ERROR: GLPK doesn't support nonlinear!

# GOOD: Use appropriate solver
using Ipopt
model = Model(Ipopt.Optimizer)
@variable(model, x)
@objective(model, Min, x^2)  # Works
```

### Not Checking Solution Status
```julia
# BAD: Assuming solution exists
optimize!(model)
result = value(x)  # May error if infeasible!

# GOOD: Check status first
optimize!(model)
if termination_status(model) == MOI.OPTIMAL
    result = value(x)
elseif termination_status(model) == MOI.INFEASIBLE
    println("Problem is infeasible")
else
    println("Solver stopped with status: ", termination_status(model))
end
```

### Type Instability in Loops
```julia
# BAD: Creating model in loop (slow)
results = []
for data in datasets
    model = Model(HiGHS.Optimizer)
    # ... build and solve ...
    push!(results, objective_value(model))
end

# GOOD: Reuse model or create once
model = Model(HiGHS.Optimizer)
results = Float64[]
for data in datasets
    empty!(model)  # Clear previous model
    # ... build and solve ...
    push!(results, objective_value(model))
end
```

## Advanced Patterns

### Callbacks and Cuts
```julia
# User-defined cuts for MIP
using HiGHS

function solve_with_cuts()
    model = Model(HiGHS.Optimizer)
    @variable(model, x[1:3], Bin)
    @objective(model, Max, sum(x))

    # Add lazy constraint callback
    function my_callback(cb_data)
        x_val = callback_value.(cb_data, x)
        if sum(x_val[1:2]) > 1.5
            con = @build_constraint(x[1] + x[2] <= 1)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    end

    set_attribute(model, MOI.LazyConstraintCallback(), my_callback)
    optimize!(model)
end
```

### Warm Starting
```julia
# Solve sequence of related problems with warm starts
function solve_sequence(n_problems)
    model = Model(Ipopt.Optimizer)
    @variable(model, x >= 0)
    @variable(model, y >= 0)

    results = []
    for i in 1:n_problems
        # Update objective
        @objective(model, Min, (x - i)^2 + (y - i)^2)

        # Use previous solution as starting point
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
# Solve with some variables fixed
model = Model(HiGHS.Optimizer)
@variable(model, x)
@variable(model, y)
@objective(model, Min, x^2 + y^2)

# Fix x to a specific value
fix(x, 2.0)
is_fixed(x)  # true

optimize!(model)
println("With x=2: y = ", value(y))

# Unfix and resolve
unfix(x)
optimize!(model)
println("Both free: x = ", value(x), ", y = ", value(y))
```

## Resources

- **JuMP Documentation**: https://jump.dev/JuMP.jl/stable/
- **JuMP Tutorials**: https://jump.dev/JuMP.jl/stable/tutorials/
- **Solver List**: https://jump.dev/JuMP.jl/stable/installation/#Getting-Solvers
- **MOI (MathOptInterface)**: https://jump.dev/MathOptInterface.jl/stable/

## Related Skills

- **optimization-patterns** (sciml-pro): For SciML-specific optimization with Optimization.jl
- **core-julia-patterns**: Type system and performance patterns
- **package-management**: Managing JuMP and solver dependencies
