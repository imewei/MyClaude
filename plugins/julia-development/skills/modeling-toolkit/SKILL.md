---
name: modeling-toolkit
description: Master symbolic problem definition with ModelingToolkit.jl for equation simplification, code generation, and declarative mathematical modeling. Use when defining symbolic differential equations (.jl files with @variables, @parameters, Differential), creating ODESystem, PDESystem, or NonlinearSystem from symbolic equations, using structural_simplify for automatic equation optimization, generating efficient code from symbolic definitions, working with component-based modeling and connections, building physics models declaratively, performing symbolic differentiation for automatic Jacobian generation, or integrating with DifferentialEquations.jl. Essential for complex mathematical modeling, physics simulations, and leveraging automatic equation simplification for performance.
---

# ModelingToolkit

Symbolic modeling with ModelingToolkit.jl for automated equation simplification and code generation.

## When to use this skill

- Defining symbolic differential equations with @variables and @parameters
- Creating ODESystem, PDESystem, NonlinearSystem from symbolic equations
- Using structural_simplify for automatic equation reduction and optimization
- Generating efficient numerical code from symbolic definitions
- Working with component-based modeling (ModelingToolkit components)
- Building physics models declaratively
- Performing symbolic differentiation for automatic Jacobian computation
- Integrating symbolic systems with DifferentialEquations.jl solvers
- Creating reusable model components with composition
- Simplifying complex equation systems automatically
- Generating optimized code for specific problem structures

## Basic Pattern
```julia
using ModelingToolkit, DifferentialEquations

@variables t x(t) y(t)
@parameters α β
D = Differential(t)

eqs = [D(x) ~ α * x, D(y) ~ -β * y]
@named sys = ODESystem(eqs, t)
sys_simple = structural_simplify(sys)
prob = ODEProblem(sys_simple, [x => 1.0, y => 1.0], (0.0, 10.0), [α => 0.5, β => 0.3])
sol = solve(prob, Tsit5())
```

## Resources
- **ModelingToolkit.jl**: https://docs.sciml.ai/ModelingToolkit/stable/
