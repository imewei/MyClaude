---
name: modeling-toolkit
description: Symbolic problem definition with ModelingToolkit.jl, equation simplification, and code generation. Use for declarative modeling and automatic optimization of equation systems.
---

# ModelingToolkit

Symbolic modeling with ModelingToolkit.jl for automated equation simplification and code generation.

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
