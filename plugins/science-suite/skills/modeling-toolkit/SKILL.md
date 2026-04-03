---
name: modeling-toolkit
maturity: "5-Expert"
specialization: Symbolic Modeling
description: Define symbolic differential equations with ModelingToolkit.jl for automatic simplification and code generation. Use when building complex mathematical models declaratively.
---

# ModelingToolkit.jl

## Expert Agent

For symbolic differential equation modeling with ModelingToolkit.jl, delegate to:

- **`julia-pro`**: Julia SciML ecosystem and symbolic modeling workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

Symbolic modeling with automatic equation simplification.

---

## Basic Pattern

```julia
using ModelingToolkit, DifferentialEquations

@variables t x(t) y(t)
@parameters α β
D = Differential(t)

eqs = [D(x) ~ α * x, D(y) ~ -β * y]
@named sys = ODESystem(eqs, t)
sys_simple = structural_simplify(sys)

prob = ODEProblem(sys_simple, [x => 1.0, y => 1.0],
                  (0.0, 10.0), [α => 0.5, β => 0.3])
sol = solve(prob, Tsit5())
```

---

## Key Features

| Feature | Benefit |
|---------|---------|
| Symbolic variables | @variables, @parameters |
| Automatic simplification | structural_simplify() |
| Code generation | Efficient numerical code |
| Component-based | Reusable model parts |

---

## Checklist

- [ ] Variables and parameters defined
- [ ] Equations specified symbolically
- [ ] System simplified with structural_simplify
- [ ] Problem converted and solved

---

**Version**: 1.0.5
