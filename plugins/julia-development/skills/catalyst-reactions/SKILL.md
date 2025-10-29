---
name: catalyst-reactions
description: Reaction network modeling with Catalyst.jl for chemical and biochemical systems. Supports deterministic ODE and stochastic simulation.
---

# Catalyst Reaction Networks

Model chemical reaction networks with Catalyst.jl.

## Reaction Network Pattern
```julia
using Catalyst, DifferentialEquations

rn = @reaction_network begin
    k1, A + B --> C
    k2, C --> A + B
end k1 k2

odesys = convert(ODESystem, rn)
prob = ODEProblem(odesys, [A => 10, B => 10, C => 0], (0.0, 10.0), [k1 => 0.1, k2 => 0.05])
sol = solve(prob, Tsit5())
```

## Resources
- **Catalyst.jl**: https://docs.sciml.ai/Catalyst/stable/
