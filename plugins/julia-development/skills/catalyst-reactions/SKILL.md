---
name: catalyst-reactions
version: "1.0.7"
maturity: "5-Expert"
specialization: Reaction Networks
description: Model chemical reaction networks with Catalyst.jl for deterministic and stochastic simulations. Use when modeling biochemical pathways or chemical kinetics.
---

# Catalyst.jl Reaction Networks

Chemical and biochemical reaction network modeling.

---

## Reaction Network Pattern

```julia
using Catalyst, DifferentialEquations

rn = @reaction_network begin
    k1, A + B --> C
    k2, C --> A + B
end k1 k2

odesys = convert(ODESystem, rn)
prob = ODEProblem(odesys, [A => 10, B => 10, C => 0],
                  (0.0, 10.0), [k1 => 0.1, k2 => 0.05])
sol = solve(prob, Tsit5())
```

---

## Simulation Types

| Type | Convert To | Use Case |
|------|------------|----------|
| Deterministic | ODESystem | Large populations |
| Stochastic | JumpSystem | Small populations |
| SDE | SDESystem | Chemical noise |

---

## Applications

- Systems biology
- Metabolic networks
- Gene regulation
- Enzyme kinetics
- Chemical engineering

---

## Checklist

- [ ] Reactions defined with @reaction_network
- [ ] Rates and species specified
- [ ] Converted to appropriate system type
- [ ] Initial conditions set

---

**Version**: 1.0.5
