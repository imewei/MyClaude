---
name: catalyst-reactions
description: Master reaction network modeling with Catalyst.jl for chemical and biochemical systems supporting both deterministic ODE and stochastic simulations. Use when modeling chemical reactions (.jl files with @reaction_network), defining reaction rates and stoichiometry, converting reaction networks to ODESystem or JumpSystem, working with mass action kinetics, implementing Gillespie stochastic simulations, modeling biochemical pathways (metabolic networks, gene regulation), specifying reaction parameters and species, integrating with DifferentialEquations.jl for simulation, or analyzing chemical/biological systems. Essential for systems biology, chemical engineering, and biochemical modeling workflows.
---

# Catalyst Reaction Networks

Model chemical reaction networks with Catalyst.jl.

## When to use this skill

- Modeling chemical reactions with @reaction_network macro
- Defining reaction rates and stoichiometric coefficients
- Converting reaction networks to ODESystem or JumpSystem
- Working with mass action kinetics and rate laws
- Implementing Gillespie algorithm for stochastic simulations
- Modeling biochemical pathways (metabolism, gene regulation, signaling)
- Specifying reaction parameters and chemical species
- Integrating with DifferentialEquations.jl for deterministic/stochastic simulation
- Analyzing steady states and bifurcations in reaction systems
- Modeling enzyme kinetics (Michaelis-Menten, Hill equations)
- Building systems biology models

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
