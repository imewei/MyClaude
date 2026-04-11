---
name: statistical-physics-hub
description: Meta-orchestrator for statistical physics and soft matter. Routes to equilibrium/non-equilibrium theory, stochastic dynamics, active matter, multiscale modeling, and advanced simulation skills. Use when modeling equilibrium/non-equilibrium systems, simulating stochastic dynamics, studying active matter, implementing multiscale methods, or running advanced MD simulations.
---

# Statistical Physics Hub

Orchestrator for statistical physics and soft matter. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`statistical-physicist`**: Specialist for statistical mechanics, field theory, and soft matter.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - *Capabilities*: Equilibrium and non-equilibrium theory, phase transitions, stochastic processes, active matter, and multiscale modeling.

## Core Skills

### [Statistical Physics](../statistical-physics/SKILL.md)
Equilibrium statistical mechanics: partition functions, phase transitions, critical phenomena, renormalization group, and Monte Carlo samplers (Metropolis / heat-bath / cluster / Wang-Landau / replica exchange) with Julia idiomatic patterns and the Python production handoff.

### [Stochastic Dynamics](../stochastic-dynamics/SKILL.md)
Langevin equations, Fokker-Planck, Brownian motion, and stochastic differential equations.

### [Non-Equilibrium Theory](../non-equilibrium-theory/SKILL.md)
Driven systems, fluctuation theorems, entropy production, and linear response theory.

### [Active Matter](../active-matter/SKILL.md)
Self-propelled particles, collective motion, motility-induced phase separation, and biological active systems.

### [Multiscale Modeling](../multiscale-modeling/SKILL.md)
Coarse-graining, effective field theories, renormalization, and bridging micro to macro scales.

### [Advanced Simulations](../advanced-simulations/SKILL.md)
Monte Carlo methods, replica exchange, umbrella sampling, and free energy calculations.

### [Rare Events Sampling](../rare-events-sampling/SKILL.md)
FFS, TIS / RETIS, TPS, AMS, WE, OPS / WESTPA / pyretis, milestoning, and cloning algorithms for thermally-activated barriers, large-deviation statistics, and avalanche exponents.

## Routing Decision Tree

```
What is the statistical physics task?
|
+-- Equilibrium thermodynamics / phase transitions / Monte Carlo sampling
|   (Metropolis / heat-bath / cluster / Wang-Landau / parallel tempering)?
|   --> statistical-physics
|
+-- Stochastic processes / Langevin / FP equations?
|   --> stochastic-dynamics
|
+-- Driven / non-equilibrium systems?
|   --> non-equilibrium-theory
|
+-- Self-propelled particles / biological active systems?
|   --> active-matter
|
+-- Coarse-graining / bridging scales?
|   --> multiscale-modeling
|
+-- Advanced MC / free energy / enhanced sampling?
|   --> advanced-simulations
|
+-- Rare events / barrier crossings / large-deviation / avalanche statistics?
    --> rare-events-sampling
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| Partition functions, RG, criticality, Metropolis / cluster / Wang-Landau / parallel tempering MC | `statistical-physics` |
| Langevin, Fokker-Planck, SDEs | `stochastic-dynamics` |
| Fluctuation theorems, entropy production | `non-equilibrium-theory` |
| Active particles, MIPS, flocking | `active-matter` |
| Coarse-graining, EFT | `multiscale-modeling` |
| Replica exchange, umbrella sampling | `advanced-simulations` |
| FFS / TIS / WE / cloning / rare events | `rare-events-sampling` |

## Checklist

- [ ] Identify equilibrium vs non-equilibrium before routing
- [ ] Verify thermodynamic limit assumptions are valid for the system size
- [ ] Check detailed balance / broken detailed balance for dynamics classification
- [ ] Confirm simulation ensemble matches physical conditions (NVT, NPT, etc.)
- [ ] Validate coarse-grained models reproduce fine-grained observables
- [ ] Use fluctuation theorems to cross-check non-equilibrium simulation results
- [ ] Document all simulation parameters and random seeds for reproducibility
