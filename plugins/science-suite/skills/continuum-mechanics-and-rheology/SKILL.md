---
name: continuum-mechanics-and-rheology
description: Meta-orchestrator for continuum mechanics, FEM/FEA, constitutive modeling, rheology/DMA, transient networks, and nanocomposites. Use when formulating finite element models, fitting viscoelastic/hyperelastic constitutive laws, interpreting DMA or rheology data, modeling covalent adaptable networks (vitrimers) or physical gels, or predicting nanocomposite effective properties.
---

# Continuum Mechanics & Rheology Hub

Orchestrator for continuum-scale materials engineering. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`continuum-mechanics-engineer`**: Specialist for FEM, constitutive modeling, DMA/rheology, transient networks, and nanocomposites.
  - *Location*: `plugins/science-suite/agents/continuum-mechanics-engineer.md`
  - *Capabilities*: Weak-form PDE discretization, hyperelastic/viscoelastic constitutive fitting, transient-network rheology, effective-medium composite modeling.

## Core Skills

### [FEM/FEA](../fem-fea/SKILL.md)
Weak-form formulation, mesh convergence, element selection, nonlinear solvers. For mesh-connectivity graph representations, see [Graph Theory](../graph-theory/SKILL.md).

### [Constitutive Equations](../constitutive-equations/SKILL.md)
Linear/nonlinear viscoelasticity, hyperelastic models (Neo-Hookean, Mooney-Rivlin), Prony series fitting.

### [DMA & Rheology](../dma-rheology/SKILL.md)
Storage/loss modulus interpretation, oscillatory shear, shear and extensional flow curves.

### [Harmonic Response & Superposition](../harmonic-response-superposition/SKILL.md)
Complex modulus under harmonic loading, WLF time-temperature superposition, master curves.

### [Transient Networks & CAN](../transient-networks-and-can/SKILL.md)
Physical gels, covalent adaptable networks (vitrimers), bond-exchange kinetics, sticky Rouse and Green-Tobolsky models.

### [Nanocomposites & Adaptive Materials](../nanocomposites-and-adaptive-materials/SKILL.md)
Effective-medium theory (Halpin-Tsai, Mori-Tanaka), percolation-aware property prediction, self-healing composites.

## Routing Decision Tree

```
What is the continuum-mechanics task?
|
+-- Weak-form discretization / mesh / convergence?
|   --> science-suite:fem-fea
|
+-- Stress-strain law selection / fitting (elastic, hyperelastic, viscoelastic)?
|   --> science-suite:constitutive-equations
|
+-- DMA data (storage/loss modulus, tan delta) / rheology (shear, extensional)?
|   --> science-suite:dma-rheology
|
+-- Harmonic/complex modulus / time-temperature superposition / master curves?
|   --> science-suite:harmonic-response-superposition
|
+-- Physical gel / vitrimer / covalent adaptable network / bond-exchange kinetics?
|   --> science-suite:transient-networks-and-can
|
+-- Filler-matrix composite / effective modulus / percolation threshold for properties?
|   --> science-suite:nanocomposites-and-adaptive-materials
|
+-- None of the above / concern is ambiguous?
    --> Delegate to continuum-mechanics-engineer for open-ended triage.
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| Weak form, mesh, FEniCS/Gridap.jl | `science-suite:fem-fea` |
| Neo-Hookean, Mooney-Rivlin, Prony series | `science-suite:constitutive-equations` |
| Storage/loss modulus, oscillatory rheology | `science-suite:dma-rheology` |
| WLF equation, master curves | `science-suite:harmonic-response-superposition` |
| Vitrimer, bond exchange, sticky Rouse | `science-suite:transient-networks-and-can` |
| Halpin-Tsai, Mori-Tanaka, percolation | `science-suite:nanocomposites-and-adaptive-materials` |

## Checklist

- [ ] Identify discretization vs. constitutive-modeling vs. experimental-interpretation before routing
- [ ] Confirm whether the material is elastic, viscoelastic, or exhibits bond-exchange dynamics before selecting a constitutive model
- [ ] Cross-check percolation/composite claims with `statistical-physics-hub` rather than re-deriving percolation theory here
- [ ] Verify mesh convergence for any FEM result before reporting it as final
