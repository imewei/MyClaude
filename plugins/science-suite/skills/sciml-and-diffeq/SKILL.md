---
name: sciml-and-diffeq
description: Meta-orchestrator for Scientific Machine Learning and differential equations in Julia. Routes to SciML ecosystem, DiffEq solvers, ModelingToolkit, optimization, neural PDE, reaction networks, and JuMP skills. Use when solving ODEs/PDEs/SDEs in Julia, using ModelingToolkit, fitting models with Optimization.jl, building neural PDEs, or modeling chemical reactions.
---

# SciML and Differential Equations

Orchestrator for Scientific Machine Learning (SciML) and differential equations in Julia. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`julia-pro`**: Specialist for SciML ecosystem, DifferentialEquations.jl, and ModelingToolkit.jl.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: ODE/SDE/PDE solvers, symbolic-numeric workflows, neural ODEs, parameter estimation.

## Core Skills

### [SciML Ecosystem](../sciml-ecosystem/SKILL.md)
Overview of the SciML organization: packages, interoperability, and when to use each.

### [SciML Modern Stack](../sciml-modern-stack/SKILL.md)
DifferentialEquations.jl v7+, SciMLBase interfaces, and modern solver selection.

### [Differential Equations](../differential-equations/SKILL.md)
ODE/SDE/DAE/DDE solvers: solver selection, stiffness detection, and error control.

### [Modeling Toolkit](../modeling-toolkit/SKILL.md)
ModelingToolkit.jl: symbolic-numeric modeling, structural simplification, and code generation.

### [Optimization Patterns](../optimization-patterns/SKILL.md)
Optimization.jl: unified interface for local/global optimizers and parameter estimation.

### [Neural PDE](../neural-pde/SKILL.md)
NeuralPDE.jl: physics-informed neural networks (PINNs) and neural operators for PDEs.

### [Catalyst Reactions](../catalyst-reactions/SKILL.md)
Catalyst.jl: reaction network modeling, mass action kinetics, and stochastic simulations.

### [JuMP Optimization](../jump-optimization/SKILL.md)
JuMP.jl: mathematical programming, LP/QP/MIP, and solver interfaces.

## Routing Decision Tree

```
What is the SciML / DiffEq task?
|
+-- Which SciML package to use?
|   --> sciml-ecosystem
|
+-- Solve ODE / SDE / DAE / DDE?
|   --> differential-equations (solver selection, stiffness)
|   --> sciml-modern-stack (SciMLBase v7+ interfaces)
|
+-- Symbolic modeling / structural analysis?
|   --> modeling-toolkit
|
+-- Parameter estimation / inverse problems?
|   --> optimization-patterns
|
+-- PDE with neural networks (PINNs)?
|   --> neural-pde
|
+-- Chemical reaction networks?
|   --> catalyst-reactions
|
+-- Linear/integer programming?
    --> jump-optimization
```

## Checklist

- [ ] Use routing tree to select solver type before writing code
- [ ] Check stiffness: use `AutoTsit5` or `Rodas5` for stiff systems
- [ ] Verify ModelingToolkit structural simplification before numerical solve
- [ ] Use `Optimization.jl` interface for solver-agnostic parameter estimation
- [ ] Validate PINN training loss includes both data and physics residual terms
- [ ] Confirm Catalyst.jl reaction networks are balanced before simulation
- [ ] Test JuMP models with a known feasible solution before scaling up
- [ ] Pin SciML package versions together (they co-evolve rapidly)
