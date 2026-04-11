---
name: sciml-and-diffeq
description: Meta-orchestrator for Scientific Machine Learning and differential equations in Julia. Routes to SciML ecosystem, DiffEq solvers, ModelingToolkit, optimization, neural PDE, reaction networks, JuMP, SINDy equation discovery, bifurcation analysis, and Bayesian UDE skills. Use when solving ODEs/PDEs/SDEs in Julia, using ModelingToolkit, fitting models with Optimization.jl, building neural PDEs, modeling chemical reactions, discovering equations from data, computing bifurcation diagrams, or fitting Bayesian neural ODEs.
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
Lux.jl explicit-parameter neural networks, SciMLSensitivity.jl adjoint/forward sensitivity, Universal Differential Equations (UDEs), UncertaintyQuantification.jl, DeepEquilibriumNetworks.jl. Use for neural-physics hybrids and AD-through-solvers. (Frozen at 78% budget — do not add new content; create a sibling skill instead.)

### [Differential Equations](../differential-equations/SKILL.md)
ODE/SDE/DAE/DDE solvers: solver selection, stiffness detection, and error control.

### [Modeling Toolkit](../modeling-toolkit/SKILL.md)
ModelingToolkit.jl: symbolic-numeric modeling, structural simplification, and code generation.

### [Optimization Patterns](../optimization-patterns/SKILL.md)
Optimization.jl: unified interface for local/global optimizers and parameter estimation.

### [Neural PDE](../neural-pde/SKILL.md)
NeuralPDE.jl: physics-informed neural networks (PINNs) and neural operators for PDEs (deterministic).

### [Bayesian PINN](../bayesian-pinn/SKILL.md)
NeuralPDE.jl's BNNODE / BayesianPINN discretizers — credible intervals on PINN solutions via internal AdvancedHMC.

### [Catalyst Reactions](../catalyst-reactions/SKILL.md)
Catalyst.jl: reaction network modeling, mass action kinetics, and stochastic simulations.

### [JuMP Optimization](../jump-optimization/SKILL.md)
JuMP.jl: mathematical programming, LP/QP/MIP, and solver interfaces.

### [Equation Discovery (SINDy)](../equation-discovery/SKILL.md)
DataDrivenDiffEq.jl: sparse identification of nonlinear dynamics from time-series data.

### [Bifurcation Analysis](../bifurcation-analysis/SKILL.md)
Numerical continuation, branch tracking, and bifurcation diagrams for ODE/PDE systems.

### [Bayesian UDE Workflow](../bayesian-ude-workflow/SKILL.md)
End-to-end Bayesian Universal Differential Equations: hybrid physics + neural correction with posterior uncertainty.

## Routing Decision Tree

```
What is the SciML / DiffEq task?
|
+-- Which SciML package to use?
|   --> sciml-ecosystem
|
+-- Solve ODE / SDE / DAE / DDE?
|   --> differential-equations (solver selection, stiffness, error control)
|
+-- Which SciML component / package to use at a glance?
|   --> sciml-ecosystem (package map, when to use what)
|
+-- UDE / neural ODE / Lux.jl / SciMLSensitivity adjoint / AD-through-solver / UQ on trained models?
|   --> sciml-modern-stack (Lux + SciMLSensitivity + UncertaintyQuantification specifics)
|   (for the end-to-end Bayesian UDE workflow with posterior sampling, use bayesian-ude-workflow instead)
|
+-- Symbolic modeling / structural analysis?
|   --> modeling-toolkit
|
+-- Parameter estimation / inverse problems?
|   --> optimization-patterns
|
+-- PDE with neural networks (deterministic PINNs)?
|   --> neural-pde
|
+-- Bayesian PINN (credible intervals via BNNODE / BayesianPINN)?
|   --> bayesian-pinn
|
+-- Chemical reaction networks?
|   --> catalyst-reactions
|
+-- Linear/integer programming?
|   --> jump-optimization
|
+-- Discover symbolic equations from time-series data (SINDy)?
|   --> equation-discovery
|
+-- Bifurcation diagrams / numerical continuation?
|   --> bifurcation-analysis
|
+-- Universal DE / neural ODE with posterior uncertainty?
    --> bayesian-ude-workflow
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
