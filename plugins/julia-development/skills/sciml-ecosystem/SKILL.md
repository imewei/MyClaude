---
name: sciml-ecosystem
description: Comprehensive overview of the Scientific Machine Learning (SciML) ecosystem including DifferentialEquations.jl, ModelingToolkit.jl, Optimization.jl, Catalyst.jl, NeuralPDE.jl, and SciMLSensitivity.jl. Use when selecting appropriate SciML packages for scientific computing tasks, understanding package relationships and integration patterns, choosing between DifferentialEquations.jl for solving equations vs ModelingToolkit.jl for symbolic modeling, deciding between Optimization.jl (SciML) vs JuMP.jl (mathematical programming), working with NeuralPDE.jl for physics-informed neural networks, integrating Catalyst.jl for reaction networks, or understanding the SciML stack architecture. Essential for navigating the SciML ecosystem and making informed package selection decisions.
---

# SciML Ecosystem

Comprehensive overview of the Scientific Machine Learning (SciML) ecosystem in Julia.

## When to use this skill

- Understanding SciML package relationships and integration patterns
- Selecting appropriate packages for differential equations (ODE, PDE, SDE, DAE)
- Choosing between symbolic (ModelingToolkit.jl) vs numeric approaches
- Understanding Optimization.jl (SciML) vs JuMP.jl (mathematical programming) differences
- Working with parameter estimation and sensitivity analysis
- Integrating NeuralPDE.jl for physics-informed neural networks
- Using Catalyst.jl for chemical reaction network modeling
- Planning SciML-based scientific computing workflows
- Combining multiple SciML packages in projects
- Navigating the SciML documentation ecosystem
- Understanding solver selection across different equation types

## Core Packages

**DifferentialEquations.jl**: ODE, PDE, SDE, DAE solving
**ModelingToolkit.jl**: Symbolic modeling and code generation
**Optimization.jl**: Parameter estimation and optimization
**NeuralPDE.jl**: Physics-informed neural networks
**Catalyst.jl**: Chemical reaction networks
**SciMLSensitivity.jl**: Sensitivity analysis
**DataDrivenDiffEq.jl**: Data-driven modeling

## Package Selection Guide

- Solving equations → DifferentialEquations.jl
- Symbolic modeling → ModelingToolkit.jl
- Parameter estimation → Optimization.jl + SciMLSensitivity.jl
- PINNs → NeuralPDE.jl
- Reactions → Catalyst.jl

## Resources
- **SciML Documentation**: https://docs.sciml.ai/
