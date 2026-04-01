---
name: sciml-ecosystem
maturity: "5-Expert"
specialization: SciML Overview
description: Navigate the SciML ecosystem including DifferentialEquations.jl, ModelingToolkit.jl, Optimization.jl, and Catalyst.jl. Use when selecting packages for scientific computing tasks.
---

# SciML Ecosystem

Overview of Scientific Machine Learning packages in Julia.

> **Important (2026):** Lux.jl has replaced Flux.jl as the standard neural network library for SciML. All new SciML neural network work should use Lux.jl for its explicit parameterization model. See `sciml-modern-stack` skill for details.

## Expert Agent

For complex SciML workflows, differential equation solving, and scientific machine learning integration, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia optimization, including SciML, DifferentialEquations.jl, ModelingToolkit.jl, and Optimization.jl.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: Stiff ODE solvers, sensitivity analysis, symbolic modeling, and physics-informed neural networks.

---

## Core Packages

| Package | Purpose |
|---------|---------|
| DifferentialEquations.jl | ODE, PDE, SDE, DAE solving |
| ModelingToolkit.jl | Symbolic modeling |
| Optimization.jl | Parameter estimation |
| NeuralPDE.jl | Physics-informed NNs |
| Catalyst.jl | Reaction networks |
| SciMLSensitivity.jl | Sensitivity analysis |

---

## Selection Guide

| Task | Package |
|------|---------|
| Solve equations | DifferentialEquations.jl |
| Symbolic modeling | ModelingToolkit.jl |
| Parameter fitting | Optimization.jl |
| PINNs | NeuralPDE.jl |
| Reactions | Catalyst.jl |
| Math programming | JuMP.jl (not SciML) |

---

## Integration

All packages integrate with DifferentialEquations.jl:
- ModelingToolkit → ODEProblem
- Catalyst → ODESystem/JumpSystem
- NeuralPDE → discretized problems

---

## Modern SciML Components

### Lux.jl (Neural Networks)

Lux.jl provides explicit parameterization for neural networks, separating model structure from parameters and state. This makes it naturally compatible with functional transformations (differentiation, optimization) used throughout SciML. See `sciml-modern-stack` skill for architecture patterns and migration guidance.

### Universal Differential Equations (UDEs)

UDEs combine known physics (differential equations) with neural networks to learn unknown dynamics from data. The standard discovery pipeline is:

1. **UDE Training** — Embed a neural network term in an ODE/SDE system, train against observed data
2. **SINDy Extraction** — Use DataDrivenDiffEq.jl to extract symbolic equations from the trained neural network

This UDE+SINDy pipeline enables automated scientific discovery: learn the dynamics neurally, then distill into interpretable equations. See `sciml-modern-stack` skill for implementation details.

### DataDrivenDiffEq.jl (Equation Discovery)

Sparse Identification of Nonlinear Dynamics (SINDy) and symbolic regression for extracting governing equations from data or trained neural networks. Core component of the UDE discovery pipeline. See `equation-discovery` skill for algorithm details and usage patterns.

### SciMLSensitivity.jl (Gradient Through Solvers)

Compute gradients through differential equation solvers for optimization and training:

| Method | Use Case |
|--------|----------|
| `ForwardDiffSensitivity` | Few parameters (<100), forward-mode AD |
| `InterpolatingAdjoint` | Many parameters (neural networks), reverse-mode AD |

Choose forward sensitivity for small parameter spaces and adjoint methods for large-scale neural ODE/UDE training. See `sciml-modern-stack` skill for sensitivity method selection.

### UncertaintyQuantification.jl

Polynomial chaos expansions and global sensitivity analysis (Sobol indices) for quantifying how input uncertainties propagate through scientific models.

---

## Checklist

- [ ] Correct package for task selected
- [ ] Integration patterns understood
- [ ] Documentation reviewed

---

**Version**: 1.0.6
