---
name: sciml-ecosystem
version: "2.2.1"
maturity: "5-Expert"
specialization: SciML Overview
description: Navigate the SciML ecosystem including DifferentialEquations.jl, ModelingToolkit.jl, Optimization.jl, and Catalyst.jl. Use when selecting packages for scientific computing tasks.
---

# SciML Ecosystem

Overview of Scientific Machine Learning packages in Julia.

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

## Checklist

- [ ] Correct package for task selected
- [ ] Integration patterns understood
- [ ] Documentation reviewed

---

**Version**: 1.0.6
