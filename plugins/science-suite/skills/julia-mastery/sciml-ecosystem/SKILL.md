---
name: sciml-ecosystem
version: "1.0.7"
maturity: "5-Expert"
specialization: SciML Overview
description: Navigate the SciML ecosystem including DifferentialEquations.jl, ModelingToolkit.jl, Optimization.jl, and Catalyst.jl. Use when selecting packages for scientific computing tasks.
---

# SciML Ecosystem

Overview of Scientific Machine Learning packages in Julia.

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

**Version**: 1.0.5
