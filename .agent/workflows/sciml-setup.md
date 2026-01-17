---
description: Interactive SciML project scaffolding with auto-detection of problem
  types (ODE, PDE, SDE, optimization) and generation of template code
triggers:
- /sciml-setup
- interactive sciml project scaffolding
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<problem-description>`
The agent should parse these arguments from the user's request.

# Interactive SciML Project Scaffolding

Auto-detect problem type and generate template code with DifferentialEquations.jl, ModelingToolkit.jl, or Optimization.jl.

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

**Docs**: [sciml-templates.md](../../plugins/julia-development/docs/sciml-templates.md) (~550 lines of complete templates)

## Requirements

$ARGUMENTS

## Workflow

### 1. Detect Problem Type (Analysis)

Auto-detect from keywords:
- **ODE**: "ordinary", "ODE", "dynamics", "population", "oscillator"
- **PDE**: "partial", "PDE", "spatial", "diffusion", "heat", "wave"
- **SDE**: "stochastic", "SDE", "noise", "Brownian"
- **Optimization**: "minimize", "maximize", "fitting", "calibration"

### 2. Select Approach (Strategy)

- **Symbolic** (ModelingToolkit): Complex systems, automatic differentiation
- **Direct API**: Simple systems, performance-critical

### 3. Generate Template (Implementation)

Structure:
```julia
# Imports
# Problem definition (ODE/PDE/SDE/Optimization)
# TODO comments for customization
# Solver call
# Visualization
# [Optional] Callbacks, ensemble, sensitivity
```

### 4. Solver Recommendation (Validation)

| Problem | Solver | Reason |
|---------|--------|--------|
| ODE (non-stiff) | `Tsit5()` | Fast, accurate |
| ODE (stiff) | `Rodas5()` | Handles stiffness |
| PDE | Method of Lines → ODE | Spatial discretization |
| SDE | `SOSRI()` | General SDEs |
| Optimization | `BFGS()` | Quasi-Newton |

### 5. Next Steps

1. Fill TODO sections
2. Run template
3. Use `/julia-optimize` for performance
4. See [sciml-templates.md](../../plugins/julia-development/docs/sciml-templates.md) for detailed examples

**Outcome**: Working SciML template with recommended solvers and clear customization points
