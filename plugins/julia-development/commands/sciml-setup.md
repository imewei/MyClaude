---
version: "1.0.5"
category: "julia-development"
command: "/sciml-setup"
description: Interactive SciML project scaffolding with auto-detection of problem types (ODE, PDE, SDE, optimization) and generation of template code
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<problem-description>"
color: purple
execution_modes:
  quick: "5-10 minutes - Generate basic template for detected problem type"
  standard: "15-20 minutes - Generate comprehensive template with callbacks and examples"
  comprehensive: "25-35 minutes - Generate full template with ensemble, sensitivity analysis, and optimization guide"
agents:
  primary:
    - sciml-pro
  conditional:
    - agent: turing-pro
      trigger: pattern "bayesian|mcmc|posterior|prior|inference"
    - agent: julia-pro
      trigger: pattern "performance|optimization|type.*stability"
  orchestrated: false
---

# Interactive SciML Project Scaffolding

Auto-detect problem type from natural language description and generate working template code with DifferentialEquations.jl, ModelingToolkit.jl, or Optimization.jl.

## Quick Reference

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| **Complete Templates** | [sciml-templates.md](../docs/sciml-templates.md) | ~550 |
| **Solver Selection** | [sciml-templates.md#solver-selection-guide](../docs/sciml-templates.md#solver-selection-guide) | ~80 |
| **Best Practices** | [sciml-templates.md#best-practices](../docs/sciml-templates.md#best-practices) | ~100 |
| **Common Pitfalls** | [sciml-templates.md#common-pitfalls](../docs/sciml-templates.md#common-pitfalls) | ~80 |

**Total External Documentation**: ~550 lines of complete templates and guidance

## Requirements

$ARGUMENTS

## Core Workflow

### Phase 1: Problem Type Detection

**Auto-detect from natural language description** using keyword analysis:

**Detection Categories**:
1. **ODE** (Ordinary Differential Equations)
   - Keywords: "ordinary differential", "ODE", "dynamics", "time evolution", "population", "predator-prey", "chemical kinetics", "oscillator", "coupled system"

2. **PDE** (Partial Differential Equations)
   - Keywords: "partial differential", "PDE", "spatial", "diffusion", "heat equation", "wave equation", "boundary conditions", "Laplacian"

3. **SDE** (Stochastic Differential Equations)
   - Keywords: "stochastic", "SDE", "noise", "random", "Brownian", "uncertainty", "fluctuations"

4. **Optimization** (Parameter Estimation)
   - Keywords: "minimize", "maximize", "optimal", "parameter estimation", "fitting", "calibration", "inverse problem"

**Scoring**: Count keyword matches, select type with highest score.

**Ambiguous Cases**: If multiple types score similarly, present options to user.

### Phase 2: Modeling Approach Selection

**Prompt user** for approach preference:

1. **Symbolic ModelingToolkit.jl** (recommended for complex systems)
   - Automatic differentiation
   - Symbolic simplification
   - Component-based modeling
   - Easier debugging (equations visible)

2. **Direct API** (for simple systems or performance-critical code)
   - Less overhead
   - More explicit control
   - Faster compilation

**Default**: Symbolic for PDE, Direct for others (unless user specifies)

### Phase 3: Feature Selection

**Interactive prompts** (standard & comprehensive modes):

- **Callbacks**: Include event detection/termination examples? (yes/no)
- **Ensemble**: Include parameter/initial condition variations? (yes/no)
- **Sensitivity**: Include sensitivity analysis setup? (yes/no)

**Quick mode**: Skip prompts, generate basic template only.

### Phase 4: Template Generation

**Generate appropriate template** based on selections:

**Template Structure**:
```julia
# Header comments (problem description, auto-generated notice)
# Imports (DifferentialEquations, ModelingToolkit, etc.)
# Problem definition (ODE/PDE/SDE/Optimization)
# TODO comments for user customization
# Solver call with recommended algorithm
# Visualization boilerplate
# [Optional] Callbacks, ensemble, sensitivity sections
```

**File Output**:
- Save to: `<sanitized-description>_sciml.jl` or user-specified filename
- Print: Next steps and usage instructions

### Phase 5: Guidance & Next Steps

**Provide user** with:
1. **Customization checklist**: TODOs to fill in
2. **Solver recommendations**: Why this solver was chosen
3. **Next steps**: How to run, modify, and extend
4. **External docs reference**: Link to detailed templates and guides

## Mode-Specific Execution

### Quick Mode (5-10 minutes)

**Phases**: 1, 2 (auto-select Direct API), 4 (basic template), 5 (brief guidance)

**Output**: Minimal working template with core structure

**Skip**: Feature selection prompts, callback/ensemble/sensitivity sections

### Standard Mode (15-20 minutes) - DEFAULT

**Phases**: All 5 phases

**Output**: Comprehensive template with selected features

**Include**: Interactive prompts, basic callbacks/ensemble examples if selected

### Comprehensive Mode (25-35 minutes)

**Phases**: All 5 phases with extended guidance

**Output**: Full template with all optional features

**Include**:
- All callback types (continuous, discrete, termination)
- Complete ensemble setup (ThreadedEnsemble, parameter sampling)
- Forward and adjoint sensitivity analysis
- Performance tips and optimization guidance
- Links to detailed external documentation

## Template Examples

### ODE Template (Brief)

See [sciml-templates.md#ode-templates](../docs/sciml-templates.md#ode-templates) for:
- Direct API template with state dynamics
- Symbolic ModelingToolkit template
- Callback examples (termination, periodic)
- Ensemble simulation setup
- Sensitivity analysis integration

### PDE Template (Brief)

See [sciml-templates.md#pde-templates](../docs/sciml-templates.md#pde-templates) for:
- Method of Lines discretization
- Boundary and initial conditions
- Domain specification
- Integration with ODE solvers

### SDE Template (Brief)

See [sciml-templates.md#sde-templates](../docs/sciml-templates.md#sde-templates) for:
- Drift and diffusion functions
- Stochastic solver selection (SOSRI)
- Ensemble trajectories
- Noise parameter handling

### Optimization Template (Brief)

See [sciml-templates.md#optimization-templates](../docs/sciml-templates.md#optimization-templates) for:
- Loss function design
- Solver sensitivity configuration
- Parameter estimation workflow
- Data fitting examples

## Solver Selection

**Automatic solver recommendation** based on problem type:

| Problem Type | Recommended Solver | Reason |
|--------------|-------------------|--------|
| **ODE (non-stiff)** | `Tsit5()` | General-purpose, fast, accurate |
| **ODE (stiff)** | `Rodas5()` | Handles stiffness well |
| **PDE** | Method of Lines → ODE solver | Discretize spatially first |
| **SDE** | `SOSRI()` | Recommended for general SDEs |
| **Optimization** | `BFGS()` | Quasi-Newton, good default |

**Full guide**: [sciml-templates.md#solver-selection-guide](../docs/sciml-templates.md#solver-selection-guide)

## Success Criteria

✅ Problem type correctly detected from description
✅ Appropriate template generated (ODE/PDE/SDE/Optimization)
✅ Code is syntactically correct Julia
✅ TODO comments guide user customization
✅ Recommended solver included
✅ Code is runnable after TODOs filled
✅ Explanatory comments provided
✅ Optional features (callbacks, ensemble, sensitivity) included if selected
✅ External documentation referenced for detailed guidance

## Agent Integration

- **sciml-pro**: Primary agent for SciML template generation and solver selection
- **turing-pro**: Triggered for Bayesian parameter estimation problems (keywords: bayesian, mcmc, posterior)
- **julia-pro**: Triggered for performance optimization questions (keywords: performance, type stability)

## Post-Generation

After template is generated, guide user to:

1. **Customize**: Fill in TODO sections with problem-specific code
2. **Test**: Run template to verify correctness
3. **Extend**: Add features from external docs if needed
4. **Optimize**: Use `/julia-optimize` if performance is critical
5. **Document**: Add docstrings and examples

**See Also**:
- `/julia-optimize` - Profile and optimize generated code
- `/julia-scaffold` - Create package structure for SciML project
- [sciml-templates.md](../docs/sciml-templates.md) - Complete template library

---

Focus on **rapid scaffolding**, **correct defaults**, and **clear guidance** to transform problem descriptions into working SciML code.
