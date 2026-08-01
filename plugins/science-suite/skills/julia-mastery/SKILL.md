---
name: julia-mastery
description: Master the Julia language for scientific computing. Covers multiple dispatch, type stability, metaprogramming, and the SciML ecosystem. Use when writing Julia code, optimizing type stability, designing multiple dispatch hierarchies, or integrating SciML packages.
---

# Julia Mastery

Expert guide for writing high-performance, idiomatic Julia code for scientific applications.

## Expert Agent

For complex Julia programming, SciML workflows, and performance optimization, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia optimization, including Core Julia, SciML (Lux.jl, UDEs), nonlinear dynamics (DynamicalSystems.jl, AUTO-07p -- BifurcationKit.jl blocked on Julia 1.12), Turing.jl, and Package Development.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: Performance tuning, stiff ODE solvers, Bayesian inference, UDEs, bifurcation analysis, equation discovery, and CI/CD setup.

## Core Skills

### [Core Julia Patterns](../core-julia-patterns/SKILL.md)
Multiple dispatch, type stability, and functional programming.

### [SciML Ecosystem](../sciml-ecosystem/SKILL.md)
Unified framework for scientific machine learning and modeling.

### [Differential Equations](../differential-equations/SKILL.md)
High-performance ODE, PDE, SDE, and DAE solvers.

### [ModelingToolkit](../modeling-toolkit/SKILL.md)
Symbolic-numeric modeling and acausal system design.

### [Neural PDE](../neural-pde/SKILL.md)
Physics-informed neural networks (PINNs) in Julia.

### [Turing Model Design](../turing-model-design/SKILL.md)
Bayesian inference and probabilistic programming.

### [Performance Tuning](../performance-tuning/SKILL.md)
Memory optimization, type stability, and benchmarking.

### [Package Development](../package-development-workflow/SKILL.md)
Scaffolding, testing, and CI/CD for Julia packages.

### [Variational Inference Patterns](../variational-inference-patterns/SKILL.md)
ADVI and approximate inference with Turing.jl.

### [Optimization Patterns](../optimization-patterns/SKILL.md)
Non-linear optimization and parameter estimation with Optimization.jl.

### [Jump Optimization](../jump-optimization/SKILL.md)
Mathematical programming (LP, QP, MIP) with JuMP.jl.

### [MCMC Diagnostics](../mcmc-diagnostics/SKILL.md)
Convergence checking and chain analysis.

### [Julia Testing Patterns](../julia-testing-patterns/SKILL.md)
Robust testing strategies with ReTestItems.jl and Aqua.jl.

### [Package Management](../package-management/SKILL.md)
Environment management and Pkg.jl workflows.

### [Catalyst Reactions](../catalyst-reactions/SKILL.md)
Chemical reaction network modeling.

### [Visualization Patterns](../visualization-patterns/SKILL.md)
Data visualization with Makie.jl and Plots.jl.

### [Web Development](../web-development-julia/SKILL.md)
Building scientific web services with Genie.jl/Oxygen.jl.

### [Modern SciML Stack](../sciml-modern-stack/SKILL.md)
Lux.jl, Universal Differential Equations (UDEs), SciMLSensitivity.jl, NeuralPDE.jl v5+.

### [Nonlinear Dynamics](../nonlinear-dynamics/SKILL.md)
Bifurcation analysis, chaos, network dynamics, pattern formation, equation discovery.

### [Interop Patterns](../interop-patterns/SKILL.md)
Calling Python and R from Julia.

### [CI/CD Patterns](../ci-cd-patterns/SKILL.md)
GitHub Actions for Julia packages.

### [Compiler Patterns](../compiler-patterns/SKILL.md)
PackageCompiler.jl and system images.

### [Parallel Computing](../parallel-computing/SKILL.md)
Multi-threading and distributed computing patterns.

## Routing Decision Tree

```
What is the primary Julia task?
|
+-- Core language: dispatch, type system, metaprogramming, or functional patterns?
|   --> science-suite:core-julia-patterns
|
+-- SciML ecosystem overview or package selection?
|   --> science-suite:sciml-ecosystem
|
+-- ODE / PDE / SDE / DAE solvers?
|   --> science-suite:differential-equations
|
+-- Symbolic-numeric modeling with ModelingToolkit?
|   --> science-suite:modeling-toolkit
|
+-- Physics-informed neural networks (PINNs) in Julia?
|   --> science-suite:neural-pde
|
+-- Bayesian inference or probabilistic programming with Turing.jl?
|   --> science-suite:turing-model-design
|
+-- Memory, type-stability, benchmarking, or allocation profiling?
|   --> science-suite:performance-tuning
|
+-- Package scaffolding, testing, or CI/CD setup?
|   --> science-suite:package-development-workflow
|
+-- Variational inference or ADVI with Turing.jl?
|   --> science-suite:variational-inference-patterns
|
+-- Non-linear optimization or parameter estimation with Optimization.jl?
|   --> science-suite:optimization-patterns
|
+-- Mathematical programming (LP / QP / MIP) with JuMP.jl?
|   --> science-suite:jump-optimization
|
+-- MCMC diagnostics, R-hat, ESS, or chain convergence?
|   --> science-suite:mcmc-diagnostics
|
+-- Writing tests with ReTestItems.jl or Aqua.jl?
|   --> science-suite:julia-testing-patterns
|
+-- Environment management or Pkg.jl workflows?
|   --> science-suite:package-management
|
+-- Chemical reaction network modeling with Catalyst.jl?
|   --> science-suite:catalyst-reactions
|
+-- Data visualization with Makie.jl or Plots.jl?
|   --> science-suite:visualization-patterns
|
+-- Scientific web services with Genie.jl or Oxygen.jl?
|   --> science-suite:web-development-julia
|
+-- Lux.jl neural networks, UDEs, or SciMLSensitivity?
|   --> science-suite:sciml-modern-stack
|
+-- Bifurcation analysis, chaos, network dynamics, or equation discovery?
|   --> science-suite:nonlinear-dynamics
|
+-- Calling Python or R from Julia?
|   --> science-suite:interop-patterns
|
+-- GitHub Actions for Julia packages?
|   --> science-suite:ci-cd-patterns
|
+-- PackageCompiler.jl or system images?
|   --> science-suite:compiler-patterns
|
+-- Multi-threading or distributed computing patterns?
|   --> science-suite:parallel-computing
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to julia-pro for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## 1. Multiple Dispatch & Type System

- **Multiple Dispatch**: Design functions that specialize based on all argument types.
- **Abstract Types**: Use abstract types (e.g., `AbstractVector`) in function signatures for flexibility.
- **Concrete Types**: Use concrete types in struct fields to ensure type stability and performance.
- **Parametric Types**: Write generic code that specializes at compile-time for specific types.

## 2. Performance Optimization

- **Type Stability**: Use `@code_warntype` to detect instabilities (red/pink output). Ensure return types are predictable.
- **Allocations**: Minimize allocations in hot loops by preallocating arrays and using mutating functions (ending in `!`).
- **SIMD & Inbounds**: Use `@simd` and `@inbounds` (after safety checks) to maximize loop performance.
- **StaticArrays**: Use `StaticArrays.jl` for small, fixed-size vectors and matrices to enable stack allocation.

## 3. The SciML Ecosystem

- **Lux.jl**: Modern neural networks with explicit parameterization (replaces Flux for SciML). See `sciml-modern-stack` skill.
- **DifferentialEquations.jl**: State-of-the-art solvers for ODEs, PDEs, SDEs, and DAEs.
- **ModelingToolkit.jl**: Symbolic-numeric modeling for simplifying complex systems.
- **SciMLSensitivity.jl**: Adjoint and forward sensitivity analysis through solvers.
- **Optimization.jl**: Unified interface for local and global optimization.
- **DataDrivenDiffEq.jl**: SINDy and equation discovery from data. See `equation-discovery` skill.
- **NeuralPDE.jl**: Physics-informed neural networks (PINNs) for solving PDEs.

## 4. Julia Development Workflow

- **Profiling**: Use `BenchmarkTools.jl` for timing and `ProfileView.jl` for bottleneck identification.
- **Metaprogramming**: Use macros (`@macro`) for code generation, but prefer functions when possible. Ensure macros use `esc()` to avoid hygiene issues.
- **Package Management**: Use `Pkg` for managing environments and dependencies.
- **Parallelism**: Leverage `Threads.@threads` for shared memory and `Distributed` for multi-node parallelism.

## Checklist

- [ ] Verify type stability with `@code_warntype` on all performance-critical functions
- [ ] Confirm struct fields use concrete types (not abstract) for allocation efficiency
- [ ] Check that hot loops preallocate output arrays and use mutating functions (ending in `!`)
- [ ] Validate multiple dispatch design: methods specialize on all argument types as intended
- [ ] Ensure `@simd` and `@inbounds` are applied only after correctness is verified
- [ ] Use `BenchmarkTools.@btime` (not `@time`) for accurate performance measurements
- [ ] Confirm `Manifest.toml` is committed for reproducible environments
- [ ] Check that macros use `esc()` correctly to avoid hygiene issues
- [ ] Validate SciML solver selection matches problem stiffness and accuracy requirements
