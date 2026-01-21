---
name: julia-mastery
version: "2.1.0"
description: Master the Julia language for scientific computing. Covers multiple dispatch, type stability, metaprogramming, and the SciML ecosystem.
---

# Julia Mastery

Expert guide for writing high-performance, idiomatic Julia code for scientific applications.

## Expert Agent

For complex Julia programming, SciML workflows, and performance optimization, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia optimization, including Core Julia, SciML, Turing.jl, and Package Development.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: Performance tuning, stiff ODE solvers, Bayesian inference, and CI/CD setup.

## Core Skills

### [Core Julia Patterns](./core-julia-patterns/SKILL.md)
Multiple dispatch, type stability, and functional programming.

### [SciML Ecosystem](./sciml-ecosystem/SKILL.md)
Unified framework for scientific machine learning and modeling.

### [Differential Equations](./differential-equations/SKILL.md)
High-performance ODE, PDE, SDE, and DAE solvers.

### [ModelingToolkit](./modeling-toolkit/SKILL.md)
Symbolic-numeric modeling and acausal system design.

### [Neural PDE](./neural-pde/SKILL.md)
Physics-informed neural networks (PINNs) in Julia.

### [Turing Model Design](./turing-model-design/SKILL.md)
Bayesian inference and probabilistic programming.

### [Performance Tuning](./performance-tuning/SKILL.md)
Memory optimization, type stability, and benchmarking.

### [Package Development](./package-development-workflow/SKILL.md)
Scaffolding, testing, and CI/CD for Julia packages.

### [Variational Inference Patterns](./variational-inference-patterns/SKILL.md)
ADVI and approximate inference with Turing.jl.

### [Optimization Patterns](./optimization-patterns/SKILL.md)
Non-linear optimization and parameter estimation with Optimization.jl.

### [Jump Optimization](./jump-optimization/SKILL.md)
Mathematical programming (LP, QP, MIP) with JuMP.jl.

### [MCMC Diagnostics](./mcmc-diagnostics/SKILL.md)
Convergence checking and chain analysis.

### [Testing Patterns](./testing-patterns/SKILL.md)
Robust testing strategies with ReTestItems.jl and Aqua.jl.

### [Package Management](./package-management/SKILL.md)
Environment management and Pkg.jl workflows.

### [Catalyst Reactions](./catalyst-reactions/SKILL.md)
Chemical reaction network modeling.

### [Visualization Patterns](./visualization-patterns/SKILL.md)
Data visualization with Makie.jl and Plots.jl.

### [Web Development](./web-development-julia/SKILL.md)
Building scientific web services with Genie.jl/Oxygen.jl.

### [Interop Patterns](./interop-patterns/SKILL.md)
Calling Python and R from Julia.

### [CI/CD Patterns](./ci-cd-patterns/SKILL.md)
GitHub Actions for Julia packages.

### [Compiler Patterns](./compiler-patterns/SKILL.md)
PackageCompiler.jl and system images.

### [Parallel Computing](./parallel-computing/SKILL.md)
Multi-threading and distributed computing patterns.

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

- **DifferentialEquations.jl**: State-of-the-art solvers for ODEs, PDEs, SDEs, and DAEs.
- **ModelingToolkit.jl**: Symbolic-numeric modeling for simplifying complex systems.
- **Optimization.jl**: Unified interface for local and global optimization.
- **NeuralPDE.jl**: Physics-informed neural networks (PINNs) for solving PDEs.

## 4. Julia Development Workflow

- **Profiling**: Use `BenchmarkTools.jl` for timing and `ProfileView.jl` for bottleneck identification.
- **Metaprogramming**: Use macros (`@macro`) for code generation, but prefer functions when possible. Ensure macros use `esc()` to avoid hygiene issues.
- **Package Management**: Use `Pkg` for managing environments and dependencies.
- **Parallelism**: Leverage `Threads.@threads` for shared memory and `Distributed` for multi-node parallelism.
