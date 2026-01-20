---
name: julia-mastery
version: "1.0.1"
description: Master the Julia language for scientific computing. Covers multiple dispatch, type stability, metaprogramming, and the SciML ecosystem.
---

# Julia Mastery

Expert guide for writing high-performance, idiomatic Julia code for scientific applications.

## Expert Agent

For complex Julia programming, SciML workflows, and performance optimization, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia optimization, including Core Julia, SciML, Turing.jl, and Package Development.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: Performance tuning, stiff ODE solvers, Bayesian inference, and CI/CD setup.

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
