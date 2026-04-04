---
name: julia-language
description: Meta-orchestrator for Julia language and ecosystem. Routes to core patterns, packages, compilation, performance, testing, CI/CD, visualization, and interop skills. Use when writing Julia code, managing packages, optimizing performance, writing tests, setting up CI/CD, or building cross-language workflows.
---

# Julia Language

Orchestrator for Julia language development. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`julia-pro`**: Specialist for Julia language, package ecosystem, and performance engineering.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: Type system, multiple dispatch, metaprogramming, package development, compiler internals.

## Core Skills

### [Julia Mastery](../julia-mastery/SKILL.md)
Core Julia idioms: multiple dispatch, type system, metaprogramming, and performance patterns.

### [Core Julia Patterns](../core-julia-patterns/SKILL.md)
Idiomatic Julia: broadcasting, comprehensions, closures, and standard library usage.

### [Package Management](../package-management/SKILL.md)
Pkg.jl workflows: environments, registries, Manifest.toml, and reproducible setups.

### [Package Development Workflow](../package-development-workflow/SKILL.md)
Creating and publishing packages: PkgTemplates.jl, documentation, versioning, and registration.

### [Compiler Patterns](../compiler-patterns/SKILL.md)
Julia compiler internals: `@code_typed`, `@code_llvm`, type inference, and specialization.

### [Performance Tuning](../performance-tuning/SKILL.md)
Profiling with `@btime`, memory allocation, SIMD, threading, and type stability.

### [Julia Testing Patterns](../julia-testing-patterns/SKILL.md)
Test.jl, TestEnv.jl, property-based testing, and test organization.

### [CI/CD Patterns](../ci-cd-patterns/SKILL.md)
GitHub Actions for Julia: test matrix, code coverage, and automated releases.

### [Visualization Patterns](../visualization-patterns/SKILL.md)
Makie.jl, Plots.jl, interactive visualization, and publication-quality figures.

### [Web Development Julia](../web-development-julia/SKILL.md)
Genie.jl, HTTP.jl, REST APIs, and web services in Julia.

### [Julia HPC Distributed](../julia-hpc-distributed/SKILL.md)
Multi-node Julia: Distributed.jl, MPI.jl, SLURM job management, and pmap/remotecall patterns.

### [Interop Patterns](../interop-patterns/SKILL.md)
PyCall.jl, RCall.jl, ccall, and cross-language data exchange.

### [Ecosystem Selection](../ecosystem-selection/SKILL.md)
Choosing the right Julia packages for a given domain or task.

## Routing Decision Tree

```
What is the Julia task?
|
+-- Language patterns / type system / dispatch?
|   --> julia-mastery or core-julia-patterns
|
+-- Package environment / dependency management?
|   --> package-management
|
+-- Creating or publishing a package?
|   --> package-development-workflow
|
+-- Compiler / type inference / specialization?
|   --> compiler-patterns
|
+-- Profiling / allocation / SIMD?
|   --> performance-tuning
|
+-- Testing / CI?
|   --> julia-testing-patterns / ci-cd-patterns
|
+-- Plotting / visualization?
|   --> visualization-patterns
|
+-- Web services?
|   --> web-development-julia
|
+-- Multi-node / MPI / SLURM / distributed?
|   --> julia-hpc-distributed
|
+-- Cross-language interop?
|   --> interop-patterns
|
+-- Which package to use?
    --> ecosystem-selection
```

## Checklist

- [ ] Use routing tree to select the most specific sub-skill
- [ ] Confirm Julia version compatibility (LTS vs current stable)
- [ ] Check type stability with `@code_warntype` before optimizing
- [ ] Use explicit environments (`Project.toml`) — never the global env
- [ ] Prefer in-place (`!`) functions for large array operations
- [ ] Validate package dependencies are pinned in `Manifest.toml`
- [ ] Run tests in a clean environment before publishing
