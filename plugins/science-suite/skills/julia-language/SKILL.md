---
name: julia-language
description: Meta-orchestrator for Julia language and ecosystem. Routes to core patterns, packages, compilation, performance, testing, CI/CD, visualization, and interop skills. Use when writing Julia code, creating, testing, documenting, or auditing a Julia package (Project.toml, Documenter, exports, precompilation, JET), profiling performance, setting up CI/CD, or building cross-language workflows. For SciML/DiffEq use sciml-and-diffeq; for Lux or GPU use julia-ml-and-dl; for dispatch, type, and metaprogramming idioms use julia-mastery.
---

> **Loading a routing target.** `dev-suite:*`, `research-suite:*`, and `science-suite:*` targets below
> are slash-only (`disable-model-invocation: true`) — the Skill tool will not fire them. Load one by
> reading its file with the Read tool at `${CLAUDE_PLUGIN_ROOT}/skills/<name>/SKILL.md` — Claude Code
> substitutes that variable with this plugin's install directory. A target in a sibling suite cannot be
> read by path (installs live in separate directories); reach it by invoking that suite's hub. Targets from other plugins (`superpowers:*`,
> `ecc:*`, …) are unaffected and invoke normally.

# Julia Language

Orchestrator for Julia language development. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`julia-implementer`**: Lead for the package lifecycle skills (generating, developing, testing, documenting, auditing, exports, precompiling, profiling, debugging, apps, JuliaCheck).
  - *Location*: `${CLAUDE_PLUGIN_ROOT}/agents/julia-implementer.md`
- **`julia-pro`**: Specialist for Julia language, package ecosystem, and performance engineering.
  - *Location*: `${CLAUDE_PLUGIN_ROOT}/agents/julia-pro.md`
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

### [Generating Julia Package](../generating-julia-package/SKILL.md)
Project environments, package layouts, UUIDs, and pre-registration checks.

### [Developing Julia Package](../developing-julia-package/SKILL.md)
Idiomatic package code: module layout, dispatch, types, formatting, docstrings.

### [Auditing Julia Package Practices](../auditing-julia-package-practices/SKILL.md)
Evidence-based package review with prioritized findings and go/no-go.

### [Documenting Julia Package](../documenting-julia-package/SKILL.md)
Documenter.jl: makedocs, doctests, GitHub Pages deployment.

### [Fixing Julia Exports](../fixing-julia-exports/SKILL.md)
Trim a package's public API: explicit test imports, `public`, deprecation.

### [Precompiling Julia Package](../precompiling-julia-package/SKILL.md)
Measure and reduce load/first-call latency with PrecompileTools.jl.

### [Profiling Julia Performance](../profiling-julia-performance/SKILL.md)
Diagnose slow runtime, type instability, allocations with Profile, JET.jl, AllocCheck.jl.

### [Debugging Julia](../debugging-julia/SKILL.md)
Non-interactive diagnosis of exceptions and wrong results: traces, `@code_typed`, JET.

### [Creating Julia App](../creating-julia-app/SKILL.md)
CLI apps and MCP servers with `@main` and `[apps]`.

### [Testing Julia Package](../testing-julia-package/SKILL.md)
Test projects (workspace or extras), `Pkg.test()`, focused and parallel runs.

### [Using JuliaCheck](../using-juliacheck/SKILL.md)
Rule-based static checks with JuliaCheck.jl.

## Routing Decision Tree

```
What is the Julia task?
|
+-- Language patterns / type system / dispatch?
|   --> science-suite:julia-mastery or science-suite:core-julia-patterns
|
+-- Package environment / dependency management?
|   --> science-suite:package-management
|
+-- Creating or publishing a package?
|   --> science-suite:package-development-workflow
|
+-- Compiler / type inference / specialization?
|   --> science-suite:compiler-patterns
|
+-- Profiling / allocation / SIMD?
|   --> science-suite:performance-tuning
|
+-- Testing / CI?
|   --> science-suite:julia-testing-patterns / science-suite:ci-cd-patterns
|
+-- Package lifecycle (create / develop / test / document / audit)?
|   --> science-suite:generating-julia-package, developing-julia-package,
|       testing-julia-package, documenting-julia-package,
|       auditing-julia-package-practices, fixing-julia-exports
|
+-- Slow first load / first call (TTFL, TTFX)?
|   --> science-suite:precompiling-julia-package
|
+-- Slow steady-state runtime / type instability / allocations?
|   --> science-suite:profiling-julia-performance
|
+-- Exception, wrong result, or crash to diagnose?
|   --> science-suite:debugging-julia
|
+-- CLI app, MCP server, or static rule checks?
|   --> science-suite:creating-julia-app / science-suite:using-juliacheck
|
+-- Plotting / visualization?
|   --> science-suite:visualization-patterns
|
+-- Web services?
|   --> science-suite:web-development-julia
|
+-- Multi-node / MPI / SLURM / distributed?
|   --> science-suite:julia-hpc-distributed
|
+-- Cross-language interop?
|   --> science-suite:interop-patterns
|
+-- Which package to use?
|   --> science-suite:ecosystem-selection
|
+-- None of the above / concern is ambiguous?
    --> Delegate to julia-pro for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Checklist

- [ ] Use routing tree to select the most specific sub-skill
- [ ] Confirm Julia version compatibility (LTS vs current stable)
- [ ] Check type stability with `@code_warntype` before optimizing
- [ ] Use explicit environments (`Project.toml`) — never the global env
- [ ] Prefer in-place (`!`) functions for large array operations
- [ ] Validate package dependencies are pinned in `Manifest.toml`
- [ ] Run tests in a clean environment before publishing
