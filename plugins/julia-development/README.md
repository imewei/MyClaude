# Julia Development

Comprehensive Julia development plugin with specialized agents for high-performance computing, package development, scientific machine learning (SciML), and Bayesian inference. Expert guidance for building robust Julia applications with optimization, monitoring, and deep learning capabilities.

**Version:** 1.0.0 | **Category:** scientific-computing | **License:** MIT

[Full Documentation →](https://docs.example.com/plugins/julia-development.html)

## Agents (4)

### julia-pro

**Status:** active

General Julia programming expert for high-performance computing, scientific simulations, data analysis, and machine learning. Master of multiple dispatch, type system, metaprogramming, JuMP optimization, and Julia ecosystem.

### julia-developer

**Status:** active

Package development specialist for creating robust Julia packages. Expert in testing patterns, CI/CD automation, PackageCompiler.jl, web development (Genie.jl), and integrating optimization, monitoring, and deep learning components.

### sciml-pro

**Status:** active

SciML ecosystem expert for scientific machine learning and differential equations. Master of DifferentialEquations.jl, ModelingToolkit.jl, Optimization.jl, NeuralPDE.jl, Catalyst.jl, performance tuning, and parallel computing.

### turing-pro

**Status:** active

Bayesian inference and probabilistic programming expert. Master of Turing.jl, MCMC methods, variational inference (ADVI), model comparison, convergence diagnostics, and integration with SciML for Bayesian ODEs.

## Commands (4)

### `sciml-setup`

**Status:** active

Interactive SciML project scaffolding with auto-detection of problem types (ODE, PDE, SDE, optimization). Generates template code with callbacks, ensemble simulations, and sensitivity analysis.

### `julia-optimize`

**Status:** active

Profile Julia code and provide optimization recommendations. Analyzes type stability, memory allocations, identifies bottlenecks, and suggests parallelization strategies.

### `julia-scaffold`

**Status:** active

Bootstrap new Julia package with proper structure following PkgTemplates.jl conventions. Creates Project.toml, testing infrastructure, documentation framework, and git repository.

### `julia-package-ci`

**Status:** active

Generate GitHub Actions CI/CD workflows for Julia packages. Configures testing matrices across Julia versions and platforms, coverage reporting, and documentation deployment.

## Skills (21)

### core-julia-patterns

Multiple dispatch, type system, parametric types, metaprogramming, type stability, and performance optimization fundamentals

### jump-optimization

Mathematical programming with JuMP.jl modeling patterns, constraints, objectives, solver selection (separate from Optimization.jl)

### visualization-patterns

Plotting with Plots.jl, Makie.jl, StatsPlots.jl for data visualization and scientific graphics

### interop-patterns

Python interop via PythonCall.jl, R via RCall.jl, C++ via CxxWrap.jl for cross-language integration

### package-management

Project.toml structure, Pkg.jl workflows, dependency management, semantic versioning

### package-development-workflow

Package structure, module organization, exports, PkgTemplates.jl conventions, documentation

### testing-patterns

Test.jl best practices, test organization, BenchmarkTools.jl, Aqua.jl quality checks, JET.jl static analysis

### compiler-patterns

PackageCompiler.jl for static compilation, creating executables, system images, deployment optimization

### web-development-julia

Genie.jl MVC framework, HTTP.jl server development, API patterns, JSON3.jl, Oxygen.jl lightweight APIs

### ci-cd-patterns

GitHub Actions for Julia, test matrices, CompatHelper.jl, TagBot.jl, documentation deployment

### sciml-ecosystem

SciML package integration: DifferentialEquations.jl, ModelingToolkit.jl, Catalyst.jl, solver selection

### differential-equations

ODE, PDE, SDE, DAE solving patterns with callbacks, ensemble simulations, sensitivity analysis

### modeling-toolkit

Symbolic problem definition with ModelingToolkit.jl, equation simplification, code generation

### optimization-patterns

Optimization.jl usage for SciML optimization (distinct from JuMP.jl mathematical programming)

### neural-pde

Physics-informed neural networks (PINNs) with NeuralPDE.jl, boundary conditions, training strategies

### catalyst-reactions

Reaction network modeling with Catalyst.jl, rate laws, species definitions, stochastic vs deterministic

### performance-tuning

Profiling with @code_warntype, @profview, BenchmarkTools.jl, allocation reduction, type stability analysis

### parallel-computing

Multi-threading, Distributed.jl, GPU computing with CUDA.jl, ensemble simulations, load balancing

### turing-model-design

Turing.jl model specification, prior selection, likelihood definition, hierarchical models, identifiability

### mcmc-diagnostics

MCMC convergence checking (trace plots, R-hat), effective sample size, divergence checking, mixing analysis

### variational-inference-patterns

ADVI with Turing.jl, Bijectors.jl transformations, ELBO monitoring, VI vs MCMC comparison

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `julia-development` plugin
3. Activate an agent (e.g., `@julia-pro`)
4. Try a command (e.g., `sciml-setup`)

## Integration

See the full documentation for integration patterns and compatible plugins.

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://docs.example.com/plugins/julia-development.html)

To build documentation locally:

```bash
cd docs/
make html
```
