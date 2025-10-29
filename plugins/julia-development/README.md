# Julia Development Plugin

Comprehensive Julia development plugin providing specialized agents for high-performance computing, package development, scientific machine learning (SciML), and Bayesian inference. Build robust Julia applications with intelligent optimization, monitoring, and deep learning capabilities leveraging Julia's JIT compilation, multiple dispatch, and extensive scientific ecosystem.

## Overview

This plugin provides four specialized agents and four prioritized commands to support the full Julia development lifecycle, from initial project scaffolding through performance optimization and package publishing.

### Agents

**julia-pro** - General Julia programming expert
- Core Julia patterns: multiple dispatch, type system, metaprogramming
- High-performance computing, scientific simulations, data analysis, machine learning
- JuMP.jl mathematical optimization
- Visualization with Plots.jl and Makie.jl
- Interoperability patterns (Python, R, C++)
- Package management and Project.toml handling

**julia-developer** - Package development specialist
- Package structure creation and organization
- Testing patterns with Test.jl, Aqua.jl, JET.jl
- CI/CD automation with GitHub Actions
- PackageCompiler.jl for executable creation
- Web development with Genie.jl and HTTP.jl
- Integration of optimization, monitoring, and deep learning components

**sciml-pro** - SciML ecosystem expert
- DifferentialEquations.jl: ODE, PDE, SDE, DAE solving
- ModelingToolkit.jl symbolic computing
- Optimization.jl (distinct from JuMP.jl)
- NeuralPDE.jl physics-informed neural networks
- Catalyst.jl reaction network modeling
- Performance tuning and parallel computing (threads, distributed, GPU)

**turing-pro** - Bayesian inference expert
- Turing.jl probabilistic programming
- MCMC methods with comprehensive diagnostics
- Variational inference (ADVI, Bijectors.jl)
- Model comparison (WAIC, LOO, Bayes factors)
- Prior and posterior predictive checks
- Integration with SciML for Bayesian ODEs

### Commands

#### Priority 1: /sciml-setup
Interactive SciML project scaffolding with auto-detection of problem types.

```julia
# Auto-detect and generate ODE solver template
/sciml-setup "coupled oscillator system"

# Generates working code with:
# - Problem definition using DifferentialEquations.jl
# - Appropriate solver selection
# - Callback examples for event handling
# - Ensemble simulation templates
# - Sensitivity analysis setup
```

**Features:**
- Auto-detects problem type: ODE, PDE, SDE, optimization
- Supports symbolic (ModelingToolkit.jl) and direct definitions
- Interactive prompts for configuration choices
- Generates runnable template code with explanatory comments

#### Priority 2: /julia-optimize
Profile Julia code and provide optimization recommendations.

```julia
# Analyze performance bottlenecks
/julia-optimize path/to/slow_function.jl

# Provides:
# - Type stability analysis with @code_warntype
# - Memory allocation profiling
# - Execution profiling with @profview
# - Ranked list of optimization opportunities
# - Parallelization suggestions
# - Before/after performance estimates
```

**Analysis includes:**
- Type instabilities with suggested fixes
- Allocation hotspots with reduction strategies
- Parallelization opportunities (threads, distributed, GPU)
- Algorithm improvement suggestions

#### Priority 3: /julia-scaffold
Bootstrap new Julia package with proper structure.

```julia
# Create new package
/julia-scaffold "MyAwesomePackage"

# Generates:
# - Project.toml with dependencies and compatibility bounds
# - src/ with module file
# - test/ with Test.jl infrastructure
# - docs/ with Documenter.jl setup
# - README.md with badges and quick start
# - LICENSE and .gitignore
```

**Follows:**
- PkgTemplates.jl conventions
- Julia community standards
- Publication-ready structure

#### Priority 4: /julia-package-ci
Generate GitHub Actions CI/CD workflows.

```julia
# Add CI/CD to existing package
/julia-package-ci

# Creates workflows:
# - .github/workflows/CI.yml: Test matrix (versions, platforms)
# - .github/workflows/Documentation.yml: Docs deployment
# - .github/workflows/CompatHelper.yml: Dependency updates
# - .github/workflows/TagBot.yml: Automated releases
```

**Configuration:**
- Test matrices across Julia versions (1.6+, nightly)
- Cross-platform testing (Linux, macOS, Windows)
- Code coverage with Codecov.jl
- Documentation deployment to GitHub Pages

## Quick Start

### Workflow 1: Scientific Computing Project

```julia
# 1. Scaffold SciML project
/sciml-setup "system of ODEs for population dynamics"

# 2. Implement your model (sciml-pro provides guidance)
# ... development ...

# 3. Optimize performance if needed
/julia-optimize src/dynamics.jl

# Achieves: Working ODE solver in <2 minutes
```

### Workflow 2: Package Development

```julia
# 1. Create package structure
/julia-scaffold "MyPackage"

# 2. Develop functionality (julia-pro provides patterns)
# ... implementation ...

# 3. Add CI/CD
/julia-package-ci

# 4. Register package (following generated instructions)
# Achieves: Publication-ready package in <10 minutes
```

### Workflow 3: Bayesian Model Implementation

```julia
# 1. Consult turing-pro for model design
# "Help me implement a hierarchical Bayesian model"

# 2. Implement model with agent guidance
# ... Turing.jl code ...

# 3. Get diagnostic recommendations
# "Check my MCMC convergence"

# 4. Consider variational inference if MCMC is slow
# "Should I use VI instead?"

# Achieves: Correctly specified Bayesian model with diagnostics
```

## Skills Overview

### julia-pro Skills
- **core-julia-patterns**: Multiple dispatch, type system, metaprogramming
- **jump-optimization**: JuMP.jl mathematical optimization (separate from Optimization.jl)
- **visualization-patterns**: Plots.jl, Makie.jl, StatsPlots.jl
- **interop-patterns**: PythonCall.jl, RCall.jl, CxxWrap.jl
- **package-management**: Project.toml, Pkg.jl workflows

### julia-developer Skills
- **package-development-workflow**: Package structure and organization
- **testing-patterns**: Test.jl, BenchmarkTools.jl, Aqua.jl
- **compiler-patterns**: PackageCompiler.jl for executables
- **web-development-julia**: Genie.jl MVC, HTTP.jl servers
- **ci-cd-patterns**: GitHub Actions, CompatHelper, TagBot

### sciml-pro Skills
- **sciml-ecosystem**: DifferentialEquations.jl, ModelingToolkit.jl, Catalyst.jl
- **differential-equations**: ODE, PDE, SDE solving with callbacks
- **modeling-toolkit**: Symbolic problem definition
- **optimization-patterns**: Optimization.jl (distinct from JuMP.jl)
- **neural-pde**: Physics-informed neural networks
- **catalyst-reactions**: Reaction network modeling
- **performance-tuning**: Profiling and optimization
- **parallel-computing**: Threads, distributed, GPU

### turing-pro Skills
- **turing-model-design**: Model specification, prior selection, hierarchical models
- **mcmc-diagnostics**: Convergence checking, R-hat, effective sample size
- **variational-inference-patterns**: ADVI, Bijectors.jl, ELBO monitoring

## Integration with Other Plugins

This plugin integrates with:

**deep-learning plugin**
- Neural network patterns applicable to Flux.jl and NeuralPDE.jl
- Training diagnostics for DiffEqFlux.jl
- Model optimization for scientific ML

**jax-implementation plugin**
- Automatic differentiation concepts parallel to Zygote.jl
- Functional programming patterns similar to Julia's style
- Scientific computing workflows for physics-informed ML

**hpc-computing plugin**
- Parallel computing strategies applicable to Julia
- GPU acceleration patterns for CUDA.jl
- HPC workflow strategies for large-scale simulations
- Julia/SciML ecosystem foundation already established

## Package Ecosystem Coverage

### Core Julia
- Base: LinearAlgebra, Statistics, Random, Distributed, Threads
- Package management: Pkg, PkgTemplates, Revise
- Testing: Test, Aqua, JET
- Documentation: Documenter, DocStringExtensions

### SciML Ecosystem (sciml-pro)
- DifferentialEquations, OrdinaryDiffEq, StochasticDiffEq
- ModelingToolkit, Symbolics
- Optimization, OptimizationOptimJL
- NeuralPDE, DiffEqFlux
- SciMLSensitivity
- Catalyst
- DataDrivenDiffEq

### Mathematical Optimization (julia-pro)
- JuMP: Mathematical programming
- Optim: Pure Julia optimization
- Solvers: GLPK, Ipopt, COSMO, HiGHS

### Bayesian Inference (turing-pro)
- Turing: Probabilistic programming
- MCMCChains: Diagnostics and visualization
- Bijectors: Variational inference transformations

### Machine Learning (julia-pro)
- Flux: Deep learning
- MLJ: Machine learning interface
- MLUtils: ML utilities

### Visualization (julia-pro)
- Plots: Unified plotting interface
- Makie: High-performance visualization
- StatsPlots: Statistical plotting

### Web Development (julia-developer)
- Genie: MVC web framework
- HTTP: Server and client
- JSON3: JSON parsing
- Oxygen: Lightweight web framework

### Performance (all agents)
- BenchmarkTools: Accurate benchmarking
- ProfileView: Profile visualization
- StaticArrays: Stack-allocated arrays
- LoopVectorization: SIMD optimization

### Parallel Computing (sciml-pro)
- Distributed: Multi-process parallelism
- CUDA: GPU computing
- MPI: Distributed HPC
- ThreadsX: Thread-based parallelism

## Technical Requirements

### Julia Version Support
- Target Julia 1.6 LTS and above
- Acknowledge Julia 1.9+ features (package extensions, improved precompilation)
- Support nightly builds in CI/CD for forward compatibility

### Development Standards
- Type-stable code (verified with @code_warntype)
- Minimal allocations in hot loops
- Follow Julia style guide (snake_case functions, CamelCase types)
- Comprehensive docstrings for public API
- Pass Aqua.jl quality checks

### Testing Standards
- Minimum 80% code coverage
- Test across supported Julia versions
- Cross-platform testing (Linux, macOS, Windows)
- @testset organization

### Documentation Standards
- Documenter.jl for documentation generation
- Getting Started guide
- API reference for public functions
- Examples directory with runnable scripts
- GitHub Pages deployment

## Success Metrics

- All four agents respond accurately to domain-specific queries
- /sciml-setup correctly auto-detects and generates runnable code for ODE/PDE/SDE/optimization
- /julia-optimize identifies real performance issues and provides actionable recommendations
- /julia-scaffold generates publication-ready package structures passing Aqua.jl checks
- /julia-package-ci creates working GitHub Actions workflows
- Skills provide reusable, runnable code patterns

## License

MIT

## Author

Scientific Computing Team
