Julia Development
=================

.. module:: julia-development
   :synopsis: Comprehensive Julia development plugin with specialized agents for high-performance computing, package development, scientific machine learning (SciML), and Bayesian inference. Expert guidance for building robust Julia applications with optimization, monitoring, and deep learning capabilities.

Description
-----------

Comprehensive Julia development plugin with specialized agents for high-performance computing, package development, scientific machine learning (SciML), and Bayesian inference. Expert guidance for building robust Julia applications with optimization, monitoring, and deep learning capabilities.

**Metadata:**

- **Version:** 1.0.4
- **Category:** scientific-computing
- **License:** MIT
- **Author:** Scientific Computing Team <https://github.com/Scientific-Computing-Team>
- **Keywords:** julia, scientific-computing, sciml, bayesian, hpc, high-performance-computing, differential-equations, ode, pde, sde, optimization, jump, turing, probabilistic-programming, mcmc, variational-inference, package-development, testing, ci-cd, multiple-dispatch, type-system, metaprogramming, performance-optimization, parallel-computing, gpu-computing, neural-pde, physics-informed-neural-networks, modeling-toolkit, catalyst, machine-learning, deep-learning, flux, visualization, plots, makie, interoperability, python-integration

Agents
------

.. agent:: julia-pro

   General Julia programming expert for high-performance computing, scientific simulations, data analysis, and machine learning. Master of multiple dispatch, type system, metaprogramming, JuMP optimization, and Julia ecosystem.

   **Status:** active

.. agent:: julia-developer

   Package development specialist for creating robust Julia packages. Expert in testing patterns, CI/CD automation, PackageCompiler.jl, web development (Genie.jl), and integrating optimization, monitoring, and deep learning components.

   **Status:** active

.. agent:: sciml-pro

   SciML ecosystem expert for scientific machine learning and differential equations. Master of DifferentialEquations.jl, ModelingToolkit.jl, Optimization.jl, NeuralPDE.jl, Catalyst.jl, performance tuning, and parallel computing.

   **Status:** active

.. agent:: turing-pro

   Bayesian inference and probabilistic programming expert. Master of Turing.jl, MCMC methods, variational inference (ADVI), model comparison, convergence diagnostics, and integration with SciML for Bayesian ODEs.

   **Status:** active

Commands
--------

.. command:: sciml-setup

   Interactive SciML project scaffolding with auto-detection of problem types (ODE, PDE, SDE, optimization). Generates template code with callbacks, ensemble simulations, and sensitivity analysis.

   **Status:** active
   **Priority:** 1

   Usage Example:

   .. code-block:: bash

      sciml-setup

.. command:: julia-optimize

   Profile Julia code and provide optimization recommendations. Analyzes type stability, memory allocations, identifies bottlenecks, and suggests parallelization strategies.

   **Status:** active
   **Priority:** 2

   Usage Example:

   .. code-block:: bash

      julia-optimize

.. command:: julia-scaffold

   Bootstrap new Julia package with proper structure following PkgTemplates.jl conventions. Creates Project.toml, testing infrastructure, documentation framework, and git repository.

   **Status:** active
   **Priority:** 3

   Usage Example:

   .. code-block:: bash

      julia-scaffold

.. command:: julia-package-ci

   Generate GitHub Actions CI/CD workflows for Julia packages. Configures testing matrices across Julia versions and platforms, coverage reporting, and documentation deployment.

   **Status:** active
   **Priority:** 4

   Usage Example:

   .. code-block:: bash

      julia-package-ci

Skills
------

.. skill:: core-julia-patterns

   Multiple dispatch, type system, parametric types, metaprogramming, type stability, and performance optimization fundamentals

   **Status:** active

.. skill:: jump-optimization

   Mathematical programming with JuMP.jl modeling patterns, constraints, objectives, solver selection (separate from Optimization.jl)

   **Status:** active

.. skill:: visualization-patterns

   Plotting with Plots.jl, Makie.jl, StatsPlots.jl for data visualization and scientific graphics

   **Status:** active

.. skill:: interop-patterns

   Python interop via PythonCall.jl, R via RCall.jl, C++ via CxxWrap.jl for cross-language integration

   **Status:** active

.. skill:: package-management

   Project.toml structure, Pkg.jl workflows, dependency management, semantic versioning

   **Status:** active

.. skill:: package-development-workflow

   Package structure, module organization, exports, PkgTemplates.jl conventions, documentation

   **Status:** active

.. skill:: testing-patterns

   Test.jl best practices, test organization, BenchmarkTools.jl, Aqua.jl quality checks, JET.jl static analysis

   **Status:** active

.. skill:: compiler-patterns

   PackageCompiler.jl for static compilation, creating executables, system images, deployment optimization

   **Status:** active

.. skill:: web-development-julia

   Genie.jl MVC framework, HTTP.jl server development, API patterns, JSON3.jl, Oxygen.jl lightweight APIs

   **Status:** active

.. skill:: ci-cd-patterns

   GitHub Actions for Julia, test matrices, CompatHelper.jl, TagBot.jl, documentation deployment

   **Status:** active

.. skill:: sciml-ecosystem

   SciML package integration: DifferentialEquations.jl, ModelingToolkit.jl, Catalyst.jl, solver selection

   **Status:** active

.. skill:: differential-equations

   ODE, PDE, SDE, DAE solving patterns with callbacks, ensemble simulations, sensitivity analysis

   **Status:** active

.. skill:: modeling-toolkit

   Symbolic problem definition with ModelingToolkit.jl, equation simplification, code generation

   **Status:** active

.. skill:: optimization-patterns

   Optimization.jl usage for SciML optimization (distinct from JuMP.jl mathematical programming)

   **Status:** active

.. skill:: neural-pde

   Physics-informed neural networks (PINNs) with NeuralPDE.jl, boundary conditions, training strategies

   **Status:** active

.. skill:: catalyst-reactions

   Reaction network modeling with Catalyst.jl, rate laws, species definitions, stochastic vs deterministic

   **Status:** active

.. skill:: performance-tuning

   Profiling with @code_warntype, @profview, BenchmarkTools.jl, allocation reduction, type stability analysis

   **Status:** active

.. skill:: parallel-computing

   Multi-threading, Distributed.jl, GPU computing with CUDA.jl, ensemble simulations, load balancing

   **Status:** active

.. skill:: turing-model-design

   Turing.jl model specification, prior selection, likelihood definition, hierarchical models, identifiability

   **Status:** active

.. skill:: mcmc-diagnostics

   MCMC convergence checking (trace plots, R-hat), effective sample size, divergence checking, mixing analysis

   **Status:** active

.. skill:: variational-inference-patterns

   ADVI with Turing.jl, Bijectors.jl transformations, ELBO monitoring, VI vs MCMC comparison

   **Status:** active

Usage Examples
--------------

Additional Examples
~~~~~~~~~~~~~~~~~~~

To build documentation locally:

.. code-block:: bash

   cd docs/
   make html

Integration
-----------

**Integrates With:**

This plugin integrates with the following plugins:

- :doc:`/plugins/deep-learning` (agent, command, documentation, integration, related, skill, workflow)
- :doc:`/plugins/hpc-computing` (agent, command, documentation, integration, related, skill, workflow)

**Common Workflows:**

This plugin is part of the following workflow patterns:

- **Machine-Learning Integration Pattern**: :doc:`/plugins/debugging-toolkit`, :doc:`/plugins/machine-learning`

- **Testing Integration Pattern**: :doc:`/plugins/full-stack-orchestration`, :doc:`/plugins/javascript-typescript`, :doc:`/plugins/python-development`

- **Scientific Computing HPC Workflow**: :doc:`/plugins/hpc-computing`

See Also
--------

- :doc:`/categories/scientific-computing`
- :doc:`/integration-map`

References
----------

*External resources and links will be added as available.*
