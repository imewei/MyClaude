Scientific Computing Suite
==========================

Consolidated suite for high-performance computing, specialized physics/chemistry simulations, and data science workflows. Optimized for Claude Code v2.1.12 with parallel agent execution and MCP server integration.

Agents
------

.. agent:: ai-engineer
   :description: Build production-ready LLM applications, advanced RAG systems, and intelligent agents. Implements vector search, multimodal AI, agent orchestration, and enterprise AI integrations. Use PROACTIVELY for LLM features, chatbots, AI agents, or AI-powered applications.
   :model: sonnet
   :version: 2.0.0

.. agent:: jax-pro
   :description: Expert JAX-based scientific computing agent. Use for Core JAX transformations (JIT/vmap/pmap), Bayesian inference (NumPyro), nonlinear optimization (NLSQ), and computational physics (JAX-MD/CFD). Handles distributed training, custom VJPs, and high-performance numerical kernels.
   :model: sonnet
   :version: 2.0.0

.. agent:: julia-pro
   :description: Expert Julia scientific computing agent. Use for Core Julia, Scientific Machine Learning (SciML), DifferentialEquations.jl, ModelingToolkit.jl, and Turing.jl. Handles high-performance optimization, package development, and cross-language interoperability.
   :model: sonnet
   :version: 2.0.0

.. agent:: ml-expert
   :description: Expert in classical ML algorithms, MLOps pipelines, and data engineering. Masters Scikit-learn, XGBoost, experiment tracking, and model deployment for production workflows. Delegates Deep Learning architecture to neural-network-master.
   :model: sonnet
   :version: 3.0.0

.. agent:: neural-network-master
   :description: Deep learning authority specializing in both theory and implementation. Master of neural architecture design (Transformers, CNNs), mathematical foundations, multi-framework implementation (Flax, Equinox, PyTorch), and training diagnostics. Provides unified guidance from architectural blueprints to theoretical proofs.
   :model: inherit
   :version: 2.0.0

.. agent:: prompt-engineer
   :description: Expert prompt engineer specializing in advanced prompting techniques, LLM optimization, and AI system design. Masters chain-of-thought, constitutional AI, and production prompt strategies. Use when building AI features, improving agent performance, or crafting system prompts.
   :model: sonnet
   :version: 2.0.0

.. agent:: python-pro
   :description: Expert Python Systems Engineer treating Python as a rigorous systems language. Specializes in type-driven development (Protocols, Generics), modern toolchains (uv, ruff), concurrency (TaskGroups, multiprocessing), and performance optimization (PyO3/Rust extensions). Enforces strict typing, zero global state, and library-first architecture.
   :model: sonnet
   :version: 2.0.0

.. agent:: research-expert
   :description: Expert in systematic research, evidence synthesis, statistical rigor, and publication-quality visualization. Guides the research lifecycle from hypothesis design to final figure generation.
   :model: sonnet
   :version: 3.0.0

.. agent:: simulation-expert
   :description: Expert in molecular dynamics, statistical mechanics, and numerical methods. Masters HPC scaling, GPU acceleration, and differentiable physics using JAX and Julia.
   :model: sonnet
   :version: 3.0.1

.. agent:: statistical-physicist
   :description: Statistical physicist expert—the bridge builder asking "How does the chaos of the microscopic world conspire to create the order of the macroscopic world?" Expert in correlation functions, non-equilibrium dynamics, JAX-accelerated GPU simulations, ensemble theory, stochastic calculus (Langevin/Fokker-Planck), phase transitions, fluctuation theorems, and modern AI-physics integration (normalizing flows, ML coarse-graining). Bridges theoretical foundations to high-performance computational analysis. Delegates JAX optimization to jax-pro.
   :model: sonnet
   :version: 1.1.0

Skills
------

.. skill:: active-matter
   :description: Simulation and analysis of active matter systems.
   :version: 1.0.0

.. skill:: advanced-ml-systems
   :description: Building and managing advanced machine learning systems.
   :version: 1.0.0

.. skill:: advanced-simulations
   :description: Master advanced simulation techniques including non-equilibrium thermodynamics, stochastic dynamics, and multiscale modeling.
   :version: 1.0.1

.. skill:: catalyst-reactions
   :description: Modeling and simulation of catalytic reactions.
   :version: 1.0.0

.. skill:: ci-cd-patterns
   :description: CI/CD patterns specifically for scientific workflows.
   :version: 1.0.0

.. skill:: compiler-patterns
   :description: Compiler optimization patterns for scientific code.
   :version: 1.0.0

.. skill:: core-julia-patterns
   :description: Core design patterns in Julia.
   :version: 1.0.0

.. skill:: correlation-computational-methods
   :description: Computational methods for correlation analysis.
   :version: 1.0.0

.. skill:: correlation-experimental-data
   :description: Correlating experimental data with theoretical models.
   :version: 1.0.0

.. skill:: correlation-math-foundations
   :description: Mathematical foundations of correlation functions.
   :version: 1.0.0

.. skill:: correlation-physical-systems
   :description: Applying correlation analysis to physical systems.
   :version: 1.0.0

.. skill:: data-analysis
   :description: General scientific data analysis techniques.
   :version: 1.0.0

.. skill:: data-wrangling-communication
   :description: Wrangling data and communicating results effectively.
   :version: 1.0.0

.. skill:: deep-learning
   :description: Master deep learning architecture design, theory, and implementation.
   :version: 2.0.0

.. skill:: deep-learning-experimentation
   :description: Designing and running deep learning experiments.
   :version: 1.0.0

.. skill:: devops-ml-infrastructure
   :description: DevOps practices for ML infrastructure.
   :version: 1.0.0

.. skill:: differential-equations
   :description: Solving differential equations numerically.
   :version: 1.0.0

.. skill:: ecosystem-selection
   :description: Selecting the right scientific ecosystem (Python vs Julia vs JAX).
   :version: 1.0.0

.. skill:: evidence-synthesis
   :description: Synthesizing evidence from multiple scientific sources.
   :version: 1.0.0

.. skill:: gpu-acceleration
   :description: Accelerating scientific code on GPUs.
   :version: 1.0.0

.. skill:: interop-patterns
   :description: Patterns for interoperability between languages (e.g. Python/Julia).
   :version: 1.0.0

.. skill:: jax-bayesian-pro
   :description: Advanced Bayesian inference with JAX.
   :version: 1.0.0

.. skill:: jax-core-programming
   :description: Core programming concepts and patterns in JAX.
   :version: 1.0.0

.. skill:: jax-diffeq-pro
   :description: Solving differential equations with JAX.
   :version: 1.0.0

.. skill:: jax-mastery
   :description: Master JAX for high-performance scientific computing.
   :version: 1.1.0

.. skill:: jax-optimization-pro
   :description: Advanced optimization techniques in JAX.
   :version: 1.0.0

.. skill:: jax-physics-applications
   :description: Applying JAX to physics problems.
   :version: 1.0.0

.. skill:: julia-mastery
   :description: Master the Julia language for scientific computing.
   :version: 1.0.1

.. skill:: jump-optimization
   :description: Optimization using JuMP in Julia.
   :version: 1.0.0

.. skill:: langchain-architecture
   :description: Architecting applications with LangChain.
   :version: 1.0.0

.. skill:: llm-application-dev
   :description: Build production-ready LLM applications.
   :version: 1.0.0

.. skill:: llm-evaluation
   :description: Evaluating LLM performance and quality.
   :version: 1.0.0

.. skill:: machine-learning
   :description: Comprehensive Classical Machine Learning suite.
   :version: 2.0.0

.. skill:: machine-learning-essentials
   :description: Essential concepts and algorithms in machine learning.
   :version: 1.0.0

.. skill:: mcmc-diagnostics
   :description: Diagnostics for MCMC simulations.
   :version: 1.0.0

.. skill:: md-simulation-setup
   :description: Setting up molecular dynamics simulations.
   :version: 1.0.0

.. skill:: ml-engineering-production
   :description: Engineering ML systems for production.
   :version: 1.0.0

.. skill:: ml-force-fields
   :description: Machine learning force fields for molecular simulation.
   :version: 1.0.0

.. skill:: ml-pipeline-workflow
   :description: Building ML pipeline workflows.
   :version: 1.0.0

.. skill:: model-deployment-serving
   :description: Deploying and serving ML models.
   :version: 1.0.0

.. skill:: model-optimization-deployment
   :description: Optimizing models for deployment.
   :version: 1.0.0

.. skill:: modeling-toolkit
   :description: Using ModelingToolkit.jl for acausal modeling.
   :version: 1.0.0

.. skill:: modern-concurrency
   :description: Modern concurrency patterns in scientific computing.
   :version: 1.0.0

.. skill:: multiscale-modeling
   :description: Modeling systems across multiple scales.
   :version: 1.0.0

.. skill:: neural-architecture-patterns
   :description: Design patterns for neural network architectures.
   :version: 1.0.0

.. skill:: neural-network-mathematics
   :description: Mathematical foundations of neural networks.
   :version: 1.0.0

.. skill:: neural-pde
   :description: Solving PDEs with neural networks (Physics-Informed Neural Networks).
   :version: 1.0.0

.. skill:: nlsq-core-mastery
   :description: Mastery of Non-Linear Least Squares optimization.
   :version: 1.0.0

.. skill:: non-equilibrium-theory
   :description: Theory of non-equilibrium statistical mechanics.
   :version: 1.0.0

.. skill:: numerical-methods-implementation
   :description: Implementing numerical methods efficiently.
   :version: 1.0.0

.. skill:: numpyro-core-mastery
   :description: Mastery of NumPyro for probabilistic programming.
   :version: 1.0.0

.. skill:: optimization-patterns
   :description: General optimization patterns and strategies.
   :version: 1.0.0

.. skill:: package-development-workflow
   :description: Workflows for developing scientific packages.
   :version: 1.0.0

.. skill:: package-management
   :description: Managing dependencies and environments.
   :version: 1.0.0

.. skill:: parallel-computing
   :description: Implement high-performance parallel computing.
   :version: 1.0.1

.. skill:: parallel-computing-strategy
   :description: Strategies for parallelizing scientific code.
   :version: 1.0.0

.. skill:: performance-tuning
   :description: Tuning code for maximum performance.
   :version: 1.0.0

.. skill:: prompt-engineering-patterns
   :description: Patterns for effective prompt engineering in science.
   :version: 1.0.0

.. skill:: python-development
   :description: Master modern Python systems engineering for scientific computing.
   :version: 2.0.0

.. skill:: python-packaging-advanced
   :description: Advanced Python packaging for scientific libraries.
   :version: 1.0.0

.. skill:: rag-implementation
   :description: Implementing Retrieval-Augmented Generation for science.
   :version: 1.0.0

.. skill:: research-methodology
   :description: Systematic framework for scientific research.
   :version: 1.1.0

.. skill:: research-paper-implementation
   :description: Implementing algorithms from research papers.
   :version: 1.0.0

.. skill:: research-quality-assessment
   :description: Assessing the quality and reproducibility of research.
   :version: 1.0.0

.. skill:: robust-testing
   :description: Robust testing strategies for scientific code.
   :version: 1.0.0

.. skill:: rust-extensions
   :description: Writing Rust extensions for Python (PyO3).
   :version: 1.0.0

.. skill:: scientific-communication
   :description: Communicating scientific results effectively.
   :version: 1.0.0

.. skill:: scientific-visualization
   :description: Create publication-quality scientific visualizations.
   :version: 1.1.0

.. skill:: sciml-ecosystem
   :description: Navigating the SciML ecosystem.
   :version: 1.0.0

.. skill:: statistical-analysis-fundamentals
   :description: Fundamentals of statistical analysis.
   :version: 1.0.0

.. skill:: statistical-physics
   :description: Comprehensive statistical physics suite.
   :version: 1.1.0

.. skill:: stochastic-dynamics
   :description: Simulation of stochastic dynamical systems.
   :version: 1.0.0

.. skill:: testing-patterns
   :description: Testing patterns for scientific software.
   :version: 1.0.0

.. skill:: training-diagnostics
   :description: Diagnosing issues in ML model training.
   :version: 1.0.0

.. skill:: trajectory-analysis
   :description: Analyzing simulation trajectories.
   :version: 1.0.0

.. skill:: turing-model-design
   :description: Designing probabilistic models with Turing.jl.
   :version: 1.0.0

.. skill:: type-driven-design
   :description: Using type systems for robust scientific code design.
   :version: 1.0.0

.. skill:: variational-inference-patterns
   :description: Patterns for variational inference.
   :version: 1.0.0

.. skill:: visualization-patterns
   :description: Patterns for effective data visualization.
   :version: 1.0.0

.. skill:: web-development-julia
   :description: Web development using Julia (Genie, etc.).
   :version: 1.0.0

