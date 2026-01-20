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

.. skill:: advanced-simulations
   :description: Master advanced simulation techniques including non-equilibrium thermodynamics, stochastic dynamics, and multiscale modeling. Bridge scales from atomistic to mesoscale.
   :version: 1.0.1

.. skill:: deep-learning
   :description: Master deep learning architecture design, theory, and implementation. Covers neural network mathematics, training diagnostics, PyTorch/JAX frameworks, and advanced patterns.
   :version: 2.0.0

.. skill:: jax-mastery
   :description: Master JAX for high-performance scientific computing. Covers functional transformations (JIT, vmap, pmap, grad), neural networks (Flax NNX), and specialized optimization (NLSQ, NumPyro).
   :version: 1.1.0

.. skill:: julia-mastery
   :description: Master the Julia language for scientific computing. Covers multiple dispatch, type stability, metaprogramming, and the SciML ecosystem.
   :version: 1.0.1

.. skill:: llm-application-dev
   :description: Build production-ready LLM applications, RAG systems, and AI agents. Covers prompt engineering, LangChain/LangGraph architecture, and evaluation.
   :version: 1.0.0

.. skill:: machine-learning
   :description: Comprehensive Classical Machine Learning suite. Covers scikit-learn, XGBoost, LightGBM, and MLOps pipelines. Focuses on tabular data, feature engineering, and production deployment.
   :version: 2.0.0

.. skill:: parallel-computing
   :description: Implement high-performance parallel computing across CPUs and GPUs using Python (CUDA/CuPy) and Julia (CUDA.jl/Distributed.jl). Master multi-threading, distributed systems, and kernel optimization.
   :version: 1.0.1

.. skill:: python-development
   :description: Master modern Python systems engineering for scientific computing. Covers type-driven design, Rust extensions (PyO3), structured concurrency (TaskGroups), robust testing (Hypothesis), and uv-based packaging.
   :version: 2.0.0

.. skill:: research-methodology
   :description: Systematic framework for scientific research, covering experimental design, statistical rigor, quality assessment, and publication readiness.
   :version: 1.1.0

.. skill:: scientific-visualization
   :description: Create publication-quality scientific visualizations across physics, biology, chemistry, and climate science. Supports uncertainty quantification, multi-dimensional data, and domain-specific plots in both Python and Julia.
   :version: 1.1.0

.. skill:: statistical-physics
   :description: Comprehensive statistical physics suite covering equilibrium and non-equilibrium statistical mechanics, active matter, stochastic dynamics, and correlation analysis. Master the bridge between microscopic laws and macroscopic behavior.
   :version: 1.1.0

