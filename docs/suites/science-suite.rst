Scientific Computing Suite
==========================

High-performance computing, physics/chemistry simulations, ML/DL, Julia, JAX, and data science workflows. Uses the :term:`Hub Skill` architecture with 14 hubs routing to 117 sub-skills. Optimized for Claude Opus 4.6 with extended context and adaptive reasoning.

**Version:** 3.3.0 | **12 Agents** | **0 Registered Commands** | **14 Hubs → 117 Sub-skills** | **6 Hook Events**

Agents
------

.. agent:: neural-network-master
   :description: Deep learning authority specializing in architecture design, theory, and implementation (Transformers, CNNs, diagnostics).
   :model: opus
   :version: 3.3.0

.. agent:: nonlinear-dynamics-expert
   :description: Expert in bifurcation analysis, chaos, coupled networks, pattern formation, and equation discovery (SINDy/UDE).
   :model: opus
   :version: 3.3.0

.. agent:: research-expert
   :description: Expert in systematic research, evidence synthesis, and publication-quality visualization.
   :model: opus
   :version: 3.3.0

.. agent:: simulation-expert
   :description: Expert in molecular dynamics, statistical mechanics, and numerical methods (HPC/GPU).
   :model: opus
   :version: 3.3.0

.. agent:: statistical-physicist
   :description: Expert in correlation functions, non-equilibrium dynamics, and ensemble theory.
   :model: opus
   :version: 3.3.0

.. agent:: ai-engineer
   :description: Build production-ready LLM applications, advanced RAG systems, and intelligent agents.
   :model: sonnet
   :version: 3.3.0

.. agent:: jax-pro
   :description: Expert in JAX-based scientific computing, functional transformations, and high-performance numerical kernels.
   :model: sonnet
   :version: 3.3.0

.. agent:: julia-ml-hpc
   :description: Expert in Julia ML, Deep Learning, and HPC (Lux.jl, MLJ.jl, CUDA.jl, MPI.jl, GraphNeuralNetworks.jl).
   :model: sonnet
   :version: 3.3.0

.. agent:: julia-pro
   :description: Expert in Julia programming, SciML (DifferentialEquations.jl, ModelingToolkit.jl), and Turing.jl.
   :model: sonnet
   :version: 3.3.0

.. agent:: ml-expert
   :description: Expert in classical ML algorithms, MLOps pipelines, and data engineering.
   :model: sonnet
   :version: 3.3.0

.. agent:: prompt-engineer
   :description: Expert prompt engineer specializing in advanced techniques and LLM performance optimization.
   :model: sonnet
   :version: 3.3.0

.. agent:: python-pro
   :description: Expert Python Systems Engineer specializing in type-driven development, Rust extensions, and performance.
   :model: sonnet
   :version: 3.3.0

Skill-Invoked Commands
----------------------

These commands are triggered by skills, not directly by users:

.. command:: analyze-data
   :description: Analyze data files with statistical tests, visualization, and reproducible reporting.

.. command:: paper-review
   :description: Structured scientific paper review with methodology and reproducibility assessment.

.. command:: run-experiment
   :description: Design and execute computational experiments with hypothesis tracking.

Hub Skills
----------

Skills use a hub architecture: 14 hub skills route to 117 specialized sub-skills.

Hub: nonlinear-dynamics (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bifurcation analysis, chaos, coupled networks, pattern formation, and equation discovery.

- ``bifurcation-analysis`` — BifurcationKit.jl: continuation, codimension-1/2 bifurcations, normal forms
- ``chaos-attractors`` — Lyapunov exponents, attractor reconstruction, fractal dimension, recurrence
- ``network-coupled-dynamics`` — Kuramoto synchronization, master stability, chimera states, epidemic models
- ``pattern-formation`` — Turing instability, dispersion relations, spiral waves, amplitude equations
- ``equation-discovery`` — SINDy, DataDrivenDiffEq.jl, PySINDy, sparse regression
- ``jax-julia-interop`` — Bridge JAX and Julia SciML ecosystems via PythonCall.jl

Hub: jax-computing (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Core JAX, optimization, Bayesian inference, differential equations, and physics.

- ``jax-mastery`` — JIT, vmap, grad, pmap, functional transformations
- ``jax-core-programming`` — Pytrees, custom primitives, XLA operations, device memory
- ``jax-optimization-pro`` — Optax, custom schedules, NLSQ, convergence diagnostics
- ``jax-bayesian-pro`` — JAX-specific NumPyro integration and GPU-accelerated sampling
- ``jax-diffeq-pro`` — Diffrax solvers, neural ODEs, stiff systems, adjoint methods
- ``jax-physics-applications`` — JAX-MD, JAX-CFD, PINNs, differentiable physics

Hub: julia-language (13 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Core Julia, packages, compilation, performance, testing, CI/CD, visualization, HPC, and interop.

- ``julia-mastery`` — Multiple dispatch, type system, metaprogramming, performance patterns
- ``core-julia-patterns`` — Broadcasting, comprehensions, closures, standard library
- ``package-management`` — Pkg.jl, Project.toml, Manifest.toml, environments
- ``package-development-workflow`` — PkgTemplates.jl, documentation, versioning, registration
- ``compiler-patterns`` — PackageCompiler.jl, system images, standalone executables
- ``performance-tuning`` — @btime, memory allocation, SIMD, threading, type stability
- ``julia-testing-patterns`` — Test.jl, Aqua.jl, JET.jl static analysis
- ``ci-cd-patterns`` — GitHub Actions for Julia: test matrix, coverage, releases
- ``visualization-patterns`` — Makie.jl, Plots.jl, interactive and publication-quality figures
- ``web-development-julia`` — Genie.jl, HTTP.jl, REST APIs
- ``julia-hpc-distributed`` — Distributed.jl, MPI.jl, SLURM, multi-node parallelism
- ``interop-patterns`` — PythonCall.jl, RCall.jl, ccall, cross-language data exchange
- ``ecosystem-selection`` — Choosing optimal Julia packages for a domain

Hub: julia-ml-and-dl (9 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Julia neural networks, architectures, training, AD backends, GPU kernels, GNNs, RL, pipelines, deployment.

- ``julia-neural-networks`` — Flux.jl and Lux.jl: model definition, training loops
- ``julia-neural-architectures`` — CNNs, RNNs, Transformers in Flux/Lux
- ``julia-training-diagnostics`` — Loss curves, gradient norms, convergence monitoring
- ``julia-ad-backends`` — Zygote.jl, Enzyme.jl, ForwardDiff.jl, DifferentiationInterface.jl
- ``julia-gpu-kernels`` — CUDA.jl, KernelAbstractions.jl, custom GPU kernels
- ``julia-graph-neural-networks`` — GraphNeuralNetworks.jl: GCN, GAT, message passing
- ``julia-reinforcement-learning`` — ReinforcementLearning.jl: DQN, PPO, environments
- ``julia-ml-pipelines`` — MLJ.jl: data pipelines, cross-validation, tuning
- ``julia-model-deployment`` — ONNX export, HTTP.jl serving, PackageCompiler sysimages

Hub: sciml-and-diffeq (8 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SciML ecosystem, DifferentialEquations.jl, ModelingToolkit, optimization, neural PDEs.

- ``sciml-ecosystem`` — Package selection guide for DifferentialEquations.jl, ModelingToolkit, Optimization.jl
- ``sciml-modern-stack`` — Lux.jl neural networks, SciMLSensitivity adjoint methods, UDEs, DEQ
- ``differential-equations`` — ODE/SDE/PDE solvers, callbacks, ensemble simulations
- ``modeling-toolkit`` — Symbolic differential equations, automatic simplification
- ``optimization-patterns`` — Optimization.jl for parameter estimation and inverse problems
- ``neural-pde`` — NeuralPDE.jl: PINNs with ModelingToolkit
- ``catalyst-reactions`` — Chemical reaction networks, deterministic and stochastic simulations
- ``jump-optimization`` — JuMP.jl: LP, QP, NLP, MIP with HiGHS, Ipopt

Hub: correlation-analysis (4 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mathematical foundations, physical systems, computational methods, and experimental data.

- ``correlation-math-foundations`` — Two-point functions, cumulants, Fourier/Laplace transforms, Wiener-Khinchin
- ``correlation-physical-systems`` — Condensed matter, soft matter, biological, non-equilibrium correlations
- ``correlation-computational-methods`` — FFT-based autocorrelation, multi-tau correlators, JAX-accelerated GPU
- ``correlation-experimental-data`` — DLS, SAXS/SANS, XPCS, FCS, rheology data interpretation

Hub: statistical-physics-hub (8 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Equilibrium/non-equilibrium statistical mechanics, stochastic dynamics, active matter, rare events, and extremes.

- ``statistical-physics`` — Ensemble theory, partition functions, phase transitions; Julia Monte Carlo idioms *(v3.1.5)*
- ``stochastic-dynamics`` — Master equations, Fokker-Planck direct PDE methods, Langevin, Green-Kubo, jump-diffusion SDEs *(Fokker-Planck v3.1.5)*
- ``non-equilibrium-theory`` — Fluctuation theorems, entropy production, linear response, BAR/Jarzynski/MBAR with pymbar worked example *(BAR example v3.1.7)*
- ``active-matter`` — Self-propelled particles, flocking, MIPS, bio-inspired materials
- ``multiscale-modeling`` — Coarse-graining, DPD, nanoscale DEM
- ``advanced-simulations`` — Non-equilibrium thermodynamics, multiscale bridging
- ``rare-events-sampling`` — Large-deviation theory, cloning / importance splitting, SOC / sandpile / avalanche statistics *(new in v3.1.4)*
- ``extreme-value-statistics`` — GEV/GPD/Hill/Pickands/POT, return levels, non-stationary EVT *(new in v3.1.4)*

Hub: deep-learning-hub (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Architecture design, mathematical foundations, training diagnostics, experimentation, large-scale systems.

- ``deep-learning`` — Core DL: feedforward networks, backpropagation, regularization
- ``neural-architecture-patterns`` — CNNs, RNNs, Transformers, diffusion models, normalizing flows
- ``neural-network-mathematics`` — Universal approximation, optimization landscapes, generalization theory
- ``training-diagnostics`` — Loss curves, gradient pathologies, learning rate tuning
- ``deep-learning-experimentation`` — Ablations, HPO, reproducibility, benchmarks
- ``advanced-ml-systems`` — Distributed training, mixed precision, gradient checkpointing

Hub: ml-and-data-science (7 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classical ML, data analysis, wrangling, statistics, visualization, curve fitting, experiment tracking.

- ``machine-learning`` — Scikit-learn, XGBoost, LightGBM, feature engineering
- ``data-analysis`` — Pandas, descriptive statistics, correlation, hypothesis testing
- ``data-wrangling-communication`` — Data cleaning, transformation, stakeholder communication
- ``statistical-analysis-fundamentals`` — Distributions, hypothesis tests, confidence intervals
- ``scientific-visualization`` — Matplotlib, seaborn, plotly, domain-specific plots
- ``nlsq-core-mastery`` — JAX-accelerated non-linear least squares curve fitting
- ``experiment-tracking`` — MLflow, Weights & Biases, DVC

Hub: llm-and-ai (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LLM application development, evaluation, LangChain, RAG, and NLP.

- ``llm-application-dev`` — LLM-powered apps: API integration, streaming, tool use, agents
- ``llm-evaluation`` — Benchmarks, LLM-as-judge, human evaluation, quality metrics
- ``langchain-architecture`` — LangChain/LangGraph: chains, agents, memory, tools
- ``rag-implementation`` — Vector stores, chunking, re-ranking, hybrid retrieval
- ``nlp-fundamentals`` — Tokenization, embeddings, NER, text classification

Hub: ml-deployment (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model serving, optimization, production engineering, pipelines, infrastructure, federated learning.

- ``model-deployment-serving`` — FastAPI, TorchServe, Triton, BentoML, REST/gRPC
- ``model-optimization-deployment`` — Quantization, pruning, ONNX, TensorRT, mobile
- ``ml-engineering-production`` — Type-safe code, testing, data pipelines, monitoring, drift
- ``ml-pipeline-workflow`` — Airflow, Prefect, Metaflow, automated retraining
- ``devops-ml-infrastructure`` — Docker, Kubernetes, GPU provisioning, cloud ML
- ``federated-learning`` — Federated averaging, differential privacy, PySyft

Hub: simulation-and-hpc (10 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MD simulations, trajectory analysis, ML force fields, parallel computing, GPU, numerical methods, and applied math.

- ``md-simulation-setup`` — GROMACS/LAMMPS force fields, equilibration protocols
- ``trajectory-analysis`` — MDAnalysis, RMSD, RDF, free energy, clustering
- ``ml-force-fields`` — NequIP, MACE, DeePMD, active learning workflows
- ``parallel-computing`` — MPI, OpenMP, Dask, Ray, scalability analysis
- ``gpu-acceleration`` — CUDA, ROCm, JAX pmap, GPU-optimized algorithms
- ``numerical-methods-implementation`` — Finite difference/element, spectral methods, iterative solvers
- ``signal-processing`` — FFT, filtering, spectral estimation, wavelet transforms
- ``time-series-analysis`` — ARIMA, state space models, changepoint detection
- ``advanced-optimization`` — Genetic algorithms, simulated annealing, basin hopping
- ``control-theory`` — PID, LQR, MPC, stability analysis

Hub: research-and-domains (18 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Research methodology, paper implementation, scientific writing, Python systems, specialized domains, and autonomous self-improvement research.

- ``research-methodology`` — Study design, literature review, experimental protocols
- ``research-paper-implementation`` — Reproduce academic paper results systematically
- ``research-quality-assessment`` — Rigor, reproducibility, statistics (CONSORT/STROBE/PRISMA)
- ``scientific-communication`` — Papers, reports, presentations (IMRaD)
- ``evidence-synthesis`` — Meta-analysis, systematic reviews (PRISMA/GRADE)
- ``robust-testing`` — Property-based, metamorphic, and tolerance-aware testing for scientific code
- ``python-development`` — Idiomatic Python, software engineering for science
- ``python-packaging-advanced`` — uv workspaces, monorepos, reproducible builds
- ``rust-extensions`` — PyO3/maturin high-performance Python extensions
- ``type-driven-design`` — Protocols, Generics, static analysis with pyright/mypy
- ``modern-concurrency`` — asyncio TaskGroups, threading, multiprocessing
- ``quantum-computing`` — Qiskit, PennyLane, VQE/QAOA
- ``bioinformatics`` — Genomics, proteomics, BioPython
- ``computer-vision`` — Image processing, detection, Vision Transformers
- ``reinforcement-learning`` — Gymnasium, Stable-Baselines3, RLlib
- ``symbolic-math`` — SymPy, CAS, algebraic solvers
- ``self-improving-ai`` — Research overview for autonomous self-improvement (research-framework counterpart to agent-core's self-improving-agents) *(new in v3.1.4)*
- ``dspy-basics`` — Depth-skill companion for DSPy programmatic prompt optimization *(new in v3.1.4)*
- ``rlaif-training`` — Depth-skill companion for Constitutional AI / RLAIF / DPO *(new in v3.1.4)*

Hub: bayesian-inference (10 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NumPyro, Turing.jl, variational inference, MCMC diagnostics, consensus / non-reversible tempering, Bayesian UDEs, Bayesian PINNs, point processes, and Bayesian SINDy equation discovery.

- ``numpyro-core-mastery`` — NumPyro: NUTS/HMC, SVI, hierarchical models, GPU inference
- ``turing-model-design`` — Turing.jl: probabilistic models, Julia-native Bayesian workflows
- ``variational-inference-patterns`` — ELBO, mean-field, normalizing flows, amortized inference
- ``mcmc-diagnostics`` — R-hat, ESS, BFMI, trace plots, ArviZ convergence diagnostics
- ``consensus-mcmc-pigeons`` — Scott-2016 divide-and-conquer Consensus MC and Pigeons.jl non-reversible parallel tempering *(new in v3.1.4)*
- ``bayesian-ude-workflow`` — Turing + DiffEq + Lux staged pipeline for Bayesian Universal Differential Equations *(new in v3.1.4)*
- ``bayesian-ude-jax`` — Python/JAX counterpart to Bayesian UDE via Diffrax + Equinox + NumPyro *(new in v3.1.4)*
- ``bayesian-pinn`` — BNNODE / BayesianPINN (extracted from neural-pde for budget management) *(new in v3.1.4)*
- ``point-processes`` — Hawkes processes, HSGP, Julia PointProcesses.jl, non-parametric Hawkes EM *(new in v3.1.4)*
- ``bayesian-sindy-workflow`` — Horseshoe-prior Bayesian SINDy with 5-stage Lorenz-63 worked example (NumPyro + NUTS + ArviZ PSIS-LOO), prior-sensitivity analysis, and Turing UQ-SINDy sidebar *(new in v3.1.7 — extracted from equation-discovery to resolve 88% budget pressure)*

Hooks
-----

6 hook events with Python script implementations *(new in v3.3.0)*:

- ``SessionStart`` — Detect JAX devices, GPU availability, Julia env
- ``PreToolUse`` — Warn before commands that could corrupt simulations
- ``PostToolUse`` — NaN/Inf check on compute job output (numerical integrity)
- ``ExecutionError`` — Capture JAX compile errors, OOM, numerical divergence
- ``SessionEnd`` — Persist structured progress summary for next session
- ``SubagentStop`` — Collect results from parallel science agents
