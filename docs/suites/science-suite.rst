Scientific Computing Suite
==========================

High-performance computing, physics/chemistry simulations, ML/DL, Julia, JAX, and data science workflows. Uses the :term:`Hub Skill` architecture with 23 hubs routing to 114 sub-skills.

**Version:** 4.0.2 | **12 Agents** | **4 Registered Commands** | **23 Hubs → 114 Sub-skills** | **5 Hook Events**

.. note::

   In v3.4.0, ``research-expert`` plus 5 methodology skills (``research-methodology``, ``research-quality-assessment``, ``research-paper-implementation``, ``scientific-communication``, ``evidence-synthesis``) moved to the new :doc:`research-suite <research-suite>`. This suite now focuses purely on *computational* work. Research-*methodology* delegations from science-suite agents route to the research-suite instead.

Agents
------

.. agent:: neural-network-master
   :description: Deep learning authority specializing in architecture design, theory, and implementation (Transformers, CNNs, diagnostics).
   :model: opus
   :version: 4.0.2

.. agent:: nonlinear-dynamics-expert
   :description: Expert in bifurcation analysis, chaos, coupled networks, pattern formation, and equation discovery (SINDy/UDE).
   :model: opus
   :version: 4.0.2

.. agent:: simulation-expert
   :description: Expert in molecular dynamics, statistical mechanics, and numerical methods (HPC/GPU).
   :model: sonnet
   :version: 4.0.2

.. agent:: statistical-physicist
   :description: Expert in correlation functions, non-equilibrium dynamics, and ensemble theory.
   :model: opus
   :version: 4.0.2

.. agent:: jax-pro
   :description: JAX expert — jit/vmap/pmap, sharding, VJP/JVP, XLA/HLO, Optax, Diffrax, Pallas, NumPyro. Delegates MD, bifurcation, general Bayes, and productionization to peers.
   :model: opus
   :version: 4.0.2

.. agent:: julia-pro
   :description: Julia/SciML expert — dispatch, type stability, DiffEq.jl, ModelingToolkit, SciMLSensitivity, UDE, SINDy, Turing, Optimization.jl. Delegates ML/HPC and productionization to peers.
   :model: opus
   :version: 4.0.2

.. agent:: julia-ml-hpc
   :description: Julia ML/HPC expert for Lux.jl, MLJ.jl, CUDA.jl, MPI.jl, and GNNLux. Delegates SciML/ODE to julia-pro.
   :model: sonnet
   :version: 4.0.2

.. agent:: pinn-engineer
   :description: Physics-informed AI for PINNs, NeuralPDE.jl, DeepXDE, BPINN/BNNODE, physics-constrained losses, and inverse PDEs.
   :model: opus
   :version: 4.0.2

.. agent:: python-pro
   :description: Python systems engineer for production Python, type-driven design, PyO3/Rust extensions, async, and uv/ruff toolchain.
   :model: sonnet
   :version: 4.0.2

.. agent:: sci-workflow-engineer
   :description: Scientific LLM workflow engineer for JAX/Julia codegen prompts, experiment templates, scientific RAG, and AI-assisted pipelines.
   :model: sonnet
   :version: 4.0.2

.. agent:: ml-expert
   :description: Classical ML/MLOps with scikit-learn, XGBoost/LightGBM, Optuna, SHAP, and MLflow/W&B. Delegates DL to neural-network-master.
   :model: haiku
   :version: 4.0.2

.. agent:: continuum-mechanics-engineer
   :description: Expert in FEM/FEA, constitutive modeling, DMA/rheology, transient networks (CAN/vitrimers), and nanocomposites.
   :model: opus
   :version: 4.0.2

Registered Commands
-------------------

Four slash commands are registered in ``plugin.json`` (``md-sim`` and ``benchmark`` since v3.5.0; ``analyze-data`` and ``run-experiment`` since v4.0.0):

.. command:: md-sim
   :description: Molecular dynamics simulation setup, running, and trajectory analysis (GROMACS/OpenMM/JAX-MD).

.. command:: benchmark
   :description: Scientific code benchmarking across backends and hardware targets, with comparison reports.

.. command:: analyze-data
   :description: Analyze data files with statistical tests, visualization, and reproducible reporting.

.. command:: run-experiment
   :description: Design and execute computational experiments with hypothesis tracking.

The legacy ``paper-review`` command moved to ``research-suite`` and was then removed in favor of the ``scientific-review`` skill (produces a ``.docx`` deliverable with journal-specific adaptation — strictly better output).

Hub Skills
----------

Skills use a hub architecture: 23 hub skills registered in ``plugin.json`` route to 114 specialized sub-skills. Hubs
overlap by design — a sub-skill reachable from several hubs is listed under each; counts below are per-hub routing
fan-out, not disjoint partitions. ``science-hub`` is the top-level router.

Hub: science-hub (1 sub-skill)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Top-level router for all scientific computing topics.

Routes to hubs: ``nonlinear-dynamics``, ``jax-computing``, ``julia-language``, ``julia-ml-and-dl``, ``sciml-and-diffeq``, ``statistical-physics-hub``, ``deep-learning-hub``, ``ml-and-data-science``, ``llm-and-ai``, ``ml-deployment``, ``simulation-and-hpc``, ``continuum-mechanics-and-rheology``, ``research-and-domains``, ``bayesian-inference``

- ``correlation-analysis`` — Second-tier hub for correlation functions: math foundations, physical systems, computational methods, experimental data

Hub: jax-computing (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for JAX scientific computing.

- ``jax-mastery`` — JIT, vmap, grad, pmap, functional transformations
- ``jax-core-programming`` — Pytrees, custom primitives, XLA operations, device memory
- ``jax-optimization-pro`` — Optax, custom schedules, NLSQ, convergence diagnostics
- ``jax-bayesian-pro`` — JAX-specific NumPyro integration and GPU-accelerated sampling
- ``jax-diffeq-pro`` — Diffrax solvers, neural ODEs, stiff systems, adjoint methods
- ``jax-physics-applications`` — JAX-MD, JAX-CFD, PINNs, differentiable physics

Hub: julia-language (12 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for Julia language and ecosystem.

Routes to hubs: ``julia-mastery``

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

Hub: julia-mastery (21 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Master the Julia language for scientific computing.

Routes to hubs: ``nonlinear-dynamics``, ``parallel-computing``

- ``core-julia-patterns`` — Broadcasting, comprehensions, closures, standard library
- ``sciml-ecosystem`` — Package selection guide for DifferentialEquations.jl, ModelingToolkit, Optimization.jl
- ``differential-equations`` — ODE/SDE/PDE solvers, callbacks, ensemble simulations
- ``modeling-toolkit`` — Symbolic differential equations, automatic simplification
- ``neural-pde`` — NeuralPDE.jl: PINNs with ModelingToolkit
- ``turing-model-design`` — Turing.jl: probabilistic models, Julia-native Bayesian workflows
- ``performance-tuning`` — @btime, memory allocation, SIMD, threading, type stability
- ``package-development-workflow`` — PkgTemplates.jl, documentation, versioning, registration
- ``variational-inference-patterns`` — ELBO, mean-field, normalizing flows, amortized inference
- ``optimization-patterns`` — Optimization.jl for parameter estimation and inverse problems
- ``jump-optimization`` — JuMP.jl: LP, QP, NLP, MIP with HiGHS, Ipopt
- ``mcmc-diagnostics`` — R-hat, ESS, BFMI, trace plots, ArviZ convergence diagnostics
- ``julia-testing-patterns`` — Test.jl, Aqua.jl, JET.jl static analysis
- ``package-management`` — Pkg.jl, Project.toml, Manifest.toml, environments
- ``catalyst-reactions`` — Chemical reaction networks, deterministic and stochastic simulations
- ``visualization-patterns`` — Makie.jl, Plots.jl, interactive and publication-quality figures
- ``web-development-julia`` — Genie.jl, HTTP.jl, REST APIs
- ``sciml-modern-stack`` — Lux.jl neural networks, SciMLSensitivity adjoint methods, UDEs, DEQ
- ``interop-patterns`` — PythonCall.jl, RCall.jl, ccall, cross-language data exchange
- ``ci-cd-patterns`` — GitHub Actions for Julia: test matrix, coverage, releases
- ``compiler-patterns`` — PackageCompiler.jl, system images, standalone executables

Hub: julia-ml-and-dl (9 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for Julia ML and deep learning.

- ``julia-neural-networks`` — Flux.jl and Lux.jl: model definition, training loops
- ``julia-neural-architectures`` — CNNs, RNNs, Transformers in Flux/Lux
- ``julia-training-diagnostics`` — Loss curves, gradient norms, convergence monitoring
- ``julia-ad-backends`` — Zygote.jl, Enzyme.jl, ForwardDiff.jl, DifferentiationInterface.jl
- ``julia-gpu-kernels`` — CUDA.jl, KernelAbstractions.jl, custom GPU kernels
- ``julia-graph-neural-networks`` — GraphNeuralNetworks.jl: GCN, GAT, message passing
- ``julia-reinforcement-learning`` — ReinforcementLearning.jl: DQN, PPO, environments
- ``julia-ml-pipelines`` — MLJ.jl: data pipelines, cross-validation, tuning
- ``julia-model-deployment`` — ONNX export, HTTP.jl serving, PackageCompiler sysimages

Hub: sciml-and-diffeq (12 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for Scientific Machine Learning and differential equations in Julia.

- ``sciml-ecosystem`` — Package selection guide for DifferentialEquations.jl, ModelingToolkit, Optimization.jl
- ``sciml-modern-stack`` — Lux.jl neural networks, SciMLSensitivity adjoint methods, UDEs, DEQ
- ``differential-equations`` — ODE/SDE/PDE solvers, callbacks, ensemble simulations
- ``modeling-toolkit`` — Symbolic differential equations, automatic simplification
- ``optimization-patterns`` — Optimization.jl for parameter estimation and inverse problems
- ``neural-pde`` — NeuralPDE.jl: PINNs with ModelingToolkit
- ``bayesian-pinn`` — BNNODE / BayesianPINN (extracted from neural-pde for budget management) *(new in v3.1.4)*
- ``catalyst-reactions`` — Chemical reaction networks, deterministic and stochastic simulations
- ``jump-optimization`` — JuMP.jl: LP, QP, NLP, MIP with HiGHS, Ipopt
- ``equation-discovery`` — SINDy, DataDrivenDiffEq.jl, PySINDy, sparse regression
- ``bifurcation-analysis`` — Numerical continuation, codim-1/2 bifurcations, normal forms; AUTO-07p engine (BifurcationKit.jl blocked on Julia 1.12)
- ``bayesian-ude-workflow`` — Turing + DiffEq + Lux staged pipeline for Bayesian Universal Differential Equations *(new in v3.1.4)*

Hub: nonlinear-dynamics (7 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for nonlinear dynamics analysis.

- ``bifurcation-analysis`` — Numerical continuation, codim-1/2 bifurcations, normal forms; AUTO-07p engine (BifurcationKit.jl blocked on Julia 1.12)
- ``chaos-attractors`` — Lyapunov exponents, attractor reconstruction, fractal dimension, recurrence
- ``network-coupled-dynamics`` — Kuramoto synchronization, master stability, chimera states, epidemic models
- ``pattern-formation`` — Turing instability, dispersion relations, spiral waves, amplitude equations
- ``equation-discovery`` — SINDy, DataDrivenDiffEq.jl, PySINDy, sparse regression
- ``jax-julia-interop`` — Bridge JAX and Julia SciML ecosystems via PythonCall.jl
- ``bayesian-ude-workflow`` — Turing + DiffEq + Lux staged pipeline for Bayesian Universal Differential Equations *(new in v3.1.4)*

Hub: statistical-physics-hub (8 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for statistical physics and soft matter.

Routes to hubs: ``statistical-physics``, ``advanced-simulations``

- ``stochastic-dynamics`` — Master equations, Fokker-Planck direct PDE methods, Langevin, Green-Kubo, jump-diffusion SDEs *(Fokker-Planck v3.1.5)*
- ``non-equilibrium-theory`` — Fluctuation theorems, entropy production, linear response, BAR/Jarzynski/MBAR with pymbar worked example *(BAR example v3.1.7)*
- ``active-matter`` — Self-propelled particles, flocking, MIPS, bio-inspired materials
- ``multiscale-modeling`` — Coarse-graining, DPD, nanoscale DEM
- ``rare-events-sampling`` — Large-deviation theory, cloning / importance splitting, SOC / sandpile / avalanche statistics *(new in v3.1.4)*
- ``extreme-value-statistics`` — GEV/GPD/Hill/Pickands/POT, return levels, non-stationary EVT *(new in v3.1.4)*
- ``glass-and-collective-dynamics`` — Glassy relaxation, jamming, aging, cooperative dynamics, percolation in disordered media
- ``physical-learning-systems`` — Coupled / contrastive Hebbian learning in physical networks, plasticity and memory in disordered materials

Hub: statistical-physics (8 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Equilibrium and non-equilibrium statistical mechanics, active matter, stochastic dynamics, and correlation analysis.

- ``stochastic-dynamics`` — Master equations, Fokker-Planck direct PDE methods, Langevin, Green-Kubo, jump-diffusion SDEs *(Fokker-Planck v3.1.5)*
- ``non-equilibrium-theory`` — Fluctuation theorems, entropy production, linear response, BAR/Jarzynski/MBAR with pymbar worked example *(BAR example v3.1.7)*
- ``active-matter`` — Self-propelled particles, flocking, MIPS, bio-inspired materials
- ``correlation-analysis`` — Second-tier hub for correlation functions: math foundations, physical systems, computational methods, experimental data
- ``correlation-math-foundations`` — Two-point functions, cumulants, Fourier/Laplace transforms, Wiener-Khinchin
- ``correlation-physical-systems`` — Condensed matter, soft matter, biological, non-equilibrium correlations
- ``correlation-computational-methods`` — FFT-based autocorrelation, multi-tau correlators, JAX-accelerated GPU
- ``correlation-experimental-data`` — DLS, SAXS/SANS, XPCS, FCS, rheology data interpretation

Hub: advanced-simulations (4 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MD setup, ML force fields, multiscale modeling, trajectory analysis, rare-event sampling, and non-equilibrium transport.

- ``md-simulation-setup`` — GROMACS/LAMMPS force fields, equilibration protocols
- ``ml-force-fields`` — NequIP, MACE, DeePMD, active learning workflows
- ``multiscale-modeling`` — Coarse-graining, DPD, nanoscale DEM
- ``trajectory-analysis`` — MDAnalysis, RMSD, RDF, free energy, clustering

Hub: simulation-and-hpc (8 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for simulation, HPC, and computational methods.

Routes to hubs: ``parallel-computing``, ``time-series-analysis``

- ``md-simulation-setup`` — GROMACS/LAMMPS force fields, equilibration protocols
- ``trajectory-analysis`` — MDAnalysis, RMSD, RDF, free energy, clustering
- ``ml-force-fields`` — NequIP, MACE, DeePMD, active learning workflows
- ``gpu-acceleration`` — CUDA, ROCm, JAX pmap, GPU-optimized algorithms
- ``numerical-methods-implementation`` — Finite difference/element, spectral methods, iterative solvers
- ``signal-processing`` — FFT, filtering, spectral estimation, wavelet transforms
- ``advanced-optimization`` — Genetic algorithms, simulated annealing, basin hopping
- ``control-theory`` — PID, LQR, MPC, stability analysis

Hub: parallel-computing (3 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implement high-performance parallel computing across CPUs and GPUs using Python (CUDA/CuPy) and Julia (CUDA.jl/Distributed.jl).

- ``ecosystem-selection`` — Choosing optimal Julia packages for a domain
- ``gpu-acceleration`` — CUDA, ROCm, JAX pmap, GPU-optimized algorithms
- ``numerical-methods-implementation`` — Finite difference/element, spectral methods, iterative solvers

Hub: continuum-mechanics-and-rheology (7 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for continuum mechanics, FEM/FEA, constitutive modeling, rheology/DMA, transient networks, and nanocomposites.

- ``fem-fea`` — Weak-form formulation, mesh strategy, element selection, convergence; FEniCS, scikit-fem, Gridap.jl, Ferrite.jl
- ``graph-theory`` — Spectral graph theory, graph algorithms, network topology metrics
- ``constitutive-equations`` — Linear elasticity, hyperelasticity (Neo-Hookean, Mooney-Rivlin, Ogden), viscoelasticity (Maxwell, Kelvin-Voigt, Prony series)
- ``dma-rheology`` — Storage/loss modulus, tan delta, oscillatory shear, shear vs extensional flow curves
- ``harmonic-response-superposition`` — Complex modulus under sinusoidal loading, WLF equation, time-temperature superposition master curves
- ``transient-networks-and-can`` — Physical gels and covalent adaptable networks (vitrimers): sticky Rouse, Green-Tobolsky, bond-exchange kinetics
- ``nanocomposites-and-adaptive-materials`` — Effective-medium theory (Halpin-Tsai, Mori-Tanaka), percolation-aware property prediction, self-healing composites

Hub: bayesian-inference (11 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for Bayesian inference and probabilistic programming.

- ``numpyro-core-mastery`` — NumPyro: NUTS/HMC, SVI, hierarchical models, GPU inference
- ``turing-model-design`` — Turing.jl: probabilistic models, Julia-native Bayesian workflows
- ``consensus-mcmc-pigeons`` — Scott-2016 divide-and-conquer Consensus MC and Pigeons.jl non-reversible parallel tempering *(new in v3.1.4)*
- ``bayesian-ude-workflow`` — Turing + DiffEq + Lux staged pipeline for Bayesian Universal Differential Equations *(new in v3.1.4)*
- ``bayesian-sindy-workflow`` — Horseshoe-prior Bayesian SINDy with 5-stage Lorenz-63 worked example (NumPyro + NUTS + ArviZ PSIS-LOO), prior-sensitivity analysis, and Turing UQ-SINDy sidebar *(new in v3.1.7 — extracted from equation-discovery to resolve 88% budget pressure)*
- ``bayesian-ude-jax`` — Python/JAX counterpart to Bayesian UDE via Diffrax + Equinox + NumPyro *(new in v3.1.4)*
- ``bayesian-pinn`` — BNNODE / BayesianPINN (extracted from neural-pde for budget management) *(new in v3.1.4)*
- ``point-processes`` — Hawkes processes, HSGP, Julia PointProcesses.jl, non-parametric Hawkes EM *(new in v3.1.4)*
- ``variational-inference-patterns`` — ELBO, mean-field, normalizing flows, amortized inference
- ``mcmc-diagnostics`` — R-hat, ESS, BFMI, trace plots, ArviZ convergence diagnostics
- ``neural-pde`` — NeuralPDE.jl: PINNs with ModelingToolkit

Hub: deep-learning-hub (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for deep learning.

Routes to hubs: ``deep-learning``

- ``neural-architecture-patterns`` — CNNs, RNNs, Transformers, diffusion models, normalizing flows
- ``neural-network-mathematics`` — Universal approximation, optimization landscapes, generalization theory
- ``training-diagnostics`` — Loss curves, gradient pathologies, learning rate tuning
- ``deep-learning-experimentation`` — Ablations, HPO, reproducibility, benchmarks
- ``advanced-ml-systems`` — Distributed training, mixed precision, gradient checkpointing

Hub: deep-learning (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Core deep-learning implementation: standard networks, training loops, regularization, loss selection.

- ``neural-architecture-patterns`` — CNNs, RNNs, Transformers, diffusion models, normalizing flows
- ``neural-network-mathematics`` — Universal approximation, optimization landscapes, generalization theory
- ``training-diagnostics`` — Loss curves, gradient pathologies, learning rate tuning
- ``model-optimization-deployment`` — Quantization, pruning, ONNX, TensorRT, mobile
- ``deep-learning-experimentation`` — Ablations, HPO, reproducibility, benchmarks

Hub: machine-learning (7 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comprehensive Classical Machine Learning suite.

- ``advanced-ml-systems`` — Distributed training, mixed precision, gradient checkpointing
- ``data-wrangling-communication`` — Data cleaning, transformation, stakeholder communication
- ``statistical-analysis-fundamentals`` — Distributions, hypothesis tests, confidence intervals
- ``ml-pipeline-workflow`` — Airflow, Prefect, Metaflow, automated retraining
- ``ml-engineering-production`` — Type-safe code, testing, data pipelines, monitoring, drift
- ``model-deployment-serving`` — FastAPI, TorchServe, Triton, BentoML, REST/gRPC
- ``devops-ml-infrastructure`` — Docker, Kubernetes, GPU provisioning, cloud ML

Hub: ml-and-data-science (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for machine learning and data science.

Routes to hubs: ``machine-learning``

- ``data-analysis`` — Pandas, descriptive statistics, correlation, hypothesis testing
- ``data-wrangling-communication`` — Data cleaning, transformation, stakeholder communication
- ``statistical-analysis-fundamentals`` — Distributions, hypothesis tests, confidence intervals
- ``scientific-visualization`` — Matplotlib, seaborn, plotly, domain-specific plots
- ``nlsq-core-mastery`` — JAX-accelerated non-linear least squares curve fitting
- ``experiment-tracking`` — MLflow, Weights & Biases, DVC

Hub: time-series-analysis (2 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forecasting and analysis of regularly sampled series: ARIMA/SARIMA, GARCH, STL, stationarity tests, change-point and anomaly detection.

- ``point-processes`` — Hawkes processes, HSGP, Julia PointProcesses.jl, non-parametric Hawkes EM *(new in v3.1.4)*
- ``extreme-value-statistics`` — GEV/GPD/Hill/Pickands/POT, return levels, non-stationary EVT *(new in v3.1.4)*

Hub: ml-deployment (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for ML model deployment and operations.

- ``model-deployment-serving`` — FastAPI, TorchServe, Triton, BentoML, REST/gRPC
- ``model-optimization-deployment`` — Quantization, pruning, ONNX, TensorRT, mobile
- ``ml-engineering-production`` — Type-safe code, testing, data pipelines, monitoring, drift
- ``ml-pipeline-workflow`` — Airflow, Prefect, Metaflow, automated retraining
- ``devops-ml-infrastructure`` — Docker, Kubernetes, GPU provisioning, cloud ML
- ``federated-learning`` — Federated averaging, differential privacy, PySyft

Hub: llm-and-ai (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for LLM integration into scientific workflows and AI engineering.

- ``llm-application-dev`` — LLM-powered apps: API integration, streaming, tool use, agents
- ``llm-evaluation`` — Benchmarks, LLM-as-judge, human evaluation, quality metrics
- ``langchain-architecture`` — LangChain/LangGraph: chains, agents, memory, tools
- ``rag-implementation`` — Vector stores, chunking, re-ranking, hybrid retrieval
- ``nlp-fundamentals`` — Tokenization, embeddings, NER, text classification

Hub: python-development (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Master modern Python systems engineering for scientific computing.

- ``type-driven-design`` — Protocols, Generics, static analysis with pyright/mypy
- ``rust-extensions`` — PyO3/maturin high-performance Python extensions
- ``modern-concurrency`` — asyncio TaskGroups, threading, multiprocessing
- ``robust-testing`` — Property-based, metamorphic, and tolerance-aware testing for scientific code
- ``python-packaging-advanced`` — uv workspaces, monorepos, reproducible builds

Hub: research-and-domains (13 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specialized scientific domains, scientific Python systems, and AI-research methods not covered by the other hubs.

Routes to hubs: ``python-development``

- ``self-improving-ai`` — Research overview for autonomous self-improvement
- ``dspy-basics`` — Depth-skill companion for DSPy programmatic prompt optimization
- ``rlaif-training`` — Depth-skill companion for Constitutional AI / RLAIF / DPO
- ``python-packaging-advanced`` — uv workspaces, monorepos, reproducible builds
- ``rust-extensions`` — PyO3/maturin high-performance Python extensions
- ``type-driven-design`` — Protocols, Generics, static analysis with pyright/mypy
- ``modern-concurrency`` — asyncio TaskGroups, threading, multiprocessing
- ``robust-testing`` — Property-based, metamorphic, and tolerance-aware testing for scientific code
- ``quantum-computing`` — Qiskit, PennyLane, VQE/QAOA
- ``bioinformatics`` — Genomics, proteomics, BioPython
- ``computer-vision`` — Image processing, detection, Vision Transformers
- ``reinforcement-learning`` — Gymnasium, Stable-Baselines3, RLlib
- ``symbolic-math`` — SymPy, CAS, algebraic solvers

Hooks
-----

5 hook events with Python script implementations (``hooks/hooks.json``):

- ``SessionStart`` — Detect JAX devices, GPU availability, Julia env
- ``UserPromptSubmit`` — Remind agent to route through the matching hub skill before implementing
- ``PostToolUse`` — NaN/Inf check on compute job output (numerical integrity)
- ``SessionEnd`` — Persist structured progress summary for next session
- ``SubagentStop`` — Collect results from parallel science agents

(``ExecutionError`` was removed in v3.4.0 — not supported by the CC v2.1.113 CLI event schema. ``PreToolUse`` is not wired in ``hooks.json``.)
