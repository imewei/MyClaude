Scientific Computing Suite
==========================

Consolidated suite for high-performance computing, specialized physics/chemistry simulations, and data science workflows. Features 80 specialized skills with extended context and adaptive reasoning for Claude Opus 4.6.

Agents
------

.. agent:: ai-engineer
   :description: Build production-ready LLM applications, advanced RAG systems, and intelligent agents.
   :model: sonnet
   :version: 2.2.1

.. agent:: jax-pro
   :description: Expert in JAX-based scientific computing, functional transformations, and high-performance numerical kernels.
   :model: sonnet
   :version: 2.2.1

.. agent:: julia-pro
   :description: Expert in Julia programming, Scientific Machine Learning (SciML), and DifferentialEquations.jl.
   :model: sonnet
   :version: 2.2.1

.. agent:: ml-expert
   :description: Expert in classical ML algorithms, MLOps pipelines, and data engineering.
   :model: sonnet
   :version: 2.2.1

.. agent:: neural-network-master
   :description: Deep learning authority specializing in theory and implementation (Transformers, CNNs, diagnostics).
   :model: inherit
   :version: 2.2.1

.. agent:: prompt-engineer
   :description: Expert prompt engineer specializing in advanced techniques and performance optimization.
   :model: sonnet
   :version: 2.2.1

.. agent:: python-pro
   :description: Expert Python Systems Engineer specializing in type-driven development and performance.
   :model: sonnet
   :version: 2.2.1

.. agent:: research-expert
   :description: Expert in systematic research, evidence synthesis, and publication-quality visualization.
   :model: sonnet
   :version: 2.2.1

.. agent:: simulation-expert
   :description: Expert in molecular dynamics, statistical mechanics, and numerical methods (HPC/GPU).
   :model: sonnet
   :version: 2.2.1

.. agent:: statistical-physicist
   :description: Expert in correlation functions, non-equilibrium dynamics, and ensemble theory.
   :model: sonnet
   :version: 2.2.1

Skills
------

.. skill:: active-matter
   :description: Model active matter including self-propelled particles, flocking, pattern formation, and collective behavior. Use when simulating ABPs, Vicsek model, MIPS, reaction-diffusion systems, or designing bio-inspired materials.
   :version: 2.2.1

.. skill:: advanced-ml-systems
   :description: Build advanced deep learning with PyTorch 2.x, TensorFlow, JAX including architectures (CNNs, Transformers, GANs), distributed training (DDP, FSDP, DeepSpeed), hyperparameter optimization (Optuna, Ray Tune), and model optimization (quantization, pruning, distillation). Use when training scripts, custom architectures, or scaling to multi-GPU/TPU.
   :version: 2.2.1

.. skill:: advanced-simulations
   :description: Master advanced simulation techniques including non-equilibrium thermodynamics, stochastic dynamics, and multiscale modeling. Bridge scales from atomistic to mesoscale.
   :version: 2.2.1

.. skill:: catalyst-reactions
   :description: Model chemical reaction networks with Catalyst.jl for deterministic and stochastic simulations. Use when modeling biochemical pathways or chemical kinetics.
   :version: 2.2.1

.. skill:: ci-cd-patterns
   :description: Master GitHub Actions for Julia packages with test matrices, CompatHelper, TagBot, and documentation deployment. Use when setting up CI workflows for Julia packages.
   :version: 2.2.1

.. skill:: compiler-patterns
   :description: Create system images and standalone executables with PackageCompiler.jl. Use when reducing startup time or deploying Julia applications without requiring Julia installation.
   :version: 2.2.1

.. skill:: core-julia-patterns
   :description: Master multiple dispatch, type system, parametric types, type stability, and metaprogramming for high-performance Julia. Use when designing type hierarchies, debugging @code_warntype issues, optimizing with @inbounds/@simd/StaticArrays, or writing macros and generated functions.
   :version: 2.2.1

.. skill:: correlation-computational-methods
   :description: Implement efficient correlation algorithms including FFT-based O(N log N) autocorrelation (50,000× speedup), multi-tau correlators for wide dynamic ranges, KD-tree spatial correlations, JAX-accelerated GPU computation (200× speedup), and statistical validation (bootstrap, convergence). Use when optimizing calculations for MD trajectories, DLS, XPCS, or large experimental datasets.
   :version: 2.2.1

.. skill:: correlation-experimental-data
   :description: Interpret experimental correlation data from DLS (g₂(τ), Siegert relation, Stokes-Einstein), SAXS/SANS (S(q), g(r), Guinier/Porod analysis), XPCS (two-time correlation, aging), FCS (diffusion time, binding kinetics), and rheology (Green-Kubo, microrheology). Extract physical parameters, perform Bayesian model fitting, and validate against theoretical predictions.
   :version: 2.2.1

.. skill:: correlation-math-foundations
   :description: Mathematical foundations of correlation functions including two-point C(r), higher-order χ₄(t), cumulants, Fourier/Laplace/wavelet transforms, Wiener-Khinchin theorem, Ornstein-Zernike equations, fluctuation-dissipation theorem, and Green's functions. Use when developing correlation theory or connecting microscopic correlations to macroscopic response.
   :version: 2.2.1

.. skill:: correlation-physical-systems
   :description: Map correlation functions to physical systems including condensed matter (spin correlations, critical exponents ξ ~ |T-Tc|^(-ν)), soft matter (polymer Rouse/Zimm dynamics, colloidal g(r), glass χ₄), biological systems (protein folding, membrane fluctuations), and non-equilibrium (active matter, transfer entropy). Use for materials characterization, transport predictions, or connecting experiments to theory.
   :version: 2.2.1

.. skill:: data-analysis
   :description: Analyze experimental correlation data from DLS, SAXS/SANS, rheology, and microscopy using Green-Kubo relations, Bayesian inference (MCMC), and model validation. Use when interpreting scattering data, validating non-equilibrium theories, or predicting transport coefficients.
   :version: 2.2.1

.. skill:: data-wrangling-communication
   :description: Comprehensive data wrangling, cleaning, feature engineering, and visualization workflows using pandas, NumPy, Matplotlib, Seaborn, and Plotly. Use when cleaning messy datasets, handling missing values, dealing with outliers, engineering features, performing EDA, creating statistical visualizations, building interactive dashboards with Plotly Dash or Streamlit, or presenting data-driven insights to stakeholders.
   :version: 2.2.1

.. skill:: deep-learning
   :description: Master deep learning architecture design, theory, and implementation. Covers neural network mathematics, training diagnostics, PyTorch/JAX frameworks, and advanced patterns.
   :version: 2.2.1

.. skill:: deep-learning-experimentation
   :description: Design systematic deep learning experiments with hyperparameter optimization, ablation studies, and reproducible workflows. Use when tuning hyperparameters, conducting ablations, setting up experiment tracking (W&B, TensorBoard, MLflow), or managing reproducibility.
   :version: 2.2.1

.. skill:: devops-ml-infrastructure
   :description: DevOps for ML with GitHub Actions pipelines, Terraform IaC, Docker/Kubernetes, and cloud ML platforms (SageMaker, Azure ML, Vertex AI). Use when automating training, deploying models, or provisioning ML infrastructure.
   :version: 2.2.1

.. skill:: differential-equations
   :description: Solve ODE/SDE/PDE with DifferentialEquations.jl. Use when defining differential equation systems, selecting solvers, implementing callbacks, or creating ensemble simulations.
   :version: 2.2.1

.. skill:: ecosystem-selection
   :description: Select optimal scientific computing ecosystems and manage multi-language workflows. Use when evaluating Python vs Julia for performance-critical numerical computing, implementing hybrid PyJulia/PyCall.jl interoperability, or setting up reproducible toolchains with Conda or Pkg.jl.
   :version: 2.2.1

.. skill:: evidence-synthesis
   :description: Conduct systematic literature reviews (PRISMA), meta-analyses, and evidence grading (GRADE). Use when synthesizing research findings, evaluating evidence quality, or conducting comprehensive literature searches.
   :version: 2.2.1

.. skill:: gpu-acceleration
   :description: Implement GPU acceleration using CUDA/CuPy (Python) and CUDA.jl (Julia) with kernel optimization and memory management. Use when offloading computations to GPU, writing custom kernels, or optimizing multi-GPU workflows.
   :version: 2.2.1

.. skill:: interop-patterns
   :description: Master cross-language integration with PythonCall.jl, RCall.jl, and CxxWrap.jl. Use when calling Python/R libraries from Julia or minimizing data transfer overhead.
   :version: 2.2.1

.. skill:: jax-bayesian-pro
   :description: This skill should be used when the user asks to "write Bayesian models in JAX", "use NumPyro or Blackjax", "implement custom MCMC samplers", "tune HMC mass matrix", "debug NUTS divergences", "run parallel MCMC chains", "integrate Diffrax with Bayesian inference", "build neural surrogate models for SBI", "implement effect handlers", "write pure log-prob functions", or needs expert guidance on probabilistic programming, simulation-based inference, or Bayesian parameter estimation for physics/soft matter.
   :version: 2.2.1

.. skill:: jax-core-programming
   :description: Master JAX functional transformations (jit, vmap, pmap, grad), Flax NNX neural networks, Optax optimizers, Orbax checkpointing, and NumPyro Bayesian inference. Use when writing JAX code, building training loops, optimizing XLA compilation, debugging tracer errors, or scaling to multi-device GPU/TPU.
   :version: 2.2.1

.. skill:: jax-diffeq-pro
   :description: This skill should be used when the user asks to "solve ODEs in JAX", "use Diffrax", "implement stiff solvers", "choose implicit vs explicit solvers", "backpropagate through ODEs", "use adjoint methods", "implement RecursiveCheckpointAdjoint", "solve SDEs", "use VirtualBrownianTree", "handle ODE events", "find steady states with Optimistix", "use Lineax for linear solves", or needs expert guidance on differentiable physics, neural ODEs, rheology simulations, or soft matter dynamics.
   :version: 2.2.1

.. skill:: jax-mastery
   :description: Master JAX for high-performance scientific computing. Covers functional transformations (JIT, vmap, pmap, grad), neural networks (Flax NNX), and specialized optimization (NLSQ, NumPyro).
   :version: 2.2.1

.. skill:: jax-optimization-pro
   :description: This skill should be used when the user asks to "optimize JAX code for production", "write JAX-first code", "debug ConcretizationError", "analyze XLA/HLO output", "implement SPMD parallelism", "use jax.sharding for TPU pods", "write Pallas/Triton kernels", "fix tracer errors", "optimize GPU/TPU memory", "handle numerical stability", "implement custom VJPs", or needs expert-level guidance on functional programming patterns, PyTree manipulation, multi-device scaling, or XLA compiler optimization.
   :version: 2.2.1

.. skill:: jax-physics-applications
   :description: Physics simulations using JAX-based libraries (JAX-MD, JAX-CFD, PINNs). Use when implementing molecular dynamics, computational fluid dynamics, physics-informed neural networks with PDE constraints, quantum computing algorithms (VQE, QAOA), multi-physics coupling, or differentiable physics for gradient-based optimization.
   :version: 2.2.1

.. skill:: julia-mastery
   :description: Master the Julia language for scientific computing. Covers multiple dispatch, type stability, metaprogramming, and the SciML ecosystem.
   :version: 2.2.1

.. skill:: jump-optimization
   :description: Master JuMP.jl for LP, QP, NLP, and MIP with HiGHS, Ipopt, and commercial solvers. Use for production planning, portfolio optimization, scheduling, and constrained optimization. Note that JuMP.jl is separate from Optimization.jl (julia-pro).
   :version: 2.2.1

.. skill:: langchain-architecture
   :description: Design LLM applications with LangChain agents, chains, memory, and tools for autonomous agents and RAG systems.

.. skill:: llm-application-dev
   :description: Build production-ready LLM applications, RAG systems, and AI agents. Covers prompt engineering, LangChain/LangGraph architecture, and evaluation.
   :version: 2.2.1

.. skill:: llm-evaluation
   :description: Implement LLM evaluation with automated metrics, LLM-as-judge patterns, human evaluation frameworks, and regression detection.

.. skill:: machine-learning
   :description: Comprehensive Classical Machine Learning suite. Covers scikit-learn, XGBoost, LightGBM, and MLOps pipelines. Focuses on tabular data, feature engineering, and production deployment.
   :version: 2.2.1

.. skill:: machine-learning-essentials
   :description: Core ML workflows with scikit-learn, XGBoost, LightGBM including algorithm selection, cross-validation, hyperparameter tuning (GridSearch, Optuna), handling imbalanced data (SMOTE), model evaluation, SHAP interpretability, and deployment. Use when building classification/regression models or evaluating ML performance.
   :version: 2.2.1

.. skill:: mcmc-diagnostics
   :description: Master MCMC convergence diagnostics with R-hat, ESS, trace plots, and divergence checking. Use when validating Bayesian inference results from Turing.jl.
   :version: 2.2.1

.. skill:: md-simulation-setup
   :description: Set up classical MD simulations using LAMMPS, GROMACS, and HOOMD-blue for materials and biomolecular systems. Use when writing input scripts, selecting force fields, configuring ensembles, or optimizing parallel execution.
   :version: 2.2.1

.. skill:: ml-engineering-production
   :description: Software and data engineering best practices for production ML. Type-safe code, pytest testing, pre-commit hooks, pandas/SQL pipelines, and modern project structure. Use when building maintainable ML systems.
   :version: 2.2.1

.. skill:: ml-force-fields
   :description: Develop ML force fields (NequIP, MACE, DeepMD) achieving quantum accuracy with 1000x speedup. Use when training neural network potentials or deploying ML force fields in MD.
   :version: 2.2.1

.. skill:: ml-pipeline-workflow
   :description: Build end-to-end MLOps pipelines with Airflow, Dagster, Kubeflow, or Prefect for data preparation, training, validation, and deployment. Use when creating DAG definitions, workflow configs, or orchestrating ML lifecycle stages.
   :version: 2.2.1

.. skill:: model-deployment-serving
   :description: Deploy ML models with FastAPI, TorchServe, BentoML, Docker, Kubernetes, and cloud platforms. Implement monitoring, A/B testing, and drift detection. Use when building model serving APIs, containerizing models, or setting up production ML infrastructure.
   :version: 2.2.1

.. skill:: model-optimization-deployment
   :description: Optimize and deploy neural networks with quantization, pruning, knowledge distillation, and production serving. Use when compressing models, converting between frameworks (ONNX, TFLite), or setting up TorchServe/Triton serving infrastructure.
   :version: 2.2.1

.. skill:: modeling-toolkit
   :description: Define symbolic differential equations with ModelingToolkit.jl for automatic simplification and code generation. Use when building complex mathematical models declaratively.
   :version: 2.2.1

.. skill:: modern-concurrency
   :description: Master structured concurrency in Python using asyncio TaskGroups and modern primitives. Use when implementing concurrent I/O, managing task lifecycles, or optimizing async applications for Python 3.11+.
   :version: 2.2.1

.. skill:: multiscale-modeling
   :description: Bridge atomistic MD to mesoscale using coarse-graining, DPD, and nanoscale DEM. Use when developing CG models, implementing DPD simulations, or coupling scales.
   :version: 2.2.1

.. skill:: neural-architecture-patterns
   :description: Design neural architectures with skip connections, attention, normalization, and encoder-decoders. Use when designing CNNs, transformers, U-Nets, or selecting architectures for vision, NLP, and multimodal tasks.
   :version: 2.2.1

.. skill:: neural-network-mathematics
   :description: Apply mathematical foundations of neural networks including linear algebra, calculus, probability theory, optimization, and information theory. Use when deriving backpropagation for custom layers, computing Jacobians/Hessians, implementing automatic differentiation, analyzing gradient flow, proving convergence, working with Bayesian deep learning, deriving loss functions from MLE principles, or understanding PAC learning theory.
   :version: 2.2.1

.. skill:: neural-pde
   :description: Solve PDEs with physics-informed neural networks using NeuralPDE.jl. Use when solving PDEs with neural networks, enforcing boundary conditions, or combining ML with physics.
   :version: 2.2.1

.. skill:: nlsq-core-mastery
   :description: Master NLSQ library for high-performance curve fitting (150-270x faster than SciPy). Use when fitting >10K points, parameter estimation, robust optimization, streaming datasets (100M+ points), or migrating from SciPy.
   :version: 2.2.1

.. skill:: non-equilibrium-theory
   :description: Apply non-equilibrium thermodynamics including fluctuation theorems, entropy production, and linear response theory. Use when modeling irreversible processes, analyzing driven systems, or deriving transport coefficients.
   :version: 2.2.1

.. skill:: numerical-methods-implementation
   :description: Implement robust numerical algorithms for ODEs (RK45, BDF), PDEs (finite difference, FEM), optimization (L-BFGS, Newton), and molecular simulations. Master solver selection, stability analysis, and differentiable physics using Python and Julia.
   :version: 2.2.1

.. skill:: numpyro-core-mastery
   :description: Master NumPyro for production Bayesian inference, MCMC sampling (NUTS/HMC), variational inference (SVI), hierarchical models, and uncertainty quantification. Use when building probabilistic models with numpyro.sample(), running MCMC with NUTS/HMC, implementing SVI with AutoGuides, diagnosing convergence (R-hat, ESS, divergences), or deploying production Bayesian pipelines.
   :version: 2.2.1

.. skill:: optimization-patterns
   :description: Use Optimization.jl for parameter estimation in differential equations. Use when fitting models to data or solving inverse problems. For LP/QP/MIP, use JuMP.jl instead.
   :version: 2.2.1

.. skill:: package-development-workflow
   :description: Create Julia packages following community standards with proper structure, exports, and PkgTemplates.jl. Use when creating new packages or organizing source code.
   :version: 2.2.1

.. skill:: package-management
   :description: Master Julia package management with Pkg.jl, Project.toml, and Manifest.toml for reproducible environments. Use when managing dependencies, specifying compatibility bounds, or setting up project environments.
   :version: 2.2.1

.. skill:: parallel-computing
   :description: Implement high-performance parallel computing across CPUs and GPUs using Python (CUDA/CuPy) and Julia (CUDA.jl/Distributed.jl). Master multi-threading, distributed systems, and kernel optimization.
   :version: 2.2.1

.. skill:: parallel-computing-strategy
   :description: Design parallel strategies with MPI (distributed memory), OpenMP (shared memory), hybrid MPI+OpenMP, SLURM scheduling, Dask/Dagger.jl workflows, and load balancing. Use when implementing multi-node parallelization, writing job scripts, or optimizing HPC workflows.
   :version: 2.2.1

.. skill:: performance-tuning
   :description: Profile and optimize Julia code with @code_warntype, @profview, and BenchmarkTools.jl. Use when debugging slow code, reducing allocations, or improving execution speed.
   :version: 2.2.1

.. skill:: prompt-engineering-patterns
   :description: Master advanced prompt engineering with chain-of-thought, few-shot learning, and production templates for maximizing LLM reliability.

.. skill:: python-development
   :description: Master modern Python systems engineering for scientific computing. Covers type-driven design, Rust extensions (PyO3), structured concurrency (TaskGroups), robust testing (Hypothesis), and uv-based packaging.
   :version: 2.2.1

.. skill:: python-packaging-advanced
   :description: Master modern Python packaging using uv, focusing on workspaces, monorepos, and reproducible builds. Use when configuring pyproject.toml for uv, setting up monorepo workspaces, managing toolchains with uv, defining dependency groups, or publishing high-performance Python libraries.
   :version: 2.2.1

.. skill:: rag-implementation
   :description: Build production RAG systems with vector databases, embeddings, chunking strategies, hybrid search, and grounded prompts.

.. skill:: research-methodology
   :description: Systematic framework for scientific research, covering experimental design, statistical rigor, quality assessment, and publication readiness.
   :version: 2.2.1

.. skill:: research-paper-implementation
   :description: Translate research papers into production implementations through systematic analysis and architecture extraction. Use when implementing novel architectures, reproducing experiments, or adapting state-of-the-art methods.
   :version: 2.2.1

.. skill:: research-quality-assessment
   :description: Evaluate scientific research quality across methodology, experimental design, statistical rigor, and publication readiness. Use when reviewing papers, grant proposals, or assessing reproducibility using CONSORT, STROBE, or PRISMA guidelines.
   :version: 2.2.1

.. skill:: robust-testing
   :description: Implement robust testing strategies using property-based testing, advanced fixtures, and mutation testing. Use when writing tests with Hypothesis, implementing complex pytest fixtures, or ensuring high reliability in scientific computations.
   :version: 2.2.1

.. skill:: rust-extensions
   :description: Build high-performance Python extensions using Rust, PyO3, and Maturin. Use when optimizing performance-critical bottlenecks, implementing native modules, or bridging Rust libraries to Python.
   :version: 2.2.1

.. skill:: scientific-communication
   :description: Structure scientific arguments, write technical reports (IMRaD), and ensure clarity and precision in scientific writing. Use when drafting papers, creating posters, or structuring technical documentation.
   :version: 2.2.1

.. skill:: scientific-visualization
   :description: Create publication-quality scientific visualizations across physics, biology, chemistry, and climate science. Supports uncertainty quantification, multi-dimensional data, and domain-specific plots in both Python and Julia.
   :version: 2.2.1

.. skill:: sciml-ecosystem
   :description: Navigate the SciML ecosystem including DifferentialEquations.jl, ModelingToolkit.jl, Optimization.jl, and Catalyst.jl. Use when selecting packages for scientific computing tasks.
   :version: 2.2.1

.. skill:: statistical-analysis-fundamentals
   :description: Comprehensive statistical analysis with scipy.stats, statsmodels, and PyMC3 including hypothesis testing, Bayesian methods, regression, experimental design, and causal inference. Use when conducting A/B tests, power analysis, or treatment effect estimation.
   :version: 2.2.1

.. skill:: statistical-physics
   :description: Comprehensive statistical physics suite covering equilibrium and non-equilibrium statistical mechanics, active matter, stochastic dynamics, and correlation analysis. Master the bridge between microscopic laws and macroscopic behavior.
   :version: 2.2.1

.. skill:: stochastic-dynamics
   :description: Model stochastic dynamics using master equations, Fokker-Planck, Langevin dynamics, and Green-Kubo transport theory. Use when simulating noise-driven systems, calculating transport coefficients, or modeling rare events.
   :version: 2.2.1

.. skill:: testing-patterns
   :description: Master Test.jl, Aqua.jl quality checks, and JET.jl static analysis for Julia testing. Use when writing unit tests, organizing test suites, or validating package quality.
   :version: 2.2.1

.. skill:: training-diagnostics
   :description: Diagnose and resolve neural network training failures through systematic analysis of gradient pathologies, loss curves, and convergence issues. Use when encountering vanishing/exploding gradients, dead ReLU neurons, loss anomalies (NaN, spikes, plateaus), overfitting/underfitting patterns, or when debugging training scripts requiring systematic troubleshooting.
   :version: 2.2.1

.. skill:: trajectory-analysis
   :description: Analyze MD trajectories to extract structural, thermodynamic, mechanical, and transport properties. Use when calculating RDF, MSD, viscosity, or validating simulations against experimental data.
   :version: 2.2.1

.. skill:: turing-model-design
   :description: Design probabilistic models with Turing.jl including prior selection, hierarchical models, and non-centered parameterization. Use when building Bayesian models for inference.
   :version: 2.2.1

.. skill:: type-driven-design
   :description: Master type-driven design in Python using Protocols, Generics, and static analysis. Use when designing library interfaces, implementing structural typing, using Generic types for reusable components, or enforcing strict type safety with pyright/mypy.
   :version: 2.2.1

.. skill:: variational-inference-patterns
   :description: Master ADVI variational inference with Turing.jl and Bijectors.jl for scalable approximate Bayesian inference. Use when MCMC is too slow for large datasets.
   :version: 2.2.1

.. skill:: visualization-patterns
   :description: Master Plots.jl and Makie.jl for data visualization in Julia. Use when creating plots, selecting backends, building statistical visualizations, or making publication-quality figures.
   :version: 2.2.1

.. skill:: web-development-julia
   :description: Build web applications with Genie.jl MVC framework and HTTP.jl. Use when creating REST APIs, handling HTTP requests, or building web services in Julia.
   :version: 2.2.1

