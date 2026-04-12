# Science Suite

Domain-specific scientific computing suite for high-performance computing, Julia ML/DL/HPC, specialized physics/chemistry simulations, and data science workflows. Agents primarily collaborate within the suite with only 2 outward delegation edges. Optimized for Claude Opus 4.6.

## Features

- **JAX Mastery**: High-performance numerical computing, differentiable physics, and Bayesian inference (NumPyro, Diffrax).
- **Julia Pro**: Scientific machine learning (SciML), differential equations, and package development.
- **Machine Learning**: End-to-end workflows from data wrangling to production deployment (scikit-learn, XGBoost, Optuna).
- **Statistical Physics**: Equilibrium and non-equilibrium thermodynamics, active matter, and correlation analysis.
- **Simulation**: Molecular dynamics (MD), computational fluid dynamics (CFD), and multiscale modeling.
- **Research**: Systematic literature reviews, evidence synthesis, and publication-quality visualization.
- **Deep Learning**: Architecture design (Transformers, CNNs), training diagnostics, and neural network mathematics.
- **AI Engineering**: RAG systems, agentic workflows, and LLM application architecture.
- **Julia ML/DL/HPC**: Neural networks (Lux.jl/Flux.jl), ML pipelines (MLJ.jl), GPU kernels (CUDA.jl), and distributed HPC (MPI.jl).

## Agents

| Agent | Model | Specialization |
|-------|-------|----------------|
| `jax-pro` | sonnet | Core JAX, NumPyro, Diffrax, JAX-MD |
| `julia-pro` | sonnet | Julia, SciML, DifferentialEquations.jl |
| `julia-ml-hpc` | sonnet | Julia ML, Deep Learning, HPC (Lux.jl, CUDA.jl, MPI.jl) |
| `ml-expert` | sonnet | Classical ML, MLOps, data engineering |
| `simulation-expert` | opus | HPC, molecular dynamics, multiscale |
| `statistical-physicist` | opus | Soft matter, non-equilibrium, correlations |
| `research-expert` | opus | Methodology, visualization, literature |
| `python-pro` | sonnet | Python systems engineering, performance |
| `ai-engineer` | sonnet | RAG, agents, LLM apps |
| `prompt-engineer` | sonnet | Prompt optimization, safety, evaluation |
| `neural-network-master` | opus | DL architecture, PINNs, theory, diagnostics |
| `nonlinear-dynamics-expert` | opus | Bifurcation theory, chaos, network dynamics, pattern formation |

## Commands (3)

| Command | Description |
|---------|-------------|
| `/run-experiment` | Design and execute reproducible scientific experiments |
| `/analyze-data` | Comprehensive data analysis with statistical tests and visualization |
| `/paper-review` | Systematic review of scientific papers |

## Skills (14 hubs → 116 sub-skills)

Organized by domain:

- **JAX**: Core programming, Bayesian inference, DiffEq, optimization, physics applications
- **Julia**: Core patterns, SciML ecosystem, performance tuning, package development
- **Machine Learning**: Algorithm selection, pipelines, deployment, deep learning, neural architectures, experiment tracking
- **Statistical Physics**: Correlation functions, stochastic dynamics (Fokker-Planck PDE), active matter, non-equilibrium theory (BAR/Jarzynski/MBAR), rare events / avalanche statistics, extreme-value statistics
- **Simulation**: MD setup, ML force fields (equivariant GNNs, Julia ACE), multiscale modeling, trajectory analysis, freud physical correlations
- **Research**: Methodology, evidence synthesis, scientific communication, visualization, robust testing, autonomous self-improvement research (DSPy, RLAIF)
- **Python**: Type-driven design, Rust extensions (PyO3), modern concurrency, packaging
- **Numerical Methods**: Solvers, parallel computing, GPU acceleration, signal processing
- **Julia ML/DL/HPC**: Neural networks, AD backends, ML pipelines, GPU kernels, HPC distributed computing, model deployment, GNNs, reinforcement learning
- **Bayesian Inference**: NumPyro/Turing, MCMC diagnostics, consensus MC & Pigeons.jl non-reversible PT, Bayesian UDEs (Turing + Diffrax), Bayesian PINNs, Hawkes/point processes
- **Nonlinear Dynamics**: Bifurcation (with Python escape hatch), chaos attractors, pattern formation, equation discovery (SINDy ecosystem), Julia ↔ Python handoff for nonlinear time-series tools (nolds, antropy, IDTxl, pyEDM)
- **Other Domains**: Computer vision, NLP, bioinformatics, time series, control theory, symbolic math, quantum computing, federated learning, reinforcement learning

## Installation

```bash
/plugin marketplace add imewei/MyClaude
/plugin install science-suite@marketplace
```

After installation, restart Claude Code for changes to take effect.

## License

MIT License
