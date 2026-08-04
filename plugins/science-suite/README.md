# Science Suite

Domain-specific scientific computing suite for high-performance computing, Julia ML/DL/HPC, specialized physics/chemistry simulations, and data science workflows. Agents primarily collaborate within the suite with only 2 outward delegation edges. Multi-tier model routing: opus for deep-math specialists, sonnet for engineering-heavy work, haiku for mechanical MLOps.

## Overview

Science Suite provides 12 specialized agents (7 opus, 4 sonnet, 1 haiku) covering JAX, Julia, physics, ML/DL, continuum mechanics, and nonlinear dynamics. Its 30 hub skills route to 107 sub-skills spanning nonlinear dynamics, Bayesian inference, statistical physics, simulation, and more. Opus agents handle deep reasoning (PINN/inverse-PDE theory, bifurcation theory, DL architecture) while sonnet agents handle implementation (JAX kernels, Julia code, Python systems, MD/HPC simulation). SessionStart hooks auto-detect JAX devices, GPU availability, and Julia environments. *(Research methodology moved to `research-suite` in v3.4.0.)*

## Quick Start / Usage Examples

```bash
# Ask JAX specialist to build a differentiable physics model
@jax-pro "Implement a Bayesian neural ODE with NumPyro + Diffrax"

# Nonlinear dynamics analysis
@nonlinear-dynamics-expert "Analyze the Lorenz system for bifurcations near r=24.74"

# Molecular dynamics simulation setup
@simulation-expert "Set up a JAX-MD Lennard-Jones NVT simulation with 10k particles"
```

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
| `jax-pro` | opus | Core JAX, NumPyro, Diffrax, JAX-MD |
| `julia-pro` | opus | Julia, SciML, DifferentialEquations.jl |
| `julia-ml-hpc` | sonnet | Julia ML, Deep Learning, HPC (Lux.jl, CUDA.jl, MPI.jl) |
| `ml-expert` | haiku | Classical ML, MLOps, data engineering |
| `simulation-expert` | sonnet | HPC, molecular dynamics, multiscale |
| `statistical-physicist` | opus | Soft matter, non-equilibrium, correlations |
| `python-pro` | sonnet | Python systems engineering, performance |
| `pinn-engineer` | opus | Physics-informed neural networks, NeuralPDE.jl, DeepXDE |
| `sci-workflow-engineer` | sonnet | LLM integration into scientific pipelines, codegen, automation |
| `neural-network-master` | opus | DL architecture, PINNs, theory, diagnostics |
| `nonlinear-dynamics-expert` | opus | Bifurcation theory, chaos, network dynamics, pattern formation |
| `continuum-mechanics-engineer` | opus | FEM/FEA, constitutive modeling, DMA/rheology, transient networks, nanocomposites |

## Commands

This suite registers **2 slash commands**: `/md-sim` (molecular dynamics simulation setup) and `/benchmark` (performance benchmarking). Reference command templates (`/run-experiment`, `/analyze-data`, `/adopt-code`) exist on disk for users to copy and adapt, but are not invoked directly. `/paper-review` was removed in v3.4.0; use `scientific-review` in `research-suite` instead.

## Skills (30 hubs → 107 sub-skills)

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
- **Bayesian Inference**: NumPyro/Turing, MCMC diagnostics, consensus MC & Pigeons.jl non-reversible PT, Bayesian UDEs (Turing + Diffrax), Bayesian PINNs, Hawkes/point processes, Bayesian SINDy equation discovery (horseshoe + NumPyro + Lorenz-63 worked example)
- **Nonlinear Dynamics**: Bifurcation (with Python escape hatch), chaos attractors, pattern formation, equation discovery (SINDy ecosystem), Julia ↔ Python handoff for nonlinear time-series tools (nolds, antropy, IDTxl, pyEDM)
- **Other Domains**: Computer vision, NLP, bioinformatics, time series, control theory, symbolic math, quantum computing, federated learning, reinforcement learning

## Hooks (6 events)

| Event | Purpose |
|-------|---------|
| SessionStart | Detect JAX devices, GPU availability, Julia env |
| UserPromptSubmit | Remind agent to route through the matching hub skill before implementing |
| PreToolUse | Warn before commands that could corrupt simulations |
| PostToolUse | NaN/Inf check on compute job output (numerical integrity) |
| SessionEnd | Persist structured progress summary for next session |
| SubagentStop | Collect results from parallel science agents |

(`ExecutionError` was removed in v3.4.0 — not supported by the CC v2.1.126 CLI event schema.)

## Integration / Workflow

Science Suite agents primarily collaborate within the suite (e.g., `simulation-expert` delegates to `jax-pro` for JAX-MD kernels, `neural-network-master` delegates to `julia-ml-hpc` for Lux.jl implementations). Cross-suite, science-suite delegates *out* to `research-suite` for methodology (power analysis, IMRaD writing, peer review). See `docs/integration-map.rst` for the full delegation graph.

## Installation

```bash
/plugin marketplace add imewei/MyClaude
/plugin install science-suite@marketplace
```

After installation, restart Claude Code for changes to take effect.

## License

MIT License
