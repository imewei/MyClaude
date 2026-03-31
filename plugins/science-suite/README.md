# Science Suite

Domain-specific scientific computing suite for high-performance computing, specialized physics/chemistry simulations, and data science workflows. Agents primarily collaborate within the suite with only 2 outward delegation edges. Optimized for Claude Opus 4.6.

## Features

- **JAX Mastery**: High-performance numerical computing, differentiable physics, and Bayesian inference (NumPyro, Diffrax).
- **Julia Pro**: Scientific machine learning (SciML), differential equations, and package development.
- **Machine Learning**: End-to-end workflows from data wrangling to production deployment (scikit-learn, XGBoost, Optuna).
- **Statistical Physics**: Equilibrium and non-equilibrium thermodynamics, active matter, and correlation analysis.
- **Simulation**: Molecular dynamics (MD), computational fluid dynamics (CFD), and multiscale modeling.
- **Research**: Systematic literature reviews, evidence synthesis, and publication-quality visualization.
- **Deep Learning**: Architecture design (Transformers, CNNs), training diagnostics, and neural network mathematics.
- **AI Engineering**: RAG systems, agentic workflows, and LLM application architecture.

## Agents

| Agent | Model | Specialization |
|-------|-------|----------------|
| `jax-pro` | sonnet | Core JAX, NumPyro, Diffrax, JAX-MD |
| `julia-pro` | sonnet | Julia, SciML, DifferentialEquations.jl |
| `ml-expert` | sonnet | Classical ML, MLOps, data engineering |
| `simulation-expert` | sonnet | HPC, molecular dynamics, multiscale |
| `statistical-physicist` | opus | Soft matter, non-equilibrium, correlations |
| `research-expert` | opus | Methodology, visualization, literature |
| `python-pro` | sonnet | Python systems engineering, performance |
| `ai-engineer` | sonnet | RAG, agents, LLM apps |
| `prompt-engineer` | sonnet | Prompt optimization, safety, evaluation |
| `neural-network-master` | sonnet | DL architecture, PINNs, theory, diagnostics |

## Skills (78)

Organized by domain:

- **JAX**: Core programming, Bayesian inference, DiffEq, optimization, physics applications
- **Julia**: Core patterns, SciML ecosystem, performance tuning, package development
- **Machine Learning**: Algorithm selection, pipelines, deployment, deep learning, neural architectures
- **Statistical Physics**: Correlation functions, stochastic dynamics, active matter, non-equilibrium theory
- **Simulation**: MD setup, ML force fields, multiscale modeling, trajectory analysis
- **Research**: Methodology, evidence synthesis, scientific communication, visualization
- **Python**: Type-driven design, Rust extensions (PyO3), modern concurrency, packaging
- **Numerical Methods**: Solvers, parallel computing, GPU acceleration

## Installation

```bash
/plugin marketplace add imewei/MyClaude
/plugin install science-suite@marketplace
```

After installation, restart Claude Code for changes to take effect.

## License

MIT License
