# JAX Implementation

Production-ready JAX-based scientific computing with systematic Chain-of-Thought frameworks, Constitutional AI principles, and comprehensive response verification protocols for NumPyro Bayesian inference (with ArviZ integration and Consensus Monte Carlo), Flax NNX neural networks, NLSQ optimization, and physics simulations with measurable quality targets and proven optimization patterns.

**Version:** 1.0.5 | **Category:** scientific-computing | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/jax-implementation.html)


## What's New in v1.0.5

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## Agents (4)

### jax-pro

**Status:** active | **Maturity:** 92% (↑22 points from v1.0.0)

Master JAX programming with **6-step decision framework** (Problem Analysis, Transformation Strategy, Performance Optimization, Flax NNX Architecture, Debugging, Production Readiness). Implements **4 Constitutional AI principles** (Functional Purity 95%, Performance 90%, Code Quality 88%, Ecosystem Best Practices 92%).

**Comprehensive examples**:
- NumPy → JAX (112x single-GPU, 562x multi-GPU speedup)
- Flax Linen → NNX (10x faster checkpointing, 52% code reduction)

Expert in JAX transformations, Flax NNX, Optax, Orbax, and GPU/TPU acceleration.

### numpyro-pro

**Status:** active | **Maturity:** 93% (↑18 points from v1.0.0)

Master NumPyro Bayesian inference with **6-step framework** (Bayesian Formulation, Model Specification, Inference Strategy, Convergence Diagnostics, Performance Optimization, Production Deployment). Implements **4 Constitutional AI principles** (Statistical Rigor 95%, Computational Efficiency 90%, Model Quality 88%, NumPyro Best Practices 92%).

**Comprehensive examples**:
- Simple Regression → Hierarchical Bayesian (R-hat < 1.01, 50x GPU speedup)
- Centered → Non-Centered Parameterization (100% divergence reduction, 40x ESS improvement)

Expert in MCMC (NUTS, HMC), variational inference (SVI), and JAX-accelerated probabilistic programming.

### jax-scientist

**Status:** active | **Maturity:** 91% (↑19 points from v1.0.0)

Master computational physics with JAX leveraging **6-step framework** (Physics Analysis, JAX-Physics Integration, Numerical Methods, Performance & Scaling, Validation, Production Deployment). Implements **4 Constitutional AI principles** (Physical Correctness 94%, Computational Efficiency 90%, Code Quality 88%, Domain Library Best Practices 91%).

**Comprehensive examples**:
- LAMMPS → JAX-MD (100x parameter optimization speedup, 50x GPU speedup)
- Finite Difference → PINN (50x faster, continuous solution, inverse problems)

Expert in JAX-MD, JAX-CFD, PINNs, quantum computing, and differentiable physics.

### nlsq-pro

**Status:** active | **Maturity:** 90% (NEW AGENT in v1.0.1)

Master GPU-accelerated nonlinear least squares with **6-step framework** (Problem Characterization, Algorithm Selection, Performance Optimization, Convergence & Robustness, Validation, Production Deployment). Implements **4 Constitutional AI principles** (Numerical Stability 92%, Computational Efficiency 90%, Code Quality 85%, NLSQ Best Practices 88%).

**Comprehensive examples**:
- SciPy → NLSQ GPU (265x speedup, 12x accuracy improvement)
- Batch L2 → Streaming Huber (100x memory reduction, 10x outlier robustness)

Expert in JAX-accelerated optimization, robust loss functions, and production curve fitting.

## Skills (4)

### jax-core-programming

Master JAX functional programming for high-performance array computing and machine learning. **Enhanced with 12+ trigger scenarios** including JAX code writing (import jax, jax.numpy), implementing transformations (jit, vmap, pmap, grad), building Flax NNX models, configuring Optax optimizers, implementing Orbax checkpointing, debugging tracer errors, scaling to multi-device GPU/TPU, and integrating with JAX ecosystem. Includes comprehensive examples, performance patterns, and production best practices.

### numpyro-core-mastery

Master NumPyro probabilistic programming for Bayesian inference using JAX. **Enhanced with 15+ trigger scenarios** including building Bayesian models (numpyro.sample), implementing MCMC (NUTS, HMC), running variational inference (SVI, AutoGuides), working with hierarchical models, diagnosing convergence (R-hat, ESS, divergences), implementing non-centered parameterization, building probabilistic ML models (BNN, GP), performing model comparison (WAIC, LOO), and deploying production Bayesian pipelines. Includes MCMC/VI workflows, diagnostics, and real-world applications.

### jax-physics-applications

Comprehensive workflows for physics simulations using JAX-based libraries (JAX-MD, JAX-CFD, PINNs). **Enhanced with 12+ trigger scenarios** including implementing molecular dynamics (energy potentials, integrators), building CFD solvers (Navier-Stokes, turbulence), designing PINNs with PDE constraints, developing quantum algorithms (VQE, QAOA), coupling multi-physics domains, validating physics correctness (energy/mass/momentum conservation), implementing differentiable physics, and scaling physics simulations to GPU/TPU. Includes complete MD/CFD/PINN/quantum examples with validation.

### nlsq-core-mastery

**NEW SKILL** - GPU/TPU-accelerated nonlinear least squares optimization using NLSQ library with JAX. **13+ trigger scenarios** including fitting nonlinear models to data (from nlsq import CurveFit), performing parameter estimation with large datasets (>10K points, 150-270x faster than SciPy), implementing robust fitting with outliers (Huber, Cauchy loss), handling streaming optimization for massive datasets, working with exponential decay/dose-response/multi-peak fitting, comparing TRF vs LM algorithms, diagnosing convergence, and deploying production curve fitting in physics/biology/chemistry/engineering. Includes complete examples, benchmarks, and diagnostics.

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `jax-implementation` plugin
3. Activate an agent (e.g., `@jax-pro`, `@numpyro-pro`, `@jax-scientist`, `@nlsq-pro`)

### Example Usage

**JAX Multi-Device Training**:
```
@jax-pro Help me convert this NumPy training loop to JAX with multi-GPU support
```

**Bayesian Inference**:
```
@numpyro-pro I'm getting divergences in my hierarchical model, can you help diagnose and fix convergence issues?
```

**Physics Simulations**:
```
@jax-scientist I need to optimize force field parameters for a molecular dynamics simulation - can you help implement differentiable MD with JAX-MD?
```

**Curve Fitting**:
```
@nlsq-pro My SciPy curve_fit is too slow for production - help me migrate to GPU-accelerated NLSQ
```

## Performance Metrics

| Agent | Speedup | Accuracy | Memory |
|-------|---------|----------|--------|
| jax-pro | 112-562x (multi-GPU) | - | 50% reduction |
| numpyro-pro | 50x (GPU MCMC) | R-hat < 1.01 | - |
| jax-scientist | 50-100x | Continuous solution | - |
| nlsq-pro | 265x | 12x improvement | 100x reduction |

## Integration

This plugin integrates with:
- **scientific-computing-workflows**: Multi-agent orchestration for complex scientific workflows
- **julia-development**: Cross-language scientific computing patterns
- **python-development**: Modern Python integration with JAX ecosystem

See the full documentation for integration patterns and compatible plugins.

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/jax-implementation.html)

To build documentation locally:

```bash
cd docs/
make html
```
