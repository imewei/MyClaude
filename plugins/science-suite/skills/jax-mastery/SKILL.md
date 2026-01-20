---
name: jax-mastery
version: "1.1.0"
description: Master JAX for high-performance scientific computing. Covers functional transformations (JIT, vmap, pmap, grad), neural networks (Flax NNX), and specialized optimization (NLSQ, NumPyro).
---

# JAX Mastery

Expert guide for leveraging JAX's functional paradigm for scientific research and high-performance computing.

## Expert Agent

For complex implementation tasks, JAX optimization, and scientific computing workflows, delegate to the expert agent:

- **`jax-pro`**: Unified specialist for Core JAX, Bayesian inference (NumPyro), Nonlinear optimization (NLSQ), and Computational Physics (JAX-MD/CFD).
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Production-ready JIT/vmap patterns, distributed computing (pmap/sharding), differentiable physics, and probabilistic programming.

## Core Skills

### [JAX Core Programming](./jax-core-programming/SKILL.md)
Functional transformations, JIT compilation, and array operations.

### [JAX Bayesian Pro](./jax-bayesian-pro/SKILL.md)
Advanced probabilistic modeling and inference with NumPyro.

### [JAX DiffEq Pro](./jax-diffeq-pro/SKILL.md)
Numerical differential equation solvers with Diffrax.

### [JAX Optimization Pro](./jax-optimization-pro/SKILL.md)
Scientific optimization and functional gradients.

### [JAX Physics Applications](./jax-physics-applications/SKILL.md)
Molecular dynamics, CFD, and physics-informed neural networks.

### [NLSQ Core Mastery](./nlsq-core-mastery/SKILL.md)
GPU-accelerated nonlinear least squares and curve fitting.

### [NumPyro Core Mastery](./numpyro-core-mastery/SKILL.md)
Foundational Bayesian inference and MCMC diagnostics.

## 1. Core Transformations

- **`jax.jit`**: XLA compilation for massive speedups. Always use for hot paths.
- **`jax.vmap` / `jax.pmap`**: Automatic vectorization and multi-device parallelization.
- **`jax.grad` / `jax.value_and_grad`**: Automatic differentiation for gradient-based optimization.

### Efficient JIT usage
```python
@jax.jit
def fast_step(state, x):
    # Pure function: no side effects, explicit RNG
    return apply_fn(state, x)
```

## 2. Neural Networks & Optimization

- **Flax NNX**: Modern, module-based neural network library for JAX.
- **Optax**: Composable gradient transformation and optimization library.
- **Orbax**: Flexible checkpointing for long-running simulations.

## 3. Specialized Scientific Libraries

- **NLSQ**: High-performance curve fitting and non-linear least squares, optimized for GPUs.
- **NumPyro**: Probabilistic programming and Bayesian inference built on JAX.
- **JAX-MD / JAX-CFD**: End-to-end differentiable physics simulations.
- **Diffrax**: Numerical differential equation solvers (ODE/SDE) with JAX.

## 4. JAX Performance Checklist

- [ ] **Pure Functions**: Ensure functions have no side effects and rely only on inputs.
- [ ] **Static Arguments**: Use `static_argnums` for arguments that determine control flow or array shapes.
- [ ] **Control Flow**: Use `jax.lax.cond`, `jax.lax.scan`, and `jax.lax.fori_loop` instead of Python loops inside JIT.
- [ ] **Memory**: Use `jax.device_put` to manage data placement on GPU/TPU.
- [ ] **Debugging**: Use `jax.disable_jit()` and `jax.debug.print` for troubleshooting.
