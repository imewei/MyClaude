---
name: jax-mastery
description: Master JAX for high-performance scientific computing. Covers functional transformations (JIT, vmap, pmap, grad), neural networks (Flax NNX), and specialized optimization (NLSQ, NumPyro). Use when writing JAX code, debugging JIT compilation, vectorizing computations with vmap, or building Flax models.
---

# JAX Mastery

Expert guide for leveraging JAX's functional paradigm for scientific research and high-performance computing.

## Expert Agent

For complex implementation tasks, JAX optimization, and scientific computing workflows, delegate to the expert agent:

- **`jax-pro`**: Unified specialist for Core JAX, Bayesian inference (NumPyro), Nonlinear optimization (NLSQ), and Computational Physics (JAX-MD/CFD).
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Production-ready JIT/vmap patterns, distributed computing (pmap/sharding), differentiable physics, and probabilistic programming.

> For routing to JAX sub-skills, use the parent hub: `science-suite:jax-computing`.

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

- **Equinox**: PyTree-native neural networks for scientific computing (Diffrax integration). See `jax-core-programming` skill.
- **Flax NNX**: Structured neural network library for large-scale ML training.
- **Optax**: Composable gradient transformation and optimization library.
- **Orbax**: Flexible checkpointing for long-running simulations.
- **Lineax**: JIT-compatible linear solvers (CG, GMRES, LU). See `jax-core-programming` skill.
- **Optimistix**: Root-finding, fixed-point iteration, least-squares. See `jax-core-programming` skill.
- **interpax**: JIT-safe interpolation (replaces scipy.interpolate). See `jax-core-programming` skill.

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
