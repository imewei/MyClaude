---
name: jax-computing
description: Meta-orchestrator for JAX scientific computing. Routes to core JAX, optimization, Bayesian inference, differential equations, and physics application skills. Use when writing JAX code, optimizing JIT compilation, implementing Bayesian models with NumPyro, solving ODEs with Diffrax, or running physics simulations.
---

# JAX Computing

Orchestrator for JAX scientific computing. Routes problems to the appropriate specialized skill based on the computation type.

## Expert Agent

- **`jax-pro`**: Specialist for JAX-based scientific computing, GPU acceleration, and JIT compilation.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: vmap/pmap, JIT, custom VJPs, GPU-accelerated physics simulations, Bayesian pipelines.

## Core Skills

### [JAX Mastery](../jax-mastery/SKILL.md)
Comprehensive JAX patterns: JIT, vmap, grad, pmap, and functional transformations. Start here for general JAX questions.

### [JAX Core Programming](../jax-core-programming/SKILL.md)
Low-level JAX: pytrees, custom primitives, XLA operations, and device memory management.

### [JAX Optimization Pro](../jax-optimization-pro/SKILL.md)
Gradient-based optimization with optax, custom schedules, NLSQ, and convergence diagnostics.

### [JAX Bayesian Pro](../jax-bayesian-pro/SKILL.md)
JAX-specific NumPyro integration patterns, GPU-accelerated sampling, and JAX-native posterior workflows. For general Bayesian modeling (model design, prior selection, MCMC diagnostics), prefer the `bayesian-inference` hub.

### [JAX DiffEq Pro](../jax-diffeq-pro/SKILL.md)
Diffrax solvers, neural ODEs, stiff systems, and adjoint-based gradient computation.

### [JAX Physics Applications](../jax-physics-applications/SKILL.md)
GPU-accelerated physics: MD simulations, field theory, and lattice models using JAX.

## Routing Decision Tree

```
What is the JAX task?
|
+-- General JAX patterns / transformations?
|   --> jax-mastery
|
+-- Low-level XLA / custom primitives / memory?
|   --> jax-core-programming
|
+-- Gradient optimization / NLSQ / schedules?
|   --> jax-optimization-pro
|
+-- JAX-specific NumPyro / GPU sampling patterns?
|   --> jax-bayesian-pro
|   (for general Bayesian modeling, see bayesian-inference hub)
|
+-- Differential equations / neural ODEs?
|   --> jax-diffeq-pro
|
+-- Physics simulations / MD / field theory?
    --> jax-physics-applications
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| vmap/pmap, JIT patterns | `jax-mastery` |
| Custom VJP, pytrees, XLA | `jax-core-programming` |
| optax, NLSQ, schedules | `jax-optimization-pro` |
| NumPyro, NUTS, ArviZ | `jax-bayesian-pro` |
| Diffrax, neural ODEs | `jax-diffeq-pro` |
| MD, lattice, field theory | `jax-physics-applications` |

## Checklist

- [ ] Identify computation type using the routing tree before selecting a sub-skill
- [ ] Confirm JIT-compatibility: no Python-side effects inside `jit`-traced functions
- [ ] Verify array shapes and dtypes at I/O boundaries before tracing
- [ ] Use `jax.debug.print` for in-JIT diagnostics, not Python `print`
- [ ] Minimize host↔device transfers; batch operations on-device
- [ ] Seed all stochastic operations with explicit `jax.random.PRNGKey`
- [ ] Profile with `jax.profiler` before declaring a bottleneck
- [ ] Validate gradients with `jax.test_util.check_grads` on new custom ops
