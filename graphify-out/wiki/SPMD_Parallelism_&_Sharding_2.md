# SPMD Parallelism & Sharding

> 27 nodes · cohesion 0.11

## Key Concepts

- **Debug Jax** (11 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **debug_jax.py** (11 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **check_nan_inf** (4 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **check_nan_inf()** (4 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **debug_jit** (3 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **monitor_training_step** (3 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **debug_jit()** (3 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **monitor_training_step()** (3 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **Enable automatic NaN checking in JAX.** (3 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **check_device_placement** (2 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **check_gradient_flow** (2 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **disable_nan_checking** (2 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **enable_nan_checking** (2 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **print_pytree_structure** (2 connections) — `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **check_device_placement()** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **check_gradient_flow()** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **disable_nan_checking()** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **enable_nan_checking()** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **good_fn()** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **print_pytree_structure()** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **Check for NaN or Inf values in JAX arrays or pytrees.      Args:         x: JAX** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **Check gradient flow through parameters.      Args:         loss_fn: Loss functio** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **Monitor a training step for common issues.      Args:         train_step_fn: Tra** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **Check which devices data is placed on.** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- **Print structure and shapes of a pytree.      Args:         pytree: JAX pytree to** (2 connections) — `science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- *... and 2 more nodes in this community*

## Relationships

- [[JAX Performance Profiling]] (2 shared connections)
- [[XLA Optimization Patterns]] (2 shared connections)

## Source Files

- `plugins/science-suite/skills/jax-core-programming/scripts/debug_jax.py`
- `science-suite/skills/jax-core-programming/scripts/debug_jax.py`

## Audit Trail

- EXTRACTED: 80 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*