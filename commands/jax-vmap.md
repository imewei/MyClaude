---
description: Vectorize functions over batches with axis specification and shape broadcasting
category: jax-core
argument-hint: "[function_name] [--in-axes] [--out-axes]"
allowed-tools: "*"
---

# /jax-vmap

Vectorize functions over batches using JAX's `vmap` transformation.

## Description

Applies `jax.vmap` to automatically vectorize functions across batch dimensions. Handles axis specification and shape broadcasting for efficient batch processing.

## Usage

```
/jax-vmap [function_name] [--in-axes] [--out-axes]
```

## What it does

1. Apply `jax.vmap` to vectorize functions over batches
2. Specify input and output axis mappings
3. Handle shape broadcasting and dimension management
4. Optimize batch processing performance

## Example output

```python
# Basic vectorization over first axis
def single_example_fn(x, y):
    return jnp.dot(x, y) + jnp.sum(x)

# Vectorize over batches (first dimension)
batch_fn = jax.vmap(single_example_fn)
# Input shapes: x (batch, features), y (batch, features)
# Output shape: (batch,)

# Specify axis for vectorization
batch_fn = jax.vmap(single_example_fn, in_axes=(0, 0), out_axes=0)

# Mixed vectorization - some inputs vectorized, others broadcast
batch_fn = jax.vmap(single_example_fn, in_axes=(0, None))
# x is vectorized over batch, y is broadcast to all examples

# Multiple batch dimensions
nested_batch_fn = jax.vmap(jax.vmap(single_example_fn))
# Input shapes: x (batch1, batch2, features), y (batch1, batch2, features)

# Per-example gradients using vmap + grad
per_example_grad = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
# Compute gradient for each example in the batch

# Advanced axis specification
complex_vmap = jax.vmap(func, in_axes={'x': 0, 'y': 1}, out_axes=2)

# vmap benefits:
# - Automatic vectorization without manual loops
# - Maintains functional programming style
# - Efficient parallel execution
# - Composes well with jit and grad
```

## Related Commands

- `/jax-jit` - JIT compile vmapped functions
- `/jax-grad` - Combine with vmap for per-example gradients
- `/jax-pmap` - Parallelize across multiple devices