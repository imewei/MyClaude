---
description: Apply JIT compilation to JAX functions with benefits explanation and pitfall warnings
category: jax-core
argument-hint: "[function_name]"
allowed-tools: "*"
---

# /jax-jit

Apply JIT (Just-In-Time) compilation to JAX functions for performance optimization.

## Description

Applies `jax.jit` to functions to compile them for faster execution. Explains compilation benefits, static shape requirements, and common pitfalls to avoid.

## Usage

```
/jax-jit [function_name]
```

## What it does

1. Wraps the specified function with `jax.jit` decorator
2. Explains JIT compilation benefits (speed improvements, XLA optimization)
3. Highlights static shape requirements and common pitfalls
4. Provides debugging tips for JIT-related issues

## Example output

```python
# Apply JIT compilation to your function
@jax.jit
def your_function(x, y):
    return jnp.dot(x, y) + jnp.sum(x)

# Alternative functional approach
your_function_jit = jax.jit(your_function)

# JIT Compilation Benefits:
# - XLA optimization for faster execution
# - Automatic GPU/TPU acceleration when available
# - Function fusion and memory optimization

# Important JIT Requirements:
# - Input shapes must be static (known at compile time)
# - Array shapes cannot change between calls
# - Use jax.jit(static_argnums=(...)) for static arguments

# Common Pitfalls:
# - Dynamic shapes will cause recompilation
# - Python control flow may not work as expected
# - Use jax.lax.cond/while_loop for control flow inside JIT
```

## Related Commands

- `/jax-debug` - Debug JIT compilation issues
- `/jax-vmap` - Vectorize functions (works well with JIT)
- `/jax-grad` - Compute gradients (combines with JIT)