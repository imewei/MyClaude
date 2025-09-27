---
description: Debug JAX code with specialized tools, disable JIT, and check for tracer leaks
category: jax-core
argument-hint: "[--disable-jit] [--check-tracers] [--print-values]"
allowed-tools: "*"
---

# /jax-debug

Debug JAX code with specialized tools and techniques.

## Description

Provides debugging utilities for JAX code including disabling JIT, adding debug prints, checking for tracer leaks, and diagnosing common JAX issues.

## Usage

```
/jax-debug [--disable-jit] [--check-tracers] [--print-values]
```

## What it does

1. Disable JIT compilation for debugging
2. Insert debug prints that work with JAX transformations
3. Check for tracer leaks and abstract value errors
4. Diagnose shape and dtype issues

## Example output

```python
# Disable JIT for debugging
jax.config.update('jax_disable_jit', True)
# Your code here - now runs in Python mode
jax.config.update('jax_disable_jit', False)  # Re-enable

# Debug prints that work with JAX transformations
def debug_fn(x):
    jax.debug.print("x shape: {x_shape}, x dtype: {x_dtype}",
                     x_shape=x.shape, x_dtype=x.dtype)
    jax.debug.print("x value: {x}", x=x)
    return x ** 2

# Use jax.debug.breakpoint() for interactive debugging
def breakpoint_fn(x):
    jax.debug.breakpoint()  # Drops into debugger
    return x + 1

# Check for tracer leaks
try:
    result = jitted_function(data)
except Exception as e:
    print(f"Tracer leak detected: {e}")
    # Common causes:
    # - Using traced values outside transformed functions
    # - Comparing traced values with Python conditionals
    # - Array shape/index operations with traced values

# Shape debugging
def check_shapes(x, y):
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    result = jnp.dot(x, y)
    print(f"Output shape: {result.shape}")
    return result

# Debug array operations
jax.config.update('jax_debug_nans', True)   # Catch NaN values
jax.config.update('jax_debug_infs', True)   # Catch infinity values

# Memory debugging
def check_memory():
    devices = jax.devices()
    for device in devices:
        print(f"Device {device}: {device.memory_stats()}")

# Common JAX debugging patterns:
# 1. Use .block_until_ready() to force computation
# 2. Check device placement with .device()
# 3. Use tree_map for nested structure debugging
# 4. Disable JIT to get normal Python errors
```

## Related Commands

- `/jax-jit` - JIT compilation that may need debugging
- `/jax-init` - Initial setup that may cause issues