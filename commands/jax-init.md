---
description: Initialize a JAX project with essential imports, PRNG setup, and Principle 3 reminders
category: jax-core
argument-hint: ""
allowed-tools: "*"
---

# /jax-init

Initialize a JAX project with essential imports and setup.

## Description

Sets up a JAX project with core imports (jax, jax.numpy, jax.random), initializes a PRNG key, and provides reminders about JAX's functional random number generation principles.

## Usage

```
/jax-init
```

## What it does

1. Import essential JAX modules: `jax`, `jax.numpy as jnp`, `jax.random`
2. Set up a PRNG key using `jax.random.PRNGKey(0)`
3. Remind about Principle 3: JAX RNG requires explicit key management and splitting for reproducible randomness

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random

# Initialize PRNG key
key = random.PRNGKey(0)

# JAX Principle 3 Reminder:
# JAX random number generation requires explicit key management.
# Always split keys when using random functions:
# key, subkey = random.split(key)
# Use subkey for random operations to maintain reproducibility.
```

## Related Commands

- `/jax-debug` - Debug JAX code issues
- `/jax-jit` - Apply JIT compilation to functions