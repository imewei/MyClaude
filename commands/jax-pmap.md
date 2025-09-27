---
description: Parallelize across multiple devices with mesh setup and sharding strategies
category: jax-core
argument-hint: "[function_name] [--axis-name] [--devices]"
allowed-tools: "*"
---

# /jax-pmap

Parallelize computation across multiple devices using JAX's `pmap`.

## Description

Sets up parallel execution across multiple devices (GPUs/TPUs) using `jax.pmap`. Handles device mesh setup, sharding strategies, and cross-device communication.

## Usage

```
/jax-pmap [function_name] [--axis-name] [--devices]
```

## What it does

1. Apply `jax.pmap` for multi-device parallelization
2. Set up device mesh and sharding configuration
3. Handle collective operations (allreduce, allgather)
4. Manage data distribution across devices

## Example output

```python
# Basic parallel map across devices
def compute_fn(x):
    return jnp.sum(x ** 2)

# Parallelize across available devices
parallel_fn = jax.pmap(compute_fn)

# Check available devices
devices = jax.devices()
print(f"Available devices: {len(devices)} - {devices}")

# Prepare data for parallel execution (shard across devices)
batch_size = len(devices)
data = jnp.arange(batch_size * 10).reshape(batch_size, 10)
result = parallel_fn(data)

# Parallel computation with axis name for collectives
@jax.pmap(axis_name='batch')
def parallel_mean(x):
    local_mean = jnp.mean(x)
    # Average across all devices
    global_mean = jax.lax.pmean(local_mean, axis_name='batch')
    return global_mean

# Parallel training step with gradient synchronization
@jax.pmap(axis_name='batch')
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    # Synchronize gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    return grads, loss

# Advanced: specify device assignment
devices = jax.devices()[:4]  # Use first 4 devices
parallel_fn = jax.pmap(compute_fn, devices=devices)

# Data sharding utilities
def shard_data(data, num_devices):
    return data.reshape(num_devices, -1, *data.shape[1:])

# pmap considerations:
# - Input first dimension must match number of devices
# - Use pmean/psum for cross-device reductions
# - Careful with device memory management
# - Combine with vmap for nested parallelism
```

## Related Commands

- `/jax-vmap` - Vectorize before parallelizing
- `/jax-jit` - JIT compile pmapped functions
- `/jax-ml-train` - Parallel training workflows