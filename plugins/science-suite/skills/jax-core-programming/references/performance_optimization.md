# JAX Performance Optimization Guide

## Overview

Comprehensive guide to optimizing JAX code for maximum performance, memory efficiency, and scalability.

---

## 1. Compilation (JIT)

### When to JIT

```python
import jax
import jax.numpy as jnp

# JIT expensive computations (>1ms)
@jax.jit
def expensive_fn(x):
    for _ in range(100):
        x = jnp.sin(x) + jnp.cos(x)
    return x

# Don't JIT trivial operations
def trivial_fn(x):
    return x + 1  # Too simple for JIT overhead
```

### Avoiding Recompilation

```python
# BAD: Dynamic shapes cause recompilation
@jax.jit
def bad_fn(x, n):
    return jnp.concatenate([x] * n)  # Recompiles for each n

# GOOD: Static arguments
@jax.partial(jax.jit, static_argnums=(1,))
def good_fn(x, n):
    return jnp.concatenate([x] * n)  # Compiles once per n

# GOOD: Fixed shapes with padding
@jax.jit
def padded_fn(x, n, max_n=10):
    padded = jnp.concatenate([x] * max_n)
    return padded[:n * x.shape[0]]  # Dynamic slicing is fine
```

### Compilation Cache

```python
# Enable persistent compilation cache
import jax
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)

# Functions will be cached and reused across runs
```

---

## 2. Vectorization (vmap)

### Efficient Batching

```python
# BAD: Python loop
def slow_batch_fn(x_batch):
    return jnp.stack([expensive_fn(x) for x in x_batch])

# GOOD: vmap
fast_batch_fn = jax.vmap(expensive_fn)

# BEST: jit + vmap
fastest_batch_fn = jax.jit(jax.vmap(expensive_fn))
```

### Optimal in_axes

```python
# Minimize broadcasting
@jax.vmap
def efficient_fn(x, shared_param):  # x batched, shared_param not
    return x * shared_param

# Usage
x_batch = jnp.ones((1000, 10))  # Batched
param = jnp.ones(10)  # Shared across batch
result = efficient_fn(x_batch, param)  # Efficient
```

---

## 3. Memory Optimization

### Rematerialization (Gradient Checkpointing)

```python
from jax.ad_checkpoint import checkpoint as remat

# Without remat: High memory, fast
def memory_heavy_model(x):
    x = large_layer1(x)  # Stored for backward
    x = large_layer2(x)  # Stored for backward
    x = large_layer3(x)  # Stored for backward
    return x

# With remat: 2-5x less memory, ~30% slower
@remat
def memory_efficient_model(x):
    x = large_layer1(x)  # Recomputed in backward
    x = large_layer2(x)  # Recomputed in backward
    x = large_layer3(x)  # Recomputed in backward
    return x

# Selective remat (recompute every N layers)
def selective_remat_model(x):
    x = large_layer1(x)
    x = large_layer2(x)
    x = remat(lambda x: large_layer3(x))(x)  # Only this layer
    return x
```

### Buffer Donation

```python
# Reuse buffers to reduce memory
@jax.jit
def update_fn(x, y):
    # x buffer can be reused since we don't use it after
    return x + y

# Enable donation
from jax._src import api
update_fn = jax.jit(update_fn, donate_argnums=(0,))  # Donate x

# Now x buffer is reused, saving memory
```

### Memory Profiling

```python
# Profile memory usage
import jax.profiler

# Start profiler
jax.profiler.start_trace('/tmp/jax_profile')

# Run code
result = expensive_fn(x)

# Stop profiler
jax.profiler.stop_trace()

# View in TensorBoard: tensorboard --logdir=/tmp/jax_profile
```

---

## 4. Mixed Precision Training

### BF16 for 2-3x Speedup

```python
from jax import lax

# Policy: use bf16 for compute, fp32 for params
def mixed_precision_matmul(x, w):
    # Cast inputs to bf16
    x_bf16 = x.astype(jnp.bfloat16)
    w_bf16 = w.astype(jnp.bfloat16)

    # Compute in bf16 (faster on TPU/GPU)
    result = x_bf16 @ w_bf16

    # Cast back to fp32 for accumulation
    return result.astype(jnp.float32)

# Automatic mixed precision
@jax.jit
def train_step(params, batch):
    # Downcast for forward/backward
    def loss_fn(params):
        params_bf16 = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
        logits = model(params_bf16, batch['x'])
        return loss(logits, batch['y'])

    # Gradients computed in bf16, then upcast
    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree_map(lambda x: x.astype(jnp.float32), grads)
    return loss, grads
```

---

## 5. Multi-Device Parallelism

### Modern Sharding API

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Create 2D device mesh (data x model parallelism)
devices = mesh_utils.create_device_mesh((4, 2))  # 8 GPUs total
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define sharding strategies
data_sharding = NamedSharding(mesh, P('data', None))
model_sharding = NamedSharding(mesh, P(None, 'model'))

# Shard arrays
x = jnp.ones((1024, 512))
x_sharded = jax.device_put(x, data_sharding)

w = jnp.ones((512, 256))
w_sharded = jax.device_put(w, model_sharding)

# JIT automatically handles communication
@jax.jit
def sharded_matmul(x, w):
    return x @ w  # JAX inserts all-gather/reduce-scatter

result = sharded_matmul(x_sharded, w_sharded)
```

### Pipeline Parallelism

```python
# For large models that don't fit on one device
def pipeline_model(x):
    # Layer 1 on device 0
    with jax.default_device(jax.devices()[0]):
        x = layer1(x)

    # Layer 2 on device 1
    with jax.default_device(jax.devices()[1]):
        x = layer2(x)

    # Layer 3 on device 2
    with jax.default_device(jax.devices()[2]):
        x = layer3(x)

    return x
```

---

## 6. Profiling

### JAX Profiler

```python
import jax.profiler

# Method 1: Programmatic profiling
with jax.profiler.trace('/tmp/profile'):
    result = expensive_fn(x)

# Method 2: Profile server (for long-running jobs)
jax.profiler.start_server(9999)  # Access at localhost:9999

# Run training
for step in range(1000):
    loss = train_step(params, batch)

# View profile in TensorBoard
# tensorboard --logdir=/tmp/profile
```

### Identifying Bottlenecks

```python
import time

def profile_fn(fn, x, n_warmup=3, n_runs=10):
    """Profile function execution time"""

    # Warmup (compilation)
    for _ in range(n_warmup):
        _ = fn(x)

    # Block until complete
    jax.block_until_ready(fn(x))

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = fn(x)
        jax.block_until_ready(result)  # Wait for GPU
        times.append(time.time() - start)

    return {
        'mean': jnp.mean(times),
        'std': jnp.std(times),
        'min': jnp.min(times),
        'max': jnp.max(times),
    }

# Usage
stats = profile_fn(expensive_fn, x)
print(f"Mean: {stats['mean']*1000:.2f}ms ± {stats['std']*1000:.2f}ms")
```

---

## 7. Optimization Checklist

```python
"""
JAX Performance Optimization Checklist:

COMPILATION:
✓ JIT expensive functions (>1ms)
✓ Use static_argnums for config/shapes
✓ Enable compilation cache
✓ Profile for recompilation hotspots

VECTORIZATION:
✓ Replace loops with vmap
✓ Order: jit(vmap(fn)) not vmap(jit(fn))
✓ Minimize in_axes complexity
✓ Batch large enough for GPU

MEMORY:
✓ Use remat for large models (2-5x savings)
✓ Enable buffer donation
✓ Profile memory with JAX profiler
✓ Use gradient accumulation for large batches

PRECISION:
✓ Use bf16/fp16 for compute (2-3x faster)
✓ Keep fp32 for parameters
✓ Test numerical stability

MULTI-DEVICE:
✓ Use Sharding API for >1 device
✓ Profile communication overhead
✓ Optimize device mesh topology
✓ Use pipelining for huge models

PROFILING:
✓ Profile before optimizing
✓ Use jax.profiler for detailed traces
✓ Check for recompilation
✓ Monitor memory usage
✓ Validate speedups with benchmarks
"""
```

---

## 8. Performance Patterns

### Training Loop Optimization

```python
@jax.jit
def optimized_train_step(params, opt_state, batch):
    """Fully optimized training step"""

    def loss_fn(params):
        # Mixed precision forward pass
        params_bf16 = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)

        # Vectorized batch processing
        logits = jax.vmap(model, in_axes=(None, 0))(params_bf16, batch['x'])

        # Loss computation
        return jnp.mean((logits - batch['y']) ** 2)

    # Gradient computation with remat
    loss, grads = jax.value_and_grad(remat(loss_fn))(params)

    # Upcast gradients to fp32
    grads = jax.tree_map(lambda x: x.astype(jnp.float32), grads)

    # Optimizer update
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

# This single function is:
# - JIT compiled (10-100x faster)
# - Mixed precision (2-3x faster)
# - Memory optimized (remat)
# - Vectorized (vmap for batch)
```

### Inference Optimization

```python
@jax.jit
def optimized_inference(params, x_batch):
    """Optimized batch inference"""

    # Cast to bf16 for speed
    params_bf16 = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    x_bf16 = x_batch.astype(jnp.bfloat16)

    # Vectorized inference
    predictions = jax.vmap(model, in_axes=(None, 0))(params_bf16, x_bf16)

    # Cast back to fp32
    return predictions.astype(jnp.float32)
```

---

## 9. Hardware-Specific Tips

### GPU Optimization

```python
# Prefer larger batch sizes (better GPU utilization)
BATCH_SIZE = 256  # GPU likes powers of 2

# Use mixed precision (bf16 on A100, fp16 on V100)
# Enable TF32 on A100
import jax
jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
```

### TPU Optimization

```python
# TPUs love bf16 and large batches
BATCH_SIZE = 1024  # TPU v3/v4 excel at large batches

# Use data parallelism across cores
from jax.sharding import Mesh
mesh = Mesh(jax.devices(), axis_names=('data',))

# Prefer 128-byte aligned shapes
def make_tpu_friendly_shape(size):
    return ((size + 127) // 128) * 128
```

---

## 10. Benchmarking

### Micro-benchmarks

```python
def benchmark_ops():
    """Compare different implementations"""
    x = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))

    # Naive
    def naive(x):
        return jnp.sum(x ** 2)

    # Optimized
    @jax.jit
    def optimized(x):
        return jnp.sum(x ** 2)

    print("Naive:", profile_fn(naive, x)['mean'] * 1000, "ms")
    print("JIT:", profile_fn(optimized, x)['mean'] * 1000, "ms")
    print("Speedup:", profile_fn(naive, x)['mean'] / profile_fn(optimized, x)['mean'], "x")
```

---

## References

- [JAX Performance Tips](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)
- [XLA Performance Guide](https://www.tensorflow.org/xla/performance)
