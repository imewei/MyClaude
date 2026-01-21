# Advanced Optimization & Performance Engineering

This reference covers what separates a "JAX user" from an "optimization engineer"—squeezing maximum utility out of GPU/TPU hardware.

## XLA & HLO Analysis

The ability to look at the HLO (High Level Optimizer) intermediate representation to understand *why* a function is slow.

### Viewing HLO Output

```python
import jax

def my_fn(x, y):
    return jnp.dot(x, y) + jnp.sin(x)

# Get the HLO text representation
hlo = jax.xla_computation(my_fn)(jnp.ones((100, 100)), jnp.ones((100, 100)))
print(hlo.as_hlo_text())
```

### Common HLO Issues

| Issue | HLO Symptom | Fix |
|-------|-------------|-----|
| **Fusion break** | Separate HLO modules where one expected | Simplify operations, check data dependencies |
| **Excessive copies** | Multiple `copy` instructions | Use in-place updates with `.at[].set()` |
| **Dynamic shapes** | `dynamic-slice` or `dynamic-update-slice` | Use static shapes with padding |
| **Broadcast overhead** | Large `broadcast` operations | Pre-broadcast or restructure computation |

### Profiling with JAX

```python
# Enable compilation logging
import os
os.environ["JAX_LOG_COMPILES"] = "1"

# Use JAX's built-in profiler
with jax.profiler.trace("/tmp/jax-trace"):
    result = my_fn(x)

# View with Perfetto: https://ui.perfetto.dev
```

## Memory Hierarchy Management

Understanding how to minimize device-to-host transfer. JAX execution is asynchronous—keep GPU/TPU saturated.

### Device Placement

```python
# Explicit device placement
x_gpu = jax.device_put(x, jax.devices('gpu')[0])

# Check where data lives
print(x_gpu.devices())

# Block until computation completes (avoid unless necessary)
result.block_until_ready()
```

### Memory-Efficient Patterns

```python
# Gradient checkpointing for large models
from jax.checkpoint import checkpoint

@jax.jit
def memory_efficient_forward(params, x):
    # Recompute activations during backward pass
    x = checkpoint(layer1)(params['layer1'], x)
    x = checkpoint(layer2)(params['layer2'], x)
    return x

# Accumulate gradients without storing all intermediates
def accumulate_gradients(params, batches):
    def scan_fn(acc_grads, batch):
        grads = jax.grad(loss_fn)(params, batch)
        return jax.tree.map(jnp.add, acc_grads, grads), None

    init_grads = jax.tree.map(jnp.zeros_like, params)
    total_grads, _ = jax.lax.scan(scan_fn, init_grads, batches)
    return jax.tree.map(lambda g: g / len(batches), total_grads)
```

### Avoiding Host-Device Transfers

```python
# Anti-pattern: forces synchronization
for i in range(1000):
    x = fn(x)
    if x[0] > threshold:  # Transfers to host!
        break

# JAX-first: use lax.while_loop
def cond_fn(state):
    x, i = state
    return (x[0] <= threshold) & (i < 1000)

def body_fn(state):
    x, i = state
    return fn(x), i + 1

final_x, final_i = jax.lax.while_loop(cond_fn, body_fn, (x, 0))
```

## SPMD Parallelism

Moving beyond simple data parallelism to model parallelism.

### jax.pmap: Replicated Data Parallelism

```python
# Simple data parallelism across all devices
@jax.pmap
def parallel_step(params, batch):
    grads = jax.grad(loss_fn)(params, batch)
    return grads

# Replicate params across devices
replicated_params = jax.device_put_replicated(params, jax.devices())

# Shard data across devices
sharded_batches = batch.reshape(num_devices, -1, *batch.shape[1:])

grads = parallel_step(replicated_params, sharded_batches)

# Aggregate gradients
grads = jax.tree.map(lambda x: x.mean(axis=0), grads)
```

### jax.sharding: Tensor Parallelism

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Create device mesh
devices = mesh_utils.create_device_mesh((4, 2))  # 4 data, 2 model parallel
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define sharding specs
data_sharding = NamedSharding(mesh, P('data', None))
model_sharding = NamedSharding(mesh, P(None, 'model'))

# Shard tensors
x_sharded = jax.device_put(x, data_sharding)
w_sharded = jax.device_put(w, model_sharding)

# Computation automatically distributed
@jax.jit
def sharded_matmul(x, w):
    return x @ w

result = sharded_matmul(x_sharded, w_sharded)
```

### shard_map: Explicit SPMD Control

```python
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh,
         in_specs=(P('data', None), P(None, 'model')),
         out_specs=P('data', 'model'))
def explicit_parallel(x, w):
    # Code runs on each shard independently
    local_result = x @ w
    # Use collective operations for cross-device communication
    return local_result
```

### Choosing Parallelism Strategy

| Strategy | Use Case | Complexity |
|----------|----------|------------|
| `jax.pmap` | Simple data parallelism, identical ops on each device | Low |
| `jax.sharding` | Tensor parallelism, compiler auto-partitions | Medium |
| `shard_map` | Custom communication patterns, maximum control | High |
| `jax.lax.with_sharding_constraint` | Hints for compiler within jit | Medium |

## Pallas: Custom Kernels

For extreme optimization cases, bypass XLA limitations with Pallas (JAX's kernel language).

### When to Use Pallas

- XLA fusion is suboptimal for your specific operation
- Need custom memory access patterns
- Implementing novel algorithms not well-supported by XLA
- Maximum performance on specific hardware

### Basic Pallas Kernel

```python
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as plgpu

def add_kernel(x_ref, y_ref, o_ref):
    # x_ref, y_ref, o_ref are references to memory
    row_idx = pl.program_id(axis=0)

    # Load from memory
    x = x_ref[row_idx, :]
    y = y_ref[row_idx, :]

    # Compute
    o_ref[row_idx, :] = x + y

@jax.jit
def pallas_add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0],),  # One program per row
    )(x, y)
```

### Pallas for Matrix Multiplication

```python
def matmul_kernel(x_ref, y_ref, o_ref, *, block_k):
    row_idx = pl.program_id(axis=0)
    col_idx = pl.program_id(axis=1)

    # Accumulator
    acc = jnp.zeros((block_k,), dtype=jnp.float32)

    def body(i, acc):
        x_block = x_ref[row_idx, i * block_k:(i + 1) * block_k]
        y_block = y_ref[i * block_k:(i + 1) * block_k, col_idx]
        return acc + x_block * y_block

    acc = jax.lax.fori_loop(0, x_ref.shape[1] // block_k, body, acc)
    o_ref[row_idx, col_idx] = acc.sum()
```

## Performance Debugging Workflow

1. **Profile first**: Use `jax.profiler.trace` to identify bottlenecks
2. **Check recompilation**: Enable `JAX_LOG_COMPILES=1`
3. **Inspect HLO**: Look for fusion breaks or unexpected operations
4. **Check data movement**: Ensure data stays on device
5. **Optimize hot paths**: Focus on the slowest 10% of code
6. **Consider Pallas**: For truly custom operations

```bash
# Full debugging environment
export JAX_LOG_COMPILES=1
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
python my_script.py

# Analyze dumps
ls /tmp/xla_dump/*.txt
```
