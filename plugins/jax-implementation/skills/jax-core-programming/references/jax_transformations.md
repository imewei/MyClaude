# JAX Transformations - Comprehensive Reference

## Overview

JAX's core power comes from composable function transformations that enable high-performance computing. This guide covers jit, vmap, pmap, grad, and their composition patterns.

---

## 1. JIT Compilation (`jax.jit`)

### Purpose
Just-In-Time compilation with XLA for 10-100x speedup by compiling pure functions to optimized machine code.

### Basic Usage

```python
import jax
import jax.numpy as jnp

# Simple function
def slow_fn(x):
    return jnp.sum(x ** 2) + jnp.sin(x).mean()

# Compiled version
fast_fn = jax.jit(slow_fn)

# Or decorator style
@jax.jit
def fast_fn_decorated(x):
    return jnp.sum(x ** 2) + jnp.sin(x).mean()

# First call compiles (slow), subsequent calls are fast
x = jnp.ones(1000)
result = fast_fn(x)  # Compilation + execution
result = fast_fn(x)  # Fast execution only
```

### Static Arguments

When functions have arguments that change shape/dtype, mark them as static to avoid recompilation:

```python
@jax.jit
def bad_dynamic_fn(x, n):
    # Recompiles every time n changes
    return jnp.concatenate([x] * n)

@jax.partial(jax.jit, static_argnums=(1,))
def good_static_fn(x, n):
    # Only compiles once per unique n value
    return jnp.concatenate([x] * n)

# Alternative: static_argnames for keyword args
@jax.partial(jax.jit, static_argnames=['n'])
def good_static_fn_kwargs(x, n=3):
    return jnp.concatenate([x] * n)
```

### Control Flow in JIT

Python control flow doesn't work with tracers. Use `jax.lax` alternatives:

```python
# BAD: Python if statement
@jax.jit
def buggy_conditional(x):
    if x.sum() > 0:  # ERROR: Can't use tracer as boolean
        return x * 2
    return x

# GOOD: jax.lax.cond
@jax.jit
def fixed_conditional(x):
    return jax.lax.cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x,
        x
    )

# GOOD: jax.lax.switch for multiple branches
@jax.jit
def multi_branch(index, x):
    branches = [
        lambda x: x * 1,
        lambda x: x * 2,
        lambda x: x * 3,
    ]
    return jax.lax.switch(index, branches, x)

# GOOD: jax.lax.select for element-wise conditions
@jax.jit
def element_wise_condition(pred, x, y):
    return jax.lax.select(pred, x, y)
```

### Debugging JIT Code

```python
# Method 1: Disable JIT temporarily
with jax.disable_jit():
    result = fast_fn(x)  # Runs in Python mode

# Method 2: Use jax.debug.print (works inside JIT)
@jax.jit
def debug_fn(x):
    jax.debug.print("x shape: {}, mean: {}", x.shape, x.mean())
    return x ** 2

# Method 3: Inspect traced code
print(jax.make_jaxpr(slow_fn)(x))
```

---

## 2. Vectorization (`jax.vmap`)

### Purpose
Automatically vectorize functions over batch dimensions for efficient SIMD parallelism.

### Basic Usage

```python
# Original function for single example
def single_loss(params, x, y):
    pred = params['w'] @ x + params['b']
    return jnp.mean((pred - y) ** 2)

# Manual batching (slow)
def manual_batch_loss(params, batch_x, batch_y):
    losses = []
    for x, y in zip(batch_x, batch_y):
        losses.append(single_loss(params, x, y))
    return jnp.mean(jnp.array(losses))

# Automatic vectorization (fast)
batch_loss = jax.vmap(single_loss, in_axes=(None, 0, 0))
# in_axes: (None, 0, 0) means params is shared, x and y are batched along axis 0

# Usage
params = {'w': jnp.ones((10, 5)), 'b': jnp.zeros(10)}
batch_x = jnp.ones((32, 5))  # 32 examples
batch_y = jnp.ones((32, 10))
loss = batch_loss(params, batch_x, batch_y)
```

### Advanced `in_axes` and `out_axes`

```python
# Different batch dimensions
def matmul_fn(A, B):
    return A @ B

# Batch over first dim of A, second dim of B
batched_matmul = jax.vmap(matmul_fn, in_axes=(0, 1))
A = jnp.ones((10, 5, 3))  # Batch of 10 matrices
B = jnp.ones((3, 4, 10))  # Batch of 10 matrices (transposed)
C = batched_matmul(A, B)  # Shape: (10, 5, 4)

# Control output batch dimension
@jax.vmap(out_axes=1)  # Batch dimension becomes axis 1
def fn(x):
    return x ** 2
```

### Nested vmap

```python
# Vectorize over two dimensions
def pairwise_distance(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))

# First vmap over y, then over x
pairwise_matrix = jax.vmap(
    jax.vmap(pairwise_distance, in_axes=(None, 0)),
    in_axes=(0, None)
)

X = jnp.ones((100, 5))  # 100 points
Y = jnp.ones((200, 5))  # 200 points
distances = pairwise_matrix(X, Y)  # Shape: (100, 200)
```

---

## 3. Multi-Device Parallelism (`jax.pmap`)

### Purpose
Distribute computation across multiple devices (GPUs/TPUs) with explicit device placement.

**Note**: `pmap` is being replaced by the Sharding API (see section 6). Use Sharding API for new code.

### Basic Usage

```python
# Check available devices
devices = jax.devices()
print(f"Available devices: {len(devices)}")

# Function to parallelize
def device_computation(x):
    return x ** 2 + jnp.sin(x)

# Parallelize across devices
parallel_fn = jax.pmap(device_computation)

# Prepare data (first dimension = number of devices)
n_devices = len(devices)
x = jnp.ones((n_devices, 1000))  # Shard along first dimension
result = parallel_fn(x)  # Each device processes one shard
```

### Collective Operations

```python
@jax.pmap
def distributed_mean(local_data):
    # Each device has local_data
    # Sum across all devices
    global_sum = jax.lax.psum(local_data, axis_name='batch')
    return global_sum / jax.device_count()

# All-gather: collect data from all devices
@jax.pmap
def gather_fn(local_data):
    return jax.lax.all_gather(local_data, axis_name='batch')
```

---

## 4. Automatic Differentiation (`jax.grad`)

### Purpose
Compute derivatives of functions for optimization and gradient-based methods.

### Basic Usage

```python
# Forward mode
def f(x):
    return x ** 3 + 2 * x ** 2 + x

# Gradient
grad_f = jax.grad(f)
print(grad_f(3.0))  # 3*9 + 4*3 + 1 = 40

# Value and gradient together
value_and_grad_f = jax.value_and_grad(f)
val, grad = value_and_grad_f(3.0)

# Gradient with respect to multiple arguments
def multi_arg_fn(x, y):
    return x ** 2 + y ** 2

# Gradient w.r.t. all args (returns tuple)
grad_multi = jax.grad(multi_arg_fn, argnums=(0, 1))
dx, dy = grad_multi(3.0, 4.0)  # (6.0, 8.0)

# Gradient w.r.t. specific arg
grad_x = jax.grad(multi_arg_fn, argnums=0)  # Only w.r.t. x
```

### Gradients with PyTrees

```python
# Works seamlessly with nested structures
def loss_fn(params, x, y):
    pred = params['w'] @ x + params['b']
    return jnp.mean((pred - y) ** 2)

# Gradient returns same structure as params
grad_fn = jax.grad(loss_fn)
params = {'w': jnp.ones((10, 5)), 'b': jnp.zeros(10)}
grads = grad_fn(params, x, y)  # grads is {'w': ..., 'b': ...}
```

### Higher-Order Derivatives

```python
# Second derivative (Hessian diagonal)
hessian_diag = jax.grad(jax.grad(f))

# Full Hessian
hessian_matrix = jax.hessian(f)

# Jacobian
def vector_fn(x):
    return jnp.array([x[0]**2, x[1]*x[0], x[1]**2])

jacobian = jax.jacobian(vector_fn)
J = jacobian(jnp.array([2.0, 3.0]))
```

### Custom Gradients

```python
@jax.custom_vjp
def stable_log1p(x):
    return jnp.log(1.0 + x)

def stable_log1p_fwd(x):
    return stable_log1p(x), x

def stable_log1p_bwd(x, g):
    # Custom gradient for numerical stability
    return (g / (1.0 + x),)

stable_log1p.defvjp(stable_log1p_fwd, stable_log1p_bwd)
```

---

## 5. Composition Patterns

### Common Compositions

```python
# Pattern 1: JIT + vmap (most common)
@jax.jit
@jax.vmap
def fast_batched_fn(x):
    return x ** 2

# Pattern 2: JIT + vmap + grad
loss_fn = lambda params, x, y: jnp.mean((params @ x - y) ** 2)
fast_grad = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0)))

# Pattern 3: vmap + grad (per-example gradients)
per_example_grad = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))

# Pattern 4: grad + jit (compile gradient computation)
compiled_grad = jax.jit(jax.grad(loss_fn))
```

### Full Training Step Composition

```python
def training_step_composition():
    """Complete example showing all transformations"""

    # 1. Define loss for single example
    def single_loss(params, x, y):
        pred = params['w'] @ x + params['b']
        return jnp.mean((pred - y) ** 2)

    # 2. Vectorize over batch
    batch_loss = jax.vmap(single_loss, in_axes=(None, 0, 0))

    # 3. Average loss
    def avg_loss(params, batch_x, batch_y):
        return jnp.mean(batch_loss(params, batch_x, batch_y))

    # 4. Compute gradient
    grad_fn = jax.grad(avg_loss)

    # 5. Compile everything
    fast_grad_fn = jax.jit(grad_fn)

    # 6. Training step with optimizer
    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        grads = grad_fn(params, batch_x, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return train_step
```

---

## 6. Modern Sharding API (2025)

### Purpose
Replace `pmap` with more flexible device placement using `Mesh` and `NamedSharding`.

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Create device mesh
devices = mesh_utils.create_device_mesh((4, 2))  # 4 data, 2 model parallel
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define sharding strategy
sharding = NamedSharding(mesh, P('data', 'model'))

# Shard arrays
x = jnp.ones((1024, 512))
x_sharded = jax.device_put(x, NamedSharding(mesh, P('data', None)))

# Automatic sharding propagation in jit
@jax.jit
def sharded_computation(x, w):
    return x @ w  # JAX handles communication

# Usage
result = sharded_computation(x_sharded, w_sharded)
```

---

## 7. Performance Best Practices

### Order of Transformations

```python
# Efficient: jit(vmap(fn))
fast_batched = jax.jit(jax.vmap(fn))  # Compile vectorized code

# Less efficient: vmap(jit(fn))
slow_batched = jax.vmap(jax.jit(fn))  # Compile each example separately
```

### Avoid Recompilation

```python
# BAD: Dynamic shapes cause recompilation
@jax.jit
def bad_fn(x):
    return jnp.concatenate([x, jnp.zeros(x.shape[0])])  # Shape depends on x

# GOOD: Static shapes
@jax.jit
def good_fn(x):
    return jnp.concatenate([x, jnp.zeros(128)])  # Fixed shape
```

### Memory Efficiency

```python
# Use rematerialization for large models
from jax.ad_checkpoint import checkpoint as remat

@remat
def memory_heavy_fn(x):
    x = large_layer1(x)
    x = large_layer2(x)
    return x

# Reduces memory by 2-5x at cost of ~30% compute
```

---

## Common Pitfalls

| Issue | Bad Pattern | Good Pattern |
|-------|------------|--------------|
| Python control flow | `if x.sum() > 0` | `jax.lax.cond` |
| Global state | `global rng` | Explicit RNG passing |
| In-place mutation | `x[0] = 1` | `x.at[0].set(1)` |
| Dynamic shapes | Varying input sizes | Padding or static args |
| Recompilation | Non-static config | `static_argnums` |

---

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
