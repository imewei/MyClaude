---
name: jax-optimization-pro
version: "2.2.1"
description: This skill should be used when the user asks to "optimize JAX code for production", "write JAX-first code", "debug ConcretizationError", "analyze XLA/HLO output", "implement SPMD parallelism", "use jax.sharding for TPU pods", "write Pallas/Triton kernels", "fix tracer errors", "optimize GPU/TPU memory", "handle numerical stability", "implement custom VJPs", or needs expert-level guidance on functional programming patterns, PyTree manipulation, multi-device scaling, or XLA compiler optimization.
---

# JAX Optimization Pro: The JAX-First Engineer

Transform from a JAX user to a JAX-first optimization engineer. Unlike PyTorch or TensorFlow which allow imperative "eager" execution, JAX demands a **functional, compiler-centric mindset**.

## Expert Agent

For advanced optimization, distributed training setup, and performance engineering, delegate to the expert agent:

- **`jax-pro`**: Unified specialist for Core JAX optimization, hardware acceleration, and sharding.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Multi-device parallelism (pmap/sharding), XLA HLO analysis, and memory optimization (remat/checkpointing).

## The JAX-First Mindset

### Functional Purist

Avoid side effects. Global state is the enemy. Every function must be pure—output depends *only* on input.

```python
# Anti-pattern: mutable state
loss_history = []
def train_step(params, x, y):
    loss = compute_loss(params, x, y)
    loss_history.append(loss)  # Side effect - breaks JIT
    return params

# JAX-first: explicit state passing
def train_step(params, x, y, loss_history):
    loss = compute_loss(params, x, y)
    return params, jnp.append(loss_history, loss)
```

### Compiler-Aware

Write code knowing it will be traced and compiled. Predict what is "static" (compile-time) versus "traced" (dynamic).

```python
# Causes ConcretizationError - shape depends on traced value
def bad_dynamic(x):
    return jnp.zeros(x.shape[0])  # x.shape[0] is traced

# JAX-first: use static_argnums
@jax.jit(static_argnums=(1,))
def good_static(x, n):
    return jnp.zeros(n)
```

### Shape-Obsessed

XLA optimization relies on static array shapes. Refactor code to keep shapes constant using padding or masking instead of dynamic resizing.

```python
# Anti-pattern: variable-length sequences
def process_variable(sequences):
    return [process(seq) for seq in sequences]  # Different shapes

# JAX-first: pad to fixed shape, use masks
def process_fixed(padded_batch, mask):
    results = jax.vmap(process)(padded_batch)
    return jnp.where(mask[:, None], results, 0.0)
```

## Core Proficiencies: The Trifecta

Master the composable transformations—not just how to use them, but how they interact when nested.

### jax.jit: XLA Compilation

- Manage cache misses with `static_argnums` for values that affect control flow
- Avoid recompilation by keeping shapes static
- Use `jax.make_jaxpr(fn)(args)` to inspect traced computation

### jax.grad & jax.value_and_grad

- Handle higher-order derivatives (Hessians) with nested `jax.grad`
- Use `jax.lax.stop_gradient` to prevent gradients through specific paths
- Implement custom VJPs with `jax.custom_vjp` for non-differentiable operations

### jax.vmap: Automatic Vectorization

Write functions for single data points, vectorize over batches. Replace manual batch dimensions entirely.

```python
# Single sample function
def single_loss(param, x, y):
    pred = param @ x
    return (pred - y) ** 2

# Vectorized over batch
batch_loss = jax.vmap(single_loss, in_axes=(None, 0, 0))
```

## PyTree Manipulation

Data in JAX is rarely a simple tensor—it's a PyTree (nested dicts/lists/tuples). Master `jax.tree_util`:

```python
# Map over nested structure
grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

# Flatten/unflatten for custom operations
leaves, treedef = jax.tree.flatten(params)
new_leaves = [leaf * 0.99 for leaf in leaves]
params = jax.tree.unflatten(treedef, new_leaves)
```

## Skills Matrix: Junior vs Expert

| Category | Junior | Expert/Pro |
|----------|--------|------------|
| **State** | Uses globals or mutable lists | Explicitly passed immutable PyTrees |
| **Parallelism** | Single GPU | `jax.sharding` for multi-host TPU pods |
| **Debugging** | Uses `print()` (fails in JIT) | `jax.debug.print` and `jax.disable_jit()` |
| **Loops** | Python `for` loops (slow unroll) | `jax.lax.scan` for compiled loops |
| **Performance** | "It feels slow" | "HLO shows fusion break at line 40" |

## Ecosystem Fluency

| Domain | Tool | Purpose |
|--------|------|---------|
| **Optimization** | Optax | Composable gradient transformations |
| **Neural Networks** | Equinox / Flax | Stateful modules in functional style |
| **Profiling** | Perfetto / JAX profiler | Visualize trace events, identify bottlenecks |

## Debugging Checklist

```python
# 1. Disable JIT to get Python errors
with jax.disable_jit():
    result = problematic_fn(x)

# 2. Enable NaN detection
jax.config.update("jax_debug_nans", True)

# 3. Print inside JIT
jax.debug.print("x = {x}", x=x)

# 4. Inspect traced computation
print(jax.make_jaxpr(fn)(args))

# 5. Check for recompilation
# Enable logging: JAX_LOG_COMPILES=1
```

## Quick Patterns

### Compiled Loops with lax.scan

```python
def step(carry, x):
    state = carry + x
    return state, state  # (new_carry, output)

final_state, all_states = jax.lax.scan(step, init_state, xs)
```

### Conditional without Python if

```python
result = jax.lax.cond(
    condition,
    true_fn,   # Called if condition is True
    false_fn,  # Called if condition is False
    operand
)
```

## Additional Resources

### Reference Files

For optimization engineering beyond basics, consult:

- **`references/advanced-optimization.md`** - XLA/HLO analysis, memory hierarchy management, SPMD parallelism (sharding, pmap, shard_map), Pallas custom kernels
- **`references/scientific-numerical.md`** - Numerical stability (float32 vs bfloat16, NaN handling), custom primitives, advanced control flow (lax.scan, lax.cond, lax.while_loop)

### Example Files

Working examples demonstrating expert patterns:

- **`examples/pure-functional-patterns.py`** - State management, PyTree manipulation, explicit RNG
- **`examples/xla-optimization.py`** - HLO analysis, fusion optimization, avoiding recompilation
- **`examples/spmd-parallelism.py`** - Sharding, mesh configuration, multi-device execution

**Outcome**: Write pure functions, explicit RNG, JIT hot paths, multi-device scaling, HLO-aware optimization.
