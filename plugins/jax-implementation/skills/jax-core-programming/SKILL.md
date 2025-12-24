---
name: jax-core-programming
version: "1.0.5"
maturity: "5-Expert"
specialization: JAX Functional Programming
description: Master JAX functional transformations (jit, vmap, pmap, grad), Flax NNX neural networks, Optax optimizers, Orbax checkpointing, and NumPyro Bayesian inference. Use when writing JAX code, building training loops, optimizing XLA compilation, debugging tracer errors, or scaling to multi-device GPU/TPU.
---

# JAX Core Programming

High-performance numerical computing with automatic differentiation and XLA compilation.

---

## JAX Transformations

| Transform | Purpose | Usage |
|-----------|---------|-------|
| `jax.jit` | XLA compilation (10-100x speedup) | Hot paths |
| `jax.vmap` | Vectorize over batch axis | Batching |
| `jax.pmap` | Parallelize across devices | Multi-GPU/TPU |
| `jax.grad` | Automatic differentiation | Training |

### Core Patterns
```python
# JIT compilation
@jax.jit
def fast_fn(x):
    return jnp.sum(x ** 2)

# Vectorization
batched_loss = jax.vmap(single_loss, in_axes=(None, 0, 0))

# Composition
fast_batched = jax.jit(jax.vmap(fn))

# Gradients
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

# Static arguments (avoid recompilation)
@jax.jit(static_argnums=(1,))
def shaped_fn(x, shape):
    return jnp.zeros(shape)
```

---

## Flax NNX

```python
from flax import nnx

class Transformer(nnx.Module):
    def __init__(self, config, rngs):
        self.embed = nnx.Embed(config.vocab, config.dim, rngs=rngs)
        self.layers = [TransformerBlock(config, rngs) for _ in range(config.n_layers)]
        self.norm = nnx.RMSNorm(config.dim, rngs=rngs)

    def __call__(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# Training step
model = Transformer(config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3))

@nnx.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        return compute_loss(model, batch)
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss
```

---

## Optax Optimizers

```python
import optax

# Chain transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.add_decayed_weights(0.01),
    optax.scale(-1e-3)
)

# With learning rate schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=10000
)
optimizer = optax.adam(learning_rate=schedule)

# Gradient accumulation
optimizer = optax.multi_steps(optax.adam(1e-3), every_k_schedule=4)
```

---

## Orbax Checkpointing

```python
import orbax.checkpoint as ocp

checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

# Save (non-blocking)
checkpointer.save('/ckpt/step_1000', {
    'model': model, 'opt_state': opt_state, 'step': 1000
})

# Load
restored = checkpointer.restore('/ckpt/step_1000')
```

---

## NumPyro Bayesian

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def bayesian_regression(x, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(alpha + beta*x, sigma), obs=y)

mcmc = MCMC(NUTS(bayesian_regression), num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), x_train, y_train)
```

---

## Multi-Device Scaling

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('data', 'model'))

data_sharding = NamedSharding(mesh, P('data', None))
x_sharded = jax.device_put(x, data_sharding)
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Python control flow | Tracer error | Use `jax.lax.cond` |
| Dynamic shapes | Recompilation | `static_argnums` or pad |
| Global RNG | Breaks purity | Explicit key splitting |
| In-place mutation | Error | Use `x.at[0].set(1)` |

### Debug Patterns
```python
# Disable JIT for debugging
with jax.disable_jit():
    result = fn(x)

# Enable NaN checking
jax.config.update("jax_debug_nans", True)

# Print inside JIT
jax.debug.print("x = {}", x)

# Check shapes
jax.tree_map(lambda x: x.shape, pytree)
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Pure functions | No side effects, no mutations |
| Explicit RNG | Split keys, pass explicitly |
| JIT hot paths | 10-100x speedup |
| vmap before pmap | Vectorize locally first |
| Profile first | Use `jax.profiler` |
| Memory efficiency | Use `jax.remat` for large models |

---

## Checklist

- [ ] Functions are pure and side-effect free
- [ ] RNG keys split and passed explicitly
- [ ] Hot paths JIT-compiled
- [ ] Static args marked for dynamic shapes
- [ ] Checkpointing with async saves
- [ ] Gradient clipping enabled
- [ ] Memory profiled for large models

---

**Version**: 1.0.5
