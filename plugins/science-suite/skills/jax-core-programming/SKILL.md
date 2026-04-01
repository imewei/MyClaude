---
name: jax-core-programming
description: Master JAX functional transformations (jit, vmap, pmap, grad), Flax NNX neural networks, Optax optimizers, Orbax checkpointing, and NumPyro Bayesian inference. Use when writing JAX code, building training loops, optimizing XLA compilation, debugging tracer errors, or scaling to multi-device GPU/TPU.
---

# JAX Core Programming

## Expert Agent

For complex JAX transformations, distributed training, and performance engineering, delegate to the expert agent:

- **`jax-pro`**: Unified specialist for Core JAX optimization, hardware acceleration, and production deployments.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Multi-device parallelism (pmap/sharding), XLA optimization, custom VJPs, and memory efficiency.

## Transforms

| Transform | Purpose |
|-----------|---------|
| `jax.jit` | XLA compile (10-100x speedup) |
| `jax.vmap` | Vectorize batch |
| `jax.pmap` | Multi-device parallel |
| `jax.grad` | Auto-differentiation |

```python
# JIT + vmap composition
fast_batched = jax.jit(jax.vmap(fn))

# Gradients
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

# Static args (avoid recompile)
@jax.jit(static_argnums=(1,))
def shaped_fn(x, shape):
    return jnp.zeros(shape)
```

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
        for layer in self.layers: x = layer(x)
        return self.norm(x)

model = Transformer(config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3))

@nnx.jit
def train_step(optimizer, batch):
    loss, grads = nnx.value_and_grad(lambda m: compute_loss(m, batch))(optimizer.model)
    optimizer.update(grads)
    return loss
```

## Optax

```python
# Chain transformations
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(),
                        optax.add_decayed_weights(0.01), optax.scale(-1e-3))

# LR schedule
schedule = optax.warmup_cosine_decay_schedule(0.0, 1e-3, 1000, 10000)
optimizer = optax.adam(learning_rate=schedule)

# Gradient accumulation
optimizer = optax.multi_steps(optax.adam(1e-3), every_k_schedule=4)
```

## Orbax

```python
import orbax.checkpoint as ocp
ckpt = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
ckpt.save('/ckpt/step_1000', {'model': model, 'opt_state': opt_state, 'step': 1000})
restored = ckpt.restore('/ckpt/step_1000')
```

## NumPyro

```python
import numpyro, numpyro.distributions as dist
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

## Parallelization Strategies

| Strategy | API | Use Case |
|----------|-----|----------|
| **SPMD** | `jax.pmap` | Parallelize across devices (simple) |
| **Sharding** | `jax.sharding` | Tensor parallelism (complex models) |
| **Manual** | `shard_map` | Explicit communication control |
| **Pipeline** | `jax.lax.scan` | Gradient accumulation / Time-pipelining |

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('data', 'model'))
x_sharded = jax.device_put(x, NamedSharding(mesh, P('data', None)))
```

## Pitfalls & Fixes

| Issue | Solution |
|-------|----------|
| Python control flow | `jax.lax.cond` |
| Dynamic shapes | `static_argnums` or pad |
| Global RNG | Explicit key splitting |
| In-place mutation | `x.at[0].set(1)` |

**Debug**:
```python
with jax.disable_jit(): result = fn(x)  # Disable JIT
jax.config.update("jax_debug_nans", True)  # NaN check
jax.debug.print("x = {}", x)  # Print in JIT
```

**Outcome**: Pure functions, explicit RNG, JIT hot paths, multi-device scaling

## Equinox: PyTree-Native Neural Networks

Equinox models ARE PyTrees — every layer, parameter, and buffer is a leaf in the tree. No separate `params` dict needed.

```python
import equinox as eqx
import jax

class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(784, 256, key=k1),
                       eqx.nn.Linear(256, 64, key=k2),
                       eqx.nn.Linear(64, 10, key=k3)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

# Filtered transforms: JIT/grad only over parameters, not static fields
@eqx.filter_jit
def train_step(model, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    model = jax.tree.map(lambda p, g: p - 0.001 * g, model, grads)
    return model, loss
```

**When to use**: Equinox for scientific computing, Diffrax integration, and PyTree-native workflows. Flax NNX for large-scale ML with ecosystem tooling (Orbax, CLU).

## Lineax: Linear Solvers

```python
import lineax as lx
import jax.numpy as jnp

A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
b = jnp.array([1.0, 2.0])
solution = lx.linear_solve(lx.MatrixLinearOperator(A), b)
x = solution.value  # [0.1, 0.6333...]
```

| Solver | Use Case |
|--------|----------|
| `lx.CG()` | Symmetric positive-definite (SPD) matrices |
| `lx.GMRES()` | General non-symmetric systems |
| `lx.LU()` | Dense direct solves |

**Note**: Diffrax implicit solvers (e.g., `Kvaerno5`) use Lineax internally for their Newton steps.

## interpax: JIT-Safe Interpolation

**Rule**: Never use `scipy.interpolate` inside JIT — it traces Python objects and breaks.

```python
import interpax

# 1D interpolation (JIT-safe)
y_new = interpax.interp1d(x_new, x_data, y_data, method="cubic")

# 2D interpolation
z_new = interpax.interp2d(x_new, y_new, x_data, y_data, z_data, method="cubic")

# B-spline variant
y_bspline = interpax.interp1d(x_new, x_data, y_data, method="cubic2")
```

All `interpax` functions are compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

## Optimistix: Root-Finding & Optimization

```python
import optimistix as optx

# Root-finding: find x such that residual_fn(x, args) = 0
sol = optx.root_find(residual_fn, optx.Newton(rtol=1e-8, atol=1e-8), x0, args=args)
x_root = sol.value

# Fixed-point iteration: find x such that g(x) = x
sol = optx.fixed_point(g_fn, optx.FixedPointIteration(rtol=1e-6, atol=1e-6), x0, args=args)

# Nonlinear least squares: minimize ||residual_fn(p, args)||^2
sol = optx.least_squares(residual_fn, optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8), p0, args=args)
params = sol.value
```

All solvers are JIT-compatible and differentiable via implicit differentiation.
