---
name: jax-core-programming
description: Master JAX functional programming for high-performance array computing and machine learning. This skill should be used when working with JAX transformations (jit, vmap, pmap, grad), Flax NNX neural networks, Optax optimizers, Orbax checkpointing, NumPyro probabilistic programming, or performance optimization with XLA compilation. Covers pure functional design, hardware acceleration (GPU/TPU), and the complete JAX ecosystem (Flax, Optax, Orbax, NumPyro).
---

# JAX Core Programming

Master JAX, the high-performance numerical computing library with automatic differentiation and XLA compilation for hardware-accelerated machine learning and scientific computing.

## What This Skill Provides

This skill equips Claude to become a JAX expert capable of:

1. **JAX Functional Transformations** - Apply jit, vmap, pmap, grad for performance optimization
2. **Flax NNX Neural Networks** - Build modern neural networks with stateful Flax NNX modules
3. **Optax Optimization** - Compose gradient transformations and advanced optimizers
4. **Orbax Persistence** - Implement async checkpointing and model serialization
5. **NumPyro Bayesian Inference** - Build probabilistic models with MCMC and variational inference
6. **Performance Engineering** - Optimize for GPU/TPU with XLA, memory efficiency, and multi-device scaling
7. **Ecosystem Integration** - Build end-to-end ML pipelines with the JAX AI stack

## When to Use This Skill

Invoke this skill when encountering:

- **Functional transformations**: jit compilation, vmap vectorization, pmap parallelization, automatic differentiation
- **Neural network development**: Flax NNX modules, transformers, LLMs, training loops
- **Optimization strategies**: Optax optimizer composition, learning rate schedules, gradient clipping
- **Checkpointing**: Orbax async saves, model persistence, multi-host checkpointing
- **Probabilistic programming**: NumPyro MCMC, variational inference, Bayesian neural networks
- **Performance optimization**: XLA compilation, memory efficiency (remat), multi-device scaling
- **Debugging JAX code**: Tracer errors, recompilation overhead, gradient issues, pytree problems
- **Pure functional programming**: Immutable pytrees, explicit RNG management, side-effect-free functions

## Core Capabilities

### 1. JAX Core: Functional Transformations and Performance Optimization

**Pure Function Design**: Craft side-effect-free functions using JAX primitives (jax.numpy), handle PyTrees for structured data, and ensure transformation compatibility.

**Key Patterns**:
```python
# Pure function with explicit RNG
def pure_layer(params, x, rng):
    rng1, rng2 = jax.random.split(rng)
    x = apply_dropout(x, rng1, rate=0.1)
    return params @ x, rng2  # Return new RNG state

# PyTree operations
params = {'w': w_matrix, 'b': b_vector}  # Structured data
grads = jax.tree_map(lambda x: x * 0.9, params)  # Apply to all leaves
```

**JIT Compilation**: Apply `jax.jit` for XLA-optimized static graphs (10-100x speedup), manage static vs dynamic arguments with `static_argnums`, minimize recompilation through caching.

```python
# Basic JIT
@jax.jit
def fast_fn(x):
    return jnp.sum(x ** 2)

# Static arguments (avoid recompilation)
@jax.jit(static_argnums=(1,))
def shaped_fn(x, shape):  # shape won't trigger recompilation
    return jnp.zeros(shape)
```

**Vectorization (vmap)**: Batch-process operations over axes for parallelism, compose with jit for acceleration.

```python
# Vectorize over batch dimension
batched_loss = jax.vmap(single_loss, in_axes=(None, 0, 0))
losses = batched_loss(params, batch_x, batch_y)

# Composition for maximum performance
fast_batched = jax.jit(jax.vmap(fn))
```

**Multi-Device Parallelism (pmap)**: Distribute computations across devices with sharding, handle data parallelism, optimize for TPUs with buffer donation.

```python
# Data parallelism
@jax.pmap
def parallel_train_step(params, local_batch):
    loss = compute_loss(params, local_batch)
    # Average across devices
    return jax.lax.pmean(loss, axis_name='batch')
```

**Automatic Differentiation**: Use `jax.grad`, `jax.value_and_grad`, or `jax.custom_vjp` for forward/backward modes.

```python
# Efficient value + gradient
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

# Higher-order derivatives
hessian = jax.hessian(loss_fn)(params)
```

**Performance Tuning**: Profile with `jax.profiler`, optimize memory via `jax.remat`, scale with prefetching and async ops.

**Reference**: See `references/jax_transformations.md` for detailed transformation patterns and composition strategies.

### 2. Flax NNX: Neural Network Modeling

**Model Construction**: Define models as `nnx.Module` subclasses, initialize layers (nnx.Linear, nnx.Conv, nnx.BatchNorm, nnx.RMSNorm) with RNGs, implement forward passes with activations.

```python
from flax import nnx

class Transformer(nnx.Module):
    def __init__(self, config, rngs):
        self.embed = nnx.Embed(config.vocab, config.dim, rngs=rngs)
        self.layers = [TransformerBlock(config, rngs)
                       for _ in range(config.n_layers)]
        self.norm = nnx.RMSNorm(config.dim, rngs=rngs)

    def __call__(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

**State Management**: Handle mutable state via Python objects, use filters for selective updates (parameters vs batch stats), integrate with JAX transforms via `@nnx.jit`.

```python
# Create model with optimizer
model = Transformer(config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3))

# Training step
@nnx.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        return compute_loss(model, batch)

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss
```

**Functional API Integration**: Split models into state and pure functions for compatibility with jit, vmap, pmap.

**Advanced Features**: Build recurrent nets with `nnx.scan`, custom modules, debug with eager mode.

**Reference**: See `references/flax_nnx_guide.md` for comprehensive module patterns and training strategies.

### 3. Optax: Gradient Transformations and Optimizers

**Optimizer Composition**: Build custom optimizers by chaining transformations for adaptive learning rates, momentum, clipping.

```python
import optax

# Chain transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),        # Gradient clipping
    optax.scale_by_adam(),                  # Adam statistics
    optax.add_decayed_weights(0.01),        # Weight decay
    optax.scale(-1e-3)                      # Learning rate
)

# Or use built-in
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.01)

# With schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=10000
)
optimizer = optax.adam(learning_rate=schedule)
```

**Stateful Updates**: Manage optimizer states in training loops, integrate with Flax models via `nnx.Optimizer`.

**Advanced Techniques**: Gradient accumulation for large batches, conditional updates (freeze layers), adaptive clipping.

```python
# Gradient accumulation (for memory-constrained training)
optimizer = optax.multi_steps(
    optax.adam(1e-3),
    every_k_schedule=4  # Accumulate 4 micro-batches
)

# Selective layer updates
mask = {'encoder': True, 'decoder': False}  # Only update encoder
optimizer = optax.masked(optax.adam(1e-3), mask)
```

**Reference**: See `references/optax_optimizers.md` for optimizer catalog and advanced composition patterns.

### 4. Orbax: Checkpointing and Persistence

**Model Saving/Loading**: Use `orbax.checkpoint.Checkpointer` for saving/loading PyTrees, support formats like Msgpack.

```python
import orbax.checkpoint as ocp

# Create checkpointer
checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

# Save (non-blocking)
checkpointer.save('/checkpoints/step_1000', {
    'model': model,
    'optimizer': opt_state,
    'step': 1000
})

# Load
restored = checkpointer.restore('/checkpoints/step_1000')
model = restored['model']
```

**Async and Sharded Checkpointing**: Implement asynchronous saves to minimize training interruptions, handle sharded states for distributed models.

```python
# Async checkpoint (non-blocking)
async_save = checkpointer.save(
    f'/checkpoints/step_{step}',
    {'model': model, 'opt_state': opt_state},
    force=False  # Skip if previous save in progress
)

# Continue training while checkpoint saves
for step in range(1000, 2000):
    model, opt_state, loss = train_step(model, opt_state, batch)

# Wait before exit
async_save.wait()
```

**Integration with JAX/Flax**: Wrap Optax states and Flax models for seamless persistence, manage versioning and recovery.

**Multi-Host Checkpointing**: Handle distributed checkpointing across multiple hosts with checkpoint managers.

**Reference**: See `references/orbax_checkpointing.md` for advanced checkpoint strategies and disaster recovery patterns.

### 5. NumPyro: Probabilistic Programming

**Model Specification**: Define Bayesian models with `numpyro.sample`, `plate` for hierarchies, distributions (numpyro.distributions.Normal, etc.).

```python
import numpyro
import numpyro.distributions as dist

def bayesian_regression(x, y=None):
    # Priors
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    # Linear model
    mu = alpha + beta * x

    # Likelihood
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)
```

**Inference Algorithms**: Run MCMC (NUTS via `numpyro.infer.MCMC`) and VI (SVI with ELBO), tune for convergence with diagnostics.

```python
from numpyro.infer import MCMC, NUTS

# MCMC inference
nuts_kernel = NUTS(bayesian_regression)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0), x_train, y_train)

# Get posterior samples
posterior_samples = mcmc.get_samples()
```

**JAX Optimizations**: Apply jit and vmap for fast sampling, reparameterization for variance reduction.

**Scalability**: Handle large datasets with mini-batching, GPU acceleration, custom guides.

**Reference**: See `references/numpyro_bayesian.md` for Bayesian modeling patterns and inference strategies. Also refer to the companion `numpyro-core-mastery` skill for deep NumPyro expertise.

### 6. Soft Skills and Ecosystem Integration

**Problem-Solving**: Debug transform issues (tracer leaks, recompilation), iterate on performance bottlenecks.

**Common Debug Patterns**:
```python
# Issue: Tracer errors with Python control flow
# Solution: Use jax.lax control flow
@jax.jit
def fixed_fn(x):
    return jax.lax.cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x,
        x
    )

# Issue: Recompilation overhead
# Solution: Static arguments
@jax.jit(static_argnums=(1,))
def shaped_fn(x, shape):
    return jnp.zeros(shape)

# Issue: NaN gradients
# Solution: Gradient clipping + monitoring
jax.config.update("jax_debug_nans", True)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3)
)
```

**End-to-End Workflows**: Build full pipelines from data prep to deployment, use tools like Weights & Biases for logging.

```python
# Complete training pipeline
def training_pipeline(config):
    # 1. Initialize model
    model = create_model(config)
    optimizer = nnx.Optimizer(model, optax.adamw(config.lr))

    # 2. Setup checkpointing
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    # 3. Training loop
    for epoch in range(config.epochs):
        for batch in dataloader:
            loss = train_step(optimizer, batch)

        # Checkpoint periodically
        if epoch % 10 == 0:
            checkpointer.save(f'/ckpt/epoch_{epoch}', {
                'model': optimizer.model,
                'opt_state': optimizer.opt_state
            })

    return optimizer.model
```

**Collaboration and Learning**: Follow 2025 trends (async checkpointing, Flax NNX, JAX sharding API), contribute to open-source.

## Workflow Patterns

### Pattern 1: Quick JAX Prototyping

For rapid experimentation with JAX transformations:

```python
import jax
import jax.numpy as jnp

# 1. Pure function
def model_fn(params, x):
    return params['w'] @ x + params['b']

# 2. Add transformations incrementally
loss_fn = lambda p, x, y: jnp.mean((model_fn(p, x) - y)**2)

# 3. Compile for speed
fast_loss = jax.jit(loss_fn)

# 4. Vectorize over batch
batched_loss = jax.vmap(fast_loss, in_axes=(None, 0, 0))

# 5. Compute gradients
grad_fn = jax.grad(fast_loss)
```

### Pattern 2: Production Flax NNX Training

For robust training with checkpointing and monitoring:

```python
from flax import nnx
import optax
import orbax.checkpoint as ocp

# 1. Define model
model = Transformer(config, rngs=nnx.Rngs(0))

# 2. Create optimizer
optimizer = nnx.Optimizer(model, optax.adamw(
    learning_rate=optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-3,
        warmup_steps=1000, decay_steps=10000
    ),
    weight_decay=0.01
))

# 3. Setup async checkpointing
checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

# 4. Training step with JIT
@nnx.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        logits = model(batch['input_ids'], train=True)
        return jnp.mean((logits - batch['labels'])**2)

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss

# 5. Training loop
for step in range(10000):
    loss = train_step(optimizer, next(dataloader))

    # Periodic checkpointing
    if step % 1000 == 0:
        checkpointer.save(f'/ckpt/step_{step}', {
            'model': optimizer.model,
            'opt_state': optimizer.opt_state,
            'step': step
        })
```

### Pattern 3: Multi-Device Scaling

For distributed training across GPUs/TPUs:

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# 1. Create device mesh
devices = mesh_utils.create_device_mesh((4, 2))  # 4 data, 2 model
mesh = Mesh(devices, axis_names=('data', 'model'))

# 2. Define sharding
data_sharding = NamedSharding(mesh, P('data', None))
model_sharding = NamedSharding(mesh, P(None, 'model'))

# 3. Shard data and model
x_sharded = jax.device_put(x, data_sharding)
params_sharded = jax.device_put(params, model_sharding)

# 4. Sharding-aware computation
@jax.jit
def sharded_train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # JAX handles cross-device communication automatically
    return loss, grads

# 5. Train with sharded data
for batch in sharded_dataloader:
    loss, grads = sharded_train_step(params_sharded, batch['x'], batch['y'])
```

### Pattern 4: Bayesian Inference with NumPyro

For probabilistic modeling and uncertainty quantification:

```python
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

# 1. Define Bayesian model
def model(x, y=None):
    w = numpyro.sample('w', dist.Normal(0, 1).expand([x.shape[1]]))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))

    mu = x @ w
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

# 2. Run MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), x_train, y_train)

# 3. Posterior predictive
posterior_samples = mcmc.get_samples()
predictive = Predictive(model, posterior_samples)
predictions = predictive(jax.random.PRNGKey(1), x_test)

# 4. Uncertainty quantification
mean_pred = predictions['obs'].mean(axis=0)
ci_lower = jnp.percentile(predictions['obs'], 2.5, axis=0)
ci_upper = jnp.percentile(predictions['obs'], 97.5, axis=0)
```

## Best Practices

### Functional Programming
- **Pure functions only**: No side effects, no mutations, no global state
- **Explicit RNG management**: Always split keys, pass explicitly
- **Immutable pytrees**: Use `jax.tree_map` for updates, never mutate
- **Transformation-ready**: Design for composition (jit ∘ vmap ∘ grad)

### Performance Optimization
- **JIT compile hot paths**: 10-100x speedup on iterative code
- **vmap before pmap**: Vectorize locally, then parallelize across devices
- **Profile before optimizing**: Use `jax.profiler`, measure don't guess
- **Memory efficiency**: Use `jax.remat` for 2-5x memory reduction

### Debugging Strategy
- **Start eager**: Disable JIT (`with jax.disable_jit()`) for debugging
- **Check shapes early**: Use `jax.tree_map(lambda x: x.shape, pytree)`
- **Enable NaN checking**: `jax.config.update("jax_debug_nans", True)`
- **Use jax.debug.print()**: For printing inside JIT functions

### Ecosystem Integration
- **Flax + Optax + Orbax**: Standard stack for neural network training
- **NumPyro for Bayesian**: Leverage JAX for fast probabilistic inference
- **2025 best practices**: Flax NNX (not Linen), async checkpointing, NamedSharding
- **Testing with Chex**: Use `chex.assert_shape`, `chex.assert_type` for correctness

## Common Pitfalls and Solutions

### Pitfall 1: Python Control Flow in JIT

**Problem**:
```python
@jax.jit
def buggy(x):
    if x.sum() > 0:  # ERROR: Can't use tracer as boolean
        return x * 2
    return x
```

**Solution**: Use `jax.lax.cond`:
```python
@jax.jit
def fixed(x):
    return jax.lax.cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x,
        x
    )
```

### Pitfall 2: Recompilation Overhead

**Problem**: Dynamic shapes trigger recompilation on every call

**Solution**: Pad to fixed shapes or use static arguments:
```python
@jax.jit(static_argnums=(1,))  # shape is static
def shaped_fn(x, shape):
    return jnp.zeros(shape)
```

### Pitfall 3: Global RNG State

**Problem**: Using global RNG breaks functional purity

**Solution**: Explicit key management:
```python
def pure_fn(key, x):
    key1, key2 = jax.random.split(key)
    x = apply_dropout(x, key1)
    return x, key2  # Return new key
```

### Pitfall 4: In-Place Mutations

**Problem**: JAX arrays are immutable

**Solution**: Use `.at[]` syntax:
```python
# BAD: x[0] = 1  # Error!
# GOOD:
x = x.at[0].set(1)
```

## Resources

This skill includes bundled resources for JAX development:

### scripts/
Utility scripts for JAX workflows:
- `jax_profiler.py`: Profile JAX code with TensorBoard integration
- `checkpoint_manager.py`: Manage checkpoints with automatic cleanup
- `pytree_visualizer.py`: Visualize pytree structure and shapes

### references/
Detailed documentation for deep dives:
- `jax_transformations.md`: Comprehensive transformation patterns (jit, vmap, pmap, grad)
- `flax_nnx_guide.md`: Flax NNX module patterns and training strategies
- `optax_optimizers.md`: Optimizer catalog and composition patterns
- `orbax_checkpointing.md`: Advanced checkpoint strategies
- `numpyro_bayesian.md`: Bayesian modeling patterns with NumPyro
- `performance_optimization.md`: Memory optimization, profiling, multi-device scaling

## Integration with Other Skills

**Combines with**:
- **numpyro-core-mastery**: For deep Bayesian inference expertise
- **neural-architecture-engineer**: For architecture design decisions
- **ml-pipeline-coordinator**: For end-to-end ML workflows and MLOps

**When to delegate**:
- Complex Bayesian models → Use `numpyro-core-mastery` skill
- Architecture design → Consult neural-architecture-engineer agent
- Production deployment → Hand off to ml-pipeline-coordinator agent

---

**JAX Core Programming** - High-performance functional programming for machine learning and scientific computing with automatic differentiation, XLA compilation, and hardware-agnostic scaling across the JAX AI Stack ecosystem.
