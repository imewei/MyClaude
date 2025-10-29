# Optax Optimizers - Complete Reference

## Overview

Optax is JAX's gradient transformation library for optimization. It provides composable transformations, built-in optimizers, and learning rate schedules.

---

## 1. Core Concepts

### Optimizer as Transformation Chain

Optax views optimizers as chains of gradient transformations:

```python
import optax

# Adam = gradient clipping + scale by Adam statistics + weight decay
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),      # Clip gradients
    optax.scale_by_adam(),                # Adam statistics
    optax.add_decayed_weights(0.01),      # Weight decay
    optax.scale(-1e-3)                    # Learning rate
)
```

### Basic Usage Pattern

```python
import jax
import jax.numpy as jnp
import optax

# 1. Create optimizer
optimizer = optax.adam(learning_rate=1e-3)

# 2. Initialize optimizer state
params = {'w': jnp.ones((10, 5)), 'b': jnp.zeros(10)}
opt_state = optimizer.init(params)

# 3. Training step
def train_step(params, opt_state, batch):
    def loss_fn(params):
        pred = params['w'] @ batch['x'] + params['b']
        return jnp.mean((pred - batch['y']) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # 4. Apply optimizer
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss
```

---

## 2. Built-in Optimizers

### Adam Family

```python
# Standard Adam
adam = optax.adam(
    learning_rate=1e-3,
    b1=0.9,          # Momentum decay
    b2=0.999,        # RMSprop decay
    eps=1e-8
)

# AdamW (Adam with decoupled weight decay)
adamw = optax.adamw(
    learning_rate=1e-3,
    weight_decay=0.01,
    b1=0.9,
    b2=0.999
)

# AdamW with gradient accumulation
adamw_accum = optax.MultiSteps(
    optax.adamw(1e-3, weight_decay=0.01),
    every_k_schedule=4  # Accumulate 4 batches
)

# Adafactor (memory-efficient Adam)
adafactor = optax.adafactor(
    learning_rate=1e-3,
    min_dim_size_to_factor=128  # Only factor large dims
)
```

### SGD Family

```python
# Standard SGD
sgd = optax.sgd(learning_rate=0.1)

# SGD with momentum
sgd_momentum = optax.sgd(
    learning_rate=0.1,
    momentum=0.9,
    nesterov=False
)

# Nesterov momentum
nesterov = optax.sgd(
    learning_rate=0.1,
    momentum=0.9,
    nesterov=True
)
```

### Advanced Optimizers

```python
# RMSprop
rmsprop = optax.rmsprop(
    learning_rate=1e-3,
    decay=0.9,
    eps=1e-8
)

# Lion (newer, memory-efficient)
lion = optax.lion(
    learning_rate=1e-4,  # Typically needs lower LR
    b1=0.9,
    b2=0.99
)

# Lamb (large batch training)
lamb = optax.lamb(
    learning_rate=1e-3,
    weight_decay=0.01
)

# Shampoo (second-order optimizer)
shampoo = optax.shampoo(
    learning_rate=1e-1,
    block_size=256
)

# SM3 (memory-efficient for large embeddings)
sm3 = optax.sm3(learning_rate=0.01)

# Adagrad
adagrad = optax.adagrad(learning_rate=0.1)
```

---

## 3. Learning Rate Schedules

### Constant Schedule

```python
# Simple constant rate
constant = optax.constant_schedule(value=1e-3)

# Piecewise constant
piecewise = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={
        1000: 0.1,   # Multiply by 0.1 at step 1000
        5000: 0.1,   # Multiply by 0.1 at step 5000
    }
)
```

### Decay Schedules

```python
# Exponential decay
exp_decay = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=1000,
    decay_rate=0.96,
    staircase=False  # Smooth vs step decay
)

# Polynomial decay
poly_decay = optax.polynomial_schedule(
    init_value=1e-3,
    end_value=1e-5,
    power=1.0,  # Linear decay
    transition_steps=10000
)

# Cosine decay
cosine_decay = optax.cosine_decay_schedule(
    init_value=1e-3,
    decay_steps=10000,
    alpha=0.1  # Min LR = 0.1 * init_value
)
```

### Warmup Schedules

```python
# Linear warmup
warmup_linear = optax.linear_schedule(
    init_value=0.0,
    end_value=1e-3,
    transition_steps=1000
)

# Warmup + cosine decay (most common for transformers)
warmup_cosine = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000,
    end_value=1e-5
)

# Warmup + exponential decay
warmup_exp = optax.warmup_exponential_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    transition_steps=10000,
    decay_rate=0.96
)
```

### Custom Schedules

```python
# Schedule composition
def custom_schedule(step):
    warmup = optax.linear_schedule(0.0, 1e-3, 1000)
    decay = optax.exponential_decay(1e-3, 1000, 0.96)
    return jax.lax.cond(
        step < 1000,
        lambda: warmup(step),
        lambda: decay(step - 1000)
    )

# Use with optimizer
optimizer = optax.adam(learning_rate=custom_schedule)
```

---

## 4. Gradient Transformations

### Gradient Clipping

```python
# Clip by global norm (most common)
clip_norm = optax.clip_by_global_norm(max_norm=1.0)

# Clip by value (per-element)
clip_value = optax.clip(min_delta=-1.0, max_delta=1.0)

# Adaptive clipping (scales with parameter norm)
adaptive_clip = optax.adaptive_grad_clip(clipping=0.01)
```

### Gradient Accumulation

```python
# Accumulate over multiple batches (useful for large models)
optimizer = optax.MultiSteps(
    optax.adam(1e-3),
    every_k_schedule=4  # Update every 4 batches
)

# Custom accumulation
accumulator = optax.chain(
    optax.scale(1.0 / 4),  # Scale by accumulation steps
    optax.adam(1e-3)
)
```

### Gradient Transformations

```python
# Scale gradients
scale = optax.scale(learning_rate=-1e-3)

# Scale by trust ratio (LARS/LAMB)
scale_by_trust_ratio = optax.scale_by_trust_ratio()

# Scale by RMS
scale_by_rms = optax.scale_by_rms()

# Scale by Adam statistics
scale_by_adam = optax.scale_by_adam(b1=0.9, b2=0.999)

# Add noise to gradients
add_noise = optax.add_noise(
    eta=0.01,
    gamma=0.55,
    seed=0
)
```

### Weight Decay

```python
# L2 regularization (weight decay)
weight_decay = optax.add_decayed_weights(
    weight_decay=0.01,
    mask=None  # Apply to all params
)

# Masked weight decay (exclude biases and norms)
def weight_decay_mask(params):
    # Don't apply to bias and norm parameters
    return jax.tree_map(
        lambda name: 'bias' not in name and 'norm' not in name,
        params
    )

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.add_decayed_weights(0.01, mask=weight_decay_mask),
    optax.scale(-1e-3)
)
```

---

## 5. Advanced Patterns

### Per-Parameter Learning Rates

```python
# Different LR for different parameter groups
def lr_multipliers(params):
    return {
        'embedding': 0.1,  # Smaller LR for embeddings
        'decoder': 1.0,    # Normal LR for decoder
    }

optimizer = optax.multi_transform(
    {
        'embedding': optax.adam(1e-4),
        'decoder': optax.adam(1e-3),
    },
    param_labels=lr_multipliers(params)
)
```

### Selective Parameter Updates

```python
# Freeze certain parameters
def trainable_mask(params):
    # Only train decoder, freeze encoder
    return {
        'encoder': False,
        'decoder': True,
    }

optimizer = optax.masked(
    optax.adam(1e-3),
    mask=trainable_mask
)
```

### Gradient Surgery

```python
# Zero out gradients for specific parameters
zero_nans = optax.zero_nans()

# Apply different transformations to different params
optimizer = optax.chain(
    optax.zero_nans(),
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.add_decayed_weights(0.01),
    optax.scale(-1e-3)
)
```

### Mixed Precision Training

```python
# Dynamic loss scaling for mixed precision
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0)
)

# With loss scaling
loss_scale = optax.scale_gradient(scale=2.0**15)

def train_step(params, opt_state, batch):
    def scaled_loss_fn(params):
        loss = loss_fn(params, batch)
        return loss * 2.0**15  # Scale loss

    grads = jax.grad(scaled_loss_fn)(params)
    grads = jax.tree_map(lambda g: g / 2.0**15, grads)  # Unscale grads

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

---

## 6. Composing Optimizers

### Chain Multiple Transformations

```python
# Custom Adam with extra features
custom_adam = optax.chain(
    optax.clip_by_global_norm(1.0),           # 1. Clip gradients
    optax.scale_by_adam(b1=0.9, b2=0.999),    # 2. Adam statistics
    optax.add_decayed_weights(0.01),          # 3. Weight decay
    optax.scale_by_schedule(schedule),        # 4. Learning rate schedule
    optax.scale(-1.0)                         # 5. Gradient descent step
)
```

### Multi-Optimizer

```python
# Different optimizers for different parameter groups
optimizer = optax.multi_transform(
    {
        'encoder': optax.sgd(0.01, momentum=0.9),
        'decoder': optax.adam(1e-3),
        'embeddings': optax.adafactor(1e-3),
    },
    param_labels={
        'encoder': 'encoder',
        'decoder': 'decoder',
        'embeddings': 'embeddings',
    }
)
```

---

## 7. Common Training Patterns

### Standard Training Loop

```python
import optax
import jax
import jax.numpy as jnp

# Setup
model = create_model()
params = model.init(rng, dummy_input)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Training step
@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['x'])
        return jnp.mean((logits - batch['y']) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        params, opt_state, loss = train_step(params, opt_state, batch)
```

### With Learning Rate Schedule

```python
# Create schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000,
    end_value=1e-5
)

# Optimizer with schedule
optimizer = optax.adam(learning_rate=schedule)
opt_state = optimizer.init(params)

# Get current LR (for logging)
def get_current_lr(schedule, step):
    return schedule(step)

for step, batch in enumerate(dataloader):
    params, opt_state, loss = train_step(params, opt_state, batch)
    current_lr = get_current_lr(schedule, step)
    print(f"Step {step}, LR: {current_lr:.6f}, Loss: {loss:.4f}")
```

### Gradient Accumulation Pattern

```python
# Optimizer with accumulation
optimizer = optax.MultiSteps(
    optax.adamw(1e-3, weight_decay=0.01),
    every_k_schedule=4  # Effective batch size = 4x
)

opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['x'])
        return jnp.mean((logits - batch['y']) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

---

## 8. Optimizer Comparison

### Performance Characteristics

| Optimizer | Memory | Convergence | Best For |
|-----------|--------|-------------|----------|
| SGD | Low | Slow | Small datasets, fine-tuning |
| Adam | Medium | Fast | General purpose, transformers |
| AdamW | Medium | Fast | NLP, vision with weight decay |
| Lion | Low | Fast | Large-scale training |
| Adafactor | Low | Medium | Large models (saves memory) |
| Lamb | Medium | Fast | Large batch training |
| Shampoo | High | Very Fast | Second-order optimization |

### Recommended Configurations

```python
# Image classification (ResNet, ViT)
image_optimizer = optax.sgd(
    learning_rate=0.1,
    momentum=0.9,
    nesterov=True
)

# NLP transformers
nlp_optimizer = optax.adamw(
    learning_rate=optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=1000,
        decay_steps=10000
    ),
    weight_decay=0.01,
    b1=0.9,
    b2=0.999
)

# Large-scale training (saves memory)
large_scale_optimizer = optax.lion(
    learning_rate=1e-4,
    b1=0.9,
    b2=0.99
)

# Fine-tuning pretrained models
finetune_optimizer = optax.sgd(
    learning_rate=optax.exponential_decay(
        init_value=1e-2,
        transition_steps=1000,
        decay_rate=0.96
    ),
    momentum=0.9
)
```

---

## 9. Debugging Optimizer Issues

### Check Gradient Statistics

```python
def log_gradient_stats(grads):
    flat_grads = jax.tree_leaves(grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grads))
    grad_mean = jnp.mean(jnp.array([jnp.mean(g) for g in flat_grads]))
    grad_std = jnp.std(jnp.array([jnp.std(g) for g in flat_grads]))

    print(f"Grad norm: {grad_norm:.4f}")
    print(f"Grad mean: {grad_mean:.6f}")
    print(f"Grad std: {grad_std:.6f}")

# In training loop
loss, grads = jax.value_and_grad(loss_fn)(params)
log_gradient_stats(grads)
```

### Check for NaN/Inf

```python
def check_grads(grads):
    for name, grad in jax.tree_leaves_with_path(grads):
        if jnp.any(jnp.isnan(grad)) or jnp.any(jnp.isinf(grad)):
            print(f"NaN/Inf in gradient: {name}")
            return False
    return True
```

### Monitor Learning Rate

```python
# Extract current LR from optimizer state
def get_lr_from_opt_state(opt_state, optimizer):
    # For scheduled optimizers
    if hasattr(opt_state, 'hyperparams'):
        return opt_state.hyperparams.get('learning_rate', None)
    return None
```

---

## 10. Best Practices

### General Guidelines

```python
# 1. Start with Adam for fast prototyping
optimizer = optax.adam(1e-3)

# 2. Add gradient clipping for stability
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3)
)

# 3. Use warmup for transformers
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000
)
optimizer = optax.adam(learning_rate=schedule)

# 4. Add weight decay for regularization
optimizer = optax.adamw(
    learning_rate=schedule,
    weight_decay=0.01
)
```

### Hyperparameter Ranges

```python
# Learning rates (order of magnitude)
lr_ranges = {
    'SGD': (1e-2, 1e-1),
    'Adam': (1e-4, 1e-3),
    'AdamW': (1e-4, 1e-3),
    'Lion': (1e-5, 1e-4),
    'Adafactor': (1e-3, 1e-2),
}

# Weight decay
weight_decay_range = (1e-5, 1e-1)

# Gradient clipping
clip_norm_range = (0.1, 10.0)
```

---

## References

- [Optax Documentation](https://optax.readthedocs.io/)
- [Optax GitHub](https://github.com/deepmind/optax)
- [Adam Paper](https://arxiv.org/abs/1412.6980)
- [Lion Paper](https://arxiv.org/abs/2302.06675)
