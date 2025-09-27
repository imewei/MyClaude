---
description: Set up optimizers with Optax including gradient transforms and learning rate schedules
category: jax-ml
argument-hint: "[--optimizer-type] [--learning-rate] [--schedule]"
allowed-tools: "*"
---

# /jax-optax-optimizer

Set up optimizers using the Optax library for JAX machine learning.

## Description

Configures optimizers from the Optax library including Adam, SGD, and custom optimization schedules. Handles gradient transforms, state initialization, and parameter updates.

## Usage

```
/jax-optax-optimizer [--optimizer-type] [--learning-rate] [--schedule]
```

## What it does

1. Set up common optimizers (Adam, SGD, RMSprop)
2. Configure learning rate schedules
3. Initialize optimizer state
4. Apply gradient updates with state management

## Example output

```python
import optax
import jax.numpy as jnp

# Basic optimizers
adam_optimizer = optax.adam(learning_rate=1e-3)
sgd_optimizer = optax.sgd(learning_rate=1e-2)
rmsprop_optimizer = optax.rmsprop(learning_rate=1e-3)

# Optimizers with momentum and decay
sgd_momentum = optax.sgd(learning_rate=1e-2, momentum=0.9)
adam_decay = optax.adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8)

# Learning rate schedules
exponential_decay = optax.exponential_decay(
    init_value=1e-3, transition_steps=1000, decay_rate=0.95
)
cosine_decay = optax.cosine_decay_schedule(
    init_value=1e-3, decay_steps=10000
)
warmup_cosine = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3, warmup_steps=1000, decay_steps=10000
)

# Optimizer with schedule
scheduled_optimizer = optax.adam(learning_rate=exponential_decay)

# Custom gradient transformations
def clip_and_adam(learning_rate, max_norm=1.0):
    return optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.adam(learning_rate)
    )

optimizer = clip_and_adam(1e-3, max_norm=1.0)

# Initialize optimizer state
params = {  # Your model parameters
    'dense1': {'kernel': jnp.ones((784, 128)), 'bias': jnp.zeros(128)},
    'dense2': {'kernel': jnp.ones((128, 10)), 'bias': jnp.zeros(10)}
}
opt_state = optimizer.init(params)

# Apply gradients (typically in training loop)
def update_step(params, opt_state, grads):
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# Advanced: Multi-optimizer setup
def multi_optimizer(lr_dense=1e-3, lr_conv=1e-4):
    return optax.multi_transform({
        'dense': optax.adam(lr_dense),
        'conv': optax.adam(lr_conv)
    }, {
        'dense': 'dense_layers',
        'conv': 'conv_layers'
    })

# Gradient preprocessing
preprocessor = optax.chain(
    optax.clip_by_global_norm(1.0),        # Gradient clipping
    optax.zero_nans(),                     # Replace NaNs with zeros
    optax.scale_by_adam(),                 # Adam scaling
    optax.scale(-1e-3)                     # Learning rate
)

# Common optimizer recipes
def get_optimizer(name, learning_rate=1e-3, **kwargs):
    optimizers = {
        'adam': optax.adam(learning_rate, **kwargs),
        'sgd': optax.sgd(learning_rate, **kwargs),
        'adamw': optax.adamw(learning_rate, **kwargs),
        'rmsprop': optax.rmsprop(learning_rate, **kwargs),
        'adagrad': optax.adagrad(learning_rate, **kwargs)
    }
    return optimizers[name]
```

## Related Commands

- `/jax-ml-train` - Use optimizers in training loops
- `/jax-flax-model` - Optimize neural network parameters
- `/jax-grad` - Compute gradients for optimization