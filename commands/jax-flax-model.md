---
description: Define neural networks using Flax Linen with module setup and variable initialization
category: jax-ml
argument-hint: "[--architecture] [--layers] [--activation]"
allowed-tools: "*"
---

# /jax-flax-model

Define neural networks using Flax Linen module system.

## Description

Creates neural network architectures using Flax Linen, including module setup, variable initialization, and the apply method for forward passes.

## Usage

```
/jax-flax-model [--architecture] [--layers] [--activation]
```

## What it does

1. Define Flax neural network modules
2. Set up parameter initialization
3. Implement forward pass with apply method
4. Handle variable collections and mutable state

## Example output

```python
import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Sequence

# Basic MLP model
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = jax.nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

# CNN model
class CNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = jax.nn.relu(x)
        x = jax.nn.avg_pool2d(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = jax.nn.relu(x)
        x = jax.nn.avg_pool2d(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

# Model with batch normalization and dropout
class ResNetBlock(nn.Module):
    features: int
    training: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        x = jax.nn.relu(x)
        x = nn.Dropout(rate=0.1, deterministic=not self.training)(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        return jax.nn.relu(x + residual)

# Initialize model parameters
def init_model(model, rng, input_shape):
    dummy_input = jnp.ones(input_shape)
    params = model.init(rng, dummy_input)
    return params

# Model with mutable state (for batch norm)
def init_model_with_state(model, rng, input_shape):
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    return params, batch_stats

# Apply model with state updates
def apply_model(model, params, batch_stats, x, training=True):
    variables = {'params': params, 'batch_stats': batch_stats}
    output, new_variables = model.apply(
        variables, x, training=training, mutable=['batch_stats']
    )
    new_batch_stats = new_variables.get('batch_stats', batch_stats)
    return output, new_batch_stats

# Usage example
key = jax.random.PRNGKey(0)
model = MLP(features=[128, 64, 10])
params = init_model(model, key, (1, 784))

# Forward pass
x = jnp.ones((32, 784))  # batch of inputs
output = model.apply(params, x)
```

## Related Commands

- `/jax-ml-train` - Use Flax models in training loops
- `/jax-optax-optimizer` - Optimize Flax model parameters
- `/jax-orbax-checkpoint` - Save/load Flax models