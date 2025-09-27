---
description: Define functional neural networks with Equinox using eqx.nn for layers
category: jax-ml
argument-hint: "[--architecture] [--functional] [--stateful]"
allowed-tools: "*"
---

# /jax-equinox-model

Define functional neural networks using the Equinox library.

## Description

Creates neural network architectures using Equinox, a JAX library for functional programming with neural networks. Uses eqx.nn for layers and provides functional model composition.

## Usage

```
/jax-equinox-model [--architecture] [--functional] [--stateful]
```

## What it does

1. Define neural networks using Equinox modules
2. Use eqx.nn for functional layer composition
3. Handle model parameters as PyTrees
4. Implement functional forward passes

## Example output

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Basic MLP model with Equinox
class MLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width_size, depth, *, key):
        keys = jr.split(key, depth + 1)
        self.layers = []

        # Input layer
        self.layers.append(eqx.nn.Linear(in_size, width_size, key=keys[0]))

        # Hidden layers
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))

        # Output layer
        self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

# CNN model with Equinox
class CNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, *, key):
        keys = jr.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=3, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, key=keys[1])
        self.linear1 = eqx.nn.Linear(64 * 5 * 5, 128, key=keys[2])  # Adjust size based on input
        self.linear2 = eqx.nn.Linear(128, 10, key=keys[3])

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.max_pool2d(x, window_shape=(2, 2))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.max_pool2d(x, window_shape=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)

# Functional model composition
def create_encoder_decoder(input_dim, latent_dim, output_dim, *, key):
    enc_key, dec_key = jr.split(key)

    encoder = MLP(input_dim, latent_dim, 64, 3, key=enc_key)
    decoder = MLP(latent_dim, output_dim, 64, 3, key=dec_key)

    def autoencoder(x):
        latent = encoder(x)
        reconstructed = decoder(latent)
        return reconstructed

    return autoencoder, encoder, decoder

# Stateful model with dropout
class DropoutMLP(eqx.Module):
    layers: list
    dropout: eqx.nn.Dropout
    inference: bool

    def __init__(self, in_size, out_size, width_size, depth, dropout_p=0.1, *, key):
        keys = jr.split(key, depth + 1)
        self.layers = []

        self.layers.append(eqx.nn.Linear(in_size, width_size, key=keys[0]))
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
        self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[-1]))

        self.dropout = eqx.nn.Dropout(dropout_p)
        self.inference = False

    def __call__(self, x, *, key=None):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
            if not self.inference:
                x = self.dropout(x, key=key)
        return self.layers[-1](x)

# Model parameter manipulation
def get_model_params(model):
    \"\"\"Extract trainable parameters from model.\"\"\"
    return eqx.filter(model, eqx.is_array)

def set_model_params(model, params):
    \"\"\"Set model parameters.\"\"\"
    return eqx.combine(params, model)

# Training utilities
def loss_fn(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)

def make_step(model, x, y, optim, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Model serialization
def save_model(model, filename):
    \"\"\"Save Equinox model to file.\"\"\"
    eqx.tree_serialise_leaves(filename, model)

def load_model(filename, model_template):
    \"\"\"Load Equinox model from file.\"\"\"
    return eqx.tree_deserialise_leaves(filename, model_template)

# Usage example
def create_and_train_model():
    key = jr.PRNGKey(42)
    model_key, data_key = jr.split(key)

    # Create model
    model = MLP(in_size=784, out_size=10, width_size=128, depth=3, key=model_key)

    # Generate dummy data
    x = jr.normal(data_key, (100, 784))
    y = jr.normal(data_key, (100, 10))

    # Set up optimizer (using Optax)
    import optax
    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Training step
    model, opt_state, loss = make_step(model, x, y, optim, opt_state)
    print(f\"Loss: {loss}\")

    return model

# Functional model transformations
@eqx.filter_jit
def evaluate_model(model, x):
    \"\"\"JIT-compiled model evaluation.\"\"\"
    return jax.vmap(model)(x)

@eqx.filter_grad
def compute_gradients(model, x, y):
    \"\"\"Compute gradients for model parameters.\"\"\"
    return loss_fn(model, x, y)
```

## Related Commands

- `/jax-ml-train` - Train Equinox models
- `/jax-optax-optimizer` - Optimize Equinox parameters
- `/jax-flax-model` - Alternative neural network framework