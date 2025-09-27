---
description: Complete ML training loop template with Flax models, Optax optimizers, and JIT compilation
category: jax-ml
argument-hint: "[--model-type] [--optimizer] [--epochs]"
allowed-tools: "*"
---

# /jax-ml-train

Outline a complete machine learning training loop with JAX ecosystem.

## Description

Provides a template for ML training loops using JAX with Flax models, Optax optimizers, and proper gradient updates. Includes JIT compilation and state management.

## Usage

```
/jax-ml-train [--model-type] [--optimizer] [--epochs]
```

## What it does

1. Set up model function (Flax/Equinox)
2. Define loss function with gradients
3. Configure optimizer from Optax
4. Create JIT-compiled training step
5. Implement training loop with state updates

## Example output

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

# 1. Define model (Flax example)
class MLP(nn.Module):
    features: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(10)(x)  # num_classes
        return x

# 2. Define loss function
def loss_fn(params, batch, model):
    x, y = batch
    logits = model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)

# 3. Set up optimizer
optimizer = optax.adam(learning_rate=1e-3)

# 4. Create training state
def create_train_state(rng, model, dummy_input):
    params = model.init(rng, dummy_input)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

# 5. JIT-compiled training step
@jax.jit
def train_step(state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# 6. Training loop
def train_model(state, train_loader, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            state, loss = train_step(state, batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return state

# 7. Initialize and run training
key = jax.random.PRNGKey(42)
model = MLP()
dummy_input = jnp.ones((1, 784))  # Example input shape
state = create_train_state(key, model, dummy_input)

# Train the model
trained_state = train_model(state, train_loader, num_epochs=10)
```

## Related Commands

- `/jax-flax-model` - Define neural network architectures
- `/jax-optax-optimizer` - Set up optimizers
- `/jax-grad` - Compute gradients for training
- `/jax-orbax-checkpoint` - Save/load model checkpoints