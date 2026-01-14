# Flax NNX - Comprehensive Guide

## Overview

Flax NNX is the modern stateful API for neural networks in JAX (2024-2025 standard). It replaces Linen with a Pythonic, object-oriented interface while maintaining JAX compatibility.

---

## 1. Core Concepts

### NNX vs Linen

**Linen (deprecated)**:
```python
class OldModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        return x

# Requires explicit initialization
variables = model.init(rng, x)
```

**NNX (modern)**:
```python
from flax import nnx

class NewModel(nnx.Module):
    def __init__(self, rngs):
        self.dense = nnx.Linear(784, 128, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)

# Direct Python initialization
model = NewModel(rngs=nnx.Rngs(0))
```

### Key Differences

| Feature | Linen | NNX |
|---------|-------|-----|
| State management | Functional (external) | Stateful (Pythonic) |
| Initialization | `init()` + `apply()` | Direct instantiation |
| Variables | `variables` dict | Python attributes |
| Training mode | `deterministic` flag | `use_running_average` |
| Optimizer | Separate optax | `nnx.Optimizer` |

---

## 2. Basic Module Definition

### Simple Linear Model

```python
from flax import nnx
import jax
import jax.numpy as jnp

class LinearModel(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            in_features=784,
            out_features=features,
            rngs=rngs
        )

    def __call__(self, x):
        return self.linear(x)

# Create model
model = LinearModel(features=10, rngs=nnx.Rngs(0))

# Forward pass
x = jnp.ones((32, 784))
y = model(x)  # Shape: (32, 10)
```

### MLP with Activation

```python
class MLP(nnx.Module):
    def __init__(self, hidden_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(784, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

model = MLP(hidden_dim=128, out_dim=10, rngs=nnx.Rngs(0))
```

---

## 3. Common Layers

### Linear Layers

```python
# Basic linear
linear = nnx.Linear(
    in_features=128,
    out_features=256,
    use_bias=True,
    rngs=rngs
)

# No bias
linear_no_bias = nnx.Linear(128, 256, use_bias=False, rngs=rngs)

# Custom initialization
linear_custom = nnx.Linear(
    128, 256,
    kernel_init=nnx.initializers.xavier_uniform(),
    bias_init=nnx.initializers.zeros,
    rngs=rngs
)
```

### Convolutional Layers

```python
class CNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # Conv2D
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            rngs=rngs
        )
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.linear = nnx.Linear(64 * 7 * 7, 10, rngs=rngs)

    def __call__(self, x):
        # Input: (batch, 28, 28, 3)
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.linear(x)
        return x
```

### Normalization Layers

```python
class NormalizedModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(128, 256, rngs=rngs)

        # Batch normalization
        self.bn = nnx.BatchNorm(
            num_features=256,
            use_running_average=False,  # Training mode
            rngs=rngs
        )

        # Layer normalization
        self.ln = nnx.LayerNorm(num_features=256, rngs=rngs)

        # RMS normalization (modern alternative)
        self.rms = nnx.RMSNorm(num_features=256, rngs=rngs)

    def __call__(self, x, train: bool = True):
        x = self.linear(x)

        # BatchNorm needs train mode
        self.bn.use_running_average = not train
        x = self.bn(x)

        x = nnx.relu(x)
        return x
```

### Dropout

```python
class DropoutModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(128, 256, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x, train: bool = True):
        x = self.linear1(x)
        x = nnx.relu(x)

        # Dropout only active in training mode
        if train:
            x = self.dropout(x)

        x = self.linear2(x)
        return x
```

---

## 4. Advanced Patterns

### Transformer Block

```python
class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            decode=False,
            rngs=rngs
        )
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)

        self.mlp = nnx.Sequential([
            nnx.Linear(d_model, 4 * d_model, rngs=rngs),
            lambda x: nnx.gelu(x),
            nnx.Linear(4 * d_model, d_model, rngs=rngs),
        ])

    def __call__(self, x):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.ln1(x)

        # MLP with residual
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)

        return x
```

### Residual Network

```python
class ResidualBlock(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(features, features, (3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(features, features, (3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(features, rngs=rngs)
        self.bn2 = nnx.BatchNorm(features, rngs=rngs)

    def __call__(self, x, train: bool = True):
        residual = x

        x = self.conv1(x)
        self.bn1.use_running_average = not train
        x = self.bn1(x)
        x = nnx.relu(x)

        x = self.conv2(x)
        self.bn2.use_running_average = not train
        x = self.bn2(x)

        x = x + residual  # Skip connection
        x = nnx.relu(x)
        return x
```

### Sequential Models

```python
# Method 1: Explicit sequential
model = nnx.Sequential([
    nnx.Linear(784, 128, rngs=rngs),
    nnx.relu,
    nnx.Dropout(0.2, rngs=rngs),
    nnx.Linear(128, 10, rngs=rngs),
])

# Method 2: Custom sequential
class CustomSequential(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(784, 256, rngs=rngs),
            nnx.Linear(256, 128, rngs=rngs),
            nnx.Linear(128, 10, rngs=rngs),
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = nnx.relu(x)
        return x
```

---

## 5. State Management

### Accessing Parameters

```python
model = MLP(128, 10, rngs=nnx.Rngs(0))

# Get all parameters
params = nnx.state(model)
print(f"Total params: {sum(p.size for p in jax.tree_leaves(params))}")

# Access specific layer
print(model.linear1.kernel.shape)  # Direct Python access
print(model.linear1.bias.shape)

# Iterate over parameters
for name, param in nnx.iter_graph(model):
    print(f"{name}: {param.shape}")
```

### Mutable State (BatchNorm)

```python
class ModelWithBN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.bn = nnx.BatchNorm(128, rngs=rngs)

    def __call__(self, x, train: bool):
        self.bn.use_running_average = not train
        return self.bn(x)

model = ModelWithBN(rngs=nnx.Rngs(0))

# Training updates running statistics
out = model(x, train=True)

# Inference uses running statistics
out = model(x, train=False)
```

---

## 6. Training with NNX

### Basic Training Loop

```python
from flax import nnx
import optax

# Model
model = MLP(128, 10, rngs=nnx.Rngs(0))

# Optimizer (NNX style)
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# Training step with JIT
@nnx.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        logits = model(batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss

# Training loop
for epoch in range(10):
    for batch in dataloader:
        loss = train_step(optimizer, batch)
        print(f"Loss: {loss}")
```

### Advanced Training with LR Schedule

```python
# Learning rate schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000,
    end_value=1e-5
)

# Optimizer with weight decay
tx = optax.adamw(
    learning_rate=schedule,
    weight_decay=0.01
)

optimizer = nnx.Optimizer(model, tx)

# Training step with metrics
@nnx.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        logits = model(batch['image'], train=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()

        # L2 regularization
        l2_loss = 0.01 * sum(
            jnp.sum(p ** 2) for p in jax.tree_leaves(nnx.state(model))
        )
        return loss + l2_loss, {'ce_loss': loss, 'l2_loss': l2_loss}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
    optimizer.update(grads)
    return loss, metrics
```

---

## 7. JAX Transform Integration

### JIT Compilation

```python
# Method 1: Decorator on training step
@nnx.jit
def train_step(optimizer, batch):
    # ... training logic
    pass

# Method 2: JIT the model forward pass
@nnx.jit
def forward(model, x):
    return model(x)
```

### Vectorization (vmap)

```python
# Per-example gradients
@nnx.jit
@jax.vmap
def per_example_grad(model, x, y):
    def loss_fn(model):
        pred = model(x)
        return ((pred - y) ** 2).mean()

    return nnx.grad(loss_fn)(model)
```

### Multi-Device (pmap/sharding)

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Create device mesh
devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('data', 'model'))

# Shard model parameters
sharding = NamedSharding(mesh, P('model', None))
model_sharded = jax.device_put(model, sharding)

# Sharded training step
@nnx.jit
def sharded_train_step(optimizer, batch):
    # JAX handles communication automatically
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss
```

---

## 8. Checkpointing

```python
import orbax.checkpoint as ocp

# Checkpoint model
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save(
    '/tmp/checkpoint',
    {'model': nnx.state(model), 'optimizer': optimizer.opt_state}
)

# Restore model
ckpt = checkpointer.restore('/tmp/checkpoint')
model = nnx.merge(model, ckpt['model'])
optimizer.opt_state = ckpt['optimizer']
```

---

## 9. Common Patterns

### Model with Config

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout_rate: float = 0.1

class ConfigurableModel(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(784 if i == 0 else config.hidden_dim, config.hidden_dim, rngs=rngs)
            for i in range(config.num_layers)
        ]
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.output = nnx.Linear(config.hidden_dim, 10, rngs=rngs)

    def __call__(self, x, train: bool = False):
        for layer in self.layers:
            x = layer(x)
            x = nnx.relu(x)
            if train:
                x = self.dropout(x)
        return self.output(x)

# Usage
config = ModelConfig(hidden_dim=256, num_layers=4)
model = ConfigurableModel(config, rngs=nnx.Rngs(0))
```

### Ensemble Models

```python
class Ensemble(nnx.Module):
    def __init__(self, n_models: int, rngs: nnx.Rngs):
        self.models = [
            MLP(128, 10, rngs=nnx.Rngs(i))
            for i in range(n_models)
        ]

    def __call__(self, x):
        predictions = [model(x) for model in self.models]
        return jnp.mean(jnp.stack(predictions), axis=0)
```

---

## 10. Migration from Linen

```python
# Linen model
class OldLinenModel(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(128)(x)
        x = nn.Dropout(0.1, deterministic=not train)(x)
        x = nn.Dense(10)(x)
        return x

# NNX equivalent
class NewNNXModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(784, 128, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.dense2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, x, train: bool = False):
        x = self.dense1(x)
        if train:
            x = self.dropout(x)
        x = self.dense2(x)
        return x
```

---

## References

- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/index.html)
- [NNX vs Linen Guide](https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/setup.html)
- [Flax Examples](https://github.com/google/flax/tree/main/examples)
