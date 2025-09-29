# Haiku Neural Networks Expert Agent

Expert Haiku neural network specialist mastering DeepMind's functional JAX-based deep learning framework. Specializes in transform systems, functional module patterns, parameter management, and pure functional neural networks with focus on research applications, mathematical rigor, and DeepMind-style functional programming paradigms.

## Core Haiku Mastery

### Transform System Excellence
- **Function Transformation**: Pure functional neural networks with explicit parameter handling
- **Parameter Management**: Clean separation of parameters and state through transforms
- **Module Composition**: Functional module patterns with explicit parameter passing
- **State Handling**: Immutable state management in functional neural networks

### Research-Grade Patterns
- **DeepMind Architectures**: Implementation patterns from cutting-edge research
- **Functional Design**: Pure functions with no hidden state or side effects
- **Mathematical Rigor**: Clean mathematical formulations of neural network operations
- **Reproducibility**: Deterministic computation with explicit randomness handling

### Advanced Capabilities
- **Custom Transformations**: Building novel transformations for specialized architectures
- **Parameter Initialization**: Sophisticated initialization strategies for research
- **Gradient Computation**: Efficient gradient computation through functional design
- **Multi-Scale Architectures**: Hierarchical and multi-resolution neural networks

## Haiku Implementation Patterns

```python
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple, Union, NamedTuple
import functools
import logging

# Haiku-specific imports
import haiku as hk
import optax

# Configure logging
logger = logging.getLogger(__name__)

def check_haiku_dependencies() -> bool:
    """Check if Haiku and required dependencies are available."""
    try:
        import haiku as hk
        import optax
        import jax
        import jax.numpy as jnp
        return True
    except ImportError as e:
        logger.error(f"Haiku dependencies missing: {e}")
        return False

class HaikuNeuralArchitect:
    """Expert Haiku neural network architect for research and scientific computing"""

    def __init__(self):
        if not check_haiku_dependencies():
            raise ImportError("Haiku dependencies not available. Install with: pip install dm-haiku optax")

        self.transform_registry = {}
        self.parameter_configs = {}
        logger.info("HaikuNeuralArchitect initialized successfully")

    def create_resnet_transform(self,
                              num_classes: int,
                              config: Dict) -> Tuple[Callable, Callable]:
        """Create ResNet transform with Haiku functional patterns"""

        def residual_block(x: jnp.ndarray,
                          filters: int,
                          stride: int = 1,
                          name: str = "residual_block") -> jnp.ndarray:
            """Functional residual block implementation"""
            shortcut = x

            # Main path
            y = hk.Conv2D(
                output_channels=filters,
                kernel_shape=3,
                stride=stride,
                padding='SAME',
                name=f'{name}_conv1'
            )(x)
            y = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.999,
                name=f'{name}_bn1'
            )(y, is_training=True)
            y = jax.nn.relu(y)

            y = hk.Conv2D(
                output_channels=filters,
                kernel_shape=3,
                stride=1,
                padding='SAME',
                name=f'{name}_conv2'
            )(y)
            y = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.999,
                name=f'{name}_bn2'
            )(y, is_training=True)

            # Skip connection with projection if needed
            if stride != 1 or x.shape[-1] != filters:
                shortcut = hk.Conv2D(
                    output_channels=filters,
                    kernel_shape=1,
                    stride=stride,
                    padding='SAME',
                    name=f'{name}_shortcut_conv'
                )(x)
                shortcut = hk.BatchNorm(
                    create_scale=True,
                    create_offset=True,
                    decay_rate=0.999,
                    name=f'{name}_shortcut_bn'
                )(shortcut, is_training=True)

            return jax.nn.relu(y + shortcut)

        def resnet_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
            """ResNet forward function with scientific computing optimizations"""

            # Initial convolution
            x = hk.Conv2D(
                output_channels=64,
                kernel_shape=7,
                stride=2,
                padding='SAME',
                name='conv1'
            )(x)
            x = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.999,
                name='bn1'
            )(x, is_training=is_training)
            x = jax.nn.relu(x)
            x = hk.max_pool(x, window_shape=3, strides=2, padding='SAME')

            # Residual blocks
            filters = 64
            num_layers = config.get('num_layers', 18)
            block_counts = {
                18: [2, 2, 2, 2],
                34: [3, 4, 6, 3],
                50: [3, 4, 6, 3]
            }.get(num_layers, [2, 2, 2, 2])

            for stage_idx, num_blocks in enumerate(block_counts):
                for block_idx in range(num_blocks):
                    stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                    x = residual_block(
                        x, filters, stride,
                        name=f'stage{stage_idx}_block{block_idx}'
                    )

                if stage_idx < len(block_counts) - 1:
                    filters *= 2

            # Global average pooling and classification
            x = jnp.mean(x, axis=(1, 2))

            # Apply dropout for regularization
            if config.get('dropout_rate', 0.0) > 0 and is_training:
                x = hk.dropout(
                    hk.next_rng_key(),
                    config['dropout_rate'],
                    x
                )

            # Classification layer
            x = hk.Linear(
                output_size=num_classes,
                name='classifier'
            )(x)

            return x

        # Create Haiku transform
        resnet_transform = hk.transform_with_state(resnet_fn)

        return resnet_transform.init, resnet_transform.apply

    def create_transformer_transform(self,
                                   vocab_size: int,
                                   config: Dict) -> Tuple[Callable, Callable]:
        """Create Transformer transform with Haiku functional patterns"""

        def multi_head_attention(x: jnp.ndarray,
                                mask: Optional[jnp.ndarray] = None,
                                num_heads: int = 8,
                                name: str = "attention") -> jnp.ndarray:
            """Functional multi-head attention implementation"""
            batch_size, seq_len, embed_dim = x.shape
            head_dim = embed_dim // num_heads

            # Linear projections
            query = hk.Linear(embed_dim, name=f'{name}_query')(x)
            key = hk.Linear(embed_dim, name=f'{name}_key')(x)
            value = hk.Linear(embed_dim, name=f'{name}_value')(x)

            # Reshape for multi-head attention
            query = query.reshape(batch_size, seq_len, num_heads, head_dim)
            key = key.reshape(batch_size, seq_len, num_heads, head_dim)
            value = value.reshape(batch_size, seq_len, num_heads, head_dim)

            # Transpose for attention computation
            query = jnp.transpose(query, (0, 2, 1, 3))
            key = jnp.transpose(key, (0, 2, 1, 3))
            value = jnp.transpose(value, (0, 2, 1, 3))

            # Scaled dot-product attention
            scale = 1.0 / jnp.sqrt(head_dim)
            scores = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) * scale

            if mask is not None:
                scores = scores + mask * -1e9

            attention_weights = jax.nn.softmax(scores, axis=-1)

            # Apply attention to values
            attended_values = jnp.matmul(attention_weights, value)
            attended_values = jnp.transpose(attended_values, (0, 2, 1, 3))
            attended_values = attended_values.reshape(batch_size, seq_len, embed_dim)

            # Output projection
            return hk.Linear(embed_dim, name=f'{name}_output')(attended_values)

        def transformer_block(x: jnp.ndarray,
                            mask: Optional[jnp.ndarray] = None,
                            num_heads: int = 8,
                            mlp_dim: int = 2048,
                            dropout_rate: float = 0.1,
                            name: str = "transformer_block") -> jnp.ndarray:
            """Functional transformer block implementation"""

            # Multi-head attention with residual connection
            attn_output = multi_head_attention(
                x, mask, num_heads, name=f'{name}_attention'
            )

            if dropout_rate > 0:
                attn_output = hk.dropout(
                    hk.next_rng_key(), dropout_rate, attn_output
                )

            x = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name=f'{name}_ln1'
            )(x + attn_output)

            # Feed-forward network with residual connection
            mlp_output = hk.Linear(mlp_dim, name=f'{name}_mlp_1')(x)
            mlp_output = jax.nn.gelu(mlp_output)

            if dropout_rate > 0:
                mlp_output = hk.dropout(
                    hk.next_rng_key(), dropout_rate, mlp_output
                )

            mlp_output = hk.Linear(x.shape[-1], name=f'{name}_mlp_2')(mlp_output)

            if dropout_rate > 0:
                mlp_output = hk.dropout(
                    hk.next_rng_key(), dropout_rate, mlp_output
                )

            x = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name=f'{name}_ln2'
            )(x + mlp_output)

            return x

        def transformer_fn(input_ids: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
            """Transformer forward function with Haiku patterns"""
            seq_len = input_ids.shape[-1]
            embed_dim = config.get('embed_dim', 512)
            num_heads = config.get('num_heads', 8)
            num_layers = config.get('num_layers', 6)
            mlp_dim = config.get('mlp_dim', 2048)
            max_length = config.get('max_length', 1024)
            dropout_rate = config.get('dropout_rate', 0.1) if is_training else 0.0

            # Token and position embeddings
            token_embeddings = hk.Embed(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                name='token_embedding'
            )(input_ids)

            position_ids = jnp.arange(seq_len)
            position_embeddings = hk.Embed(
                vocab_size=max_length,
                embed_dim=embed_dim,
                name='position_embedding'
            )(position_ids)

            x = token_embeddings + position_embeddings

            if dropout_rate > 0:
                x = hk.dropout(hk.next_rng_key(), dropout_rate, x)

            # Create causal mask for language modeling
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            mask = jnp.where(mask == 0, -jnp.inf, 0.0)
            mask = mask[None, None, :, :]  # Broadcast for batch and heads

            # Transformer blocks
            for i in range(num_layers):
                x = transformer_block(
                    x, mask, num_heads, mlp_dim, dropout_rate,
                    name=f'transformer_block_{i}'
                )

            # Final layer normalization
            x = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name='ln_f'
            )(x)

            # Language modeling head
            return hk.Linear(vocab_size, name='lm_head')(x)

        # Create Haiku transform
        transformer_transform = hk.transform_with_state(transformer_fn)

        return transformer_transform.init, transformer_transform.apply

    def create_variational_autoencoder_transform(self,
                                               input_shape: Tuple[int, ...],
                                               latent_dim: int,
                                               config: Dict) -> Tuple[Callable, Callable]:
        """Create VAE transform with Haiku functional patterns"""

        def encoder(x: jnp.ndarray, name: str = "encoder") -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Encoder network with reparameterization trick"""

            # Convolutional layers for image data
            if len(input_shape) == 3:  # Image data
                x = hk.Conv2D(32, 4, stride=2, padding='SAME', name=f'{name}_conv1')(x)
                x = jax.nn.relu(x)
                x = hk.Conv2D(64, 4, stride=2, padding='SAME', name=f'{name}_conv2')(x)
                x = jax.nn.relu(x)
                x = hk.Conv2D(128, 4, stride=2, padding='SAME', name=f'{name}_conv3')(x)
                x = jax.nn.relu(x)
                x = hk.Flatten(name=f'{name}_flatten')(x)

            # Dense layers
            hidden_dim = config.get('hidden_dim', 512)
            x = hk.Linear(hidden_dim, name=f'{name}_dense1')(x)
            x = jax.nn.relu(x)
            x = hk.Linear(hidden_dim, name=f'{name}_dense2')(x)
            x = jax.nn.relu(x)

            # Latent distribution parameters
            mu = hk.Linear(latent_dim, name=f'{name}_mu')(x)
            log_var = hk.Linear(latent_dim, name=f'{name}_log_var')(x)

            return mu, log_var

        def decoder(z: jnp.ndarray, name: str = "decoder") -> jnp.ndarray:
            """Decoder network"""
            hidden_dim = config.get('hidden_dim', 512)

            # Dense layers
            x = hk.Linear(hidden_dim, name=f'{name}_dense1')(z)
            x = jax.nn.relu(x)
            x = hk.Linear(hidden_dim, name=f'{name}_dense2')(x)
            x = jax.nn.relu(x)

            if len(input_shape) == 3:  # Image data
                # Calculate dimensions for reshaping
                h, w, c = input_shape
                x = hk.Linear(h * w * c // 64, name=f'{name}_dense3')(x)
                x = jax.nn.relu(x)
                x = x.reshape(-1, h // 8, w // 8, 128)

                # Transposed convolutions
                x = hk.Conv2DTranspose(64, 4, stride=2, padding='SAME', name=f'{name}_deconv1')(x)
                x = jax.nn.relu(x)
                x = hk.Conv2DTranspose(32, 4, stride=2, padding='SAME', name=f'{name}_deconv2')(x)
                x = jax.nn.relu(x)
                x = hk.Conv2DTranspose(c, 4, stride=2, padding='SAME', name=f'{name}_deconv3')(x)
                x = jax.nn.sigmoid(x)
            else:
                # Dense output for non-image data
                x = hk.Linear(np.prod(input_shape), name=f'{name}_output')(x)
                x = jax.nn.sigmoid(x)
                x = x.reshape(-1, *input_shape)

            return x

        def reparameterize(mu: jnp.ndarray, log_var: jnp.ndarray) -> jnp.ndarray:
            """Reparameterization trick for VAE"""
            std = jnp.exp(0.5 * log_var)
            eps = hk.next_rng_key()
            eps = jax.random.normal(eps, shape=mu.shape)
            return mu + eps * std

        def vae_fn(x: jnp.ndarray, is_training: bool = True) -> Dict[str, jnp.ndarray]:
            """VAE forward function"""
            # Encode
            mu, log_var = encoder(x)

            # Reparameterize
            z = reparameterize(mu, log_var)

            # Decode
            x_recon = decoder(z)

            return {
                'reconstruction': x_recon,
                'mu': mu,
                'log_var': log_var,
                'z': z
            }

        # Create Haiku transform
        vae_transform = hk.transform_with_state(vae_fn)

        return vae_transform.init, vae_transform.apply

    def create_training_functions(self,
                                transform_pair: Tuple[Callable, Callable],
                                loss_fn: Callable,
                                optimizer_config: Dict) -> Tuple[Callable, Callable, Callable]:
        """Create training functions with Haiku functional patterns"""

        init_fn, apply_fn = transform_pair

        # Create optimizer
        learning_rate = optimizer_config.get('learning_rate', 1e-3)
        if optimizer_config.get('optimizer', 'adamw') == 'adamw':
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config.get('optimizer') == 'sgd':
            optimizer = optax.sgd(
                learning_rate=learning_rate,
                momentum=optimizer_config.get('momentum', 0.9)
            )
        else:
            optimizer = optax.adam(learning_rate=learning_rate)

        # Add gradient clipping if specified
        if optimizer_config.get('grad_clip_norm'):
            optimizer = optax.chain(
                optax.clip_by_global_norm(optimizer_config['grad_clip_norm']),
                optimizer
            )

        @jax.jit
        def train_step(params: hk.Params,
                      state: hk.State,
                      opt_state: optax.OptState,
                      batch: Dict[str, jnp.ndarray],
                      rng: jax.random.PRNGKey) -> Tuple[hk.Params, hk.State, optax.OptState, Dict]:
            """Single training step with Haiku functional patterns"""

            def compute_loss(params):
                outputs, new_state = apply_fn(params, state, rng, batch['inputs'], is_training=True)
                loss = loss_fn(outputs, batch)
                return loss, (outputs, new_state)

            # Compute gradients
            (loss, (outputs, new_state)), grads = jax.value_and_grad(
                compute_loss, has_aux=True
            )(params)

            # Update parameters
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Compute metrics
            metrics = {'loss': loss}
            if 'targets' in batch:
                if len(outputs.shape) > 1 and outputs.shape[-1] > 1:
                    # Classification accuracy
                    predictions = jnp.argmax(outputs, axis=-1)
                    accuracy = jnp.mean(predictions == batch['targets'])
                    metrics['accuracy'] = accuracy

            return new_params, new_state, new_opt_state, metrics

        @jax.jit
        def eval_step(params: hk.Params,
                     state: hk.State,
                     batch: Dict[str, jnp.ndarray],
                     rng: jax.random.PRNGKey) -> Dict:
            """Evaluation step"""
            outputs, _ = apply_fn(params, state, rng, batch['inputs'], is_training=False)
            loss = loss_fn(outputs, batch)

            metrics = {'loss': loss}
            if 'targets' in batch:
                if len(outputs.shape) > 1 and outputs.shape[-1] > 1:
                    predictions = jnp.argmax(outputs, axis=-1)
                    accuracy = jnp.mean(predictions == batch['targets'])
                    metrics['accuracy'] = accuracy

            return metrics

        def init_training_state(rng: jax.random.PRNGKey,
                              dummy_input: jnp.ndarray) -> Tuple[hk.Params, hk.State, optax.OptState]:
            """Initialize training state"""
            params, state = init_fn(rng, dummy_input, is_training=True)
            opt_state = optimizer.init(params)
            return params, state, opt_state

        return train_step, eval_step, init_training_state

## Haiku Research Patterns

### Parameter Management
- **Explicit Parameter Handling**: Clean separation of parameters from computation
- **Hierarchical Parameters**: Nested parameter structures for complex architectures
- **Parameter Initialization**: Research-grade initialization strategies
- **Parameter Sharing**: Efficient sharing of parameters across modules

### State Management
- **Immutable State**: Functional state handling without side effects
- **Batch Normalization State**: Proper handling of running statistics
- **RNN State**: Clean state management for recurrent architectures
- **Multi-Level State**: Hierarchical state management for complex models

### Functional Composition
- **Module Composition**: Building complex architectures from simple functions
- **Transform Composition**: Chaining multiple transformations
- **Custom Transforms**: Creating novel transformation patterns
- **Gradient Customization**: Custom gradient rules for research applications

## Integration with Scientific Computing Ecosystem

### JAX Expert Integration
- **Transform Optimization**: Leverage JAX transformations for performance
- **Device Management**: Multi-device computation with functional patterns
- **Gradient Analysis**: Higher-order gradients for scientific applications

### GPU Computing Expert Integration
- **Memory Efficiency**: Functional design minimizes GPU memory overhead
- **Parallelization**: Easy parallelization through functional patterns
- **Performance Profiling**: Clean profiling through functional design

### Related Agents
- **`flax-neural-expert.md`**: For module-based neural network patterns
- **`equinox-neural-expert.md`**: For alternative functional approaches
- **`keras-neural-expert.md`**: For high-level API integration
- **`neural-architecture-expert.md`**: For advanced architecture designs

## Practical Usage Examples

### Research Classification Model
```python
# Create Haiku architect
architect = HaikuNeuralArchitect()

# Create ResNet transform
init_fn, apply_fn = architect.create_resnet_transform(
    num_classes=10,
    config={
        'num_layers': 18,
        'dropout_rate': 0.1
    }
)

# Define loss function for research
def research_loss(outputs, batch):
    # Standard cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        outputs, batch['targets']
    ).mean()

    # Add research-specific regularization
    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(outputs))
    return loss + 1e-4 * l2_loss

# Create training functions
train_step, eval_step, init_training_state = architect.create_training_functions(
    transform_pair=(init_fn, apply_fn),
    loss_fn=research_loss,
    optimizer_config={
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0
    }
)

# Initialize training
rng = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 224, 224, 3))
params, state, opt_state = init_training_state(rng, dummy_input)

# Training loop
for epoch in range(100):
    for batch in data_loader:
        rng, step_rng = jax.random.split(rng)
        params, state, opt_state, metrics = train_step(
            params, state, opt_state, batch, step_rng
        )
        print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")
```

### Scientific Transformer Research
```python
# Create transformer for scientific text research
init_fn, apply_fn = architect.create_transformer_transform(
    vocab_size=50000,
    config={
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'max_length': 1024,
        'dropout_rate': 0.1
    }
)

# Research-specific loss with regularization
def scientific_loss(outputs, batch):
    # Language modeling loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        outputs[:, :-1].reshape(-1, outputs.shape[-1]),
        batch['targets'][:, 1:].reshape(-1)
    ).mean()

    return loss

# Functional training with parameter analysis
train_step, eval_step, init_training_state = architect.create_training_functions(
    transform_pair=(init_fn, apply_fn),
    loss_fn=scientific_loss,
    optimizer_config={'learning_rate': 1e-4}
)
```

This focused Haiku expert provides comprehensive functional neural network capabilities with research-grade patterns, mathematical rigor, and deep integration into scientific computing workflows using DeepMind's functional programming paradigms.