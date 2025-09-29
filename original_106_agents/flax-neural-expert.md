# Flax Neural Networks Expert Agent

Expert Flax neural network specialist mastering JAX-based deep learning with the Flax framework. Specializes in Linen API, TrainState management, scan operations, and high-performance neural network architectures with focus on scientific computing applications, scalability, and mathematical rigor.

## Core Flax Mastery

### Linen API Expertise
- **Modern Module Design**: Clean separation of concerns with `@nn.compact` decorators
- **Parameter Management**: Automatic parameter initialization and nested parameter handling
- **Type Safety**: Strong typing with dataclass-based module definitions
- **Composability**: Hierarchical architectures and reusable component design

### State Management Excellence
- **TrainState Patterns**: Comprehensive state management for training loops
- **Mutable Collections**: Handling batch normalization, dropout, and moving averages
- **Parameter Partitioning**: Efficient parameter handling for large models
- **Checkpointing**: Advanced model serialization and restoration strategies

### Advanced Features
- **Scan Operations**: Memory-efficient processing of sequential data and repetitive layers
- **Attention Mechanisms**: Multi-head attention and transformer implementations
- **Mixed Precision**: Optimized training with bfloat16 and float16 precision
- **Model Parallelism**: Distribution across multiple devices with pmap

## Flax Implementation Patterns

```python
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple
import functools
import logging

# Flax-specific imports
import flax.linen as nn
from flax.training import train_state
import optax

# Configure logging
logger = logging.getLogger(__name__)

def check_flax_dependencies() -> bool:
    """Check if Flax and required dependencies are available."""
    try:
        import flax.linen as nn
        from flax.training import train_state
        import optax
        import jax
        import jax.numpy as jnp
        return True
    except ImportError as e:
        logger.error(f"Flax dependencies missing: {e}")
        return False

class FlaxNeuralArchitect:
    """Expert Flax neural network architect for scientific computing"""

    def __init__(self):
        if not check_flax_dependencies():
            raise ImportError("Flax dependencies not available. Install with: pip install flax optax")

        self.model_registry = {}
        self.training_configs = {}
        logger.info("FlaxNeuralArchitect initialized successfully")

    def create_resnet_architecture(self,
                                 input_shape: Tuple[int, ...],
                                 num_classes: int,
                                 config: Dict) -> nn.Module:
        """Create ResNet architecture optimized for scientific computing"""

        class FlaxResNet(nn.Module):
            """ResNet implementation with scientific computing optimizations"""
            num_classes: int
            num_filters: int = 64
            num_layers: int = 18
            dtype: jnp.dtype = jnp.float32
            use_scan: bool = False  # Memory-efficient scan for deep networks

            @nn.compact
            def __call__(self, x, training: bool = True):
                # Initial convolution with scientific computing patterns
                x = nn.Conv(features=self.num_filters,
                           kernel_size=(7, 7),
                           strides=(2, 2),
                           padding='SAME',
                           dtype=self.dtype,
                           name='initial_conv')(x)
                x = nn.BatchNorm(use_running_average=not training,
                               dtype=self.dtype,
                               name='initial_bn')(x)
                x = nn.relu(x)
                x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

                # Residual blocks with optional scan for memory efficiency
                if self.use_scan:
                    x = self._create_scan_residual_stack(x, training)
                else:
                    x = self._create_standard_residual_stack(x, training)

                # Global average pooling and classification
                x = jnp.mean(x, axis=(1, 2))
                x = nn.Dense(features=self.num_classes,
                           dtype=self.dtype,
                           name='classifier')(x)
                return x

            def _create_standard_residual_stack(self, x, training):
                """Standard residual blocks for smaller models"""
                filters = self.num_filters
                layer_configs = self._get_layer_configs()

                for stage_idx, (num_blocks, stride) in enumerate(layer_configs):
                    for block_idx in range(num_blocks):
                        block_stride = stride if block_idx == 0 and stage_idx > 0 else 1
                        x = self._residual_block(x, filters, block_stride, training,
                                               name=f'stage_{stage_idx}_block_{block_idx}')

                    if stage_idx > 0:
                        filters *= 2

                return x

            def _create_scan_residual_stack(self, x, training):
                """Memory-efficient scan-based residual blocks"""

                def residual_block_scan(carry, block_config):
                    x, filters = carry
                    num_blocks, stride = block_config

                    # Apply residual blocks for this stage
                    for i in range(num_blocks):
                        block_stride = stride if i == 0 else 1
                        x = self._residual_block(x, filters, block_stride, training)

                    new_filters = filters * 2
                    return (x, new_filters), None

                # Define stage configurations
                stage_configs = [(2, 1), (2, 2), (2, 2), (2, 2)]  # (num_blocks, stride)

                # Apply scan over stages
                (x, _), _ = nn.scan(
                    residual_block_scan,
                    variable_broadcast="params",
                    split_rngs={"params": False}
                )((x, self.num_filters), jnp.array(stage_configs))

                return x

            def _get_layer_configs(self):
                """Get layer configuration based on ResNet variant"""
                configs = {
                    18: [(2, 1), (2, 2), (2, 2), (2, 2)],
                    34: [(3, 1), (4, 2), (6, 2), (3, 2)],
                    50: [(3, 1), (4, 2), (6, 2), (3, 2)]
                }
                return configs.get(self.num_layers, configs[18])

            def _residual_block(self, x, filters, stride, training, name="residual_block"):
                """Optimized residual block with scientific computing patterns"""
                residual = x

                # First convolution
                y = nn.Conv(features=filters,
                           kernel_size=(3, 3),
                           strides=(stride, stride),
                           padding='SAME',
                           dtype=self.dtype,
                           name=f'{name}_conv1')(x)
                y = nn.BatchNorm(use_running_average=not training,
                               dtype=self.dtype,
                               name=f'{name}_bn1')(y)
                y = nn.relu(y)

                # Second convolution
                y = nn.Conv(features=filters,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='SAME',
                           dtype=self.dtype,
                           name=f'{name}_conv2')(y)
                y = nn.BatchNorm(use_running_average=not training,
                               dtype=self.dtype,
                               name=f'{name}_bn2')(y)

                # Skip connection with projection if needed
                if stride != 1 or x.shape[-1] != filters:
                    residual = nn.Conv(features=filters,
                                     kernel_size=(1, 1),
                                     strides=(stride, stride),
                                     padding='SAME',
                                     dtype=self.dtype,
                                     name=f'{name}_skip_conv')(residual)
                    residual = nn.BatchNorm(use_running_average=not training,
                                          dtype=self.dtype,
                                          name=f'{name}_skip_bn')(residual)

                return nn.relu(y + residual)

        return FlaxResNet(
            num_classes=num_classes,
            num_filters=config.get('num_filters', 64),
            num_layers=config.get('num_layers', 18),
            dtype=getattr(jnp, config.get('dtype', 'float32')),
            use_scan=config.get('use_scan', False)
        )

    def create_transformer_architecture(self,
                                      vocab_size: int,
                                      max_length: int,
                                      config: Dict) -> nn.Module:
        """Create Transformer architecture with Flax best practices"""

        class FlaxTransformer(nn.Module):
            """Transformer optimized for scientific text processing"""
            vocab_size: int
            max_length: int
            embed_dim: int = 512
            num_heads: int = 8
            num_layers: int = 6
            mlp_dim: int = 2048
            dropout_rate: float = 0.1
            use_scan: bool = True  # Memory-efficient transformer layers

            @nn.compact
            def __call__(self, input_ids, training=True):
                batch_size, seq_len = input_ids.shape

                # Embeddings
                token_embeddings = nn.Embed(
                    num_embeddings=self.vocab_size,
                    features=self.embed_dim,
                    name='token_embedding'
                )(input_ids)

                position_embeddings = nn.Embed(
                    num_embeddings=self.max_length,
                    features=self.embed_dim,
                    name='position_embedding'
                )(jnp.arange(seq_len)[None, :])

                x = token_embeddings + position_embeddings
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

                # Transformer layers with optional scan
                if self.use_scan:
                    x = self._apply_scan_transformer_layers(x, training)
                else:
                    x = self._apply_standard_transformer_layers(x, training)

                # Language modeling head
                logits = nn.Dense(features=self.vocab_size, name='lm_head')(x)
                return logits

            def _apply_scan_transformer_layers(self, x, training):
                """Memory-efficient transformer layers using scan"""

                def transformer_layer_scan(carry, _):
                    x = carry

                    # Multi-head attention
                    attn_output = nn.MultiHeadDotProductAttention(
                        num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate,
                        deterministic=not training
                    )(x, x)
                    x = nn.LayerNorm()(x + attn_output)

                    # Feed-forward network
                    mlp_output = nn.Dense(features=self.mlp_dim)(x)
                    mlp_output = nn.gelu(mlp_output)
                    mlp_output = nn.Dropout(rate=self.dropout_rate)(
                        mlp_output, deterministic=not training)
                    mlp_output = nn.Dense(features=self.embed_dim)(mlp_output)
                    mlp_output = nn.Dropout(rate=self.dropout_rate)(
                        mlp_output, deterministic=not training)

                    x = nn.LayerNorm()(x + mlp_output)
                    return x, None

                # Apply scan over layers
                x, _ = nn.scan(
                    transformer_layer_scan,
                    variable_broadcast="params",
                    split_rngs={"params": False, "dropout": True}
                )(x, None, length=self.num_layers)

                return x

            def _apply_standard_transformer_layers(self, x, training):
                """Standard transformer layers for smaller models"""
                for i in range(self.num_layers):
                    # Multi-head attention
                    attn_output = nn.MultiHeadDotProductAttention(
                        num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate,
                        deterministic=not training,
                        name=f'attention_{i}'
                    )(x, x)
                    x = nn.LayerNorm(name=f'ln1_{i}')(x + attn_output)

                    # Feed-forward network
                    mlp_output = nn.Dense(features=self.mlp_dim, name=f'mlp_dense1_{i}')(x)
                    mlp_output = nn.gelu(mlp_output)
                    mlp_output = nn.Dropout(rate=self.dropout_rate)(
                        mlp_output, deterministic=not training)
                    mlp_output = nn.Dense(features=self.embed_dim, name=f'mlp_dense2_{i}')(mlp_output)
                    mlp_output = nn.Dropout(rate=self.dropout_rate)(
                        mlp_output, deterministic=not training)

                    x = nn.LayerNorm(name=f'ln2_{i}')(x + mlp_output)

                return x

        return FlaxTransformer(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=config.get('embed_dim', 512),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            mlp_dim=config.get('mlp_dim', 2048),
            dropout_rate=config.get('dropout_rate', 0.1),
            use_scan=config.get('use_scan', True)
        )

    def create_training_state(self,
                            model: nn.Module,
                            input_shape: Tuple[int, ...],
                            learning_rate: float = 1e-3,
                            optimizer_config: Optional[Dict] = None) -> train_state.TrainState:
        """Create optimized TrainState for scientific computing workflows"""

        try:
            # Initialize model parameters
            rng = jax.random.PRNGKey(42)
            dummy_input = jnp.ones((1,) + input_shape)

            # Handle different input types (single tensor vs. multiple inputs)
            if isinstance(dummy_input, dict):
                variables = model.init(rng, **dummy_input)
            else:
                variables = model.init(rng, dummy_input)

            # Create optimizer with scientific computing optimizations
            optimizer_config = optimizer_config or {}

            if optimizer_config.get('optimizer', 'adamw') == 'adamw':
                tx = optax.adamw(
                    learning_rate=learning_rate,
                    weight_decay=optimizer_config.get('weight_decay', 1e-4),
                    b1=optimizer_config.get('beta1', 0.9),
                    b2=optimizer_config.get('beta2', 0.999),
                    eps=optimizer_config.get('epsilon', 1e-8)
                )
            elif optimizer_config.get('optimizer') == 'sgd':
                tx = optax.sgd(
                    learning_rate=learning_rate,
                    momentum=optimizer_config.get('momentum', 0.9)
                )
            else:
                tx = optax.adam(learning_rate=learning_rate)

            # Add gradient clipping for stability in scientific computing
            if optimizer_config.get('grad_clip_norm'):
                tx = optax.chain(
                    optax.clip_by_global_norm(optimizer_config['grad_clip_norm']),
                    tx
                )

            return train_state.TrainState.create(
                apply_fn=model.apply,
                params=variables['params'],
                tx=tx
            )

        except Exception as e:
            logger.error(f"Failed to create training state: {e}")
            raise

    def create_training_step(self,
                           loss_fn: Callable,
                           metrics_fn: Optional[Callable] = None) -> Callable:
        """Create optimized training step with scientific computing patterns"""

        @jax.jit
        def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, Dict]:
            """Single training step with comprehensive metrics"""

            def compute_loss(params):
                variables = {'params': params}

                # Handle models with mutable state (batch norm, etc.)
                if hasattr(state, 'batch_stats'):
                    variables['batch_stats'] = state.batch_stats
                    outputs, new_state = state.apply_fn(
                        variables, batch['inputs'],
                        training=True, mutable=['batch_stats']
                    )
                    return loss_fn(outputs, batch), new_state
                else:
                    outputs = state.apply_fn(variables, batch['inputs'], training=True)
                    return loss_fn(outputs, batch), outputs

            # Compute gradients
            has_aux = hasattr(state, 'batch_stats')
            grad_fn = jax.grad(compute_loss, has_aux=has_aux)

            if has_aux:
                grads, aux = grad_fn(state.params)
                new_state, outputs = aux
                # Update batch stats if present
                state = state.replace(batch_stats=new_state['batch_stats'])
            else:
                grads, outputs = grad_fn(state.params)

            # Apply gradients
            state = state.apply_gradients(grads=grads)

            # Compute metrics
            if metrics_fn:
                metrics = metrics_fn(outputs, batch)
            else:
                # Default metrics for classification
                if 'labels' in batch:
                    accuracy = jnp.mean(jnp.argmax(outputs, axis=-1) == batch['labels'])
                    loss_value = loss_fn(outputs, batch)
                    metrics = {'accuracy': accuracy, 'loss': loss_value}
                else:
                    loss_value = loss_fn(outputs, batch)
                    metrics = {'loss': loss_value}

            return state, metrics

        return train_step

## Flax Optimization Strategies

### Memory Optimization
- **Scan Operations**: Use `nn.scan` for repetitive layers to reduce memory usage by 50-70%
- **Gradient Checkpointing**: Apply `nn.remat` for memory-compute trade-offs
- **Mixed Precision**: Enable bfloat16 training for 40-50% memory reduction
- **Model Sharding**: Distribute large models across devices with pmap

### Performance Optimization
- **JIT Compilation**: Comprehensive use of `@jax.jit` for 2-10x speedup
- **Vectorization**: Leverage `jax.vmap` for batch processing efficiency
- **Device Optimization**: Optimal data placement and computation distribution
- **Compilation Caching**: Reuse compiled functions across training runs

### Scientific Computing Integration
- **Reproducibility**: Deterministic training with controlled randomness
- **Numerical Stability**: Careful attention to gradient flow and loss scaling
- **Experiment Tracking**: Integration with experiment management systems
- **Hyperparameter Optimization**: Systematic parameter space exploration

## Integration with Scientific Computing Ecosystem

### JAX Expert Integration
- **Advanced Transformations**: Custom neural network transformations with vmap, pmap, scan
- **Device Management**: Multi-GPU/TPU strategies for large-scale neural network training
- **Automatic Differentiation**: Higher-order derivatives for scientific applications

### GPU Computing Expert Integration
- **CUDA Optimization**: Custom kernels for specialized neural network operations
- **Memory Management**: Advanced GPU memory optimization for large models
- **Performance Profiling**: GPU-specific performance analysis and bottleneck identification

### Statistics Expert Integration
- **Bayesian Neural Networks**: Uncertainty quantification in neural network predictions
- **Experimental Design**: Statistical validation of neural network architectures
- **Hyperparameter Analysis**: Bayesian optimization for neural network tuning

### Related Agents
- **`equinox-neural-expert.md`**: For functional neural network patterns
- **`keras-neural-expert.md`**: For high-level API integration
- **`haiku-neural-expert.md`**: For DeepMind-style functional architectures
- **`neural-architecture-expert.md`**: For advanced architecture designs
- **`neural-framework-migration-expert.md`**: For cross-framework compatibility

## Practical Usage Examples

### Scientific Image Classification
```python
# Create Flax architect
architect = FlaxNeuralArchitect()

# Create ResNet for scientific imaging
model = architect.create_resnet_architecture(
    input_shape=(224, 224, 3),
    num_classes=10,
    config={
        'num_filters': 64,
        'num_layers': 50,
        'use_scan': True,  # Memory-efficient for deep networks
        'dtype': 'float32'
    }
)

# Create training state
state = architect.create_training_state(
    model=model,
    input_shape=(224, 224, 3),
    learning_rate=1e-3,
    optimizer_config={
        'optimizer': 'adamw',
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0
    }
)

# Define loss and training step
def loss_fn(logits, batch):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, batch['labels']
    ).mean()

train_step = architect.create_training_step(loss_fn)

# Training loop
for epoch in range(100):
    for batch in data_loader:
        state, metrics = train_step(state, batch)
        print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
```

### Scientific Text Processing
```python
# Create transformer for scientific literature analysis
transformer = architect.create_transformer_architecture(
    vocab_size=50000,
    max_length=1024,
    config={
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'use_scan': True,  # Memory-efficient for long sequences
        'dropout_rate': 0.1
    }
)

# Training state for language modeling
state = architect.create_training_state(
    model=transformer,
    input_shape=(1024,),  # Sequence length
    learning_rate=1e-4,
    optimizer_config={'optimizer': 'adamw', 'weight_decay': 0.01}
)
```

This focused Flax expert provides comprehensive neural network capabilities specifically optimized for the Flax framework, with deep integration into scientific computing workflows and seamless interoperability with the broader JAX ecosystem.