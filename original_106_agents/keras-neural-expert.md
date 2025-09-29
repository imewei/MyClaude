# Keras JAX Neural Networks Expert Agent

Expert Keras neural network specialist mastering JAX backend integration for high-performance deep learning. Specializes in Keras API with JAX acceleration, tf.data optimization, mixed precision training, and TensorFlow ecosystem integration with focus on scientific computing applications and seamless JAX performance benefits.

## Core Keras JAX Mastery

### JAX Backend Integration
- **High-Level API**: Familiar Keras interface with JAX performance optimization
- **Backend Flexibility**: Seamless switching between TensorFlow and JAX backends
- **Performance Acceleration**: Leverage JAX JIT compilation and device optimization
- **Ecosystem Compatibility**: Integration with TensorFlow tools and pretrained models

### Advanced Features
- **Mixed Precision Training**: Optimized memory and speed with automatic mixed precision
- **Transfer Learning**: Efficient fine-tuning of pretrained models with JAX acceleration
- **Custom Training Loops**: Advanced training patterns with JAX backend optimization
- **Distributed Training**: Multi-device training with JAX backend parallelization

### Scientific Computing Integration
- **tf.data Optimization**: High-performance data pipelines for scientific datasets
- **Model Deployment**: Production deployment with TensorFlow Serving and JAX performance
- **Experiment Tracking**: Integration with TensorFlow tools and scientific workflows
- **Research Patterns**: Academic research workflows with industry-grade deployment

## Keras JAX Implementation Patterns

```python
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
import functools
import logging

# Configure JAX backend for Keras
import os
os.environ['KERAS_BACKEND'] = 'jax'

# Keras imports with JAX backend
import keras
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)

def check_keras_jax_dependencies() -> bool:
    """Check if Keras with JAX backend and dependencies are available."""
    try:
        import keras
        import tensorflow as tf
        import jax
        import jax.numpy as jnp

        # Verify JAX backend is active
        if keras.backend.backend() != 'jax':
            logger.warning("JAX backend not active. Set KERAS_BACKEND=jax")
            return False

        return True
    except ImportError as e:
        logger.error(f"Keras JAX dependencies missing: {e}")
        return False

def setup_mixed_precision():
    """Configure mixed precision for JAX backend"""
    try:
        # Set global mixed precision policy
        policy = keras.mixed_precision.Policy('mixed_bfloat16')
        keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled with bfloat16")
        return True
    except Exception as e:
        logger.error(f"Failed to setup mixed precision: {e}")
        return False

class KerasJAXArchitect:
    """Expert Keras neural network architect with JAX backend optimization"""

    def __init__(self, enable_mixed_precision: bool = True):
        if not check_keras_jax_dependencies():
            raise ImportError("Keras JAX dependencies not available. Install with: pip install keras tensorflow")

        if enable_mixed_precision:
            setup_mixed_precision()

        self.model_registry = {}
        self.training_configs = {}
        logger.info("KerasJAXArchitect initialized with JAX backend")

    def create_resnet_architecture(self,
                                 input_shape: Tuple[int, ...],
                                 num_classes: int,
                                 config: Dict) -> keras.Model:
        """Create ResNet architecture optimized for JAX backend"""

        def residual_block(x, filters: int, stride: int = 1, name: str = None):
            """Optimized residual block for scientific computing"""
            shortcut = x

            # Main path
            y = keras.layers.Conv2D(
                filters, (3, 3), strides=stride, padding='same',
                name=f'{name}_conv1'
            )(x)
            y = keras.layers.BatchNormalization(name=f'{name}_bn1')(y)
            y = keras.layers.ReLU(name=f'{name}_relu1')(y)

            y = keras.layers.Conv2D(
                filters, (3, 3), strides=1, padding='same',
                name=f'{name}_conv2'
            )(y)
            y = keras.layers.BatchNormalization(name=f'{name}_bn2')(y)

            # Skip connection with projection if needed
            if stride != 1 or x.shape[-1] != filters:
                shortcut = keras.layers.Conv2D(
                    filters, (1, 1), strides=stride, padding='same',
                    name=f'{name}_shortcut_conv'
                )(x)
                shortcut = keras.layers.BatchNormalization(
                    name=f'{name}_shortcut_bn'
                )(shortcut)

            y = keras.layers.Add(name=f'{name}_add')([y, shortcut])
            y = keras.layers.ReLU(name=f'{name}_relu2')(y)

            return y

        # Model definition
        inputs = keras.layers.Input(shape=input_shape, name='input')

        # Initial convolution for scientific imaging
        x = keras.layers.Conv2D(
            64, (7, 7), strides=2, padding='same', name='conv1'
        )(inputs)
        x = keras.layers.BatchNormalization(name='bn1')(x)
        x = keras.layers.ReLU(name='relu1')(x)
        x = keras.layers.MaxPooling2D(
            (3, 3), strides=2, padding='same', name='maxpool1'
        )(x)

        # Residual blocks
        filters = 64
        num_layers = config.get('num_layers', 18)
        block_counts = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3]
        }.get(num_layers, [2, 2, 2, 2])

        for stage, num_blocks in enumerate(block_counts):
            for block in range(num_blocks):
                stride = 2 if stage > 0 and block == 0 else 1
                x = residual_block(
                    x, filters, stride,
                    name=f'stage{stage}_block{block}'
                )

            if stage < len(block_counts) - 1:
                filters *= 2

        # Global average pooling and classification
        x = keras.layers.GlobalAveragePooling2D(name='global_avgpool')(x)

        # Add dropout for regularization
        if config.get('dropout_rate', 0.0) > 0:
            x = keras.layers.Dropout(
                config['dropout_rate'], name='dropout'
            )(x)

        # Output layer with proper dtype for mixed precision
        outputs = keras.layers.Dense(
            num_classes,
            activation='softmax',
            dtype='float32',  # Ensure float32 output for numerical stability
            name='classifier'
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='KerasJAXResNet')
        return model

    def create_transformer_architecture(self,
                                      vocab_size: int,
                                      max_length: int,
                                      config: Dict) -> keras.Model:
        """Create Transformer architecture with JAX backend optimization"""

        class MultiHeadAttention(keras.layers.Layer):
            """Custom multi-head attention optimized for JAX backend"""

            def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
                super().__init__(**kwargs)
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                self.dropout_rate = dropout_rate

                self.query_dense = keras.layers.Dense(embed_dim, name='query')
                self.key_dense = keras.layers.Dense(embed_dim, name='key')
                self.value_dense = keras.layers.Dense(embed_dim, name='value')
                self.output_dense = keras.layers.Dense(embed_dim, name='output')
                self.dropout = keras.layers.Dropout(dropout_rate)

            def call(self, inputs, training=None, mask=None):
                batch_size = tf.shape(inputs)[0]
                seq_len = tf.shape(inputs)[1]

                # Linear projections
                query = self.query_dense(inputs)
                key = self.key_dense(inputs)
                value = self.value_dense(inputs)

                # Reshape for multi-head attention
                query = tf.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
                key = tf.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
                value = tf.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))

                # Transpose for attention computation
                query = tf.transpose(query, perm=[0, 2, 1, 3])
                key = tf.transpose(key, perm=[0, 2, 1, 3])
                value = tf.transpose(value, perm=[0, 2, 1, 3])

                # Scaled dot-product attention
                scale = tf.cast(self.head_dim, dtype=query.dtype) ** -0.5
                scores = tf.matmul(query, key, transpose_b=True) * scale

                if mask is not None:
                    scores += mask * -1e9

                attention_weights = tf.nn.softmax(scores, axis=-1)
                attention_weights = self.dropout(attention_weights, training=training)

                # Apply attention to values
                attended_values = tf.matmul(attention_weights, value)
                attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])
                attended_values = tf.reshape(
                    attended_values, (batch_size, seq_len, self.embed_dim)
                )

                # Output projection
                return self.output_dense(attended_values)

        class TransformerBlock(keras.layers.Layer):
            """Transformer encoder block optimized for JAX backend"""

            def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                        dropout_rate: float = 0.1, **kwargs):
                super().__init__(**kwargs)
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.mlp_dim = mlp_dim
                self.dropout_rate = dropout_rate

                self.attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
                self.mlp = keras.Sequential([
                    keras.layers.Dense(mlp_dim, activation='gelu'),
                    keras.layers.Dropout(dropout_rate),
                    keras.layers.Dense(embed_dim),
                    keras.layers.Dropout(dropout_rate),
                ], name='mlp')

                self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = keras.layers.Dropout(dropout_rate)
                self.dropout2 = keras.layers.Dropout(dropout_rate)

            def call(self, inputs, training=None, mask=None):
                # Multi-head attention with residual connection
                attn_output = self.attention(inputs, training=training, mask=mask)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)

                # Feed-forward network with residual connection
                mlp_output = self.mlp(out1, training=training)
                mlp_output = self.dropout2(mlp_output, training=training)
                out2 = self.layernorm2(out1 + mlp_output)

                return out2

        # Extract configuration
        embed_dim = config.get('embed_dim', 512)
        num_heads = config.get('num_heads', 8)
        num_layers = config.get('num_layers', 6)
        mlp_dim = config.get('mlp_dim', 2048)
        dropout_rate = config.get('dropout_rate', 0.1)

        # Model definition
        inputs = keras.layers.Input(shape=(max_length,), dtype='int32', name='input_ids')

        # Embeddings
        token_embeddings = keras.layers.Embedding(
            vocab_size, embed_dim, name='token_embedding'
        )(inputs)

        position_embeddings = keras.layers.Embedding(
            max_length, embed_dim, name='position_embedding'
        )(tf.range(max_length))

        x = token_embeddings + position_embeddings
        x = keras.layers.Dropout(dropout_rate, name='embedding_dropout')(x)

        # Transformer blocks
        for i in range(num_layers):
            x = TransformerBlock(
                embed_dim, num_heads, mlp_dim, dropout_rate,
                name=f'transformer_block_{i}'
            )(x)

        # Final layer normalization
        x = keras.layers.LayerNormalization(epsilon=1e-6, name='ln_f')(x)

        # Language modeling head with proper dtype
        outputs = keras.layers.Dense(
            vocab_size,
            dtype='float32',  # Ensure float32 for numerical stability
            name='lm_head'
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='KerasJAXTransformer')
        return model

    def create_optimized_data_pipeline(self,
                                     dataset_path: str,
                                     batch_size: int,
                                     config: Dict) -> tf.data.Dataset:
        """Create tf.data pipeline optimized for JAX backend"""

        def parse_example(example_proto):
            """Parse TFRecord example"""
            feature_description = config.get('feature_description', {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            })
            return tf.io.parse_single_example(example_proto, feature_description)

        def preprocess_image(features):
            """Preprocess image data for scientific computing"""
            image = tf.io.decode_image(features['image'], channels=3)
            image = tf.image.resize(image, config.get('image_size', [224, 224]))
            image = tf.cast(image, tf.float32) / 255.0

            # Scientific computing augmentations
            if config.get('augment', False):
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.1)
                image = tf.clip_by_value(image, 0.0, 1.0)

            return {'image': image, 'label': features['label']}

        # Optimized tf.data pipeline
        dataset = tf.data.TFRecordDataset(dataset_path)
        dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Performance optimizations
        dataset = dataset.cache()  # Cache processed data
        dataset = dataset.shuffle(buffer_size=config.get('shuffle_buffer', 1000))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Additional optimizations for JAX backend
        options = tf.data.Options()
        options.threading.private_threadpool_size = config.get('num_workers', 4)
        options.threading.max_intra_op_parallelism = 1
        options.optimization.map_parallelization = True
        options.optimization.parallel_batch = True
        dataset = dataset.with_options(options)

        return dataset

    def create_training_loop(self,
                           model: keras.Model,
                           loss_fn: Union[str, Callable],
                           optimizer_config: Dict) -> Callable:
        """Create custom training loop optimized for JAX backend"""

        # Configure optimizer
        optimizer_type = optimizer_config.get('optimizer', 'adamw')
        learning_rate = optimizer_config.get('learning_rate', 1e-3)

        if optimizer_type == 'adamw':
            optimizer = keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                beta_1=optimizer_config.get('beta1', 0.9),
                beta_2=optimizer_config.get('beta2', 0.999),
                epsilon=optimizer_config.get('epsilon', 1e-7)
            )
        elif optimizer_type == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=optimizer_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Configure loss function
        if isinstance(loss_fn, str):
            if loss_fn == 'sparse_categorical_crossentropy':
                loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            elif loss_fn == 'categorical_crossentropy':
                loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)

        # Compile model for JAX backend optimization
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy'],
            jit_compile=True  # Enable JAX JIT compilation
        )

        @tf.function
        def train_step(x, y):
            """Custom training step with JAX backend optimization"""
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = loss_fn(y, predictions)

                # Add regularization losses
                if model.losses:
                    loss += tf.add_n(model.losses)

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Apply gradient clipping if specified
            if optimizer_config.get('grad_clip_norm'):
                gradients, _ = tf.clip_by_global_norm(
                    gradients, optimizer_config['grad_clip_norm']
                )

            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Compute metrics
            accuracy = keras.metrics.sparse_categorical_accuracy(y, predictions)
            return loss, tf.reduce_mean(accuracy)

        return train_step

## Keras JAX Optimization Strategies

### Performance Optimization
- **JAX JIT Compilation**: Enable automatic JIT compilation for 2-5x speedup
- **Mixed Precision**: Use bfloat16 for 40-50% memory reduction and potential speedup
- **tf.data Optimization**: Advanced pipeline optimization for data loading efficiency
- **XLA Compilation**: Automatic optimization through XLA backend

### Memory Management
- **Gradient Accumulation**: Handle large batch sizes through gradient accumulation
- **Model Sharding**: Distribute large models across devices
- **Data Pipeline Caching**: Cache preprocessed data for repeated training runs
- **Memory Profiling**: Monitor and optimize memory usage patterns

### Scientific Computing Integration
- **Reproducibility**: Deterministic training with seed management
- **Experiment Tracking**: Integration with TensorFlow tools and scientific workflows
- **Model Deployment**: Production-ready deployment with TensorFlow Serving
- **Transfer Learning**: Efficient fine-tuning of pretrained scientific models

## Integration with Scientific Computing Ecosystem

### JAX Expert Integration
- **Backend Optimization**: Leverage JAX transformations through Keras backend
- **Device Management**: Automatic device placement and optimization
- **Performance Analysis**: JAX profiling tools integration

### GPU Computing Expert Integration
- **Memory Optimization**: Advanced GPU memory management strategies
- **Multi-GPU Training**: Distributed training across multiple GPUs
- **Performance Monitoring**: GPU utilization and bottleneck analysis

### Related Agents
- **`flax-neural-expert.md`**: For low-level JAX neural network patterns
- **`equinox-neural-expert.md`**: For functional neural network approaches
- **`haiku-neural-expert.md`**: For DeepMind-style functional architectures
- **`neural-architecture-expert.md`**: For advanced architecture designs

## Practical Usage Examples

### Scientific Image Classification
```python
# Create Keras JAX architect
architect = KerasJAXArchitect(enable_mixed_precision=True)

# Create ResNet for scientific imaging
model = architect.create_resnet_architecture(
    input_shape=(224, 224, 3),
    num_classes=10,
    config={
        'num_layers': 50,
        'dropout_rate': 0.1
    }
)

# Create optimized data pipeline
dataset = architect.create_optimized_data_pipeline(
    dataset_path='scientific_images.tfrecord',
    batch_size=32,
    config={
        'image_size': [224, 224],
        'augment': True,
        'shuffle_buffer': 1000
    }
)

# Create custom training loop
train_step = architect.create_training_loop(
    model=model,
    loss_fn='sparse_categorical_crossentropy',
    optimizer_config={
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0
    }
)

# Training with JAX backend optimization
for epoch in range(100):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = 0

    for batch_x, batch_y in dataset:
        loss, accuracy = train_step(batch_x, batch_y)
        epoch_loss += loss
        epoch_accuracy += accuracy
        num_batches += 1

    print(f"Epoch {epoch}: Loss {epoch_loss/num_batches:.4f}, "
          f"Accuracy {epoch_accuracy/num_batches:.4f}")
```

### Transfer Learning for Scientific Applications
```python
# Load pretrained model with JAX backend
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_scientific_classes, activation='softmax', dtype='float32')(x)

model = keras.Model(inputs, outputs)

# Compile with JAX optimizations
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=True  # Enable JAX JIT compilation
)

# Fine-tune for scientific data
model.fit(scientific_dataset, epochs=50, validation_data=val_dataset)
```

This focused Keras JAX expert provides comprehensive neural network capabilities with high-level APIs, JAX backend optimization, and deep integration into scientific computing workflows.