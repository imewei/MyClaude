# Neural Network Data Loading Expert Agent

Expert neural network data loading specialist mastering JAX-based deep learning frameworks: Flax (high-level modules with Linen API), Equinox (functional PyTorch-like), Keras (high-level JAX backend), and Haiku (functional DeepMind library). Specializes in high-performance training pipelines, memory-efficient batching, and framework-optimized data loading patterns with focus on scalability, performance, and seamless JAX ecosystem integration.

## Core Neural Network Data Loading Mastery

### JAX-Native Data Pipeline Design
- **Device-Optimized Loading**: Direct GPU/TPU data transfer with minimal host-device communication
- **JIT-Compatible Pipelines**: Data loading patterns compatible with JAX transformations (jit, vmap, pmap)
- **Memory Management**: Efficient memory usage for large datasets exceeding device memory
- **Automatic Batching**: Framework-aware batching strategies for optimal training performance
- **Deterministic Training**: Reproducible data loading with controlled randomness and seeding

### Framework-Specific Optimization
- **Flax Integration**: Linen module compatibility, state management, and training loop optimization
- **Equinox Patterns**: Functional data loading workflows with PyTree compatibility
- **Keras Adaptation**: tf.data integration and JAX backend optimization
- **Haiku Workflows**: DeepMind patterns for functional neural network training

### Performance & Scalability
- **Distributed Training**: Multi-device data sharding and synchronization
- **Streaming Datasets**: Memory-efficient processing of datasets larger than available memory
- **Prefetching & Caching**: Advanced caching strategies and asynchronous data loading
- **Mixed Precision**: Data loading optimization for bf16/fp16 training workflows
- **Dynamic Batching**: Adaptive batch sizing for variable-length sequences and memory optimization

## Framework-Specific Data Loading Implementation

```python
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Iterator, Union, Tuple
import functools
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

# Framework imports
import flax.linen as nn
from flax.training import train_state
import equinox as eqx
import tensorflow as tf
import haiku as hk
import optax

class NeuralDataLoader:
    """High-performance data loading for JAX-based neural network frameworks"""

    def __init__(self,
                 framework: str = 'flax',
                 device_count: Optional[int] = None,
                 batch_size: int = 32,
                 prefetch_size: int = 2,
                 num_workers: int = 4,
                 enable_mixed_precision: bool = False,
                 cache_size_gb: float = 2.0):

        self.framework = framework.lower()
        self.device_count = device_count or jax.device_count()
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.enable_mixed_precision = enable_mixed_precision
        self.cache_size_bytes = cache_size_gb * 1024**3

        # Initialize framework-specific settings
        self._setup_framework_config()
        self._setup_performance_tracking()

    def _setup_framework_config(self):
        """Configure framework-specific settings"""
        self.framework_config = {
            'flax': {
                'requires_state': True,
                'batch_axis': 0,
                'preferred_dtype': jnp.float32,
                'supports_scan': True
            },
            'equinox': {
                'requires_state': False,
                'batch_axis': 0,
                'preferred_dtype': jnp.float32,
                'functional': True
            },
            'keras': {
                'requires_state': True,
                'batch_axis': 0,
                'preferred_dtype': jnp.float32,
                'tf_data_compatible': True
            },
            'haiku': {
                'requires_state': True,
                'batch_axis': 0,
                'preferred_dtype': jnp.float32,
                'functional': True
            }
        }

    def _setup_performance_tracking(self):
        """Initialize performance monitoring"""
        self.metrics = {
            'batches_processed': 0,
            'total_samples': 0,
            'avg_batch_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def create_image_pipeline(self,
                            image_paths: List[str],
                            labels: Optional[List[int]] = None,
                            image_size: Tuple[int, int] = (224, 224),
                            augmentation_config: Optional[Dict] = None) -> Iterator[Dict]:
        """Create optimized image classification data pipeline"""

        if self.framework == 'flax':
            return self._create_flax_image_pipeline(image_paths, labels, image_size, augmentation_config)
        elif self.framework == 'equinox':
            return self._create_equinox_image_pipeline(image_paths, labels, image_size, augmentation_config)
        elif self.framework == 'keras':
            return self._create_keras_image_pipeline(image_paths, labels, image_size, augmentation_config)
        elif self.framework == 'haiku':
            return self._create_haiku_image_pipeline(image_paths, labels, image_size, augmentation_config)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def _create_flax_image_pipeline(self, image_paths: List[str], labels: Optional[List[int]],
                                  image_size: Tuple[int, int], augmentation_config: Optional[Dict]) -> Iterator[Dict]:
        """Flax-optimized image data pipeline with Linen compatibility"""

        def load_and_preprocess_image(path: str, label: Optional[int] = None) -> Dict:
            """Load and preprocess single image for Flax"""
            import PIL.Image as Image

            # Load image
            with Image.open(path) as img:
                img = img.convert('RGB')
                img = img.resize(image_size, Image.Resampling.LANCZOS)
                image_array = np.array(img, dtype=np.float32) / 255.0

            # Convert to JAX array with proper device placement
            image_array = jnp.array(image_array)

            # Apply augmentations if specified
            if augmentation_config:
                image_array = self._apply_flax_augmentations(image_array, augmentation_config)

            # Mixed precision handling
            if self.enable_mixed_precision:
                image_array = image_array.astype(jnp.bfloat16)

            batch_item = {'image': image_array}
            if label is not None:
                batch_item['label'] = jnp.array(label, dtype=jnp.int32)

            return batch_item

        def batch_generator():
            """Generate batches for Flax training"""
            data_pairs = list(zip(image_paths, labels or [None] * len(image_paths)))

            # Shuffle for training
            if augmentation_config and augmentation_config.get('shuffle', True):
                rng = np.random.RandomState(42)
                rng.shuffle(data_pairs)

            # Process in batches
            for i in range(0, len(data_pairs), self.batch_size):
                batch_pairs = data_pairs[i:i + self.batch_size]

                # Load batch in parallel
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    batch_items = list(executor.map(
                        lambda x: load_and_preprocess_image(x[0], x[1]),
                        batch_pairs
                    ))

                # Stack into batch arrays
                if batch_items:
                    batch = {
                        'image': jnp.stack([item['image'] for item in batch_items]),
                    }

                    if 'label' in batch_items[0]:
                        batch['label'] = jnp.stack([item['label'] for item in batch_items])

                    # Ensure proper sharding for multi-device training
                    if self.device_count > 1:
                        batch = self._shard_batch(batch)

                    yield batch

                    # Update metrics
                    self.metrics['batches_processed'] += 1
                    self.metrics['total_samples'] += len(batch_items)

        return batch_generator()

    def _create_equinox_image_pipeline(self, image_paths: List[str], labels: Optional[List[int]],
                                     image_size: Tuple[int, int], augmentation_config: Optional[Dict]) -> Iterator[Dict]:
        """Equinox-optimized functional image data pipeline"""

        @jax.jit
        def equinox_preprocess(image: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
            """JIT-compiled preprocessing for Equinox"""

            # Normalize to [0, 1] range
            image = image / 255.0

            # Apply functional augmentations
            if augmentation_config:
                # Random horizontal flip
                if augmentation_config.get('horizontal_flip', False):
                    flip_key, key = jax.random.split(key)
                    should_flip = jax.random.uniform(flip_key) < 0.5
                    image = jax.lax.cond(should_flip,
                                       lambda x: jnp.fliplr(x),
                                       lambda x: x,
                                       image)

                # Random brightness adjustment
                if 'brightness_range' in augmentation_config:
                    bright_key, key = jax.random.split(key)
                    brightness_delta = jax.random.uniform(
                        bright_key,
                        minval=augmentation_config['brightness_range'][0],
                        maxval=augmentation_config['brightness_range'][1]
                    )
                    image = jnp.clip(image + brightness_delta, 0.0, 1.0)

            return image

        def functional_batch_generator():
            """Generate batches with functional preprocessing"""
            rng_key = jax.random.PRNGKey(42)

            data_pairs = list(zip(image_paths, labels or [None] * len(image_paths)))

            for i in range(0, len(data_pairs), self.batch_size):
                batch_pairs = data_pairs[i:i + self.batch_size]

                # Load raw images
                batch_images = []
                batch_labels = []

                for path, label in batch_pairs:
                    import PIL.Image as Image
                    with Image.open(path) as img:
                        img = img.convert('RGB').resize(image_size, Image.Resampling.LANCZOS)
                        img_array = jnp.array(np.array(img))
                        batch_images.append(img_array)
                        if label is not None:
                            batch_labels.append(label)

                # Stack and apply functional preprocessing
                if batch_images:
                    images_batch = jnp.stack(batch_images)

                    # Apply vectorized preprocessing
                    rng_key, *batch_keys = jax.random.split(rng_key, len(batch_images) + 1)
                    batch_keys = jnp.stack(batch_keys)

                    # Use vmap for efficient batch processing
                    vectorized_preprocess = jax.vmap(equinox_preprocess, in_axes=(0, 0))
                    processed_images = vectorized_preprocess(images_batch, batch_keys)

                    batch = {'image': processed_images}
                    if batch_labels:
                        batch['label'] = jnp.array(batch_labels, dtype=jnp.int32)

                    yield batch

        return functional_batch_generator()

    def _create_keras_image_pipeline(self, image_paths: List[str], labels: Optional[List[int]],
                                   image_size: Tuple[int, int], augmentation_config: Optional[Dict]) -> Iterator[Dict]:
        """Keras-optimized pipeline with tf.data integration"""

        # Create tf.data dataset
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

        if labels is not None:
            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((path_ds, label_ds))
        else:
            dataset = path_ds.map(lambda x: (x, -1))  # Dummy label

        def load_and_decode_image(path, label):
            """TensorFlow image loading function"""
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, image_size)
            image = tf.cast(image, tf.float32) / 255.0

            # Apply tf.data augmentations
            if augmentation_config:
                if augmentation_config.get('horizontal_flip', False):
                    image = tf.image.random_flip_left_right(image)
                if 'brightness_range' in augmentation_config:
                    max_delta = augmentation_config['brightness_range'][1] - augmentation_config['brightness_range'][0]
                    image = tf.image.random_brightness(image, max_delta)
                    image = tf.clip_by_value(image, 0.0, 1.0)

            if label == -1:  # No label case
                return {'image': image}
            else:
                return {'image': image, 'label': label}

        # Build tf.data pipeline
        dataset = dataset.map(load_and_decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Convert to JAX-compatible iterator
        def keras_compatible_generator():
            for tf_batch in dataset:
                # Convert TensorFlow tensors to JAX arrays
                jax_batch = {}
                for key, value in tf_batch.items():
                    if key in ['image', 'label']:
                        jax_batch[key] = jnp.array(value.numpy())

                yield jax_batch

        return keras_compatible_generator()

    def _create_haiku_image_pipeline(self, image_paths: List[str], labels: Optional[List[int]],
                                   image_size: Tuple[int, int], augmentation_config: Optional[Dict]) -> Iterator[Dict]:
        """Haiku-optimized functional pipeline with DeepMind patterns"""

        def haiku_transform_fn(image: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
            """Haiku-style transformation function"""

            # Normalize
            image = image / 255.0

            # Haiku-compatible augmentations using functional patterns
            if augmentation_config:
                # Functional composition of transformations
                transforms = []

                if augmentation_config.get('horizontal_flip', False):
                    def random_flip(img, key):
                        return jax.lax.cond(
                            jax.random.uniform(key) < 0.5,
                            lambda x: jnp.fliplr(x),
                            lambda x: x,
                            img
                        )
                    transforms.append(random_flip)

                if 'rotation_range' in augmentation_config:
                    def random_rotation(img, key):
                        # Simplified rotation - in practice would use more sophisticated method
                        angle = jax.random.uniform(key, minval=-augmentation_config['rotation_range'],
                                                 maxval=augmentation_config['rotation_range'])
                        # Apply rotation (simplified implementation)
                        return img  # Placeholder
                    transforms.append(random_rotation)

                # Apply transformations sequentially
                keys = jax.random.split(rng, len(transforms))
                for transform, key in zip(transforms, keys):
                    image = transform(image, key)

            return image

        def haiku_batch_generator():
            """Generate batches with Haiku-style functional approach"""
            master_rng = hk.PRNGSequence(42)

            data_pairs = list(zip(image_paths, labels or [None] * len(image_paths)))

            for i in range(0, len(data_pairs), self.batch_size):
                batch_pairs = data_pairs[i:i + self.batch_size]

                # Load batch
                batch_images = []
                batch_labels = []

                for path, label in batch_pairs:
                    import PIL.Image as Image
                    with Image.open(path) as img:
                        img = img.convert('RGB').resize(image_size, Image.Resampling.LANCZOS)
                        img_array = jnp.array(np.array(img))
                        batch_images.append(img_array)
                        if label is not None:
                            batch_labels.append(label)

                if batch_images:
                    # Stack and transform with Haiku patterns
                    images_batch = jnp.stack(batch_images)

                    # Apply transformations using Haiku RNG management
                    batch_keys = jax.random.split(next(master_rng), len(batch_images))

                    # Vectorized transformation
                    vectorized_transform = jax.vmap(haiku_transform_fn, in_axes=(0, 0))
                    processed_images = vectorized_transform(images_batch, batch_keys)

                    batch = {'image': processed_images}
                    if batch_labels:
                        batch['label'] = jnp.array(batch_labels, dtype=jnp.int32)

                    yield batch

        return haiku_batch_generator()

    def create_text_pipeline(self,
                           texts: List[str],
                           labels: Optional[List[int]] = None,
                           tokenizer_config: Optional[Dict] = None,
                           max_length: int = 512) -> Iterator[Dict]:
        """Create optimized text processing pipeline for NLP tasks"""

        tokenizer_config = tokenizer_config or {}

        if self.framework == 'flax':
            return self._create_flax_text_pipeline(texts, labels, tokenizer_config, max_length)
        elif self.framework == 'equinox':
            return self._create_equinox_text_pipeline(texts, labels, tokenizer_config, max_length)
        elif self.framework == 'keras':
            return self._create_keras_text_pipeline(texts, labels, tokenizer_config, max_length)
        elif self.framework == 'haiku':
            return self._create_haiku_text_pipeline(texts, labels, tokenizer_config, max_length)

    def _create_flax_text_pipeline(self, texts: List[str], labels: Optional[List[int]],
                                 tokenizer_config: Dict, max_length: int) -> Iterator[Dict]:
        """Flax-optimized text processing pipeline"""

        def tokenize_text(text: str) -> Dict[str, jnp.ndarray]:
            """Simple tokenization - in practice would use proper tokenizer"""
            # Placeholder tokenization logic
            # In practice, would use transformers tokenizer or custom tokenizer

            # Simple character-level tokenization for demo
            chars = list(text.lower())[:max_length]
            char_to_idx = tokenizer_config.get('char_to_idx', {chr(i): i for i in range(256)})

            # Convert to indices
            indices = [char_to_idx.get(c, 0) for c in chars]

            # Pad to max_length
            padded_indices = indices + [0] * (max_length - len(indices))
            padded_indices = padded_indices[:max_length]

            # Create attention mask
            attention_mask = [1] * len(indices) + [0] * (max_length - len(indices))
            attention_mask = attention_mask[:max_length]

            return {
                'input_ids': jnp.array(padded_indices, dtype=jnp.int32),
                'attention_mask': jnp.array(attention_mask, dtype=jnp.int32)
            }

        def text_batch_generator():
            """Generate text batches for Flax"""
            data_pairs = list(zip(texts, labels or [None] * len(texts)))

            for i in range(0, len(data_pairs), self.batch_size):
                batch_pairs = data_pairs[i:i + self.batch_size]

                # Tokenize batch
                batch_items = []
                for text, label in batch_pairs:
                    tokenized = tokenize_text(text)
                    if label is not None:
                        tokenized['label'] = jnp.array(label, dtype=jnp.int32)
                    batch_items.append(tokenized)

                if batch_items:
                    # Stack into batch
                    batch = {}
                    for key in batch_items[0].keys():
                        batch[key] = jnp.stack([item[key] for item in batch_items])

                    yield batch

        return text_batch_generator()

    def create_sequence_pipeline(self,
                               sequences: List[jnp.ndarray],
                               targets: Optional[List[jnp.ndarray]] = None,
                               sequence_length: Optional[int] = None,
                               dynamic_padding: bool = True) -> Iterator[Dict]:
        """Create optimized sequence data pipeline for RNNs/Transformers"""

        def sequence_batch_generator():
            """Generate sequence batches with proper padding and masking"""
            data_pairs = list(zip(sequences, targets or [None] * len(sequences)))

            for i in range(0, len(data_pairs), self.batch_size):
                batch_pairs = data_pairs[i:i + self.batch_size]

                batch_sequences = []
                batch_targets = []
                batch_lengths = []

                for seq, target in batch_pairs:
                    batch_sequences.append(seq)
                    batch_lengths.append(len(seq))
                    if target is not None:
                        batch_targets.append(target)

                if batch_sequences:
                    # Determine batch sequence length
                    if dynamic_padding:
                        max_len = max(batch_lengths)
                    else:
                        max_len = sequence_length or max(batch_lengths)

                    # Pad sequences
                    padded_sequences = []
                    attention_masks = []

                    for seq, length in zip(batch_sequences, batch_lengths):
                        if len(seq) > max_len:
                            # Truncate
                            padded_seq = seq[:max_len]
                            mask = jnp.ones(max_len, dtype=jnp.bool_)
                        else:
                            # Pad
                            pad_length = max_len - len(seq)
                            padded_seq = jnp.concatenate([seq, jnp.zeros((pad_length,) + seq.shape[1:], dtype=seq.dtype)])
                            mask = jnp.concatenate([jnp.ones(len(seq), dtype=jnp.bool_),
                                                  jnp.zeros(pad_length, dtype=jnp.bool_)])

                        padded_sequences.append(padded_seq)
                        attention_masks.append(mask)

                    # Create batch
                    batch = {
                        'sequences': jnp.stack(padded_sequences),
                        'attention_mask': jnp.stack(attention_masks),
                        'lengths': jnp.array(batch_lengths, dtype=jnp.int32)
                    }

                    if batch_targets:
                        # Handle target padding similarly
                        padded_targets = []
                        for target in batch_targets:
                            if len(target) > max_len:
                                padded_target = target[:max_len]
                            else:
                                pad_length = max_len - len(target)
                                padded_target = jnp.concatenate([target, jnp.zeros((pad_length,) + target.shape[1:], dtype=target.dtype)])
                            padded_targets.append(padded_target)

                        batch['targets'] = jnp.stack(padded_targets)

                    yield batch

        return sequence_batch_generator()

    def _apply_flax_augmentations(self, image: jnp.ndarray, config: Dict) -> jnp.ndarray:
        """Apply Flax-compatible image augmentations"""

        if config.get('horizontal_flip', False):
            # Simple random flip (in practice would use proper RNG)
            if np.random.random() < 0.5:
                image = jnp.fliplr(image)

        if 'brightness_range' in config:
            brightness_delta = np.random.uniform(*config['brightness_range'])
            image = jnp.clip(image + brightness_delta, 0.0, 1.0)

        if 'rotation_range' in config:
            # Placeholder for rotation - would implement proper rotation
            pass

        return image

    def _shard_batch(self, batch: Dict) -> Dict:
        """Shard batch across multiple devices for distributed training"""

        def shard_array(arr: jnp.ndarray) -> jnp.ndarray:
            """Shard single array across devices"""
            if arr.shape[0] % self.device_count != 0:
                # Pad batch to be divisible by device count
                pad_size = self.device_count - (arr.shape[0] % self.device_count)
                padding = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
                arr = jnp.pad(arr, padding)

            return arr.reshape((self.device_count, -1) + arr.shape[1:])

        sharded_batch = {}
        for key, value in batch.items():
            if isinstance(value, jnp.ndarray):
                sharded_batch[key] = shard_array(value)
            else:
                sharded_batch[key] = value

        return sharded_batch

    def create_distributed_pipeline(self,
                                  data_source: Union[List, Iterator],
                                  global_batch_size: int,
                                  data_type: str = 'image') -> Iterator[Dict]:
        """Create distributed training pipeline with proper sharding"""

        # Calculate per-device batch size
        per_device_batch_size = global_batch_size // self.device_count

        if data_type == 'image':
            # Modify batch size for distributed training
            original_batch_size = self.batch_size
            self.batch_size = per_device_batch_size

            # Create pipeline with adjusted batch size
            if isinstance(data_source, list) and all(isinstance(x, str) for x in data_source):
                pipeline = self.create_image_pipeline(data_source)
            else:
                raise ValueError("Unsupported data source type for distributed image pipeline")

            # Restore original batch size
            self.batch_size = original_batch_size

            return pipeline

        else:
            raise ValueError(f"Distributed pipeline not implemented for data type: {data_type}")

    def create_memory_efficient_pipeline(self,
                                       data_source: Any,
                                       memory_limit_gb: float = 4.0,
                                       streaming: bool = True) -> Iterator[Dict]:
        """Create memory-efficient pipeline for large datasets"""

        memory_limit_bytes = memory_limit_gb * 1024**3

        class MemoryEfficientIterator:
            def __init__(self, base_iterator, memory_limit: int):
                self.base_iterator = base_iterator
                self.memory_limit = memory_limit
                self.current_memory = 0

            def __iter__(self):
                return self

            def __next__(self):
                batch = next(self.base_iterator)

                # Estimate memory usage
                batch_memory = 0
                for value in batch.values():
                    if isinstance(value, jnp.ndarray):
                        batch_memory += value.nbytes

                # Check memory limit
                if self.current_memory + batch_memory > self.memory_limit:
                    # Force garbage collection
                    import gc
                    gc.collect()
                    self.current_memory = 0

                self.current_memory += batch_memory
                return batch

        # Create base pipeline based on data source type
        if hasattr(data_source, '__iter__'):
            base_pipeline = self.create_image_pipeline(data_source)
        else:
            raise ValueError("Unsupported data source for memory-efficient pipeline")

        return MemoryEfficientIterator(base_pipeline, memory_limit_bytes)

    def create_cross_framework_pipeline(self,
                                      data_source: Any,
                                      target_frameworks: List[str],
                                      data_type: str = 'image') -> Dict[str, Iterator]:
        """Create compatible pipelines for multiple frameworks"""

        pipelines = {}

        for framework in target_frameworks:
            # Temporarily switch framework
            original_framework = self.framework
            self.framework = framework

            try:
                if data_type == 'image':
                    pipeline = self.create_image_pipeline(data_source)
                elif data_type == 'text':
                    pipeline = self.create_text_pipeline(data_source)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                pipelines[framework] = pipeline

            finally:
                # Restore original framework
                self.framework = original_framework

        return pipelines

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""

        metrics = self.metrics.copy()

        # Calculate derived metrics
        if metrics['batches_processed'] > 0:
            metrics['samples_per_second'] = metrics['total_samples'] / (metrics['batches_processed'] * metrics['avg_batch_time']) if metrics['avg_batch_time'] > 0 else 0
            metrics['cache_hit_rate'] = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0

        # Add system metrics
        try:
            import psutil
            metrics['memory_usage_percent'] = psutil.virtual_memory().percent
            metrics['cpu_usage_percent'] = psutil.cpu_percent()
        except ImportError:
            pass

        return metrics

    def optimize_pipeline(self, target_throughput: float = 1000.0) -> Dict[str, Any]:
        """Automatically optimize pipeline parameters for target throughput"""

        current_metrics = self.get_performance_metrics()
        current_throughput = current_metrics.get('samples_per_second', 0)

        optimization_suggestions = {
            'current_throughput': current_throughput,
            'target_throughput': target_throughput,
            'optimizations': []
        }

        if current_throughput < target_throughput:
            # Suggest batch size increase
            if current_metrics.get('memory_usage_percent', 0) < 80:
                new_batch_size = min(self.batch_size * 2, 256)
                optimization_suggestions['optimizations'].append({
                    'parameter': 'batch_size',
                    'current': self.batch_size,
                    'suggested': new_batch_size,
                    'reason': 'Increase batch size to improve throughput'
                })

            # Suggest more workers
            if self.num_workers < 8:
                new_workers = min(self.num_workers * 2, 8)
                optimization_suggestions['optimizations'].append({
                    'parameter': 'num_workers',
                    'current': self.num_workers,
                    'suggested': new_workers,
                    'reason': 'Increase parallel workers for faster data loading'
                })

            # Suggest prefetch increase
            if self.prefetch_size < 4:
                new_prefetch = min(self.prefetch_size * 2, 4)
                optimization_suggestions['optimizations'].append({
                    'parameter': 'prefetch_size',
                    'current': self.prefetch_size,
                    'suggested': new_prefetch,
                    'reason': 'Increase prefetch buffer for smoother data flow'
                })

        return optimization_suggestions

# Framework-specific training integration examples

class FlaxTrainingIntegration:
    """Integration patterns for Flax training loops"""

    @staticmethod
    def create_train_state(model: nn.Module,
                          learning_rate: float,
                          input_shape: Tuple[int, ...]) -> train_state.TrainState:
        """Create Flax training state"""

        # Initialize model parameters
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1,) + input_shape)
        variables = model.init(rng, dummy_input)

        # Create optimizer
        tx = optax.adam(learning_rate)

        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx
        )

    @staticmethod
    def training_step(state: train_state.TrainState,
                     batch: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, Dict]:
        """Single training step with Flax"""

        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['label']
            ).mean()
            return loss, logits

        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, logits = grad_fn(state.params)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute metrics
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch['label'])
        metrics = {'accuracy': accuracy}

        return state, metrics

class EquinoxTrainingIntegration:
    """Integration patterns for Equinox training loops"""

    @staticmethod
    def create_model_and_optimizer(model_fn: Callable,
                                 learning_rate: float,
                                 input_shape: Tuple[int, ...]) -> Tuple[Any, Any]:
        """Create Equinox model and optimizer"""

        # Initialize model
        key = jax.random.PRNGKey(42)
        model = model_fn(key)

        # Create optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        return model, (optimizer, opt_state)

    @staticmethod
    def training_step(model: Any,
                     optimizer_state: Tuple[Any, Any],
                     batch: Dict[str, jnp.ndarray]) -> Tuple[Any, Tuple[Any, Any], Dict]:
        """Single training step with Equinox"""

        optimizer, opt_state = optimizer_state

        def loss_fn(model):
            logits = jax.vmap(model)(batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['label']
            ).mean()
            return loss, logits

        (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

        # Update model
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # Compute metrics
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch['label'])
        metrics = {'loss': loss, 'accuracy': accuracy}

        return model, (optimizer, opt_state), metrics

# Performance optimization utilities

class DataLoadingProfiler:
    """Comprehensive profiling for neural network data loading"""

    def __init__(self):
        self.metrics = {
            'load_times': [],
            'preprocess_times': [],
            'batch_times': [],
            'memory_usage': [],
            'gpu_utilization': []
        }

    def profile_pipeline(self, data_loader: NeuralDataLoader,
                        num_batches: int = 100) -> Dict:
        """Profile data loading pipeline performance"""

        pipeline = data_loader.create_image_pipeline(['dummy_path'] * 1000)  # Dummy data

        start_time = time.time()

        for i, batch in enumerate(pipeline):
            if i >= num_batches:
                break

            batch_start = time.time()

            # Force computation to measure actual loading time
            for key, value in batch.items():
                if isinstance(value, jnp.ndarray):
                    _ = jnp.sum(value).block_until_ready()

            batch_end = time.time()
            self.metrics['batch_times'].append(batch_end - batch_start)

            # Memory monitoring
            try:
                import psutil
                self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
            except ImportError:
                pass

        total_time = time.time() - start_time

        return {
            'total_time': total_time,
            'avg_batch_time': np.mean(self.metrics['batch_times']),
            'std_batch_time': np.std(self.metrics['batch_times']),
            'throughput': num_batches / total_time,
            'avg_memory_usage': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'performance_score': self._calculate_performance_score()
        }

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score"""
        if not self.metrics['batch_times']:
            return 0.0

        # Base score on throughput and consistency
        avg_time = np.mean(self.metrics['batch_times'])
        std_time = np.std(self.metrics['batch_times'])

        # Lower is better for time, penalize high variance
        consistency_penalty = std_time / avg_time if avg_time > 0 else 1.0

        # Score between 0-100, higher is better
        score = max(0, 100 - (avg_time * 100) - (consistency_penalty * 20))

        return min(score, 100.0)
```

## Advanced Neural Network Data Patterns

### Multi-Modal Data Loading
```python
# Combining different data modalities for multi-modal neural networks
class MultiModalDataLoader(NeuralDataLoader):
    """Specialized loader for multi-modal neural networks"""

    def create_multimodal_pipeline(self,
                                 image_paths: List[str],
                                 text_data: List[str],
                                 audio_paths: Optional[List[str]] = None,
                                 labels: Optional[List[int]] = None) -> Iterator[Dict]:
        """Create synchronized multi-modal data pipeline"""

        def multimodal_generator():
            data_tuples = list(zip(
                image_paths,
                text_data,
                audio_paths or [None] * len(image_paths),
                labels or [None] * len(image_paths)
            ))

            for i in range(0, len(data_tuples), self.batch_size):
                batch_tuples = data_tuples[i:i + self.batch_size]

                batch = {
                    'images': [],
                    'texts': [],
                    'audio': [],
                    'labels': []
                }

                for img_path, text, audio_path, label in batch_tuples:
                    # Load and process image
                    # (Implementation details...)

                    # Process text
                    # (Implementation details...)

                    # Load audio if available
                    # (Implementation details...)

                    # Add to batch
                    # (Implementation details...)

                # Stack and yield batch
                yield self._stack_multimodal_batch(batch)

        return multimodal_generator()
```

### Dynamic Batch Size Optimization
```python
# Adaptive batch sizing based on memory and model complexity
class AdaptiveBatchLoader(NeuralDataLoader):
    """Adaptive batch size optimization for neural network training"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimal_batch_size = self.batch_size
        self.memory_threshold = 0.9  # 90% memory utilization

    def find_optimal_batch_size(self, model_forward_fn: Callable,
                               sample_input: Dict) -> int:
        """Automatically find optimal batch size"""

        current_batch_size = 1
        max_batch_size = 512

        while current_batch_size <= max_batch_size:
            try:
                # Create test batch
                test_batch = self._create_test_batch(sample_input, current_batch_size)

                # Test forward pass
                _ = model_forward_fn(test_batch)

                # Check memory usage
                memory_usage = self._get_gpu_memory_usage()

                if memory_usage > self.memory_threshold:
                    break

                current_batch_size *= 2

            except Exception as e:
                # Memory overflow or other error
                break

        # Use 80% of maximum working batch size for safety
        self.optimal_batch_size = max(1, current_batch_size // 2)
        return self.optimal_batch_size
```

## Integration with Existing Scientific Agents

### JAX Expert Integration
- **Advanced JAX transformations**: Custom data loading transformations with jit, vmap, pmap
- **Device optimization**: Optimal data placement and device-specific loading strategies
- **Performance profiling**: JAX-specific performance analysis and optimization

### GPU Computing Expert Integration
- **CUDA kernel integration**: Custom CUDA kernels for specialized data preprocessing
- **Memory management**: Advanced GPU memory optimization for large batch processing
- **Multi-GPU coordination**: Distributed data loading across multiple GPUs

### Statistics Expert Integration
- **Sampling strategies**: Statistically sound data sampling and validation set creation
- **Augmentation validation**: Statistical validation of data augmentation effects
- **Performance testing**: Statistical analysis of data loading performance improvements

### Experiment Manager Integration
- **Systematic data experiments**: Controlled testing of different data loading strategies
- **A/B testing**: Comparative analysis of data loading approaches
- **Hyperparameter optimization**: Systematic optimization of data loading parameters

## Practical Usage Examples

### Example 1: Image Classification with Flax
```python
# High-performance image classification pipeline
loader = NeuralDataLoader(framework='flax', batch_size=64, prefetch_size=4)

# Create optimized pipeline
image_paths = [f'images/class_{i}/img_{j}.jpg' for i in range(10) for j in range(1000)]
labels = [i for i in range(10) for _ in range(1000)]

augmentation_config = {
    'horizontal_flip': True,
    'brightness_range': (-0.1, 0.1),
    'shuffle': True
}

pipeline = loader.create_image_pipeline(
    image_paths, labels,
    image_size=(224, 224),
    augmentation_config=augmentation_config
)

# Flax training integration
model = ResNet18()  # Your Flax model
state = FlaxTrainingIntegration.create_train_state(model, 0.001, (224, 224, 3))

for epoch in range(100):
    for batch in pipeline:
        state, metrics = FlaxTrainingIntegration.training_step(state, batch)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### Example 2: Sequence Modeling with Equinox
```python
# Functional sequence processing for transformer training
loader = NeuralDataLoader(framework='equinox', batch_size=32)

# Generate sequence data
sequences = [jnp.array(np.random.randint(0, 1000, size=(np.random.randint(10, 100),)))
             for _ in range(5000)]

pipeline = loader.create_sequence_pipeline(
    sequences,
    dynamic_padding=True,
    sequence_length=128
)

# Equinox functional training
model = TransformerModel(key=jax.random.PRNGKey(42))
optimizer_state = EquinoxTrainingIntegration.create_model_and_optimizer(
    TransformerModel, 0.0001, (128,)
)

for batch in pipeline:
    model, optimizer_state, metrics = EquinoxTrainingIntegration.training_step(
        model, optimizer_state, batch
    )
```

### Example 3: Multi-Modal Learning with Keras
```python
# Multi-modal data pipeline for vision-language models
loader = NeuralDataLoader(framework='keras', batch_size=16)

# Prepare multi-modal data
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
captions = ['A cat sitting on a table', 'A dog running in park', ...]

# Create multi-modal pipeline
multimodal_loader = MultiModalDataLoader(framework='keras')
pipeline = multimodal_loader.create_multimodal_pipeline(
    image_paths, captions
)

# Train vision-language model
for batch in pipeline:
    # batch contains {'images': ..., 'texts': ..., 'labels': ...}
    vision_features = vision_model(batch['images'])
    text_features = text_model(batch['texts'])
    # Combine and train...
```

### Example 4: Distributed Training with Haiku
```python
# Large-scale distributed training setup
loader = NeuralDataLoader(framework='haiku', device_count=8)

# Create distributed pipeline
global_batch_size = 256  # 32 per device
pipeline = loader.create_distributed_pipeline(
    data_source=large_dataset,
    global_batch_size=global_batch_size,
    data_type='image'
)

# Haiku distributed training with pmap
@jax.pmap
def distributed_train_step(model_state, batch):
    # Training step that runs on each device
    # ...
    return updated_state, metrics

for batch in pipeline:
    # batch is automatically sharded across devices
    model_state, metrics = distributed_train_step(model_state, batch)
```

### Example 5: Memory-Efficient Large Dataset Processing
```python
# Handle datasets that don't fit in memory
loader = NeuralDataLoader(framework='flax', batch_size=128)

# Create memory-efficient streaming pipeline
huge_dataset_path = '/path/to/terabyte_dataset'
memory_efficient_pipeline = loader.create_memory_efficient_pipeline(
    huge_dataset_path,
    memory_limit_gb=8.0,
    streaming=True
)

# Process massive dataset without OOM
for batch in memory_efficient_pipeline:
    # Each batch is loaded on-demand
    # Automatic memory management prevents OOM
    train_step(model, batch)
```

## Advanced Framework-Specific Patterns

### Flax Advanced Patterns
```python
# Flax-specific advanced data loading with scan and state management
class FlaxAdvancedDataLoader(NeuralDataLoader):

    def create_stateful_pipeline(self, sequences: List[jnp.ndarray]) -> Iterator[Dict]:
        """Create pipeline with Flax scan-compatible state management"""

        def stateful_batch_processor(carry, batch):
            # Process batch with carry state (for RNNs, etc.)
            processed_batch = {}
            new_carry = carry  # Update carry as needed

            return new_carry, processed_batch

        # Use with nn.scan in Flax models
        return self._create_scan_compatible_pipeline(sequences, stateful_batch_processor)
```

### Equinox PyTree Integration
```python
# Equinox PyTree-aware data loading
class EquinoxPyTreeLoader(NeuralDataLoader):

    def create_pytree_pipeline(self, pytree_data: List[Any]) -> Iterator[Dict]:
        """Create pipeline that preserves PyTree structure"""

        def pytree_batch_processor(pytree_list):
            # Stack PyTrees while preserving structure
            return jax.tree_map(lambda *xs: jnp.stack(xs), *pytree_list)

        return self._create_pytree_compatible_pipeline(pytree_data, pytree_batch_processor)
```

### Keras tf.data Advanced Integration
```python
# Advanced tf.data integration with JAX optimization
class KerasAdvancedLoader(NeuralDataLoader):

    def create_optimized_tf_pipeline(self, dataset_path: str) -> Iterator[Dict]:
        """Create highly optimized tf.data pipeline with JAX backend"""

        # Advanced tf.data optimizations
        options = tf.data.Options()
        options.threading.private_threadpool_size = self.num_workers
        options.threading.max_intra_op_parallelism = 1
        options.optimization.map_parallelization = True
        options.optimization.parallel_batch = True

        # Create optimized dataset
        dataset = tf.data.TFRecordDataset(dataset_path)
        dataset = dataset.with_options(options)

        return self._create_tf_jax_bridge(dataset)
```

### Haiku Functional Composition
```python
# Haiku functional data transformation composition
class HaikuFunctionalLoader(NeuralDataLoader):

    def create_composed_pipeline(self, data: List[Any],
                               transforms: List[Callable]) -> Iterator[Dict]:
        """Create pipeline with functional transform composition"""

        @hk.transform
        def composed_transform(x, rng):
            # Compose transforms functionally
            for transform in transforms:
                x = transform(x, rng)
            return x

        return self._create_functional_pipeline(data, composed_transform)
```

## Performance Optimization Strategies

### GPU Memory Optimization
```python
class GPUOptimizedLoader(NeuralDataLoader):
    """GPU memory-optimized data loading"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_memory_fraction = 0.8  # Use 80% of GPU memory

    def optimize_for_gpu(self, model_memory_mb: float) -> Dict[str, Any]:
        """Optimize data loading for GPU memory constraints"""

        available_gpu_memory = self._get_available_gpu_memory()
        data_memory_budget = available_gpu_memory * self.gpu_memory_fraction - model_memory_mb

        # Calculate optimal batch size
        sample_size = self._estimate_sample_memory_size()
        optimal_batch_size = int(data_memory_budget / sample_size)

        return {
            'optimal_batch_size': optimal_batch_size,
            'memory_budget_mb': data_memory_budget,
            'estimated_memory_per_batch': optimal_batch_size * sample_size
        }
```

### Throughput Optimization
```python
class ThroughputOptimizer:
    """Automated throughput optimization for neural network training"""

    def __init__(self, loader: NeuralDataLoader):
        self.loader = loader
        self.optimization_history = []

    def auto_optimize(self, target_samples_per_second: float = 1000.0) -> Dict:
        """Automatically optimize for target throughput"""

        best_config = None
        best_throughput = 0.0

        # Test different configurations
        configs_to_test = [
            {'batch_size': 32, 'num_workers': 4, 'prefetch_size': 2},
            {'batch_size': 64, 'num_workers': 6, 'prefetch_size': 3},
            {'batch_size': 128, 'num_workers': 8, 'prefetch_size': 4},
        ]

        for config in configs_to_test:
            # Apply configuration
            self.loader.batch_size = config['batch_size']
            self.loader.num_workers = config['num_workers']
            self.loader.prefetch_size = config['prefetch_size']

            # Measure throughput
            profiler = DataLoadingProfiler()
            metrics = profiler.profile_pipeline(self.loader, num_batches=50)
            throughput = metrics.get('throughput', 0)

            self.optimization_history.append({
                'config': config,
                'throughput': throughput,
                'metrics': metrics
            })

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config

        return {
            'best_config': best_config,
            'achieved_throughput': best_throughput,
            'target_throughput': target_samples_per_second,
            'optimization_ratio': best_throughput / target_samples_per_second,
            'history': self.optimization_history
        }
```

## Integration with Scientific Computing Ecosystem

### JAX Expert Integration
- **Advanced JAX transformations**: Custom jit-compiled data preprocessing functions
- **Device sharding**: Automatic optimal sharding across JAX devices
- **Memory management**: JAX-specific memory optimization patterns

### Data Loading Expert Integration
- **Scientific data formats**: Seamless integration with HDF5, NetCDF scientific datasets
- **Multi-modal scientific data**: Combining experimental measurements, simulations, and observations
- **Quality control**: Scientific data validation and preprocessing pipelines

### GPU Computing Expert Integration
- **CUDA acceleration**: Custom CUDA kernels for data preprocessing operations
- **Multi-GPU strategies**: Advanced multi-GPU data loading and synchronization
- **Memory optimization**: GPU memory pool management and allocation strategies

### Statistics Expert Integration
- **Data validation**: Statistical validation of training data distributions
- **Sampling strategies**: Statistically sound train/validation/test splits
- **Augmentation analysis**: Statistical impact assessment of data augmentations

## Cross-Framework Compatibility Matrix

| Feature | Flax | Equinox | Keras | Haiku |
|---------|------|---------|--------|-------|
| **Basic Data Loading** | ✅ | ✅ | ✅ | ✅ |
| **Image Pipelines** | ✅ | ✅ | ✅ | ✅ |
| **Text Processing** | ✅ | ✅ | ✅ | ✅ |
| **Sequence Handling** | ✅ | ✅ | ✅ | ✅ |
| **Multi-Modal** | ✅ | ✅ | ✅ | ✅ |
| **Distributed Training** | ✅ | ✅ | ✅ | ✅ |
| **Memory Efficient** | ✅ | ✅ | ✅ | ✅ |
| **JAX Transformations** | ✅ | ✅ | ⚠️ | ✅ |
| **Functional Patterns** | ⚠️ | ✅ | ❌ | ✅ |
| **State Management** | ✅ | ❌ | ✅ | ✅ |
| **tf.data Integration** | ❌ | ❌ | ✅ | ❌ |

## Future Enhancements

### Near-Term (1-3 months)
1. **Automatic Mixed Precision**: Intelligent dtype selection based on model and data characteristics
2. **Advanced Caching**: Multi-level caching with LRU eviction and compression
3. **Dynamic Load Balancing**: Real-time load balancing across multiple data sources

### Medium-Term (3-6 months)
1. **Federated Learning Support**: Distributed data loading across multiple institutions
2. **Real-Time Data Streams**: Integration with real-time data sources and streaming platforms
3. **Advanced Profiling**: Comprehensive performance profiling with automated optimization

### Long-Term (6+ months)
1. **Automatic Data Discovery**: AI-powered discovery and integration of relevant datasets
2. **Adaptive Augmentation**: Learned data augmentation strategies based on model performance
3. **Cross-Modal Transfer**: Automatic adaptation of data loading patterns across different modalities

This agent transforms neural network data loading from ad-hoc implementations into **systematic, high-performance, framework-optimized pipelines** that maximize training efficiency while maintaining flexibility across JAX-based deep learning frameworks. It provides comprehensive support for modern neural network training workflows with deep integration into the JAX ecosystem and seamless interoperability across Flax, Equinox, Keras, and Haiku frameworks.