title: "JAX data load"
---
description: Set up data loading pipelines with Grain or TF Datasets optimized for JAX workflows
subcategory: jax-ecosystem
complexity: intermediate
category: jax-ml
argument-hint: "[--framework=grain|tf] [--batch-size=32] [--shuffle] [--prefetch] [--agents=auto|jax|scientific|ai|data|all] [--orchestrate] [--intelligent] [--breakthrough] [--distributed] [--optimize]"
allowed-tools: "*"
model: inherit
---

# JAX Data Load

Set up data loading pipelines with Grain or TensorFlow Datasets optimized for JAX workflows.

```bash
/jax-data-load [--framework=grain|tf] [--batch-size=32] [--shuffle] [--prefetch] [--agents=auto|jax|scientific|ai|data|all] [--orchestrate] [--intelligent] [--breakthrough] [--distributed] [--optimize]
```

## Options

- `--framework=<framework>`: Data loading framework (grain, tf)
- `--batch-size=<size>`: Batch size for data loading (default: 32)
- `--shuffle`: Enable data shuffling
- `--prefetch`: Enable data prefetching
- `--agents=<agents>`: Agent selection (auto, jax, scientific, ai, data, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with data intelligence
- `--intelligent`: Enable intelligent agent selection based on data analysis
- `--breakthrough`: Enable breakthrough data processing optimization
- `--distributed`: Enable distributed data processing across multiple agents
- `--optimize`: Apply performance optimization to data loading pipelines

## What it does

1. **Data Pipeline Setup**: Configure Grain or TF Datasets for JAX
2. **Batching Strategy**: Implement efficient batching with proper shapes
3. **Preprocessing**: Add transformations and data augmentation
4. **Device Optimization**: Handle device placement and memory management
5. **Performance Tuning**: Enable prefetching and parallel loading
6. **23-Agent Data Intelligence**: Multi-agent collaboration for optimal data processing strategies
7. **Distributed Processing**: Agent-coordinated data processing across multiple domains
8. **Advanced Optimization**: Agent-driven data pipeline performance optimization

## 23-Agent Intelligent Data Processing System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes data characteristics, processing requirements, and performance constraints to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Data Type Detection → Agent Selection
- Large-Scale Data Processing → data-professional + ai-systems-architect + systems-architect
- Scientific Data Processing → scientific-computing-master + data-professional + jax-pro
- ML Data Pipelines → ai-systems-architect + neural-networks-master + data-professional
- Real-Time Data Streams → data-professional + systems-architect + jax-pro
- Multi-Modal Data → neural-networks-master + data-professional + multi-agent-orchestrator
```

### Core JAX Data Processing Agents

#### **`data-professional`** - Data Processing & Pipeline Expert
- **Data Architecture**: Expert data pipeline design and ETL optimization strategies
- **Performance Engineering**: High-performance data loading and processing optimization
- **Multi-Format Support**: Efficient handling of diverse data formats and sources
- **Streaming Optimization**: Real-time and streaming data processing strategies
- **Quality Management**: Data validation, cleaning, and quality assurance protocols

#### **`jax-pro`** - JAX Data Optimization Specialist
- **JAX Integration**: Deep expertise in JAX-optimized data loading and preprocessing
- **Device Management**: Optimal device placement and memory management for data
- **Vectorization**: Efficient vectorized data operations and transformations
- **Performance Tuning**: JAX-specific data pipeline performance optimization
- **Memory Efficiency**: Advanced memory management for large-scale data processing

#### **`ai-systems-architect`** - AI Data Infrastructure & Scalability
- **ML Data Pipelines**: Scalable data infrastructure for machine learning workflows
- **Distributed Processing**: Multi-device and multi-node data processing architecture
- **Production Data Systems**: Data pipeline design for production AI systems
- **Resource Optimization**: Efficient resource allocation for data processing workloads
- **Integration Architecture**: Data system integration with larger AI infrastructure

#### **`systems-architect`** - System-Level Data Processing
- **Infrastructure Design**: System-level data processing architecture and optimization
- **Resource Management**: Computational resource allocation for data processing
- **Scalability Engineering**: Data system design for large-scale processing
- **Performance Monitoring**: Real-time data processing performance tracking
- **Fault Tolerance**: Robust data processing systems with failure recovery

### Specialized Data Processing Agents

#### **`scientific-computing-master`** - Scientific Data Processing
- **Scientific Data Formats**: Expert handling of scientific data formats and standards
- **Domain Integration**: Data processing for specific scientific domains
- **Multi-Scale Data**: Processing strategies for multi-scale scientific datasets
- **Research Standards**: Data processing for research-grade reproducibility
- **Computational Efficiency**: High-performance scientific data processing

#### **`neural-networks-master`** - ML Data Processing Expert
- **Training Data Optimization**: Data pipeline optimization for neural network training
- **Augmentation Strategies**: Advanced data augmentation and preprocessing techniques
- **Multi-Modal Processing**: Data processing for complex multi-modal learning
- **Feature Engineering**: Intelligent feature extraction and processing strategies
- **Training Integration**: Data pipeline design for efficient training workflows

#### **`research-intelligence-master`** - Research Data Methodology
- **Experimental Data Design**: Data collection and processing for research experiments
- **Reproducibility**: Research-grade data processing with reproducibility standards
- **Innovation Synthesis**: Advanced data processing techniques from cutting-edge research
- **Academic Standards**: Data processing methodologies for academic publication
- **Breakthrough Discovery**: Novel data processing approaches and optimization

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Data Processing
Automatically analyzes data requirements and selects optimal agent combinations:
- **Data Analysis**: Detects data types, sizes, processing requirements, performance needs
- **Resource Assessment**: Evaluates computational resources and processing constraints
- **Agent Matching**: Maps data processing challenges to relevant agent expertise
- **Optimization Focus**: Balances comprehensive processing with performance efficiency

#### **`jax`** - JAX-Specialized Data Processing Team
- `jax-pro` (JAX ecosystem lead)
- `data-professional` (data pipeline design)
- `ai-systems-architect` (ML integration)
- `systems-architect` (system optimization)

#### **`scientific`** - Scientific Computing Data Processing Team
- `scientific-computing-master` (lead)
- `data-professional` (data pipeline expertise)
- `jax-pro` (JAX implementation)
- `research-intelligence-master` (research methodology)
- Domain-specific experts based on scientific application

#### **`ai`** - AI/ML Data Processing Team
- `ai-systems-architect` (lead)
- `neural-networks-master` (ML data processing)
- `data-professional` (data pipeline design)
- `jax-pro` (JAX optimization)
- `systems-architect` (infrastructure)

#### **`data`** - Dedicated Data Processing Team
- `data-professional` (lead)
- `systems-architect` (system design)
- `ai-systems-architect` (AI integration)
- `jax-pro` (JAX optimization)
- `research-intelligence-master` (methodology)

#### **`all`** - Complete 23-Agent Data Processing Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough data processing optimization.

### 23-Agent Data Processing Orchestration (`--orchestrate`)

#### **Multi-Agent Data Pipeline**
1. **Data Analysis Phase**: Multiple agents analyze data characteristics simultaneously
2. **Pipeline Design**: Collaborative data pipeline architecture development
3. **Optimization Strategy**: Multi-agent optimization of data processing performance
4. **Distributed Processing**: Agent-coordinated data processing across resources
5. **Real-Time Monitoring**: Continuous multi-agent data processing monitoring

#### **Breakthrough Data Processing Discovery (`--breakthrough`)**
- **Cross-Domain Innovation**: Data processing techniques from multiple domains
- **Emergent Optimization**: Novel data processing strategies through agent collaboration
- **Research-Grade Processing**: Academic and industry-leading data processing standards
- **Adaptive Pipelines**: Dynamic data processing optimization based on real-time analysis

### Advanced 23-Agent Data Processing Examples

```bash
# Intelligent auto-selection for data processing optimization
/jax-data-load --agents=auto --intelligent --framework=grain --optimize

# Scientific computing data processing with specialized agents
/jax-data-load --agents=scientific --framework=tf --orchestrate --distributed

# AI/ML data pipeline optimization with scalability focus
/jax-data-load --agents=ai --batch-size=128 --optimize --breakthrough

# Research-grade data processing development
/jax-data-load --agents=all --breakthrough --orchestrate --distributed

# JAX-specialized data loading optimization
/jax-data-load --agents=jax --framework=grain --prefetch --optimize

# Complete 23-agent data processing ecosystem
/jax-data-load --agents=all --orchestrate --breakthrough --intelligent

# Large-scale distributed data processing
/jax-data-load large_dataset.py --agents=data --distributed --optimize

# Scientific data processing with domain expertise
/jax-data-load scientific_data.py --agents=scientific --intelligent --breakthrough

# Production ML data pipeline
/jax-data-load ml_pipeline.py --agents=ai --optimize --distributed

# Real-time data streaming optimization
/jax-data-load streaming_data.py --agents=data --orchestrate --intelligent

# Multi-modal data processing
/jax-data-load multimodal_data.py --agents=ai --breakthrough --orchestrate

# Research data processing workflow
/jax-data-load research_data.py --agents=all --distributed --breakthrough
```

### Intelligent Agent Selection Examples

```bash
# Data Type Detection → Intelligent Agent Selection

# Large-scale image dataset processing
/jax-data-load image_dataset.py --agents=auto --intelligent
# → Selects: ai-systems-architect + neural-networks-master + data-professional

# Scientific simulation data processing
/jax-data-load simulation_data.py --agents=auto --intelligent
# → Selects: scientific-computing-master + data-professional + jax-pro

# Production ML data pipeline
/jax-data-load production_data.py --agents=auto --intelligent
# → Selects: ai-systems-architect + data-professional + systems-architect

# Real-time streaming data processing
/jax-data-load streaming_pipeline.py --agents=auto --intelligent
# → Selects: data-professional + systems-architect + jax-pro

# Multi-modal research data processing
/jax-data-load multimodal_research.py --agents=auto --intelligent
# → Selects: research-intelligence-master + neural-networks-master + data-professional

# Comprehensive data processing audit
/jax-data-load complete_pipeline.py --agents=all --orchestrate --breakthrough
# → Activates: All 23 agents with intelligent coordination for maximum data processing efficiency
```

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from typing import Iterator, Tuple, Dict, Any

# Initialize PRNG key
key = random.PRNGKey(0)

# ============================================================================
# 1. TENSORFLOW DATASETS FOR JAX
# ============================================================================

def setup_tf_dataset(dataset_name: str = "mnist", batch_size: int = 32,
                     shuffle: bool = True, prefetch: bool = True):
    """Set up TensorFlow Datasets pipeline for JAX"""

    # Load dataset
    ds_train, ds_test = tfds.load(
        dataset_name,
        split=['train', 'test'],
        shuffle_files=shuffle,
        as_supervised=True
    )

    def preprocess(image, label):
        """Preprocess function for images"""
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Convert to JAX-compatible format
        image = tf.reshape(image, [-1])  # Flatten for simple models
        label = tf.one_hot(label, depth=10)  # One-hot encoding

        return image, label

    # Configure training pipeline
    train_ds = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        train_ds = train_ds.shuffle(1000)

    train_ds = train_ds.batch(batch_size)

    if prefetch:
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Configure test pipeline
    test_ds = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)

    if prefetch:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds

def tf_dataset_to_jax(tf_dataset) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Convert TensorFlow dataset to JAX arrays"""

    for batch in tf_dataset:
        x, y = batch
        # Convert to JAX arrays
        x_jax = jnp.array(x.numpy())
        y_jax = jnp.array(y.numpy())
        yield x_jax, y_jax

# ============================================================================
# 2. GRAIN DATA LOADING
# ============================================================================

try:
    import grain.tensorflow as grain_tf
    import grain.python as grain_py
    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False
    print("Grain not available. Using TensorFlow Datasets only.")

def setup_grain_dataset(data_source, batch_size: int = 32,
                       shuffle: bool = True, num_workers: int = 4):
    """Set up Grain data loading pipeline"""

    if not GRAIN_AVAILABLE:
        raise ImportError("Grain is not installed")

    # Create data source
    sampler = grain_py.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=42
    )

    # Create data loader
    loader = grain_py.DataLoader(
        data_source=data_source,
        sampler=sampler,
        num_workers=num_workers,
        worker_buffer_size=2,
        worker_count=num_workers
    )

    # Add batching
    batched_loader = loader.batch(batch_size)

    return batched_loader

# ============================================================================
# 3. CUSTOM DATA SOURCES
# ============================================================================

class NumpyDataSource:
    """Custom data source for numpy arrays"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {'input': self.x[index], 'target': self.y[index]}

def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 10):
    """Create synthetic dataset for testing"""

    key, subkey = random.split(key)
    x = random.normal(subkey, (n_samples, n_features))
    y = random.randint(key, (n_samples,), 0, 2)  # Binary classification

    return np.array(x), np.array(y)

# ============================================================================
# 4. DATA AUGMENTATION AND PREPROCESSING
# ============================================================================

def create_image_preprocessing():
    """Create image preprocessing pipeline"""

    def preprocess_image(image, label):
        # Normalize
        image = tf.cast(image, tf.float32) / 255.0

        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random brightness
        image = tf.image.random_brightness(image, 0.1)

        # Random contrast
        image = tf.image.random_contrast(image, 0.9, 1.1)

        return image, label

    return preprocess_image

def create_text_preprocessing(vocab_size: int = 10000, max_length: int = 128):
    """Create text preprocessing pipeline"""

    def preprocess_text(text, label):
        # Tokenize (simplified)
        tokens = tf.strings.split(text)

        # Convert to IDs (simplified - use proper tokenizer in practice)
        token_ids = tf.py_function(
            lambda x: np.random.randint(0, vocab_size, max_length),
            [tokens],
            tf.int32
        )
        token_ids.set_shape([max_length])

        return token_ids, label

    return preprocess_text

# ============================================================================
# 5. DEVICE PLACEMENT AND SHARDING
# ============================================================================

def shard_batch_for_devices(batch_x, batch_y, n_devices: int):
    """Shard batch across multiple devices"""

    batch_size = batch_x.shape[0]
    per_device_batch_size = batch_size // n_devices

    # Reshape for device sharding
    sharded_x = batch_x[:n_devices * per_device_batch_size]
    sharded_y = batch_y[:n_devices * per_device_batch_size]

    sharded_x = sharded_x.reshape(n_devices, per_device_batch_size, *batch_x.shape[1:])
    sharded_y = sharded_y.reshape(n_devices, per_device_batch_size, *batch_y.shape[1:])

    return sharded_x, sharded_y

def create_device_iterator(dataset, n_devices: int):
    """Create iterator that shards data across devices"""

    for batch_x, batch_y in dataset:
        if batch_x.shape[0] < n_devices:
            continue  # Skip incomplete batches

        sharded_x, sharded_y = shard_batch_for_devices(batch_x, batch_y, n_devices)
        yield sharded_x, sharded_y

# ============================================================================
# 6. PERFORMANCE OPTIMIZATION
# ============================================================================

def optimize_tf_dataset(dataset, cache: bool = True, repeat: bool = True):
    """Apply performance optimizations to TF dataset"""

    # Cache dataset in memory
    if cache:
        dataset = dataset.cache()

    # Repeat dataset
    if repeat:
        dataset = dataset.repeat()

    # Prefetch data
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def benchmark_data_loading(data_loader, n_batches: int = 100):
    """Benchmark data loading performance"""

    import time

    start_time = time.time()

    for i, batch in enumerate(data_loader):
        if i >= n_batches:
            break

        # Simulate processing
        if hasattr(batch[0], 'block_until_ready'):
            batch[0].block_until_ready()

    total_time = time.time() - start_time
    batches_per_second = n_batches / total_time

    print(f"Data loading performance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Batches per second: {batches_per_second:.2f}")

    return batches_per_second

# ============================================================================
# 7. COMPLETE DATA PIPELINE EXAMPLES
# ============================================================================

def create_image_classification_pipeline(dataset_name: str = "cifar10",
                                       batch_size: int = 32,
                                       train_split: str = "train",
                                       test_split: str = "test"):
    """Complete image classification data pipeline"""

    # Load datasets
    ds_train, ds_test = tfds.load(
        dataset_name,
        split=[train_split, test_split],
        as_supervised=True,
        shuffle_files=True
    )

    # Get dataset info
    ds_info = tfds.builder(dataset_name).info
    num_classes = ds_info.features['label'].num_classes

    def preprocess_for_classification(image, label):
        # Normalize image
        image = tf.cast(image, tf.float32) / 255.0

        # One-hot encode label
        label = tf.one_hot(label, depth=num_classes)

        return image, label

    # Training pipeline with augmentation
    train_pipeline = (ds_train
                     .map(create_image_preprocessing(), num_parallel_calls=tf.data.AUTOTUNE)
                     .map(preprocess_for_classification, num_parallel_calls=tf.data.AUTOTUNE)
                     .shuffle(1000)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    # Test pipeline (no augmentation)
    test_pipeline = (ds_test
                    .map(preprocess_for_classification, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

    return train_pipeline, test_pipeline, num_classes

def create_regression_pipeline(n_samples: int = 10000, n_features: int = 20,
                             batch_size: int = 32, validation_split: float = 0.2):
    """Complete regression data pipeline"""

    # Generate synthetic regression data
    key, subkey = random.split(key)
    X = random.normal(subkey, (n_samples, n_features))

    # Create target with some noise
    true_weights = random.normal(key, (n_features,))
    y = jnp.dot(X, true_weights) + 0.1 * random.normal(key, (n_samples,))

    # Convert to numpy for TF compatibility
    X_np = np.array(X)
    y_np = np.array(y)

    # Split into train/validation
    split_idx = int(n_samples * (1 - validation_split))
    X_train, X_val = X_np[:split_idx], X_np[split_idx:]
    y_train, y_val = y_np[:split_idx], y_np[split_idx:]

    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Apply batching and optimization
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

def run_data_loading_examples():
    """Run various data loading examples"""

    print("=== JAX Data Loading Examples ===")

    # Example 1: MNIST with TensorFlow Datasets
    print("\n1. MNIST with TensorFlow Datasets:")
    train_ds, test_ds = setup_tf_dataset("mnist", batch_size=64)

    # Convert to JAX iterator
    jax_train_iter = tf_dataset_to_jax(train_ds)

    # Process first batch
    batch_x, batch_y = next(jax_train_iter)
    print(f"  Batch shape: X={batch_x.shape}, Y={batch_y.shape}")
    print(f"  Data types: X={batch_x.dtype}, Y={batch_y.dtype}")

    # Example 2: CIFAR-10 with preprocessing
    print("\n2. CIFAR-10 with preprocessing:")
    cifar_train, cifar_test, num_classes = create_image_classification_pipeline("cifar10")

    # Get first batch
    cifar_batch = next(iter(cifar_train))
    print(f"  CIFAR batch: X={cifar_batch[0].shape}, Y={cifar_batch[1].shape}")
    print(f"  Number of classes: {num_classes}")

    # Example 3: Synthetic regression data
    print("\n3. Synthetic regression data:")
    reg_train, reg_val = create_regression_pipeline(n_samples=1000, batch_size=32)

    reg_batch = next(iter(reg_train))
    print(f"  Regression batch: X={reg_batch[0].shape}, Y={reg_batch[1].shape}")

    # Example 4: Multi-device sharding
    print("\n4. Multi-device data sharding:")
    n_devices = jax.device_count()
    print(f"  Available devices: {n_devices}")

    if n_devices > 1:
        device_iter = create_device_iterator(tf_dataset_to_jax(train_ds), n_devices)
        sharded_x, sharded_y = next(device_iter)
        print(f"  Sharded X shape: {sharded_x.shape}")
        print(f"  Sharded Y shape: {sharded_y.shape}")

    # Example 5: Performance benchmarking
    print("\n5. Data loading performance:")
    performance = benchmark_data_loading(tf_dataset_to_jax(train_ds), n_batches=50)

# Run examples
run_data_loading_examples()
```

## Data Loading Best Practices

### Framework Selection
- **TensorFlow Datasets**: Mature ecosystem, many pre-built datasets
- **Grain**: Google's next-generation data loading, better performance
- **Custom sources**: For specialized data formats or preprocessing

### Performance Optimization
- Use `tf.data.AUTOTUNE` for automatic parallelism tuning
- Enable prefetching to overlap data loading and computation
- Cache datasets in memory when possible
- Use proper batch sizes for your hardware

### Device Management
- Shard data across multiple devices for distributed training
- Use explicit device placement for large datasets
- Monitor memory usage during data loading
- Consider gradient accumulation for large effective batch sizes

### Preprocessing Pipeline
- Apply data augmentation during training
- Normalize data appropriately for your model
- Handle different data types (images, text, structured data)
- Use vectorized operations for efficiency

## Common Issues and Solutions

### Memory Problems
- **Large datasets**: Use streaming and avoid loading everything into memory
- **Memory leaks**: Properly close dataset iterators and free resources
- **Device memory**: Monitor GPU memory usage during data loading

### Performance Issues
- **Slow loading**: Enable prefetching and parallel processing
- **CPU bottlenecks**: Increase number of parallel workers
- **I/O bound**: Use SSD storage and optimize file formats

### Data Format Issues
- **Shape mismatches**: Ensure consistent batch dimensions
- **Type mismatches**: Convert data types appropriately for JAX
- **Missing data**: Handle NaN values and missing entries properly

## Agent-Enhanced Data Processing Integration Patterns

### Complete Data Processing Workflow
```bash
# Intelligent data pipeline design and optimization
/jax-data-load --agents=auto --intelligent --framework=grain --optimize
/jax-training --agents=auto --optimizer=adam --data-efficient
/jax-performance --agents=data --technique=memory --optimization
```

### Scientific Data Processing Pipeline
```bash
# High-performance scientific data processing
/jax-data-load --agents=scientific --breakthrough --orchestrate
/jax-essentials --agents=scientific --operation=vmap --distributed
/run-all-tests --agents=scientific --data-validation --performance
```

### Production ML Data Infrastructure
```bash
# Large-scale production data processing optimization
/jax-data-load --agents=ai --distributed --optimize --breakthrough
/jax-models --agents=ai --framework=flax --data-integration
/ci-setup --agents=ai --data-monitoring --performance
```

## Related Commands

**Prerequisites**: Commands to run before data processing setup
- `/jax-init --agents=auto` - JAX project setup with data considerations
- `/jax-essentials --agents=auto` - Core JAX operations for data handling

**Core Workflow**: Data processing development with agent intelligence
- `/jax-training --agents=jax` - Training workflows with optimized data loading
- `/jax-performance --agents=data` - Data loading performance optimization
- `/jax-models --agents=auto` - Model integration with data pipelines

**Advanced Integration**: Specialized data processing development
- `/jax-sparse-ops --agents=scientific` - Sparse data processing optimization
- `/jax-numpyro-prob --agents=scientific` - Probabilistic data processing
- `/python-debug-prof --agents=data` - Data pipeline profiling and debugging

**Quality Assurance**: Data processing validation and monitoring
- `/generate-tests --agents=auto --type=data` - Generate data processing tests
- `/run-all-tests --agents=data --benchmark` - Comprehensive data processing testing
- `/optimize --agents=data --category=io` - Data I/O optimization with specialized agents

**Research & Documentation**: Advanced data processing workflows
- `/update-docs --agents=research --type=api` - Research-grade data processing documentation
- `/reflection --agents=data --type=scientific` - Data processing methodology analysis
- `/multi-agent-optimize --agents=all --focus=data` - Comprehensive data processing optimization

ARGUMENTS: [--framework=grain|tf] [--batch-size=32] [--shuffle] [--prefetch] [--agents=auto|jax|scientific|ai|data|all] [--orchestrate] [--intelligent] [--breakthrough] [--distributed] [--optimize]