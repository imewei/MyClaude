---
description: Set up data loading pipelines with Grain or TF Datasets optimized for JAX workflows
category: jax-ml
argument-hint: "[--framework] [--batch-size] [--shuffle] [--prefetch]"
allowed-tools: "*"
---

# /jax-data-load

Set up efficient data loading for JAX machine learning workflows.

## Description

Configures data loading pipelines using Grain or TensorFlow Datasets, optimized for JAX. Handles batching, shuffling, preprocessing, and device placement.

## Usage

```
/jax-data-load [--framework] [--batch-size] [--shuffle] [--prefetch]
```

## What it does

1. Set up data loading with Grain or TF Datasets
2. Configure batching and shuffling strategies
3. Optimize data pipeline for JAX training
4. Handle device placement and memory management

## Example output

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np

# Basic TensorFlow Datasets setup for JAX
def load_dataset_tfds(dataset_name='mnist', batch_size=32, split='train'):
    \"\"\"Load dataset using TensorFlow Datasets for JAX.\"\"\"
    ds = tfds.load(dataset_name, split=split, as_supervised=True)

    def preprocess(image, label):
        # Convert to float32 and normalize
        image = tf.cast(image, tf.float32) / 255.0
        # Flatten for simple models (remove for CNNs)
        image = tf.reshape(image, [-1])
        return image, label

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# Convert TF dataset to JAX arrays
def tf_to_jax_iterator(tf_dataset):
    \"\"\"Convert TensorFlow dataset to JAX-compatible iterator.\"\"\"
    for batch in tf_dataset:
        x, y = batch
        yield jnp.array(x.numpy()), jnp.array(y.numpy())

# Grain data loading (Google's JAX-native data loader)
def setup_grain_loader(data_source, batch_size=32):
    \"\"\"Set up Grain data loader for JAX.\"\"\"
    import grain.tensorflow as grain_tf

    # Create data source
    source = grain_tf.TfDataSource(data_source)

    # Define transformations
    transformations = [
        grain_tf.Batch(batch_size),
        grain_tf.Shuffle(buffer_size=1000),
    ]

    # Create loader
    loader = grain_tf.TfDataLoader(
        source=source,
        transformations=transformations
    )

    return loader

# Custom data loader for NumPy arrays
class JAXDataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = jnp.array(X)
        self.y = jnp.array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(X)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            key = jax.random.PRNGKey(np.random.randint(0, 2**32))
            indices = jax.random.permutation(key, self.num_samples)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
        else:
            X_shuffled, y_shuffled = self.X, self.y

        for i in range(0, self.num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_samples)
            batch_X = X_shuffled[i:end_idx]
            batch_y = y_shuffled[i:end_idx]
            yield batch_X, batch_y

    def __len__(self):
        return self.num_batches

# Data loading with device placement
def create_device_data_loader(dataset, devices=None):
    \"\"\"Create data loader that distributes data across devices.\"\"\"
    if devices is None:
        devices = jax.devices()

    def device_put_batch(batch):
        x, y = batch
        # Replicate across devices for pmap
        x_devices = jax.device_put_replicated(x, devices)
        y_devices = jax.device_put_replicated(y, devices)
        return x_devices, y_devices

    for batch in dataset:
        yield device_put_batch(batch)

# Optimized data loading pipeline
def create_optimized_pipeline(dataset_name, batch_size=32, prefetch_size=2):
    \"\"\"Create highly optimized data loading pipeline.\"\"\"
    ds = tfds.load(dataset_name, split='train', as_supervised=True)

    # Preprocessing pipeline
    def preprocess_fn(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        # Add any data augmentation here
        return image, label

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # Cache after preprocessing
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch_size)

    return ds

# Usage example
def setup_data_for_training():
    # Load training data
    train_ds = load_dataset_tfds('mnist', batch_size=32, split='train')
    val_ds = load_dataset_tfds('mnist', batch_size=32, split='test')

    # Convert to JAX iterators
    train_iter = tf_to_jax_iterator(train_ds)
    val_iter = tf_to_jax_iterator(val_ds)

    return train_iter, val_iter

# Integration with training loop
def train_with_data_loader(state, data_loader, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in data_loader:
            state, loss = train_step(state, (batch_x, batch_y))
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f\"Epoch {epoch}: Average Loss = {avg_loss:.4f}\")

    return state
```

## Related Commands

- `/jax-ml-train` - Use data loaders in training loops
- `/jax-pmap` - Distribute data across multiple devices
- `/jax-flax-model` - Process data through neural networks