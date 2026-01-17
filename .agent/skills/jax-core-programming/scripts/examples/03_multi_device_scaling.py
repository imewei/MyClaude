#!/usr/bin/env python3
"""
Workflow Pattern 3: Multi-Device Scaling

Demonstrates data and model parallelism using JAX Sharding API.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


def example_multi_device_scaling():
    """Complete multi-device scaling example"""

    print("=" * 60)
    print("Workflow Pattern 3: Multi-Device Scaling")
    print("=" * 60)

    # Check available devices
    devices = jax.devices()
    n_devices = len(devices)
    print(f"\nAvailable devices: {n_devices}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")

    if n_devices < 2:
        print("\nNote: Only 1 device available. Demonstrating API patterns.")
        print("      For actual multi-device benefits, run on multi-GPU/TPU setup.")

    # Step 1: Create device mesh
    print("\nStep 1: Create device mesh")

    # For demo: if we have 8 devices, create 4x2 mesh (data x model)
    # Otherwise, create 1D mesh
    if n_devices >= 8:
        mesh_shape = (4, 2)
        axis_names = ('data', 'model')
    elif n_devices >= 4:
        mesh_shape = (2, 2)
        axis_names = ('data', 'model')
    else:
        mesh_shape = (n_devices,)
        axis_names = ('data',)

    devices_array = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices_array, axis_names=axis_names)

    print(f"Device mesh shape: {mesh_shape}")
    print(f"Axis names: {axis_names}")

    # Step 2: Define sharding strategies
    print("\nStep 2: Define sharding strategies")

    if len(axis_names) == 2:
        # 2D mesh: data and model parallelism
        data_sharding = NamedSharding(mesh, P('data', None))
        model_sharding = NamedSharding(mesh, P(None, 'model'))
        full_sharding = NamedSharding(mesh, P('data', 'model'))
    else:
        # 1D mesh: data parallelism only
        data_sharding = NamedSharding(mesh, P('data'))
        model_sharding = NamedSharding(mesh, P(None))
        full_sharding = NamedSharding(mesh, P('data'))

    print("Sharding strategies created:")
    print(f"  Data sharding: shard along {axis_names[0]}")
    if len(axis_names) == 2:
        print(f"  Model sharding: shard along {axis_names[1]}")

    # Step 3: Shard data
    print("\nStep 3: Shard data across devices")

    batch_size = 128 * n_devices  # Scale batch with device count
    features = 512

    # Create data
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, features))
    y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10))

    # Shard data (each device gets a slice)
    x_sharded = jax.device_put(x, data_sharding)
    y_sharded = jax.device_put(y, data_sharding)

    print(f"Data shape: {x.shape}")
    print(f"Data sharding: {x_sharded.sharding}")
    print(f"Per-device data: {x.shape[0] // n_devices} examples")

    # Step 4: Shard model parameters
    print("\nStep 4: Shard model parameters")

    # Model parameters (large weight matrix)
    w1 = jax.random.normal(jax.random.PRNGKey(2), (features, 256))
    w2 = jax.random.normal(jax.random.PRNGKey(3), (256, 10))

    if len(axis_names) == 2:
        # Model parallelism: shard weights across model dimension
        w1_sharded = jax.device_put(w1, model_sharding)
        w2_sharded = jax.device_put(w2, model_sharding)
        print(f"Weight w1 shape: {w1.shape}")
        print(f"Weight w1 sharding: {w1_sharded.sharding}")
    else:
        # No model parallelism: replicate weights
        w1_sharded = jax.device_put(w1, NamedSharding(mesh, P(None)))
        w2_sharded = jax.device_put(w2, NamedSharding(mesh, P(None)))
        print(f"Weight w1 shape: {w1.shape}")
        print(f"Weight w1 sharding: replicated")

    # Step 5: Sharded computation
    print("\nStep 5: Sharded computation (automatic communication)")

    @jax.jit
    def sharded_forward(x, w1, w2):
        """Forward pass with automatic device communication"""
        # JAX inserts all-gather/reduce-scatter as needed
        hidden = x @ w1
        hidden = jax.nn.relu(hidden)
        output = hidden @ w2
        return output

    # Run sharded computation
    logits = sharded_forward(x_sharded, w1_sharded, w2_sharded)
    print(f"Output shape: {logits.shape}")
    print(f"Output sharding: {logits.sharding}")

    # Step 6: Gradient computation with sharding
    print("\nStep 6: Sharded gradient computation")

    @jax.jit
    def loss_fn(w1, w2, x, y):
        """Loss function with sharding"""
        logits = sharded_forward(x, w1, w2)
        return jnp.mean((logits - y) ** 2)

    # Compute gradients (sharded automatically)
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_w1, grad_w2 = grad_fn(w1_sharded, w2_sharded, x_sharded, y_sharded)

    print(f"Gradient w1 shape: {grad_w1.shape}")
    print(f"Gradient w1 sharding: {grad_w1.sharding}")

    # Step 7: Collective operations
    print("\nStep 7: Collective operations (all-reduce, all-gather)")

    @jax.jit
    def distributed_batch_stats(x):
        """Compute global batch statistics across devices"""
        # Local mean
        local_mean = jnp.mean(x, axis=0)

        # All-reduce to get global mean
        # JAX handles this automatically with proper sharding
        return local_mean

    batch_mean = distributed_batch_stats(x_sharded)
    print(f"Global batch mean computed across {n_devices} devices")
    print(f"Mean shape: {batch_mean.shape}")

    # Step 8: Training step with sharding
    print("\nStep 8: Complete sharded training step")

    @jax.jit
    def sharded_train_step(w1, w2, x, y, learning_rate=0.01):
        """Training step with automatic sharding"""

        def loss_fn_inner(w1, w2):
            logits = sharded_forward(x, w1, w2)
            return jnp.mean((logits - y) ** 2)

        # Compute loss and gradients
        loss, (grad_w1, grad_w2) = jax.value_and_grad(
            loss_fn_inner, argnums=(0, 1)
        )(w1, w2)

        # Update parameters (sharding preserved)
        w1 = w1 - learning_rate * grad_w1
        w2 = w2 - learning_rate * grad_w2

        return w1, w2, loss

    # Run training step
    w1_new, w2_new, loss = sharded_train_step(
        w1_sharded, w2_sharded, x_sharded, y_sharded
    )

    print(f"Loss: {loss:.4f}")
    print(f"Updated w1 sharding preserved: {w1_new.sharding == w1_sharded.sharding}")

    # Step 9: Performance comparison
    print("\nStep 9: Performance comparison")

    # Unsharded (single device)
    @jax.jit
    def unsharded_forward(x, w1, w2):
        hidden = x @ w1
        hidden = jax.nn.relu(hidden)
        return hidden @ w2

    # Benchmark
    import time

    # Warmup
    _ = unsharded_forward(x, w1, w2)
    _ = sharded_forward(x_sharded, w1_sharded, w2_sharded)

    # Time unsharded
    start = time.time()
    for _ in range(10):
        _ = unsharded_forward(x, w1, w2)
        jax.block_until_ready(_)
    unsharded_time = (time.time() - start) / 10

    # Time sharded
    start = time.time()
    for _ in range(10):
        _ = sharded_forward(x_sharded, w1_sharded, w2_sharded)
        jax.block_until_ready(_)
    sharded_time = (time.time() - start) / 10

    print(f"Unsharded time: {unsharded_time * 1000:.2f}ms")
    print(f"Sharded time: {sharded_time * 1000:.2f}ms")
    print(f"Speedup: {unsharded_time / sharded_time:.2f}x")

    # Step 10: Visualize sharding
    print("\nStep 10: Sharding visualization")
    print(f"x_sharded.addressable_shards: {len(x_sharded.addressable_shards)} shards")
    for i, shard in enumerate(x_sharded.addressable_shards[:3]):  # Show first 3
        print(f"  Shard {i}: device={shard.device}, shape={shard.data.shape}")

    print("\n✓ Multi-device scaling workflow complete!")
    print(f"Demonstrated {n_devices}-device parallelism with JAX Sharding API")


if __name__ == '__main__':
    example_multi_device_scaling()
