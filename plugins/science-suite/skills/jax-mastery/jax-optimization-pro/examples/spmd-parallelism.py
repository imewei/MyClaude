"""
SPMD Parallelism Patterns for JAX

Demonstrates sharding, mesh configuration, and multi-device execution
for scaling JAX programs across GPUs and TPUs.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


# =============================================================================
# Pattern 1: Basic Device Placement
# =============================================================================

def show_available_devices():
    """Display available devices."""
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d}")
    return devices


def explicit_device_placement():
    """Explicitly place data on specific devices."""
    devices = jax.devices()

    # Place on first device
    x = jax.device_put(jnp.ones(1000), devices[0])
    print(f"x is on: {x.devices()}")

    # Replicate across all devices
    x_replicated = jax.device_put_replicated(jnp.ones(1000), devices)
    print(f"Replicated shape: {x_replicated.shape}")

    # Shard across devices (first axis)
    x_sharded = jax.device_put_sharded(
        [jnp.ones(1000) * i for i in range(len(devices))],
        devices
    )
    print(f"Sharded on: {x_sharded.devices()}")


# =============================================================================
# Pattern 2: pmap - Simple Data Parallelism
# =============================================================================

@jax.pmap
def pmap_train_step(params, batch):
    """Training step replicated across devices.

    Each device gets:
    - Full copy of params
    - Shard of batch (first axis split across devices)
    """
    def loss_fn(p, b):
        pred = p['w'] @ b['x'] + p['b']
        return jnp.mean((pred - b['y']) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    return loss, grads


def run_pmap_example():
    """Complete pmap training example."""
    n_devices = jax.device_count()
    print(f"Running pmap on {n_devices} devices")

    # Initialize params (replicated)
    params = {
        'w': jnp.ones((10, 10)),
        'b': jnp.zeros(10),
    }
    replicated_params = jax.device_put_replicated(params, jax.devices())

    # Create sharded batch (first axis = n_devices)
    batch_per_device = 32
    batch_per_device * n_devices
    batch = {
        'x': jnp.ones((n_devices, batch_per_device, 10)),
        'y': jnp.ones((n_devices, batch_per_device, 10)),
    }

    # Run parallel step
    losses, grads = pmap_train_step(replicated_params, batch)

    # Aggregate gradients (mean across devices)
    mean_grads = jax.tree.map(lambda g: g.mean(axis=0), grads)
    mean_loss = losses.mean()

    print(f"Mean loss: {mean_loss:.4f}")
    return mean_grads


# =============================================================================
# Pattern 3: Mesh and Sharding - Advanced Parallelism
# =============================================================================

def create_mesh_examples():
    """Different mesh configurations."""
    devices = jax.devices()
    n_devices = len(devices)

    # 1D mesh: data parallelism only
    if n_devices >= 1:
        mesh_1d = Mesh(devices, axis_names=('data',))
        print(f"1D mesh (data parallel): {mesh_1d.shape}")

    # 2D mesh: data + model parallelism (requires 4+ devices)
    if n_devices >= 4:
        devices_2d = mesh_utils.create_device_mesh((2, n_devices // 2))
        mesh_2d = Mesh(devices_2d, axis_names=('data', 'model'))
        print(f"2D mesh (data x model): {mesh_2d.shape}")

    # 3D mesh: data + model + pipeline (requires 8+ devices)
    if n_devices >= 8:
        devices_3d = mesh_utils.create_device_mesh((2, 2, n_devices // 4))
        mesh_3d = Mesh(devices_3d, axis_names=('data', 'model', 'pipeline'))
        print(f"3D mesh (data x model x pipeline): {mesh_3d.shape}")


def sharding_example():
    """Demonstrate different sharding patterns."""
    devices = jax.devices()
    n_devices = len(devices)

    if n_devices < 2:
        print("Need 2+ devices for sharding example")
        return

    # Create mesh
    mesh = Mesh(devices, axis_names=('devices',))

    with mesh:
        # Fully replicated (same data on all devices)
        replicated = NamedSharding(mesh, P())

        # Sharded on first axis
        sharded_0 = NamedSharding(mesh, P('devices'))

        # Sharded on second axis
        sharded_1 = NamedSharding(mesh, P(None, 'devices'))

        # Create data
        x = jnp.ones((n_devices * 32, 128))

        # Apply shardings
        x_rep = jax.device_put(x, replicated)
        x_sh0 = jax.device_put(x, sharded_0)
        jax.device_put(x[:, :n_devices * 16].reshape(n_devices * 32, -1),
                               sharded_1)

        print(f"Replicated: {x_rep.sharding}")
        print(f"Sharded axis 0: {x_sh0.sharding}")


# =============================================================================
# Pattern 4: Model Parallelism with Sharding
# =============================================================================

def model_parallel_matmul():
    """Distribute matrix multiplication across devices."""
    devices = jax.devices()
    n_devices = len(devices)

    if n_devices < 2:
        print("Need 2+ devices for model parallelism")
        return

    # 2D mesh if we have enough devices, else 1D
    if n_devices >= 4:
        device_mesh = mesh_utils.create_device_mesh((2, n_devices // 2))
        mesh = Mesh(device_mesh, axis_names=('data', 'model'))
        data_axis = 'data'
        model_axis = 'model'
    else:
        mesh = Mesh(devices, axis_names=('devices',))
        data_axis = 'devices'
        model_axis = None

    with mesh:
        # Input: sharded on batch dimension
        x_sharding = NamedSharding(mesh, P(data_axis, None))

        # Weight: sharded on output dimension (model parallel)
        if model_axis:
            w_sharding = NamedSharding(mesh, P(None, model_axis))
        else:
            w_sharding = NamedSharding(mesh, P(None, data_axis))

        # Create sharded data
        batch_size = n_devices * 32
        hidden_dim = 1024
        output_dim = 2048

        x = jax.device_put(
            jnp.ones((batch_size, hidden_dim)),
            x_sharding
        )
        w = jax.device_put(
            jnp.ones((hidden_dim, output_dim)) * 0.01,
            w_sharding
        )

        # JIT with output sharding constraint
        @jax.jit
        def parallel_matmul(x, w):
            return x @ w

        result = parallel_matmul(x, w)
        print(f"Input sharding: {x.sharding}")
        print(f"Weight sharding: {w.sharding}")
        print(f"Output sharding: {result.sharding}")
        print(f"Output shape: {result.shape}")


# =============================================================================
# Pattern 5: Sharding Constraints
# =============================================================================

def with_sharding_constraints():
    """Use sharding constraints to guide XLA partitioning."""
    devices = jax.devices()
    if len(devices) < 2:
        print("Need 2+ devices")
        return

    mesh = Mesh(devices, axis_names=('devices',))

    def forward(x, w1, w2):
        # First layer
        h = x @ w1
        h = jax.nn.relu(h)

        # Force specific sharding after first layer
        h = jax.lax.with_sharding_constraint(
            h,
            NamedSharding(mesh, P('devices', None))
        )

        # Second layer
        y = h @ w2
        return y

    with mesh:
        x = jnp.ones((len(devices) * 32, 128))
        w1 = jnp.ones((128, 256)) * 0.01
        w2 = jnp.ones((256, 64)) * 0.01

        # Shard input
        x_sharded = jax.device_put(
            x,
            NamedSharding(mesh, P('devices', None))
        )

        result = jax.jit(forward)(x_sharded, w1, w2)
        print(f"Result shape: {result.shape}")
        print(f"Result sharding: {result.sharding}")


# =============================================================================
# Pattern 6: Complete SPMD Training Example
# =============================================================================

def spmd_training_example():
    """Full SPMD training loop with proper sharding."""
    devices = jax.devices()
    n_devices = len(devices)

    print(f"\nSPMD Training on {n_devices} device(s)")

    # Configuration
    batch_size = n_devices * 32  # Batch per device * num devices
    hidden_dim = 512
    output_dim = 10

    # Create mesh
    mesh = Mesh(devices, axis_names=('batch',))

    # Define shardings
    data_sharding = NamedSharding(mesh, P('batch', None))
    replicated = NamedSharding(mesh, P())

    # Initialize model (replicated)
    def init_model(rng):
        return {
            'w1': jax.random.normal(rng, (784, hidden_dim)) * 0.01,
            'b1': jnp.zeros(hidden_dim),
            'w2': jax.random.normal(rng, (hidden_dim, output_dim)) * 0.01,
            'b2': jnp.zeros(output_dim),
        }

    rng = jax.random.PRNGKey(0)
    params = init_model(rng)

    # Replicate params across devices
    params = jax.device_put(params, replicated)

    def forward(params, x):
        h = jax.nn.relu(x @ params['w1'] + params['b1'])
        return h @ params['w2'] + params['b2']

    def loss_fn(params, x, y):
        logits = forward(params, x)
        return jnp.mean((logits - y) ** 2)

    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        # Simple SGD
        params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
        return params, loss

    # Training loop
    for step in range(5):
        # Create sharded batch
        x = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(step), (batch_size, 784)),
            data_sharding
        )
        y = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(step + 100), (batch_size, output_dim)),
            data_sharding
        )

        params, loss = train_step(params, x, y)
        print(f"Step {step}: loss = {loss:.4f}")

    print("SPMD training complete!")


# =============================================================================
# Pattern 7: Debugging Sharding
# =============================================================================

def debug_sharding(array, name="array"):
    """Print sharding information for debugging."""
    print(f"\n{name}:")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Sharding: {array.sharding}")
    print(f"  Devices: {array.devices()}")

    # Check if fully replicated
    if hasattr(array.sharding, 'is_fully_replicated'):
        print(f"  Fully replicated: {array.sharding.is_fully_replicated}")


def visualize_sharding():
    """Visualize how data is distributed."""
    devices = jax.devices()
    n_devices = len(devices)

    if n_devices < 2:
        print("Need 2+ devices to visualize sharding")
        return

    mesh = Mesh(devices, axis_names=('d',))

    with mesh:
        # Create data with visible values per shard
        data = jnp.arange(n_devices * 4).reshape(n_devices, 4)
        jax.device_put(data, NamedSharding(mesh, P('d', None)))

        print("Sharding visualization:")
        for i, device in enumerate(devices):
            print(f"  Device {i}: {data[i]}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPMD Parallelism Patterns")
    print("=" * 60)

    # Show devices
    show_available_devices()

    # Run examples
    print("\n" + "=" * 60)
    print("Running SPMD Training Example")
    print("=" * 60)
    spmd_training_example()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key SPMD Patterns:
1. jax.pmap - Simple data parallelism (legacy, but simple)
2. Mesh + NamedSharding - Modern sharding API
3. PartitionSpec (P) - Specify how axes map to mesh dimensions
4. with_sharding_constraint - Guide XLA's partitioner
5. jax.device_put with sharding - Explicit data placement

Sharding Specs:
- P() = replicated (same on all devices)
- P('axis') = shard first dim across 'axis'
- P(None, 'axis') = shard second dim across 'axis'
- P('a', 'b') = shard first dim on 'a', second on 'b'
""")
