# Orbax Checkpointing - Complete Guide

## Overview

Orbax provides robust checkpointing for JAX models with async saves, multi-host support, and disaster recovery.

---

## 1. Basic Checkpointing

### Simple Save/Load

```python
import orbax.checkpoint as ocp
import jax.numpy as jnp

# Create checkpointer
checkpointer = ocp.PyTreeCheckpointer()

# Save checkpoint
params = {'w': jnp.ones((10, 5)), 'b': jnp.zeros(10)}
checkpointer.save('/tmp/checkpoint', params)

# Restore checkpoint
restored_params = checkpointer.restore('/tmp/checkpoint')
```

### With Metadata

```python
# Save with metadata
checkpointer.save(
    '/tmp/checkpoint',
    params,
    save_args=ocp.args.StandardSave(metadata={'step': 1000, 'loss': 0.123})
)

# Restore with metadata
restored = checkpointer.restore(
    '/tmp/checkpoint',
    restore_args=ocp.args.StandardRestore()
)
```

---

## 2. Async Checkpointing

### Non-Blocking Saves

```python
from orbax.checkpoint import AsyncCheckpointer

# Create async checkpointer (doesn't block training)
checkpointer = AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

# Training loop
for step in range(10000):
    # Training step
    params, loss = train_step(params, batch)

    # Save checkpoint without blocking
    if step % 1000 == 0:
        checkpointer.save(
            f'/checkpoints/step_{step}',
            args=ocp.args.StandardSave(params)
        )
        # Training continues immediately

# Wait for all saves to complete
checkpointer.wait_until_finished()
```

### With Queue Management

```python
# Configure async behavior
checkpointer = AsyncCheckpointer(
    ocp.PyTreeCheckpointHandler(),
    timeout_secs=300  # Max wait time
)

# Save with priority
checkpointer.save(
    '/checkpoints/important',
    args=ocp.args.StandardSave(params),
    force=True  # Higher priority
)
```

---

## 3. Checkpoint Managers

### Automatic Cleanup

```python
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions

# Create manager with automatic cleanup
options = CheckpointManagerOptions(
    max_to_keep=3,  # Keep only 3 most recent checkpoints
    keep_time_interval=3600,  # Keep checkpoints every hour
    keep_period=5,  # Keep every 5th checkpoint
    create=True
)

manager = CheckpointManager(
    '/checkpoints',
    checkpointer,
    options=options
)

# Save checkpoint (automatic cleanup)
for step in range(10000):
    params, loss = train_step(params, batch)

    if step % 1000 == 0:
        manager.save(
            step,
            args=ocp.args.StandardSave(params),
            metrics={'loss': float(loss)}
        )

# Restore latest checkpoint
latest_step = manager.latest_step()
params = manager.restore(latest_step)
```

### Best Checkpoint Tracking

```python
# Save best checkpoint based on metric
manager = CheckpointManager(
    '/checkpoints',
    checkpointer,
    options=CheckpointManagerOptions(
        max_to_keep=3,
        best_fn=lambda metrics: metrics['accuracy'],  # Keep best accuracy
        best_mode='max'
    )
)

# Save with metrics
manager.save(step, args=ocp.args.StandardSave(params), metrics={'accuracy': 0.95})
```

---

## 4. Multi-Host Checkpointing

### Distributed Save/Load

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Setup distributed environment
devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('data', 'model'))

# Shard parameters
sharding = NamedSharding(mesh, P('model', None))
params_sharded = jax.device_put(params, sharding)

# Multi-host checkpointer
checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

# Save sharded parameters (each host saves its shard)
checkpointer.save(
    '/distributed_checkpoint',
    args=ocp.args.StandardSave(params_sharded)
)

# Restore to same sharding
restored_params = checkpointer.restore(
    '/distributed_checkpoint',
    args=ocp.args.StandardRestore(params_sharded)
)
```

---

## 5. Flax NNX Integration

### Save/Restore NNX Models

```python
from flax import nnx
import orbax.checkpoint as ocp

# Create model
model = nnx.Linear(784, 10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# Save model and optimizer
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/checkpoint', {
    'model': nnx.state(model),
    'optimizer': optimizer.opt_state,
    'step': step
})

# Restore
ckpt = checkpointer.restore('/checkpoint')
model = nnx.merge(model, ckpt['model'])
optimizer.opt_state = ckpt['optimizer']
step = ckpt['step']
```

---

## 6. Advanced Patterns

### Partial Restoration

```python
# Restore only specific parameters
restore_args = ocp.args.StandardRestore(
    item={'model': model_state},  # Only restore model, not optimizer
)
restored = checkpointer.restore('/checkpoint', restore_args=restore_args)
```

### Custom Serialization

```python
class CustomHandler(ocp.CheckpointHandler):
    def save(self, directory, item):
        # Custom save logic
        pass

    def restore(self, directory):
        # Custom restore logic
        pass

checkpointer = ocp.Checkpointer(CustomHandler())
```

### Checkpoint Conversion

```python
# Convert between checkpoint formats
old_ckpt = checkpointer.restore('/old_checkpoint')
new_ckpt = convert_params(old_ckpt)
checkpointer.save('/new_checkpoint', new_ckpt)
```

---

## 7. Production Patterns

### Complete Training Setup

```python
import orbax.checkpoint as ocp
from flax import nnx
import optax

def setup_checkpointing(checkpoint_dir, model, optimizer):
    """Setup production checkpointing"""

    # Async checkpointer
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    # Manager with automatic cleanup
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        keep_time_interval=3600,  # Keep hourly
        create=True
    )

    manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointer,
        options=options
    )

    # Restore if checkpoint exists
    if manager.latest_step() is not None:
        latest_step = manager.latest_step()
        ckpt = manager.restore(latest_step)

        model = nnx.merge(model, ckpt['model'])
        optimizer.opt_state = ckpt['optimizer']
        start_step = ckpt['step']
        print(f"Restored from step {start_step}")
    else:
        start_step = 0

    return manager, start_step


# Usage in training
manager, start_step = setup_checkpointing('/checkpoints', model, optimizer)

for step in range(start_step, num_steps):
    params, loss = train_step(params, batch)

    # Periodic checkpoint
    if step % 1000 == 0:
        manager.save(
            step,
            args=ocp.args.StandardSave({
                'model': nnx.state(model),
                'optimizer': optimizer.opt_state,
                'step': step
            }),
            metrics={'loss': float(loss)}
        )

manager.wait_until_finished()
```

### Disaster Recovery

```python
def safe_checkpoint_save(manager, step, params, max_retries=3):
    """Save with retry logic"""
    for attempt in range(max_retries):
        try:
            manager.save(
                step,
                args=ocp.args.StandardSave(params)
            )
            return True
        except Exception as e:
            print(f"Checkpoint save failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                # Save to backup location
                backup_path = f'/backup/emergency_checkpoint_{step}'
                manager.save(backup_path, args=ocp.args.StandardSave(params))
    return False
```

---

## 8. Performance Optimization

### Checkpoint Size Reduction

```python
# Use compression
checkpointer = ocp.AsyncCheckpointer(
    ocp.PyTreeCheckpointHandler(),
    compression='zstd'  # Fast compression
)

# Save only trainable params
trainable_params = extract_trainable(model)
checkpointer.save('/checkpoint', trainable_params)
```

### Faster Saves

```python
# Optimize I/O
options = ocp.CheckpointManagerOptions(
    max_to_keep=3,
    create=True,
    enable_async_checkpointing=True,  # Enable async
    write_tree_metadata=False  # Skip metadata for speed
)
```

---

## 9. Common Patterns

### Resume Training

```python
def resume_training(checkpoint_dir, model, optimizer):
    """Resume from latest checkpoint or start fresh"""
    manager = ocp.CheckpointManager(
        checkpoint_dir,
        ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    )

    if manager.latest_step() is not None:
        step = manager.latest_step()
        ckpt = manager.restore(step)
        model = nnx.merge(model, ckpt['model'])
        optimizer.opt_state = ckpt['optimizer']
        print(f"Resumed from step {step}")
        return model, optimizer, step + 1

    print("Starting fresh training")
    return model, optimizer, 0
```

### Export to SavedModel

```python
# Export for serving
from orbax.export import ExportManager

export_manager = ExportManager(model, serving_config)
export_manager.save('/serving/model')
```

---

## 10. Best Practices

### Checklist

```python
"""
Checkpointing Best Practices:

1. Use AsyncCheckpointer for non-blocking saves
2. Use CheckpointManager for automatic cleanup
3. Save every N steps and at end of training
4. Keep both recent and best checkpoints
5. Save optimizer state for exact resumption
6. Test restoration logic regularly
7. Use multi-host for distributed training
8. Monitor checkpoint size and save time
9. Implement retry logic for reliability
10. Version checkpoint formats
"""
```

### Configuration Template

```python
CHECKPOINT_CONFIG = {
    'checkpoint_dir': '/checkpoints',
    'save_interval_steps': 1000,
    'max_to_keep': 3,
    'keep_time_interval': 3600,  # 1 hour
    'async_checkpointing': True,
    'compression': 'zstd',
    'timeout_secs': 600,
}

def create_checkpoint_manager(config):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config['max_to_keep'],
        keep_time_interval=config['keep_time_interval'],
        create=True
    )

    checkpointer = ocp.AsyncCheckpointer(
        ocp.PyTreeCheckpointHandler(),
        timeout_secs=config['timeout_secs']
    )

    return ocp.CheckpointManager(
        config['checkpoint_dir'],
        checkpointer,
        options=options
    )
```

---

## References

- [Orbax Documentation](https://orbax.readthedocs.io/)
- [Orbax GitHub](https://github.com/google/orbax)
- [JAX Checkpointing Guide](https://jax.readthedocs.io/en/latest/jax-101/06-checkpointing.html)
