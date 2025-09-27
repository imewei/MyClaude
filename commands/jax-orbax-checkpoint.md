---
description: Handle model checkpointing with Orbax including async operations and distributed training
category: jax-ml
argument-hint: "[--save] [--restore] [--async] [--distributed]"
allowed-tools: "*"
---

# /jax-orbax-checkpoint

Handle model checkpointing and persistence with Orbax.

## Description

Sets up model checkpointing using Orbax for saving and restoring JAX model states, parameters, and training progress. Supports async operations and distributed training.

## Usage

```
/jax-orbax-checkpoint [--save] [--restore] [--async] [--distributed]
```

## What it does

1. Save model parameters and training state
2. Restore models from checkpoints
3. Handle async and distributed checkpointing
4. Manage checkpoint versioning and cleanup

## Example output

```python
import orbax.checkpoint as ocp
from flax.training import train_state
import jax

# Basic checkpoint manager setup
checkpoint_dir = './checkpoints'
checkpointer = ocp.PyTreeCheckpointer()

# Save model state
def save_checkpoint(state, step, checkpoint_dir):
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(
        f'{checkpoint_dir}/checkpoint_{step}',
        state,
        save_args=save_args
    )

# Restore model state
def restore_checkpoint(checkpoint_path, target_state):
    restored_state = checkpointer.restore(checkpoint_path, item=target_state)
    return restored_state

# Advanced checkpoint manager with options
options = ocp.CheckpointManagerOptions(
    max_to_keep=5,              # Keep last 5 checkpoints
    save_interval_steps=100,    # Save every 100 steps
    async_save=True            # Enable async saving
)

checkpoint_manager = ocp.CheckpointManager(
    checkpoint_dir,
    checkpointers={
        'state': ocp.PyTreeCheckpointer(),
        'metadata': ocp.JsonCheckpointer()
    },
    options=options
)

# Save with checkpoint manager
def save_with_manager(state, step, metrics=None):
    checkpoint_manager.save(
        step,
        items={
            'state': state,
            'metadata': {
                'step': step,
                'metrics': metrics or {},
                'timestamp': time.time()
            }
        }
    )

# Restore latest checkpoint
def restore_latest(target_state):
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        restored = checkpoint_manager.restore(
            latest_step,
            items={'state': target_state}
        )
        return restored['state'], latest_step
    return target_state, 0

# Distributed checkpointing for multi-device training
def setup_distributed_checkpointing():
    # Use with jax.experimental.multihost_utils
    multihost_utils.sync_global_devices("checkpoint_sync")

    # Only save on host 0
    if jax.process_index() == 0:
        save_checkpoint(state, step, checkpoint_dir)

    # Sync all hosts
    multihost_utils.sync_global_devices("checkpoint_done")

# Custom save/restore for specific needs
class CustomCheckpointer:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpointer = ocp.PyTreeCheckpointer()

    def save_training_state(self, state, step, loss_history=None):
        checkpoint_data = {
            'params': state.params,
            'opt_state': state.opt_state,
            'step': step,
            'loss_history': loss_history or []
        }

        save_path = f'{self.checkpoint_dir}/train_state_{step}'
        self.checkpointer.save(save_path, checkpoint_data)

    def restore_training_state(self, step=None):
        if step is None:
            # Find latest checkpoint
            checkpoints = list(Path(self.checkpoint_dir).glob('train_state_*'))
            if not checkpoints:
                return None
            latest = max(checkpoints, key=lambda p: int(p.name.split('_')[-1]))
            save_path = str(latest)
        else:
            save_path = f'{self.checkpoint_dir}/train_state_{step}'

        return self.checkpointer.restore(save_path)

# Usage in training loop
def training_loop_with_checkpointing(state, train_loader, num_epochs):
    custom_checkpointer = CustomCheckpointer('./checkpoints')

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            state, loss = train_step(state, batch)

            # Save checkpoint every 1000 steps
            if step % 1000 == 0:
                custom_checkpointer.save_training_state(state, step)

    return state
```

## Related Commands

- `/jax-ml-train` - Integrate checkpointing in training loops
- `/jax-flax-model` - Save/restore Flax model parameters