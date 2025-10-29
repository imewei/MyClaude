#!/usr/bin/env python3
"""
Workflow Pattern 2: Production Flax NNX Training

Complete production-ready training pipeline with Flax NNX, Optax, and Orbax.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import tempfile
import os


class MLP(nnx.Module):
    """Multi-layer perceptron for classification"""

    def __init__(self, hidden_dim: int, num_classes: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(784, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim // 2, num_classes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.2, rngs=rngs)

    def __call__(self, x, train: bool = False):
        x = self.linear1(x)
        x = nnx.relu(x)
        if train:
            x = self.dropout(x)

        x = self.linear2(x)
        x = nnx.relu(x)
        if train:
            x = self.dropout(x)

        x = self.linear3(x)
        return x


def create_dataset(batch_size=32, num_batches=10):
    """Create synthetic dataset"""
    rng = jax.random.PRNGKey(0)

    for _ in range(num_batches):
        x_key, y_key, rng = jax.random.split(rng, 3)
        x = jax.random.normal(x_key, (batch_size, 784))
        y = jax.random.randint(y_key, (batch_size,), 0, 10)
        yield {'x': x, 'y': y}


def example_production_training():
    """Complete production training example"""

    print("=" * 60)
    print("Workflow Pattern 2: Production Flax NNX Training")
    print("=" * 60)

    # Configuration
    config = {
        'hidden_dim': 128,
        'num_classes': 10,
        'learning_rate': 1e-3,
        'warmup_steps': 100,
        'decay_steps': 1000,
        'weight_decay': 0.01,
        'num_epochs': 3,
        'batch_size': 32,
        'checkpoint_interval': 50,
    }

    # Step 1: Model initialization
    print("\nStep 1: Initialize model")
    model = MLP(
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        rngs=nnx.Rngs(0)
    )
    print(f"Model created with {sum(p.size for p in jax.tree_leaves(nnx.state(model)))} parameters")

    # Step 2: Optimizer with learning rate schedule
    print("\nStep 2: Create optimizer with LR schedule")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        decay_steps=config['decay_steps'],
        end_value=config['learning_rate'] * 0.1
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config['weight_decay']
        )
    )
    print(f"Optimizer initialized with AdamW")

    # Step 3: Setup async checkpointing
    print("\nStep 3: Setup async checkpointing")
    checkpoint_dir = tempfile.mkdtemp()
    print(f"Checkpoint dir: {checkpoint_dir}")

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointer,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=3,
            create=True
        )
    )

    # Step 4: Define training step
    print("\nStep 4: Define JIT-compiled training step")

    @nnx.jit
    def train_step(optimizer, batch):
        """Single training step with JIT compilation"""

        def loss_fn(model):
            logits = model(batch['x'], train=True)
            # Cross-entropy loss
            one_hot = jax.nn.one_hot(batch['y'], num_classes=config['num_classes'])
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
        optimizer.update(grads)
        return loss

    # Step 5: Define evaluation step
    @nnx.jit
    def eval_step(model, batch):
        """Evaluation step"""
        logits = model(batch['x'], train=False)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch['y'])
        return accuracy

    # Step 6: Training loop
    print("\nStep 6: Training loop")
    print("-" * 60)

    step = 0
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Training
        epoch_losses = []
        for batch in create_dataset(config['batch_size'], num_batches=10):
            loss = train_step(optimizer, batch)
            epoch_losses.append(float(loss))
            step += 1

            # Log progress
            if step % 20 == 0:
                current_lr = schedule(step)
                print(f"  Step {step}: Loss = {loss:.4f}, LR = {current_lr:.6f}")

            # Periodic checkpoint
            if step % config['checkpoint_interval'] == 0:
                print(f"  Saving checkpoint at step {step}...")
                manager.save(
                    step,
                    args=ocp.args.StandardSave({
                        'model': nnx.state(optimizer.model),
                        'optimizer': optimizer.opt_state,
                        'step': step
                    }),
                    metrics={'loss': float(loss)}
                )

        # Epoch summary
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        # Evaluation
        accuracies = []
        for batch in create_dataset(config['batch_size'], num_batches=5):
            acc = eval_step(optimizer.model, batch)
            accuracies.append(float(acc))

        avg_accuracy = jnp.mean(jnp.array(accuracies))
        print(f"  Epoch {epoch + 1} accuracy: {avg_accuracy:.4f}")

    # Step 7: Final checkpoint
    print("\nStep 7: Save final checkpoint")
    manager.save(
        step,
        args=ocp.args.StandardSave({
            'model': nnx.state(optimizer.model),
            'optimizer': optimizer.opt_state,
            'step': step
        }),
        metrics={'loss': float(loss), 'accuracy': float(avg_accuracy)}
    )

    # Wait for async saves
    manager.wait_until_finished()
    print(f"Final checkpoint saved at step {step}")

    # Step 8: Demonstrate checkpoint restoration
    print("\nStep 8: Demonstrate checkpoint restoration")

    # Create fresh model
    new_model = MLP(
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        rngs=nnx.Rngs(42)  # Different seed
    )

    # Restore from checkpoint
    latest_step = manager.latest_step()
    ckpt = manager.restore(latest_step)

    new_model = nnx.merge(new_model, ckpt['model'])
    print(f"✓ Restored model from step {ckpt['step']}")

    # Verify restored model works
    test_batch = next(create_dataset(config['batch_size'], num_batches=1))
    test_acc = eval_step(new_model, test_batch)
    print(f"✓ Restored model accuracy: {test_acc:.4f}")

    # Cleanup
    import shutil
    shutil.rmtree(checkpoint_dir)
    print(f"\n✓ Production training workflow complete!")
    print(f"Trained for {step} steps across {config['num_epochs']} epochs")


if __name__ == '__main__':
    example_production_training()
