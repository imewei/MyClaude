---
description: Handle model checkpointing with Orbax including async operations and distributed training
category: jax-ml
argument-hint: "[--save] [--restore] [--async] [--distributed] [--agents=auto|jax|scientific|ai|production|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--mlops]"
allowed-tools: "*"
model: inherit
---

# JAX Orbax Checkpoint

Handle model checkpointing with Orbax including async operations and distributed training.

```bash
/jax-orbax-checkpoint [--save] [--restore] [--async] [--distributed] [--agents=auto|jax|scientific|ai|production|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--mlops]
```

## Options

- `--save`: Enable model parameter saving
- `--restore`: Enable model parameter restoration
- `--async`: Use asynchronous checkpointing operations
- `--distributed`: Enable distributed training checkpointing
- `--agents=<agents>`: Agent selection (auto, jax, scientific, ai, production, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with checkpoint intelligence
- `--intelligent`: Enable intelligent agent selection based on checkpoint analysis
- `--breakthrough`: Enable breakthrough checkpoint optimization and management
- `--optimize`: Apply performance optimization to checkpoint operations
- `--mlops`: Advanced MLOps integration with production checkpoint workflows

## What it does

1. **Model Checkpointing**: Save and restore JAX model parameters with Orbax
2. **Training State Management**: Handle optimizer state, metrics, and training progress
3. **Async Operations**: Non-blocking checkpoint saves for better performance
4. **Distributed Support**: Multi-device and multi-host checkpointing
5. **Checkpoint Management**: Versioning, cleanup, and best checkpoints
6. **23-Agent Checkpoint Intelligence**: Multi-agent collaboration for optimal checkpoint strategies
7. **Production MLOps**: Agent-driven production model lifecycle management
8. **Advanced Optimization**: Agent-coordinated checkpoint performance and reliability

## 23-Agent Intelligent Checkpoint System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes checkpoint requirements, model complexity, and production needs to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Checkpoint Use Case Detection → Agent Selection
- Production ML Systems → ai-systems-architect + systems-architect + jax-pro
- Research Model Versioning → research-intelligence-master + scientific-computing-master + jax-pro
- Large-Scale Training → ai-systems-architect + neural-networks-master + multi-agent-orchestrator
- Distributed Training → systems-architect + ai-systems-architect + jax-pro
- Scientific Computing → scientific-computing-master + research-intelligence-master + jax-pro
```

### Core JAX Checkpoint Management Agents

#### **`ai-systems-architect`** - Production ML Checkpoint Systems
- **MLOps Integration**: Production model lifecycle and checkpoint management
- **Deployment Automation**: Automated checkpoint deployment and rollback strategies
- **System Reliability**: Fault-tolerant checkpoint systems and disaster recovery
- **Scalability Engineering**: Large-scale model checkpoint architecture
- **Performance Optimization**: Production checkpoint performance and resource management

#### **`jax-pro`** - JAX Checkpoint Optimization Expert
- **Orbax Mastery**: Deep expertise in Orbax checkpoint optimization and best practices
- **JAX Integration**: Optimal JAX transformation integration with checkpoint operations
- **Memory Efficiency**: JAX-specific checkpoint memory optimization and management
- **Device Management**: Multi-device and GPU/TPU checkpoint coordination
- **Performance Engineering**: JAX checkpoint performance optimization and acceleration

#### **`systems-architect`** - Checkpoint Infrastructure & Performance
- **Storage Architecture**: Optimal checkpoint storage systems and infrastructure design
- **Resource Management**: Computational and storage resource optimization for checkpoints
- **Network Optimization**: Distributed checkpoint transfer and synchronization
- **Infrastructure Monitoring**: Real-time checkpoint system performance tracking
- **Fault Tolerance**: Robust checkpoint systems with failure recovery and resilience

#### **`neural-networks-master`** - ML Model Checkpoint Expertise
- **Model State Management**: Neural network state checkpointing and restoration strategies
- **Training Optimization**: Checkpoint integration with training loop optimization
- **Model Versioning**: Neural network model version management and compatibility
- **Large Model Checkpoints**: Checkpoint strategies for large-scale neural networks
- **Training Recovery**: Advanced training resumption and state recovery techniques

### Specialized Checkpoint Agents

#### **`research-intelligence-master`** - Research Checkpoint Standards
- **Reproducibility**: Research-grade checkpoint reproducibility and validation
- **Experimental Tracking**: Advanced experiment checkpoint management and organization
- **Academic Standards**: Checkpoint management for academic and research publication
- **Innovation Framework**: Checkpoint strategies for breakthrough research workflows
- **Collaboration**: Multi-researcher checkpoint sharing and version control

#### **`scientific-computing-master`** - Scientific Computing Checkpoints
- **Computational Workflows**: Checkpoint management for scientific computing pipelines
- **Simulation Checkpoints**: Long-running simulation checkpoint and recovery strategies
- **Research Computing**: High-performance computing checkpoint optimization
- **Domain Integration**: Checkpoint strategies for specific scientific domains
- **Performance Computing**: Scientific computing checkpoint performance optimization

#### **`data-professional`** - Data & Checkpoint Integration
- **Data Pipeline Checkpoints**: Checkpoint integration with data processing workflows
- **Model-Data Versioning**: Coordinated model and data checkpoint management
- **Data Consistency**: Data-model checkpoint consistency and validation
- **Pipeline Recovery**: Data pipeline checkpoint and recovery strategies
- **Data Governance**: Checkpoint data governance and compliance management

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Checkpoint Management
Automatically analyzes checkpoint requirements and selects optimal agent combinations:
- **Use Case Analysis**: Detects production vs research, scale, complexity requirements
- **Performance Assessment**: Evaluates computational and storage constraints
- **Agent Matching**: Maps checkpoint challenges to relevant agent expertise
- **Optimization Focus**: Balances comprehensive checkpoint management with efficiency

#### **`jax`** - JAX-Specialized Checkpoint Team
- `jax-pro` (JAX ecosystem lead)
- `ai-systems-architect` (ML system integration)
- `systems-architect` (infrastructure)
- `neural-networks-master` (model management)

#### **`scientific`** - Scientific Computing Checkpoint Team
- `scientific-computing-master` (lead)
- `research-intelligence-master` (research methodology)
- `jax-pro` (JAX implementation)
- `systems-architect` (system performance)
- Domain-specific experts based on scientific application

#### **`ai`** - AI/ML Production Checkpoint Team
- `ai-systems-architect` (lead)
- `neural-networks-master` (ML checkpoint strategies)
- `jax-pro` (JAX optimization)
- `systems-architect` (infrastructure)
- `data-professional` (data integration)

#### **`production`** - Production Checkpoint Management Team
- `ai-systems-architect` (lead)
- `systems-architect` (infrastructure)
- `jax-pro` (JAX optimization)
- `neural-networks-master` (model management)
- `data-professional` (data coordination)

#### **`all`** - Complete 23-Agent Checkpoint Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough checkpoint management.

### 23-Agent Checkpoint Orchestration (`--orchestrate`)

#### **Multi-Agent Checkpoint Pipeline**
1. **Requirements Analysis Phase**: Multiple agents analyze checkpoint requirements simultaneously
2. **Strategy Development**: Collaborative checkpoint strategy development across domains
3. **Implementation Optimization**: Agent-coordinated checkpoint system implementation
4. **Performance Monitoring**: Multi-agent checkpoint performance and reliability tracking
5. **Recovery Planning**: Comprehensive disaster recovery and checkpoint validation

#### **Breakthrough Checkpoint Innovation (`--breakthrough`)**
- **Cross-Domain Techniques**: Checkpoint strategies from multiple domains and applications
- **Emergent Optimization**: Novel checkpoint approaches through agent collaboration
- **Production-Grade Standards**: Industry-leading checkpoint reliability and performance
- **Adaptive Management**: Dynamic checkpoint strategy optimization based on usage patterns

### Advanced 23-Agent Checkpoint Examples

```bash
# Intelligent auto-selection for checkpoint management
/jax-orbax-checkpoint --agents=auto --intelligent --save --optimize

# Production ML checkpoint systems with specialized agents
/jax-orbax-checkpoint --agents=ai --mlops --breakthrough --orchestrate

# Scientific computing checkpoint optimization
/jax-orbax-checkpoint --agents=scientific --async --optimize --breakthrough

# Research-grade checkpoint development
/jax-orbax-checkpoint --agents=all --breakthrough --orchestrate --mlops

# JAX-specialized checkpoint optimization
/jax-orbax-checkpoint --agents=jax --distributed --optimize --intelligent

# Complete 23-agent checkpoint ecosystem
/jax-orbax-checkpoint --agents=all --orchestrate --breakthrough --intelligent

# Large-scale distributed training checkpoints
/jax-orbax-checkpoint distributed_training.py --agents=ai --distributed --orchestrate

# Research model versioning and reproducibility
/jax-orbax-checkpoint research_model.py --agents=scientific --intelligent --mlops

# Production model deployment checkpoints
/jax-orbax-checkpoint production_model.py --agents=production --optimize --breakthrough

# High-performance checkpoint optimization
/jax-orbax-checkpoint hpc_model.py --agents=all --optimize --breakthrough

# Scientific simulation checkpoint management
/jax-orbax-checkpoint simulation.py --agents=scientific --async --intelligent

# MLOps checkpoint automation
/jax-orbax-checkpoint mlops_pipeline.py --agents=ai --mlops --orchestrate
```

### Intelligent Agent Selection Examples

```bash
# Checkpoint Use Case Detection → Intelligent Agent Selection

# Production ML system checkpoints
/jax-orbax-checkpoint production_ml.py --agents=auto --intelligent
# → Selects: ai-systems-architect + systems-architect + jax-pro

# Research model development checkpoints
/jax-orbax-checkpoint research_dev.py --agents=auto --intelligent
# → Selects: research-intelligence-master + scientific-computing-master + jax-pro

# Large-scale distributed training
/jax-orbax-checkpoint distributed_large.py --agents=auto --intelligent
# → Selects: ai-systems-architect + neural-networks-master + multi-agent-orchestrator

# Scientific computing simulation checkpoints
/jax-orbax-checkpoint simulation_checkpoint.py --agents=auto --intelligent
# → Selects: scientific-computing-master + systems-architect + research-intelligence-master

# Complex multi-model checkpoint system
/jax-orbax-checkpoint complex_system.py --agents=auto --intelligent
# → Selects: ai-systems-architect + multi-agent-orchestrator + neural-networks-master

# Comprehensive checkpoint platform
/jax-orbax-checkpoint checkpoint_platform.py --agents=all --orchestrate --breakthrough
# → Activates: All 23 agents with intelligent coordination for maximum checkpoint effectiveness
```

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
import orbax.checkpoint as ocp
from flax.training import train_state
import flax.linen as nn
import optax
import asyncio
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import time

# Initialize PRNG key
key = random.PRNGKey(0)

# ============================================================================
# 1. BASIC ORBAX CHECKPOINT SETUP
# ============================================================================

def setup_checkpoint_manager(checkpoint_dir: str, max_to_keep: int = 5):
    """Set up Orbax checkpoint manager"""

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint manager
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True
    )

    checkpoint_manager = ocp.CheckpointManager(
        directory=checkpoint_dir,
        checkpointers=ocp.StandardCheckpointer(),
        options=options
    )

    return checkpoint_manager

def create_sample_model_and_state():
    """Create a sample model and training state for demonstration"""

    # Simple MLP model
    class MLP(nn.Module):
        hidden_dims: list
        output_dim: int

        @nn.compact
        def __call__(self, x):
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)
            x = nn.Dense(self.output_dim)(x)
            return x

    # Initialize model
    model = MLP(hidden_dims=[64, 32], output_dim=10)

    # Create dummy input
    dummy_input = jnp.ones((1, 20))

    # Initialize parameters
    key, subkey = random.split(key)
    params = model.init(subkey, dummy_input)

    # Create optimizer
    optimizer = optax.adam(learning_rate=0.001)

    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    return model, state

# ============================================================================
# 2. BASIC SAVE AND RESTORE OPERATIONS
# ============================================================================

def save_checkpoint(checkpoint_manager, state, step: int, metrics: Optional[Dict] = None):
    """Save checkpoint with training state"""

    # Prepare checkpoint data
    checkpoint_data = {
        'model': state,
        'step': step
    }

    # Add metrics if provided
    if metrics is not None:
        checkpoint_data['metrics'] = metrics

    # Save checkpoint
    save_args = orbax.checkpoint.args.StandardSave(checkpoint_data)
    checkpoint_manager.save(step, checkpoint_data, save_kwargs={'save_args': save_args})

    print(f"Checkpoint saved at step {step}")

def restore_checkpoint(checkpoint_manager, step: Optional[int] = None):
    """Restore checkpoint from training state"""

    if step is None:
        # Restore latest checkpoint
        step = checkpoint_manager.latest_step()

    if step is None:
        print("No checkpoints found")
        return None

    # Create restore args
    # First we need to get the structure
    abstract_state = checkpoint_manager.restore(step, args=ocp.args.StandardRestore())

    print(f"Checkpoint restored from step {step}")
    return abstract_state

def list_available_checkpoints(checkpoint_manager):
    """List all available checkpoints"""

    steps = checkpoint_manager.all_steps()

    if not steps:
        print("No checkpoints found")
        return []

    print("Available checkpoints:")
    for step in sorted(steps):
        checkpoint_info = checkpoint_manager.item_metadata(step)
        print(f"  Step {step}: {checkpoint_info}")

    return steps

# ============================================================================
# 3. ADVANCED CHECKPOINT FEATURES
# ============================================================================

def save_best_checkpoint(checkpoint_manager, state, step: int, metric_value: float,
                        metric_name: str = "accuracy", higher_is_better: bool = True):
    """Save checkpoint only if it's the best so far"""

    # Define best checkpoint path
    best_checkpoint_path = checkpoint_manager.directory / "best_checkpoint.json"

    # Load current best metric if exists
    current_best = None
    if best_checkpoint_path.exists():
        import json
        with open(best_checkpoint_path, 'r') as f:
            best_info = json.load(f)
            current_best = best_info.get(metric_name)

    # Check if this is the best checkpoint
    is_best = False
    if current_best is None:
        is_best = True
    elif higher_is_better and metric_value > current_best:
        is_best = True
    elif not higher_is_better and metric_value < current_best:
        is_best = True

    if is_best:
        # Save regular checkpoint
        save_checkpoint(checkpoint_manager, state, step)

        # Update best checkpoint info
        best_info = {
            metric_name: metric_value,
            "step": step,
            "timestamp": time.time()
        }

        import json
        with open(best_checkpoint_path, 'w') as f:
            json.dump(best_info, f, indent=2)

        print(f"New best checkpoint saved! {metric_name}: {metric_value}")

    return is_best

def save_checkpoint_with_metadata(checkpoint_manager, state, step: int,
                                metadata: Dict[str, Any]):
    """Save checkpoint with custom metadata"""

    # Prepare checkpoint data with metadata
    checkpoint_data = {
        'model': state,
        'step': step,
        'metadata': metadata
    }

    save_args = orbax.checkpoint.args.StandardSave(checkpoint_data)
    checkpoint_manager.save(step, checkpoint_data, save_kwargs={'save_args': save_args})

    print(f"Checkpoint with metadata saved at step {step}")

# ============================================================================
# 4. ASYNCHRONOUS CHECKPOINTING
# ============================================================================

class AsyncCheckpointManager:
    """Asynchronous checkpoint manager for non-blocking saves"""

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_manager = setup_checkpoint_manager(checkpoint_dir, max_to_keep)
        self.save_queue = asyncio.Queue()
        self.save_task = None

    async def start_save_worker(self):
        """Start background worker for async saves"""

        async def save_worker():
            while True:
                try:
                    save_data = await self.save_queue.get()
                    if save_data is None:  # Shutdown signal
                        break

                    state, step, metadata = save_data

                    # Perform actual save in background
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._sync_save, state, step, metadata
                    )

                    self.save_queue.task_done()

                except Exception as e:
                    print(f"Error in async save: {e}")

        self.save_task = asyncio.create_task(save_worker())

    def _sync_save(self, state, step, metadata):
        """Synchronous save operation"""
        if metadata:
            save_checkpoint_with_metadata(self.checkpoint_manager, state, step, metadata)
        else:
            save_checkpoint(self.checkpoint_manager, state, step)

    async def save_async(self, state, step: int, metadata: Optional[Dict] = None):
        """Queue checkpoint for asynchronous save"""
        await self.save_queue.put((state, step, metadata))
        print(f"Checkpoint queued for async save at step {step}")

    async def shutdown(self):
        """Shutdown async checkpoint manager"""
        await self.save_queue.put(None)  # Shutdown signal
        if self.save_task:
            await self.save_task

# ============================================================================
# 5. DISTRIBUTED TRAINING CHECKPOINTING
# ============================================================================

def setup_distributed_checkpointing(checkpoint_dir: str, process_id: int = 0,
                                   is_primary_process: bool = True):
    """Set up checkpointing for distributed training"""

    if is_primary_process:
        # Only primary process manages checkpoints
        checkpoint_manager = setup_checkpoint_manager(checkpoint_dir)
        print(f"Primary process {process_id} managing checkpoints")
    else:
        checkpoint_manager = None
        print(f"Secondary process {process_id} not managing checkpoints")

    return checkpoint_manager

def save_distributed_checkpoint(checkpoint_manager, state, step: int,
                               process_id: int = 0, is_primary_process: bool = True):
    """Save checkpoint in distributed training setup"""

    if is_primary_process and checkpoint_manager is not None:
        # Only primary process saves checkpoints
        save_checkpoint(checkpoint_manager, state, step)
        print(f"Distributed checkpoint saved by process {process_id}")
    else:
        print(f"Process {process_id} skipping checkpoint save")

def synchronize_checkpoint_restore(checkpoint_manager, step: Optional[int] = None):
    """Restore checkpoint synchronized across processes"""

    if checkpoint_manager is not None:
        # Restore checkpoint
        restored_state = restore_checkpoint(checkpoint_manager, step)

        # In real distributed training, you would broadcast the state
        # to all processes here using JAX's distributed communication
        print("Checkpoint restored and synchronized across processes")

        return restored_state
    else:
        print("Waiting for checkpoint broadcast from primary process")
        return None

# ============================================================================
# 6. CHECKPOINT MANAGEMENT UTILITIES
# ============================================================================

def cleanup_old_checkpoints(checkpoint_manager, keep_every_n_steps: int = 1000):
    """Clean up old checkpoints keeping only important ones"""

    all_steps = checkpoint_manager.all_steps()

    steps_to_remove = []
    for step in all_steps:
        # Keep checkpoints at regular intervals
        if step % keep_every_n_steps != 0:
            steps_to_remove.append(step)

    # Remove old checkpoints
    for step in steps_to_remove[:-5]:  # Keep last 5 regardless
        try:
            checkpoint_manager.delete(step)
            print(f"Removed checkpoint at step {step}")
        except Exception as e:
            print(f"Error removing checkpoint at step {step}: {e}")

def backup_checkpoint(checkpoint_manager, step: int, backup_dir: str):
    """Create backup of specific checkpoint"""

    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    # Restore checkpoint
    state = restore_checkpoint(checkpoint_manager, step)

    if state is not None:
        # Create backup checkpoint manager
        backup_manager = setup_checkpoint_manager(backup_dir, max_to_keep=100)

        # Save to backup location
        save_checkpoint(backup_manager, state['model'], step)

        print(f"Checkpoint at step {step} backed up to {backup_dir}")

def validate_checkpoint_integrity(checkpoint_manager, step: int):
    """Validate checkpoint integrity"""

    try:
        # Try to restore checkpoint
        state = restore_checkpoint(checkpoint_manager, step)

        if state is None:
            print(f"Checkpoint at step {step} is corrupted or missing")
            return False

        # Basic validation
        if 'model' not in state:
            print(f"Checkpoint at step {step} missing model state")
            return False

        print(f"Checkpoint at step {step} is valid")
        return True

    except Exception as e:
        print(f"Error validating checkpoint at step {step}: {e}")
        return False

# ============================================================================
# 7. INTEGRATION WITH TRAINING LOOPS
# ============================================================================

def training_loop_with_checkpointing(model, initial_state, train_data,
                                   checkpoint_dir: str, save_every: int = 100):
    """Example training loop with integrated checkpointing"""

    # Setup checkpoint manager
    checkpoint_manager = setup_checkpoint_manager(checkpoint_dir)

    # Try to restore from existing checkpoint
    restored_state = restore_checkpoint(checkpoint_manager)
    if restored_state is not None:
        state = restored_state['model']
        start_step = restored_state['step']
        print(f"Resumed training from step {start_step}")
    else:
        state = initial_state
        start_step = 0
        print("Starting training from scratch")

    # Training loop
    for step in range(start_step, 1000):
        # Simulate training step
        batch = next(train_data)  # Get next batch

        # Compute loss and gradients (simplified)
        def loss_fn(params):
            predictions = model.apply(params, batch['x'])
            return jnp.mean((predictions - batch['y']) ** 2)

        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Save checkpoint periodically
        if step % save_every == 0:
            metrics = {'loss': float(loss_value), 'step': step}
            save_checkpoint(checkpoint_manager, state, step, metrics)

        # Print progress
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss_value:.4f}")

    print("Training completed")
    return state

async def async_training_loop(model, initial_state, train_data, checkpoint_dir: str):
    """Training loop with asynchronous checkpointing"""

    # Setup async checkpoint manager
    async_manager = AsyncCheckpointManager(checkpoint_dir)
    await async_manager.start_save_worker()

    state = initial_state

    try:
        for step in range(1000):
            # Simulate training step
            batch = next(train_data)

            # Compute loss and update state (simplified)
            def loss_fn(params):
                predictions = model.apply(params, batch['x'])
                return jnp.mean((predictions - batch['y']) ** 2)

            loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)

            # Async checkpoint save
            if step % 100 == 0:
                metadata = {'loss': float(loss_value), 'timestamp': time.time()}
                await async_manager.save_async(state, step, metadata)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss_value:.4f}")

    finally:
        # Shutdown async manager
        await async_manager.shutdown()

    return state

# ============================================================================
# 8. COMPREHENSIVE EXAMPLES
# ============================================================================

def run_checkpoint_examples():
    """Run comprehensive checkpointing examples"""

    print("=== JAX Orbax Checkpoint Examples ===")

    # Create sample model and state
    model, initial_state = create_sample_model_and_state()

    # Example 1: Basic checkpointing
    print("\n1. Basic checkpoint save and restore:")
    checkpoint_dir = "/tmp/jax_checkpoints"
    checkpoint_manager = setup_checkpoint_manager(checkpoint_dir)

    # Save checkpoint
    save_checkpoint(checkpoint_manager, initial_state, step=100)

    # List checkpoints
    list_available_checkpoints(checkpoint_manager)

    # Restore checkpoint
    restored_state = restore_checkpoint(checkpoint_manager, step=100)

    # Example 2: Best checkpoint saving
    print("\n2. Best checkpoint management:")
    # Simulate saving best checkpoints
    for step, accuracy in [(100, 0.85), (200, 0.87), (300, 0.86), (400, 0.90)]:
        is_best = save_best_checkpoint(
            checkpoint_manager, initial_state, step, accuracy, "accuracy"
        )
        print(f"  Step {step}: accuracy={accuracy:.3f}, is_best={is_best}")

    # Example 3: Checkpoint validation
    print("\n3. Checkpoint validation:")
    all_steps = checkpoint_manager.all_steps()
    for step in all_steps:
        is_valid = validate_checkpoint_integrity(checkpoint_manager, step)
        print(f"  Step {step}: valid={is_valid}")

    # Example 4: Cleanup demonstration
    print("\n4. Checkpoint cleanup:")
    print(f"Before cleanup: {len(checkpoint_manager.all_steps())} checkpoints")
    cleanup_old_checkpoints(checkpoint_manager, keep_every_n_steps=200)
    print(f"After cleanup: {len(checkpoint_manager.all_steps())} checkpoints")

# Async example
async def run_async_checkpoint_example():
    """Run asynchronous checkpointing example"""

    print("\n5. Async checkpointing example:")

    # Create sample model and data
    model, initial_state = create_sample_model_and_state()

    # Simulate training data
    def generate_batch():
        while True:
            key_x, key_y = random.split(random.PRNGKey(42))
            x = random.normal(key_x, (32, 20))
            y = random.normal(key_y, (32, 10))
            yield {'x': x, 'y': y}

    train_data = generate_batch()

    # Run async training loop
    final_state = await async_training_loop(
        model, initial_state, train_data, "/tmp/async_checkpoints"
    )

    print("Async training completed")

# Run examples
run_checkpoint_examples()

# Run async example (in a real script, you'd use asyncio.run())
# asyncio.run(run_async_checkpoint_example())
```

## Checkpoint Management Best Practices

### Checkpoint Frequency
- **Save regularly**: Every 100-1000 steps for long training runs
- **Save best checkpoints**: Based on validation metrics
- **Save at milestones**: End of epochs, before evaluation
- **Emergency saves**: Before shutdown or interruption

### Storage Management
- **Cleanup old checkpoints**: Keep only recent and best checkpoints
- **Backup important checkpoints**: To separate storage locations
- **Monitor disk usage**: Large models can consume significant space
- **Use compression**: For long-term storage

### Distributed Training
- **Primary process saves**: Avoid multiple processes writing simultaneously
- **Synchronize restoration**: Ensure all processes use same checkpoint
- **Handle failures**: Graceful degradation when checkpoints are corrupted
- **Network considerations**: Minimize checkpoint transfer overhead

### Performance Optimization
- **Async saves**: Non-blocking checkpoint operations
- **Batched metadata**: Group multiple operations
- **Efficient serialization**: Use optimized formats
- **Memory management**: Avoid memory spikes during saves

## Common Issues and Solutions

### Storage Problems
- **Disk space**: Monitor and cleanup old checkpoints regularly
- **Permission errors**: Ensure write access to checkpoint directory
- **Network storage**: Handle temporary network failures gracefully
- **Concurrent access**: Use file locking for multi-process safety

### Restoration Issues
- **Version compatibility**: Handle model architecture changes
- **Missing checkpoints**: Implement fallback strategies
- **Corrupted files**: Validate checkpoint integrity
- **State mismatches**: Handle optimizer state changes

### Performance Issues
- **Slow saves**: Use async operations and faster storage
- **Memory spikes**: Optimize checkpoint data structure
- **Network overhead**: Compress data for distributed saves
- **Blocking operations**: Separate checkpoint thread from training

## Agent-Enhanced Checkpoint Integration Patterns

### Complete Production ML Checkpoint Workflow
```bash
# Intelligent checkpoint management and MLOps pipeline
/jax-orbax-checkpoint --agents=auto --intelligent --mlops --optimize
/jax-training --agents=ai --checkpoint-integration --distributed
/ci-setup --agents=ai --mlops --model-deployment
```

### Scientific Computing Checkpoint Pipeline
```bash
# High-performance scientific computing checkpoint management
/jax-orbax-checkpoint --agents=scientific --breakthrough --orchestrate
/jax-performance --agents=scientific --technique=memory --optimization
/run-all-tests --agents=scientific --reproducible --checkpoint-validation
```

### Research Reproducibility Checkpoint Infrastructure
```bash
# Research-grade checkpoint management and versioning
/jax-orbax-checkpoint --agents=all --breakthrough --orchestrate --mlops
/jax-models --agents=research --versioning --reproducibility
/update-docs --agents=research --type=reproducibility
```

## Related Commands

**Prerequisites**: Commands to run before checkpoint setup
- `/jax-init --agents=auto` - JAX project setup with checkpoint configuration
- `/jax-models --agents=auto` - Model definition and parameter management

**Core Workflow**: Checkpoint development with agent intelligence
- `/jax-training --agents=ai` - Training workflows with intelligent checkpoint integration
- `/jax-performance --agents=jax` - Checkpoint performance optimization
- `/jax-debug --agents=auto` - Debug checkpoint operations and state management

**Advanced Integration**: Specialized checkpoint development
- `/jax-data-load --agents=ai` - Data pipeline checkpoint coordination
- `/jax-sparse-ops --agents=scientific` - Sparse model checkpoint optimization
- `/jax-numpyro-prob --agents=scientific` - Probabilistic model checkpoint management

**Quality Assurance**: Checkpoint validation and testing
- `/generate-tests --agents=auto --type=checkpoint` - Generate checkpoint system tests
- `/run-all-tests --agents=ai --reproducible` - Comprehensive checkpoint testing
- `/check-code-quality --agents=production --mlops` - Production checkpoint code quality

**MLOps & Production**: Advanced checkpoint workflows
- `/ci-setup --agents=ai --mlops` - Automated checkpoint CI/CD integration
- `/update-docs --agents=production --type=mlops` - Production checkpoint documentation
- `/multi-agent-optimize --agents=all --focus=checkpoint` - Comprehensive checkpoint optimization

ARGUMENTS: [--save] [--restore] [--async] [--distributed] [--agents=auto|jax|scientific|ai|production|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--mlops]