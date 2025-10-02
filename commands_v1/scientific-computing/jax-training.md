---
description: Complete ML training workflows with Optax optimizers, learning rate schedules, and JIT-compiled training loops
category: jax-ml
argument-hint: "[--optimizer=adam|sgd|rmsprop] [--schedule=constant|cosine|exponential] [--epochs=100] [--agents=auto|jax|scientific|ai|research|all] [--orchestrate] [--intelligent] [--breakthrough] [--distributed] [--optimize]"
allowed-tools: "*"
model: inherit
---

# JAX Training

Complete ML training workflows with Optax optimizers, learning rate schedules, and JIT-compiled training loops.

```bash
/jax-training [--optimizer=adam|sgd|rmsprop] [--schedule=constant|cosine|exponential] [--epochs=100] [--agents=auto|jax|scientific|ai|research|all] [--orchestrate] [--intelligent] [--breakthrough] [--distributed] [--optimize]
```

## Options

- `--optimizer=<optimizer>`: Optimizer type (adam, sgd, rmsprop)
- `--schedule=<schedule>`: Learning rate schedule (constant, cosine, exponential)
- `--epochs=<epochs>`: Number of training epochs (default: 100)
- `--agents=<agents>`: Agent selection (auto, jax, scientific, ai, research, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with training intelligence
- `--intelligent`: Enable intelligent agent selection based on training analysis
- `--breakthrough`: Enable breakthrough training optimization discovery
- `--distributed`: Enable distributed training across multiple agents
- `--optimize`: Apply performance optimization to training workflows

## What it does

1. **Optax Optimizers**: Set up Adam, SGD, RMSprop with proper state management
2. **Learning Rate Schedules**: Implement constant, cosine, exponential schedules
3. **Training Loops**: JIT-compiled training steps with gradient computation
4. **State Management**: Handle optimizer state and model parameters
5. **Progress Tracking**: Monitor loss, metrics, and training progress
6. **23-Agent Training Intelligence**: Multi-agent collaboration for optimal training strategies
7. **Distributed Training**: Agent-coordinated distributed training workflows
8. **Performance Optimization**: Agent-driven training loop and optimizer optimization

## 23-Agent Intelligent Training System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes training requirements, model complexity, and performance constraints to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Training Type Detection → Agent Selection
- Deep Learning Research → neural-networks-master + research-intelligence-master + jax-pro
- Scientific ML Training → scientific-computing-master + jax-pro + neural-networks-master
- Production Training → ai-systems-architect + jax-pro + systems-architect
- Large-Scale Training → ai-systems-architect + neural-networks-master + multi-agent-orchestrator
- Experimental Training → research-intelligence-master + neural-networks-master + jax-pro
```

### Core JAX Training Optimization Agents

#### **`neural-networks-master`** - Deep Learning Training Expert
- **Training Strategy**: Advanced training methodologies and optimization strategies
- **Architecture-Specific Training**: Optimal training patterns for different neural architectures
- **Learning Dynamics**: Training stability, convergence analysis, and optimization tuning
- **Advanced Techniques**: Gradient accumulation, mixed precision, and distributed training
- **Research Integration**: Cutting-edge training methodologies and best practices

#### **`jax-pro`** - JAX Training Optimization Specialist
- **JAX Ecosystem Mastery**: Deep expertise in JAX, Optax, and training loop optimization
- **Performance Engineering**: JIT compilation optimization and memory-efficient training
- **GPU/TPU Optimization**: Device-specific training optimization and parallel computation
- **Scientific ML**: JAX training patterns for scientific computing applications
- **Advanced Transformations**: Optimal use of grad, vmap, pmap for training efficiency

#### **`ai-systems-architect`** - Training Infrastructure & Scalability
- **Distributed Training**: Multi-device and multi-node training architecture
- **Scalability Engineering**: Training system design for large-scale applications
- **Resource Optimization**: Memory management and computational resource allocation
- **Production Training**: Training infrastructure for deployment and monitoring
- **MLOps Integration**: Training pipeline design for continuous integration

#### **`research-intelligence-master`** - Research-Grade Training Methodologies
- **Experimental Design**: Research methodology for training experiments and validation
- **Innovation Synthesis**: Advanced training techniques from cutting-edge research
- **Reproducibility**: Research-grade training reproducibility and experimental rigor
- **Academic Standards**: Training methodologies for academic and research publication
- **Breakthrough Discovery**: Novel training approaches and optimization innovations

### Specialized Training Agents

#### **`scientific-computing-master`** - Scientific ML Training
- **Physics-Informed Training**: Training methodologies for scientific ML applications
- **Numerical Optimization**: Training strategies for computational science applications
- **Domain Integration**: Training approaches for specific scientific domains
- **Multi-Scale Training**: Training strategies for multi-scale scientific problems
- **Computational Efficiency**: Scientific computing performance optimization

#### **`systems-architect`** - Training System Design & Integration
- **Training Infrastructure**: System-level training architecture and resource management
- **Integration Patterns**: Training system integration with larger application architectures
- **Performance Monitoring**: System-level training performance tracking and optimization
- **Fault Tolerance**: Robust training systems with failure recovery and resilience
- **Deployment Integration**: Training system design for production deployment

#### **`data-professional`** - Data-Centric Training Optimization
- **Data Pipeline Integration**: Training optimization for data loading and preprocessing
- **Streaming Training**: Training strategies for streaming and real-time data
- **Data Efficiency**: Training approaches for limited or expensive data scenarios
- **Feature Engineering**: Training integration with feature processing and selection
- **Data Quality**: Training robustness and data quality management

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Training
Automatically analyzes training requirements and selects optimal agent combinations:
- **Training Analysis**: Detects model complexity, data characteristics, performance requirements
- **Resource Assessment**: Evaluates computational resources and training constraints
- **Agent Matching**: Maps training patterns to relevant agent expertise
- **Efficiency Optimization**: Balances comprehensive training with computational efficiency

#### **`jax`** - JAX-Specialized Training Team
- `jax-pro` (JAX ecosystem lead)
- `neural-networks-master` (training methodology)
- `ai-systems-architect` (distributed training)
- `scientific-computing-master` (scientific applications)

#### **`scientific`** - Scientific Computing Training Team
- `scientific-computing-master` (lead)
- `jax-pro` (JAX implementation)
- `neural-networks-master` (ML methodology)
- `research-intelligence-master` (research methodology)
- Domain-specific experts based on scientific application

#### **`ai`** - AI/ML Production Training Team
- `ai-systems-architect` (lead)
- `neural-networks-master` (training expertise)
- `jax-pro` (JAX optimization)
- `data-professional` (data integration)
- `systems-architect` (infrastructure)

#### **`research`** - Research-Grade Training Development
- `research-intelligence-master` (lead)
- `neural-networks-master` (training innovation)
- `jax-pro` (JAX research tools)
- `scientific-computing-master` (computational research)
- `ai-systems-architect` (scalable research infrastructure)

#### **`all`** - Complete 23-Agent Training Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough training optimization.

### 23-Agent Training Orchestration (`--orchestrate`)

#### **Multi-Agent Training Pipeline**
1. **Training Strategy Phase**: Multiple agents analyze training requirements simultaneously
2. **Optimization Design**: Collaborative optimization strategy development across agent expertise
3. **Distributed Execution**: Agent-coordinated training across resources and methodologies
4. **Performance Monitoring**: Multi-agent monitoring and optimization during training
5. **Adaptive Optimization**: Real-time training strategy adjustment based on performance

#### **Breakthrough Training Discovery (`--breakthrough`)**
- **Cross-Domain Innovation**: Training techniques from multiple domains and research areas
- **Emergent Optimization**: Novel training strategies discovered through agent collaboration
- **Research-Grade Performance**: Academic and industry-leading training standards
- **Adaptive Methodologies**: Dynamic training strategy adaptation based on performance

### Advanced 23-Agent Training Examples

```bash
# Intelligent auto-selection for training optimization
/jax-training --agents=auto --intelligent --optimizer=adam --epochs=1000

# Scientific computing training with specialized agents
/jax-training --agents=scientific --schedule=cosine --optimize --orchestrate

# AI/ML production training with scalability focus
/jax-training --agents=ai --distributed --optimize --breakthrough

# Research-grade training development
/jax-training --agents=research --breakthrough --orchestrate --distributed

# JAX-specialized training optimization
/jax-training --agents=jax --optimizer=adamw --schedule=warmup_cosine --optimize

# Complete 23-agent training ecosystem
/jax-training --agents=all --orchestrate --breakthrough --distributed

# Large-scale transformer training
/jax-training transformer_model.py --agents=ai --distributed --optimize --epochs=500

# Physics-informed neural network training
/jax-training physics_model.py --agents=scientific --intelligent --breakthrough

# Experimental architecture training
/jax-training experimental.py --agents=research --orchestrate --breakthrough

# Production CNN training optimization
/jax-training production_cnn.py --agents=ai --optimize --distributed

# Multi-modal model training
/jax-training multimodal.py --agents=all --orchestrate --intelligent

# Scientific simulation training
/jax-training simulation.py --agents=scientific --optimize --breakthrough
```

### Intelligent Agent Selection Examples

```bash
# Training Type Detection → Intelligent Agent Selection

# Large-scale deep learning project
/jax-training large_model.py --agents=auto --intelligent
# → Selects: ai-systems-architect + neural-networks-master + jax-pro

# Scientific ML research training
/jax-training physics_research.py --agents=auto --intelligent
# → Selects: scientific-computing-master + research-intelligence-master + jax-pro

# Production model training
/jax-training production_model.py --agents=auto --intelligent
# → Selects: ai-systems-architect + jax-pro + systems-architect

# Computer vision training
/jax-training vision_model.py --agents=auto --intelligent
# → Selects: neural-networks-master + ai-systems-architect + jax-pro

# Experimental training methodology
/jax-training experimental_training.py --agents=auto --intelligent
# → Selects: research-intelligence-master + neural-networks-master + multi-agent-orchestrator

# Distributed training optimization
/jax-training distributed_model.py --agents=all --orchestrate --distributed
# → Activates: All 23 agents with intelligent coordination for maximum training efficiency
```

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap, value_and_grad
import optax
import time

# Initialize PRNG key
key = random.PRNGKey(0)

# ============================================================================
# 1. BASIC TRAINING SETUP
# ============================================================================

def create_model_and_data(key, n_features=10, n_samples=1000):
    """Create simple model and synthetic data"""

    # Model parameters
    key, subkey = random.split(key)
    params = {
        'weights': random.normal(subkey, (n_features, 1)) * 0.1,
        'bias': jnp.zeros((1,))
    }

    # Synthetic data
    key, subkey = random.split(key)
    X = random.normal(subkey, (n_samples, n_features))
    true_weights = random.normal(key, (n_features, 1))
    y = jnp.dot(X, true_weights) + 0.1 * random.normal(key, (n_samples, 1))

    return params, X, y

def predict(params, X):
    """Simple linear model prediction"""
    return jnp.dot(X, params['weights']) + params['bias']

def loss_fn(params, X, y):
    """Mean squared error loss"""
    predictions = predict(params, X)
    return jnp.mean((predictions - y) ** 2)

# ============================================================================
# 2. OPTAX OPTIMIZERS
# ============================================================================

def setup_optimizer(optimizer_name, learning_rate=0.01):
    """Set up Optax optimizer"""

    if optimizer_name == 'adam':
        optimizer = optax.adam(learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optax.sgd(learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optax.rmsprop(learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = optax.adamw(learning_rate, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer

def setup_advanced_optimizer(learning_rate=0.01):
    """Advanced optimizer with gradient clipping and transformations"""

    # Chain multiple transformations
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.scale_by_adam(),           # Adam scaling
        optax.scale(-learning_rate)      # Learning rate scaling
    )

    return optimizer

# ============================================================================
# 3. LEARNING RATE SCHEDULES
# ============================================================================

def create_lr_schedule(schedule_name, learning_rate=0.01, total_steps=1000):
    """Create learning rate schedule"""

    if schedule_name == 'constant':
        schedule = optax.constant_schedule(learning_rate)

    elif schedule_name == 'exponential':
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=100,
            decay_rate=0.95
        )

    elif schedule_name == 'cosine':
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps
        )

    elif schedule_name == 'polynomial':
        schedule = optax.polynomial_schedule(
            init_value=learning_rate,
            end_value=learning_rate * 0.01,
            power=1.0,
            transition_steps=total_steps
        )

    elif schedule_name == 'warmup_cosine':
        warmup_steps = total_steps // 10
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps
        )

    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    return schedule

# ============================================================================
# 4. TRAINING STEP FUNCTIONS
# ============================================================================

@jit
def train_step(params, opt_state, X, y, optimizer):
    """Single training step with gradient computation"""

    # Compute loss and gradients
    loss_value, grads = value_and_grad(loss_fn)(params, X, y)

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_value

@jit
def train_step_with_metrics(params, opt_state, X, y, optimizer):
    """Training step with additional metrics"""

    # Compute loss and gradients
    loss_value, grads = value_and_grad(loss_fn)(params, X, y)

    # Compute additional metrics
    predictions = predict(params, X)
    mae = jnp.mean(jnp.abs(predictions - y))

    # Gradient norm for monitoring
    grad_norm = optax.global_norm(grads)

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    metrics = {
        'loss': loss_value,
        'mae': mae,
        'grad_norm': grad_norm
    }

    return new_params, new_opt_state, metrics

# ============================================================================
# 5. COMPLETE TRAINING LOOPS
# ============================================================================

def train_model(params, X, y, optimizer, n_epochs=100, batch_size=32):
    """Complete training loop"""

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Training history
    history = {'loss': [], 'mae': [], 'grad_norm': []}

    n_samples = X.shape[0]
    n_batches = n_samples // batch_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_grad_norm = 0.0

        # Shuffle data each epoch
        key = random.PRNGKey(epoch)
        perm = random.permutation(key, n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        # Mini-batch training
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Training step
            params, opt_state, metrics = train_step_with_metrics(
                params, opt_state, X_batch, y_batch, optimizer
            )

            epoch_loss += metrics['loss']
            epoch_mae += metrics['mae']
            epoch_grad_norm += metrics['grad_norm']

        # Average metrics over batches
        avg_loss = epoch_loss / n_batches
        avg_mae = epoch_mae / n_batches
        avg_grad_norm = epoch_grad_norm / n_batches

        history['loss'].append(float(avg_loss))
        history['mae'].append(float(avg_mae))
        history['grad_norm'].append(float(avg_grad_norm))

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.6f}, mae={avg_mae:.6f}, grad_norm={avg_grad_norm:.6f}")

    return params, history

def train_with_validation(params, X_train, y_train, X_val, y_val,
                         optimizer, n_epochs=100, batch_size=32):
    """Training with validation monitoring"""

    opt_state = optimizer.init(params)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': []
    }

    best_val_loss = float('inf')
    best_params = params
    patience_counter = 0
    patience = 10

    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    for epoch in range(n_epochs):
        # Training phase
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0

        key = random.PRNGKey(epoch)
        perm = random.permutation(key, n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            params, opt_state, metrics = train_step_with_metrics(
                params, opt_state, X_batch, y_batch, optimizer
            )

            epoch_train_loss += metrics['loss']
            epoch_train_mae += metrics['mae']

        avg_train_loss = epoch_train_loss / n_batches
        avg_train_mae = epoch_train_mae / n_batches

        # Validation phase
        val_loss = loss_fn(params, X_val, y_val)
        val_predictions = predict(params, X_val)
        val_mae = jnp.mean(jnp.abs(val_predictions - y_val))

        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_mae'].append(float(avg_train_mae))
        history['val_mae'].append(float(val_mae))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

    return best_params, history

# ============================================================================
# 6. ADVANCED TRAINING FEATURES
# ============================================================================

def train_with_lr_schedule(params, X, y, base_lr=0.01, n_epochs=100):
    """Training with learning rate scheduling"""

    total_steps = n_epochs * (X.shape[0] // 32)  # Approximate steps

    # Create schedule and optimizer
    lr_schedule = create_lr_schedule('warmup_cosine', base_lr, total_steps)
    optimizer = optax.adam(lr_schedule)

    opt_state = optimizer.init(params)
    history = {'loss': [], 'learning_rate': []}

    step = 0
    for epoch in range(n_epochs):
        # Get current learning rate
        current_lr = lr_schedule(step)

        # Training step (simplified for demonstration)
        params, opt_state, loss_value = train_step(
            params, opt_state, X, y, optimizer
        )

        history['loss'].append(float(loss_value))
        history['learning_rate'].append(float(current_lr))

        step += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss_value:.6f}, lr={current_lr:.6f}")

    return params, history

def train_with_gradient_accumulation(params, X, y, optimizer,
                                   accumulation_steps=4, n_epochs=100):
    """Training with gradient accumulation for large effective batch sizes"""

    opt_state = optimizer.init(params)
    batch_size = 32
    n_samples = X.shape[0]

    for epoch in range(n_epochs):
        # Shuffle data
        key = random.PRNGKey(epoch)
        perm = random.permutation(key, n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        accumulated_grads = jax.tree_map(jnp.zeros_like, params)
        accumulated_loss = 0.0

        # Accumulate gradients over multiple mini-batches
        for step in range(accumulation_steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size

            if end_idx > n_samples:
                break

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Compute gradients without updating
            loss_value, grads = value_and_grad(loss_fn)(params, X_batch, y_batch)

            # Accumulate gradients
            accumulated_grads = jax.tree_map(
                lambda acc, g: acc + g / accumulation_steps,
                accumulated_grads, grads
            )
            accumulated_loss += loss_value / accumulation_steps

        # Apply accumulated gradients
        updates, opt_state = optimizer.update(accumulated_grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: accumulated_loss={accumulated_loss:.6f}")

    return params

# ============================================================================
# 7. EXAMPLE USAGE
# ============================================================================

def run_training_examples():
    """Run various training examples"""

    print("=== JAX Training Examples ===")

    # Create model and data
    key = random.PRNGKey(42)
    params, X, y = create_model_and_data(key)

    # Split into train/validation
    n_train = int(0.8 * X.shape[0])
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

    # Example 1: Basic training with Adam
    print("\n1. Basic training with Adam:")
    optimizer = setup_optimizer('adam', learning_rate=0.01)
    trained_params, history = train_model(params, X_train, y_train, optimizer, n_epochs=50)
    print(f"Final loss: {history['loss'][-1]:.6f}")

    # Example 2: Training with validation
    print("\n2. Training with validation:")
    optimizer = setup_optimizer('adam', learning_rate=0.01)
    best_params, val_history = train_with_validation(
        params, X_train, y_train, X_val, y_val, optimizer, n_epochs=100
    )
    print(f"Best validation loss: {min(val_history['val_loss']):.6f}")

    # Example 3: Training with learning rate schedule
    print("\n3. Training with learning rate schedule:")
    scheduled_params, lr_history = train_with_lr_schedule(
        params, X_train, y_train, base_lr=0.01, n_epochs=50
    )
    print(f"Final loss: {lr_history['loss'][-1]:.6f}")

    print("\nTraining examples completed!")

# Run training examples
run_training_examples()
```

## Training Best Practices

### Optimizer Selection
- **Adam**: Good default choice, works well for most problems
- **SGD**: Simple and reliable, good for well-tuned problems
- **RMSprop**: Good for RNNs and non-stationary problems
- **AdamW**: Adam with weight decay for better generalization

### Learning Rate Scheduling
- Start with constant learning rate for baseline
- Use cosine decay for longer training runs
- Add warmup for large batch sizes or transformers
- Monitor learning rate vs. loss curves

### Training Loop Design
- Always use JIT compilation for training steps
- Implement proper gradient accumulation for large batches
- Include validation monitoring and early stopping
- Track multiple metrics, not just loss

### Memory and Performance
- Use appropriate batch sizes for your hardware
- Consider gradient checkpointing for large models
- Monitor gradient norms to detect training issues
- Profile training loops to identify bottlenecks

## Common Training Issues

### Optimization Problems
- **Vanishing gradients**: Use gradient clipping, better initialization
- **Exploding gradients**: Apply gradient clipping with optax.clip_by_global_norm
- **Slow convergence**: Increase learning rate, use learning rate schedules
- **Unstable training**: Reduce learning rate, add weight decay

### Implementation Issues
- **Memory errors**: Reduce batch size, use gradient accumulation
- **Slow training**: Ensure JIT compilation, check device placement
- **Poor generalization**: Add regularization, use validation monitoring
- **NaN losses**: Check learning rates, add gradient clipping

## Agent-Enhanced Training Integration Patterns

### Complete Training Workflow with Agents
```bash
# Intelligent model definition and training pipeline
/jax-models --agents=auto --intelligent --framework=flax
/jax-training --agents=auto --intelligent --optimizer=adam --epochs=1000
/jax-performance --agents=jax --technique=caching --gpu-accel
```

### Scientific Computing Training Pipeline
```bash
# Physics-informed neural network training workflow
/jax-models --agents=scientific --breakthrough --architecture=mlp
/jax-training --agents=scientific --optimize --orchestrate --schedule=cosine
/jax-essentials --agents=scientific --operation=grad --higher-order
```

### Production Training Infrastructure
```bash
# Large-scale production training with monitoring
/jax-training --agents=ai --distributed --optimize --breakthrough
/jax-performance --agents=ai --optimization --gpu-accel
/run-all-tests --agents=ai --benchmark --coverage
```

## Related Commands

**Prerequisites**: Commands to run before training setup
- `/jax-models --agents=auto` - Neural network model definitions with agent optimization
- `/jax-essentials --agents=auto` - Core JAX operations with intelligent assistance
- `/jax-init --agents=auto` - JAX project setup with agent intelligence

**Core Workflow**: Training development with agent intelligence
- `/jax-performance --agents=jax` - Training performance optimization with specialized agents
- `/jax-debug --agents=auto` - Debug training issues with intelligent assistance
- `/jax-data-load --agents=auto` - Optimized data loading for training workflows

**Advanced Integration**: Specialized training development
- `/jax-numpyro-prob --agents=scientific` - Probabilistic training with scientific agents
- `/jax-orbax-checkpoint --agents=ai` - Training checkpointing with production agents
- `/jax-sparse-ops --agents=scientific` - Sparse training optimization with scientific agents

**Quality Assurance**: Validation and optimization
- `/generate-tests --agents=auto --type=jax` - Generate training tests with agent intelligence
- `/run-all-tests --agents=ai --benchmark` - Comprehensive training validation
- `/optimize --agents=jax --language=jax` - Training code optimization with JAX agents

**Research & Documentation**: Advanced workflows
- `/update-docs --agents=research --type=api` - Research-grade training documentation
- `/reflection --agents=research --type=scientific` - Training methodology analysis
- `/multi-agent-optimize --agents=all --mode=optimize` - Comprehensive training optimization