---
title: "JAX Essentials"
description: "Core JAX operations: JIT, gradients, vectorization (vmap), and parallelization (pmap)"
category: scientific-computing
subcategory: jax-core
complexity: intermediate
argument-hint: "[--operation=jit|grad|vmap|pmap] [--higher-order] [--static-args] [--devices] [--agents=auto|jax|scientific|ai|research|all] [--orchestrate] [--intelligent] [--breakthrough]"
allowed-tools: "*"
model: inherit
tags: jax, jit, gradients, vmap, pmap, scientific-computing
dependencies: [jax-init]
related: [jax-performance, jax-debug, jax-models, jax-training, optimize]
workflows: [jax-development, performance-optimization, scientific-computing]
version: "2.0"
last-updated: "2025-09-28"
---

# JAX Essentials

Core JAX operations: JIT compilation, automatic differentiation, vectorization, and parallelization.

## Quick Start

```bash
# Learn core JAX operations
/jax-essentials --operation=jit

# Advanced transformations
/jax-essentials --operation=grad --higher-order

# Multi-device operations
/jax-essentials --operation=pmap --devices

# Complete JAX mastery
/jax-essentials --operation=all --agents=jax --intelligent
```

## Usage

```bash
/jax-essentials [options]
```

**Parameters:**
- `options` - JAX operation focus, complexity level, and agent configuration

## Options

- `--operation=<op>`: Focus on specific operation (jit, grad, vmap, pmap)
- `--higher-order`: Include higher-order derivatives and compositions
- `--static-args`: Show static argument patterns and optimizations
- `--devices`: Include multi-device and hardware-specific examples
- `--agents=<agents>`: Agent selection (auto, jax, scientific, ai, research, all)
- `--orchestrate`: Enable 23-agent orchestration for complex JAX workflows
- `--intelligent`: Enable intelligent agent selection based on code analysis
- `--breakthrough`: Enable breakthrough optimization discovery across domains

## 23-Agent Scientific Computing Integration

### Intelligent JAX Agent Selection (`--intelligent`)
**Auto-JAX Algorithm**: Analyzes JAX code patterns, transformations, and performance requirements to automatically deploy optimal agent combinations from the 23-agent library for maximum JAX efficiency.

```bash
# JAX Pattern Detection → Agent Selection
- Core JAX operations → jax-pro + scientific-computing-master
- Neural network models → jax-pro + neural-networks-master + ai-systems-architect
- Scientific simulations → jax-pro + scientific-computing-master + research-intelligence-master
- Quantum computing → advanced-quantum-computing-expert + jax-pro + scientific-computing-master
- High-performance computing → jax-pro + systems-architect + devops-security-engineer
```

### Core JAX Agents

#### **`jax-pro`** - JAX Ecosystem Specialist
- **JAX Mastery**: Deep expertise in JAX transformations, XLA optimization, and GPU acceleration
- **Performance Optimization**: Advanced JIT compilation, memory management, and hardware utilization
- **Scientific ML**: JAX-based machine learning, Flax/Equinox models, and scientific computing
- **Advanced Transformations**: Complex vmap/pmap patterns, custom derivatives, and parallel computing
- **Ecosystem Integration**: NumPyro, Optax, Orbax, and JAX scientific computing libraries

#### **`scientific-computing-master`** - Scientific Computing Authority
- **Numerical Methods**: Advanced numerical algorithms, scientific simulations, and computational science
- **Multi-Language Integration**: Python, Julia, JAX ecosystem coordination and optimization
- **Research Workflows**: Publication-ready computational methods and reproducible science
- **Performance Analysis**: Scientific computing optimization and computational efficiency
- **Domain Integration**: Physics, chemistry, biology, and engineering computational methods

#### **`neural-networks-master`** - Deep Learning Architecture Expert
- **JAX Deep Learning**: Flax, Equinox, and JAX-based neural network architectures
- **Model Optimization**: Architecture design, training efficiency, and model performance
- **Advanced Architectures**: Transformers, CNNs, RNNs, and custom neural network designs
- **Training Workflows**: Optimization algorithms, learning rate schedules, and training pipelines
- **Research Integration**: Cutting-edge deep learning research and implementation

### Advanced Scientific Computing Agents

#### **Research & Innovation**
- **`research-intelligence-master`**: Research methodology, innovation discovery, and scientific breakthroughs
- **`advanced-quantum-computing-expert`**: Quantum computing algorithms and quantum-classical hybrid systems
- **`ai-systems-architect`**: AI system design, scalability, and intelligent architecture

#### **Engineering & Architecture**
- **`systems-architect`**: High-performance computing architecture and system design
- **`code-quality-master`**: Scientific code quality, testing strategies, and validation
- **`devops-security-engineer`**: Scientific computing infrastructure and deployment

#### **Domain-Specific Scientific Experts**
- **`data-professional`**: Scientific data pipelines, analytics, and data processing optimization
- **`correlation-function-expert`**: Statistical analysis, correlation methods, and statistical computing
- **`nonequilibrium-stochastic-expert`**: Stochastic processes, nonequilibrium systems, and probabilistic methods

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for JAX
Automatically analyzes JAX code and selects optimal agent combinations:
- **Pattern Recognition**: Detects JAX transformations, ML patterns, and scientific computing requirements
- **Performance Optimization**: Matches computational needs to agent expertise
- **Resource Management**: Balances comprehensive analysis with execution efficiency
- **Dynamic Scaling**: Adjusts agent allocation based on problem complexity

#### **`jax`** - JAX-Focused Agent Team
- `jax-pro` (lead JAX specialist)
- `neural-networks-master` (deep learning focus)
- `scientific-computing-master` (scientific computing integration)
- `systems-architect` (performance architecture)

#### **`scientific`** - Scientific Computing Focus
- `scientific-computing-master` (lead scientific expert)
- `jax-pro` (JAX ecosystem)
- `research-intelligence-master` (research methodology)
- `advanced-quantum-computing-expert` (quantum computing)
- Domain-specific experts based on computational domain

#### **`ai`** - AI/ML JAX Specialization
- `neural-networks-master` (lead ML expert)
- `jax-pro` (JAX deep learning)
- `ai-systems-architect` (AI system design)
- `data-professional` (data pipeline optimization)

#### **`research`** - Research-Grade JAX Computing
- `research-intelligence-master` (lead research expert)
- `jax-pro` (advanced JAX techniques)
- `scientific-computing-master` (computational methods)
- `advanced-quantum-computing-expert` (quantum computing research)

#### **`all`** - Complete 23-Agent JAX Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough JAX computing and cross-domain innovation.

### JAX Breakthrough Discovery (`--breakthrough`)
- **Cross-Domain JAX Innovation**: Novel JAX techniques from multiple scientific domains
- **Emergent JAX Patterns**: Advanced JAX patterns discovered through agent collaboration
- **Research-Grade Performance**: Academic and industry-leading JAX optimization standards
- **Quantum-JAX Integration**: Cutting-edge quantum-classical hybrid computing with JAX

## What it does

1. **JIT Compilation**: Apply `jax.jit` for performance optimization
2. **Gradients**: Compute gradients with `jax.grad` and `jax.value_and_grad`
3. **Vectorization**: Batch operations with `jax.vmap`
4. **Parallelization**: Multi-device execution with `jax.pmap`
5. **Combined Usage**: How to compose these transformations effectively

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap, pmap, value_and_grad

# Initialize PRNG key
key = random.PRNGKey(0)

# ============================================================================
# 1. JIT COMPILATION
# ============================================================================

# Basic JIT usage
@jit
def compute_sum(x, y):
    return jnp.dot(x, y) + jnp.sum(x)

# Alternative functional syntax
compute_sum_jit = jit(compute_sum)

# JIT with static arguments
@jit(static_argnums=(1,))  # Second argument is static
def compute_with_static(x, n_steps):
    result = x
    for _ in range(n_steps):  # n_steps must be static
        result = result * 2
    return result

# JIT Benefits:
# - XLA optimization for faster execution
# - Automatic GPU/TPU acceleration
# - Function fusion and memory optimization

# JIT Requirements and Pitfalls:
# - Input shapes must be static (known at compile time)
# - Array shapes cannot change between calls
# - Python control flow may require jax.lax.cond/while_loop

# ============================================================================
# 2. AUTOMATIC DIFFERENTIATION
# ============================================================================

# Basic gradient computation
def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

# Compute gradient with respect to first argument (params)
grad_fn = grad(loss_fn)
# gradients = grad_fn(params, x, y)

# Get both value and gradient (more efficient)
value_and_grad_fn = value_and_grad(loss_fn)
# loss_value, gradients = value_and_grad_fn(params, x, y)

# Gradients with respect to multiple arguments
grad_fn_multi = grad(loss_fn, argnums=(0, 1))  # w.r.t. params and x

# Higher-order derivatives
def simple_fn(x):
    return x**4 + 2*x**3 + x**2

first_deriv = grad(simple_fn)
second_deriv = grad(grad(simple_fn))  # Hessian for scalar functions

# For vector outputs, use jacobian
def vector_fn(x):
    return jnp.array([x**2, x**3, jnp.sin(x)])

jacobian_fn = jax.jacobian(vector_fn)

# Gradient tips:
# - Function must return scalar for jax.grad
# - Use jax.jacobian for vector-valued functions
# - Use has_aux=True if function returns auxiliary data

# ============================================================================
# 3. VECTORIZATION (VMAP)
# ============================================================================

# Basic vmap usage - batch processing
def single_prediction(params, x):
    return jnp.dot(x, params)

# Vectorize over batch dimension (first axis of x)
batch_prediction = vmap(single_prediction, in_axes=(None, 0))
# predictions = batch_prediction(params, batch_x)

# Vectorize gradients over batch
batch_grad = vmap(grad(loss_fn), in_axes=(None, 0, 0))
# batch_gradients = batch_grad(params, batch_x, batch_y)

# Custom axis mapping
def matrix_multiply(A, B):
    return jnp.dot(A, B)

# Vectorize over first axis of A, second axis of B
vmapped_matmul = vmap(matrix_multiply, in_axes=(0, 1), out_axes=0)

# Nested vmap for higher-dimensional batching
def element_wise_op(x, y):
    return x * y + x**2

# Double vectorization for 2D batching
double_vmap = vmap(vmap(element_wise_op, in_axes=(0, 0)), in_axes=(0, 0))

# Vmap benefits:
# - Automatic batching without explicit loops
# - Better performance than manual batching
# - Cleaner code for batch operations

# ============================================================================
# 4. PARALLELIZATION (PMAP)
# ============================================================================

# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")
print(f"Device count: {jax.device_count()}")

# Basic pmap usage - parallel across devices
def parallel_computation(x):
    return jnp.sum(x**2)

# Parallelize across all available devices
parallel_fn = pmap(parallel_computation)

# Use pmap with data sharded across devices
if jax.device_count() > 1:
    # Shape data for multiple devices
    data_shape = (jax.device_count(), 1000)  # (n_devices, data_per_device)
    key, subkey = random.split(key)
    sharded_data = random.normal(subkey, data_shape)

    # Parallel computation
    # results = parallel_fn(sharded_data)

# Pmap with axis_name for collective operations
@pmap(axis_name='devices')
def parallel_mean(x):
    return jax.lax.pmean(x, axis_name='devices')

# Pmap with gradients for distributed training
@pmap(axis_name='batch')
def parallel_grad_step(params, batch_x, batch_y):
    def loss(p):
        pred = vmap(lambda x: jnp.dot(x, p))(batch_x)
        return jnp.mean((pred - batch_y)**2)

    grad_p = grad(loss)(params)
    # Average gradients across devices
    return jax.lax.pmean(grad_p, axis_name='batch')

# ============================================================================
# 5. COMPOSITION AND BEST PRACTICES
# ============================================================================

# Effective composition of transformations
@jit
@vmap
def jitted_batch_operation(x, y):
    return jnp.dot(x, y)

# JIT + grad for fast gradient computation
fast_grad = jit(grad(loss_fn))

# Vmap + grad for batch gradients
batch_gradients = jit(vmap(grad(loss_fn), in_axes=(None, 0, 0)))

# Full training step with all transformations
@jit
def training_step(params, batch_x, batch_y, learning_rate):
    def batch_loss(p):
        predictions = vmap(lambda x: jnp.dot(x, p))(batch_x)
        return jnp.mean((predictions - batch_y)**2)

    loss_val, grad_p = value_and_grad(batch_loss)(params)
    new_params = params - learning_rate * grad_p
    return new_params, loss_val

# Memory-efficient gradient accumulation
def gradient_accumulation_step(params, batches, accumulation_steps):
    """Accumulate gradients over multiple batches"""
    def compute_batch_grad(batch):
        x, y = batch
        return grad(lambda p: loss_fn(p, x, y))(params)

    # Compute gradients for each batch
    all_grads = vmap(compute_batch_grad)(batches)
    # Average gradients
    return jnp.mean(all_grads, axis=0)

# Performance optimization patterns
def optimized_training_loop():
    """Best practices for performance"""

    # 1. JIT compile the entire training step
    @jit
    def step(params, batch):
        loss_val, grad_p = value_and_grad(loss_fn)(params, *batch)
        return params - 0.01 * grad_p, loss_val

    # 2. Use vmap for batch processing
    @vmap
    def process_batch(single_example):
        return single_example * 2  # Example processing

    # 3. Combine transformations efficiently
    optimized_fn = jit(vmap(grad(simple_fn)))

    return step, process_batch, optimized_fn

# Common transformation patterns
class JAXTransformations:
    """Common JAX transformation patterns"""

    @staticmethod
    def batch_apply(fn, batch_data):
        """Apply function to batch of data"""
        return vmap(fn)(batch_data)

    @staticmethod
    def parallel_batch_apply(fn, sharded_data):
        """Apply function in parallel across devices"""
        return pmap(vmap(fn))(sharded_data)

    @staticmethod
    def fast_gradient(loss_fn):
        """Fast compiled gradient function"""
        return jit(grad(loss_fn))

    @staticmethod
    def batch_gradients(loss_fn):
        """Batch gradient computation"""
        return jit(vmap(grad(loss_fn), in_axes=(None, 0, 0)))

# Debugging tips for transformations
def debugging_guide():
    """
    Common issues and solutions:

    JIT Issues:
    - Dynamic shapes cause recompilation
    - Use static_argnums for constants
    - Python control flow needs jax.lax equivalents

    Grad Issues:
    - Function must return scalar
    - Use has_aux=True for auxiliary outputs
    - Check for non-differentiable operations

    Vmap Issues:
    - Axis specifications must match data shapes
    - Broadcasting rules apply
    - Some functions may not be vmappable

    Pmap Issues:
    - Data must be sharded correctly
    - Device count must match leading dimension
    - Collective operations need axis_name
    """
    pass

# Example usage demonstration
def demonstrate_essentials():
    """Demonstrate core JAX operations"""

    # Sample data
    key, subkey = random.split(key)
    params = random.normal(subkey, (10,))
    x = random.normal(key, (100, 10))
    y = random.normal(key, (100,))

    print("=== JAX Essentials Demonstration ===")

    # 1. JIT compilation
    print("\n1. JIT Compilation:")
    jit_fn = jit(lambda p, x: jnp.dot(x, p))
    print("JIT compilation applied successfully")

    # 2. Gradient computation
    print("\n2. Gradient Computation:")
    grad_fn = grad(lambda p: jnp.sum((jnp.dot(x, p) - y)**2))
    print("Gradient function created")

    # 3. Vectorization
    print("\n3. Vectorization:")
    vmapped_fn = vmap(lambda xi: jnp.dot(xi, params))
    print("Vectorized function created")

    # 4. Combined usage
    print("\n4. Combined Transformations:")
    combined_fn = jit(vmap(grad(lambda p, xi: jnp.dot(xi, p)), in_axes=(None, 0)))
    print("Combined JIT + vmap + grad function created")

    return "JAX essentials demonstration completed!"

# Run demonstration
demonstrate_essentials()
```

## Key Concepts

### JIT Compilation
- Compiles functions for faster execution using XLA
- Requires static shapes and control flow
- Use `static_argnums` for compile-time constants
- Automatically handles GPU/TPU acceleration

### Automatic Differentiation
- Forward and reverse mode automatic differentiation
- Use `grad` for scalar-valued functions
- Use `jacobian` for vector-valued functions
- Compose with other transformations for complex workflows

### Vectorization (vmap)
- Efficiently batch operations without loops
- Specify input/output axes for flexible batching
- Can be nested for multi-dimensional operations
- Integrates seamlessly with other transformations

### Parallelization (pmap)
- Distributes computation across multiple devices
- Requires properly sharded input data
- Use collective operations (pmean, psum) for communication
- Essential for large-scale distributed training

## Common Workflows

### Basic JAX Development with Intelligent Agents
```bash
# 1. Initialize JAX project with intelligent setup
/jax-init --agents=auto --intelligent

# 2. Learn core operations with JAX specialist guidance
/jax-essentials --operation=jit --static-args --agents=jax --intelligent

# 3. Debug and optimize with expert agents
/jax-debug --check-tracers --agents=jax --orchestrate
/jax-performance --technique=caching --agents=jax --breakthrough
```

### Machine Learning with JAX and AI Agents
```bash
# 1. Set up core transformations with AI expertise
/jax-essentials --operation=grad --higher-order --agents=ai --intelligent

# 2. Build models with neural network specialists
/jax-models --framework=flax --architecture=mlp --agents=ai --orchestrate

# 3. Training workflow with optimization experts
/jax-training --optimizer=adam --schedule=cosine --agents=ai --breakthrough
```

### Performance Optimization with Scientific Computing Agents
```bash
# 1. Profile current code with JAX specialists
/jax-essentials --operation=jit --devices --agents=jax --intelligent
/jax-performance --gpu-accel --profiling --agents=scientific --orchestrate

# 2. Optimize transformations with breakthrough techniques
/optimize jax_code.py --language=jax --implement --agents=scientific --breakthrough

# 3. Validate performance with comprehensive testing
/run-all-tests --gpu --benchmark --agents=scientific --orchestrate
```

### Research-Grade JAX Computing
```bash
# 1. Research methodology with expert guidance
/jax-essentials --operation=all --agents=research --intelligent --breakthrough

# 2. Advanced scientific computing patterns
/jax-sparse-ops --operation=jacobian --agents=research --orchestrate
/jax-numpyro-prob --model-type=hierarchical --agents=research --breakthrough

# 3. Publication-ready validation and documentation
/run-all-tests --reproducible --agents=research --orchestrate
/update-docs jax_research/ --type=research --agents=research --publish
```

### Cross-Domain JAX Innovation
```bash
# 1. Multi-agent ecosystem analysis
/jax-essentials --operation=all --agents=all --orchestrate --breakthrough

# 2. Quantum-classical hybrid computing
/jax-essentials --operation=pmap --agents=quantum --breakthrough
/optimize quantum_jax.py --agents=quantum --implement

# 3. Complete JAX mastery with all agents
/jax-performance --agents=all --orchestrate --breakthrough
/multi-agent-optimize jax_project/ --agents=all --mode=research
```

## Related Commands

**Prerequisites**: Commands to run first
- `/jax-init` - Initialize JAX project with imports

**Core JAX Operations**: Related JAX functionality
- `/jax-debug` - Debug JAX compilation and transformation issues
- `/jax-performance` - Performance optimization techniques
- `/jax-models` - Neural network models with Flax/Equinox
- `/jax-training` - Training workflows with optimization

**Integration**: Commands that work with JAX
- `/optimize` - General optimization including JAX
- `/debug --gpu` - GPU debugging for JAX code
- `/generate-tests` - JAX-specific test generation

**Follow-up**: Next steps after learning essentials
- `/jax-models --framework=equinox` - Build models
- `/jax-numpyro-prob` - Probabilistic programming
- `/jax-orbax-checkpoint` - Model checkpointing

## Integration Patterns

### JAX Learning Path
```bash
# Progressive JAX mastery
/jax-init                           # Setup
/jax-essentials --operation=jit     # Core concepts
/jax-models --framework=flax        # Model building
/jax-training --epochs=100          # Training
```

### JAX Debugging Workflow
```bash
# Debug JAX transformations
/jax-essentials --operation=grad --higher-order
/jax-debug --disable-jit --print-values
/jax-performance --optimization
```

### Scientific Computing Pipeline
```bash
# Complete JAX scientific workflow
/jax-essentials --devices --static-args
/jax-sparse-ops --operation=jacobian
/jax-nlsq --algorithm=TRR --gpu-accel
```

ARGUMENTS: [--operation=jit|grad|vmap|pmap] [--higher-order] [--static-args] [--devices] [--agents=auto|jax|scientific|ai|research|all] [--orchestrate] [--intelligent] [--breakthrough]