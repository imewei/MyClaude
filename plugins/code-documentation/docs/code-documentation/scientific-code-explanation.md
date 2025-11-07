# Scientific Code Explanation

**Version**: 1.0.3
**Category**: code-documentation
**Purpose**: Specialized explanations for scientific computing, numerical methods, and data science code

## Overview

Domain-specific code explanations for NumPy/SciPy, JAX, Pandas, Julia, molecular dynamics simulations, and machine learning training loops with emphasis on numerical methods and performance.

## NumPy/SciPy Array Operations

### Array Broadcasting and Vectorization

```markdown
## Array Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes efficiently.

### Broadcasting Rules:
1. If arrays have different numbers of dimensions, prepend 1s to smaller shape
2. Arrays are compatible if dimensions are equal or one of them is 1
3. Result shape is element-wise maximum of input shapes

### Example:
```python
import numpy as np

# Shape (3, 1)
a = np.array([[1],
              [2],
              [3]])

# Shape (4,)
b = np.array([10, 20, 30, 40])

# Broadcasting: (3, 1) + (4,) -> (3, 4)
result = a + b
# [[11, 21, 31, 41],
#  [12, 22, 32, 42],
#  [13, 23, 33, 43]]
```

### Step-by-Step Transformation:
```
a.shape = (3, 1)
b.shape = (4,)

Step 1: Prepend 1 to b
b.shape = (1, 4)

Step 2: Stretch dimensions of size 1
a broadcasted: (3, 4) - repeat each row 4 times
b broadcasted: (3, 4) - repeat single row 3 times

Step 3: Element-wise operation
result.shape = (3, 4)
```

### Vectorization Performance:
```python
# Slow: Python loop
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# Fast: Vectorized
def fast_sum(arr):
    return np.sum(arr)

# Benchmark (1M elements):
# slow_sum: 150ms
# fast_sum: 1.2ms (125x faster!)
```

### Memory Layout (C vs Fortran Order):
```python
# C-order (row-major): default in NumPy
arr_c = np.array([[1, 2, 3],
                  [4, 5, 6]], order='C')
# Memory: [1, 2, 3, 4, 5, 6]

# Fortran-order (column-major): used in MATLAB, Julia
arr_f = np.array([[1, 2, 3],
                  [4, 5, 6]], order='F')
# Memory: [1, 4, 2, 5, 3, 6]

# Performance implications:
# C-order: Fast row-wise operations
# F-order: Fast column-wise operations
```

### Numerical Precision Considerations:
```python
# Float32 vs Float64
arr32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 4 bytes/element
arr64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 8 bytes/element

# Precision vs Memory trade-off:
# float32: ~7 decimal digits, half memory
# float64: ~15 decimal digits, standard precision

# Catastrophic cancellation example:
a = np.float32(1.0)
b = np.float32(1.0 + 1e-7)
result = b - a  # Loss of precision!
```
```

## JAX Functional Transformations

### Understanding JAX Transformations

```markdown
## JAX Function Transformations

JAX provides functional transformations for high-performance numerical computing.

### 1. @jax.jit - Just-In-Time Compilation

**Purpose**: Compile to XLA for GPU/TPU execution

```python
import jax
import jax.numpy as jnp

def slow_function(x):
    return jnp.sum(x ** 2)

# JIT compiled version
@jax.jit
def fast_function(x):
    return jnp.sum(x ** 2)

x = jnp.arange(1000000)

# First call: Compiles (slow ~100ms)
result = fast_function(x)

# Subsequent calls: Use cached compilation (fast ~1ms)
result = fast_function(x)
```

**Requirements**:
- Pure functions (no side effects)
- No Python control flow (use jnp.where, lax.cond instead)
- Traceable operations only

### 2. jax.grad - Automatic Differentiation

**Purpose**: Compute gradients automatically

```python
import jax

def loss_function(params, x, y):
    pred = params['w'] * x + params['b']
    return jnp.mean((pred - y) ** 2)

# Gradient with respect to params
grad_fn = jax.grad(loss_function)

params = {'w': 2.0, 'b': 1.0}
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([3.0, 5.0, 7.0])

# Compute gradients
grads = grad_fn(params, x, y)
# grads = {'w': ..., 'b': ...}
```

**Gradient Modes**:
- **Reverse-mode**: Default, efficient for scalar outputs
- **Forward-mode**: Use `jax.jvp` for Jacobian-vector products

### 3. jax.vmap - Auto-Vectorization

**Purpose**: Map function over batch dimension

```python
# Without vmap: Manual loop
def apply_to_batch_slow(model, batch):
    return jnp.stack([model(x) for x in batch])

# With vmap: Automatic vectorization
apply_to_batch_fast = jax.vmap(model)

batch = jnp.array([[1, 2], [3, 4], [5, 6]])
result = apply_to_batch_fast(batch)
```

**Benefits**:
- No manual loop writing
- Automatically parallelizes
- Composes with jit and grad

### 4. jax.pmap - Multi-Device Parallelism

**Purpose**: Data parallelism across GPUs/TPUs

```python
# Replicate function across devices
@jax.pmap
def parallel_computation(x):
    return x ** 2

# Data split across 8 GPUs
n_devices = 8
x = jnp.arange(n_devices * 100).reshape(n_devices, 100)
result = parallel_computation(x)
```

### Key JAX Concepts:

**1. Pure Functional Programming**
```python
# Bad: Mutation
def impure(x):
    x[0] = 0  # Mutates input!
    return x

# Good: No mutation
def pure(x):
    return x.at[0].set(0)  # Returns new array
```

**2. Explicit Random Key Threading**
```python
import jax.random as random

# Bad: Implicit global state (not allowed)
# random.seed(0)
# x = random.normal()

# Good: Explicit key management
key = random.PRNGKey(0)
key, subkey = random.split(key)
x = random.normal(subkey, shape=(100,))
```

**3. PyTrees for Nested Structures**
```python
# JAX understands nested dicts/lists
params = {
    'layer1': {'w': jnp.array(...), 'b': jnp.array(...)},
    'layer2': {'w': jnp.array(...), 'b': jnp.array(...)}
}

# grad/vmap work on PyTrees
grads = jax.grad(loss)(params)
# Returns same structure with gradients
```
```

## Pandas Data Operations

### Efficient DataFrame Operations

```markdown
## Pandas Method Chaining and Performance

### Method Chaining vs Assignment
```python
import pandas as pd

# Method chaining (preferred)
result = (df
    .query('age > 18')
    .assign(age_squared=lambda x: x['age'] ** 2)
    .groupby('category')
    .agg({'age': 'mean', 'age_squared': 'sum'})
    .reset_index()
)

# Multiple assignments (less readable)
result = df[df['age'] > 18].copy()
result['age_squared'] = result['age'] ** 2
result = result.groupby('category').agg({'age': 'mean', 'age_squared': 'sum'})
result = result.reset_index()
```

### GroupBy Split-Apply-Combine Pattern
```python
# Group data by category
grouped = df.groupby('category')

# Apply aggregation
result = grouped.agg({
    'sales': ['sum', 'mean', 'count'],
    'profit': 'sum'
})

# Custom aggregation functions
def custom_metric(x):
    return (x.max() - x.min()) / x.mean()

result = grouped['sales'].agg([
    ('total', 'sum'),
    ('average', 'mean'),
    ('volatility', custom_metric)
])
```

### Memory-Efficient Operations
```python
# Bad: Creates copy
df_filtered = df[df['age'] > 18].copy()

# Better: View (when possible)
df_view = df.loc[df['age'] > 18]

# Use category dtype for repeated strings
df['category'] = df['category'].astype('category')
# Memory reduction: 50-90% for categorical data

# Use appropriate numeric types
df['age'] = df['age'].astype('int8')  # -128 to 127
df['count'] = df['count'].astype('uint32')  # 0 to 4B
```

### Index Alignment
```python
# Pandas automatically aligns on index
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])

result = s1 + s2
# a    NaN
# b    6.0
# c    8.0
# d    NaN
```
```

## Julia High-Performance Computing

### Julia Performance Patterns

```markdown
## Julia Type Stability

Type stability means the return type is predictable from input types, enabling LLVM optimization.

### Type-Stable Code:
```julia
# Type-stable: Always returns Float64
function stable_func(x::Float64)::Float64
    if x > 0
        return sqrt(x)
    else
        return 0.0  # Same type as sqrt
    end
end

# Check type stability
@code_warntype stable_func(5.0)
# Body::Float64 (good!)
```

### Type-Unstable Code:
```julia
# Type-unstable: Returns Float64 OR Int64
function unstable_func(x::Float64)
    if x > 0
        return sqrt(x)  # Float64
    else
        return 0        # Int64 (BAD!)
    end
end

# Check type stability
@code_warntype unstable_func(5.0)
# Body::Union{Float64, Int64} (bad!)
```

### Multiple Dispatch
```julia
# Define function for different types
area(r::Real) = π * r^2                    # Circle
area(w::Real, h::Real) = w * h              # Rectangle
area(a::Real, b::Real, c::Real) = ...       # Triangle

# Julia selects method based on ALL argument types
area(5)         # Calls circle version
area(3, 4)      # Calls rectangle version
area(3, 4, 5)   # Calls triangle version
```

### Performance Annotations
```julia
function optimized_sum(arr::Vector{Float64})
    total = 0.0
    @inbounds @simd for i in eachindex(arr)
        total += arr[i]
    end
    return total
end

# @inbounds: Skip bounds checking (unsafe but fast)
# @simd: SIMD vectorization hints
```

### Memory Management
```julia
# Pre-allocate arrays
function efficient_compute!(result::Vector{Float64}, input::Vector{Float64})
    @. result = sqrt(input)  # In-place, no allocation
end

# Avoid allocations in loops
function no_allocation_loop(n::Int)
    sum = 0.0
    @inbounds for i in 1:n
        sum += i^2  # No array allocation
    end
    return sum
end

# Broadcast with .= for in-place operations
result .= input .+ 1  # In-place
result = input .+ 1   # Allocates new array
```
```

## Molecular Dynamics and Simulations

### Simulation Code Structure

```markdown
## Velocity Verlet Integration

Standard integrator for molecular dynamics simulations.

### Algorithm:
```
1. Compute forces: F(t)
2. Update velocities to half-step: v(t+dt/2) = v(t) + (F(t)/m) * (dt/2)
3. Update positions: x(t+dt) = x(t) + v(t+dt/2) * dt
4. Recompute forces: F(t+dt)
5. Complete velocity update: v(t+dt) = v(t+dt/2) + (F(t+dt)/m) * (dt/2)
```

### Python Implementation:
```python
import numpy as np

def velocity_verlet(positions, velocities, forces, mass, dt):
    """
    Velocity Verlet integrator

    Args:
        positions: (N, 3) array
        velocities: (N, 3) array
        forces: (N, 3) array
        mass: float or (N,) array
        dt: timestep
    """
    # Half-step velocity update
    velocities += 0.5 * (forces / mass) * dt

    # Full-step position update
    positions += velocities * dt

    # Recompute forces
    forces = compute_forces(positions)

    # Complete velocity update
    velocities += 0.5 * (forces / mass) * dt

    return positions, velocities, forces
```

### Performance Considerations:
- **Neighbor Lists**: O(N) force calculation instead of O(N²)
- **Timestep Selection**: dt < 1/10 of fastest oscillation period
- **Energy Conservation**: Monitor total energy as quality check
- **Periodic Boundary Conditions**: Minimum image convention

### Frameworks:
- **LAMMPS**: Large-scale parallel MD
- **GROMACS**: Biomolecular simulations
- **ASE**: Atomistic Simulation Environment (Python)
- **HOOMD-blue**: GPU-accelerated MD
```

## Machine Learning Training Loops

### Training Architecture

```markdown
## Standard ML Training Loop

### PyTorch Pattern:
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 1. Forward pass
        predictions = model(batch_x)

        # 2. Compute loss
        loss = criterion(predictions, batch_y)

        # 3. Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()         # Compute gradients

        # 4. Update parameters
        optimizer.step()

        # 5. Track metrics
        train_loss += loss.item()
```

### JAX Pattern (Functional):
```python
import jax
import jax.numpy as jnp
import optax

def loss_fn(params, x, y):
    predictions = model_apply(params, x)
    return jnp.mean((predictions - y) ** 2)

# Initialize
params = init_params()
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 1. Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y)

        # 2. Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
```

### Key Differences:
**PyTorch**:
- Imperative style
- Automatic gradient tracking
- In-place parameter updates

**JAX**:
- Functional style
- Explicit state threading
- Immutable parameters (returns new params)
```

## Numerical Stability

```markdown
## Common Numerical Issues

### 1. Log-Sum-Exp Trick
```python
# Problem: exp(x) overflows for large x
def naive_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))  # Overflow!

# Solution: Subtract max before exp
def stable_softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
```

### 2. Catastrophic Cancellation
```python
# Problem: a - b when a ≈ b loses precision
a = 1.0 + 1e-15
b = 1.0
result = a - b  # Loss of precision!

# Solution: Reformulate algorithm
# Example: Quadratic formula
# Bad:  (-b - sqrt(b^2 - 4ac)) / 2a
# Good: -2c / (b + sqrt(b^2 - 4ac))
```

### 3. Condition Numbers
```python
import numpy as np

A = np.array([[1, 2], [1.0001, 2]])
cond = np.linalg.cond(A)  # Condition number

# cond >> 1: Ill-conditioned (small changes → big errors)
# cond ~  1: Well-conditioned
```
```
