# Code Optimization Patterns

## Overview

This guide contains proven optimization patterns with before/after code examples, expected speedups, and applicability guidelines. All patterns include real-world metrics from production optimizations.

**Quick Reference**: See [Domain-Specific Patterns](#domain-specific-patterns) for specialized optimizations (scientific computing, ML, web).

---

## Table of Contents

1. [Vectorization](#vectorization)
2. [JIT Compilation](#jit-compilation)
3. [Caching & Memoization](#caching--memoization)
4. [Parallelization](#parallelization)
5. [GPU Acceleration](#gpu-acceleration)
6. [Memory Optimization](#memory-optimization)
7. [Algorithm Selection](#algorithm-selection)
8. [I/O Optimization](#io-optimization)

---

## Vectorization

### Pattern: Replace Python Loops with NumPy Operations

**Applicability**: Array/matrix operations, element-wise computations, mathematical transformations

**Expected Speedup**: 10-100x

**Before**:
```python
import numpy as np

# Slow: Python loop (0.85s for 1M elements)
data = np.random.rand(1000000)
result = np.zeros_like(data)
for i in range(len(data)):
    result[i] = data[i] ** 2 + 2 * data[i] + 1
```

**After**:
```python
# Fast: Vectorized (0.008s for 1M elements)
result = data ** 2 + 2 * data + 1
# Speedup: 106x
```

**Why it works**: NumPy operations are implemented in C, avoid Python interpreter overhead, and use SIMD instructions.

### Pattern: Pandas apply() → Vectorized Operations

**Before**:
```python
import pandas as pd

df = pd.DataFrame({'A': range(1000000), 'B': range(1000000)})

# Slow: apply() with lambda (12.3s)
df['C'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
```

**After**:
```python
# Fast: Vectorized column operations (0.05s)
df['C'] = df['A'] + df['B']
# Speedup: 246x
```

**When to use vectorization**:
- ✅ Element-wise operations
- ✅ Mathematical functions (sin, exp, log)
- ✅ Boolean masking and filtering
- ❌ Complex conditional logic (use np.where or numba instead)
- ❌ Operations with side effects

---

## JIT Compilation

### Pattern: JAX @jit Decorator

**Applicability**: Pure functions with NumPy-like operations, especially in loops

**Expected Speedup**: 5-50x (first call slow due to compilation)

**Before**:
```python
import jax.numpy as jnp

# Without JIT (0.42s)
def compute_distance(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))

# Many calls
for i in range(10000):
    result = compute_distance(arr1, arr2)
```

**After**:
```python
from jax import jit

# With JIT (0.009s)
@jit
def compute_distance(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))

# First call: compilation overhead (~1s)
# Subsequent calls: 47x faster
for i in range(10000):
    result = compute_distance(arr1, arr2)
```

**JAX JIT Requirements**:
- Function must be pure (no side effects)
- All array shapes must be known at compile time
- No Python control flow depending on array values (use jax.lax instead)

### Pattern: Numba @njit for CPU-Bound Loops

**Before**:
```python
# Slow Python loop (5.2s)
def monte_carlo_pi(n_samples):
    inside = 0
    for i in range(n_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 < 1:
            inside += 1
    return 4 * inside / n_samples
```

**After**:
```python
import numba

@numba.njit
def monte_carlo_pi(n_samples):
    inside = 0
    for i in range(n_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 < 1:
            inside += 1
    return 4 * inside / n_samples
# Speedup: 35x (0.15s)
```

---

## Caching & Memoization

### Pattern: @lru_cache for Expensive Pure Functions

**Applicability**: Repeated calls with same inputs, recursive functions, expensive computations

**Expected Speedup**: 2-10x for repeated calls, up to 1000x for recursive algorithms

**Before**:
```python
# Slow: Recomputes every time (fibonacci 35: 3.2s)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(35)  # 9227465 recursive calls
```

**After**:
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(35)  # Only 36 unique calls (0.0001s)
# Speedup: 32000x
```

**When to use caching**:
- ✅ Pure functions (same input → same output)
- ✅ Expensive computations
- ✅ Frequent repeated calls
- ❌ Functions with side effects
- ❌ Large result objects (memory cost)
- ❌ Functions with mutable arguments

### Pattern: Database Query Result Caching

**Before**:
```python
# Slow: Query database on every call
def get_user_profile(user_id):
    return db.query("SELECT * FROM users WHERE id = ?", user_id)

# Called 1000x/sec: Heavy DB load
```

**After**:
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_user_profile(user_id):
    return db.query("SELECT * FROM users WHERE id = ?", user_id)

# First call: DB query
# Subsequent calls: Instant from cache
# Cache invalidation: Manually clear on user updates
```

---

## Parallelization

### Pattern: Multiprocessing for CPU-Bound Tasks

**Applicability**: Independent iterations, CPU-bound tasks, no shared state

**Expected Speedup**: Nx (N = number of cores)

**Before**:
```python
# Serial processing (45s on 8-core machine)
results = []
for item in large_dataset:  # 10,000 items
    result = expensive_cpu_function(item)
    results.append(result)
```

**After**:
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Parallel processing (6.2s on 8-core machine)
with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    results = list(executor.map(expensive_cpu_function, large_dataset))
# Speedup: 7.3x (near-linear scaling)
```

**Parallelization Considerations**:
- **Overhead**: Data serialization, process spawning
- **GIL**: Python's Global Interpreter Lock (use multiprocessing, not threading for CPU-bound)
- **Memory**: Each process has separate memory space
- **Order**: Use `executor.map()` to preserve order

### Pattern: Async I/O for I/O-Bound Tasks

**Before**:
```python
import requests

# Serial API calls (15.3s for 100 calls)
results = []
for url in urls:  # 100 URLs
    response = requests.get(url)
    results.append(response.json())
```

**After**:
```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Parallel API calls (0.8s for 100 calls)
results = asyncio.run(fetch_all(urls))
# Speedup: 19x
```

---

## GPU Acceleration

### Pattern: NumPy → JAX for GPU Computation

**Applicability**: Large matrix operations, deep learning, scientific simulations

**Expected Speedup**: 10-1000x (depending on GPU vs CPU)

**Before**:
```python
import numpy as np

# CPU computation (2.4s)
A = np.random.rand(5000, 5000)
B = np.random.rand(5000, 5000)
C = A @ B  # Matrix multiplication
```

**After**:
```python
import jax.numpy as jnp
from jax import device_put

# GPU computation (0.04s)
A = device_put(jnp.array(A))  # Move to GPU
B = device_put(jnp.array(B))
C = A @ B  # Computed on GPU
# Speedup: 60x (on NVIDIA A100)
```

**GPU Optimization Tips**:
1. **Minimize data transfer**: Keep data on GPU as long as possible
2. **Batch operations**: Combine small operations into larger ones
3. **Use mixed precision**: fp16 for 2x speedup with minimal accuracy loss
4. **Profile**: Use `jax.profiler` to identify bottlenecks

---

## Memory Optimization

### Pattern: Generator Expressions vs List Comprehensions

**Before**:
```python
# Memory: 800MB for 100M integers
large_list = [i ** 2 for i in range(100_000_000)]
result = sum(large_list)
```

**After**:
```python
# Memory: 128 bytes (constant)
large_gen = (i ** 2 for i in range(100_000_000))
result = sum(large_gen)
# Memory reduction: 6250x
```

### Pattern: In-Place Operations

**Before**:
```python
# Creates new array (memory copy)
data = data + 1  # New memory allocation
```

**After**:
```python
# Modifies in-place (no memory copy)
data += 1  # Same memory location
```

**JAX In-Place Updates**:
```python
# JAX arrays are immutable, use .at[] for "in-place" style
data = data.at[indices].set(new_values)
```

---

## Algorithm Selection

### Pattern: O(n²) → O(n log n) with Better Algorithm

**Before**:
```python
# Bubble sort: O(n²) - 12.5s for 10,000 elements
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

**After**:
```python
# TimSort (Python's built-in): O(n log n) - 0.003s
sorted_arr = sorted(arr)
# Speedup: 4167x
```

### Pattern: scipy.spatial.cKDTree for Nearest Neighbors

**Before**:
```python
# Naive nearest neighbor: O(n²) - 8.2s for 10,000 points
def find_nearest(point, points):
    distances = [np.linalg.norm(point - p) for p in points]
    return np.argmin(distances)
```

**After**:
```python
from scipy.spatial import cKDTree

# KD-Tree: O(log n) per query - 0.05s
tree = cKDTree(points)
distance, index = tree.query(point)
# Speedup: 164x
```

---

## I/O Optimization

### Pattern: Batch Database Operations

**Before**:
```python
# 1000 individual inserts (45s)
for user in users:
    db.execute("INSERT INTO users VALUES (?, ?, ?)", user)
```

**After**:
```python
# Single batch insert (0.8s)
db.executemany("INSERT INTO users VALUES (?, ?, ?)", users)
# Speedup: 56x
```

### Pattern: Read Files in Chunks

**Before**:
```python
# Loads entire 10GB file into memory
with open('large_file.txt', 'r') as f:
    data = f.read()
    process(data)
```

**After**:
```python
# Streams file in 1MB chunks
with open('large_file.txt', 'r') as f:
    for chunk in iter(lambda: f.read(1024*1024), ''):
        process(chunk)
# Memory: 10GB → 1MB (10000x reduction)
```

---

## Domain-Specific Patterns

### Scientific Computing
See [scientific-patterns.md](scientific-patterns.md) for:
- NumPy broadcasting
- SciPy sparse matrices
- FFT-based convolution
- Numerical stability (log-sum-exp trick)

### Machine Learning
See [ml-optimization.md](ml-optimization.md) for:
- Mixed precision training
- Gradient accumulation
- Data loading optimization
- Model quantization

### Web Performance
See [web-performance.md](web-performance.md) for:
- N+1 query elimination
- Connection pooling
- CDN and caching strategies
- Lazy loading

---

## Pattern Selection Guide

| Bottleneck | Pattern | Speedup | Effort |
|------------|---------|---------|--------|
| Python loops | Vectorization | 10-100x | Low |
| Repeated function calls | Caching | 2-1000x | Very Low |
| CPU-bound loops | JIT (Numba) | 10-50x | Low |
| Mathematical operations | JIT (JAX) | 5-50x | Medium |
| Independent tasks | Parallelization | Nx | Medium |
| Large matrix ops | GPU (JAX/CUDA) | 10-1000x | High |
| Database queries | Batching | 10-100x | Low |
| Memory issues | Generators/Streaming | N/A | Low |
| Wrong algorithm | Algorithm change | 10-10000x | High |

---

## Best Practices

1. **Profile first**: Use cProfile, line_profiler to find actual bottlenecks
2. **Measure impact**: Benchmark before/after with realistic data
3. **Start simple**: Vectorization and caching often sufficient
4. **Read-only data**: Most optimizations require immutability
5. **Test correctness**: Optimizations can introduce bugs
6. **Document tradeoffs**: Note memory/complexity costs

---

## Additional Resources

- [Scientific Patterns](scientific-patterns.md)
- [ML Optimization](ml-optimization.md)
- [Web Performance](web-performance.md)
- [Profiling Tools](profiling-tools.md)
- [Performance Engineering](performance-engineering.md)

---

**Last Updated**: 2025-06-11
**Version**: 1.0.0
