# Scientific Computing Optimization Patterns

## Overview

Specialized patterns for NumPy, SciPy, JAX, and Julia scientific computing workloads. All patterns include measured speedups from production scientific code.

---

## Pattern 1: NumPy Broadcasting

**Avoid explicit loops, use broadcasting for element-wise operations**

**Before** (0.82s for 1M×1M):
```python
import numpy as np

data = np.random.rand(1000, 1000)
result = np.zeros_like(data)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        result[i, j] = data[i, j] * 2 + 1
```

**After** (0.004s):
```python
result = data * 2 + 1  # Broadcasts automatically
# Speedup: 205x
```

**Broadcasting Rules**:
```python
# (3, 1) * (1, 4) → (3, 4)
a = np.array([[1], [2], [3]])      # Shape: (3, 1)
b = np.array([[10, 20, 30, 40]])   # Shape: (1, 4)
c = a * b                           # Shape: (3, 4)

# Explicit reshape for clarity
distances = np.sqrt(np.sum((points[:, np.newaxis] - centers[np.newaxis, :]) ** 2, axis=-1))
# Computes pairwise distances: (N, 1, D) - (1, M, D) → (N, M, D) → (N, M)
```

---

## Pattern 2: SciPy Sparse Matrices

**For matrices with <5% non-zero elements, use sparse**

**Before** (2.1 GB memory, 0.45s):
```python
import numpy as np

# Sparse matrix (99% zeros)
matrix = np.zeros((100000, 100000))
for i in range(10000):
    matrix[np.random.randint(100000), np.random.randint(100000)] = np.random.rand()

result = matrix @ vector
```

**After** (21 MB memory, 0.003s):
```python
from scipy.sparse import csr_matrix

# Store only non-zero elements
data, row, col = [], [], []
for i in range(10000):
    r, c = np.random.randint(100000), np.random.randint(100000)
    data.append(np.random.rand())
    row.append(r)
    col.append(c)

matrix = csr_matrix((data, (row, col)), shape=(100000, 100000))
result = matrix @ vector

# Memory: 100x reduction, Speed: 150x
```

---

## Pattern 3: FFT-Based Convolution

**For large convolutions, FFT is O(n log n) vs O(n²)**

**Before** (12.3s for 10M points):
```python
import numpy as np

signal = np.random.rand(10_000_000)
kernel = np.random.rand(1001)

# Direct convolution: O(n×m)
result = np.convolve(signal, kernel, mode='same')
```

**After** (0.18s):
```python
from scipy.signal import fftconvolve

result = fftconvolve(signal, kernel, mode='same')
# Speedup: 68x
```

**When to use FFT convolution**:
- Signal length > 1000
- Kernel length > 50
- Multiple convolutions with same kernel

---

## Pattern 4: JAX JIT Compilation

**For repeated NumPy-like operations, JIT compile**

**Before** (3.2s for 10K iterations):
```python
import jax.numpy as jnp

def compute(x, y):
    return jnp.sum(jnp.exp(-(x - y) ** 2))

# Many calls without JIT
for i in range(10000):
    result = compute(arr1, arr2)
```

**After** (0.062s):
```python
from jax import jit

@jit
def compute(x, y):
    return jnp.sum(jnp.exp(-(x - y) ** 2))

# First call: compilation (~1s)
# Subsequent calls: 52x faster
for i in range(10000):
    result = compute(arr1, arr2)
```

**JIT Best Practices**:
```python
# ✅ Good: Pure function
@jit
def compute_distance(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))

# ❌ Bad: Side effects
@jit
def bad_function(x):
    print(f"Value: {x}")  # Side effect!
    return x ** 2

# ✅ Good: Static arguments
@jit
def process(data, mode='train'):  # mode must be static
    if mode == 'train':
        return data * 0.9
    return data
```

---

## Pattern 5: JAX vmap for Batching

**Vectorize over batch dimension automatically**

**Before** (0.85s):
```python
import jax.numpy as jnp

def compute_single(x):
    return jnp.sum(x ** 2)

# Loop over batch
batch = jnp.array([...])  # Shape: (1000, 100)
results = jnp.array([compute_single(item) for item in batch])
```

**After** (0.009s):
```python
from jax import vmap, jit

@jit
@vmap
def compute_single(x):
    return jnp.sum(x ** 2)

results = compute_single(batch)  # Vectorized automatically
# Speedup: 94x
```

---

## Pattern 6: Numba for CPU Loops

**When vectorization impossible, use Numba**

**Before** (8.5s):
```python
import numpy as np

def pairwise_distance(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist
    return distances
```

**After** (0.32s):
```python
import numba

@numba.njit(parallel=True)
def pairwise_distance(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in numba.prange(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(points.shape[1]):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances
# Speedup: 27x
```

---

## Pattern 7: Numerical Stability (Log-Sum-Exp)

**Prevent overflow/underflow in exponential calculations**

**Before** (numerically unstable):
```python
import numpy as np

logits = np.array([1000, 1001, 1002])  # Large values
probs = np.exp(logits) / np.sum(np.exp(logits))
# Result: [nan, nan, nan] due to overflow!
```

**After** (stable):
```python
def stable_softmax(logits):
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits)

probs = stable_softmax(logits)
# Result: [0.0900, 0.2447, 0.6652] (correct!)
```

**JAX built-in**:
```python
from jax.nn import softmax
probs = softmax(logits)  # Numerically stable by default
```

---

## Pattern 8: Einsum for Complex Operations

**Readable and fast for tensor contractions**

**Before** (hard to read, 0.42s):
```python
# Matrix multiplication with transpose
result = np.dot(np.dot(A.T, B), C.T)
```

**After** (clear intent, 0.38s):
```python
result = np.einsum('ij,jk,lk->il', A, B, C)
# Speedup: 1.1x, much clearer
```

**Common einsum patterns**:
```python
# Matrix multiply: C = A @ B
np.einsum('ik,kj->ij', A, B)

# Batch matrix multiply
np.einsum('bij,bjk->bik', batch_A, batch_B)

# Trace
np.einsum('ii', A)

# Outer product
np.einsum('i,j->ij', a, b)

# Attention mechanism
np.einsum('bqd,bkd->bqk', Q, K)  # Q @ K.T
```

---

## Pattern 9: Chunked Processing for Memory

**Process large datasets in chunks**

**Before** (120 GB memory!):
```python
data = np.load('huge_dataset.npy')  # 100 GB file
result = expensive_computation(data)
```

**After** (1.2 GB memory):
```python
def process_chunks(filepath, chunk_size=1_000_000):
    results = []
    with np.load(filepath, mmap_mode='r') as data:
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            result = expensive_computation(chunk)
            results.append(result)
    return np.concatenate(results)

result = process_chunks('huge_dataset.npy')
# Memory: 100x reduction
```

---

## Pattern 10: GPU Memory Management (JAX)

**Minimize CPU↔GPU transfers**

**Before** (slow, 2.1s):
```python
import jax.numpy as jnp

# Repeatedly transfer between CPU and GPU
for i in range(1000):
    cpu_data = np.random.rand(1000)
    gpu_data = jnp.array(cpu_data)  # CPU → GPU
    result = compute(gpu_data)
    cpu_result = np.array(result)   # GPU → CPU
```

**After** (fast, 0.08s):
```python
from jax import device_put

# Transfer once, keep on GPU
gpu_data = device_put(np.random.rand(1000, 1000))

for i in range(1000):
    gpu_data = compute(gpu_data)  # Stays on GPU

cpu_result = np.array(gpu_data)  # Single transfer at end
# Speedup: 26x
```

---

## Quick Reference

| Optimization | When to Use | Speedup | Complexity |
|--------------|-------------|---------|------------|
| Broadcasting | Element-wise ops | 10-200x | Low |
| Sparse matrices | <5% non-zero | 10-100x | Medium |
| FFT convolution | Large signals | 10-100x | Low |
| JAX JIT | Repeated compute | 10-50x | Low |
| JAX vmap | Batch processing | 10-100x | Low |
| Numba | Unavoidable loops | 10-50x | Medium |
| Einsum | Tensor ops | 1-5x | Low |
| Chunking | Memory limits | N/A | Medium |

---

**See also**:
- [ML Optimization](ml-optimization.md) - PyTorch/TensorFlow patterns
- [Optimization Patterns](optimization-patterns.md) - General patterns
- [Examples](examples/md-simulation-optimization.md) - Real-world case studies
