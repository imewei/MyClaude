# Performance Optimization Techniques

**Version**: 1.0.3
**Category**: code-migration
**Purpose**: Systematic approaches to parallelization, computational efficiency, and hardware utilization

## Parallelization Opportunities

### SIMD Vectorization

**NumPy Broadcasting**:
```python
# Scalar loop (slow)
for i in range(N):
    c[i] = a[i] * b[i] + d[i]

# Vectorized (10-100x faster)
c = a * b + d
```

**JAX vmap** (automatic vectorization):
```python
import jax
import jax.numpy as jnp

def process_single(x):
    return jnp.dot(x, x.T)

# Vectorize over batch dimension
process_batch = jax.vmap(process_single)

batch = jnp.ones((1000, 100))
result = process_batch(batch)  # Parallel SIMD execution
```

---

### Multi-Threading (Data Parallelism)

**NumPy Multi-Core**:
```python
import numpy as np
import os

# Enable multi-threading
os.environ['OMP_NUM_THREADS'] = '8'

# Large matrix operations use all cores
A = np.random.rand(5000, 5000)
B = np.random.rand(5000, 5000)
C = A @ B  # Parallel BLAS (OpenBLAS/MKL)
```

**Julia @threads**:
```julia
using Base.Threads

function parallel_sum(arr)
    partial_sums = zeros(nthreads())

    @threads for i in 1:length(arr)
        tid = threadid()
        partial_sums[tid] += arr[i]
    end

    return sum(partial_sums)
end
```

---

### GPU Acceleration

**JAX GPU Patterns**:
```python
import jax
import jax.numpy as jnp

@jax.jit  # JIT compile for GPU
def gpu_kernel(x):
    return jnp.fft.fft(x) * jnp.conj(jnp.fft.fft(x))

# Runs on GPU if available
x = jnp.ones((1_000_000,))
result = gpu_kernel(x)
```

**Expected Speedups**:
- Embarrassingly parallel: 100-1000x
- Memory-bound operations: 10-50x
- Complex algorithms: 5-20x

---

### Distributed Computing

**Dask Parallel Arrays**:
```python
import dask.array as da

# Distributed array across cluster
x = da.random.random((100000, 100000), chunks=(1000, 1000))
y = da.sum(x, axis=0)

# Compute triggers distributed execution
result = y.compute()
```

**JAX pmap** (multi-GPU):
```python
@jax.pmap  # Parallelize over devices
def multi_gpu_function(x):
    return x ** 2

# Data sharded across 8 GPUs
x = jnp.ones((8, 1000, 1000))
result = multi_gpu_function(x)
```

---

## Computational Efficiency

### Algorithm Complexity Reduction

**O(N²) → O(N log N)** via FFT:
```python
# Convolution: O(N²) naive
def slow_convolve(signal, kernel):
    N = len(signal)
    result = np.zeros(N)
    for i in range(N):
        for j in range(len(kernel)):
            if i-j >= 0:
                result[i] += signal[i-j] * kernel[j]
    return result

# FFT-based: O(N log N)
def fast_convolve(signal, kernel):
    return np.fft.ifft(np.fft.fft(signal) * np.fft.fft(kernel, len(signal))).real
```

---

### Cache Optimization

**Loop Blocking/Tiling**:
```python
def blocked_matmul(A, B, block_size=64):
    """Cache-friendly matrix multiplication"""
    N = A.shape[0]
    C = np.zeros((N, N))

    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            for k in range(0, N, block_size):
                # Process block (fits in cache)
                C[i:i+block_size, j:j+block_size] += \
                    A[i:i+block_size, k:k+block_size] @ B[k:k+block_size, j:j+block_size]

    return C
```

---

### Memory Allocation Optimization

**Pre-Allocation**:
```python
# Bad: Repeated allocation
result = []
for i in range(N):
    result.append(compute(i))  # Reallocates each time

# Good: Pre-allocate
result = np.empty(N)
for i in range(N):
    result[i] = compute(i)  # No allocation
```

**In-Place Operations**:
```python
# Allocates new array
result = array + 1

# In-place (no allocation)
array += 1
```

---

### JIT Compilation

**JAX JIT**:
```python
import jax

@jax.jit
def optimized_function(x):
    for _ in range(100):
        x = jnp.sin(x) + jnp.cos(x)
    return x

# First call: compile (slow)
result = optimized_function(x)

# Subsequent calls: use compiled version (fast)
result = optimized_function(x)  # 10-100x faster
```

**Numba JIT** (Python):
```python
from numba import jit

@jit(nopython=True)
def fast_loop(x):
    total = 0.0
    for val in x:
        total += val ** 2
    return total
```

---

## Hardware Utilization

### CPU Architecture Optimization

**AVX-512 SIMD** (modern Intel CPUs):
- 512-bit vectors (8 x float64 or 16 x float32)
- Automatic via NumPy/JAX compiled code
- Manual via intrinsics (advanced)

**ARM NEON** (Apple Silicon, mobile):
- 128-bit vectors (2 x float64 or 4 x float32)
- Excellent for mobile/edge deployment

---

### GPU Utilization Strategy

**Kernel Candidates** (good for GPU):
- Large parallel loops (1M+ iterations)
- Matrix operations (BLAS level 2/3)
- FFT, convolution
- Element-wise operations

**Poor GPU Candidates**:
- Small problem sizes (<10K elements)
- Heavy branching/conditionals
- Sequential algorithms
- CPU-GPU data transfer overhead

---

### Memory Hierarchy Optimization

**Bandwidth vs. Latency**:
- **L1 Cache**: 1-4 cycles, ~32 KB
- **L2 Cache**: 10-20 cycles, ~256 KB
- **L3 Cache**: 40-75 cycles, ~8 MB
- **RAM**: 200+ cycles, ~16 GB
- **GPU VRAM**: 400+ cycles, ~24 GB

**Optimization Strategy**:
1. Maximize data reuse (cache locality)
2. Minimize memory transfers
3. Use contiguous arrays
4. Align data to cache lines (64 bytes)

---

## Profiling and Benchmarking

### CPU Profiling

**cProfile**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

result = expensive_computation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

---

### GPU Profiling

**JAX Profiling**:
```python
import jax.profiler

# Profile GPU kernels
jax.profiler.start_trace("/tmp/tensorboard")
result = gpu_function(x).block_until_ready()
jax.profiler.stop_trace()

# View in TensorBoard
# tensorboard --logdir=/tmp/tensorboard
```

---

### Benchmarking Best Practices

1. **Warm-up runs**: JIT compilation overhead
2. **Multiple iterations**: Statistical significance
3. **Block until ready**: GPU async execution
4. **Same hardware**: Fair comparison
5. **Document conditions**: CPU freq, GPU model, library versions

---

## Performance Targets

| Optimization | Expected Speedup |
|--------------|------------------|
| Vectorization (NumPy) | 10-100x vs. Python loops |
| JIT Compilation (JAX/Numba) | 10-50x vs. interpreted |
| GPU Acceleration (JAX/CUDA) | 10-1000x vs. CPU |
| Multi-core (OpenMP/threads) | 4-8x on 8-core CPU |
| Distributed (Dask/MPI) | 10-100x on cluster |

---

## References

- JAX Performance Guide: https://jax.readthedocs.io/en/latest/faq.html#performance
- NumPy Performance: https://numpy.org/doc/stable/user/c-info.beyond-basics.html
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
