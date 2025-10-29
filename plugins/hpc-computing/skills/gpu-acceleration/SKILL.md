---
name: gpu-acceleration
description: Implement GPU acceleration using CUDA/ROCm for NVIDIA/AMD GPUs. Use when offloading computations with CuPy/Numba (Python) or CUDA.jl (Julia), optimizing GPU kernels for matrix operations and PDE solvers, or managing hybrid CPU-GPU pipelines with memory optimization and multi-device orchestration.
---

# GPU Acceleration

## Overview

Implement GPU acceleration for scientific computing using CUDA (NVIDIA) and ROCm (AMD), covering framework integration, kernel optimization, memory management, and hybrid CPU-GPU workflows.

## Core GPU Programming

### CUDA/ROCm Kernels

**CUDA Example:**
```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}
```

**Memory Management:**
```cuda
float *d_A;
cudaMalloc(&d_A, N * sizeof(float));
cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
cudaFree(d_A);
```

### Framework Integration

**CuPy (Python/NVIDIA):**
```python
import cupy as cp

# GPU arrays (NumPy-like API)
x_gpu = cp.random.random((10000, 10000))
y_gpu = cp.matmul(x_gpu, x_gpu.T)  # Matrix multiply on GPU
result = cp.asnumpy(y_gpu)  # Transfer to CPU
```

**Numba (Python CUDA JIT):**
```python
from numba import cuda

@cuda.jit
def gpu_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = x[idx] + y[idx]

# Launch: gpu_kernel[blocks, threads](x, y, out)
```

**CUDA.jl (Julia):**
```julia
using CUDA

x_gpu = CUDA.rand(10000, 10000)
y_gpu = x_gpu * x_gpu'  # Matrix multiply on GPU
result = Array(y_gpu)  # Transfer to CPU
```

## Optimization Strategies

### Memory Optimization

**Coalescing**: Threads access consecutive memory
**Shared Memory**: Fast on-chip cache
**Pinned Memory**: Faster CPU-GPU transfers

```python
# Pinned memory for fast transfers
x_pinned = cp.cuda.alloc_pinned_memory(n * 4)
```

### Kernel Optimization

- **Occupancy**: 128-512 threads/block
- **Reduce Divergence**: Minimize branching
- **Loop Unrolling**: Manual unrolling for performance

### GPU-Accelerated Numerics

**GPU ODE Solver:**
```python
def gpu_runge_kutta_4(f, y0, t_span, dt):
    t = cp.arange(*t_span, dt)
    y = cp.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt/2 * k1)
        k3 = f(t[i] + dt/2, y[i] + dt/2 * k2)
        k4 = f(t[i] + dt, y[i] + dt * k3)
        y[i+1] = y[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y
```

**GPU PDE Solver:**
```python
@cuda.jit
def heat_equation_2d(u, u_new, dx, dt, alpha):
    i, j = cuda.grid(2)
    if 0 < i < u.shape[0] - 1 and 0 < j < u.shape[1] - 1:
        u_new[i,j] = u[i,j] + alpha * dt / (dx**2) * (
            u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]
        )
```

### Hybrid CPU-GPU Workflows

**Async Transfers:**
```python
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

with stream1:
    gpu_data1 = cp.asarray(chunk1)
    result1 = process(gpu_data1)

with stream2:
    gpu_data2 = cp.asarray(chunk2)
    result2 = process(gpu_data2)
```

**Multi-GPU:**
```python
num_gpus = cp.cuda.runtime.getDeviceCount()
for gpu_id in range(num_gpus):
    with cp.cuda.Device(gpu_id):
        result = process_on_gpu(data_chunk[gpu_id])
```

## Profiling & Best Practices

**Profile with Nsight:**
```bash
nsys profile --stats=true python code.py  # System-level
ncu --set full python code.py            # Kernel-level
```

**When to Use GPU:**
- ✅ Large data-parallel problems (>10⁶ elements)
- ✅ Dense linear algebra, Monte Carlo, PDE solvers
- ❌ Small problems (transfer overhead)
- ❌ Sequential algorithms with dependencies

**Optimization Checklist:**
- [ ] Maximize occupancy (threads per SM)
- [ ] Ensure memory coalescing
- [ ] Minimize host-device transfers
- [ ] Use pinned memory
- [ ] Overlap computation/communication
- [ ] Profile with Nsight

Load references for advanced optimization patterns and multi-GPU strategies.
