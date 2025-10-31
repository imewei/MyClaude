---
name: gpu-acceleration
description: Implement GPU acceleration for scientific computing using CUDA (NVIDIA) and ROCm (AMD) with framework integration and kernel optimization. Use this skill when offloading array computations to GPU with CuPy (Python) or CUDA.jl (Julia) for NumPy-like GPU operations, writing custom CUDA kernels using @cuda.jit decorators in Numba or CUDA.jl kernel macros for domain-specific algorithms, optimizing GPU memory management with pinned memory and asynchronous transfers for minimizing CPU-GPU communication overhead, implementing GPU-accelerated ODE/PDE solvers for large-scale numerical simulations, parallelizing matrix operations (matrix multiplication, SVD, eigenvalue decomposition) on GPU for linear algebra computations, managing multi-GPU workflows for distributed GPU computing across multiple devices, profiling GPU performance with NVIDIA Nsight Systems or Nsight Compute to identify bottlenecks and optimize kernel occupancy, designing hybrid CPU-GPU pipelines that overlap computation with data transfer using CUDA streams, implementing GPU-accelerated Monte Carlo simulations or particle-based methods with massive parallelism, optimizing memory coalescing and shared memory usage in custom CUDA kernels for performance, working with .cu CUDA source files or GPU-enabled Python/Julia scripts, configuring GPU resources in HPC job schedulers (SLURM --gres=gpu), or migrating CPU-bound numerical code to GPU for 10-100x speedups in data-parallel workloads.
---

# GPU Acceleration

## When to use this skill

- When implementing GPU-accelerated array operations using CuPy or CUDA.jl
- When writing custom CUDA kernels with Numba's @cuda.jit or CUDA.jl kernel syntax
- When optimizing GPU memory transfers and managing pinned memory for performance
- When parallelizing large-scale matrix operations (matmul, SVD, eigensolvers) on GPU
- When implementing GPU-accelerated ODE/PDE solvers for scientific simulations
- When working with .cu CUDA source files or GPU-enabled Python/Julia scripts
- When managing multi-GPU computations across multiple devices
- When profiling GPU code with NVIDIA Nsight Systems or Nsight Compute
- When designing hybrid CPU-GPU workflows with asynchronous data transfers
- When optimizing CUDA kernel parameters (block size, grid size, shared memory)
- When implementing GPU-accelerated Monte Carlo methods or particle simulations
- When configuring GPU resources in SLURM job scripts (--gres=gpu:4)
- When migrating CPU NumPy code to GPU CuPy for performance improvements
- When ensuring memory coalescing and minimizing divergent branches in GPU kernels
- When benchmarking CPU versus GPU performance for specific numerical algorithms
- When working with GPU-enabled scientific libraries (cuBLAS, cuFFT, cuSolver)

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
