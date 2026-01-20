---
name: parallel-computing
version: "1.0.1"
description: Implement high-performance parallel computing across CPUs and GPUs using Python (CUDA/CuPy) and Julia (CUDA.jl/Distributed.jl). Master multi-threading, distributed systems, and kernel optimization.
---

# Parallel Computing Suite

Comprehensive guide for scaling scientific computations across multiple cores, machines, and GPUs.

## Expert Agent

For high-performance computing, GPU optimization, and distributed systems, delegate to the expert agent:

- **`jax-pro`** (for Python/JAX):
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Multi-device parallelism (`pmap`), sharding, and TPU optimization.
- **`julia-pro`** (for Julia):
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: Native GPU arrays (`CUDA.jl`), distributed computing (`Distributed.jl`), and multi-threading.

## Core Skills

### [Ecosystem Selection](./ecosystem-selection/SKILL.md)
Selection criteria for parallel computing frameworks (Python vs Julia).

### [GPU Acceleration](./gpu-acceleration/SKILL.md)
Native GPU computing using CUDA, CuPy, and CUDA.jl.

### [Numerical Methods Implementation](./numerical-methods-implementation/SKILL.md)
High-performance solvers for ODEs, PDEs, and linear systems.

### [Parallel Computing Strategy](./parallel-computing-strategy/SKILL.md)
Architecture patterns for multi-threading and distributed computing.

## 1. GPU Acceleration (Cross-Platform)

### Framework Selection

| Framework | Language | Primary Use Case |
|-----------|----------|------------------|
| **CuPy** | Python | NumPy-like GPU arrays & linear algebra |
| **CUDA.jl** | Julia | Native GPU arrays with high-level syntax |
| **Numba CUDA**| Python | Custom kernels for specialized logic |
| **DiffEqGPU** | Julia | Parallelized ODE/PDE solving on GPUs |

### Implementation Examples

#### Python (CuPy & Numba)
```python
import cupy as cp
from numba import cuda

# High-level array operations
x_gpu = cp.random.random((10000, 10000))
y_gpu = cp.matmul(x_gpu, x_gpu.T)

# Custom CUDA Kernel
@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = x[idx] + y[idx]
```

#### Julia (CUDA.jl)
```julia
using CUDA, DiffEqGPU

# Native GPU arrays
x_gpu = CUDA.rand(10000, 10000)
y_gpu = x_gpu * x_gpu'

# Parallel ODE Solving
sol_gpu = solve(prob_gpu, Tsit5(), EnsembleGPUArray(), trajectories=10000)
```

## 2. Multi-Threading & Distributed Computing

### Shared Memory (Multi-Threading)
- **Python**: Use `concurrent.futures`, `multiprocessing`, or Numba's `@njit(parallel=True)`.
- **Julia**: Use `Threads.@threads` or `EnsembleThreads()` in SciML. Ensure `JULIA_NUM_THREADS` is set.

### Distributed Memory (Multi-Machine)
- **Python**: Use `Dask`, `Ray`, or `mpi4py`.
- **Julia**: Use `Distributed.jl`. Add processes via `addprocs(n)` and use `@everywhere` for loading dependencies.

## 3. Performance & Optimization Checklist

- [ ] **Memory Management**: Minimize host-to-device transfers. Use pinned memory for faster I/O.
- [ ] **Coalescing**: Ensure threads access contiguous memory locations.
- [ ] **Occupancy**: Optimize threads/block (typically 128-512) to maximize hardware utilization.
- [ ] **Streams**: Use asynchronous streams to overlap computation with data transfers.

## 4. Profiling Tools

- **NSight Systems**: `nsys profile --stats=true python/julia script.py`
- **NSight Compute**: `ncu --set full ...` for detailed kernel analysis.
- **Julia Profiler**: Use `Profile` and `PProf.jl` for CPU/Memory profiling.
