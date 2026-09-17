---
name: gpu-acceleration
description: Implement GPU acceleration using CUDA/CuPy (Python) and CUDA.jl (Julia) with kernel optimization and memory management. Use when offloading computations to GPU, writing custom kernels, or optimizing multi-GPU workflows.
---

# GPU Acceleration

## Expert Agent

For GPU optimization strategies and kernel implementation, delegate to:

- **`jax-pro`**: For JAX-based GPU acceleration, sharding, and Pallas kernels.
- **`julia-pro`**: For CUDA.jl, KernelAbstractions.jl, and Julia GPU kernels.
- **`julia-ml-hpc`**: For advanced Julia GPU kernels, KernelAbstractions.jl, and multi-GPU with NCCL.jl.
  - *Julia skill*: See `julia-gpu-kernels` for detailed Julia GPU programming.

## Framework Selection

| Framework | Language | Use |
|-----------|----------|-----|
| CuPy | Python | NumPy-like GPU arrays |
| Numba CUDA | Python | Custom kernels |
| CUDA.jl | Julia | Native GPU arrays |
| cuBLAS/cuFFT | C/C++ | Optimized linear algebra |

## CuPy

```python
import cupy as cp
x_gpu = cp.random.random((10000, 10000))
y_gpu = cp.matmul(x_gpu, x_gpu.T)
result = cp.asnumpy(y_gpu)  # Back to CPU
```

## Custom Kernels

```python
from numba import cuda
@cuda.jit
def gpu_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < out.size: out[idx] = x[idx] + y[idx]
# Launch: gpu_kernel[blocks, threads](x, y, out)
```

## CUDA.jl

```julia
using CUDA
x_gpu = CUDA.rand(10000, 10000)
y_gpu = x_gpu * x_gpu'
result = Array(y_gpu)
```

## Memory Optimization

| Technique | Benefit |
|-----------|---------|
| Coalesced access | Consecutive memory |
| Shared memory | On-chip cache |
| Pinned memory | Faster transfers |
| Async transfers | Overlap compute |

## Parallelization Patterns

| Pattern | Implementation | Use Case |
|---------|----------------|----------|
| **Grid-Stride Loop** | CUDA Kernel | Arbitrary input sizes |
| **Reduction** | Tree-based | Sum/Max of arrays |
| **Stream Processing** | Async CUDA Streams | Overlap copy & compute |
| **Multi-GPU** | NCCL / NVLink | Distributed training/simulation |

```python
stream1 = cp.cuda.Stream()
with stream1:
    gpu_data = cp.asarray(chunk)
    result = process(gpu_data)
```

## Kernel Optimization

- Threads/block: 128-512
- Minimize divergence
- Loop unrolling
- Maximize occupancy

## Multi-GPU

```python
for gpu_id in range(cp.cuda.runtime.getDeviceCount()):
    with cp.cuda.Device(gpu_id):
        result = process_on_gpu(data_chunk[gpu_id])
```

## Profiling

```bash
nsys profile --stats=true python code.py
ncu --set full python code.py
```

## When to Use GPU

**Use**: Large data-parallel (>10^6), dense linear algebra, Monte Carlo
**Avoid**: Small problems (transfer overhead), sequential algorithms

**Outcome**: Maximize occupancy, coalescing, minimize transfers, overlap compute

## Checklist

- [ ] Verify problem size exceeds GPU transfer overhead threshold (>10^6 elements for dense operations)
- [ ] Confirm memory access patterns are coalesced (consecutive threads access consecutive memory)
- [ ] Check that shared memory is used for frequently accessed data within thread blocks
- [ ] Validate thread block size is between 128-512 for optimal occupancy
- [ ] Ensure host-to-device transfers use pinned memory for maximum bandwidth
- [ ] Overlap compute and data transfer using CUDA streams or async operations
- [ ] Profile with `nsys` or `ncu` to identify bottlenecks before manual optimization
- [ ] Verify multi-GPU workload distribution is balanced across devices
- [ ] Check that kernel launches avoid warp divergence in conditional branches
