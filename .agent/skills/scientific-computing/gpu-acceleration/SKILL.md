---
name: gpu-acceleration
version: "1.0.7"
description: Implement GPU acceleration using CUDA/CuPy (Python) and CUDA.jl (Julia) with kernel optimization and memory management. Use when offloading computations to GPU, writing custom kernels, or optimizing multi-GPU workflows.
---

# GPU Acceleration

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
