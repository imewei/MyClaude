---
name: gpu-acceleration
version: "1.0.6"
maturity: "5-Expert"
specialization: GPU Computing
description: Implement GPU acceleration using CUDA/CuPy (Python) and CUDA.jl (Julia) with kernel optimization and memory management. Use when offloading computations to GPU, writing custom kernels, or optimizing multi-GPU workflows.
---

# GPU Acceleration

GPU acceleration for scientific computing with CUDA, CuPy, and CUDA.jl.

---

## Framework Selection

| Framework | Language | Use Case |
|-----------|----------|----------|
| CuPy | Python | NumPy-like GPU arrays |
| Numba CUDA | Python | Custom kernels |
| CUDA.jl | Julia | Native GPU arrays |
| cuBLAS/cuFFT | C/C++ | Optimized linear algebra |

---

## CuPy (Python)

```python
import cupy as cp

x_gpu = cp.random.random((10000, 10000))
y_gpu = cp.matmul(x_gpu, x_gpu.T)  # GPU matmul
result = cp.asnumpy(y_gpu)  # Transfer to CPU
```

---

## Custom CUDA Kernels (Numba)

```python
from numba import cuda

@cuda.jit
def gpu_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = x[idx] + y[idx]

# Launch: gpu_kernel[blocks, threads](x, y, out)
```

---

## CUDA.jl (Julia)

```julia
using CUDA

x_gpu = CUDA.rand(10000, 10000)
y_gpu = x_gpu * x_gpu'  # GPU matmul
result = Array(y_gpu)  # Transfer to CPU
```

---

## Memory Optimization

| Technique | Benefit |
|-----------|---------|
| Coalesced access | Threads access consecutive memory |
| Shared memory | Fast on-chip cache |
| Pinned memory | Faster CPU-GPU transfers |
| Async transfers | Overlap compute + transfer |

```python
# Async streams
stream1 = cp.cuda.Stream()
with stream1:
    gpu_data = cp.asarray(chunk)
    result = process(gpu_data)
```

---

## Kernel Optimization

| Parameter | Guideline |
|-----------|-----------|
| Threads/block | 128-512 for occupancy |
| Divergence | Minimize branching |
| Loop unrolling | Manual for performance |
| Occupancy | Maximize threads per SM |

---

## Multi-GPU

```python
num_gpus = cp.cuda.runtime.getDeviceCount()
for gpu_id in range(num_gpus):
    with cp.cuda.Device(gpu_id):
        result = process_on_gpu(data_chunk[gpu_id])
```

---

## Profiling

```bash
nsys profile --stats=true python code.py  # System-level
ncu --set full python code.py             # Kernel-level
```

---

## When to Use GPU

| Use GPU | Avoid GPU |
|---------|-----------|
| Large data-parallel (>10^6 elements) | Small problems (transfer overhead) |
| Dense linear algebra | Sequential algorithms |
| Monte Carlo, PDE solvers | Memory-bound I/O |

---

## Checklist

- [ ] Maximize occupancy (threads per SM)
- [ ] Ensure memory coalescing
- [ ] Minimize host-device transfers
- [ ] Use pinned memory for transfers
- [ ] Overlap computation/communication
- [ ] Profile with Nsight

---

**Version**: 1.0.5
