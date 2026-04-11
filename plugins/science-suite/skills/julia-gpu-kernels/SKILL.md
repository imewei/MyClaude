---
name: julia-gpu-kernels
description: Write high-performance GPU code in Julia with CUDA.jl and KernelAbstractions.jl. Covers custom kernel writing, shared memory optimization, multi-GPU data parallelism, memory management (unified/pinned), profiling with NVTX.jl, and portable kernels across CUDA/ROCm/oneAPI/Metal backends. Use when writing custom GPU kernels or optimizing GPU performance in Julia.
---

# Julia GPU Kernels

## Expert Agent

For GPU kernel development and optimization in Julia, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for CUDA.jl, KernelAbstractions.jl, custom GPU kernels, and multi-GPU patterns.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`

## CUDA.jl Basics

CuArray broadcasting and linear algebra work out of the box:

```julia
using CUDA

# Array operations on GPU
x = CUDA.rand(Float32, 10_000, 10_000)
y = x * x'                    # cuBLAS gemm
z = x .+ sin.(x)              # Fused broadcast kernel
result = Array(z)              # Transfer back to CPU
```

Linear algebra dispatches to cuBLAS/cuSOLVER automatically:

```julia
using LinearAlgebra
F = lu(x)                     # cuSOLVER LU
vals = svdvals(x)             # cuSOLVER SVD
```

Move a Lux model to GPU:

```julia
using Lux, CUDA

model = Chain(Dense(784, 256, relu), Dense(256, 10))
ps, st = Lux.setup(Random.default_rng(), model)

dev = gpu_device()
ps_gpu = ps |> dev
st_gpu = st |> dev
x_gpu  = CUDA.rand(Float32, 784, 64)

y_gpu, st_gpu = model(x_gpu, ps_gpu, st_gpu)
```

## Custom CUDA Kernel

Write a custom kernel using `@cuda`:

```julia
function vector_add_kernel!(C, A, B, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds C[i] = A[i] + B[i]
    end
    return nothing
end

N = 1_000_000
A = CUDA.rand(Float32, N)
B = CUDA.rand(Float32, N)
C = similar(A)

threads = 256
blocks  = cld(N, threads)
@cuda threads=threads blocks=blocks vector_add_kernel!(C, A, B, N)
synchronize()
```

## Grid-Stride Loop Pattern

Handle arbitrary input sizes with a single kernel launch:

```julia
function grid_stride_kernel!(out, x, N)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while idx <= N
        @inbounds out[idx] = x[idx]^2 + 2 * x[idx] + 1
        idx += stride
    end
    return nothing
end
```

## KernelAbstractions.jl Portable Kernels

Write backend-agnostic kernels:

```julia
using KernelAbstractions

@kernel function ka_vector_add!(C, A, B)
    i = @index(Global)
    @inbounds C[i] = A[i] + B[i]
end

backend = get_backend(A)          # Auto-detect: CUDABackend, ROCBackend, etc.
kernel! = ka_vector_add!(backend, 256)
kernel!(C, A, B; ndrange=length(A))
synchronize(backend)
```

### Backend Portability

| Backend | Package | Hardware |
|---------|---------|----------|
| `CUDABackend` | CUDA.jl | NVIDIA GPUs |
| `ROCBackend` | AMDGPU.jl | AMD GPUs |
| `oneAPIBackend` | oneAPI.jl | Intel GPUs |
| `MetalBackend` | Metal.jl | Apple Silicon |
| `CPU` | KernelAbstractions.jl | Any CPU |

## Shared Memory Reduction Kernel

Use shared memory for efficient parallel reductions:

```julia
function reduce_sum_kernel!(output, input, N)
    shmem = @cuStaticSharedMem(Float32, 256)

    tid = threadIdx().x
    gid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # Load to shared memory
    shmem[tid] = gid <= N ? input[gid] : 0f0
    sync_threads()

    # Tree reduction in shared memory
    s = blockDim().x >> 1
    while s > 0
        if tid <= s
            shmem[tid] += shmem[tid + s]
        end
        sync_threads()
        s >>= 1
    end

    # Write block result
    if tid == 1
        output[blockIdx().x] = shmem[1]
    end
    return nothing
end
```

## Memory Management

| Type | API | Latency | Use Case |
|------|-----|---------|----------|
| Device | `CuArray(...)` | Lowest on GPU | Default GPU computation |
| Unified | `cu(x; unified=true)` | Auto-migrate | Mixed CPU/GPU access |
| Pinned | `CUDA.Mem.pin(x)` | Fast H2D/D2H | Staging for async transfers |
| Async | Stream-ordered | Overlapped | Pipeline copy + compute |

Stream-ordered memory operations:

```julia
s1 = CuStream()
s2 = CuStream()

# Overlap data transfer and computation
CUDA.@sync begin
    @async begin
        copyto!(d_a, h_a; stream=s1)
        @cuda stream=s1 threads=256 blocks=blocks compute_kernel!(d_a, N)
    end
    @async begin
        copyto!(d_b, h_b; stream=s2)
        @cuda stream=s2 threads=256 blocks=blocks compute_kernel!(d_b, N)
    end
end
```

## Multi-GPU with NCCL.jl

Distribute work across multiple GPUs:

```julia
using CUDA, NCCL

ndevs = length(CUDA.devices())
comms = NCCL.Communicators(ndevs)

@sync for (i, dev) in enumerate(CUDA.devices())
    @async begin
        CUDA.device!(dev)
        sendbuf = CUDA.rand(Float32, 1024)
        recvbuf = similar(sendbuf)
        NCCL.Allreduce!(sendbuf, recvbuf, +, comms[i])
    end
end
```

## Profiling with NVTX.jl

Annotate code regions for profiling:

```julia
using NVTX

NVTX.@range "data_loading" begin
    data = load_data()
end

NVTX.@range "forward_pass" begin
    output = model(data)
end
```

Profile with NVIDIA tools:

```bash
# System-wide timeline
nsys profile --trace=cuda,nvtx julia script.jl

# Kernel-level metrics
ncu --set full --target-processes all julia script.jl

# Export for Nsight Systems GUI
nsys profile -o report.nsys-rep julia script.jl
```

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Scalar indexing `x[1]` | Triggers GPU-CPU sync | Use `CUDA.@allowscalar` only for debug |
| Allocating in kernel | Not supported | Pre-allocate all buffers |
| Non-coalesced access | Low bandwidth | Access consecutive memory per warp |
| Excessive `synchronize()` | Kills concurrency | Sync only at pipeline boundaries |
| Small kernel launches | Launch overhead dominates | Batch work, use grid-stride |
| `Float64` by default | Half the throughput | Use `Float32` explicitly |

## Occupancy Optimization

Calculate optimal launch configuration:

```julia
kernel = @cuda launch=false vector_add_kernel!(C, A, B, N)
config = launch_configuration(kernel.fun)

threads = min(N, config.threads)
blocks  = cld(N, threads)

kernel(C, A, B, N; threads=threads, blocks=blocks)
```

## Checklist

- [ ] Use `CuArray` broadcasting before writing custom kernels
- [ ] Profile with `CUDA.@time` and `CUDA.@profile` before optimizing
- [ ] Prefer `Float32` over `Float64` for throughput
- [ ] Use grid-stride loops for arbitrary input sizes
- [ ] Avoid scalar indexing on GPU arrays
- [ ] Use `KernelAbstractions.jl` for multi-backend portability
- [ ] Pin host memory for async transfer pipelines
- [ ] Annotate with `NVTX.@range` for profiler visibility
