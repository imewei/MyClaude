---
name: benchmark
description: Profile JAX/Julia/HPC code — wall time, memory, GPU utilization, JIT compilation overhead — and suggest optimizations.
argument-hint: "[--target path/to/script] [--backend jax|julia|cuda] [--profile memory|time|both]"
allowed-tools: ["Read", "Bash", "Glob"]
---

# /benchmark — Scientific Code Benchmarking

Routes to `jax-pro` (JAX/CUDA), `julia-pro` (Julia), or `systems-engineer` (C/Fortran/HPC) based on `--backend`.

## Usage

```
/benchmark --target src/train.py --backend jax --profile both
/benchmark --target scripts/simulate.jl --backend julia --profile time
/benchmark --target src/md_kernel.cu --backend cuda --profile memory
```

## What This Does

1. Reads `--target` file and identifies hot paths
2. Runs profiling appropriate for `--backend`
3. Reports wall time, peak memory, and (for JAX) JIT compile overhead vs runtime
4. Suggests targeted optimizations (vmap, pmap, type stability, allocation reduction)

## Backend Routing

| `--backend` | Routes To | Tool |
|---|---|---|
| `jax` | jax-pro | `jax.profiler`, `jax.make_jaxpr`, nvtx |
| `julia` | julia-pro | `@btime`, `@profile`, `Cthulhu.jl` |
| `cuda` | systems-engineer | `nvprof`, `Nsight Compute` |

## Token Strategy

Profiling templates and tool flags load only when `--profile` is specified. Omitting `--profile` uses `time` as default.
