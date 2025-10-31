---
name: parallel-computing
description: Master multi-threading, Distributed.jl, and GPU computing with CUDA.jl for parallel scientific computing and high-performance workloads. Use when scaling computations with Threads.@threads for multi-core CPU parallelism, implementing distributed computing with Distributed.jl and @distributed macros, running ensemble simulations with EnsembleThreads or EnsembleDistributed, accelerating array operations on GPUs with CUDA.jl and CuArray, working with DiffEqGPU for GPU-accelerated differential equations, using pmap for parallel map operations, managing worker processes with addprocs, or optimizing parallel performance. Essential for large-scale scientific computing, Monte Carlo simulations, and computationally intensive workloads requiring parallelization.
---

# Parallel Computing

Master parallel computing in Julia with threads, distributed processing, and GPUs.

## When to use this skill

- Scaling computations with multi-threading (Threads.@threads, @spawn)
- Implementing distributed computing across multiple processes (Distributed.jl)
- Running ensemble simulations (EnsembleThreads, EnsembleDistributed, EnsembleGPUArray)
- Accelerating array operations on GPUs (CUDA.jl, CuArray, GPU kernels)
- Using DiffEqGPU for GPU-accelerated differential equations
- Parallel map operations with pmap and @distributed
- Managing worker processes with addprocs and @everywhere
- Sharing data across workers with SharedArrays or DistributedArrays
- Optimizing parallel performance and load balancing
- Choosing between threading vs distributed vs GPU parallelism
- Running Monte Carlo simulations in parallel

## Multi-Threading
```julia
using DifferentialEquations

# Set threads: export JULIA_NUM_THREADS=8
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=1000)
```

## GPU Computing
```julia
using CUDA, DiffEqGPU

prob_gpu = remake(prob, u0=CuArray(u0))
sol_gpu = solve(prob_gpu, Tsit5(), EnsembleGPUArray(), trajectories=10000)
```

## Distributed
```julia
using Distributed
addprocs(4)
@everywhere using DifferentialEquations

sol = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories=1000)
```

## Resources
- **Parallel Computing**: https://docs.julialang.org/en/v1/manual/parallel-computing/
