---
name: parallel-computing
description: Multi-threading, Distributed.jl, and GPU computing with CUDA.jl for parallel scientific computing. Use for ensemble simulations and large-scale computations.
---

# Parallel Computing

Master parallel computing in Julia with threads, distributed processing, and GPUs.

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
