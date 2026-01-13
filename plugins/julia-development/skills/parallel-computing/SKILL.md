---
name: parallel-computing
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Parallelism
description: Master multi-threading, Distributed.jl, and GPU computing with CUDA.jl. Use when scaling computations across CPUs or GPUs for scientific computing.
---

# Julia Parallel Computing

Multi-threading, distributed, and GPU parallelism.

---

## Multi-Threading

```julia
using DifferentialEquations

# Set threads: export JULIA_NUM_THREADS=8
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=1000)
```

---

## GPU Computing

```julia
using CUDA, DiffEqGPU

prob_gpu = remake(prob, u0=CuArray(u0))
sol_gpu = solve(prob_gpu, Tsit5(), EnsembleGPUArray(), trajectories=10000)
```

---

## Distributed

```julia
using Distributed
addprocs(4)
@everywhere using DifferentialEquations

sol = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories=1000)
```

---

## Selection Guide

| Method | Use Case |
|--------|----------|
| Threads | Shared memory, single machine |
| Distributed | Multi-machine, large scale |
| GPU | Data-parallel, massive parallelism |

---

## Checklist

- [ ] Parallelism type selected
- [ ] Thread count configured
- [ ] Data properly distributed
- [ ] Performance benchmarked

---

**Version**: 1.0.5
