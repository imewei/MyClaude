---
name: julia-hpc-distributed
description: Scale Julia computations across clusters with Distributed.jl, MPI.jl, and SLURM job management. Covers multi-node data parallelism, AllReduce for gradient aggregation, pmap/remotecall patterns, Dagger.jl task DAGs, and SLURM batch scripting for HPC facilities. Use when scaling Julia beyond a single node.
---

# Julia HPC & Distributed Computing

## Expert Agent

For scaling Julia across nodes and HPC clusters, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for Distributed.jl, MPI.jl, SLURM job management, and multi-node scaling.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`

## Distributed.jl Fundamentals

### Adding Workers

```julia
using Distributed

# Local workers
addprocs(4)

# Remote workers via SSH
addprocs([("node1", 4), ("node2", 4)];
         exeflags="--project=@.",
         tunnel=true)

nworkers()  # 8
```

### @everywhere and pmap

```julia
@everywhere using LinearAlgebra

# Parallel map -- auto load-balanced across workers
results = pmap(1:100) do i
    eigvals(randn(100, 100))
end
```

### @distributed Reduction

```julia
total = @distributed (+) for i in 1:1_000_000
    expensive_computation(i)
end
```

### remotecall / fetch / @spawnat

```julia
# Explicit remote execution
f = remotecall(rand, 2, 100, 100)   # Run on worker 2
result = fetch(f)                     # Block until ready

# Spawn on specific worker
r = @spawnat 3 begin
    A = randn(1000, 1000)
    svdvals(A)
end
fetch(r)
```

## RemoteChannel Producer-Consumer

```julia
const jobs    = RemoteChannel(() -> Channel{Int}(128))
const results = RemoteChannel(() -> Channel{Tuple{Int,Float64}}(128))

@everywhere function worker_loop(jobs, results)
    while true
        id = take!(jobs)
        id == -1 && break
        result = expensive_computation(id)
        put!(results, (id, result))
    end
end

# Start workers
for w in workers()
    remote_do(worker_loop, w, jobs, results)
end

# Feed jobs
for i in 1:1000
    put!(jobs, i)
end
for _ in workers()
    put!(jobs, -1)  # Poison pill
end
```

## SharedArrays

```julia
using SharedArrays

S = SharedArray{Float64}(1000, 1000)

@distributed for i in 1:1000
    for j in 1:1000
        S[i, j] = compute(i, j)
    end
end
```

## MPI.jl

### Basic Operations

```julia
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Broadcast from root
data = rank == 0 ? rand(Float64, 1000) : Vector{Float64}(undef, 1000)
MPI.Bcast!(data, 0, comm)

# Scatter / Gather
sendbuf = rank == 0 ? rand(Float32, 100 * nprocs) : nothing
recvbuf = Vector{Float32}(undef, 100)
MPI.Scatter!(sendbuf, recvbuf, 0, comm)

gathered = MPI.Gather(recvbuf, 0, comm)

# AllReduce
local_sum = [sum(recvbuf)]
global_sum = similar(local_sum)
MPI.Allreduce!(local_sum, global_sum, +, comm)
```

## Distributed Gradient Aggregation

Synchronize gradients across ranks using MPI AllReduce:

```julia
using MPI, Lux, Zygote, Optimisers

function distributed_train_step!(model, ps, st, opt_state, x_local, y_local, comm)
    nranks = MPI.Comm_size(comm)

    # Local forward + backward
    (loss, st), grads = Zygote.withgradient(ps) do p
        y_pred, st_ = model(x_local, p, st)
        sum(abs2, y_pred .- y_local), st_
    end

    # AllReduce gradients (average across ranks)
    for g in Functors.fleaves(grads)
        MPI.Allreduce!(g, +, comm)
        g ./= nranks
    end

    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
    return ps, st, opt_state, loss
end
```

## SLURM Batch Scripts

### CPU Job

```bash
#!/bin/bash
#SBATCH --job-name=julia-dist
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute

module load julia/1.11

srun julia --project=@. distributed_script.jl
```

### GPU-Aware Job

```bash
#!/bin/bash
#SBATCH --job-name=julia-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu

module load julia/1.11 cuda/12.4

export JULIA_CUDA_MEMORY_POOL=none  # Let SLURM manage GPU memory
srun julia --project=@. gpu_training.jl
```

## ClusterManagers.jl

Launch Julia workers via SLURM or PBS:

```julia
using ClusterManagers

# SLURM -- request 32 workers
addprocs(SlurmManager(32);
         time="02:00:00",
         partition="compute",
         exeflags="--project=@.")

# PBS
addprocs(PBSManager(16);
         queue="batch",
         exeflags="--project=@.")
```

## Dagger.jl Task DAGs

Express computation as a directed acyclic graph:

```julia
using Dagger

a = Dagger.@spawn load_data("input_a.h5")
b = Dagger.@spawn load_data("input_b.h5")
c = Dagger.@spawn process(a, b)          # Depends on a, b
d = Dagger.@spawn summarize(c)           # Depends on c

result = fetch(d)                          # Triggers full DAG execution
```

## Decision Tree: Which Framework?

| Scenario | Framework | Why |
|----------|-----------|-----|
| Embarrassingly parallel | `pmap` / `@distributed` | Simple, built-in |
| Producer-consumer pipeline | `RemoteChannel` | Decoupled, back-pressure |
| Shared-memory multiprocessing | `SharedArrays` | Zero-copy on single node |
| Tightly-coupled numerical (stencils, PDE) | `MPI.jl` | Low-latency collectives |
| GPU gradient sync | `MPI.jl` + `NCCL.jl` | Hardware-accelerated AllReduce |
| Complex task DAGs | `Dagger.jl` | Auto-scheduling, fault tolerance |

## Performance Tips

| Tip | Details |
|-----|---------|
| Minimize data transfer | Send only what workers need, not entire datasets |
| Use `@everywhere` sparingly | Load code once at startup, not per iteration |
| Prefer `pmap` over `@distributed` | Better load balancing for uneven workloads |
| Pin MPI ranks to cores | `--bind-to core` or SLURM `--cpu-bind=cores` |
| Profile communication | `MPI.Wtick()` / `MPI.Wtime()` to measure overhead |

## Checklist

- [ ] Start with `pmap` before reaching for MPI
- [ ] Use `@everywhere` to load packages on all workers
- [ ] Set `--project=@.` in `exeflags` for reproducible environments
- [ ] Test with 2 workers locally before submitting SLURM jobs
- [ ] Use `MPI.Allreduce!` (in-place) to minimize allocations
- [ ] Pin processes to cores for NUMA-aware execution
- [ ] Monitor with `SLURM sacct` and Julia `@elapsed`
