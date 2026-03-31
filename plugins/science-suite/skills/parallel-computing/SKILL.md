---
name: parallel-computing
description: Implement high-performance parallel computing across CPUs and GPUs using Python (CUDA/CuPy) and Julia (CUDA.jl/Distributed.jl). Design parallel strategies with MPI (distributed memory), OpenMP (shared memory), hybrid MPI+OpenMP, SLURM scheduling, Dask/Dagger.jl workflows, and load balancing. Master multi-threading, distributed systems, and kernel optimization.
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
- **`simulation-expert`**: For HPC scaling, MPI/OpenMP strategies, and job scheduling.

## Core Skills

### [Ecosystem Selection](./ecosystem-selection/SKILL.md)
Selection criteria for parallel computing frameworks (Python vs Julia).

### [GPU Acceleration](./gpu-acceleration/SKILL.md)
Native GPU computing using CUDA, CuPy, and CUDA.jl.

### [Numerical Methods Implementation](./numerical-methods-implementation/SKILL.md)
High-performance solvers for ODEs, PDEs, and linear systems.

## 1. Parallelism Types

| Type | Best For |
|------|----------|
| Data | Same operation, different data |
| Task | Independent operations |
| Pipeline | Sequential stages on different data |

## 2. GPU Acceleration (Cross-Platform)

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

## 3. MPI (Distributed Memory)

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
chunk = N // size
local_result = np.sum(np.arange(rank*chunk, (rank+1)*chunk) ** 2)
total = comm.reduce(local_result, op=MPI.SUM, root=0)
```

**Patterns**: `send/recv`, `broadcast`, `scatter`, `gather`, `reduce`, `allreduce`, `isend/irecv`

## 4. OpenMP (Shared Memory)

```c
#include <omp.h>
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < 1000000; i++) sum += i * i;

#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        { /* Task 1 */ }
        #pragma omp taskwait
    }
}
```

**Clauses**: `private/shared`, `reduction`, `schedule(dynamic)`, `num_threads`

## 5. Hybrid MPI+OpenMP

```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#pragma omp parallel for
for (int i = 0; i < 1000; i++) { /* Computation */ }
MPI_Finalize();
```

## 6. Multi-Threading & Distributed Computing

### Shared Memory (Multi-Threading)
- **Python**: Use `concurrent.futures`, `multiprocessing`, or Numba's `@njit(parallel=True)`.
- **Julia**: Use `Threads.@threads` or `EnsembleThreads()` in SciML. Ensure `JULIA_NUM_THREADS` is set.

### Distributed Memory (Multi-Machine)
- **Python**: Use `Dask`, `Ray`, or `mpi4py`.
- **Julia**: Use `Distributed.jl`. Add processes via `addprocs(n)` and use `@everywhere` for loading dependencies.

## 7. SLURM Job Scheduling

```bash
#!/bin/bash
#SBATCH --nodes=4 --ntasks-per-node=32 --cpus-per-task=1
#SBATCH --time=24:00:00 --mem=64GB
module load gcc openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python simulation.py
```

**Array jobs**:
```bash
#SBATCH --array=1-100
PARAM=$(awk "NR==$SLURM_ARRAY_TASK_ID" parameters.txt)
```

## 8. Load Balancing

```python
# Master-worker pattern
if rank == 0:  # Master distributes tasks
    for worker in range(1, size):
        comm.send(tasks[task_idx], dest=worker); task_idx += 1
    while task_idx < len(tasks):
        result = comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(tasks[task_idx], dest=status.Get_source()); task_idx += 1
else:  # Worker processes tasks
    while True:
        task = comm.recv(source=0)
        if task is None: break
        comm.send(process_task(task), dest=0)
```

## 9. Dask (Python Distributed)

```python
from dask.distributed import Client
client = Client(n_workers=4)
x = da.random.random((100000, 100000), chunks=(10000, 10000))
result = ((x + x.T) / 2).sum().compute()
```

## Parallelization Best Practices

| Strategy | Goal | Technique |
|----------|------|-----------|
| **Domain Decomposition** | Scale memory | Split spatial grid (Ghost cells) |
| **Task Parallelism** | Scale throughput | Dynamic load balancing (Work stealing) |
| **Hybrid** | Maximize hardware | MPI (Inter-node) + OpenMP (Intra-node) |
| **Vectorization** | CPU utilization | SIMD instructions (AVX-512) |

## Scaling Laws

- **Strong**: Fixed problem, increase processors. Goal: Speedup = T₁/Tₙ
- **Weak**: Problem scales with processors. Goal: Constant runtime
- **Amdahl's Law**: Speedup = 1 / (s + (1-s)/n), s = serial fraction

## Performance & Optimization Checklist

- [ ] **Memory Management**: Minimize host-to-device transfers. Use pinned memory for faster I/O.
- [ ] **Coalescing**: Ensure threads access contiguous memory locations.
- [ ] **Occupancy**: Optimize threads/block (typically 128-512) to maximize hardware utilization.
- [ ] **Streams**: Use asynchronous streams to overlap computation with data transfers.

## Profiling Tools

- **NSight Systems**: `nsys profile --stats=true python/julia script.py`
- **NSight Compute**: `ncu --set full ...` for detailed kernel analysis.
- **Julia Profiler**: Use `Profile` and `PProf.jl` for CPU/Memory profiling.
