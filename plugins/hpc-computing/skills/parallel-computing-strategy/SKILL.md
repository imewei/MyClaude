---
name: parallel-computing-strategy
version: "1.0.7"
description: Design parallel strategies with MPI (distributed memory), OpenMP (shared memory), hybrid MPI+OpenMP, SLURM scheduling, Dask/Dagger.jl workflows, and load balancing. Use when implementing multi-node parallelization, writing job scripts, or optimizing HPC workflows.
---

# Parallel Computing Strategy

## Parallelism Types

| Type | Best For |
|------|----------|
| Data | Same operation, different data |
| Task | Independent operations |
| Pipeline | Sequential stages on different data |

## MPI (Distributed)

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
chunk = N // size
local_result = np.sum(np.arange(rank*chunk, (rank+1)*chunk) ** 2)
total = comm.reduce(local_result, op=MPI.SUM, root=0)
```

**Patterns**: `send/recv`, `broadcast`, `scatter`, `gather`, `reduce`, `allreduce`, `isend/irecv`

## OpenMP (Shared)

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

## Hybrid MPI+OpenMP

```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#pragma omp parallel for
for (int i = 0; i < 1000; i++) { /* Computation */ }
MPI_Finalize();
```

## SLURM

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

## Load Balancing

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

## Dask

```python
from dask.distributed import Client
client = Client(n_workers=4)
x = da.random.random((100000, 100000), chunks=(10000, 10000))
result = ((x + x.T) / 2).sum().compute()
```

## Scaling

- **Strong**: Fixed problem, increase processors. Goal: Speedup = T₁/Tₙ
- **Weak**: Problem scales with processors. Goal: Constant runtime
- **Amdahl's Law**: Speedup = 1 / (s + (1-s)/n), s = serial fraction

**Outcome**: Efficient multi-node parallelization with optimized communication and load balancing
