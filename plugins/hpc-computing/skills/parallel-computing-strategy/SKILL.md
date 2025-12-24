---
name: parallel-computing-strategy
version: "1.0.6"
maturity: "5-Expert"
specialization: HPC Parallelization
description: Design parallel strategies with MPI (distributed memory), OpenMP (shared memory), hybrid MPI+OpenMP, SLURM scheduling, Dask/Dagger.jl workflows, and load balancing. Use when implementing multi-node parallelization, writing job scripts, or optimizing HPC workflows.
---

# Parallel Computing Strategy

MPI, OpenMP, hybrid parallelism, and workflow orchestration for HPC.

---

## Parallelism Types

| Type | Best For | Example |
|------|----------|---------|
| Data | Same operation, different data | Array operations, SIMD |
| Task | Independent operations | Parameter sweeps |
| Pipeline | Sequential stages on different data | ETL workflows |

---

## MPI (Distributed Memory)

```python
from mpi4py import MPI
import numpy as np

def mpi_data_parallel():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Decompose problem
    N = 1000000
    chunk = N // size
    start, end = rank * chunk, (rank + 1) * chunk

    # Local computation
    local_result = np.sum(np.arange(start, end) ** 2)

    # Reduction
    total = comm.reduce(local_result, op=MPI.SUM, root=0)
    return total
```

**Communication Patterns**:
- **Point-to-point**: `send/recv` for direct
- **Collective**: `broadcast`, `scatter`, `gather`, `reduce`, `allreduce`
- **Non-blocking**: `isend/irecv` for overlap

---

## OpenMP (Shared Memory)

```c
#include <omp.h>

void parallel_loop() {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 1000000; i++) {
        sum += i * i;
    }
}

void task_parallelism() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            { /* Task 1 */ }
            #pragma omp task
            { /* Task 2 */ }
            #pragma omp taskwait
        }
    }
}
```

**Clauses**: `private/shared`, `reduction`, `schedule(dynamic)`, `num_threads`

---

## Hybrid MPI+OpenMP

```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

#pragma omp parallel
{
    #pragma omp for
    for (int i = 0; i < 1000; i++) {
        // Computation
    }
}

MPI_Finalize();
```

**Use when**: Multi-node clusters with many cores per node

---

## SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --output=job_%j.out

module load gcc/11.2.0 openmpi/4.1.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python simulation.py
```

### Array Jobs (Parameter Sweeps)

```bash
#SBATCH --array=1-100
PARAM=$(awk "NR==$SLURM_ARRAY_TASK_ID" parameters.txt)
python simulation.py --param $PARAM
```

---

## Load Balancing (Master-Worker)

```python
def dynamic_work_distribution(tasks):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:  # Master
        task_idx = 0
        for worker in range(1, size):
            if task_idx < len(tasks):
                comm.send(tasks[task_idx], dest=worker)
                task_idx += 1
        # Receive results, send new tasks
        while task_idx < len(tasks):
            result = comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(tasks[task_idx], dest=status.Get_source())
            task_idx += 1
    else:  # Worker
        while True:
            task = comm.recv(source=0)
            if task is None: break
            result = process_task(task)
            comm.send(result, dest=0)
```

---

## Dask Workflow

```python
import dask
import dask.array as da
from dask.distributed import Client

client = Client(n_workers=4)

# Large array computation
x = da.random.random((100000, 100000), chunks=(10000, 10000))
y = (x + x.T) / 2
result = y.sum().compute()

# Task graph
@dask.delayed
def process(data):
    return data.groupby('key').sum()

processed = [process(load(f)) for f in files]
dask.compute(*processed)
```

---

## Scaling Analysis

| Type | Definition | Goal |
|------|------------|------|
| Strong | Fixed problem size, increase processors | Speedup = T₁/Tₙ |
| Weak | Problem scales with processors | Constant runtime |

**Amdahl's Law**: Speedup = 1 / (s + (1-s)/n), where s = serial fraction

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Minimize communication | Overlap computation with non-blocking |
| Right-size jobs | Don't over-allocate |
| Test serial first | Validate correctness |
| Profile | Scalasca, VTune, TAU |
| Balance loads | Dynamic scheduling for heterogeneous work |

---

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Too much communication | Dominates computation time |
| Deadlocks | MPI send/recv ordering |
| Race conditions | OpenMP shared variables |
| Load imbalance | Idle processors |

---

## Checklist

- [ ] Parallelization strategy chosen (data/task/pipeline)
- [ ] Communication overhead estimated
- [ ] Load balancing mechanism designed
- [ ] Serial version tested
- [ ] Scaling benchmarked
- [ ] SLURM resources right-sized

---

**Version**: 1.0.5
