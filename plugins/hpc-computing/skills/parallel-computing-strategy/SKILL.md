---
name: parallel-computing-strategy
description: Design and implement parallel computing strategies for distributed and shared-memory HPC systems using MPI, OpenMP, and workflow orchestration frameworks. Use this skill when implementing MPI distributed-memory parallelization with mpi4py (Python) or MPI.jl (Julia) for multi-node cluster computing, designing OpenMP shared-memory parallelization with #pragma omp directives for loop-level multi-threading on single nodes, creating hybrid MPI+OpenMP applications that combine distributed and shared-memory parallelism for hierarchical parallelization, writing SLURM job scripts with #SBATCH directives for resource allocation (--nodes, --ntasks-per-node, --cpus-per-task, --gres=gpu) on HPC clusters, implementing dynamic load balancing using master-worker patterns or work-stealing algorithms for heterogeneous workloads, orchestrating parallel workflows with Dask (Python) or Dagger.jl (Julia) for task graphs and out-of-core computation, designing data parallelism for array operations where the same operation applies to different data elements, implementing task parallelism for independent computational tasks with dynamic scheduling, optimizing MPI communication patterns (point-to-point, collective operations, non-blocking communication) to minimize overhead, creating SLURM array jobs for parameter sweeps and ensemble simulations, implementing MPI domain decomposition for PDE solvers with halo exchanges and ghost cell communication, designing scalability strategies and analyzing strong scaling (fixed problem size) versus weak scaling (scaled problem size), profiling parallel performance with Scalasca, Score-P, Intel VTune, or TAU performance tools, managing job dependencies and workflow pipelines in SLURM with --dependency flags, implementing parallel I/O strategies for reading/writing large datasets efficiently across nodes, working with .sh SLURM batch scripts or MPI-enabled Python/Julia/C/Fortran source files, or optimizing resource utilization and queue wait times on shared HPC infrastructure.
---

# Parallel Computing Strategy

## When to use this skill

- When implementing MPI parallelization with mpi4py (Python) or MPI.jl (Julia)
- When writing OpenMP parallel loops with #pragma omp directives in C/C++/Fortran
- When creating hybrid MPI+OpenMP applications for multi-node clusters
- When writing SLURM job scripts (.sh files) with #SBATCH resource directives
- When configuring --nodes, --ntasks-per-node, --cpus-per-task in SLURM submissions
- When implementing dynamic load balancing with master-worker or work-stealing patterns
- When orchestrating workflows with Dask (Python) or Dagger.jl (Julia)
- When designing data parallelism for array computations across multiple processors
- When implementing task parallelism for independent computational workflows
- When optimizing MPI communication (MPI_Send, MPI_Recv, MPI_Bcast, MPI_Reduce)
- When creating SLURM array jobs for parameter sweeps (#SBATCH --array=1-100)
- When implementing domain decomposition for PDE solvers with halo exchanges
- When analyzing strong scaling and weak scaling performance of parallel applications
- When profiling parallel code with Scalasca, Score-P, Intel VTune, or TAU
- When managing SLURM job dependencies with --dependency=afterok:jobid
- When implementing parallel I/O for large-scale data processing across nodes
- When working with MPI-enabled Python, Julia, C, C++, or Fortran source files
- When optimizing queue wait times and resource allocation on shared HPC clusters
- When designing collective communication patterns to minimize inter-node overhead
- When balancing computational loads across heterogeneous computing resources

## Overview

Design and implement parallel computing strategies for scientific workflows across distributed and shared-memory systems. This skill covers task and data parallelism patterns, job scheduling optimization, workflow orchestration, load balancing, and communication strategies for HPC environments.

## Core Capabilities

### 1. Task and Data Parallelism Design

#### Understanding Parallelism Types

**Data Parallelism:**
- Same operation applied to different data elements
- Ideal for array operations, matrix computations, image processing
- Scales well with problem size, low communication overhead
- Example: Element-wise operations, SIMD vectorization

**Task Parallelism:**
- Different operations executed concurrently
- Suitable for heterogeneous workflows with independent tasks
- Dynamic task scheduling for load balancing
- Example: Parameter sweeps, ensemble simulations

**Pipeline Parallelism:**
- Sequential stages processed concurrently on different data
- Maximizes resource utilization in multi-stage workflows
- Example: Data ingestion → processing → analysis → visualization

#### MPI (Message Passing Interface) for Distributed Memory

**When to Use MPI:**
- Multi-node clusters with distributed memory
- Large-scale simulations requiring 100s-1000s of cores
- Fine-grained control over communication
- Scalability to supercomputer scale

**MPI Programming Patterns:**

```python
# Python (mpi4py)
from mpi4py import MPI
import numpy as np

def mpi_data_parallel():
    """Distribute array computation across MPI ranks."""
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Problem size and decomposition
    N = 1000000
    chunk_size = N // size
    start = rank * chunk_size
    end = start + chunk_size if rank < size - 1 else N

    # Local computation
    local_data = np.arange(start, end)
    local_result = np.sum(local_data ** 2)

    # Reduction to gather results
    total = comm.reduce(local_result, op=MPI.SUM, root=0)
    return total

def mpi_collective_ops():
    """Collective communication patterns."""
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Scatter: distribute data from root
    if rank == 0:
        data = np.arange(size * 10).reshape(size, 10)
    else:
        data = None
    local_data = comm.scatter(data, root=0)

    # Process local data
    local_result = np.sum(local_data)

    # Allreduce: reduction with broadcast
    total_sum = comm.allreduce(local_result, op=MPI.SUM)
    return total_sum
```

**MPI Communication Patterns:**
- **Point-to-Point**: `send/recv` for direct communication
- **Collective**: `broadcast`, `scatter`, `gather`, `reduce`, `allreduce`
- **Non-Blocking**: `isend/irecv` for overlapping computation/communication

**MPI Best Practices:**
1. Minimize communication frequency
2. Overlap communication with computation (non-blocking)
3. Use collective operations vs multiple point-to-point
4. Balance workload to avoid idle ranks

#### OpenMP for Shared-Memory Parallelism

**When to Use OpenMP:**
- Single-node multi-core systems
- Shared memory architecture
- Loop-level parallelization
- Quick parallelization with minimal code changes

**OpenMP Programming:**

```c
// C/C++ with OpenMP
#include <omp.h>

void openmp_parallel_loop() {
    int n = 1000000;
    double sum = 0.0;

    // Parallel loop with reduction
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += i * i;
    }
}

void openmp_task_parallelism() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                // Task 1: heavy computation
            }
            #pragma omp task
            {
                // Task 2: heavy computation
            }
            #pragma omp taskwait
        }
    }
}
```

**OpenMP Clauses:**
- `private/shared`: Variable scope control
- `reduction`: Combine results from threads
- `schedule`: Loop iteration distribution (static, dynamic, guided)
- `num_threads`: Thread count control

#### Hybrid MPI+OpenMP

**When to Use Hybrid:**
- Multi-node clusters with many cores per node
- Reduce MPI communication overhead
- Better memory usage, hierarchical parallelism

```c
// Hybrid MPI+OpenMP
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI rank spawns OpenMP threads
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // Parallel computation within node
        #pragma omp for
        for (int i = 0; i < 1000; i++) {
            // Computation
        }
    }

    MPI_Finalize();
    return 0;
}
```

### 2. Job Scheduling and Orchestration

#### SLURM Job Scheduling

**SLURM Job Script Template:**
```bash
#!/bin/bash
#SBATCH --job-name=simulation       # Job name
#SBATCH --partition=compute         # Partition (queue)
#SBATCH --nodes=4                   # Nodes
#SBATCH --ntasks-per-node=32        # MPI tasks per node
#SBATCH --cpus-per-task=1           # OpenMP threads
#SBATCH --time=24:00:00             # Wall time
#SBATCH --mem=64GB                  # Memory per node
#SBATCH --output=job_%j.out         # stdout
#SBATCH --error=job_%j.err          # stderr
#SBATCH --mail-type=END,FAIL

# Load modules
module load gcc/11.2.0 openmpi/4.1.1 python/3.10

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run application
srun python simulation.py
```

**SLURM Array Jobs (Parameter Sweeps):**
```bash
#!/bin/bash
#SBATCH --array=1-100               # 100 array tasks
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=output_%A_%a.out   # %A = job ID, %a = task ID

# Use SLURM_ARRAY_TASK_ID for parameter
PARAM=$(awk "NR==$SLURM_ARRAY_TASK_ID" parameters.txt)
python simulation.py --param $PARAM
```

**Resource Allocation:**
- **Exclusive mode**: `--exclusive` for dedicated nodes
- **GPU allocation**: `--gres=gpu:4`
- **Memory**: `--mem-per-cpu=4GB`
- **Dependencies**: `--dependency=afterok:12345`

#### Load Balancing Strategies

**Dynamic Work Distribution (Master-Worker):**

```python
# Python: Dynamic load balancing with MPI
from mpi4py import MPI

def dynamic_work_distribution(tasks):
    """Master-worker pattern for load balancing."""
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        # Master: distribute tasks dynamically
        task_idx = 0
        finished = 0

        # Send initial tasks
        for worker in range(1, size):
            if task_idx < len(tasks):
                comm.send(tasks[task_idx], dest=worker, tag=1)
                task_idx += 1

        # Receive results, send new tasks
        while finished < size - 1:
            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            worker = status.Get_source()

            if task_idx < len(tasks):
                comm.send(tasks[task_idx], dest=worker, tag=1)
                task_idx += 1
            else:
                comm.send(None, dest=worker, tag=1)
                finished += 1
    else:
        # Worker: process tasks
        while True:
            task = comm.recv(source=0, tag=1)
            if task is None:
                break
            result = process_task(task)
            comm.send(result, dest=0, tag=2)
```

**Load Balancing Approaches:**
- **Static**: Equal work distribution (known workload)
- **Dynamic**: Work queue/task pool (heterogeneous)
- **Work-stealing**: Idle workers steal from busy ones

### 3. Workflow Management with Dask and Dagger.jl

#### Dask (Python)

**When to Use Dask:**
- Python-centric workflows
- Out-of-core computation (data > memory)
- Dynamic task graphs
- NumPy/Pandas integration

**Dask Programming:**

```python
import dask
import dask.array as da
from dask.distributed import Client

# Start Dask cluster
client = Client(n_workers=4, threads_per_worker=2)

def dask_array_computation():
    """Large-scale array computation."""
    # Lazy array
    x = da.random.random((100000, 100000), chunks=(10000, 10000))

    # Lazy operations
    y = (x + x.T) / 2  # Symmetrize
    z = da.linalg.svd(y)

    # Trigger computation
    return z[0].compute()

def dask_delayed_workflow():
    """Custom task graph."""
    @dask.delayed
    def load_data(filename):
        return pd.read_csv(filename)

    @dask.delayed
    def process_data(data):
        return data.groupby('key').sum()

    # Build task graph
    files = ['data1.csv', 'data2.csv']
    loaded = [load_data(f) for f in files]
    processed = [process_data(d) for d in loaded]

    # Execute
    return dask.compute(*processed)
```

**Dask Best Practices:**
1. Chunk sizes: 10MB-1GB
2. Use `persist()` for reused intermediates
3. Monitor task graph size
4. Profile with Dask dashboard

#### Dagger.jl (Julia)

**Workflow Orchestration in Julia:**

```julia
using Dagger
using Distributed

function dagger_workflow()
    """Parallel workflow with Dagger.jl."""
    addprocs(4)

    # Define delayed tasks
    a = Dagger.@spawn expensive_computation_1()
    b = Dagger.@spawn expensive_computation_2()

    # Compose tasks
    c = Dagger.@spawn combine_results(a, b)

    # Fetch result
    return fetch(c)
end
```

## Performance Optimization

### Communication Optimization

**Minimize Communication:**
1. Increase computation-to-communication ratio
2. Use halo exchanges for ghost cells
3. Overlap communication with computation (non-blocking)
4. Aggregate small messages

**Communication Patterns:**
- Point-to-point: Neighbor communication
- Collective: Broadcast, reduction trees
- Custom topologies: Cartesian, graph

### Scalability Analysis

**Strong Scaling:**
- Fixed problem size, increase processors
- Speedup = T₁ / Tₙ, Efficiency = Speedup / n

**Weak Scaling:**
- Problem size scales with processors
- Goal: Constant runtime

**Amdahl's Law:**
- Speedup = 1 / (s + (1-s)/n) where s = serial fraction

### Profiling Tools

- **Scalasca/Score-P**: MPI performance
- **Intel VTune**: Thread-level profiling
- **ARM MAP**: HPC profiling/debugging
- **TAU**: Performance instrumentation

## Best Practices

### Parallel Design Checklist

- [ ] Identify parallel regions and dependencies
- [ ] Choose parallelization strategy (data/task/pipeline)
- [ ] Estimate communication overhead
- [ ] Design load balancing mechanism
- [ ] Select programming model (MPI/OpenMP/hybrid)

### Resource Allocation

1. Right-size jobs (don't over-allocate)
2. Node-aware placement (minimize inter-node communication)
3. Memory management (avoid swapping)
4. I/O optimization (parallel file systems)

### Debugging Parallel Code

1. Test serial version first
2. Start with small processor counts
3. Check for deadlocks in MPI
4. Use race condition detectors
5. Validate data dependencies

## Resources

### references/

- `mpi_patterns.md`: Common MPI communication patterns
- `openmp_optimization.md`: OpenMP performance tuning
- `scheduler_reference.md`: SLURM/PBS advanced features
- `workflow_examples.md`: Real-world workflow examples

Load references for detailed patterns, optimization techniques, and troubleshooting.
