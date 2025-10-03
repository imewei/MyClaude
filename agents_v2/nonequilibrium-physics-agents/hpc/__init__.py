"""HPC Integration for Optimal Control.

This module provides high-performance computing capabilities:
1. SLURM cluster integration
2. Dask distributed computing
3. Parallel optimization
4. Batch job management

Enable scaling to large-scale problems and parameter sweeps.

Author: Nonequilibrium Physics Agents
"""

__version__ = "4.2.0-dev"

# Unified Scheduler Interface (Week 29-30)
try:
    from .schedulers import (
        JobStatus,
        ResourceRequirements,
        JobInfo,
        Scheduler,
        SLURMScheduler as UnifiedSLURM,
        PBSScheduler,
        LocalScheduler,
        JobManager
    )
    SCHEDULERS_AVAILABLE = True
except ImportError:
    SCHEDULERS_AVAILABLE = False
    JobStatus = None
    ResourceRequirements = None
    JobInfo = None
    Scheduler = None
    UnifiedSLURM = None
    PBSScheduler = None
    LocalScheduler = None
    JobManager = None

# SLURM Integration
try:
    from .slurm import (
        SLURMConfig,
        SLURMJob,
        SLURMScheduler,
        submit_job,
        submit_array_job,
        monitor_jobs,
        cancel_job
    )
    SLURM_AVAILABLE = True
except ImportError:
    SLURM_AVAILABLE = False
    SLURMConfig = None
    SLURMJob = None
    SLURMScheduler = None
    submit_job = None
    submit_array_job = None
    monitor_jobs = None
    cancel_job = None

# Dask Distributed
try:
    from .distributed import (
        DaskCluster,
        ParallelExecutor,
        distribute_computation,
        create_local_cluster,
        create_slurm_cluster
    )
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    DaskCluster = None
    ParallelExecutor = None
    distribute_computation = None
    create_local_cluster = None
    create_slurm_cluster = None

# Parallel Optimization
try:
    from .parallel import (
        ParallelOptimizer,
        ParameterSweep,
        GridSearch,
        RandomSearch,
        BayesianOptimization,
        run_parallel_sweep
    )
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    ParallelOptimizer = None
    ParameterSweep = None
    GridSearch = None
    RandomSearch = None
    BayesianOptimization = None
    run_parallel_sweep = None

__all__ = [
    # Unified Schedulers (Week 29-30)
    'JobStatus',
    'ResourceRequirements',
    'JobInfo',
    'Scheduler',
    'UnifiedSLURM',
    'PBSScheduler',
    'LocalScheduler',
    'JobManager',
    'SCHEDULERS_AVAILABLE',

    # SLURM
    'SLURMConfig',
    'SLURMJob',
    'SLURMScheduler',
    'submit_job',
    'submit_array_job',
    'monitor_jobs',
    'cancel_job',
    'SLURM_AVAILABLE',

    # Dask
    'DaskCluster',
    'ParallelExecutor',
    'distribute_computation',
    'create_local_cluster',
    'create_slurm_cluster',
    'DASK_AVAILABLE',

    # Parallel Optimization
    'ParallelOptimizer',
    'ParameterSweep',
    'GridSearch',
    'RandomSearch',
    'BayesianOptimization',
    'run_parallel_sweep',
    'PARALLEL_AVAILABLE',
]
