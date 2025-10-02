"""Dask Distributed Computing for Optimal Control.

This module provides Dask integration for parallel and distributed computing.

Features:
1. Local cluster for multi-core parallelism
2. SLURM cluster integration
3. Distributed map/reduce operations
4. Automatic task scheduling
5. Result collection and aggregation

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from pathlib import Path
import numpy as np

# Check if Dask is available
try:
    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster, as_completed, wait
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Check if dask-jobqueue is available (for SLURM)
try:
    from dask_jobqueue import SLURMCluster
    DASK_JOBQUEUE_AVAILABLE = True
except ImportError:
    DASK_JOBQUEUE_AVAILABLE = False


# =============================================================================
# Cluster Management
# =============================================================================

class DaskCluster:
    """Wrapper for Dask cluster management.

    Provides unified interface for local and SLURM clusters.
    """

    def __init__(
        self,
        cluster_type: str = "local",
        n_workers: int = 4,
        threads_per_worker: int = 1,
        memory_limit: str = "4GB",
        **kwargs
    ):
        """Initialize Dask cluster.

        Args:
            cluster_type: "local" or "slurm"
            n_workers: Number of workers
            threads_per_worker: Threads per worker
            memory_limit: Memory per worker
            **kwargs: Additional cluster-specific arguments
        """
        if not DASK_AVAILABLE:
            raise ImportError("Dask not available. Install with: pip install dask distributed")

        self.cluster_type = cluster_type
        self.n_workers = n_workers
        self.cluster = None
        self.client = None

        if cluster_type == "local":
            self.cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                **kwargs
            )
        elif cluster_type == "slurm":
            if not DASK_JOBQUEUE_AVAILABLE:
                raise ImportError("dask-jobqueue not available. Install with: pip install dask-jobqueue")

            self.cluster = SLURMCluster(
                cores=threads_per_worker,
                memory=memory_limit,
                **kwargs
            )
            self.cluster.scale(n_workers)
        else:
            raise ValueError(f"Unknown cluster type: {cluster_type}")

        self.client = Client(self.cluster)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close cluster and client."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()

    def scale(self, n_workers: int):
        """Scale cluster to n workers.

        Args:
            n_workers: Target number of workers
        """
        if self.cluster:
            self.cluster.scale(n_workers)

    def get_dashboard_link(self) -> Optional[str]:
        """Get link to Dask dashboard.

        Returns:
            Dashboard URL if available
        """
        if self.cluster:
            return self.cluster.dashboard_link
        return None

    def submit(self, func: Callable, *args, **kwargs):
        """Submit task to cluster.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object
        """
        return self.client.submit(func, *args, **kwargs)

    def map(self, func: Callable, *iterables, **kwargs):
        """Map function over iterables.

        Args:
            func: Function to map
            *iterables: Iterables to map over
            **kwargs: Keyword arguments for func

        Returns:
            List of futures
        """
        return self.client.map(func, *iterables, **kwargs)

    def gather(self, futures):
        """Gather results from futures.

        Args:
            futures: Future or list of futures

        Returns:
            Result or list of results
        """
        return self.client.gather(futures)

    def compute(self, *args, **kwargs):
        """Compute dask collections.

        Args:
            *args: Dask collections
            **kwargs: Compute options

        Returns:
            Results
        """
        return dask.compute(*args, **kwargs)


# =============================================================================
# Parallel Executor
# =============================================================================

class ParallelExecutor:
    """High-level parallel execution interface.

    Simplifies common parallel patterns.
    """

    def __init__(self, cluster: Optional[DaskCluster] = None):
        """Initialize executor.

        Args:
            cluster: DaskCluster (creates local cluster if None)
        """
        if cluster is None:
            self.cluster = create_local_cluster()
            self.owns_cluster = True
        else:
            self.cluster = cluster
            self.owns_cluster = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.owns_cluster:
            self.cluster.close()

    def map(
        self,
        func: Callable,
        inputs: List[Any],
        show_progress: bool = True
    ) -> List[Any]:
        """Map function over inputs in parallel.

        Args:
            func: Function to apply
            inputs: List of inputs
            show_progress: Whether to show progress bar

        Returns:
            List of results
        """
        futures = self.cluster.map(func, inputs)

        if show_progress:
            from dask.distributed import progress
            progress(futures)

        return self.cluster.gather(futures)

    def starmap(
        self,
        func: Callable,
        inputs: List[Tuple],
        show_progress: bool = True
    ) -> List[Any]:
        """Map function over tuple arguments.

        Args:
            func: Function to apply
            inputs: List of tuples (each tuple is unpacked as args)
            show_progress: Whether to show progress bar

        Returns:
            List of results
        """
        def wrapper(args):
            return func(*args)

        return self.map(wrapper, inputs, show_progress)

    def map_reduce(
        self,
        map_func: Callable,
        reduce_func: Callable,
        inputs: List[Any],
        show_progress: bool = True
    ) -> Any:
        """Map-reduce pattern.

        Args:
            map_func: Map function
            reduce_func: Reduce function (should be associative)
            inputs: List of inputs
            show_progress: Whether to show progress bar

        Returns:
            Reduced result
        """
        # Map phase
        mapped = self.map(map_func, inputs, show_progress)

        # Reduce phase
        while len(mapped) > 1:
            # Pairwise reduction
            pairs = [
                (mapped[i], mapped[i+1])
                for i in range(0, len(mapped)-1, 2)
            ]

            # Handle odd length
            if len(mapped) % 2 == 1:
                pairs.append((mapped[-1],))

            def reduce_pair(pair):
                if len(pair) == 1:
                    return pair[0]
                return reduce_func(pair[0], pair[1])

            mapped = self.map(reduce_pair, pairs, show_progress=False)

        return mapped[0] if mapped else None

    def batch_compute(
        self,
        func: Callable,
        inputs: List[Any],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> List[Any]:
        """Compute in batches (useful for memory management).

        Args:
            func: Function to apply
            inputs: List of inputs
            batch_size: Number of inputs per batch
            show_progress: Whether to show progress bar

        Returns:
            List of results
        """
        results = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_results = self.map(func, batch, show_progress)
            results.extend(batch_results)

        return results


# =============================================================================
# Distributed Computation Utilities
# =============================================================================

def distribute_computation(
    func: Callable,
    inputs: List[Any],
    n_workers: int = 4,
    cluster_type: str = "local",
    show_progress: bool = True,
    **cluster_kwargs
) -> List[Any]:
    """Distribute computation across workers.

    Args:
        func: Function to apply to each input
        inputs: List of inputs
        n_workers: Number of parallel workers
        cluster_type: "local" or "slurm"
        show_progress: Whether to show progress
        **cluster_kwargs: Additional cluster arguments

    Returns:
        List of results
    """
    with DaskCluster(
        cluster_type=cluster_type,
        n_workers=n_workers,
        **cluster_kwargs
    ) as cluster:
        executor = ParallelExecutor(cluster)
        return executor.map(func, inputs, show_progress)


def parallel_solve(
    solver_func: Callable,
    problem_configs: List[Dict[str, Any]],
    n_workers: int = 4,
    cluster_type: str = "local"
) -> List[Any]:
    """Solve multiple optimal control problems in parallel.

    Args:
        solver_func: Solver function (takes config dict, returns result)
        problem_configs: List of problem configurations
        n_workers: Number of parallel workers
        cluster_type: "local" or "slurm"

    Returns:
        List of solver results
    """
    return distribute_computation(
        solver_func,
        problem_configs,
        n_workers=n_workers,
        cluster_type=cluster_type
    )


def parallel_parameter_sweep(
    objective_func: Callable,
    parameter_grid: List[Dict[str, Any]],
    n_workers: int = 4,
    cluster_type: str = "local"
) -> Tuple[List[Any], List[Dict]]:
    """Run parameter sweep in parallel.

    Args:
        objective_func: Function to evaluate (takes params, returns score)
        parameter_grid: List of parameter dictionaries
        n_workers: Number of parallel workers
        cluster_type: "local" or "slurm"

    Returns:
        Tuple of (scores, parameter_configs)
    """
    scores = distribute_computation(
        objective_func,
        parameter_grid,
        n_workers=n_workers,
        cluster_type=cluster_type
    )

    return scores, parameter_grid


# =============================================================================
# Distributed Array Operations
# =============================================================================

def create_distributed_array(
    shape: Tuple[int, ...],
    chunks: Union[str, Tuple[int, ...]] = "auto",
    dtype: type = np.float64
) -> 'da.Array':
    """Create distributed Dask array.

    Args:
        shape: Array shape
        chunks: Chunk size ("auto" or tuple)
        dtype: Data type

    Returns:
        Dask array
    """
    if not DASK_AVAILABLE:
        raise ImportError("Dask not available")

    return da.zeros(shape, chunks=chunks, dtype=dtype)


def distribute_array(
    array: np.ndarray,
    chunks: Union[str, Tuple[int, ...]] = "auto"
) -> 'da.Array':
    """Convert NumPy array to distributed Dask array.

    Args:
        array: NumPy array
        chunks: Chunk size

    Returns:
        Dask array
    """
    if not DASK_AVAILABLE:
        raise ImportError("Dask not available")

    return da.from_array(array, chunks=chunks)


def parallel_matmul(
    A: Union[np.ndarray, 'da.Array'],
    B: Union[np.ndarray, 'da.Array'],
    chunks: Union[str, Tuple[int, ...]] = "auto"
) -> np.ndarray:
    """Parallel matrix multiplication.

    Args:
        A: Matrix A
        B: Matrix B
        chunks: Chunk size for computation

    Returns:
        A @ B computed in parallel
    """
    if not DASK_AVAILABLE:
        return A @ B

    # Convert to Dask arrays if needed
    if not isinstance(A, da.Array):
        A = da.from_array(A, chunks=chunks)
    if not isinstance(B, da.Array):
        B = da.from_array(B, chunks=chunks)

    # Compute
    C = da.matmul(A, B)
    return C.compute()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_local_cluster(
    n_workers: int = 4,
    threads_per_worker: int = 1,
    memory_limit: str = "4GB"
) -> DaskCluster:
    """Create local Dask cluster.

    Args:
        n_workers: Number of workers
        threads_per_worker: Threads per worker
        memory_limit: Memory per worker

    Returns:
        DaskCluster object
    """
    return DaskCluster(
        cluster_type="local",
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )


def create_slurm_cluster(
    n_workers: int = 10,
    cores: int = 1,
    memory: str = "4GB",
    queue: str = "general",
    walltime: str = "01:00:00",
    **kwargs
) -> DaskCluster:
    """Create SLURM Dask cluster.

    Args:
        n_workers: Number of workers
        cores: Cores per worker
        memory: Memory per worker
        queue: SLURM partition
        walltime: Wall time limit
        **kwargs: Additional SLURM arguments

    Returns:
        DaskCluster object
    """
    if not DASK_JOBQUEUE_AVAILABLE:
        raise ImportError("dask-jobqueue not available")

    return DaskCluster(
        cluster_type="slurm",
        n_workers=n_workers,
        threads_per_worker=cores,
        memory_limit=memory,
        queue=queue,
        walltime=walltime,
        **kwargs
    )


def get_cluster_info(cluster: DaskCluster) -> Dict[str, Any]:
    """Get cluster information.

    Args:
        cluster: DaskCluster object

    Returns:
        Dictionary with cluster info
    """
    info = {
        "type": cluster.cluster_type,
        "n_workers": len(cluster.client.scheduler_info()["workers"]),
        "total_cores": sum(
            w["nthreads"]
            for w in cluster.client.scheduler_info()["workers"].values()
        ),
        "total_memory": sum(
            w["memory_limit"]
            for w in cluster.client.scheduler_info()["workers"].values()
        ),
        "dashboard": cluster.get_dashboard_link()
    }

    return info


# =============================================================================
# Advanced Patterns
# =============================================================================

def adaptive_batch_processing(
    func: Callable,
    inputs: List[Any],
    initial_batch_size: int = 10,
    memory_threshold: float = 0.8,
    cluster: Optional[DaskCluster] = None
) -> List[Any]:
    """Adaptive batch processing with memory monitoring.

    Automatically adjusts batch size based on memory usage.

    Args:
        func: Function to apply
        inputs: List of inputs
        initial_batch_size: Starting batch size
        memory_threshold: Reduce batch size if memory > threshold
        cluster: DaskCluster (creates local if None)

    Returns:
        List of results
    """
    if cluster is None:
        cluster = create_local_cluster()
        owns_cluster = True
    else:
        owns_cluster = False

    results = []
    batch_size = initial_batch_size
    i = 0

    try:
        while i < len(inputs):
            # Get current memory usage
            workers = cluster.client.scheduler_info()["workers"]
            avg_memory = np.mean([
                w["memory"] / w["memory_limit"]
                for w in workers.values()
            ])

            # Adjust batch size
            if avg_memory > memory_threshold:
                batch_size = max(1, batch_size // 2)
            elif avg_memory < memory_threshold / 2:
                batch_size = min(len(inputs), batch_size * 2)

            # Process batch
            batch = inputs[i:i+batch_size]
            futures = cluster.map(func, batch)
            batch_results = cluster.gather(futures)
            results.extend(batch_results)

            i += batch_size

    finally:
        if owns_cluster:
            cluster.close()

    return results


def fault_tolerant_map(
    func: Callable,
    inputs: List[Any],
    max_retries: int = 3,
    cluster: Optional[DaskCluster] = None
) -> Tuple[List[Any], List[Tuple[int, Exception]]]:
    """Map with automatic retry on failure.

    Args:
        func: Function to apply
        inputs: List of inputs
        max_retries: Maximum retry attempts
        cluster: DaskCluster

    Returns:
        Tuple of (results, failures)
        - results: List with successful results or None for failures
        - failures: List of (index, exception) for failed inputs
    """
    if cluster is None:
        cluster = create_local_cluster()
        owns_cluster = True
    else:
        owns_cluster = False

    results = [None] * len(inputs)
    failures = []

    try:
        # Track remaining work
        work_queue = list(enumerate(inputs))

        for attempt in range(max_retries):
            if not work_queue:
                break

            # Submit all remaining work
            futures_map = {}
            for idx, inp in work_queue:
                future = cluster.submit(func, inp)
                futures_map[future] = idx

            # Collect results
            completed = as_completed(futures_map.keys())
            new_work_queue = []

            for future in completed:
                idx = futures_map[future]

                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Final attempt failed
                        failures.append((idx, e))
                    else:
                        # Retry
                        new_work_queue.append((idx, inputs[idx]))

            work_queue = new_work_queue

    finally:
        if owns_cluster:
            cluster.close()

    return results, failures


# =============================================================================
# Week 31-32 Enhancements: Advanced Distributed Features
# =============================================================================

def distributed_optimization(
    objective: Callable,
    parameter_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 100,
    cluster: Optional[DaskCluster] = None,
    method: str = "random"
) -> Tuple[Dict[str, float], float]:
    """Distributed hyperparameter optimization.

    Args:
        objective: Function to minimize (takes dict of parameters)
        parameter_ranges: Dict mapping parameter name to (min, max)
        n_samples: Number of samples to evaluate
        cluster: Dask cluster (None = create local)
        method: Sampling method ("random", "grid", "latin")

    Returns:
        Tuple of (best_params, best_value)
    """
    if cluster is None:
        cluster = create_local_cluster()
        owns_cluster = True
    else:
        owns_cluster = False

    try:
        # Generate parameter samples
        param_names = list(parameter_ranges.keys())

        if method == "random":
            # Random sampling
            samples = []
            for _ in range(n_samples):
                sample = {
                    name: np.random.uniform(parameter_ranges[name][0],
                                           parameter_ranges[name][1])
                    for name in param_names
                }
                samples.append(sample)

        elif method == "grid":
            # Grid sampling
            from itertools import product
            n_per_dim = int(np.ceil(n_samples ** (1.0 / len(param_names))))
            grids = [
                np.linspace(parameter_ranges[name][0],
                           parameter_ranges[name][1],
                           n_per_dim)
                for name in param_names
            ]
            samples = [
                dict(zip(param_names, combo))
                for combo in product(*grids)
            ][:n_samples]

        else:  # latin hypercube
            try:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=len(param_names))
                unit_samples = sampler.random(n=n_samples)

                samples = []
                for unit_sample in unit_samples:
                    sample = {
                        param_names[i]: parameter_ranges[param_names[i]][0] +
                                       unit_sample[i] * (parameter_ranges[param_names[i]][1] -
                                                        parameter_ranges[param_names[i]][0])
                        for i in range(len(param_names))
                    }
                    samples.append(sample)
            except ImportError:
                # Fall back to random
                samples = []
                for _ in range(n_samples):
                    sample = {
                        name: np.random.uniform(parameter_ranges[name][0],
                                               parameter_ranges[name][1])
                        for name in param_names
                    }
                    samples.append(sample)

        # Evaluate in parallel
        futures = [cluster.submit(objective, sample) for sample in samples]
        results = cluster.gather(futures)

        # Find best
        best_idx = np.argmin(results)
        best_params = samples[best_idx]
        best_value = results[best_idx]

        return best_params, best_value

    finally:
        if owns_cluster:
            cluster.close()


def pipeline(
    stages: List[Callable],
    initial_data: Any,
    cluster: Optional[DaskCluster] = None,
    persist_intermediate: bool = False
) -> Any:
    """Execute data processing pipeline with Dask.

    Args:
        stages: List of functions to apply sequentially
        initial_data: Input data
        cluster: Dask cluster
        persist_intermediate: Whether to persist intermediate results

    Returns:
        Final processed data
    """
    if cluster is None:
        cluster = create_local_cluster()
        owns_cluster = True
    else:
        owns_cluster = False

    try:
        # Wrap initial data as delayed
        if not isinstance(initial_data, dask.delayed.Delayed):
            data = dask.delayed(lambda x: x)(initial_data)
        else:
            data = initial_data

        # Apply stages
        for i, stage in enumerate(stages):
            data = dask.delayed(stage)(data)

            if persist_intermediate:
                # Compute and persist
                data = cluster.client.persist(data)

        # Compute final result
        result = data.compute()

        return result

    finally:
        if owns_cluster:
            cluster.close()


def distributed_cross_validation(
    model_fn: Callable,
    train_fn: Callable,
    evaluate_fn: Callable,
    data: Any,
    n_folds: int = 5,
    cluster: Optional[DaskCluster] = None
) -> Dict[str, Any]:
    """Distributed k-fold cross-validation.

    Args:
        model_fn: Function that creates a new model instance
        train_fn: Function(model, train_data) -> trained_model
        evaluate_fn: Function(model, test_data) -> score
        data: Full dataset (will be split into folds)
        n_folds: Number of CV folds
        cluster: Dask cluster

    Returns:
        Dict with scores, mean, std
    """
    if cluster is None:
        cluster = create_local_cluster()
        owns_cluster = True
    else:
        owns_cluster = False

    try:
        # Create folds
        n_samples = len(data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_size = n_samples // n_folds

        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else n_samples
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_idx, test_idx))

        # Define CV job for single fold
        def cv_fold(fold_data):
            train_idx, test_idx = fold_data
            train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            model = model_fn()
            model = train_fn(model, train_data)
            score = evaluate_fn(model, test_data)
            return score

        # Execute in parallel
        futures = [cluster.submit(cv_fold, fold) for fold in folds]
        scores = cluster.gather(futures)

        results = {
            "scores": scores,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "n_folds": n_folds
        }

        return results

    finally:
        if owns_cluster:
            cluster.close()


def scatter_gather_reduction(
    data: List[Any],
    map_fn: Callable,
    reduce_fn: Callable,
    cluster: Optional[DaskCluster] = None
) -> Any:
    """MapReduce pattern with scatter/gather.

    Args:
        data: Input data list
        map_fn: Function to apply to each element
        reduce_fn: Function to reduce results (takes list, returns single value)
        cluster: Dask cluster

    Returns:
        Reduced result
    """
    if cluster is None:
        cluster = create_local_cluster()
        owns_cluster = True
    else:
        owns_cluster = False

    try:
        # Scatter data to workers
        scattered = cluster.scatter(data)

        # Map phase
        mapped_futures = [cluster.submit(map_fn, item) for item in scattered]

        # Gather mapped results
        mapped_results = cluster.gather(mapped_futures)

        # Reduce phase (locally, as it's typically small)
        final_result = reduce_fn(mapped_results)

        return final_result

    finally:
        if owns_cluster:
            cluster.close()


def checkpoint_computation(
    computation_fn: Callable,
    checkpoint_path: Union[str, Path],
    cluster: Optional[DaskCluster] = None,
    force_recompute: bool = False
) -> Any:
    """Computation with checkpointing for fault tolerance.

    Args:
        computation_fn: Function to execute
        checkpoint_path: Path to save/load checkpoint
        cluster: Dask cluster
        force_recompute: Ignore existing checkpoint

    Returns:
        Computation result
    """
    checkpoint_path = Path(checkpoint_path)

    # Check for existing checkpoint
    if checkpoint_path.exists() and not force_recompute:
        import pickle
        with open(checkpoint_path, 'rb') as f:
            result = pickle.load(f)
        return result

    # Compute
    if cluster is None:
        result = computation_fn()
    else:
        future = cluster.submit(computation_fn)
        result = future.result()

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(result, f)

    return result
