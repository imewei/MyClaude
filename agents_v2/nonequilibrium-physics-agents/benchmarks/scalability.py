"""
Scalability Benchmarking Suite

Tests parallel and distributed scaling performance:
- Strong scaling: Fixed problem size, varying number of workers
- Weak scaling: Problem size scales with number of workers
- Distributed execution efficiency
- Network overhead analysis

Author: Nonequilibrium Physics Agents
Date: 2025-10-01
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Check for Dask availability
try:
    import dask
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Distributed scaling tests will be skipped.")

from ml_optimal_control.performance import Timer


@dataclass
class ScalingResult:
    """Results from a scaling benchmark.

    Attributes
    ----------
    test_name : str
        Name of the scaling test
    num_workers : int
        Number of parallel workers
    problem_size : int
        Total problem size
    execution_time : float
        Wall-clock execution time
    speedup : float
        Speedup relative to serial execution
    efficiency : float
        Parallel efficiency (speedup / num_workers)
    overhead : float
        Estimated parallel overhead in seconds
    """
    test_name: str
    num_workers: int
    problem_size: int
    execution_time: float
    speedup: float
    efficiency: float
    overhead: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'num_workers': self.num_workers,
            'problem_size': self.problem_size,
            'execution_time': self.execution_time,
            'speedup': self.speedup,
            'efficiency': self.efficiency,
            'overhead': self.overhead
        }


class StrongScalingBenchmark:
    """Strong scaling benchmark.

    Fixed problem size, varying number of workers.
    Ideal: linear speedup (speedup = num_workers).

    Parameters
    ----------
    problem_size : int
        Total number of tasks (fixed)
    task_duration : float
        Approximate duration of each task in seconds
    """

    def __init__(self, problem_size: int = 1000, task_duration: float = 0.01):
        self.problem_size = problem_size
        self.task_duration = task_duration

    def compute_task(self, task_id: int) -> float:
        """Simulate a computational task.

        Parameters
        ----------
        task_id : int
            Task identifier

        Returns
        -------
        float
            Task result
        """
        # Simulate computation with sleep + work
        time.sleep(self.task_duration * 0.1)  # Shorter for benchmark speed

        # Some actual computation
        result = 0.0
        for i in range(100):
            result += np.sin(task_id + i) * np.cos(task_id - i)

        return result

    def run_serial(self) -> float:
        """Run serial baseline.

        Returns
        -------
        float
            Execution time
        """
        timer = Timer()
        timer.start()

        for task_id in range(self.problem_size):
            self.compute_task(task_id)

        return timer.stop()

    def run_parallel(self, num_workers: int) -> float:
        """Run parallel execution.

        Parameters
        ----------
        num_workers : int
            Number of parallel workers

        Returns
        -------
        float
            Execution time
        """
        if not DASK_AVAILABLE:
            warnings.warn("Dask not available, using serial execution")
            return self.run_serial()

        from hpc.distributed import create_local_cluster, distribute_computation

        cluster = create_local_cluster(n_workers=num_workers)

        try:
            timer = Timer()
            timer.start()

            tasks = list(range(self.problem_size))
            results = distribute_computation(self.compute_task, tasks, cluster)

            elapsed = timer.stop()
        finally:
            cluster.close()

        return elapsed

    def run_benchmark(self, worker_counts: List[int] = None) -> List[ScalingResult]:
        """Run strong scaling benchmark.

        Parameters
        ----------
        worker_counts : List[int], optional
            List of worker counts to test, defaults to [1, 2, 4, 8]

        Returns
        -------
        List[ScalingResult]
            Scaling results for each worker count
        """
        if worker_counts is None:
            worker_counts = [1, 2, 4, 8]

        print("\n" + "=" * 80)
        print("STRONG SCALING BENCHMARK")
        print("=" * 80)
        print(f"Fixed problem size: {self.problem_size} tasks")
        print()

        # Serial baseline
        print("Running serial baseline...")
        serial_time = self.run_serial()
        print(f"  Serial time: {serial_time:.4f}s")

        results = []

        for num_workers in worker_counts:
            print(f"\nRunning with {num_workers} workers...")

            if num_workers == 1:
                parallel_time = serial_time
            else:
                parallel_time = self.run_parallel(num_workers)

            speedup = serial_time / parallel_time if parallel_time > 0 else 0.0
            efficiency = speedup / num_workers if num_workers > 0 else 0.0

            # Estimate overhead
            ideal_time = serial_time / num_workers
            overhead = parallel_time - ideal_time

            result = ScalingResult(
                test_name="StrongScaling",
                num_workers=num_workers,
                problem_size=self.problem_size,
                execution_time=parallel_time,
                speedup=speedup,
                efficiency=efficiency,
                overhead=max(0, overhead)
            )

            results.append(result)

            print(f"  Time: {parallel_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Efficiency: {efficiency*100:.1f}%")

        return results


class WeakScalingBenchmark:
    """Weak scaling benchmark.

    Problem size scales with number of workers (constant work per worker).
    Ideal: constant execution time as workers increase.

    Parameters
    ----------
    tasks_per_worker : int
        Number of tasks per worker (constant)
    task_duration : float
        Approximate duration of each task
    """

    def __init__(self, tasks_per_worker: int = 100, task_duration: float = 0.01):
        self.tasks_per_worker = tasks_per_worker
        self.task_duration = task_duration

    def compute_task(self, task_id: int) -> float:
        """Simulate a computational task.

        Parameters
        ----------
        task_id : int
            Task identifier

        Returns
        -------
        float
            Task result
        """
        time.sleep(self.task_duration * 0.1)

        result = 0.0
        for i in range(100):
            result += np.sin(task_id + i) * np.cos(task_id - i)

        return result

    def run_benchmark(self, worker_counts: List[int] = None) -> List[ScalingResult]:
        """Run weak scaling benchmark.

        Parameters
        ----------
        worker_counts : List[int], optional
            List of worker counts to test

        Returns
        -------
        List[ScalingResult]
            Scaling results
        """
        if worker_counts is None:
            worker_counts = [1, 2, 4, 8]

        print("\n" + "=" * 80)
        print("WEAK SCALING BENCHMARK")
        print("=" * 80)
        print(f"Tasks per worker: {self.tasks_per_worker}")
        print()

        results = []
        baseline_time = None

        for num_workers in worker_counts:
            problem_size = self.tasks_per_worker * num_workers

            print(f"\nRunning with {num_workers} workers ({problem_size} total tasks)...")

            if num_workers == 1 or not DASK_AVAILABLE:
                # Serial execution
                timer = Timer()
                timer.start()

                for task_id in range(problem_size):
                    self.compute_task(task_id)

                elapsed = timer.stop()
            else:
                # Parallel execution
                from hpc.distributed import create_local_cluster, distribute_computation

                cluster = create_local_cluster(n_workers=num_workers)

                try:
                    timer = Timer()
                    timer.start()

                    tasks = list(range(problem_size))
                    results_data = distribute_computation(self.compute_task, tasks, cluster)

                    elapsed = timer.stop()
                finally:
                    cluster.close()

            if baseline_time is None:
                baseline_time = elapsed

            # For weak scaling, efficiency is baseline_time / current_time
            efficiency = baseline_time / elapsed if elapsed > 0 else 0.0

            result = ScalingResult(
                test_name="WeakScaling",
                num_workers=num_workers,
                problem_size=problem_size,
                execution_time=elapsed,
                speedup=1.0,  # Not applicable for weak scaling
                efficiency=efficiency,
                overhead=elapsed - baseline_time
            )

            results.append(result)

            print(f"  Time: {elapsed:.4f}s")
            print(f"  Efficiency: {efficiency*100:.1f}%")
            print(f"  Overhead: {elapsed - baseline_time:.4f}s")

        return results


class NetworkOverheadBenchmark:
    """Benchmark network communication overhead.

    Measures overhead of data transfer in distributed execution.

    Parameters
    ----------
    data_sizes : List[int]
        List of data sizes to test (number of elements)
    """

    def __init__(self, data_sizes: List[int] = None):
        if data_sizes is None:
            self.data_sizes = [100, 1000, 10000, 100000]
        else:
            self.data_sizes = data_sizes

    def run_benchmark(self) -> List[Dict]:
        """Run network overhead benchmark.

        Returns
        -------
        List[Dict]
            Results for each data size
        """
        if not DASK_AVAILABLE:
            print("Dask not available, skipping network overhead benchmark")
            return []

        from hpc.distributed import create_local_cluster

        print("\n" + "=" * 80)
        print("NETWORK OVERHEAD BENCHMARK")
        print("=" * 80)
        print()

        results = []

        cluster = create_local_cluster(n_workers=2)

        try:
            for size in self.data_sizes:
                print(f"Testing data size: {size} elements...")

                # Create test data
                data = np.random.randn(size)

                # Measure scatter-compute-gather time
                timer = Timer()
                timer.start()

                future = cluster.client.scatter(data)
                result_future = cluster.client.submit(np.sum, future)
                result = result_future.result()

                elapsed = timer.stop()

                # Estimate data transfer time
                # Assume computation is negligible for sum
                data_bytes = data.nbytes
                throughput = data_bytes / elapsed if elapsed > 0 else 0

                results.append({
                    'data_size': size,
                    'data_bytes': data_bytes,
                    'transfer_time': elapsed,
                    'throughput_MB_s': throughput / 1e6
                })

                print(f"  Time: {elapsed:.6f}s")
                print(f"  Throughput: {throughput/1e6:.2f} MB/s")

        finally:
            cluster.close()

        return results


def run_scalability_suite() -> Dict[str, List]:
    """Run complete scalability benchmark suite.

    Returns
    -------
    Dict[str, List]
        All scalability results
    """
    results = {}

    # Strong scaling
    print("\nRunning Strong Scaling Benchmark...")
    strong = StrongScalingBenchmark(problem_size=100, task_duration=0.01)
    results['strong_scaling'] = strong.run_benchmark([1, 2, 4])

    # Weak scaling
    print("\nRunning Weak Scaling Benchmark...")
    weak = WeakScalingBenchmark(tasks_per_worker=50, task_duration=0.01)
    results['weak_scaling'] = weak.run_benchmark([1, 2, 4])

    # Network overhead (if Dask available)
    if DASK_AVAILABLE:
        print("\nRunning Network Overhead Benchmark...")
        network = NetworkOverheadBenchmark(data_sizes=[1000, 10000, 100000])
        results['network_overhead'] = network.run_benchmark()

    return results


def print_scalability_summary(results: Dict[str, List]) -> None:
    """Print summary of scalability results.

    Parameters
    ----------
    results : Dict[str, List]
        Scalability benchmark results
    """
    print("\n" + "=" * 80)
    print("SCALABILITY BENCHMARK SUMMARY")
    print("=" * 80)

    if 'strong_scaling' in results:
        print("\nStrong Scaling (Fixed Problem Size):")
        print("-" * 80)
        print(f"{'Workers':>10s} {'Time (s)':>12s} {'Speedup':>12s} {'Efficiency':>12s}")
        print("-" * 80)

        for result in results['strong_scaling']:
            print(f"{result.num_workers:10d} {result.execution_time:12.4f} "
                  f"{result.speedup:12.2f}x {result.efficiency*100:11.1f}%")

    if 'weak_scaling' in results:
        print("\nWeak Scaling (Constant Work Per Worker):")
        print("-" * 80)
        print(f"{'Workers':>10s} {'Problem Size':>15s} {'Time (s)':>12s} {'Efficiency':>12s}")
        print("-" * 80)

        for result in results['weak_scaling']:
            print(f"{result.num_workers:10d} {result.problem_size:15d} "
                  f"{result.execution_time:12.4f} {result.efficiency*100:11.1f}%")

    if 'network_overhead' in results and results['network_overhead']:
        print("\nNetwork Communication Overhead:")
        print("-" * 80)
        print(f"{'Data Size':>12s} {'Data (MB)':>12s} {'Time (s)':>12s} {'Throughput':>15s}")
        print("-" * 80)

        for result in results['network_overhead']:
            print(f"{result['data_size']:12d} {result['data_bytes']/1e6:12.2f} "
                  f"{result['transfer_time']:12.6f} {result['throughput_MB_s']:12.2f} MB/s")


if __name__ == "__main__":
    results = run_scalability_suite()
    print_scalability_summary(results)
