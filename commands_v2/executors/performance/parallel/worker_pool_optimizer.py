#!/usr/bin/env python3
"""
Worker Pool Optimizer
=====================

Calculates optimal worker counts for parallel execution.

Features:
- CPU vs I/O bound task detection
- Optimal thread/process pool sizing
- Diminishing returns analysis
- Load-based dynamic adjustment

Author: Claude Code Framework
Version: 2.0
"""

import psutil
import logging
import multiprocessing as mp
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Task type classification"""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class WorkerPoolConfig:
    """Worker pool configuration"""
    max_workers: int
    thread_pool_size: int
    process_pool_size: int
    queue_size: int
    task_type: TaskType
    use_threads: bool
    use_processes: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_workers": self.max_workers,
            "thread_pool_size": self.thread_pool_size,
            "process_pool_size": self.process_pool_size,
            "queue_size": self.queue_size,
            "task_type": self.task_type.value,
            "use_threads": self.use_threads,
            "use_processes": self.use_processes
        }


class WorkerPoolOptimizer:
    """
    Optimizes worker pool configuration for parallel execution.

    Determines:
    - Optimal number of workers
    - Thread vs process pool usage
    - Queue sizes
    - Load balancing parameters
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cpu_count = mp.cpu_count()

    def optimize(
        self,
        task_type: Optional[TaskType] = None,
        avg_task_duration_ms: Optional[float] = None,
        memory_per_task_mb: Optional[float] = None
    ) -> WorkerPoolConfig:
        """
        Optimize worker pool configuration.

        Args:
            task_type: Type of tasks (auto-detected if None)
            avg_task_duration_ms: Average task duration
            memory_per_task_mb: Memory per task

        Returns:
            Optimized worker pool configuration
        """
        self.logger.info("Optimizing worker pool configuration")

        # Detect task type if not provided
        if task_type is None:
            task_type = self._detect_task_type(
                avg_task_duration_ms,
                memory_per_task_mb
            )

        self.logger.info(f"Task type: {task_type.value}")

        # Calculate optimal worker counts
        config = self._calculate_optimal_workers(
            task_type,
            avg_task_duration_ms,
            memory_per_task_mb
        )

        self.logger.info(
            f"Optimal config: {config.max_workers} workers, "
            f"threads={config.thread_pool_size}, "
            f"processes={config.process_pool_size}"
        )

        return config

    def _detect_task_type(
        self,
        avg_duration_ms: Optional[float],
        memory_per_task_mb: Optional[float]
    ) -> TaskType:
        """Detect task type from characteristics"""

        # Use heuristics to classify
        if avg_duration_ms is None or avg_duration_ms < 10:
            # Very fast tasks are likely I/O bound
            return TaskType.IO_BOUND

        if memory_per_task_mb and memory_per_task_mb > 100:
            # Memory-intensive tasks
            return TaskType.CPU_BOUND

        # Default to mixed
        return TaskType.MIXED

    def _calculate_optimal_workers(
        self,
        task_type: TaskType,
        avg_duration_ms: Optional[float],
        memory_per_task_mb: Optional[float]
    ) -> WorkerPoolConfig:
        """Calculate optimal worker configuration"""

        # Get system resources
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)

        # Base calculations
        if task_type == TaskType.CPU_BOUND:
            # CPU-bound: workers = CPU count * (1.0 to 1.5)
            max_workers = int(self.cpu_count * 1.0)
            thread_pool = self.cpu_count
            process_pool = self.cpu_count
            use_threads = False
            use_processes = True

        elif task_type == TaskType.IO_BOUND:
            # I/O-bound: workers = CPU count * (2.0 to 4.0)
            max_workers = int(self.cpu_count * 3.0)
            thread_pool = self.cpu_count * 4
            process_pool = 0
            use_threads = True
            use_processes = False

        else:  # MIXED or UNKNOWN
            # Mixed: workers = CPU count * 2.0
            max_workers = int(self.cpu_count * 2.0)
            thread_pool = self.cpu_count * 2
            process_pool = self.cpu_count
            use_threads = True
            use_processes = True

        # Adjust for memory constraints
        if memory_per_task_mb:
            # Ensure we don't exceed available memory
            max_memory_workers = int(
                (available_memory_gb * 1024 * 0.7) / memory_per_task_mb
            )
            if max_memory_workers < max_workers:
                self.logger.warning(
                    f"Reducing workers from {max_workers} to {max_memory_workers} "
                    "due to memory constraints"
                )
                max_workers = max_memory_workers
                thread_pool = min(thread_pool, max_workers)
                process_pool = min(process_pool, max_workers)

        # Calculate queue size (2x workers)
        queue_size = max_workers * 2

        # Apply limits
        max_workers = max(2, min(max_workers, 64))
        thread_pool = max(2, min(thread_pool, 128))
        process_pool = max(0, min(process_pool, self.cpu_count * 2))
        queue_size = max(100, min(queue_size, 10000))

        return WorkerPoolConfig(
            max_workers=max_workers,
            thread_pool_size=thread_pool,
            process_pool_size=process_pool,
            queue_size=queue_size,
            task_type=task_type,
            use_threads=use_threads,
            use_processes=use_processes
        )

    def calculate_speedup(
        self,
        num_workers: int,
        task_type: TaskType,
        overhead_ms: float = 5.0
    ) -> float:
        """
        Calculate expected speedup with given workers.

        Args:
            num_workers: Number of workers
            task_type: Task type
            overhead_ms: Parallelization overhead

        Returns:
            Expected speedup factor
        """
        # Amdahl's law with overhead
        if task_type == TaskType.CPU_BOUND:
            # CPU-bound has high parallelization potential
            parallel_fraction = 0.95
        elif task_type == TaskType.IO_BOUND:
            # I/O-bound has very high parallelization potential
            parallel_fraction = 0.98
        else:
            # Mixed workload
            parallel_fraction = 0.90

        # Calculate speedup
        serial_fraction = 1.0 - parallel_fraction
        theoretical_speedup = 1.0 / (serial_fraction + parallel_fraction / num_workers)

        # Apply overhead penalty
        overhead_factor = 1.0 - (overhead_ms / 100.0) * (num_workers / self.cpu_count)
        actual_speedup = theoretical_speedup * overhead_factor

        return max(1.0, actual_speedup)

    def find_optimal_worker_count(
        self,
        task_type: TaskType,
        max_workers: int = 64
    ) -> Tuple[int, float]:
        """
        Find optimal worker count with diminishing returns analysis.

        Args:
            task_type: Task type
            max_workers: Maximum workers to test

        Returns:
            Tuple of (optimal_workers, speedup)
        """
        best_efficiency = 0.0
        best_workers = 1
        best_speedup = 1.0

        for workers in range(1, min(max_workers, 65)):
            speedup = self.calculate_speedup(workers, task_type)
            efficiency = speedup / workers  # Efficiency = speedup per worker

            # Stop when efficiency drops below 70%
            if efficiency < 0.7 and workers > self.cpu_count:
                break

            if speedup > best_speedup:
                best_speedup = speedup
                best_workers = workers
                best_efficiency = efficiency

        self.logger.info(
            f"Optimal workers: {best_workers} (speedup: {best_speedup:.2f}x, "
            f"efficiency: {best_efficiency:.2f})"
        )

        return best_workers, best_speedup


def main():
    """Test worker pool optimizer"""
    logging.basicConfig(level=logging.INFO)

    optimizer = WorkerPoolOptimizer()

    # Test different task types
    for task_type in [TaskType.CPU_BOUND, TaskType.IO_BOUND, TaskType.MIXED]:
        print(f"\n{task_type.value.upper()}")
        print("=" * 60)

        config = optimizer.optimize(task_type=task_type)

        print(f"Max Workers: {config.max_workers}")
        print(f"Thread Pool: {config.thread_pool_size}")
        print(f"Process Pool: {config.process_pool_size}")
        print(f"Queue Size: {config.queue_size}")
        print(f"Use Threads: {config.use_threads}")
        print(f"Use Processes: {config.use_processes}")

        # Calculate speedup
        optimal, speedup = optimizer.find_optimal_worker_count(task_type)
        print(f"\nOptimal Workers (diminishing returns): {optimal}")
        print(f"Expected Speedup: {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())