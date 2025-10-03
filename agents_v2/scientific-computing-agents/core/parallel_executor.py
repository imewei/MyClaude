"""
Parallel Execution Framework for Scientific Computing Agents.

This module provides parallel execution capabilities with support for:
- Thread-based parallelism (I/O-bound tasks)
- Process-based parallelism (CPU-bound tasks)
- Async/await patterns (I/O-bound with async support)
- Dependency graph resolution
- Error handling and result aggregation
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
from queue import Queue
import threading


class ParallelMode(Enum):
    """Parallel execution modes."""
    THREADS = "threads"  # For I/O-bound tasks
    PROCESSES = "processes"  # For CPU-bound tasks
    ASYNC = "async"  # For async I/O operations


@dataclass
class Task:
    """Represents a task to be executed."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"TaskResult(id={self.task_id}, status={status}, time={self.execution_time:.3f}s)"


class DependencyGraph:
    """Manages task dependencies and execution order."""

    def __init__(self, tasks: List[Task]):
        self.tasks = {task.task_id: task for task in tasks}
        self._build_graph()

    def _build_graph(self):
        """Build dependency graph."""
        self.dependents = {task_id: [] for task_id in self.tasks}
        self.dependencies_count = {task_id: 0 for task_id in self.tasks}

        for task_id, task in self.tasks.items():
            self.dependencies_count[task_id] = len(task.dependencies)
            for dep in task.dependencies:
                if dep in self.dependents:
                    self.dependents[dep].append(task_id)

    def get_ready_tasks(self, completed: set) -> List[str]:
        """Get tasks that are ready to execute (all dependencies satisfied)."""
        ready = []
        for task_id, task in self.tasks.items():
            if task_id not in completed:
                deps_satisfied = all(dep in completed for dep in task.dependencies)
                if deps_satisfied:
                    ready.append(task_id)
        return ready

    def has_cycles(self) -> bool:
        """Check if dependency graph has cycles."""
        visited = set()
        rec_stack = set()

        def has_cycle_util(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            for dependent in self.dependents.get(task_id, []):
                if dependent not in visited:
                    if has_cycle_util(dependent):
                        return True
                elif dependent in rec_stack:
                    return True

            rec_stack.remove(task_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle_util(task_id):
                    return True
        return False

    def topological_sort(self) -> List[List[str]]:
        """
        Return tasks in topological order, grouped by execution level.
        Tasks in the same level can be executed in parallel.
        """
        if self.has_cycles():
            raise ValueError("Dependency graph contains cycles")

        levels = []
        completed = set()

        while len(completed) < len(self.tasks):
            ready = self.get_ready_tasks(completed)
            if not ready:
                # This shouldn't happen if no cycles
                raise ValueError("Unable to resolve dependencies")
            levels.append(ready)
            completed.update(ready)

        return levels


class ParallelExecutor:
    """
    Executes tasks in parallel with dependency management.

    Supports multiple execution modes:
    - THREADS: For I/O-bound tasks (network, file I/O)
    - PROCESSES: For CPU-bound tasks (numerical computation)
    - ASYNC: For async I/O operations
    """

    def __init__(
        self,
        mode: ParallelMode = ParallelMode.THREADS,
        max_workers: Optional[int] = None
    ):
        """
        Initialize parallel executor.

        Args:
            mode: Execution mode (threads, processes, or async)
            max_workers: Maximum number of workers (default: CPU count)
        """
        self.mode = mode
        self.max_workers = max_workers or mp.cpu_count()

    def execute(self, tasks: List[Task]) -> Dict[str, TaskResult]:
        """
        Execute tasks in parallel, respecting dependencies.

        Args:
            tasks: List of tasks to execute

        Returns:
            Dictionary mapping task_id to TaskResult
        """
        if not tasks:
            return {}

        # Build dependency graph
        graph = DependencyGraph(tasks)

        # Check for cycles
        if graph.has_cycles():
            raise ValueError("Task dependencies contain cycles")

        # Get execution levels
        levels = graph.topological_sort()

        # Execute level by level
        results = {}
        for level_tasks in levels:
            level_results = self._execute_level(
                [graph.tasks[tid] for tid in level_tasks],
                results
            )
            results.update(level_results)

        return results

    def _execute_level(
        self,
        tasks: List[Task],
        previous_results: Dict[str, TaskResult]
    ) -> Dict[str, TaskResult]:
        """Execute all tasks in a level in parallel."""
        if self.mode == ParallelMode.THREADS:
            return self._execute_with_threads(tasks, previous_results)
        elif self.mode == ParallelMode.PROCESSES:
            return self._execute_with_processes(tasks, previous_results)
        elif self.mode == ParallelMode.ASYNC:
            return self._execute_with_async(tasks, previous_results)
        else:
            raise ValueError(f"Unknown execution mode: {self.mode}")

    def _execute_with_threads(
        self,
        tasks: List[Task],
        previous_results: Dict[str, TaskResult]
    ) -> Dict[str, TaskResult]:
        """Execute tasks using thread pool."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                # Inject dependency results into kwargs
                task_kwargs = task.kwargs.copy()
                task_kwargs['_dependency_results'] = {
                    dep: previous_results[dep] for dep in task.dependencies
                }

                future = executor.submit(
                    self._execute_task_wrapper,
                    task.function,
                    task.args,
                    task_kwargs,
                    task.task_id
                )
                future_to_task[future] = task

            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.task_id] = result
                except Exception as e:
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        success=False,
                        error=str(e)
                    )

        return results

    def _execute_with_processes(
        self,
        tasks: List[Task],
        previous_results: Dict[str, TaskResult]
    ) -> Dict[str, TaskResult]:
        """Execute tasks using process pool."""
        results = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {}
            for task in tasks:
                # For process pool, we can't easily inject dependency results
                # due to pickling constraints. Tasks should access shared state differently.
                future = executor.submit(
                    self._execute_task_wrapper,
                    task.function,
                    task.args,
                    task.kwargs,
                    task.task_id
                )
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.task_id] = result
                except Exception as e:
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        success=False,
                        error=str(e)
                    )

        return results

    def _execute_with_async(
        self,
        tasks: List[Task],
        previous_results: Dict[str, TaskResult]
    ) -> Dict[str, TaskResult]:
        """Execute tasks using asyncio."""
        # Create or get event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async tasks
        results = loop.run_until_complete(
            self._async_execute_tasks(tasks, previous_results)
        )
        return results

    async def _async_execute_tasks(
        self,
        tasks: List[Task],
        previous_results: Dict[str, TaskResult]
    ) -> Dict[str, TaskResult]:
        """Execute tasks asynchronously."""
        async_tasks = []
        for task in tasks:
            task_kwargs = task.kwargs.copy()
            task_kwargs['_dependency_results'] = {
                dep: previous_results[dep] for dep in task.dependencies
            }
            async_tasks.append(
                self._async_execute_task(
                    task.function,
                    task.args,
                    task_kwargs,
                    task.task_id
                )
            )

        results_list = await asyncio.gather(*async_tasks, return_exceptions=True)

        results = {}
        for i, result in enumerate(results_list):
            task_id = tasks[i].task_id
            if isinstance(result, Exception):
                results[task_id] = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(result)
                )
            else:
                results[task_id] = result

        return results

    @staticmethod
    def _execute_task_wrapper(
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
        task_id: str
    ) -> TaskResult:
        """Wrapper for executing a task with timing."""
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=elapsed
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=elapsed
            )

    @staticmethod
    async def _async_execute_task(
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
        task_id: str
    ) -> TaskResult:
        """Async wrapper for executing a task."""
        start_time = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)

            elapsed = time.perf_counter() - start_time
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=elapsed
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=elapsed
            )


def execute_parallel(
    functions: List[Callable],
    args_list: Optional[List[Tuple]] = None,
    kwargs_list: Optional[List[Dict]] = None,
    mode: ParallelMode = ParallelMode.THREADS,
    max_workers: Optional[int] = None
) -> List[TaskResult]:
    """
    Simple parallel execution of independent functions.

    Args:
        functions: List of functions to execute
        args_list: List of argument tuples for each function
        kwargs_list: List of keyword argument dicts for each function
        mode: Execution mode
        max_workers: Maximum number of workers

    Returns:
        List of TaskResult objects

    Example:
        def task1(x): return x ** 2
        def task2(x): return x ** 3

        results = execute_parallel(
            [task1, task2],
            args_list=[(2,), (3,)],
            mode=ParallelMode.THREADS
        )
    """
    if args_list is None:
        args_list = [() for _ in functions]
    if kwargs_list is None:
        kwargs_list = [{} for _ in functions]

    # Create tasks
    tasks = []
    for i, func in enumerate(functions):
        tasks.append(Task(
            task_id=f"task_{i}",
            function=func,
            args=args_list[i],
            kwargs=kwargs_list[i]
        ))

    # Execute
    executor = ParallelExecutor(mode=mode, max_workers=max_workers)
    results_dict = executor.execute(tasks)

    # Return in order
    return [results_dict[f"task_{i}"] for i in range(len(functions))]


if __name__ == "__main__":
    # Example usage
    import math

    print("=" * 70)
    print("Parallel Executor Demo")
    print("=" * 70)

    # Example 1: Simple independent tasks
    print("\n1. Independent Tasks (Thread Pool)")
    print("-" * 70)

    def compute_factorial(n):
        """Compute factorial."""
        time.sleep(0.1)  # Simulate I/O
        return math.factorial(n)

    def compute_fibonacci(n):
        """Compute Fibonacci."""
        time.sleep(0.1)  # Simulate I/O
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    def compute_power(base, exp):
        """Compute power."""
        time.sleep(0.1)  # Simulate I/O
        return base ** exp

    results = execute_parallel(
        [compute_factorial, compute_fibonacci, compute_power],
        args_list=[(10,), (20,), (2, 10)],
        mode=ParallelMode.THREADS
    )

    for result in results:
        print(f"{result.task_id}: {result.result} (time: {result.execution_time:.3f}s)")

    # Example 2: Tasks with dependencies
    print("\n2. Tasks with Dependencies")
    print("-" * 70)

    def task_a():
        """First task."""
        time.sleep(0.2)
        return "A complete"

    def task_b():
        """Second task (independent of A)."""
        time.sleep(0.2)
        return "B complete"

    def task_c(_dependency_results=None):
        """Third task (depends on A and B)."""
        deps = _dependency_results or {}
        time.sleep(0.1)
        return f"C complete (after {list(deps.keys())})"

    tasks = [
        Task(task_id="A", function=task_a),
        Task(task_id="B", function=task_b),
        Task(task_id="C", function=task_c, dependencies=["A", "B"])
    ]

    executor = ParallelExecutor(mode=ParallelMode.THREADS)
    results = executor.execute(tasks)

    for task_id in ["A", "B", "C"]:
        result = results[task_id]
        print(f"{task_id}: {result.result} (time: {result.execution_time:.3f}s)")

    print(f"\nTotal time: {sum(r.execution_time for r in results.values()):.3f}s")
    print("(Note: A and B ran in parallel)")

    print("\n" + "=" * 70)
