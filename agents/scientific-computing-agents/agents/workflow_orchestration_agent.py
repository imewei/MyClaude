"""
Workflow Orchestration Agent - Coordinate multi-agent workflows with parallel execution.

This agent manages complex workflows involving multiple agents, with support for:
- Sequential execution
- Parallel execution (threads/processes)
- Dependency management
- Result aggregation
- Error handling and recovery
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parallel_executor import (
    ParallelExecutor,
    ParallelMode,
    Task,
    TaskResult,
    DependencyGraph
)


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""
    step_id: str
    agent: Any  # Agent instance
    method: str  # Method name to call
    inputs: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    steps_completed: List[str]
    results: Dict[str, Any]
    total_time: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"WorkflowResult(status={status}, steps={len(self.steps_completed)}, time={self.total_time:.3f}s)"


class WorkflowOrchestrationAgent:
    """
    Agent for orchestrating multi-agent workflows with parallel execution.

    Features:
    - Sequential and parallel execution
    - Automatic dependency resolution
    - Result passing between steps
    - Error handling
    - Performance tracking
    """

    def __init__(
        self,
        parallel_mode: ParallelMode = ParallelMode.THREADS,
        max_workers: Optional[int] = None
    ):
        """
        Initialize workflow orchestration agent.

        Args:
            parallel_mode: Mode for parallel execution
            max_workers: Maximum number of parallel workers
        """
        self.parallel_mode = parallel_mode
        self.max_workers = max_workers
        self.executor = ParallelExecutor(mode=parallel_mode, max_workers=max_workers)

    def execute_workflow(
        self,
        steps: List[WorkflowStep],
        parallel: bool = True
    ) -> WorkflowResult:
        """
        Execute a workflow with multiple steps.

        Args:
            steps: List of workflow steps
            parallel: Whether to execute independent steps in parallel

        Returns:
            WorkflowResult with execution details
        """
        start_time = time.perf_counter()

        if not steps:
            return WorkflowResult(
                success=True,
                steps_completed=[],
                results={},
                total_time=0.0
            )

        try:
            if parallel:
                results_dict = self._execute_parallel(steps)
            else:
                results_dict = self._execute_sequential(steps)

            elapsed = time.perf_counter() - start_time

            # Check for failures
            failed_steps = [
                step_id for step_id, result in results_dict.items()
                if isinstance(result, TaskResult) and not result.success
            ]

            if failed_steps:
                errors = [
                    f"Step {step_id} failed: {results_dict[step_id].error}"
                    for step_id in failed_steps
                ]
                return WorkflowResult(
                    success=False,
                    steps_completed=[s for s in results_dict.keys() if s not in failed_steps],
                    results=results_dict,
                    total_time=elapsed,
                    errors=errors
                )

            return WorkflowResult(
                success=True,
                steps_completed=list(results_dict.keys()),
                results=results_dict,
                total_time=elapsed
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return WorkflowResult(
                success=False,
                steps_completed=[],
                results={},
                total_time=elapsed,
                errors=[f"Workflow execution failed: {str(e)}"]
            )

    def _execute_parallel(self, steps: List[WorkflowStep]) -> Dict[str, Any]:
        """Execute workflow steps in parallel where possible."""
        # Convert workflow steps to tasks
        tasks = []
        for step in steps:
            task = Task(
                task_id=step.step_id,
                function=self._execute_step_wrapper,
                args=(step,),
                dependencies=step.dependencies,
                metadata=step.metadata
            )
            tasks.append(task)

        # Execute with dependency resolution
        task_results = self.executor.execute(tasks)

        # Extract actual results from TaskResult objects
        results = {}
        for step_id, task_result in task_results.items():
            if task_result.success:
                results[step_id] = task_result.result
            else:
                results[step_id] = task_result  # Keep TaskResult for error info

        return results

    def _execute_sequential(self, steps: List[WorkflowStep]) -> Dict[str, Any]:
        """Execute workflow steps sequentially."""
        # Build dependency graph for ordering
        tasks = [
            Task(task_id=step.step_id, function=lambda: None, dependencies=step.dependencies)
            for step in steps
        ]
        graph = DependencyGraph(tasks)

        # Check for cycles
        if graph.has_cycles():
            raise ValueError("Workflow contains circular dependencies")

        # Get execution order
        levels = graph.topological_sort()
        execution_order = [step_id for level in levels for step_id in level]

        # Execute in order
        results = {}
        step_dict = {step.step_id: step for step in steps}

        for step_id in execution_order:
            step = step_dict[step_id]

            # Prepare inputs with dependency results
            inputs = step.inputs.copy()
            inputs['_dependency_results'] = {
                dep: results[dep] for dep in step.dependencies
            }

            # Execute step
            try:
                start = time.perf_counter()
                result = self._execute_step(step, inputs)
                elapsed = time.perf_counter() - start

                results[step_id] = result
            except Exception as e:
                results[step_id] = TaskResult(
                    task_id=step_id,
                    success=False,
                    error=str(e)
                )
                break  # Stop on error in sequential mode

        return results

    def _execute_step_wrapper(self, step: WorkflowStep, _dependency_results: Optional[Dict] = None) -> Any:
        """Wrapper for executing a workflow step."""
        # Prepare inputs
        inputs = step.inputs.copy()
        if _dependency_results:
            inputs['_dependency_results'] = _dependency_results

        return self._execute_step(step, inputs)

    @staticmethod
    def _execute_step(step: WorkflowStep, inputs: Dict[str, Any]) -> Any:
        """Execute a single workflow step."""
        # Get the method from the agent
        agent = step.agent
        method = getattr(agent, step.method)

        # Call the method
        result = method(inputs)
        return result

    def execute_agents_parallel(
        self,
        agents: List[Any],
        method_name: str,
        inputs_list: List[Dict[str, Any]]
    ) -> List[TaskResult]:
        """
        Execute the same method on multiple agents in parallel.

        Args:
            agents: List of agent instances
            method_name: Name of method to call on each agent
            inputs_list: List of input dicts for each agent

        Returns:
            List of TaskResult objects

        Example:
            # Solve multiple PDEs in parallel
            results = orchestrator.execute_agents_parallel(
                agents=[pde_agent, pde_agent, pde_agent],
                method_name='solve_pde_2d',
                inputs_list=[problem1, problem2, problem3]
            )
        """
        tasks = []
        for i, (agent, inputs) in enumerate(zip(agents, inputs_list)):
            # Wrapper to call method with inputs dict as single argument
            method = getattr(agent, method_name)
            task = Task(
                task_id=f"agent_{i}",
                function=method,
                args=(inputs,)  # Pass inputs as single positional argument
            )
            tasks.append(task)

        results = self.executor.execute(tasks)
        return [results[f"agent_{i}"] for i in range(len(agents))]


def create_simple_workflow(
    agent_methods: List[tuple],  # [(agent, method_name, inputs), ...]
    dependencies: Optional[List[List[int]]] = None
) -> List[WorkflowStep]:
    """
    Helper to create a simple workflow from a list of agent methods.

    Args:
        agent_methods: List of (agent, method_name, inputs) tuples
        dependencies: List of dependency indices for each step

    Returns:
        List of WorkflowStep objects

    Example:
        steps = create_simple_workflow([
            (problem_analyzer, 'analyze', {'problem': data}),
            (algorithm_selector, 'select', {}),
            (executor, 'execute', {})
        ], dependencies=[[], [0], [0, 1]])
    """
    if dependencies is None:
        dependencies = [[] for _ in agent_methods]

    steps = []
    for i, ((agent, method, inputs), deps) in enumerate(zip(agent_methods, dependencies)):
        step = WorkflowStep(
            step_id=f"step_{i}",
            agent=agent,
            method=method,
            inputs=inputs,
            dependencies=[f"step_{d}" for d in deps]
        )
        steps.append(step)

    return steps


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Workflow Orchestration Agent Demo")
    print("=" * 70)

    # Mock agent for demonstration
    class MockAgent:
        def __init__(self, name):
            self.name = name

        def process(self, data):
            """Mock processing method."""
            import time
            time.sleep(0.2)  # Simulate work
            value = data.get('value', 0)
            return {'result': value * 2, 'agent': self.name}

    # Create orchestrator
    orchestrator = WorkflowOrchestrationAgent(
        parallel_mode=ParallelMode.THREADS,
        max_workers=4
    )

    # Example 1: Independent parallel execution
    print("\n1. Parallel Agent Execution (Independent)")
    print("-" * 70)

    agents = [MockAgent(f"Agent_{i}") for i in range(4)]
    inputs_list = [{'value': i * 10} for i in range(4)]

    start = time.perf_counter()
    results = orchestrator.execute_agents_parallel(
        agents=agents,
        method_name='process',
        inputs_list=inputs_list
    )
    elapsed_parallel = time.perf_counter() - start

    print("Parallel Results:")
    for result in results:
        print(f"  {result}")
    print(f"Total time (parallel): {elapsed_parallel:.3f}s")

    # Example 2: Workflow with dependencies
    print("\n2. Workflow with Dependencies")
    print("-" * 70)

    agent_a = MockAgent("A")
    agent_b = MockAgent("B")
    agent_c = MockAgent("C")

    steps = [
        WorkflowStep(
            step_id="step_1",
            agent=agent_a,
            method="process",
            inputs={'value': 10}
        ),
        WorkflowStep(
            step_id="step_2",
            agent=agent_b,
            method="process",
            inputs={'value': 20}
        ),
        WorkflowStep(
            step_id="step_3",
            agent=agent_c,
            method="process",
            inputs={'value': 5},
            dependencies=["step_1", "step_2"]  # Depends on both
        )
    ]

    workflow_result = orchestrator.execute_workflow(steps, parallel=True)
    print(workflow_result)
    print(f"\nSteps completed: {workflow_result.steps_completed}")
    print(f"Total time: {workflow_result.total_time:.3f}s")
    print("(Note: step_1 and step_2 ran in parallel)")

    print("\n" + "=" * 70)
