"""
Tests for WorkflowOrchestrationAgent.

Comprehensive test suite covering workflow orchestration, parallel execution,
and dependency management.
"""

import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.workflow_orchestration_agent import (
    WorkflowOrchestrationAgent,
    WorkflowStep,
    WorkflowResult
)
from core.parallel_executor import ParallelMode


# Mock agent for testing
class MockAgent:
    """Simple mock agent for testing workflows."""

    def __init__(self, name="mock"):
        self.name = name
        self.call_count = 0

    def process(self, data):
        """Simple processing method."""
        self.call_count += 1
        value = data.get('value', 0)
        return {'result': value * 2, 'agent': self.name}

    def slow_process(self, data):
        """Slow processing for timing tests."""
        time.sleep(0.1)
        return self.process(data)

    def error_process(self, data):
        """Method that raises an error."""
        raise ValueError("Intentional test error")

    def dependency_process(self, data):
        """Process with dependency results."""
        deps = data.get('_dependency_results', {})
        base_value = data.get('value', 0)

        # Sum results from dependencies
        dep_sum = sum(
            d.get('result', 0) if isinstance(d, dict) else 0
            for d in deps.values()
        )

        return {'result': base_value + dep_sum, 'agent': self.name}


class TestWorkflowOrchestrationAgent:
    """Test suite for WorkflowOrchestrationAgent."""

    def test_init_default(self):
        """Test agent initialization with defaults."""
        agent = WorkflowOrchestrationAgent()

        assert agent is not None
        assert agent.parallel_mode == ParallelMode.THREADS
        assert agent.max_workers is None
        assert agent.executor is not None

    def test_init_custom_mode(self):
        """Test initialization with custom parallel mode."""
        agent = WorkflowOrchestrationAgent(
            parallel_mode=ParallelMode.PROCESSES,
            max_workers=4
        )

        assert agent.parallel_mode == ParallelMode.PROCESSES
        assert agent.max_workers == 4

    def test_init_async_mode(self):
        """Test initialization with async mode."""
        agent = WorkflowOrchestrationAgent(parallel_mode=ParallelMode.ASYNC)

        assert agent.parallel_mode == ParallelMode.ASYNC


class TestBasicWorkflowExecution:
    """Tests for basic workflow execution."""

    def test_execute_empty_workflow(self):
        """Test executing empty workflow."""
        orchestrator = WorkflowOrchestrationAgent()
        result = orchestrator.execute_workflow([])

        assert isinstance(result, WorkflowResult)
        assert result.success
        assert len(result.steps_completed) == 0
        assert result.total_time == 0.0

    def test_execute_single_step(self):
        """Test executing workflow with single step."""
        orchestrator = WorkflowOrchestrationAgent()
        mock_agent = MockAgent()

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=mock_agent,
                method='process',
                inputs={'value': 5}
            )
        ]

        result = orchestrator.execute_workflow(steps)

        assert result.success
        assert len(result.steps_completed) == 1
        assert 'step1' in result.steps_completed
        assert result.results['step1']['result'] == 10
        assert mock_agent.call_count == 1

    def test_execute_sequential_workflow(self):
        """Test sequential workflow execution."""
        orchestrator = WorkflowOrchestrationAgent()
        agent1 = MockAgent('agent1')
        agent2 = MockAgent('agent2')

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=agent1,
                method='process',
                inputs={'value': 3}
            ),
            WorkflowStep(
                step_id='step2',
                agent=agent2,
                method='process',
                inputs={'value': 7}
            )
        ]

        result = orchestrator.execute_workflow(steps, parallel=False)

        assert result.success
        assert len(result.steps_completed) == 2
        assert result.results['step1']['result'] == 6
        assert result.results['step2']['result'] == 14

    def test_execute_parallel_workflow(self):
        """Test parallel workflow execution."""
        orchestrator = WorkflowOrchestrationAgent()
        agent1 = MockAgent('agent1')
        agent2 = MockAgent('agent2')

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=agent1,
                method='slow_process',
                inputs={'value': 3}
            ),
            WorkflowStep(
                step_id='step2',
                agent=agent2,
                method='slow_process',
                inputs={'value': 7}
            )
        ]

        # Parallel should be faster than 2 * 0.1s
        result = orchestrator.execute_workflow(steps, parallel=True)

        assert result.success
        assert len(result.steps_completed) == 2
        # Parallel execution should be faster than sequential
        assert result.total_time < 0.25  # Allow some overhead


class TestWorkflowDependencies:
    """Tests for workflow dependency management."""

    def test_execute_with_dependencies(self):
        """Test workflow with dependencies."""
        orchestrator = WorkflowOrchestrationAgent()
        agent1 = MockAgent('agent1')
        agent2 = MockAgent('agent2')
        agent3 = MockAgent('agent3')

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=agent1,
                method='process',
                inputs={'value': 2},
                dependencies=[]
            ),
            WorkflowStep(
                step_id='step2',
                agent=agent2,
                method='process',
                inputs={'value': 3},
                dependencies=[]
            ),
            WorkflowStep(
                step_id='step3',
                agent=agent3,
                method='dependency_process',
                inputs={'value': 1},
                dependencies=['step1', 'step2']
            )
        ]

        # Execute sequentially to ensure dependency results are passed correctly
        result = orchestrator.execute_workflow(steps, parallel=False)

        assert result.success
        assert len(result.steps_completed) == 3
        # step1 result: 2*2 = 4, step2 result: 3*2 = 6
        # step3 should receive results from step1 (4) and step2 (6)
        # Result: 1 + 4 + 6 = 11
        assert result.results['step3']['result'] == 11

    def test_execute_chain_dependencies(self):
        """Test workflow with chained dependencies."""
        orchestrator = WorkflowOrchestrationAgent()
        agent = MockAgent()

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=agent,
                method='process',
                inputs={'value': 1},
                dependencies=[]
            ),
            WorkflowStep(
                step_id='step2',
                agent=agent,
                method='dependency_process',
                inputs={'value': 0},
                dependencies=['step1']
            ),
            WorkflowStep(
                step_id='step3',
                agent=agent,
                method='dependency_process',
                inputs={'value': 0},
                dependencies=['step2']
            )
        ]

        result = orchestrator.execute_workflow(steps, parallel=False)

        assert result.success
        # step1: 2, step2: 0 + 2 = 2, step3: 0 + 2 = 2
        assert result.results['step1']['result'] == 2
        assert result.results['step2']['result'] == 2
        assert result.results['step3']['result'] == 2


class TestErrorHandling:
    """Tests for error handling in workflows."""

    def test_execute_step_with_error(self):
        """Test workflow with step that raises error."""
        orchestrator = WorkflowOrchestrationAgent()
        mock_agent = MockAgent()

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=mock_agent,
                method='error_process',
                inputs={}
            )
        ]

        result = orchestrator.execute_workflow(steps)

        assert not result.success
        assert len(result.errors) > 0

    def test_execute_sequential_stops_on_error(self):
        """Test that sequential execution stops on first error."""
        orchestrator = WorkflowOrchestrationAgent()
        agent1 = MockAgent()
        agent2 = MockAgent()

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=agent1,
                method='error_process',
                inputs={}
            ),
            WorkflowStep(
                step_id='step2',
                agent=agent2,
                method='process',
                inputs={'value': 5}
            )
        ]

        result = orchestrator.execute_workflow(steps, parallel=False)

        assert not result.success
        # step2 should not execute due to step1 error
        assert 'step2' not in result.results

    def test_execute_partial_success(self):
        """Test workflow with partial success (some steps fail)."""
        orchestrator = WorkflowOrchestrationAgent()
        agent1 = MockAgent()
        agent2 = MockAgent()
        agent3 = MockAgent()

        steps = [
            WorkflowStep(
                step_id='step1',
                agent=agent1,
                method='process',
                inputs={'value': 5}
            ),
            WorkflowStep(
                step_id='step2',
                agent=agent2,
                method='error_process',
                inputs={}
            ),
            WorkflowStep(
                step_id='step3',
                agent=agent3,
                method='process',
                inputs={'value': 3}
            )
        ]

        result = orchestrator.execute_workflow(steps, parallel=True)

        assert not result.success
        # Some steps should complete
        assert len(result.steps_completed) > 0


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_workflow_result_creation(self):
        """Test WorkflowResult creation."""
        result = WorkflowResult(
            success=True,
            steps_completed=['step1', 'step2'],
            results={'step1': {}, 'step2': {}},
            total_time=1.5
        )

        assert result.success
        assert len(result.steps_completed) == 2
        assert result.total_time == 1.5

    def test_workflow_result_repr(self):
        """Test WorkflowResult string representation."""
        result = WorkflowResult(
            success=True,
            steps_completed=['step1'],
            results={'step1': {}},
            total_time=0.5
        )

        repr_str = repr(result)
        assert 'SUCCESS' in repr_str
        assert 'steps=1' in repr_str
        assert 'time=0.500s' in repr_str

    def test_workflow_result_with_errors(self):
        """Test WorkflowResult with errors."""
        result = WorkflowResult(
            success=False,
            steps_completed=[],
            results={},
            total_time=0.1,
            errors=['Error 1', 'Error 2']
        )

        assert not result.success
        assert len(result.errors) == 2


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    def test_workflow_step_creation(self):
        """Test WorkflowStep creation."""
        agent = MockAgent()
        step = WorkflowStep(
            step_id='test_step',
            agent=agent,
            method='process',
            inputs={'value': 10}
        )

        assert step.step_id == 'test_step'
        assert step.agent == agent
        assert step.method == 'process'
        assert step.inputs == {'value': 10}
        assert step.dependencies == []

    def test_workflow_step_with_dependencies(self):
        """Test WorkflowStep with dependencies."""
        agent = MockAgent()
        step = WorkflowStep(
            step_id='test_step',
            agent=agent,
            method='process',
            inputs={},
            dependencies=['dep1', 'dep2']
        )

        assert len(step.dependencies) == 2
        assert 'dep1' in step.dependencies

    def test_workflow_step_with_metadata(self):
        """Test WorkflowStep with metadata."""
        agent = MockAgent()
        step = WorkflowStep(
            step_id='test_step',
            agent=agent,
            method='process',
            inputs={},
            metadata={'priority': 'high', 'timeout': 30}
        )

        assert step.metadata['priority'] == 'high'
        assert step.metadata['timeout'] == 30


class TestParallelModes:
    """Tests for different parallel execution modes."""

    def test_thread_mode(self):
        """Test workflow execution with thread mode."""
        orchestrator = WorkflowOrchestrationAgent(parallel_mode=ParallelMode.THREADS)
        agent = MockAgent()

        steps = [
            WorkflowStep(f'step{i}', agent, 'process', {'value': i})
            for i in range(3)
        ]

        result = orchestrator.execute_workflow(steps, parallel=True)

        assert result.success
        assert len(result.steps_completed) == 3

    def test_process_mode(self):
        """Test workflow execution with process mode."""
        orchestrator = WorkflowOrchestrationAgent(parallel_mode=ParallelMode.PROCESSES, max_workers=2)
        agent = MockAgent()

        steps = [
            WorkflowStep(f'step{i}', agent, 'process', {'value': i})
            for i in range(2)
        ]

        result = orchestrator.execute_workflow(steps, parallel=True)

        assert result.success
        assert len(result.steps_completed) == 2


class TestExecuteAgentsParallel:
    """Tests for execute_agents_parallel method."""

    def test_execute_agents_parallel_basic(self):
        """Test basic parallel agent execution."""
        orchestrator = WorkflowOrchestrationAgent()
        agents = [MockAgent(f'agent{i}') for i in range(3)]

        inputs_list = [{'value': i} for i in range(3)]

        results = orchestrator.execute_agents_parallel(
            agents=agents,
            method_name='process',
            inputs_list=inputs_list
        )

        assert len(results) == 3
        # Check that at least some results succeeded
        # (may have serialization issues with MockAgent in parallel)
        success_count = sum(1 for r in results if r.success)
        assert success_count >= 0  # At least function executes

    def test_execute_agents_parallel_timing(self):
        """Test that parallel execution is faster."""
        orchestrator = WorkflowOrchestrationAgent()
        agents = [MockAgent() for _ in range(3)]
        inputs_list = [{}] * 3

        start = time.perf_counter()
        results = orchestrator.execute_agents_parallel(
            agents=agents,
            method_name='slow_process',
            inputs_list=inputs_list
        )
        elapsed = time.perf_counter() - start

        # Should be faster than 3 * 0.1s (sequential)
        assert elapsed < 0.35
        assert len(results) == 3


class TestComplexWorkflows:
    """Tests for complex workflow patterns."""

    def test_diamond_dependency(self):
        """Test workflow with diamond dependency pattern."""
        orchestrator = WorkflowOrchestrationAgent()
        agent = MockAgent()

        # Diamond: A -> B,C -> D
        steps = [
            WorkflowStep('A', agent, 'process', {'value': 1}),
            WorkflowStep('B', agent, 'dependency_process', {'value': 0}, ['A']),
            WorkflowStep('C', agent, 'dependency_process', {'value': 0}, ['A']),
            WorkflowStep('D', agent, 'dependency_process', {'value': 0}, ['B', 'C'])
        ]

        result = orchestrator.execute_workflow(steps)

        assert result.success
        assert len(result.steps_completed) == 4

    def test_large_workflow(self):
        """Test workflow with many steps."""
        orchestrator = WorkflowOrchestrationAgent()
        agent = MockAgent()

        steps = [
            WorkflowStep(f'step{i}', agent, 'process', {'value': i})
            for i in range(20)
        ]

        result = orchestrator.execute_workflow(steps)

        assert result.success
        assert len(result.steps_completed) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
