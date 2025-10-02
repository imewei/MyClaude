"""Tests for ExecutorValidatorAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.executor_validator_agent import (
    ExecutorValidatorAgent, ValidationLevel, QualityMetric
)
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return ExecutorValidatorAgent()

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "ExecutorValidatorAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 4
    cap_names = [c.name for c in caps]
    assert 'execute_workflow' in cap_names
    assert 'validate_solution' in cap_names
    assert 'check_convergence' in cap_names
    assert 'generate_report' in cap_names

# Validation
def test_validate_missing_task_type(agent):
    val = agent.validate_input({})
    assert not val.valid
    assert any('task_type' in err for err in val.errors)

def test_validate_invalid_task_type(agent):
    val = agent.validate_input({'task_type': 'invalid'})
    assert not val.valid

def test_validate_valid_validate_task(agent):
    val = agent.validate_input({
        'task_type': 'validate',
        'solution': np.array([1, 2, 3])
    })
    assert val.valid

def test_validate_convergence_missing_residuals(agent):
    val = agent.validate_input({'task_type': 'check_convergence'})
    assert not val.valid
    assert any('residuals' in err for err in val.errors)

# Workflow Execution
def test_execute_workflow_simple(agent):
    """Test simple workflow execution."""
    workflow = [
        {'step': 1, 'agent': 'Agent1', 'action': 'action1'},
        {'step': 2, 'agent': 'Agent2', 'action': 'action2'}
    ]

    result = agent.execute({
        'task_type': 'execute_workflow',
        'workflow': workflow,
        'problem_data': {}
    })

    assert result.success
    assert 'execution_log' in result.data
    assert len(result.data['execution_log']) == 2

def test_execute_workflow_status(agent):
    """Test workflow execution status."""
    workflow = [
        {'step': 1, 'agent': 'TestAgent', 'action': 'test'}
    ]

    result = agent.execute({
        'task_type': 'execute_workflow',
        'workflow': workflow
    })

    assert result.success
    assert result.data['workflow_status'] == 'completed'
    assert result.data['successful_steps'] == 1

def test_execute_workflow_empty(agent):
    """Test workflow execution with empty workflow."""
    result = agent.execute({
        'task_type': 'execute_workflow',
        'workflow': []
    })

    assert result.success
    assert result.data['total_steps'] == 0

# Solution Validation
def test_validate_solution_valid_vector(agent):
    """Test validation of valid solution vector."""
    solution = np.array([1.0, 2.0, 3.0])

    result = agent.execute({
        'task_type': 'validate',
        'solution': solution
    })

    assert result.success
    checks = result.data['validation_checks']
    assert any(c['check'] == 'solution_exists' and c['passed'] for c in checks)

def test_validate_solution_with_nan(agent):
    """Test validation catches NaN values."""
    solution = np.array([1.0, np.nan, 3.0])

    result = agent.execute({
        'task_type': 'validate',
        'solution': solution
    })

    assert result.success
    checks = result.data['validation_checks']
    nan_check = next((c for c in checks if c['check'] == 'no_nan'), None)
    assert nan_check is not None
    assert not nan_check['passed']

def test_validate_solution_with_inf(agent):
    """Test validation catches Inf values."""
    solution = np.array([1.0, 2.0, np.inf])

    result = agent.execute({
        'task_type': 'validate',
        'solution': solution
    })

    assert result.success
    checks = result.data['validation_checks']
    inf_check = next((c for c in checks if c['check'] == 'no_inf'), None)
    assert inf_check is not None
    assert not inf_check['passed']

def test_validate_solution_shape(agent):
    """Test solution shape validation."""
    solution = np.array([1.0, 2.0, 3.0])
    expected_shape = (3,)

    result = agent.execute({
        'task_type': 'validate',
        'solution': solution,
        'problem_data': {'expected_shape': expected_shape}
    })

    assert result.success
    checks = result.data['validation_checks']
    shape_check = next((c for c in checks if c['check'] == 'correct_shape'), None)
    assert shape_check is not None
    assert shape_check['passed']

def test_validate_solution_wrong_shape(agent):
    """Test validation catches wrong shape."""
    solution = np.array([1.0, 2.0, 3.0])
    expected_shape = (5,)

    result = agent.execute({
        'task_type': 'validate',
        'solution': solution,
        'problem_data': {'expected_shape': expected_shape}
    })

    assert result.success
    checks = result.data['validation_checks']
    shape_check = next((c for c in checks if c['check'] == 'correct_shape'), None)
    assert shape_check is not None
    assert not shape_check['passed']

def test_validate_solution_residual(agent):
    """Test residual-based validation."""
    # Ax = b with exact solution
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([3.0, 3.0])
    x = np.array([1.0, 1.0])  # Exact solution

    result = agent.execute({
        'task_type': 'validate',
        'solution': x,
        'problem_data': {'matrix_A': A, 'vector_b': b}
    })

    assert result.success
    checks = result.data['validation_checks']
    residual_check = next((c for c in checks if c['check'] == 'residual'), None)
    assert residual_check is not None
    assert residual_check['passed']

def test_validate_solution_large_residual(agent):
    """Test validation catches large residuals."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([3.0, 3.0])
    x = np.array([5.0, 5.0])  # Wrong solution

    result = agent.execute({
        'task_type': 'validate',
        'solution': x,
        'problem_data': {'matrix_A': A, 'vector_b': b}
    })

    assert result.success
    checks = result.data['validation_checks']
    residual_check = next((c for c in checks if c['check'] == 'residual'), None)
    assert residual_check is not None
    assert not residual_check['passed']

def test_validate_solution_quality_metrics(agent):
    """Test quality metrics computation."""
    solution = np.array([1.0, 2.0, 3.0])

    result = agent.execute({
        'task_type': 'validate',
        'solution': solution
    })

    assert result.success
    assert 'quality_metrics' in result.data
    metrics = result.data['quality_metrics']
    assert 'accuracy' in metrics
    assert 'consistency' in metrics
    assert 'stability' in metrics

def test_validate_solution_overall_quality(agent):
    """Test overall quality assessment."""
    solution = np.array([1.0, 2.0, 3.0])

    result = agent.execute({
        'task_type': 'validate',
        'solution': solution
    })

    assert result.success
    assert 'overall_quality' in result.data
    assert result.data['overall_quality'] in ['excellent', 'good', 'acceptable', 'poor', 'unknown']

# Convergence Checking
def test_check_convergence_converged(agent):
    """Test convergence check for converged method."""
    residuals = np.array([1.0, 0.1, 0.01, 0.001, 1e-7])

    result = agent.execute({
        'task_type': 'check_convergence',
        'residuals': residuals,
        'tolerance': 1e-6
    })

    assert result.success
    assert result.data['converged']
    assert result.data['final_residual'] < 1e-6

def test_check_convergence_not_converged(agent):
    """Test convergence check for non-converged method."""
    residuals = np.array([1.0, 0.9, 0.85, 0.8, 0.75])

    result = agent.execute({
        'task_type': 'check_convergence',
        'residuals': residuals,
        'tolerance': 1e-6
    })

    assert result.success
    assert not result.data['converged']

def test_check_convergence_rate(agent):
    """Test convergence rate estimation."""
    # Linear convergence with rate ~ 0.1
    residuals = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])

    result = agent.execute({
        'task_type': 'check_convergence',
        'residuals': residuals,
        'tolerance': 1e-6
    })

    assert result.success
    assert 'convergence_rate' in result.data
    # Should detect linear convergence
    assert result.data['convergence_rate'] is not None

def test_check_convergence_quality(agent):
    """Test convergence quality assessment."""
    residuals = np.array([1.0, 0.1, 1e-7])  # Fast convergence

    result = agent.execute({
        'task_type': 'check_convergence',
        'residuals': residuals,
        'tolerance': 1e-6,
        'max_iterations': 100
    })

    assert result.success
    assert 'convergence_quality' in result.data
    # Fast convergence should be excellent or good
    assert result.data['convergence_quality'] in ['excellent', 'good']

def test_check_convergence_empty_residuals(agent):
    """Test convergence check with empty residuals."""
    result = agent.execute({
        'task_type': 'check_convergence',
        'residuals': [],
        'tolerance': 1e-6
    })

    assert result.success
    assert not result.data['converged']

# Report Generation
def test_generate_report_basic(agent):
    """Test basic report generation."""
    results = {'solution': [1, 2, 3]}
    validation = {'all_checks_passed': True, 'overall_quality': 'good'}

    result = agent.execute({
        'task_type': 'generate_report',
        'results': results,
        'validation': validation
    })

    assert result.success
    assert 'summary' in result.data
    assert 'results_overview' in result.data
    assert 'validation_summary' in result.data

def test_generate_report_summary(agent):
    """Test report summary generation."""
    results = {'step1': 'result1'}
    validation = {'all_checks_passed': True, 'overall_quality': 'excellent'}

    result = agent.execute({
        'task_type': 'generate_report',
        'results': results,
        'validation': validation
    })

    assert result.success
    summary = result.data['summary']
    assert 'passed' in summary.lower()
    assert 'excellent' in summary.lower()

def test_generate_report_recommendations(agent):
    """Test report recommendations."""
    results = {}
    validation = {'all_checks_passed': False, 'overall_quality': 'poor'}

    result = agent.execute({
        'task_type': 'generate_report',
        'results': results,
        'validation': validation
    })

    assert result.success
    assert 'recommendations' in result.data
    assert len(result.data['recommendations']) > 0

def test_generate_report_performance_metrics(agent):
    """Test performance metrics in report."""
    result = agent.execute({
        'task_type': 'generate_report',
        'results': {},
        'validation': {},
        'metadata': {'execution_time_sec': 1.5}
    })

    assert result.success
    assert 'performance_metrics' in result.data
    assert result.data['performance_metrics']['execution_time_sec'] == 1.5

# Error handling
def test_invalid_task_type(agent):
    """Test invalid task type handling."""
    result = agent.execute({
        'task_type': 'invalid_task'
    })

    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0

# Integration tests
def test_full_workflow_execution_and_validation(agent):
    """Test complete workflow execution followed by validation."""
    # Step 1: Execute workflow
    workflow = [
        {'step': 1, 'agent': 'LinearAlgebraAgent', 'action': 'solve'},
        {'step': 2, 'agent': 'ValidationAgent', 'action': 'check'}
    ]

    workflow_result = agent.execute({
        'task_type': 'execute_workflow',
        'workflow': workflow
    })

    assert workflow_result.success

    # Step 2: Validate solution
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    x = np.array([1.0, 2.0, 3.0])

    validation_result = agent.execute({
        'task_type': 'validate',
        'solution': x,
        'problem_data': {'matrix_A': A, 'vector_b': b}
    })

    assert validation_result.success
    assert validation_result.data['all_checks_passed']

    # Step 3: Check convergence
    convergence_result = agent.execute({
        'task_type': 'check_convergence',
        'residuals': np.array([1.0, 0.1, 0.01, 1e-7]),
        'tolerance': 1e-6
    })

    assert convergence_result.success
    assert convergence_result.data['converged']

    # Step 4: Generate report
    report_result = agent.execute({
        'task_type': 'generate_report',
        'results': workflow_result.data,
        'validation': validation_result.data
    })

    assert report_result.success
    assert 'summary' in report_result.data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
