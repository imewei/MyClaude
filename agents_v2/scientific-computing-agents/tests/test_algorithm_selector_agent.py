"""Tests for AlgorithmSelectorAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.algorithm_selector_agent import AlgorithmSelectorAgent, AlgorithmCategory
from agents.problem_analyzer_agent import ProblemType, ProblemComplexity
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return AlgorithmSelectorAgent()

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "AlgorithmSelectorAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 4
    cap_names = [c.name for c in caps]
    assert 'select_algorithm' in cap_names
    assert 'select_agents' in cap_names
    assert 'tune_parameters' in cap_names
    assert 'design_workflow' in cap_names

# Validation
def test_validate_missing_selection_type(agent):
    val = agent.validate_input({})
    assert not val.valid
    assert any('selection_type' in err for err in val.errors)

def test_validate_invalid_selection_type(agent):
    val = agent.validate_input({'selection_type': 'invalid'})
    assert not val.valid

def test_validate_valid_algorithm_selection(agent):
    val = agent.validate_input({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.LINEAR_SYSTEM.value
    })
    assert val.valid

# Algorithm Selection
def test_select_algorithm_linear_simple(agent):
    """Test algorithm selection for simple linear system."""
    result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'selected_algorithm' in result.data
    # Simple problems should prefer direct methods
    assert 'LU' in result.data['selected_algorithm'] or 'Direct' in result.data.get('algorithm_type', '')

def test_select_algorithm_linear_complex(agent):
    """Test algorithm selection for complex linear system."""
    result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.COMPLEX.value
    })

    assert result.success
    algo = result.data['selected_algorithm']
    # Large problems should prefer iterative methods
    assert 'Conjugate' in algo or 'GMRES' in algo or 'iterative' in result.data.get('algorithm_type', '')

def test_select_algorithm_optimization(agent):
    """Test algorithm selection for optimization."""
    result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'complexity_class': ProblemComplexity.MODERATE.value
    })

    assert result.success
    assert 'selected_algorithm' in result.data
    assert result.data['confidence'] > 0.0

def test_select_algorithm_uq(agent):
    """Test algorithm selection for UQ."""
    result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.UNCERTAINTY_QUANTIFICATION.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    algo = result.data['selected_algorithm']
    assert 'Monte Carlo' in algo or 'Latin Hypercube' in algo or 'Polynomial Chaos' in algo

def test_select_algorithm_ode(agent):
    """Test algorithm selection for ODE."""
    result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.ODE_IVP.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'selected_algorithm' in result.data

def test_select_algorithm_alternatives(agent):
    """Test that alternatives are provided."""
    result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.MODERATE.value
    })

    assert result.success
    assert 'alternatives' in result.data
    assert isinstance(result.data['alternatives'], list)

def test_select_algorithm_performance_estimate(agent):
    """Test that performance estimate is provided."""
    result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'expected_performance' in result.data
    perf = result.data['expected_performance']
    assert 'expected_runtime' in perf
    assert 'memory_usage' in perf

# Agent Selection
def test_select_agents_linear(agent):
    """Test agent selection for linear system."""
    result = agent.execute({
        'selection_type': 'agents',
        'problem_type': ProblemType.LINEAR_SYSTEM.value
    })

    assert result.success
    assert 'LinearAlgebraAgent' in result.data['primary_agents']

def test_select_agents_optimization(agent):
    """Test agent selection for optimization."""
    result = agent.execute({
        'selection_type': 'agents',
        'problem_type': ProblemType.OPTIMIZATION.value
    })

    assert result.success
    assert 'OptimizationAgent' in result.data['primary_agents']

def test_select_agents_with_uncertainty(agent):
    """Test agent selection with UQ requirement."""
    result = agent.execute({
        'selection_type': 'agents',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'optional_requirements': ['uncertainty']
    })

    assert result.success
    assert 'UncertaintyQuantificationAgent' in result.data['supporting_agents']

def test_select_agents_execution_order(agent):
    """Test that execution order is provided."""
    result = agent.execute({
        'selection_type': 'agents',
        'problem_type': ProblemType.OPTIMIZATION.value
    })

    assert result.success
    assert 'execution_order' in result.data
    order = result.data['execution_order']
    assert len(order) > 0
    assert all('agent' in item and 'order' in item for item in order)

def test_select_agents_total_count(agent):
    """Test total agent count."""
    result = agent.execute({
        'selection_type': 'agents',
        'problem_type': ProblemType.INVERSE_PROBLEM.value
    })

    assert result.success
    assert 'total_agents' in result.data
    assert result.data['total_agents'] > 0

# Parameter Tuning
def test_tune_parameters_cg(agent):
    """Test parameter tuning for CG."""
    result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': 'Conjugate Gradient',
        'problem_size': 500
    })

    assert result.success
    params = result.data['recommended_parameters']
    assert 'tolerance' in params
    assert 'max_iterations' in params

def test_tune_parameters_gmres(agent):
    """Test parameter tuning for GMRES."""
    result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': 'GMRES',
        'problem_size': 1000
    })

    assert result.success
    params = result.data['recommended_parameters']
    assert 'restart' in params
    assert 'tolerance' in params

def test_tune_parameters_bfgs(agent):
    """Test parameter tuning for BFGS."""
    result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': 'L-BFGS',
        'problem_size': 100
    })

    assert result.success
    params = result.data['recommended_parameters']
    assert 'tolerance' in params
    assert 'max_iterations' in params

def test_tune_parameters_monte_carlo(agent):
    """Test parameter tuning for Monte Carlo."""
    result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': 'Monte Carlo',
        'desired_tolerance': 0.01
    })

    assert result.success
    params = result.data['recommended_parameters']
    assert 'n_samples' in params
    # For tolerance 0.01, need n ~ 1/0.01^2 = 10000 samples
    assert params['n_samples'] >= 1000

def test_tune_parameters_rk45(agent):
    """Test parameter tuning for RK45."""
    result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': 'RK45',
        'desired_tolerance': 1e-8
    })

    assert result.success
    params = result.data['recommended_parameters']
    assert 'atol' in params
    assert 'rtol' in params

def test_tune_parameters_rationale(agent):
    """Test that tuning rationale is provided."""
    result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': 'GMRES',
        'problem_size': 500
    })

    assert result.success
    assert 'tuning_rationale' in result.data
    assert isinstance(result.data['tuning_rationale'], str)

def test_tune_parameters_sensitivity(agent):
    """Test that parameter sensitivity is provided."""
    result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': 'Conjugate Gradient',
        'problem_size': 100
    })

    assert result.success
    assert 'sensitivity' in result.data
    assert isinstance(result.data['sensitivity'], dict)

# Workflow Design
def test_design_workflow_linear(agent):
    """Test workflow design for linear system."""
    result = agent.execute({
        'selection_type': 'workflow',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'workflow_steps' in result.data
    steps = result.data['workflow_steps']
    assert len(steps) > 0

    # Should have ProblemAnalyzer, AlgorithmSelector, and ExecutorValidator
    agent_names = [s['agent'] for s in steps]
    assert 'ProblemAnalyzerAgent' in agent_names
    assert 'AlgorithmSelectorAgent' in agent_names

def test_design_workflow_optimization(agent):
    """Test workflow design for optimization."""
    result = agent.execute({
        'selection_type': 'workflow',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'complexity_class': ProblemComplexity.MODERATE.value
    })

    assert result.success
    steps = result.data['workflow_steps']
    agent_names = [s['agent'] for s in steps]
    assert 'OptimizationAgent' in agent_names

def test_design_workflow_dependencies(agent):
    """Test workflow dependencies."""
    result = agent.execute({
        'selection_type': 'workflow',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'dependencies' in result.data
    deps = result.data['dependencies']
    assert len(deps) > 0
    assert all('step' in d and 'depends_on' in d for d in deps)

def test_design_workflow_with_uq(agent):
    """Test workflow with UQ requirement."""
    result = agent.execute({
        'selection_type': 'workflow',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'complexity_class': ProblemComplexity.SIMPLE.value,
        'requirements': {'uncertainty_quantification': True}
    })

    assert result.success
    steps = result.data['workflow_steps']
    agent_names = [s['agent'] for s in steps]
    assert 'UncertaintyQuantificationAgent' in agent_names

def test_design_workflow_runtime_estimate(agent):
    """Test workflow runtime estimation."""
    result = agent.execute({
        'selection_type': 'workflow',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'estimated_runtime' in result.data
    assert isinstance(result.data['estimated_runtime'], str)

def test_design_workflow_parallel_opportunities(agent):
    """Test parallel step identification."""
    result = agent.execute({
        'selection_type': 'workflow',
        'problem_type': ProblemType.UNCERTAINTY_QUANTIFICATION.value,
        'complexity_class': ProblemComplexity.MODERATE.value
    })

    assert result.success
    assert 'parallel_opportunities' in result.data
    assert isinstance(result.data['parallel_opportunities'], list)

def test_design_workflow_total_steps(agent):
    """Test total steps count."""
    result = agent.execute({
        'selection_type': 'workflow',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'total_steps' in result.data
    assert result.data['total_steps'] == len(result.data['workflow_steps'])

# Error handling
def test_invalid_selection_type(agent):
    """Test invalid selection type handling."""
    result = agent.execute({
        'selection_type': 'invalid_type'
    })

    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0

# Integration tests
def test_full_workflow_algorithm_to_agents(agent):
    """Test complete workflow from algorithm to agent selection."""
    # Step 1: Select algorithm
    algo_result = agent.execute({
        'selection_type': 'algorithm',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.MODERATE.value
    })

    assert algo_result.success
    selected_algo = algo_result.data['selected_algorithm']

    # Step 2: Tune parameters
    param_result = agent.execute({
        'selection_type': 'parameters',
        'algorithm': selected_algo,
        'problem_size': 500
    })

    assert param_result.success
    params = param_result.data['recommended_parameters']

    # Step 3: Select agents
    agent_result = agent.execute({
        'selection_type': 'agents',
        'problem_type': ProblemType.LINEAR_SYSTEM.value
    })

    assert agent_result.success
    assert len(agent_result.data['primary_agents']) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
