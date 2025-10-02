"""Tests for ProblemAnalyzerAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.problem_analyzer_agent import (
    ProblemAnalyzerAgent, ProblemType, ProblemComplexity
)
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return ProblemAnalyzerAgent()

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "ProblemAnalyzerAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 4
    cap_names = [c.name for c in caps]
    assert 'classify_problem' in cap_names
    assert 'estimate_complexity' in cap_names
    assert 'identify_requirements' in cap_names
    assert 'recommend_approach' in cap_names

# Validation
def test_validate_missing_analysis_type(agent):
    val = agent.validate_input({})
    assert not val.valid
    assert any('analysis_type' in err for err in val.errors)

def test_validate_invalid_analysis_type(agent):
    val = agent.validate_input({'analysis_type': 'invalid'})
    assert not val.valid

def test_validate_valid_classify(agent):
    val = agent.validate_input({
        'analysis_type': 'classify',
        'problem_description': 'Solve ODE'
    })
    assert val.valid

# Problem Classification
def test_classify_ode_ivp(agent):
    """Test ODE IVP classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Solve an ODE initial value problem'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.ODE_IVP.value
    assert result.data['confidence'] > 0.8

def test_classify_ode_bvp(agent):
    """Test ODE BVP classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Solve a boundary value problem'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.ODE_BVP.value

def test_classify_pde(agent):
    """Test PDE classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Solve the heat equation PDE'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.PDE.value

def test_classify_linear_system(agent):
    """Test linear system classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Solve linear system Ax=b'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.LINEAR_SYSTEM.value
    assert result.data['confidence'] > 0.9

def test_classify_eigenvalue(agent):
    """Test eigenvalue classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Compute eigenvalues and eigenvectors'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.EIGENVALUE.value

def test_classify_optimization(agent):
    """Test optimization classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Minimize objective function'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.OPTIMIZATION.value

def test_classify_integration(agent):
    """Test integration classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Integrate a function using quadrature'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.INTEGRATION.value

def test_classify_inverse_problem(agent):
    """Test inverse problem classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Parameter estimation inverse problem'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.INVERSE_PROBLEM.value

def test_classify_uncertainty_quantification(agent):
    """Test UQ classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Monte Carlo uncertainty analysis'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.UNCERTAINTY_QUANTIFICATION.value

def test_classify_surrogate_modeling(agent):
    """Test surrogate modeling classification."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Build Gaussian process surrogate'
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.SURROGATE_MODELING.value

def test_classify_with_data_matrix(agent):
    """Test classification using matrix data."""
    A = np.random.randn(10, 10)
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': '',
        'problem_data': {'matrix_A': A}
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.LINEAR_SYSTEM.value

def test_classify_with_objective(agent):
    """Test classification using objective function."""
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': '',
        'problem_data': {'objective_function': lambda x: x**2}
    })

    assert result.success
    assert result.data['problem_type'] == ProblemType.OPTIMIZATION.value

def test_classify_characteristics(agent):
    """Test that characteristics are extracted."""
    A = np.random.randn(5, 5)
    result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Solve linear system',
        'problem_data': {'matrix_A': A}
    })

    assert result.success
    assert 'characteristics' in result.data
    chars = result.data['characteristics']
    assert 'dimensions' in chars
    assert chars['is_square']

# Complexity Estimation
def test_estimate_complexity_small_linear(agent):
    """Test complexity estimation for small linear system."""
    result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'problem_data': {'n_dof': 50}
    })

    assert result.success
    assert result.data['complexity_class'] == ProblemComplexity.SIMPLE.value
    assert result.data['memory_requirement'] == 'LOW'

def test_estimate_complexity_medium_linear(agent):
    """Test complexity for medium linear system."""
    result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'problem_data': {'n_dof': 500}
    })

    assert result.success
    assert result.data['complexity_class'] == ProblemComplexity.MODERATE.value
    assert result.data['memory_requirement'] == 'MEDIUM'

def test_estimate_complexity_large_linear(agent):
    """Test complexity for large linear system."""
    result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'problem_data': {'n_dof': 5000}
    })

    assert result.success
    assert result.data['complexity_class'] == ProblemComplexity.COMPLEX.value
    assert result.data['time_requirement'] == 'SLOW'

def test_estimate_complexity_very_large_linear(agent):
    """Test complexity for very large linear system."""
    result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'problem_data': {'n_dof': 50000}
    })

    assert result.success
    assert result.data['complexity_class'] == ProblemComplexity.VERY_COMPLEX.value
    assert result.data['memory_requirement'] == 'VERY_HIGH'

def test_estimate_complexity_optimization(agent):
    """Test complexity estimation for optimization."""
    result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'problem_data': {'n_variables': 50}
    })

    assert result.success
    assert result.data['complexity_class'] == ProblemComplexity.MODERATE.value

def test_estimate_complexity_uq(agent):
    """Test complexity estimation for UQ."""
    result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': ProblemType.UNCERTAINTY_QUANTIFICATION.value,
        'problem_data': {'n_samples': 5000}
    })

    assert result.success
    assert result.data['complexity_class'] == ProblemComplexity.MODERATE.value

def test_estimate_complexity_with_matrix(agent):
    """Test complexity estimation from matrix size."""
    A = np.random.randn(200, 200)
    result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'problem_data': {'matrix_A': A}
    })

    assert result.success
    assert result.data['complexity_class'] == ProblemComplexity.MODERATE.value

# Requirements Identification
def test_identify_requirements_ode(agent):
    """Test requirements for ODE problem."""
    result = agent.execute({
        'analysis_type': 'requirements',
        'problem_type': ProblemType.ODE_IVP.value
    })

    assert result.success
    assert 'ODEPDESolverAgent' in result.data['required_agents']
    assert 'solve_ode_ivp' in result.data['required_capabilities']

def test_identify_requirements_linear(agent):
    """Test requirements for linear system."""
    result = agent.execute({
        'analysis_type': 'requirements',
        'problem_type': ProblemType.LINEAR_SYSTEM.value
    })

    assert result.success
    assert 'LinearAlgebraAgent' in result.data['required_agents']
    assert 'solve_linear_system' in result.data['required_capabilities']

def test_identify_requirements_optimization(agent):
    """Test requirements for optimization."""
    result = agent.execute({
        'analysis_type': 'requirements',
        'problem_type': ProblemType.OPTIMIZATION.value
    })

    assert result.success
    assert 'OptimizationAgent' in result.data['required_agents']

def test_identify_requirements_inverse(agent):
    """Test requirements for inverse problem."""
    result = agent.execute({
        'analysis_type': 'requirements',
        'problem_type': ProblemType.INVERSE_PROBLEM.value
    })

    assert result.success
    assert 'InverseProblemsAgent' in result.data['required_agents']

def test_identify_requirements_with_uncertainty(agent):
    """Test optional UQ agent."""
    result = agent.execute({
        'analysis_type': 'requirements',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'requires_uncertainty': True
    })

    assert result.success
    assert 'UncertaintyQuantificationAgent' in result.data['optional_agents']

def test_identify_requirements_expensive_model(agent):
    """Test optional surrogate for expensive model."""
    result = agent.execute({
        'analysis_type': 'requirements',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'expensive_model': True
    })

    assert result.success
    assert 'SurrogateModelingAgent' in result.data['optional_agents']

# Approach Recommendation
def test_recommend_linear_simple(agent):
    """Test recommendation for simple linear system."""
    result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert len(result.data['recommendations']) > 0
    assert 'Direct solver' in result.data['recommendations'][0]['method']

def test_recommend_linear_complex(agent):
    """Test recommendation for complex linear system."""
    result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.COMPLEX.value
    })

    assert result.success
    rec = result.data['recommendations'][0]
    assert 'Iterative solver' in rec['method']
    assert rec['agent'] == 'LinearAlgebraAgent'

def test_recommend_optimization(agent):
    """Test recommendation for optimization."""
    result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert len(result.data['recommendations']) > 0
    assert 'BFGS' in result.data['recommendations'][0]['method']

def test_recommend_optimization_complex(agent):
    """Test recommendation for complex optimization."""
    result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'complexity_class': ProblemComplexity.COMPLEX.value
    })

    assert result.success
    # Should have global optimization recommendation
    assert len(result.data['recommendations']) > 1

def test_recommend_uq(agent):
    """Test recommendation for UQ."""
    result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': ProblemType.UNCERTAINTY_QUANTIFICATION.value,
        'complexity_class': ProblemComplexity.MODERATE.value
    })

    assert result.success
    recs = result.data['recommendations']
    methods = [r['method'] for r in recs]
    assert any('Monte Carlo' in m for m in methods)

def test_recommend_execution_plan(agent):
    """Test that execution plan is generated."""
    result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': ProblemType.LINEAR_SYSTEM.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'execution_plan' in result.data
    plan = result.data['execution_plan']
    assert len(plan) > 0
    assert all('step' in item and 'action' in item for item in plan)

def test_recommend_estimated_steps(agent):
    """Test estimated_steps field."""
    result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': ProblemType.OPTIMIZATION.value,
        'complexity_class': ProblemComplexity.SIMPLE.value
    })

    assert result.success
    assert 'estimated_steps' in result.data
    assert result.data['estimated_steps'] > 0

# Error handling
def test_invalid_analysis_type(agent):
    """Test invalid analysis type handling."""
    result = agent.execute({
        'analysis_type': 'invalid_type'
    })

    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0

# Integration tests
def test_full_workflow_classification_to_recommendation(agent):
    """Test complete workflow from classification to recommendation."""
    # Step 1: Classify
    classify_result = agent.execute({
        'analysis_type': 'classify',
        'problem_description': 'Solve large linear system Ax=b with n=5000'
    })

    assert classify_result.success
    problem_type = classify_result.data['problem_type']

    # Step 2: Estimate complexity
    complexity_result = agent.execute({
        'analysis_type': 'complexity',
        'problem_type': problem_type,
        'problem_data': {'n_dof': 5000}
    })

    assert complexity_result.success
    complexity_class = complexity_result.data['complexity_class']

    # Step 3: Identify requirements
    req_result = agent.execute({
        'analysis_type': 'requirements',
        'problem_type': problem_type
    })

    assert req_result.success
    assert len(req_result.data['required_agents']) > 0

    # Step 4: Recommend approach
    rec_result = agent.execute({
        'analysis_type': 'recommend',
        'problem_type': problem_type,
        'complexity_class': complexity_class
    })

    assert rec_result.success
    assert len(rec_result.data['recommendations']) > 0
    assert len(rec_result.data['execution_plan']) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
