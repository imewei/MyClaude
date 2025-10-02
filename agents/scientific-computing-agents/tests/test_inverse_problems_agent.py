"""Tests for InverseProblemsAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.inverse_problems_agent import InverseProblemsAgent
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return InverseProblemsAgent(config={'tolerance': 1e-6})

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "InverseProblemsAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 4
    cap_names = [c.name for c in caps]
    assert 'bayesian_inference' in cap_names
    assert 'ensemble_kalman_filter' in cap_names
    assert 'variational_assimilation' in cap_names
    assert 'regularized_inversion' in cap_names

# Validation
def test_validate_bayesian_valid(agent):
    data = {
        'problem_type': 'bayesian',
        'observations': np.array([1, 2, 3]),
        'forward_model': lambda x: x[0] * np.array([1, 2, 3])
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_bayesian_missing_observations(agent):
    data = {
        'problem_type': 'bayesian',
        'forward_model': lambda x: x
    }
    val = agent.validate_input(data)
    assert not val.valid

def test_validate_enkf_valid(agent):
    data = {
        'problem_type': 'enkf',
        'ensemble': np.random.randn(50, 10),
        'observations': np.array([1, 2, 3])
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_variational_valid(agent):
    data = {
        'problem_type': 'variational',
        'background': np.array([1, 2, 3]),
        'observations': np.array([1.1, 2.1, 3.1])
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_regularized_valid(agent):
    data = {
        'problem_type': 'regularized',
        'forward_matrix': np.eye(5),
        'observations': np.array([1, 2, 3, 4, 5])
    }
    val = agent.validate_input(data)
    assert val.valid

# Resource estimation
def test_estimate_resources_bayesian(agent):
    data = {
        'problem_type': 'bayesian',
        'observations': np.random.randn(100),
        'forward_model': lambda x: x
    }
    res = agent.estimate_resources(data)
    assert res.cpu_cores >= 1

def test_estimate_resources_enkf_large(agent):
    data = {
        'problem_type': 'enkf',
        'ensemble': np.random.randn(2000, 15000),
        'observations': np.random.randn(100)
    }
    res = agent.estimate_resources(data)
    assert res.cpu_cores >= 4
    assert res.memory_gb >= 8

# Bayesian Inference
def test_bayesian_linear_model(agent):
    """Test Bayesian inference on linear model y = a*x."""
    true_a = 2.5
    x_data = np.linspace(0, 1, 10)
    y_obs = true_a * x_data + np.random.normal(0, 0.1, 10)

    def forward_model(params):
        return params[0] * x_data

    result = agent.execute({
        'problem_type': 'bayesian',
        'observations': y_obs,
        'forward_model': forward_model,
        'prior': {
            'mean': np.array([1.0]),
            'covariance': np.array([[4.0]])
        },
        'observation_noise': 0.1
    })

    assert result.success
    posterior_mean = result.data['solution']['posterior_mean']
    assert len(posterior_mean) == 1
    # Should estimate close to true value
    assert abs(posterior_mean[0] - true_a) < 1.0

def test_bayesian_credible_intervals(agent):
    """Test that credible intervals are computed."""
    def forward_model(params):
        return params

    result = agent.execute({
        'problem_type': 'bayesian',
        'observations': np.array([1.0, 2.0]),
        'forward_model': forward_model,
        'prior': {
            'mean': np.array([0.5, 1.5]),
            'covariance': np.eye(2)
        }
    })

    assert result.success
    assert 'credible_intervals' in result.data['solution']
    intervals = result.data['solution']['credible_intervals']
    assert intervals.shape == (2, 2)  # (n_params, 2)

# Ensemble Kalman Filter
def test_enkf_simple(agent):
    """Test EnKF on simple linear observation."""
    n_ensemble = 50
    n_state = 5
    n_obs = 3

    # Create ensemble
    ensemble = np.random.randn(n_ensemble, n_state)

    # True state
    true_state = np.array([1, 2, 3, 4, 5])

    # Observations (first 3 components)
    observations = true_state[:n_obs] + np.random.normal(0, 0.1, n_obs)

    # Observation operator
    H = np.zeros((n_obs, n_state))
    H[:n_obs, :n_obs] = np.eye(n_obs)

    result = agent.execute({
        'problem_type': 'enkf',
        'ensemble': ensemble,
        'observations': observations,
        'observation_operator': H,
        'observation_noise': 0.1
    })

    assert result.success
    ensemble_updated = result.data['solution']['ensemble_updated']
    analysis_mean = result.data['solution']['analysis_mean']

    assert ensemble_updated.shape == (n_ensemble, n_state)
    assert len(analysis_mean) == n_state

    # First 3 components should be close to observations
    assert np.allclose(analysis_mean[:n_obs], observations, atol=0.5)

def test_enkf_kalman_gain(agent):
    """Test that Kalman gain is computed."""
    ensemble = np.random.randn(100, 10)
    observations = np.array([1, 2, 3])

    result = agent.execute({
        'problem_type': 'enkf',
        'ensemble': ensemble,
        'observations': observations
    })

    assert result.success
    assert 'kalman_gain' in result.data['solution']
    K = result.data['solution']['kalman_gain']
    assert K.shape[1] == len(observations)

# Variational Assimilation
def test_variational_3dvar(agent):
    """Test 3D-Var data assimilation."""
    n_state = 10
    background = np.ones(n_state)
    observations = np.array([1.5, 2.0, 2.5])

    # Observe first 3 components
    H = np.zeros((3, n_state))
    H[:3, :3] = np.eye(3)

    result = agent.execute({
        'problem_type': 'variational',
        'background': background,
        'observations': observations,
        'observation_operator': H,
        'background_error': 1.0,
        'observation_error': 0.1
    })

    assert result.success
    analysis = result.data['solution']['analysis']
    assert len(analysis) == n_state

    # First 3 components should be closer to observations
    assert np.allclose(analysis[:3], observations, atol=0.5)

def test_variational_cost_reduction(agent):
    """Test that cost function is reduced."""
    background = np.array([0, 0, 0])
    observations = np.array([1, 1, 1])

    result = agent.execute({
        'problem_type': 'variational',
        'background': background,
        'observations': observations
    })

    assert result.success
    cost_reduction = result.data['solution']['cost_reduction']
    assert cost_reduction > 0  # Cost should decrease

# Regularized Inversion
def test_regularized_tikhonov(agent):
    """Test Tikhonov regularization."""
    # Ill-conditioned problem: deblurring
    n = 10
    A = np.eye(n) + 0.1 * np.random.randn(n, n)
    true_x = np.random.randn(n)
    b = A @ true_x

    result = agent.execute({
        'problem_type': 'regularized',
        'forward_matrix': A,
        'observations': b,
        'regularization_type': 'tikhonov',
        'regularization_parameter': 0.01
    })

    assert result.success
    solution = result.data['solution']['solution']
    assert len(solution) == n

    # Should recover close to true solution
    assert np.linalg.norm(solution - true_x) / np.linalg.norm(true_x) < 0.5

def test_regularized_truncated_svd(agent):
    """Test truncated SVD regularization."""
    A = np.array([[1, 0], [0, 1], [0, 0]])  # Rank 2
    b = np.array([1, 1, 0])

    result = agent.execute({
        'problem_type': 'regularized',
        'forward_matrix': A,
        'observations': b,
        'regularization_type': 'truncated_svd',
        'regularization_parameter': 0.1
    })

    assert result.success
    solution = result.data['solution']['solution']
    assert len(solution) == 2

def test_regularized_residual(agent):
    """Test residual computation."""
    A = np.eye(5)
    b = np.array([1, 2, 3, 4, 5])

    result = agent.execute({
        'problem_type': 'regularized',
        'forward_matrix': A,
        'observations': b,
        'regularization_parameter': 0.001
    })

    assert result.success
    residual_norm = result.data['solution']['residual_norm']
    # Should have small residual for well-conditioned identity problem
    assert residual_norm < 1.0

# Metadata and provenance
def test_metadata(agent):
    """Test metadata is populated."""
    def forward_model(params):
        return params

    result = agent.execute({
        'problem_type': 'bayesian',
        'observations': np.array([1, 2]),
        'forward_model': forward_model
    })

    assert result.success
    assert 'n_parameters' in result.data['metadata']
    assert 'n_observations' in result.data['metadata']

def test_provenance(agent):
    """Test provenance tracking."""
    A = np.eye(3)
    b = np.array([1, 2, 3])

    result = agent.execute({
        'problem_type': 'regularized',
        'forward_matrix': A,
        'observations': b
    })

    assert result.provenance is not None
    assert result.provenance.agent_name == "InverseProblemsAgent"

# Error handling
def test_invalid_problem_type(agent):
    """Test handling of invalid problem type."""
    result = agent.execute({
        'problem_type': 'invalid_type'
    })

    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
