"""Tests for SurrogateModelingAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.surrogate_modeling_agent import SurrogateModelingAgent
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return SurrogateModelingAgent(config={'tolerance': 1e-6})

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "SurrogateModelingAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 4
    cap_names = [c.name for c in caps]
    assert 'gaussian_process' in cap_names
    assert 'polynomial_chaos' in cap_names
    assert 'kriging' in cap_names
    assert 'reduced_order_model' in cap_names

# Validation
def test_validate_gp_valid(agent):
    data = {
        'problem_type': 'gp_regression',
        'training_x': np.array([[1], [2], [3]]),
        'training_y': np.array([1, 4, 9])
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_gp_missing_data(agent):
    data = {'problem_type': 'gp_regression'}
    val = agent.validate_input(data)
    assert not val.valid
    assert any('training_x' in e for e in val.errors)

def test_validate_gp_mismatched_lengths(agent):
    data = {
        'problem_type': 'gp_regression',
        'training_x': np.array([[1], [2]]),
        'training_y': np.array([1, 4, 9])
    }
    val = agent.validate_input(data)
    assert not val.valid

def test_validate_pce_valid(agent):
    data = {
        'problem_type': 'polynomial_chaos',
        'samples': np.random.randn(100, 2),
        'polynomial_order': 3
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_rom_valid(agent):
    data = {
        'problem_type': 'rom',
        'snapshots': np.random.randn(50, 10)
    }
    val = agent.validate_input(data)
    assert val.valid

# Resource estimation
def test_estimate_resources_gp_small(agent):
    data = {
        'problem_type': 'gp_regression',
        'training_x': np.random.randn(100, 2),
        'training_y': np.random.randn(100)
    }
    res = agent.estimate_resources(data)
    assert res.cpu_cores >= 1
    assert res.memory_gb >= 1.0

def test_estimate_resources_gp_large(agent):
    data = {
        'problem_type': 'gp_regression',
        'training_x': np.random.randn(2000, 2),
        'training_y': np.random.randn(2000)
    }
    res = agent.estimate_resources(data)
    assert res.cpu_cores >= 4
    assert res.estimated_time_sec > 10

# Gaussian Process Regression
def test_gp_regression_1d(agent):
    """Test GP regression on 1D data."""
    # Training data: y = x^2
    X_train = np.array([[0], [1], [2], [3], [4]])
    y_train = np.array([0, 1, 4, 9, 16])

    # Test points
    X_test = np.array([[0.5], [1.5], [2.5]])

    result = agent.execute({
        'problem_type': 'gp_regression',
        'training_x': X_train,
        'training_y': y_train,
        'test_x': X_test,
        'kernel': 'rbf',
        'length_scale': 1.0
    })

    assert result.success
    assert 'predictions' in result.data['solution']
    assert 'uncertainties' in result.data['solution']

    predictions = result.data['solution']['predictions']
    assert len(predictions) == 3

    # Check predictions are reasonable (should be close to x^2)
    expected = np.array([0.25, 2.25, 6.25])
    assert np.allclose(predictions, expected, atol=2.0)

def test_gp_regression_2d(agent):
    """Test GP regression on 2D data."""
    # Training data: z = x + y
    X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y_train = np.array([0, 1, 1, 2])

    X_test = np.array([[0.5, 0.5]])

    result = agent.execute({
        'problem_type': 'gp_regression',
        'training_x': X_train,
        'training_y': y_train,
        'test_x': X_test,
        'kernel': 'rbf'
    })

    assert result.success
    predictions = result.data['solution']['predictions']
    # Should predict close to 1.0 for (0.5, 0.5)
    assert abs(predictions[0] - 1.0) < 0.5

def test_gp_different_kernels(agent):
    """Test GP with different kernel types."""
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([1, 2, 3, 4])

    for kernel in ['rbf', 'matern', 'linear']:
        result = agent.execute({
            'problem_type': 'gp_regression',
            'training_x': X_train,
            'training_y': y_train,
            'kernel': kernel
        })
        assert result.success
        assert result.data['metadata']['kernel'] == kernel

# Polynomial Chaos Expansion
def test_pce_basic(agent):
    """Test basic PCE without function values."""
    samples = np.random.randn(100, 2)

    result = agent.execute({
        'problem_type': 'polynomial_chaos',
        'samples': samples,
        'polynomial_order': 2
    })

    assert result.success
    assert 'basis' in result.data['solution']
    assert result.data['solution']['basis'].shape[0] == 100

def test_pce_with_function_values(agent):
    """Test PCE with function evaluation."""
    # Generate samples
    samples = np.random.randn(50, 2)

    # Evaluate function: f(x, y) = x^2 + y
    function_values = samples[:, 0]**2 + samples[:, 1]

    result = agent.execute({
        'problem_type': 'polynomial_chaos',
        'samples': samples,
        'function_values': function_values,
        'polynomial_order': 3
    })

    assert result.success
    assert 'coefficients' in result.data['solution']
    assert result.data['solution']['coefficients'] is not None

def test_pce_sensitivity_indices(agent):
    """Test Sobol sensitivity indices computation."""
    samples = np.random.randn(100, 3)
    function_values = samples[:, 0]**2 + 0.5 * samples[:, 1] + 0.1 * samples[:, 2]

    result = agent.execute({
        'problem_type': 'polynomial_chaos',
        'samples': samples,
        'function_values': function_values,
        'polynomial_order': 2
    })

    assert result.success
    sensitivity_indices = result.data['solution']['sensitivity_indices']
    assert sensitivity_indices is not None
    assert 'S1' in sensitivity_indices  # First variable index

# Kriging
def test_kriging_1d(agent):
    """Test kriging interpolation in 1D."""
    locations = np.array([[0], [1], [2], [3], [4]])
    values = np.array([0, 1, 4, 9, 16])  # y = x^2

    prediction_locations = np.array([[1.5], [2.5]])

    result = agent.execute({
        'problem_type': 'kriging',
        'locations': locations,
        'values': values,
        'prediction_locations': prediction_locations
    })

    assert result.success
    assert 'predictions' in result.data['solution']
    predictions = result.data['solution']['predictions']
    assert len(predictions) == 2

def test_kriging_2d(agent):
    """Test kriging in 2D."""
    # Grid of points
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y)
    locations = np.column_stack([X.ravel(), Y.ravel()])
    values = X.ravel() + Y.ravel()  # f(x,y) = x + y

    prediction_locations = np.array([[0.5, 0.5], [0.25, 0.75]])

    result = agent.execute({
        'problem_type': 'kriging',
        'locations': locations,
        'values': values,
        'prediction_locations': prediction_locations
    })

    assert result.success
    predictions = result.data['solution']['predictions']
    # Should be close to [1.0, 1.0]
    assert np.allclose(predictions, [1.0, 1.0], atol=0.2)

# Reduced-Order Models
def test_rom_basic(agent):
    """Test basic ROM/POD."""
    # Generate snapshot matrix: random data
    snapshots = np.random.randn(100, 20)

    result = agent.execute({
        'problem_type': 'rom',
        'snapshots': snapshots,
        'n_modes': 5
    })

    assert result.success
    assert 'reduced_basis' in result.data['solution']
    basis = result.data['solution']['reduced_basis']
    assert basis.shape == (100, 5)

def test_rom_reconstruction(agent):
    """Test ROM reconstruction accuracy."""
    # Generate correlated snapshots
    t = np.linspace(0, 10, 50)
    mode1 = np.sin(t).reshape(-1, 1)
    mode2 = np.cos(t).reshape(-1, 1)

    # Create snapshots as linear combination
    snapshots = mode1 @ np.random.randn(1, 20) + mode2 @ np.random.randn(1, 20)

    result = agent.execute({
        'problem_type': 'rom',
        'snapshots': snapshots,
        'n_modes': 2
    })

    assert result.success
    energy_captured = result.data['solution']['energy_captured']
    # Should capture most energy with 2 modes
    assert energy_captured > 0.7

    reconstruction_error = result.data['metadata']['reconstruction_error']
    assert reconstruction_error < 0.5

def test_rom_energy_capture(agent):
    """Test energy captured by ROM."""
    # Simple rank-1 matrix
    snapshots = np.outer(np.ones(50), np.arange(10))

    result = agent.execute({
        'problem_type': 'rom',
        'snapshots': snapshots,
        'n_modes': 1
    })

    assert result.success
    energy_captured = result.data['solution']['energy_captured']
    # Rank-1 matrix should be captured perfectly by 1 mode
    assert energy_captured > 0.99

# Helper methods
def test_compute_kernel_rbf(agent):
    """Test RBF kernel computation."""
    X1 = np.array([[0], [1]])
    X2 = np.array([[0], [0.5], [1]])

    K = agent._compute_kernel(X1, X2, 'rbf', 1.0)

    assert K.shape == (2, 3)
    # K[0, 0] should be 1 (same point)
    assert np.isclose(K[0, 0], 1.0)

def test_generate_polynomial_basis(agent):
    """Test polynomial basis generation."""
    samples = np.array([[1, 2], [3, 4]])
    basis = agent._generate_polynomial_basis(samples, order=2, n_dims=2)

    assert basis.shape[0] == 2
    # Should have (order+1)^n_dims terms for total degree
    assert basis.shape[1] >= 4

# Provenance
def test_provenance(agent):
    """Test provenance tracking."""
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 3])

    result = agent.execute({
        'problem_type': 'gp_regression',
        'training_x': X_train,
        'training_y': y_train
    })

    assert result.provenance is not None
    assert result.provenance.agent_name == "SurrogateModelingAgent"

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
