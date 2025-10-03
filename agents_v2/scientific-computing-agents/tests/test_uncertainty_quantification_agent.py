"""Tests for UncertaintyQuantificationAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.uncertainty_quantification_agent import UncertaintyQuantificationAgent
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return UncertaintyQuantificationAgent(config={'n_samples': 1000})

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "UncertaintyQuantificationAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 5
    cap_names = [c.name for c in caps]
    assert 'monte_carlo_sampling' in cap_names
    assert 'latin_hypercube_sampling' in cap_names
    assert 'sensitivity_analysis' in cap_names

# Validation
def test_validate_monte_carlo_valid(agent):
    data = {
        'problem_type': 'monte_carlo',
        'model': lambda x: x[0]**2 + x[1],
        'input_distributions': [
            {'type': 'normal', 'mean': 0, 'std': 1},
            {'type': 'uniform', 'low': -1, 'high': 1}
        ]
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_lhs_valid(agent):
    data = {
        'problem_type': 'lhs',
        'bounds': [[0, 1], [0, 1]],
        'n_samples': 100
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_sensitivity_valid(agent):
    data = {
        'problem_type': 'sensitivity',
        'model': lambda x: x[0] + x[1]**2,
        'input_ranges': [[-1, 1], [-1, 1]]
    }
    val = agent.validate_input(data)
    assert val.valid

# Resource estimation
def test_estimate_resources_small(agent):
    data = {
        'problem_type': 'monte_carlo',
        'n_samples': 1000
    }
    res = agent.estimate_resources(data)
    assert res.cpu_cores >= 1

def test_estimate_resources_large(agent):
    data = {
        'problem_type': 'monte_carlo',
        'n_samples': 50000
    }
    res = agent.estimate_resources(data)
    assert res.cpu_cores >= 4

# Monte Carlo Sampling
def test_monte_carlo_simple(agent):
    """Test MC on simple linear model."""
    def model(x):
        return x[0] + 2 * x[1]

    result = agent.execute({
        'problem_type': 'monte_carlo',
        'model': model,
        'input_distributions': [
            {'type': 'normal', 'mean': 0, 'std': 1},
            {'type': 'normal', 'mean': 0, 'std': 1}
        ],
        'n_samples': 1000,
        'seed': 42
    })

    assert result.success
    assert 'mean' in result.data['solution']
    assert 'std' in result.data['solution']

    # E[X1 + 2*X2] = 0, Var[X1 + 2*X2] = 1 + 4 = 5
    mean = result.data['solution']['mean']
    std = result.data['solution']['std']

    assert abs(mean) < 0.3  # Should be close to 0
    assert abs(std - np.sqrt(5)) < 0.5  # Should be close to sqrt(5)

def test_monte_carlo_statistics(agent):
    """Test that all statistics are computed."""
    def model(x):
        return x[0]**2

    result = agent.execute({
        'problem_type': 'monte_carlo',
        'model': model,
        'input_distributions': [
            {'type': 'normal', 'mean': 0, 'std': 1}
        ],
        'n_samples': 500
    })

    assert result.success
    sol = result.data['solution']

    assert 'mean' in sol
    assert 'std' in sol
    assert 'variance' in sol
    assert 'confidence_interval' in sol
    assert 'median' in sol
    assert 'skewness' in sol
    assert 'kurtosis' in sol
    assert 'percentiles' in sol

def test_monte_carlo_uniform(agent):
    """Test MC with uniform distribution."""
    def model(x):
        return x[0]

    result = agent.execute({
        'problem_type': 'monte_carlo',
        'model': model,
        'input_distributions': [
            {'type': 'uniform', 'low': 0, 'high': 10}
        ],
        'n_samples': 1000,
        'seed': 42
    })

    assert result.success
    mean = result.data['solution']['mean']
    # E[U(0,10)] = 5
    assert abs(mean - 5.0) < 1.0

# Latin Hypercube Sampling
def test_lhs_basic(agent):
    """Test basic LHS generation."""
    result = agent.execute({
        'problem_type': 'lhs',
        'bounds': [[0, 1], [0, 1]],
        'n_samples': 50,
        'seed': 42
    })

    assert result.success
    samples = result.data['solution']['samples']
    assert samples.shape == (50, 2)

    # Check bounds
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)

def test_lhs_higher_dimensions(agent):
    """Test LHS in higher dimensions."""
    result = agent.execute({
        'problem_type': 'lhs',
        'n_dimensions': 5,
        'n_samples': 100,
        'seed': 42
    })

    assert result.success
    samples = result.data['solution']['samples']
    assert samples.shape == (100, 5)

def test_lhs_discrepancy(agent):
    """Test that discrepancy is computed."""
    result = agent.execute({
        'problem_type': 'lhs',
        'bounds': [[0, 1], [0, 1], [0, 1]],
        'n_samples': 100
    })

    assert result.success
    assert 'discrepancy' in result.data['solution']
    discrepancy = result.data['solution']['discrepancy']
    assert 0 <= discrepancy <= 1

# Sensitivity Analysis
def test_sensitivity_linear(agent):
    """Test sensitivity on linear model."""
    def model(x):
        return 3 * x[0] + x[1]  # x[0] should be more important

    result = agent.execute({
        'problem_type': 'sensitivity',
        'model': model,
        'input_ranges': [[-1, 1], [-1, 1]],
        'n_samples': 500,
        'seed': 42
    })

    assert result.success
    first_order = result.data['solution']['first_order_indices']

    assert 'S1' in first_order
    assert 'S2' in first_order

    # S1 should be larger (x[0] has coefficient 3 vs 1)
    assert first_order['S1'] > first_order['S2']

def test_sensitivity_sobol_indices(agent):
    """Test that both first and total order indices are computed."""
    def model(x):
        return x[0] + x[1]**2

    result = agent.execute({
        'problem_type': 'sensitivity',
        'model': model,
        'input_ranges': [[-1, 1], [-1, 1]],
        'n_samples': 300
    })

    assert result.success
    sol = result.data['solution']

    assert 'first_order_indices' in sol
    assert 'total_order_indices' in sol

    first = sol['first_order_indices']
    total = sol['total_order_indices']

    # Both should have 2 entries
    assert len(first) == 2
    assert len(total) == 2

def test_sensitivity_metadata(agent):
    """Test sensitivity metadata."""
    def model(x):
        return np.sum(x)

    result = agent.execute({
        'problem_type': 'sensitivity',
        'model': model,
        'input_ranges': [[-1, 1], [-1, 1], [-1, 1]],
        'n_samples': 100
    })

    assert result.success
    metadata = result.data['metadata']

    assert 'n_parameters' in metadata
    assert metadata['n_parameters'] == 3
    assert 'total_evaluations' in metadata

# Confidence Intervals
def test_confidence_interval_basic(agent):
    """Test basic confidence interval computation."""
    samples = np.random.randn(1000)

    result = agent.execute({
        'problem_type': 'confidence_interval',
        'samples': samples,
        'confidence_level': 0.95
    })

    assert result.success
    sol = result.data['solution']

    assert 'mean' in sol
    assert 'std' in sol
    assert 'confidence_interval_mean' in sol
    assert 'prediction_interval' in sol

def test_confidence_interval_mean(agent):
    """Test CI for mean of known distribution."""
    # N(5, 1) samples
    samples = np.random.normal(5, 1, 500)

    result = agent.execute({
        'problem_type': 'confidence_interval',
        'samples': samples,
        'confidence_level': 0.95
    })

    assert result.success
    mean = result.data['solution']['mean']
    ci_mean = result.data['solution']['confidence_interval_mean']

    # Mean should be close to 5
    assert abs(mean - 5.0) < 0.5

    # CI should contain true mean (5)
    assert ci_mean[0] < 5.0 < ci_mean[1]

def test_confidence_interval_percentiles(agent):
    """Test percentile-based intervals."""
    samples = np.random.uniform(0, 10, 1000)

    result = agent.execute({
        'problem_type': 'confidence_interval',
        'samples': samples
    })

    assert result.success
    assert 'percentile_interval' in result.data['solution']

# Rare Event Estimation
def test_rare_event_simple(agent):
    """Test rare event probability estimation."""
    def model(x):
        return x**2

    result = agent.execute({
        'problem_type': 'rare_event',
        'model': model,
        'threshold': 4,  # P(X^2 > 4) for X ~ N(0,1)
        'n_samples': 5000,
        'n_dimensions': 1,
        'seed': 42
    })

    assert result.success
    p_failure = result.data['solution']['failure_probability']

    # For X ~ N(0,1), P(X^2 > 4) = P(|X| > 2) ≈ 0.046
    assert 0 <= p_failure <= 1
    assert abs(p_failure - 0.046) < 0.03

def test_rare_event_confidence_bounds(agent):
    """Test confidence bounds for rare event."""
    def model(x):
        return x

    result = agent.execute({
        'problem_type': 'rare_event',
        'model': model,
        'threshold': 2,
        'n_samples': 1000,
        'n_dimensions': 1
    })

    assert result.success
    assert 'confidence_bounds' in result.data['solution']

    bounds = result.data['solution']['confidence_bounds']
    assert len(bounds) == 2
    assert 0 <= bounds[0] <= bounds[1] <= 1

def test_rare_event_no_failures(agent):
    """Test rare event when no failures occur."""
    def model(x):
        return 0  # Always below threshold

    result = agent.execute({
        'problem_type': 'rare_event',
        'model': model,
        'threshold': 10,
        'n_samples': 100
    })

    assert result.success
    p_failure = result.data['solution']['failure_probability']
    assert p_failure == 0.0

    n_failures = result.data['solution']['n_failures']
    assert n_failures == 0

# Provenance
def test_provenance(agent):
    """Test provenance tracking."""
    result = agent.execute({
        'problem_type': 'lhs',
        'n_dimensions': 2,
        'n_samples': 50
    })

    assert result.provenance is not None
    assert result.provenance.agent_name == "UncertaintyQuantificationAgent"

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
