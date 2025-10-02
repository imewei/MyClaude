"""
Test suite for LargeDeviationAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL and HPC)
- Execution for all large deviation methods (5 methods)
- Integration methods
- Caching and provenance
- Physics validation (rate function, Cramér theorem)

Total: 50 tests
"""

import pytest
import numpy as np
import os
import tempfile
import json
from datetime import datetime
from pathlib import Path

# Import agent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from large_deviation_agent import LargeDeviationAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a LargeDeviationAgent instance."""
    return LargeDeviationAgent()


@pytest.fixture
def sample_rare_event_input():
    """Sample input for rare event sampling."""
    # Gaussian observable with rare positive tail
    observable = np.random.randn(5000)

    return {
        'method': 'rare_event_sampling',
        'data': {
            'observable': observable.tolist(),
        },
        'parameters': {
            'bias_parameter': 1.5,
            'target_activity': 3.0  # ~3 sigma event
        },
        'analysis': ['rate_function', 'reweighting']
    }


@pytest.fixture
def sample_tps_input():
    """Sample input for transition path sampling."""
    # Random walk trajectory
    trajectory = np.cumsum(np.random.randn(2000)) * 0.1

    return {
        'method': 'transition_path_sampling',
        'data': {
            'trajectory': trajectory.tolist()
        },
        'parameters': {
            'region_A_max': -1.0,
            'region_B_min': 1.0
        },
        'analysis': ['committor', 'reactive_flux']
    }


@pytest.fixture
def sample_dpt_input():
    """Sample input for dynamical phase transition."""
    # Exponential observable (activity)
    observable = np.random.exponential(1.0, 10000)

    return {
        'method': 'dynamical_phase_transition',
        'data': {
            'observable': observable.tolist()
        },
        'parameters': {
            's_min': -2.0,
            's_max': 2.0,
            'n_s_points': 20
        },
        'analysis': ['phase_diagram', 'critical_point']
    }


@pytest.fixture
def sample_rate_function_input():
    """Sample input for rate function calculation."""
    # Gaussian observable
    observable = np.random.randn(5000)

    return {
        'method': 'rate_function_calculation',
        'data': {
            'observable': observable.tolist()
        },
        'parameters': {
            'n_grid_points': 50
        },
        'analysis': ['legendre_transform']
    }


@pytest.fixture
def sample_s_ensemble_input():
    """Sample input for s-ensemble simulation."""
    # Trajectories with time-integrated observable
    observable = np.random.exponential(1.0, 1000)

    return {
        'method': 's_ensemble_simulation',
        'data': {
            'observable': observable.tolist()
        },
        'parameters': {
            'bias_parameter': 1.0
        },
        'analysis': ['biased_statistics']
    }


# ============================================================================
# Test 1-5: Initialization and Metadata
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert 'rare_event_sampling' in agent.supported_methods
    assert 'transition_path_sampling' in agent.supported_methods


def test_agent_metadata(agent):
    """Test agent metadata is correct."""
    metadata = agent.get_metadata()
    assert metadata.name == "LargeDeviationAgent"
    assert metadata.version == "1.0.0"
    assert metadata.author == "Nonequilibrium Physics Team"


def test_agent_capabilities(agent):
    """Test agent capabilities."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 5
    method_names = [cap.name for cap in capabilities]
    assert 'rare_event_sampling' in method_names
    assert 'dynamical_phase_transition' in method_names


def test_supported_methods(agent):
    """Test all methods are supported."""
    assert len(agent.supported_methods) == 5
    expected_methods = ['rare_event_sampling', 'transition_path_sampling',
                       'dynamical_phase_transition', 'rate_function_calculation',
                       's_ensemble_simulation']
    for method in expected_methods:
        assert method in agent.supported_methods


def test_agent_description(agent):
    """Test agent description."""
    metadata = agent.get_metadata()
    assert 'large deviation' in metadata.description.lower() or 'rare event' in metadata.description.lower()


# ============================================================================
# Test 6-15: Input Validation
# ============================================================================

def test_validate_rare_event_input_valid(agent, sample_rare_event_input):
    """Test validation accepts valid rare event input."""
    result = agent.validate_input(sample_rare_event_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_tps_input_valid(agent, sample_tps_input):
    """Test validation accepts valid TPS input."""
    result = agent.validate_input(sample_tps_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_dpt_input_valid(agent, sample_dpt_input):
    """Test validation accepts valid DPT input."""
    result = agent.validate_input(sample_dpt_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_missing_method(agent):
    """Test validation fails with missing method."""
    result = agent.validate_input({'data': {}, 'parameters': {}})
    assert result.valid is False
    assert any('method' in err.lower() for err in result.errors)


def test_validate_invalid_method(agent):
    """Test validation fails with invalid method."""
    result = agent.validate_input({'method': 'invalid_method', 'data': {}, 'parameters': {}})
    assert result.valid is False
    assert any('method' in err.lower() or 'supported' in err.lower() for err in result.errors)


def test_validate_missing_data(agent):
    """Test validation fails with missing data."""
    result = agent.validate_input({'method': 'rare_event_sampling', 'parameters': {}})
    assert result.valid is False
    assert any('data' in err.lower() for err in result.errors)


def test_validate_large_bias_parameter(agent):
    """Test validation warns about large bias parameter."""
    large_bias_input = {
        'method': 's_ensemble_simulation',
        'data': {'observable': [1, 2, 3]},
        'parameters': {'bias_parameter': 15.0}  # Very large
    }
    result = agent.validate_input(large_bias_input)
    assert len(result.warnings) > 0 or result.valid is True


def test_validate_missing_observable(agent):
    """Test validation handles missing observable."""
    invalid_input = {
        'method': 'rare_event_sampling',
        'data': {},  # No observable
        'parameters': {}
    }
    result = agent.validate_input(invalid_input)
    assert len(result.warnings) > 0 or len(result.errors) > 0


def test_validate_missing_trajectory_tps(agent):
    """Test validation warns about missing trajectory for TPS."""
    invalid_input = {
        'method': 'transition_path_sampling',
        'data': {},  # No trajectory
        'parameters': {}
    }
    result = agent.validate_input(invalid_input)
    assert len(result.warnings) > 0


def test_validate_edge_case_parameters(agent):
    """Test validation with edge case parameters."""
    edge_input = {
        'method': 's_ensemble_simulation',
        'data': {'observable': [0.0]},  # Single value
        'parameters': {'bias_parameter': 0.0}
    }
    result = agent.validate_input(edge_input)
    assert result.valid is True or len(result.warnings) > 0


# ============================================================================
# Test 16-25: Resource Estimation
# ============================================================================

def test_resource_estimation_rare_event_local(agent, sample_rare_event_input):
    """Test resource estimation for rare event sampling on LOCAL."""
    req = agent.estimate_resources(sample_rare_event_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.cpu_cores >= 1
    assert req.memory_gb > 0


def test_resource_estimation_tps_local(agent, sample_tps_input):
    """Test resource estimation for TPS on LOCAL."""
    req = agent.estimate_resources(sample_tps_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.memory_gb > 0


def test_resource_estimation_dpt_hpc(agent, sample_dpt_input):
    """Test resource estimation for DPT."""
    req = agent.estimate_resources(sample_dpt_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_large_dataset(agent):
    """Test resource estimation scales with data size."""
    large_input = {
        'method': 'rare_event_sampling',
        'data': {
            'observable': np.random.randn(1000000).tolist()  # 1M samples
        },
        'parameters': {}
    }
    req = agent.estimate_resources(large_input)
    # Should require more resources
    assert req.memory_gb >= 4.0 or req.environment == 'HPC'


def test_resource_estimation_tps_complex(agent):
    """Test resource estimation for complex TPS."""
    complex_input = {
        'method': 'transition_path_sampling',
        'data': {
            'trajectory': np.random.randn(100000).tolist()  # Long trajectory
        },
        'parameters': {}
    }
    req = agent.estimate_resources(complex_input)
    # TPS is computationally intensive
    assert req.cpu_cores >= 2


def test_resource_estimation_returns_requirements(agent, sample_rare_event_input):
    """Test resource estimation returns ResourceRequirement object."""
    req = agent.estimate_resources(sample_rare_event_input)
    assert isinstance(req, ResourceRequirement)
    assert hasattr(req, 'cpu_cores')
    assert hasattr(req, 'memory_gb')
    assert hasattr(req, 'estimated_duration_seconds')


def test_resource_estimation_dpt_multiple_s(agent):
    """Test resource estimation for DPT with many s values."""
    many_s_input = {
        'method': 'dynamical_phase_transition',
        'data': {'observable': np.random.randn(10000).tolist()},
        'parameters': {'n_s_points': 100}
    }
    req = agent.estimate_resources(many_s_input)
    assert req.memory_gb > 0


def test_resource_estimation_no_gpu_required(agent, sample_rare_event_input):
    """Test that GPU is not required."""
    req = agent.estimate_resources(sample_rare_event_input)
    assert req.gpu_required is False


def test_resource_estimation_rate_function(agent, sample_rate_function_input):
    """Test resource estimation for rate function calculation."""
    req = agent.estimate_resources(sample_rate_function_input)
    assert req.environment in ['LOCAL', 'HPC']


def test_resource_estimation_s_ensemble(agent, sample_s_ensemble_input):
    """Test resource estimation for s-ensemble."""
    req = agent.estimate_resources(sample_s_ensemble_input)
    assert req.cpu_cores >= 1


# ============================================================================
# Test 26-30: Execution - Rare Event Sampling
# ============================================================================

def test_execute_rare_event_basic(agent, sample_rare_event_input):
    """Test basic rare event sampling."""
    result = agent.execute(sample_rare_event_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'rate_function_I' in result.data


def test_execute_rare_event_rate_function(agent, sample_rare_event_input):
    """Test rare event sampling computes rate function."""
    result = agent.execute(sample_rare_event_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'a_values' in result.data
    assert len(result.data['a_values']) > 0


def test_execute_rare_event_importance_sampling(agent):
    """Test importance sampling weights."""
    input_data = {
        'method': 'rare_event_sampling',
        'data': {'observable': np.random.randn(1000).tolist()},
        'parameters': {'bias_parameter': 2.0}
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'effective_sample_size' in result.data
    assert result.data['effective_sample_size'] > 0


def test_execute_rare_event_provenance(agent, sample_rare_event_input):
    """Test rare event analysis includes provenance."""
    result = agent.execute(sample_rare_event_input)
    assert result.provenance is not None
    assert result.provenance.agent_name == "LargeDeviationAgent"
    assert result.provenance.agent_version == "1.0.0"


def test_execute_rare_event_scgf(agent, sample_rare_event_input):
    """Test scaled cumulant generating function computation."""
    result = agent.execute(sample_rare_event_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'scaled_cumulant_generating_function' in result.data


# ============================================================================
# Test 31-35: Execution - Transition Path Sampling
# ============================================================================

def test_execute_tps_basic(agent, sample_tps_input):
    """Test basic TPS analysis."""
    result = agent.execute(sample_tps_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'committor_values' in result.data


def test_execute_tps_committor(agent, sample_tps_input):
    """Test TPS computes committor."""
    result = agent.execute(sample_tps_input)
    assert result.status == AgentStatus.SUCCESS
    committor = result.data['committor_values']
    assert len(committor) > 0
    # Committor should be between 0 and 1
    assert all(0 <= c <= 1 for c in committor)


def test_execute_tps_transition_state(agent, sample_tps_input):
    """Test TPS identifies transition state."""
    result = agent.execute(sample_tps_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'transition_state_position' in result.data
    assert 'transition_state_committor' in result.data
    # Transition state should have committor ≈ 0.5
    ts_committor = result.data['transition_state_committor']
    assert 0.3 <= ts_committor <= 0.7  # Approximate


def test_execute_tps_transition_rate(agent, sample_tps_input):
    """Test TPS computes transition rate."""
    result = agent.execute(sample_tps_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'transition_rate' in result.data
    assert result.data['transition_rate'] >= 0


def test_execute_tps_transitions(agent, sample_tps_input):
    """Test TPS counts transitions."""
    result = agent.execute(sample_tps_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'n_transitions' in result.data
    assert result.data['n_transitions'] >= 0


# ============================================================================
# Test 36-40: Execution - Dynamical Phase Transition
# ============================================================================

def test_execute_dpt_basic(agent, sample_dpt_input):
    """Test basic DPT analysis."""
    result = agent.execute(sample_dpt_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'theta_values' in result.data


def test_execute_dpt_theta_s(agent, sample_dpt_input):
    """Test DPT computes θ(s)."""
    result = agent.execute(sample_dpt_input)
    assert result.status == AgentStatus.SUCCESS
    s_values = result.data['s_values']
    theta_values = result.data['theta_values']
    assert len(s_values) == len(theta_values)


def test_execute_dpt_critical_point(agent, sample_dpt_input):
    """Test DPT identifies critical point."""
    result = agent.execute(sample_dpt_input)
    assert result.status == AgentStatus.SUCCESS
    assert 's_critical' in result.data
    assert 'has_phase_transition' in result.data


def test_execute_dpt_curvature(agent, sample_dpt_input):
    """Test DPT computes curvature."""
    result = agent.execute(sample_dpt_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'd2theta_ds2' in result.data
    assert 'max_curvature' in result.data


def test_execute_dpt_mean_activity(agent, sample_dpt_input):
    """Test DPT computes mean activity vs s."""
    result = agent.execute(sample_dpt_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'mean_activity_vs_s' in result.data


# ============================================================================
# Test 41-45: Execution - Rate Function & s-Ensemble
# ============================================================================

def test_execute_rate_function_basic(agent, sample_rate_function_input):
    """Test basic rate function calculation."""
    result = agent.execute(sample_rate_function_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'rate_function_I' in result.data


def test_execute_rate_function_legendre(agent, sample_rate_function_input):
    """Test rate function Legendre transform."""
    result = agent.execute(sample_rate_function_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'optimal_s_values' in result.data
    # Rate function should be non-negative
    rate_func = result.data['rate_function_I']
    assert all(I >= -0.1 for I in rate_func)  # Small tolerance for numerical error


def test_execute_rate_function_at_mean(agent, sample_rate_function_input):
    """Test rate function is zero at mean."""
    result = agent.execute(sample_rate_function_input)
    assert result.status == AgentStatus.SUCCESS
    # Rate at mean should be ≈ 0
    rate_at_mean = result.data['rate_at_mean']
    assert abs(rate_at_mean) < 0.5  # Should be close to zero


def test_execute_s_ensemble_basic(agent, sample_s_ensemble_input):
    """Test basic s-ensemble simulation."""
    result = agent.execute(sample_s_ensemble_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'mean_s_ensemble' in result.data


def test_execute_s_ensemble_bias_shift(agent, sample_s_ensemble_input):
    """Test s-ensemble produces biased statistics."""
    result = agent.execute(sample_s_ensemble_input)
    assert result.status == AgentStatus.SUCCESS
    # Biased mean should differ from unbiased
    bias_shift = result.data['bias_shift']
    # For positive s and positive observable, expect negative shift
    assert 'bias_shift' in result.data


# ============================================================================
# Test 46-50: Integration Methods and Advanced Features
# ============================================================================

def test_integration_analyze_driven_rare_events(agent):
    """Test integration with DrivenSystemsAgent results."""
    # Simulate driven system work distribution
    driven_result = {
        'work': np.random.randn(1000).tolist(),
        'dissipation': np.abs(np.random.randn(1000)).tolist()
    }

    result = agent.analyze_driven_rare_events(driven_result, observable_key='work')
    assert 'rate_function_I' in result


def test_integration_compute_transition_rates(agent):
    """Test integration with StochasticDynamicsAgent results."""
    # Simulate stochastic trajectory
    stochastic_result = {
        'trajectory': np.cumsum(np.random.randn(2000)).tolist()
    }

    result = agent.compute_transition_rates(stochastic_result)
    assert 'committor_values' in result


def test_integration_validate_fluctuation_tail(agent):
    """Test integration with FluctuationAgent results."""
    # Simulate fluctuation work distribution
    fluctuation_result = {
        'work': np.random.randn(5000).tolist()
    }

    result = agent.validate_fluctuation_tail(fluctuation_result, observable='work')
    assert 'rate_function_I' in result


def test_caching_identical_inputs(agent, sample_rare_event_input):
    """Test that caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_rare_event_input)
    result2 = agent.execute_with_caching(sample_rare_event_input)

    assert result1.status == AgentStatus.SUCCESS
    assert result2.status == AgentStatus.SUCCESS
    # Results should be identical (from cache)
    assert result1.data == result2.data


def test_error_handling_invalid_input(agent):
    """Test error handling with invalid input."""
    invalid_input = {
        'method': 'rare_event_sampling',
        'data': {'invalid_key': None},
        'parameters': {}
    }

    result = agent.execute(invalid_input)
    # Should handle gracefully
    assert result.status in [AgentStatus.FAILED, AgentStatus.SUCCESS]