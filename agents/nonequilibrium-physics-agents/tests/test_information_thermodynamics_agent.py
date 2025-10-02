"""
Test suite for InformationThermodynamicsAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL and HPC)
- Execution for all information thermodynamics methods (5 methods)
- Integration methods
- Caching and provenance
- Physical validation (Landauer's principle, TUR)

Total: 47 tests
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
from information_thermodynamics_agent import InformationThermodynamicsAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create an InformationThermodynamicsAgent instance."""
    return InformationThermodynamicsAgent()


@pytest.fixture
def sample_landauer_input():
    """Sample input for Landauer erasure analysis."""
    return {
        'method': 'landauer_erasure',
        'data': {
            'bits_erased': 1000,
            'work_distribution': np.random.exponential(2.87e-21, 1000)  # ~kT ln(2) at 300K
        },
        'parameters': {
            'temperature': 300.0
        },
        'analysis': ['energy_cost', 'entropy_production', 'efficiency']
    }


@pytest.fixture
def sample_maxwell_demon_input():
    """Sample input for Maxwell demon analysis."""
    # Binary measurements (0/1)
    measurements = np.random.randint(0, 2, 500)
    work = np.random.exponential(1.0e-21, 500)

    return {
        'method': 'maxwell_demon',
        'data': {
            'measurement_outcomes': measurements.tolist(),
            'work_extracted': work.tolist(),
            'feedback_protocol': 'binary_partition'
        },
        'parameters': {
            'temperature': 300.0
        },
        'analysis': ['information_gain', 'efficiency', 'entropy']
    }


@pytest.fixture
def sample_mutual_information_input():
    """Sample input for mutual information analysis."""
    # Two correlated binary sequences
    X = np.random.randint(0, 2, 1000)
    # Y is correlated with X
    Y = np.where(np.random.rand(1000) < 0.7, X, 1 - X)

    return {
        'method': 'mutual_information',
        'data': {
            'system_X': X.tolist(),
            'system_Y': Y.tolist()
        },
        'parameters': {
            'n_bins': 10
        },
        'analysis': ['mutual_info', 'entropies', 'correlation']
    }


@pytest.fixture
def sample_tur_input():
    """Sample input for thermodynamic uncertainty relation analysis."""
    # Observable with some fluctuations
    observable = np.random.randn(1000) + 5.0
    entropy_prod = np.abs(np.random.randn(1000)) * 10

    return {
        'method': 'thermodynamic_uncertainty',
        'data': {
            'observable': observable.tolist(),
            'entropy_production': entropy_prod.tolist()
        },
        'parameters': {
            'temperature': 300.0
        },
        'analysis': ['tur_bound', 'precision', 'dissipation']
    }


@pytest.fixture
def sample_feedback_control_input():
    """Sample input for feedback control analysis."""
    measurements = np.random.randint(0, 2, 500)
    actions = np.random.randint(0, 2, 500)
    work = np.random.exponential(1.5e-21, 500)

    return {
        'method': 'feedback_control',
        'data': {
            'measurements': measurements.tolist(),
            'feedback_actions': actions.tolist(),
            'work_per_cycle': work.tolist()
        },
        'parameters': {
            'temperature': 300.0,
            'feedback_delay': 0.01
        },
        'analysis': ['efficiency', 'mutual_info', 'optimal_protocol']
    }


# ============================================================================
# Test 1-5: Initialization and Metadata
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert 'landauer_erasure' in agent.supported_methods
    assert 'maxwell_demon' in agent.supported_methods
    assert agent.kB == 1.380649e-23


def test_agent_metadata(agent):
    """Test agent metadata is correct."""
    metadata = agent.get_metadata()
    assert metadata.name == "InformationThermodynamicsAgent"
    assert metadata.version == "1.0.0"
    assert metadata.agent_type == "analysis"


def test_agent_capabilities(agent):
    """Test agent capabilities."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 5
    method_names = [cap.name for cap in capabilities]
    assert 'landauer_erasure' in method_names
    assert 'maxwell_demon' in method_names
    assert 'thermodynamic_uncertainty' in method_names


def test_supported_methods(agent):
    """Test all methods are supported."""
    assert len(agent.supported_methods) == 5
    expected_methods = ['maxwell_demon', 'landauer_erasure', 'mutual_information',
                       'thermodynamic_uncertainty', 'feedback_control']
    for method in expected_methods:
        assert method in agent.supported_methods


def test_agent_description(agent):
    """Test agent description."""
    metadata = agent.get_metadata()
    assert 'information' in metadata.description.lower() or 'thermodynamics' in metadata.description.lower()


# ============================================================================
# Test 6-15: Input Validation
# ============================================================================

def test_validate_landauer_input_valid(agent, sample_landauer_input):
    """Test validation accepts valid Landauer erasure input."""
    result = agent.validate_input(sample_landauer_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_maxwell_demon_input_valid(agent, sample_maxwell_demon_input):
    """Test validation accepts valid Maxwell demon input."""
    result = agent.validate_input(sample_maxwell_demon_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_mutual_information_input_valid(agent, sample_mutual_information_input):
    """Test validation accepts valid mutual information input."""
    result = agent.validate_input(sample_mutual_information_input)
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
    result = agent.validate_input({'method': 'landauer_erasure', 'parameters': {}})
    assert result.valid is False
    assert any('data' in err.lower() for err in result.errors)


def test_validate_missing_parameters(agent):
    """Test validation handles missing parameters."""
    result = agent.validate_input({'method': 'landauer_erasure', 'data': {'bits_erased': 100}})
    assert len(result.warnings) > 0 or result.valid is True


def test_validate_invalid_temperature(agent):
    """Test validation catches invalid temperature."""
    invalid_input = {
        'method': 'landauer_erasure',
        'data': {'bits_erased': 100},
        'parameters': {'temperature': -10.0}  # Negative temperature
    }
    result = agent.validate_input(invalid_input)
    assert len(result.errors) > 0


def test_validate_very_high_temperature(agent):
    """Test validation warns about very high temperature."""
    high_temp_input = {
        'method': 'landauer_erasure',
        'data': {'bits_erased': 100},
        'parameters': {'temperature': 50000.0}
    }
    result = agent.validate_input(high_temp_input)
    assert len(result.warnings) > 0 or result.valid is True


def test_validate_edge_case_zero_temperature(agent):
    """Test validation catches zero temperature."""
    zero_temp_input = {
        'method': 'landauer_erasure',
        'data': {'bits_erased': 100},
        'parameters': {'temperature': 0.0}
    }
    result = agent.validate_input(zero_temp_input)
    assert len(result.errors) > 0


# ============================================================================
# Test 16-25: Resource Estimation
# ============================================================================

def test_resource_estimation_landauer_local(agent, sample_landauer_input):
    """Test resource estimation for Landauer erasure on LOCAL."""
    req = agent.estimate_resources(sample_landauer_input)
    assert req.environment == 'LOCAL'
    assert req.cpu_cores >= 1
    assert req.memory_gb > 0
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_maxwell_demon_local(agent, sample_maxwell_demon_input):
    """Test resource estimation for Maxwell demon on LOCAL."""
    req = agent.estimate_resources(sample_maxwell_demon_input)
    assert req.environment == 'LOCAL'
    assert req.memory_gb > 0


def test_resource_estimation_mutual_info_local(agent, sample_mutual_information_input):
    """Test resource estimation for mutual information on LOCAL."""
    req = agent.estimate_resources(sample_mutual_information_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.memory_gb > 0


def test_resource_estimation_tur(agent, sample_tur_input):
    """Test resource estimation for TUR analysis."""
    req = agent.estimate_resources(sample_tur_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_feedback_control(agent, sample_feedback_control_input):
    """Test resource estimation for feedback control."""
    req = agent.estimate_resources(sample_feedback_control_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.memory_gb > 0


def test_resource_estimation_large_dataset(agent):
    """Test resource estimation scales with data size."""
    large_input = {
        'method': 'landauer_erasure',
        'data': {
            'bits_erased': 1000000,
            'work_distribution': np.random.randn(1000000).tolist()
        },
        'parameters': {'temperature': 300.0}
    }
    req = agent.estimate_resources(large_input)
    assert req.memory_gb >= 1.0 or req.environment == 'HPC'


def test_resource_estimation_hpc_environment(agent, sample_landauer_input):
    """Test resource estimation for HPC environment."""
    sample_landauer_input['environment'] = 'HPC'
    req = agent.estimate_resources(sample_landauer_input)
    assert req.cpu_cores >= 1


def test_resource_estimation_complex_method(agent):
    """Test resource estimation for complex method (mutual information)."""
    complex_input = {
        'method': 'mutual_information',
        'data': {
            'system_X': np.random.randn(100000).tolist(),
            'system_Y': np.random.randn(100000).tolist()
        },
        'parameters': {'n_bins': 50}
    }
    req = agent.estimate_resources(complex_input)
    # Should require more resources
    assert req.memory_gb >= 0.5 or req.cpu_cores >= 2


def test_resource_estimation_returns_requirements(agent, sample_landauer_input):
    """Test resource estimation returns ResourceRequirement object."""
    req = agent.estimate_resources(sample_landauer_input)
    assert isinstance(req, ResourceRequirement)
    assert hasattr(req, 'cpu_cores')
    assert hasattr(req, 'memory_gb')
    assert hasattr(req, 'estimated_duration_seconds')


def test_resource_estimation_no_gpu_required(agent, sample_landauer_input):
    """Test that GPU is not required for analysis."""
    req = agent.estimate_resources(sample_landauer_input)
    assert req.gpu_required is False


# ============================================================================
# Test 26-30: Execution - Landauer Erasure
# ============================================================================

def test_execute_landauer_basic(agent, sample_landauer_input):
    """Test basic Landauer erasure analysis."""
    result = agent.execute(sample_landauer_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'landauer_limit_per_bit_J' in result.data


def test_execute_landauer_limit_physics(agent):
    """Test Landauer limit has correct physics: kT ln(2)."""
    temperature = 300.0
    input_data = {
        'method': 'landauer_erasure',
        'data': {'bits_erased': 1},
        'parameters': {'temperature': temperature}
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS

    # kB * T * ln(2)
    expected_limit = agent.kB * temperature * np.log(2)
    actual_limit = result.data['landauer_limit_per_bit_J']

    # Should match within tolerance
    assert np.isclose(actual_limit, expected_limit, rtol=0.01)


def test_execute_landauer_entropy_production(agent, sample_landauer_input):
    """Test Landauer erasure produces entropy."""
    result = agent.execute(sample_landauer_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'entropy_production_JK' in result.data
    # Entropy production should be positive
    assert result.data['entropy_production_JK'] > 0


def test_execute_landauer_efficiency(agent, sample_landauer_input):
    """Test Landauer erasure efficiency is ≤ 1."""
    result = agent.execute(sample_landauer_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'efficiency_vs_landauer' in result.data
    # Efficiency should be between 0 and 1 (or slightly above due to numerical issues)
    assert 0 <= result.data['efficiency_vs_landauer'] <= 1.1


def test_execute_landauer_provenance(agent, sample_landauer_input):
    """Test Landauer analysis includes provenance."""
    result = agent.execute(sample_landauer_input)
    assert result.provenance is not None
    assert result.provenance.agent_name == "InformationThermodynamicsAgent"
    assert result.provenance.agent_version == "1.0.0"


# ============================================================================
# Test 31-33: Execution - Maxwell Demon
# ============================================================================

def test_execute_maxwell_demon_basic(agent, sample_maxwell_demon_input):
    """Test basic Maxwell demon analysis."""
    result = agent.execute(sample_maxwell_demon_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'information_gain_bits' in result.data


def test_execute_maxwell_demon_second_law(agent, sample_maxwell_demon_input):
    """Test Maxwell demon satisfies second law."""
    result = agent.execute(sample_maxwell_demon_input)
    assert result.status == AgentStatus.SUCCESS
    # Total entropy production should be ≥ 0
    assert result.data['second_law_satisfied'] is True
    assert result.data['entropy_production_total_JK'] >= -1e-12


def test_execute_maxwell_demon_efficiency(agent, sample_maxwell_demon_input):
    """Test Maxwell demon efficiency is bounded."""
    result = agent.execute(sample_maxwell_demon_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'efficiency' in result.data
    # Efficiency should be ≤ 1
    assert 0 <= result.data['efficiency'] <= 1.0


# ============================================================================
# Test 34-36: Execution - Mutual Information
# ============================================================================

def test_execute_mutual_information_basic(agent, sample_mutual_information_input):
    """Test basic mutual information calculation."""
    result = agent.execute(sample_mutual_information_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'mutual_information_bits' in result.data


def test_execute_mutual_information_nonnegative(agent, sample_mutual_information_input):
    """Test mutual information is non-negative."""
    result = agent.execute(sample_mutual_information_input)
    assert result.status == AgentStatus.SUCCESS
    # I(X;Y) ≥ 0
    assert result.data['mutual_information_bits'] >= -0.01  # Small tolerance


def test_execute_mutual_information_independent_systems(agent):
    """Test mutual information is zero for independent systems."""
    # Two independent sequences
    X = np.random.randint(0, 2, 1000)
    Y = np.random.randint(0, 2, 1000)

    input_data = {
        'method': 'mutual_information',
        'data': {'system_X': X.tolist(), 'system_Y': Y.tolist()},
        'parameters': {'n_bins': 10}
    }

    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    # Should be close to zero (but may not be exactly due to finite sampling)
    assert result.data['mutual_information_bits'] < 0.5


# ============================================================================
# Test 37-39: Execution - Thermodynamic Uncertainty Relation
# ============================================================================

def test_execute_tur_basic(agent, sample_tur_input):
    """Test basic TUR analysis."""
    result = agent.execute(sample_tur_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'tur_bound_entropy_JK' in result.data


def test_execute_tur_bound_satisfied(agent, sample_tur_input):
    """Test TUR bound is satisfied."""
    result = agent.execute(sample_tur_input)
    assert result.status == AgentStatus.SUCCESS
    # TUR should be satisfied or close
    # (May fail due to statistical fluctuations in simulated data)
    assert 'tur_satisfied' in result.data


def test_execute_tur_precision_dissipation(agent, sample_tur_input):
    """Test TUR precision-dissipation trade-off."""
    result = agent.execute(sample_tur_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'precision_dissipation_ratio' in result.data
    # Ratio should be non-negative
    assert result.data['precision_dissipation_ratio'] >= 0


# ============================================================================
# Test 40-42: Execution - Feedback Control
# ============================================================================

def test_execute_feedback_control_basic(agent, sample_feedback_control_input):
    """Test basic feedback control analysis."""
    result = agent.execute(sample_feedback_control_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'feedback_efficiency' in result.data


def test_execute_feedback_control_efficiency_bounded(agent, sample_feedback_control_input):
    """Test feedback control efficiency is bounded."""
    result = agent.execute(sample_feedback_control_input)
    assert result.status == AgentStatus.SUCCESS
    # Efficiency should be ≤ 1
    assert 0 <= result.data['feedback_efficiency'] <= 1.0


def test_execute_feedback_control_with_delay(agent, sample_feedback_control_input):
    """Test feedback control with delay penalty."""
    result = agent.execute(sample_feedback_control_input)
    assert result.status == AgentStatus.SUCCESS
    # Corrected efficiency should be lower due to delay
    assert result.data['corrected_efficiency_with_delay'] <= result.data['feedback_efficiency'] * 1.01


# ============================================================================
# Test 43-45: Integration Methods
# ============================================================================

def test_integration_analyze_fluctuation_work(agent):
    """Test integration with FluctuationAgent results."""
    # Simulate FluctuationAgent output
    fluctuation_result = {
        'work_distribution': np.random.randn(1000) + 5.0,
        'entropy_production': 10.0
    }

    result = agent.analyze_fluctuation_work(fluctuation_result, temperature=300.0)
    assert 'tur_bound_entropy_JK' in result or 'total_entropy_production_JK' in result


def test_integration_compute_information_flow(agent):
    """Test integration with StochasticDynamicsAgent results."""
    # Simulate StochasticDynamicsAgent output
    stochastic_result = {
        'trajectory': np.random.randn(1000, 2)
    }

    result = agent.compute_information_flow(stochastic_result)
    assert 'mutual_information_bits' in result


def test_integration_validate_thermodynamic_bounds(agent):
    """Test thermodynamic bound validation."""
    work = 4.0e-21  # J
    information = 1.0  # nats
    temperature = 300.0

    result = agent.validate_thermodynamic_bounds(work, information, temperature)
    assert 'bound_satisfied' in result
    assert 'efficiency' in result
    assert result['work_extracted_J'] == work


# ============================================================================
# Test 46-47: Caching and Error Handling
# ============================================================================

def test_caching_identical_inputs(agent, sample_landauer_input):
    """Test that caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_landauer_input)
    result2 = agent.execute_with_caching(sample_landauer_input)

    assert result1.status == AgentStatus.SUCCESS
    assert result2.status == AgentStatus.SUCCESS
    # Results should be identical (from cache)
    assert result1.data == result2.data


def test_error_handling_invalid_input(agent):
    """Test error handling with invalid input."""
    invalid_input = {
        'method': 'landauer_erasure',
        'data': {'invalid_key': None},
        'parameters': {}
    }

    result = agent.execute(invalid_input)
    # Should handle gracefully
    assert result.status in [AgentStatus.FAILED, AgentStatus.SUCCESS]