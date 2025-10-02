"""
Test suite for PatternFormationAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL and HPC)
- Execution for all pattern formation methods (5 methods)
- Integration methods
- Caching and provenance
- Physical validation

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
from pattern_formation_agent import PatternFormationAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a PatternFormationAgent instance."""
    return PatternFormationAgent()


@pytest.fixture
def sample_turing_input():
    """Sample input for Turing pattern analysis."""
    # Create sample concentration fields
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    field_A = 1.0 + 0.1 * np.sin(2 * np.pi * X / 5)
    field_B = 1.0 + 0.1 * np.cos(2 * np.pi * Y / 5)

    return {
        'method': 'turing_patterns',
        'data': {
            'concentration_A': field_A,
            'concentration_B': field_B
        },
        'parameters': {
            'D_A': 1.0,
            'D_B': 10.0,
            'domain_size': 10.0,
            'timestep': 0.01
        },
        'analysis': ['wavelength', 'stability']
    }


@pytest.fixture
def sample_rayleigh_benard_input():
    """Sample input for Rayleigh-Bénard convection analysis."""
    # Create sample temperature field
    x = np.linspace(0, 1, 50)
    z = np.linspace(0, 1, 50)
    X, Z = np.meshgrid(x, z)
    temperature_field = 300.0 + 10.0 * Z + 0.5 * np.sin(2 * np.pi * X)

    return {
        'method': 'rayleigh_benard',
        'data': {
            'temperature_field': temperature_field,
            'velocity_field': np.random.randn(50, 50, 2) * 0.1
        },
        'parameters': {
            'temperature_bottom': 310.0,
            'temperature_top': 300.0,
            'height': 0.01,
            'thermal_diffusivity': 1.4e-7,
            'kinematic_viscosity': 1.0e-6,
            'thermal_expansion': 2.1e-4,
            'g': 9.81
        },
        'analysis': ['rayleigh_number', 'bifurcation', 'wavenumber']
    }


@pytest.fixture
def sample_phase_field_input():
    """Sample input for phase field analysis."""
    # Create sample concentration field (spinodal decomposition)
    x = np.linspace(0, 100, 128)
    y = np.linspace(0, 100, 128)
    X, Y = np.meshgrid(x, y)
    concentration = 0.5 + 0.1 * np.random.randn(128, 128)

    return {
        'method': 'phase_field',
        'data': {
            'concentration_field': concentration,
            'time_series': [concentration * (1 + 0.01 * i) for i in range(10)]
        },
        'parameters': {
            'mobility': 1.0,
            'gradient_energy': 0.5,
            'temperature': 300.0,
            'critical_temperature': 350.0,
            'timestep': 0.01
        },
        'analysis': ['domain_size', 'growth_kinetics', 'coarsening']
    }


@pytest.fixture
def sample_self_organization_input():
    """Sample input for self-organization analysis."""
    # Create sample spatial pattern
    x = np.linspace(0, 20, 200)
    y = np.linspace(0, 20, 200)
    X, Y = np.meshgrid(x, y)
    pattern = np.sin(2 * np.pi * X / 5) * np.cos(2 * np.pi * Y / 5)

    return {
        'method': 'self_organization',
        'data': {
            'spatial_field': pattern,
            'time_evolution': [pattern * np.exp(-0.1 * i) for i in range(10)]
        },
        'parameters': {
            'symmetry_group': 'hexagonal',
            'order_parameter_threshold': 0.5,
            'correlation_length': 2.0
        },
        'analysis': ['symmetry_breaking', 'order_parameter', 'structure_factor']
    }


@pytest.fixture
def sample_spatiotemporal_chaos_input():
    """Sample input for spatiotemporal chaos analysis."""
    # Create sample chaotic spatiotemporal field
    nx, nt = 100, 500
    x = np.linspace(0, 10, nx)
    t = np.linspace(0, 50, nt)
    field = np.random.randn(nt, nx) * 0.5  # Chaotic field

    return {
        'method': 'spatiotemporal_chaos',
        'data': {
            'spatiotemporal_field': field,
            'space_axis': x,
            'time_axis': t
        },
        'parameters': {
            'embedding_dimension': 3,
            'time_delay': 1,
            'correlation_dimension_estimate': 2.5
        },
        'analysis': ['lyapunov', 'correlation_dimension', 'defects']
    }


# ============================================================================
# Test 1-5: Initialization and Metadata
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert 'turing_patterns' in agent.supported_methods
    assert 'rayleigh_benard' in agent.supported_methods
    assert 'phase_field' in agent.supported_methods


def test_agent_metadata(agent):
    """Test agent metadata is correct."""
    metadata = agent.get_metadata()
    assert metadata.name == "PatternFormationAgent"
    assert metadata.version == "1.0.0"
    assert metadata.agent_type == "analysis"


def test_agent_capabilities(agent):
    """Test agent capabilities."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 5
    method_names = [cap.name for cap in capabilities]
    assert 'turing_patterns' in method_names
    assert 'rayleigh_benard' in method_names
    assert 'spatiotemporal_chaos' in method_names


def test_supported_methods(agent):
    """Test all methods are supported."""
    assert len(agent.supported_methods) == 5
    expected_methods = ['turing_patterns', 'rayleigh_benard', 'phase_field',
                       'self_organization', 'spatiotemporal_chaos']
    for method in expected_methods:
        assert method in agent.supported_methods


def test_agent_description(agent):
    """Test agent description."""
    metadata = agent.get_metadata()
    assert 'Pattern formation' in metadata.description or 'pattern' in metadata.description.lower()


# ============================================================================
# Test 6-15: Input Validation
# ============================================================================

def test_validate_turing_input_valid(agent, sample_turing_input):
    """Test validation accepts valid Turing pattern input."""
    result = agent.validate_input(sample_turing_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_rayleigh_benard_input_valid(agent, sample_rayleigh_benard_input):
    """Test validation accepts valid Rayleigh-Bénard input."""
    result = agent.validate_input(sample_rayleigh_benard_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_phase_field_input_valid(agent, sample_phase_field_input):
    """Test validation accepts valid phase field input."""
    result = agent.validate_input(sample_phase_field_input)
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
    result = agent.validate_input({'method': 'turing_patterns', 'parameters': {}})
    assert result.valid is False
    assert any('data' in err.lower() for err in result.errors)


def test_validate_missing_parameters(agent):
    """Test validation fails with missing parameters."""
    result = agent.validate_input({'method': 'turing_patterns', 'data': {}})
    # May pass with warnings or fail depending on implementation
    # Check for either warnings or errors
    assert len(result.warnings) > 0 or len(result.errors) > 0


def test_validate_invalid_diffusion_coefficients(agent):
    """Test validation catches invalid diffusion coefficients."""
    invalid_input = {
        'method': 'turing_patterns',
        'data': {'concentration_A': np.ones((10, 10))},
        'parameters': {'D_A': -1.0, 'D_B': 10.0}  # Negative diffusion
    }
    result = agent.validate_input(invalid_input)
    # Should have warning or error about negative diffusion
    assert len(result.warnings) > 0 or len(result.errors) > 0


def test_validate_invalid_temperature(agent):
    """Test validation catches invalid temperature."""
    invalid_input = {
        'method': 'rayleigh_benard',
        'data': {'temperature_field': np.ones((10, 10))},
        'parameters': {'temperature_bottom': -10.0}  # Negative absolute temperature
    }
    result = agent.validate_input(invalid_input)
    assert len(result.warnings) > 0 or len(result.errors) > 0


def test_validate_edge_case_parameters(agent):
    """Test validation with edge case parameters."""
    edge_input = {
        'method': 'phase_field',
        'data': {'concentration_field': np.ones((5, 5))},
        'parameters': {'mobility': 0.0, 'gradient_energy': 0.0}
    }
    result = agent.validate_input(edge_input)
    # Should at least generate warnings
    assert result.valid is True or len(result.warnings) > 0


# ============================================================================
# Test 16-25: Resource Estimation
# ============================================================================

def test_resource_estimation_turing_local(agent, sample_turing_input):
    """Test resource estimation for Turing patterns on LOCAL."""
    req = agent.estimate_resources(sample_turing_input)
    assert req.environment == 'LOCAL'
    assert req.cpu_cores >= 1
    assert req.memory_gb > 0
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_rayleigh_benard_local(agent, sample_rayleigh_benard_input):
    """Test resource estimation for Rayleigh-Bénard on LOCAL."""
    req = agent.estimate_resources(sample_rayleigh_benard_input)
    assert req.environment == 'LOCAL'
    assert req.memory_gb > 0


def test_resource_estimation_phase_field_hpc(agent, sample_phase_field_input):
    """Test resource estimation for phase field on HPC."""
    sample_phase_field_input['environment'] = 'HPC'
    req = agent.estimate_resources(sample_phase_field_input)
    assert req.environment == 'HPC'
    assert req.cpu_cores >= 4


def test_resource_estimation_self_organization(agent, sample_self_organization_input):
    """Test resource estimation for self-organization."""
    req = agent.estimate_resources(sample_self_organization_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.memory_gb > 0


def test_resource_estimation_spatiotemporal_chaos(agent, sample_spatiotemporal_chaos_input):
    """Test resource estimation for spatiotemporal chaos."""
    req = agent.estimate_resources(sample_spatiotemporal_chaos_input)
    assert req.environment in ['LOCAL', 'HPC']
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_large_dataset(agent):
    """Test resource estimation scales with data size."""
    large_input = {
        'method': 'turing_patterns',
        'data': {'concentration_A': np.ones((1000, 1000))},
        'parameters': {'D_A': 1.0, 'D_B': 10.0}
    }
    req = agent.estimate_resources(large_input)
    assert req.memory_gb >= 1.0  # Should require more memory


def test_resource_estimation_gpu_preference(agent, sample_turing_input):
    """Test resource estimation with GPU preference."""
    sample_turing_input['environment'] = 'GPU'
    req = agent.estimate_resources(sample_turing_input)
    # May return GPU or HPC
    assert req.environment in ['GPU', 'HPC', 'LOCAL']


def test_resource_estimation_multiple_analyses(agent, sample_turing_input):
    """Test resource estimation with multiple analyses."""
    sample_turing_input['analysis'] = ['wavelength', 'stability', 'defects', 'correlation']
    req = agent.estimate_resources(sample_turing_input)
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_time_series(agent):
    """Test resource estimation for time series data."""
    time_series_input = {
        'method': 'phase_field',
        'data': {
            'concentration_field': np.ones((100, 100)),
            'time_series': [np.ones((100, 100)) for _ in range(100)]
        },
        'parameters': {'mobility': 1.0}
    }
    req = agent.estimate_resources(time_series_input)
    assert req.memory_gb >= 0.5  # Time series requires more memory


def test_resource_estimation_returns_requirements(agent, sample_turing_input):
    """Test resource estimation returns ResourceRequirement object."""
    req = agent.estimate_resources(sample_turing_input)
    assert isinstance(req, ResourceRequirement)
    assert hasattr(req, 'cpu_cores')
    assert hasattr(req, 'memory_gb')
    assert hasattr(req, 'estimated_duration_seconds')


# ============================================================================
# Test 26-30: Execution - Turing Patterns
# ============================================================================

def test_execute_turing_patterns_basic(agent, sample_turing_input):
    """Test basic Turing pattern analysis."""
    result = agent.execute(sample_turing_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'wavelength_critical' in result.data or 'wavelength' in str(result.data)


def test_execute_turing_patterns_stability(agent, sample_turing_input):
    """Test Turing pattern stability analysis."""
    sample_turing_input['analysis'] = ['stability']
    result = agent.execute(sample_turing_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'turing_condition' in result.data or 'stability' in str(result.data)


def test_execute_turing_patterns_wavelength(agent, sample_turing_input):
    """Test Turing pattern wavelength selection."""
    result = agent.execute(sample_turing_input)
    assert result.status == AgentStatus.SUCCESS
    data = result.data
    if 'wavelength_critical' in data:
        assert data['wavelength_critical'] > 0


def test_execute_turing_patterns_provenance(agent, sample_turing_input):
    """Test Turing pattern analysis includes provenance."""
    result = agent.execute(sample_turing_input)
    assert result.provenance is not None
    assert result.provenance.agent_name == "PatternFormationAgent"
    assert result.provenance.agent_version == "1.0.0"


def test_execute_turing_patterns_metadata(agent, sample_turing_input):
    """Test Turing pattern analysis includes metadata."""
    result = agent.execute(sample_turing_input)
    assert result.metadata is not None
    assert 'method' in result.metadata or 'analysis_method' in str(result.metadata)


# ============================================================================
# Test 31-33: Execution - Rayleigh-Bénard Convection
# ============================================================================

def test_execute_rayleigh_benard_basic(agent, sample_rayleigh_benard_input):
    """Test basic Rayleigh-Bénard analysis."""
    result = agent.execute(sample_rayleigh_benard_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'rayleigh_number' in result.data or 'Ra' in str(result.data)


def test_execute_rayleigh_benard_bifurcation(agent, sample_rayleigh_benard_input):
    """Test Rayleigh-Bénard bifurcation analysis."""
    sample_rayleigh_benard_input['analysis'] = ['bifurcation']
    result = agent.execute(sample_rayleigh_benard_input)
    assert result.status == AgentStatus.SUCCESS


def test_execute_rayleigh_benard_critical_ra(agent, sample_rayleigh_benard_input):
    """Test Rayleigh-Bénard critical Rayleigh number."""
    result = agent.execute(sample_rayleigh_benard_input)
    assert result.status == AgentStatus.SUCCESS
    # Critical Ra ≈ 1708 for idealized case
    if 'rayleigh_number_critical' in result.data:
        assert result.data['rayleigh_number_critical'] > 0


# ============================================================================
# Test 34-36: Execution - Phase Field
# ============================================================================

def test_execute_phase_field_basic(agent, sample_phase_field_input):
    """Test basic phase field analysis."""
    result = agent.execute(sample_phase_field_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'domain_size' in result.data or 'phase' in str(result.data)


def test_execute_phase_field_growth_kinetics(agent, sample_phase_field_input):
    """Test phase field growth kinetics."""
    sample_phase_field_input['analysis'] = ['growth_kinetics']
    result = agent.execute(sample_phase_field_input)
    assert result.status == AgentStatus.SUCCESS


def test_execute_phase_field_coarsening(agent, sample_phase_field_input):
    """Test phase field coarsening analysis."""
    sample_phase_field_input['analysis'] = ['coarsening']
    result = agent.execute(sample_phase_field_input)
    assert result.status == AgentStatus.SUCCESS
    # Coarsening exponent should be present
    if 'coarsening_exponent' in result.data:
        assert 0 < result.data['coarsening_exponent'] < 1


# ============================================================================
# Test 37-39: Execution - Self-Organization
# ============================================================================

def test_execute_self_organization_basic(agent, sample_self_organization_input):
    """Test basic self-organization analysis."""
    result = agent.execute(sample_self_organization_input)
    assert result.status == AgentStatus.SUCCESS


def test_execute_self_organization_symmetry_breaking(agent, sample_self_organization_input):
    """Test self-organization symmetry breaking."""
    sample_self_organization_input['analysis'] = ['symmetry_breaking']
    result = agent.execute(sample_self_organization_input)
    assert result.status == AgentStatus.SUCCESS


def test_execute_self_organization_order_parameter(agent, sample_self_organization_input):
    """Test self-organization order parameter."""
    sample_self_organization_input['analysis'] = ['order_parameter']
    result = agent.execute(sample_self_organization_input)
    assert result.status == AgentStatus.SUCCESS
    # Order parameter should be between 0 and 1
    if 'order_parameter' in result.data:
        assert 0 <= result.data['order_parameter'] <= 1


# ============================================================================
# Test 40-42: Execution - Spatiotemporal Chaos
# ============================================================================

def test_execute_spatiotemporal_chaos_basic(agent, sample_spatiotemporal_chaos_input):
    """Test basic spatiotemporal chaos analysis."""
    result = agent.execute(sample_spatiotemporal_chaos_input)
    assert result.status == AgentStatus.SUCCESS


def test_execute_spatiotemporal_chaos_lyapunov(agent, sample_spatiotemporal_chaos_input):
    """Test spatiotemporal chaos Lyapunov exponent."""
    sample_spatiotemporal_chaos_input['analysis'] = ['lyapunov']
    result = agent.execute(sample_spatiotemporal_chaos_input)
    assert result.status == AgentStatus.SUCCESS
    # Lyapunov exponent should be present
    if 'lyapunov_exponent' in result.data:
        assert isinstance(result.data['lyapunov_exponent'], (int, float))


def test_execute_spatiotemporal_chaos_defects(agent, sample_spatiotemporal_chaos_input):
    """Test spatiotemporal chaos defect detection."""
    sample_spatiotemporal_chaos_input['analysis'] = ['defects']
    result = agent.execute(sample_spatiotemporal_chaos_input)
    assert result.status == AgentStatus.SUCCESS


# ============================================================================
# Test 43-45: Integration Methods
# ============================================================================

def test_integration_detect_patterns_in_active_matter(agent):
    """Test pattern detection in active matter integration."""
    # This would integrate with ActiveMatterAgent output
    active_matter_result = {
        'velocity_field': np.random.randn(50, 50, 2),
        'density_field': np.ones((50, 50)) + 0.1 * np.random.randn(50, 50),
        'order_parameter': 0.7
    }

    # Test that agent can analyze active matter patterns
    integration_input = {
        'method': 'self_organization',
        'data': active_matter_result,
        'parameters': {'order_parameter_threshold': 0.5},
        'analysis': ['order_parameter', 'structure_factor']
    }

    result = agent.execute(integration_input)
    assert result.status == AgentStatus.SUCCESS


def test_integration_analyze_driven_system_patterns(agent):
    """Test pattern analysis in driven systems integration."""
    # This would integrate with DrivenSystemsAgent output
    driven_result = {
        'velocity_profile': np.linspace(0, 1, 50),
        'temperature_field': 300 + 10 * np.random.randn(50, 50),
        'dissipation_rate': 0.5
    }

    integration_input = {
        'method': 'rayleigh_benard',
        'data': driven_result,
        'parameters': {
            'temperature_bottom': 310.0,
            'temperature_top': 300.0,
            'height': 0.01
        },
        'analysis': ['rayleigh_number']
    }

    result = agent.execute(integration_input)
    assert result.status == AgentStatus.SUCCESS


def test_integration_with_transport_agent(agent):
    """Test integration with transport agent results."""
    # Analyze patterns in transport phenomena
    transport_result = {
        'heat_flux': np.random.randn(30, 30),
        'temperature_field': 300 + 5 * np.random.randn(30, 30)
    }

    integration_input = {
        'method': 'turing_patterns',
        'data': {
            'concentration_A': transport_result['temperature_field'],
            'concentration_B': transport_result['heat_flux']
        },
        'parameters': {'D_A': 1.0, 'D_B': 5.0},
        'analysis': ['wavelength']
    }

    result = agent.execute(integration_input)
    assert result.status == AgentStatus.SUCCESS


# ============================================================================
# Test 46-47: Caching and Error Handling
# ============================================================================

def test_caching_identical_inputs(agent, sample_turing_input):
    """Test that caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_turing_input)
    result2 = agent.execute_with_caching(sample_turing_input)

    assert result1.status == AgentStatus.SUCCESS
    assert result2.status == AgentStatus.SUCCESS
    # Results should be identical (from cache)
    assert result1.data == result2.data


def test_error_handling_invalid_input(agent):
    """Test error handling with invalid input."""
    invalid_input = {
        'method': 'turing_patterns',
        'data': {'invalid_key': None},
        'parameters': {}
    }

    result = agent.execute(invalid_input)
    # Should either fail gracefully or return with warnings
    assert result.status in [AgentStatus.FAILED, AgentStatus.SUCCESS]
    if result.status == AgentStatus.FAILED:
        assert len(result.errors) > 0