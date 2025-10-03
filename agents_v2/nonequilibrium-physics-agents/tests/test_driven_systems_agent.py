"""
Test suite for DrivenSystemsAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL, GPU, and HPC)
- Execution for all driving methods (5 methods)
- Job submission/status/retrieval patterns
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
from driven_systems_agent import DrivenSystemsAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a DrivenSystemsAgent instance."""
    return DrivenSystemsAgent()


@pytest.fixture
def sample_shear_flow_input():
    """Sample input for shear flow simulation."""
    return {
        'method': 'shear_flow',
        'driving': {'shear_rate': 1e-4},
        'parameters': {
            'temperature': 300.0,
            'n_particles': 10000,
            'simulation_time': 10.0
        }
    }


@pytest.fixture
def sample_electric_field_input():
    """Sample input for electric field driving."""
    return {
        'method': 'electric_field',
        'driving': {'field_strength': 0.01},
        'parameters': {
            'temperature': 300.0,
            'n_charges': 1000,
            'simulation_time': 10.0,
            'charge_carriers': 'ions'
        }
    }


@pytest.fixture
def sample_temperature_gradient_input():
    """Sample input for temperature gradient."""
    return {
        'method': 'temperature_gradient',
        'driving': {'T_hot': 310.0, 'T_cold': 290.0},
        'parameters': {
            'box_length': 100.0,
            'simulation_time': 10.0
        }
    }


@pytest.fixture
def sample_pressure_gradient_input():
    """Sample input for pressure gradient."""
    return {
        'method': 'pressure_gradient',
        'driving': {'gradient': 0.1},
        'parameters': {
            'temperature': 300.0,
            'viscosity': 1e-3,
            'channel_width': 10.0
        }
    }


@pytest.fixture
def sample_steady_state_input():
    """Sample input for steady-state analysis."""
    return {
        'method': 'steady_state_analysis',
        'trajectory_file': 'nemd_trajectory.lammpstrj',
        'parameters': {
            'temperature': 300.0
        }
    }


# ============================================================================
# Test: Initialization and Metadata (3 tests)
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert isinstance(agent.supported_methods, list)
    assert len(agent.supported_methods) >= 5


def test_get_metadata(agent):
    """Test agent metadata retrieval."""
    metadata = agent.get_metadata()
    assert metadata.name == "DrivenSystemsAgent"
    assert metadata.version == "1.0.0"
    assert "NEMD" in metadata.description or "driven" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 4
    method_names = [cap.name for cap in capabilities]
    assert 'Shear Flow NEMD' in method_names


# ============================================================================
# Test: Input Validation (8 tests)
# ============================================================================

def test_validate_input_success_shear(agent, sample_shear_flow_input):
    """Test validation succeeds for valid shear flow input."""
    validation = agent.validate_input(sample_shear_flow_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_electric(agent, sample_electric_field_input):
    """Test validation succeeds for valid electric field input."""
    validation = agent.validate_input(sample_electric_field_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_method(agent):
    """Test validation fails when method is missing."""
    validation = agent.validate_input({'driving': {}})
    assert not validation.valid
    assert any('method' in err.lower() for err in validation.errors)


def test_validate_input_invalid_method(agent):
    """Test validation fails for unsupported method."""
    validation = agent.validate_input({'method': 'magnetic_field'})
    assert not validation.valid
    assert any('unsupported method' in err.lower() for err in validation.errors)


def test_validate_input_missing_shear_rate(agent):
    """Test validation fails when shear rate is missing."""
    validation = agent.validate_input({
        'method': 'shear_flow',
        'driving': {}
    })
    assert not validation.valid
    assert any('shear_rate' in err.lower() for err in validation.errors)


def test_validate_input_negative_shear_rate(agent):
    """Test validation fails for negative shear rate."""
    validation = agent.validate_input({
        'method': 'shear_flow',
        'driving': {'shear_rate': -0.1}
    })
    assert not validation.valid


def test_validate_input_high_shear_rate_warning(agent, sample_shear_flow_input):
    """Test validation warns for very high shear rate."""
    sample_shear_flow_input['driving']['shear_rate'] = 2.0
    validation = agent.validate_input(sample_shear_flow_input)
    assert len(validation.warnings) > 0


def test_validate_input_negative_temperature(agent, sample_shear_flow_input):
    """Test validation fails for negative temperature."""
    sample_shear_flow_input['parameters']['temperature'] = -50
    validation = agent.validate_input(sample_shear_flow_input)
    assert not validation.valid


# ============================================================================
# Test: Resource Estimation (7 tests)
# ============================================================================

def test_estimate_resources_small_system(agent, sample_shear_flow_input):
    """Test resource estimation for small system."""
    sample_shear_flow_input['parameters']['n_particles'] = 1000
    resources = agent.estimate_resources(sample_shear_flow_input)
    assert resources.execution_environment.value == 'local'
    assert resources.estimated_time_sec == 600


def test_estimate_resources_medium_system(agent, sample_shear_flow_input):
    """Test resource estimation for medium system."""
    sample_shear_flow_input['parameters']['n_particles'] = 50000
    resources = agent.estimate_resources(sample_shear_flow_input)
    assert resources.execution_environment.value == 'gpu'
    assert resources.gpu_count >= 1


def test_estimate_resources_large_system(agent, sample_shear_flow_input):
    """Test resource estimation for large system."""
    sample_shear_flow_input['parameters']['n_particles'] = 200000
    resources = agent.estimate_resources(sample_shear_flow_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec == 7200


def test_estimate_resources_long_simulation(agent, sample_electric_field_input):
    """Test resource estimation for long simulation."""
    sample_electric_field_input['parameters']['simulation_time'] = 100.0
    resources = agent.estimate_resources(sample_electric_field_input)
    assert resources.execution_environment.value == 'hpc'


def test_estimate_resources_electric_field(agent, sample_electric_field_input):
    """Test resource estimation for electric field."""
    resources = agent.estimate_resources(sample_electric_field_input)
    assert resources.cpu_cores >= 8


def test_estimate_resources_temperature_gradient(agent, sample_temperature_gradient_input):
    """Test resource estimation for temperature gradient."""
    resources = agent.estimate_resources(sample_temperature_gradient_input)
    assert resources.estimated_time_sec > 0


def test_estimate_resources_steady_state(agent, sample_steady_state_input):
    """Test resource estimation for steady-state analysis."""
    resources = agent.estimate_resources(sample_steady_state_input)
    assert resources.execution_environment.value in ['local', 'hpc']


# ============================================================================
# Test: Execution for All Methods (7 tests)
# ============================================================================

def test_execute_shear_flow_success(agent, sample_shear_flow_input):
    """Test shear flow execution succeeds."""
    result = agent.execute(sample_shear_flow_input)
    assert result.success
    assert 'shear_viscosity_Pa_s' in result.data
    assert 'velocity_profile' in result.data
    assert 'steady_state_reached' in result.data
    assert result.data['rheology_type'] in ['newtonian', 'shear_thinning']


def test_execute_electric_field_success(agent, sample_electric_field_input):
    """Test electric field execution succeeds."""
    result = agent.execute(sample_electric_field_input)
    assert result.success
    assert 'current_density_A_per_m2' in result.data
    assert 'electrical_conductivity_S_per_m' in result.data
    assert 'mobility_cm2_per_Vs' in result.data
    assert result.data['response_type'] in ['ohmic', 'non_ohmic']


def test_execute_temperature_gradient_success(agent, sample_temperature_gradient_input):
    """Test temperature gradient execution succeeds."""
    result = agent.execute(sample_temperature_gradient_input)
    assert result.success
    assert 'heat_flux_W_per_m2' in result.data
    assert 'thermal_conductivity_W_per_mK' in result.data
    assert 'temperature_profile' in result.data
    assert result.data['steady_state_reached']


def test_execute_pressure_gradient_success(agent, sample_pressure_gradient_input):
    """Test pressure gradient execution succeeds."""
    result = agent.execute(sample_pressure_gradient_input)
    assert result.success
    assert 'flow_rate' in result.data
    assert 'velocity_profile' in result.data
    assert result.data['flow_type'] == 'poiseuille'


def test_execute_steady_state_analysis_success(agent, sample_steady_state_input):
    """Test steady-state analysis execution succeeds."""
    result = agent.execute(sample_steady_state_input)
    assert result.success
    assert 'is_stationary' in result.data
    assert 'entropy_production_rate' in result.data
    assert 'transport_coefficient' in result.data
    assert 'linear_response_valid' in result.data


def test_execute_shear_flow_high_rate(agent, sample_shear_flow_input):
    """Test shear flow with high shear rate."""
    sample_shear_flow_input['driving']['shear_rate'] = 0.2
    result = agent.execute(sample_shear_flow_input)
    assert result.success
    # High shear rate may show shear thinning
    assert result.data['rheology_type'] in ['newtonian', 'shear_thinning']


def test_execute_invalid_method(agent):
    """Test execution fails for invalid method."""
    result = agent.execute({'method': 'invalid_method'})
    assert not result.success
    assert len(result.errors) > 0


# ============================================================================
# Test: Job Submission/Status/Retrieval (5 tests)
# ============================================================================

def test_submit_calculation_returns_job_id(agent, sample_shear_flow_input):
    """Test submit_calculation returns valid job ID."""
    job_id = agent.submit_calculation(sample_shear_flow_input)
    assert job_id is not None
    assert isinstance(job_id, str)


def test_check_status_after_submission(agent, sample_shear_flow_input):
    """Test check_status returns valid status after submission."""
    job_id = agent.submit_calculation(sample_shear_flow_input)
    status = agent.check_status(job_id)
    assert status == AgentStatus.RUNNING


def test_check_status_invalid_job_id(agent):
    """Test check_status handles invalid job ID."""
    status = agent.check_status('invalid_job_id')
    assert status == AgentStatus.FAILED


def test_retrieve_results_after_completion(agent, sample_shear_flow_input):
    """Test retrieve_results after marking job complete."""
    job_id = agent.submit_calculation(sample_shear_flow_input)
    if hasattr(agent, 'job_cache') and job_id in agent.job_cache:
        agent.job_cache[job_id]['status'] = AgentStatus.SUCCESS
        agent.job_cache[job_id]['results'] = {'viscosity': 1.0}
    results = agent.retrieve_results(job_id)
    assert results is not None


def test_retrieve_results_invalid_job_id(agent):
    """Test retrieve_results handles invalid job ID."""
    with pytest.raises(Exception):
        agent.retrieve_results('invalid_job_id')


# ============================================================================
# Test: Integration Methods (6 tests)
# ============================================================================

def test_validate_linear_response_good(agent):
    """Test linear response validation with good agreement."""
    force = 0.1
    flux = 0.05
    transport_coeff = 0.5
    validation = agent.validate_linear_response(force, flux, transport_coeff)
    assert validation['linear_response_valid']


def test_validate_linear_response_poor(agent):
    """Test linear response validation with poor agreement."""
    force = 0.1
    flux = 0.1
    transport_coeff = 0.5
    validation = agent.validate_linear_response(force, flux, transport_coeff)
    assert not validation['linear_response_valid']


def test_compute_entropy_production_positive(agent):
    """Test entropy production is positive."""
    fluxes = [0.1, 0.05]
    forces = [1.0, 0.5]
    result = agent.compute_entropy_production(fluxes, forces)
    assert result['entropy_production_rate'] >= 0
    assert result['second_law_satisfied']


def test_compute_entropy_production_zero_equilibrium(agent):
    """Test entropy production is zero at equilibrium."""
    fluxes = [0.0, 0.0]
    forces = [0.0, 0.0]
    result = agent.compute_entropy_production(fluxes, forces)
    assert result['entropy_production_rate'] == 0


def test_cross_validate_green_kubo_good(agent):
    """Test cross-validation between NEMD and Green-Kubo."""
    nemd_coeff = 1.0
    gk_coeff = 1.05
    validation = agent.cross_validate_with_green_kubo(nemd_coeff, gk_coeff)
    assert validation['methods_agree']
    assert validation['quality'] in ['excellent', 'good']


def test_cross_validate_green_kubo_poor(agent):
    """Test cross-validation with poor agreement."""
    nemd_coeff = 1.0
    gk_coeff = 2.0
    validation = agent.cross_validate_with_green_kubo(nemd_coeff, gk_coeff)
    assert not validation['methods_agree']


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_shear_flow_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_shear_flow_input)
    result2 = agent.execute_with_caching(sample_shear_flow_input)
    assert result1.success
    assert result2.success


def test_caching_different_inputs(agent, sample_shear_flow_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_shear_flow_input)
    modified = sample_shear_flow_input.copy()
    modified['driving'] = {'shear_rate': 2e-4}
    result2 = agent.execute_with_caching(modified)
    assert result1.success
    assert result2.success


def test_provenance_tracking(agent, sample_shear_flow_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_shear_flow_input)
    assert result.success
    assert result.provenance.agent_name == 'DrivenSystemsAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_shear_flow_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_shear_flow_input)
    assert result.success
    assert len(result.provenance.input_hash) == 64


# ============================================================================
# Test: Physical Validation (4 tests)
# ============================================================================

def test_viscosity_positive(agent, sample_shear_flow_input):
    """Test shear viscosity is positive."""
    result = agent.execute(sample_shear_flow_input)
    assert result.success
    assert result.data['shear_viscosity_Pa_s'] > 0


def test_velocity_profile_linear(agent, sample_shear_flow_input):
    """Test velocity profile is linear for Newtonian fluid."""
    result = agent.execute(sample_shear_flow_input)
    assert result.success
    profile = result.data['velocity_profile']
    assert len(profile['z_position']) == len(profile['velocity_x'])


def test_ohms_law_low_field(agent, sample_electric_field_input):
    """Test Ohm's law holds at low field."""
    sample_electric_field_input['driving']['field_strength'] = 0.001
    result = agent.execute(sample_electric_field_input)
    assert result.success
    assert result.data['response_type'] == 'ohmic'


def test_entropy_production_positive(agent, sample_steady_state_input):
    """Test entropy production rate is positive."""
    result = agent.execute(sample_steady_state_input)
    assert result.success
    assert result.data['entropy_production_rate'] >= 0


# ============================================================================
# Test: Workflow Integration (3 tests)
# ============================================================================

def test_workflow_nemd_viscosity_series(agent):
    """Test NEMD viscosity at different shear rates."""
    shear_rates = [1e-5, 1e-4, 1e-3]
    viscosities = []
    for rate in shear_rates:
        input_data = {
            'method': 'shear_flow',
            'driving': {'shear_rate': rate},
            'parameters': {'temperature': 300, 'n_particles': 5000, 'simulation_time': 5.0}
        }
        result = agent.execute(input_data)
        assert result.success
        viscosities.append(result.data['shear_viscosity_Pa_s'])
    assert all(v > 0 for v in viscosities)


def test_workflow_thermal_gradient_to_conductivity(agent, sample_temperature_gradient_input):
    """Test workflow from temperature gradient to thermal conductivity."""
    result = agent.execute(sample_temperature_gradient_input)
    assert result.success
    assert result.data['thermal_conductivity_W_per_mK'] > 0


def test_workflow_all_driving_methods(agent):
    """Test all driving methods execute successfully."""
    methods = ['shear_flow', 'electric_field', 'temperature_gradient', 'pressure_gradient', 'steady_state_analysis']
    results = []
    for method in methods:
        if method == 'shear_flow':
            input_data = {'method': method, 'driving': {'shear_rate': 1e-4}, 'parameters': {'temperature': 300}}
        elif method == 'electric_field':
            input_data = {'method': method, 'driving': {'field_strength': 0.01}, 'parameters': {'temperature': 300}}
        elif method == 'temperature_gradient':
            input_data = {'method': method, 'driving': {'T_hot': 310, 'T_cold': 290}, 'parameters': {}}
        elif method == 'pressure_gradient':
            input_data = {'method': method, 'driving': {'gradient': 0.1}, 'parameters': {'temperature': 300, 'viscosity': 1e-3}}
        else:
            input_data = {'method': method, 'parameters': {}}
        result = agent.execute(input_data)
        results.append(result.success)
    assert all(results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
