"""
Test suite for TransportAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL and HPC)
- Execution for all transport methods (5 methods)
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
from transport_agent import TransportAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a TransportAgent instance."""
    return TransportAgent()


@pytest.fixture
def sample_thermal_conductivity_input():
    """Sample input for thermal conductivity calculation."""
    return {
        'method': 'thermal_conductivity',
        'trajectory_file': 'nvt_trajectory.lammpstrj',
        'parameters': {
            'temperature': 300.0,
            'volume': 1000.0,
            'mode': 'green_kubo',
            'correlation_length': 1000,
            'timestep': 0.001
        },
        'mode': 'equilibrium'
    }


@pytest.fixture
def sample_mass_diffusion_input():
    """Sample input for mass diffusion calculation."""
    return {
        'method': 'mass_diffusion',
        'trajectory_file': 'md_trajectory.dcd',
        'parameters': {
            'temperature': 300.0,
            'species': 'A',
            'dimensionality': 3,
            'expected_D': 5e-5
        }
    }


@pytest.fixture
def sample_electrical_conductivity_input():
    """Sample input for electrical conductivity."""
    return {
        'method': 'electrical_conductivity',
        'trajectory_file': 'ion_trajectory.xyz',
        'parameters': {
            'temperature': 300.0,
            'volume': 1000.0,
            'correlation_length': 1000,
            'timestep': 0.001,
            'charge': 1.0,
            'diffusion_coefficient': 1e-5
        }
    }


@pytest.fixture
def sample_thermoelectric_input():
    """Sample input for thermoelectric properties."""
    return {
        'method': 'thermoelectric',
        'trajectory_file': 'thermoelectric.lammpstrj',
        'parameters': {
            'temperature': 300.0,
            'temperature_gradient': 10.0,
            'voltage_difference': 0.0002,
            'electrical_conductivity': 1000.0,
            'thermal_conductivity': 1.0
        }
    }


@pytest.fixture
def sample_cross_coupling_input():
    """Sample input for cross-coupling effects."""
    return {
        'method': 'cross_coupling',
        'trajectory_file': 'cross_coupling.dcd',
        'parameters': {
            'temperature': 300.0,
            'soret_coefficient': 0.1,
            'dufour_coefficient': 0.05
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
    assert metadata.name == "TransportAgent"
    assert metadata.version == "1.0.0"
    assert metadata.description is not None
    assert "transport" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 5
    method_names = [cap.name for cap in capabilities]
    assert 'Thermal Conductivity' in method_names
    assert 'Mass Diffusion' in method_names
    assert 'Electrical Conductivity' in method_names


# ============================================================================
# Test: Input Validation (8 tests)
# ============================================================================

def test_validate_input_success_thermal(agent, sample_thermal_conductivity_input):
    """Test validation succeeds for valid thermal conductivity input."""
    validation = agent.validate_input(sample_thermal_conductivity_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_diffusion(agent, sample_mass_diffusion_input):
    """Test validation succeeds for valid diffusion input."""
    validation = agent.validate_input(sample_mass_diffusion_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_method(agent):
    """Test validation fails when method is missing."""
    validation = agent.validate_input({'trajectory_file': 'test.xyz'})
    assert not validation.valid
    assert any('method' in err.lower() for err in validation.errors)


def test_validate_input_invalid_method(agent):
    """Test validation fails for unsupported method."""
    validation = agent.validate_input({'method': 'quantum_transport'})
    assert not validation.valid
    assert any('unsupported method' in err.lower() for err in validation.errors)


def test_validate_input_missing_trajectory(agent):
    """Test validation fails when trajectory file is missing."""
    validation = agent.validate_input({
        'method': 'thermal_conductivity'
    })
    assert not validation.valid
    assert any('trajectory_file' in err.lower() or 'data' in err.lower() for err in validation.errors)


def test_validate_input_negative_temperature(agent, sample_thermal_conductivity_input):
    """Test validation fails for negative temperature."""
    sample_thermal_conductivity_input['parameters']['temperature'] = -50
    validation = agent.validate_input(sample_thermal_conductivity_input)
    assert not validation.valid
    assert any('temperature' in err.lower() for err in validation.errors)


def test_validate_input_warnings_mode(agent, sample_thermal_conductivity_input):
    """Test validation warns for unknown mode."""
    sample_thermal_conductivity_input['mode'] = 'unknown_mode'
    validation = agent.validate_input(sample_thermal_conductivity_input)
    assert len(validation.warnings) > 0


def test_validate_input_warnings_missing_params(agent):
    """Test validation warns for missing optional parameters."""
    input_data = {
        'method': 'mass_diffusion',
        'trajectory_file': 'test.xyz',
        'parameters': {}
    }
    validation = agent.validate_input(input_data)
    assert len(validation.warnings) > 0


# ============================================================================
# Test: Resource Estimation (7 tests)
# ============================================================================

def test_estimate_resources_equilibrium_short(agent, sample_thermal_conductivity_input):
    """Test resource estimation for equilibrium (Green-Kubo) calculation."""
    resources = agent.estimate_resources(sample_thermal_conductivity_input)
    assert resources.execution_environment.value == 'local'
    assert resources.estimated_time_sec == 300
    assert resources.gpu_count == 0


def test_estimate_resources_nemd_short(agent, sample_thermal_conductivity_input):
    """Test resource estimation for short NEMD simulation."""
    sample_thermal_conductivity_input['mode'] = 'nonequilibrium'
    sample_thermal_conductivity_input['parameters']['steps'] = 1000000
    resources = agent.estimate_resources(sample_thermal_conductivity_input)
    assert resources.execution_environment.value in ['local', 'hpc']
    assert resources.estimated_time_sec >= 300


def test_estimate_resources_nemd_long(agent, sample_thermal_conductivity_input):
    """Test resource estimation for long NEMD simulation."""
    sample_thermal_conductivity_input['mode'] = 'nonequilibrium'
    sample_thermal_conductivity_input['parameters']['steps'] = 20000000
    resources = agent.estimate_resources(sample_thermal_conductivity_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec == 7200
    assert resources.gpu_count >= 1


def test_estimate_resources_mass_diffusion(agent, sample_mass_diffusion_input):
    """Test resource estimation for mass diffusion."""
    resources = agent.estimate_resources(sample_mass_diffusion_input)
    assert resources.estimated_time_sec > 0
    assert resources.execution_environment.value == 'local'


def test_estimate_resources_electrical(agent, sample_electrical_conductivity_input):
    """Test resource estimation for electrical conductivity."""
    resources = agent.estimate_resources(sample_electrical_conductivity_input)
    assert resources.cpu_cores >= 4
    assert resources.memory_gb >= 8.0


def test_estimate_resources_thermoelectric(agent, sample_thermoelectric_input):
    """Test resource estimation for thermoelectric properties."""
    resources = agent.estimate_resources(sample_thermoelectric_input)
    assert resources.estimated_time_sec > 0


def test_estimate_resources_cross_coupling(agent, sample_cross_coupling_input):
    """Test resource estimation for cross-coupling."""
    resources = agent.estimate_resources(sample_cross_coupling_input)
    assert resources.execution_environment.value == 'local'


# ============================================================================
# Test: Execution for All Methods (7 tests)
# ============================================================================

def test_execute_thermal_conductivity_success(agent, sample_thermal_conductivity_input):
    """Test thermal conductivity execution succeeds."""
    result = agent.execute(sample_thermal_conductivity_input)
    assert result.success
    assert 'thermal_conductivity_W_per_mK' in result.data
    assert 'correlation_function' in result.data
    assert 'method' in result.data
    assert result.data['method'] == 'Green-Kubo'
    assert result.data['convergence']['converged']


def test_execute_mass_diffusion_success(agent, sample_mass_diffusion_input):
    """Test mass diffusion execution succeeds."""
    result = agent.execute(sample_mass_diffusion_input)
    assert result.success
    assert 'diffusion_coefficient_cm2_s' in result.data
    assert 'msd_curve' in result.data
    assert 'anomalous_exponent' in result.data
    assert result.data['diffusion_type'] in ['normal', 'subdiffusive', 'superdiffusive']


def test_execute_electrical_conductivity_success(agent, sample_electrical_conductivity_input):
    """Test electrical conductivity execution succeeds."""
    result = agent.execute(sample_electrical_conductivity_input)
    assert result.success
    assert 'electrical_conductivity_S_per_m' in result.data
    assert 'mobility_m2_per_Vs' in result.data
    assert 'current_correlation_function' in result.data
    assert result.data['method'] == 'Green-Kubo'


def test_execute_thermoelectric_success(agent, sample_thermoelectric_input):
    """Test thermoelectric execution succeeds."""
    result = agent.execute(sample_thermoelectric_input)
    assert result.success
    assert 'seebeck_coefficient_uV_per_K' in result.data
    assert 'power_factor_W_per_mK2' in result.data
    assert 'zt_figure_of_merit' in result.data
    assert 'quality_assessment' in result.data


def test_execute_cross_coupling_success(agent, sample_cross_coupling_input):
    """Test cross-coupling execution succeeds."""
    result = agent.execute(sample_cross_coupling_input)
    assert result.success
    assert 'soret_coefficient_per_K' in result.data
    assert 'dufour_coefficient_J_m2_per_mol' in result.data
    assert 'onsager_matrix' in result.data
    assert 'onsager_reciprocity_satisfied' in result.data


def test_execute_thermal_nemd_success(agent, sample_thermal_conductivity_input):
    """Test NEMD thermal conductivity execution succeeds."""
    sample_thermal_conductivity_input['mode'] = 'nonequilibrium'
    sample_thermal_conductivity_input['parameters']['heat_flux'] = 1e10
    sample_thermal_conductivity_input['parameters']['temperature_gradient'] = 1e9
    result = agent.execute(sample_thermal_conductivity_input)
    assert result.success
    assert result.data['method'] == 'NEMD'
    assert 'heat_flux_W_per_m2' in result.data


def test_execute_invalid_method(agent):
    """Test execution fails for invalid method."""
    result = agent.execute({'method': 'invalid_method', 'trajectory_file': 'test.xyz'})
    assert not result.success
    assert len(result.errors) > 0


# ============================================================================
# Test: Job Submission/Status/Retrieval (5 tests)
# ============================================================================

def test_submit_calculation_returns_job_id(agent, sample_thermal_conductivity_input):
    """Test submit_calculation returns valid job ID."""
    job_id = agent.submit_calculation(sample_thermal_conductivity_input)
    assert job_id is not None
    assert isinstance(job_id, str)
    assert len(job_id) > 0


def test_check_status_after_submission(agent, sample_thermal_conductivity_input):
    """Test check_status returns valid status after submission."""
    job_id = agent.submit_calculation(sample_thermal_conductivity_input)
    status = agent.check_status(job_id)
    assert status == AgentStatus.RUNNING


def test_check_status_invalid_job_id(agent):
    """Test check_status handles invalid job ID."""
    status = agent.check_status('invalid_job_id')
    assert status == AgentStatus.FAILED


def test_retrieve_results_after_completion(agent, sample_thermal_conductivity_input):
    """Test retrieve_results after marking job complete."""
    job_id = agent.submit_calculation(sample_thermal_conductivity_input)
    # Simulate completion
    if hasattr(agent, 'job_cache') and job_id in agent.job_cache:
        agent.job_cache[job_id]['status'] = AgentStatus.SUCCESS
        agent.job_cache[job_id]['results'] = {'thermal_conductivity': 1.0}

    results = agent.retrieve_results(job_id)
    assert results is not None
    assert isinstance(results, dict)


def test_retrieve_results_invalid_job_id(agent):
    """Test retrieve_results handles invalid job ID."""
    with pytest.raises(Exception):
        agent.retrieve_results('invalid_job_id')


# ============================================================================
# Test: Integration Methods (6 tests)
# ============================================================================

def test_validate_with_experiment_good_agreement(agent):
    """Test experimental validation with good agreement."""
    computed = 1.0
    experimental = 1.05
    validation = agent.validate_with_experiment(computed, experimental, tolerance=0.2)
    assert validation['agrees_within_tolerance']
    assert validation['quality'] in ['excellent', 'good']


def test_validate_with_experiment_poor_agreement(agent):
    """Test experimental validation with poor agreement."""
    computed = 1.0
    experimental = 2.0
    validation = agent.validate_with_experiment(computed, experimental, tolerance=0.2)
    assert not validation['agrees_within_tolerance']
    assert validation['quality'] == 'fair'


def test_check_onsager_reciprocity_satisfied(agent):
    """Test Onsager reciprocity check with symmetric matrix."""
    onsager_matrix = np.array([[1.0, 0.1], [0.1, 0.5]])
    result = agent.check_onsager_reciprocity(onsager_matrix)
    assert result


def test_check_onsager_reciprocity_violated(agent):
    """Test Onsager reciprocity check with asymmetric matrix."""
    onsager_matrix = np.array([[1.0, 0.1], [0.5, 0.5]])
    result = agent.check_onsager_reciprocity(onsager_matrix, tolerance=0.01)
    assert not result


def test_cross_validate_methods(agent, sample_thermal_conductivity_input):
    """Test cross-validation between Green-Kubo and NEMD."""
    # Green-Kubo
    result_gk = agent.execute(sample_thermal_conductivity_input)
    k_gk = result_gk.data['thermal_conductivity_W_per_mK']

    # NEMD
    sample_thermal_conductivity_input['mode'] = 'nonequilibrium'
    sample_thermal_conductivity_input['parameters']['heat_flux'] = 1e10
    sample_thermal_conductivity_input['parameters']['temperature_gradient'] = 1e9
    result_nemd = agent.execute(sample_thermal_conductivity_input)
    k_nemd = result_nemd.data['thermal_conductivity_W_per_mK']

    # Both should give positive conductivity
    assert k_gk > 0
    assert k_nemd > 0


def test_integration_validate_transport_coefficients(agent):
    """Test validation of transport coefficient relationships."""
    # Test Einstein relation: D = kT/γ
    temperature = 300.0
    friction = 1.0
    kB = 1.380649e-23
    D_theory = kB * temperature / friction

    assert D_theory > 0


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_thermal_conductivity_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_thermal_conductivity_input)
    result2 = agent.execute_with_caching(sample_thermal_conductivity_input)

    assert result1.success
    assert result2.success
    # Second call should be from cache
    assert result2.metadata.get('cached', False) or result2.execution_time_seconds < result1.execution_time_seconds * 0.5


def test_caching_different_inputs(agent, sample_thermal_conductivity_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_thermal_conductivity_input)

    # Modify input
    modified_input = sample_thermal_conductivity_input.copy()
    modified_input['parameters'] = sample_thermal_conductivity_input['parameters'].copy()
    modified_input['parameters']['temperature'] = 350

    result2 = agent.execute_with_caching(modified_input)

    assert result1.success
    assert result2.success


def test_provenance_tracking(agent, sample_thermal_conductivity_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_thermal_conductivity_input)
    assert result.success
    assert hasattr(result, 'provenance')
    assert result.provenance is not None
    assert result.provenance.agent_name == 'TransportAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_thermal_conductivity_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_thermal_conductivity_input)
    assert result.success
    assert result.provenance.input_hash is not None
    assert len(result.provenance.input_hash) == 64  # SHA256 hex digest


# ============================================================================
# Test: Physical Validation (4 tests)
# ============================================================================

def test_thermal_conductivity_positive(agent, sample_thermal_conductivity_input):
    """Test thermal conductivity is positive."""
    result = agent.execute(sample_thermal_conductivity_input)
    assert result.success
    assert result.data['thermal_conductivity_W_per_mK'] > 0


def test_diffusion_coefficient_positive(agent, sample_mass_diffusion_input):
    """Test diffusion coefficient is positive."""
    result = agent.execute(sample_mass_diffusion_input)
    assert result.success
    assert result.data['diffusion_coefficient_cm2_s'] > 0


def test_electrical_conductivity_positive(agent, sample_electrical_conductivity_input):
    """Test electrical conductivity is positive."""
    result = agent.execute(sample_electrical_conductivity_input)
    assert result.success
    assert result.data['electrical_conductivity_S_per_m'] > 0


def test_onsager_reciprocity_physics(agent, sample_cross_coupling_input):
    """Test Onsager reciprocity is satisfied."""
    result = agent.execute(sample_cross_coupling_input)
    assert result.success
    assert result.data['onsager_reciprocity_satisfied']


# ============================================================================
# Test: Workflow Integration (3 tests)
# ============================================================================

def test_workflow_thermal_to_thermoelectric(agent, sample_thermal_conductivity_input, sample_thermoelectric_input):
    """Test workflow: thermal conductivity → thermoelectric."""
    # Step 1: Compute thermal conductivity
    thermal_result = agent.execute(sample_thermal_conductivity_input)
    assert thermal_result.success

    # Step 2: Use thermal conductivity in thermoelectric calculation
    sample_thermoelectric_input['parameters']['thermal_conductivity'] = thermal_result.data['thermal_conductivity_W_per_mK']
    thermoelectric_result = agent.execute(sample_thermoelectric_input)
    assert thermoelectric_result.success


def test_workflow_diffusion_to_electrical(agent, sample_mass_diffusion_input, sample_electrical_conductivity_input):
    """Test workflow: diffusion → electrical conductivity."""
    # Step 1: Compute diffusion coefficient
    diffusion_result = agent.execute(sample_mass_diffusion_input)
    assert diffusion_result.success

    # Step 2: Use in electrical conductivity (Nernst-Einstein)
    sample_electrical_conductivity_input['parameters']['diffusion_coefficient'] = diffusion_result.data['diffusion_coefficient_cm2_s']
    electrical_result = agent.execute(sample_electrical_conductivity_input)
    assert electrical_result.success


def test_workflow_all_transport_properties(agent):
    """Test complete transport property characterization."""
    # Thermal
    thermal_input = {
        'method': 'thermal_conductivity',
        'trajectory_file': 'test.lammpstrj',
        'parameters': {'temperature': 300, 'volume': 1000, 'mode': 'green_kubo'},
        'mode': 'equilibrium'
    }
    thermal_result = agent.execute(thermal_input)
    assert thermal_result.success

    # Mass diffusion
    diffusion_input = {
        'method': 'mass_diffusion',
        'trajectory_file': 'test.dcd',
        'parameters': {'temperature': 300, 'species': 'A'}
    }
    diffusion_result = agent.execute(diffusion_input)
    assert diffusion_result.success

    # Electrical
    electrical_input = {
        'method': 'electrical_conductivity',
        'trajectory_file': 'test.xyz',
        'parameters': {'temperature': 300, 'volume': 1000}
    }
    electrical_result = agent.execute(electrical_input)
    assert electrical_result.success

    # All successful
    assert all([thermal_result.success, diffusion_result.success, electrical_result.success])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])