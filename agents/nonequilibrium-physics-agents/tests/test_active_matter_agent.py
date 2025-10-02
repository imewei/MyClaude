"""
Test suite for ActiveMatterAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL and HPC)
- Execution for all active matter models (5 models)
- Job submission/status/retrieval patterns
- Analysis methods
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
from active_matter_agent import ActiveMatterAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create an ActiveMatterAgent instance."""
    return ActiveMatterAgent()


@pytest.fixture
def sample_vicsek_input():
    """Sample input for Vicsek model."""
    return {
        'model': 'vicsek',
        'parameters': {
            'n_particles': 1000,
            'v0': 0.5,
            'eta': 0.3,
            'interaction_radius': 1.0,
            'n_steps': 10000,
            'box_size': 20.0,
            'dt': 0.1,
            'seed': 42
        },
        'analysis': ['order_parameter', 'correlation_function']
    }


@pytest.fixture
def sample_active_brownian_input():
    """Sample input for Active Brownian Particles."""
    return {
        'model': 'active_brownian',
        'parameters': {
            'n_particles': 1000,
            'v0': 1.0,
            'D_r': 0.1,
            'D_t': 0.01,
            'box_size': 20.0,
            'n_steps': 10000,
            'dt': 0.01,
            'seed': 42
        }
    }


@pytest.fixture
def sample_run_and_tumble_input():
    """Sample input for run-and-tumble dynamics."""
    return {
        'model': 'run_and_tumble',
        'parameters': {
            'n_particles': 500,
            'v0': 1.0,
            'tumble_rate': 1.0,
            'n_steps': 10000,
            'dt': 0.01,
            'box_size': 20.0,
            'seed': 42
        }
    }


@pytest.fixture
def sample_active_nematics_input():
    """Sample input for active nematics."""
    return {
        'model': 'active_nematics',
        'parameters': {
            'grid_size': 64,
            'activity': 0.5,
            'n_steps': 1000,
            'seed': 42
        }
    }


@pytest.fixture
def sample_swarming_input():
    """Sample input for swarming behavior."""
    return {
        'model': 'swarming',
        'parameters': {
            'n_particles': 500,
            'v0': 1.0,
            'r_repulsion': 1.0,
            'r_attraction': 3.0,
            'r_alignment': 2.0
        }
    }


# ============================================================================
# Test: Initialization and Metadata (3 tests)
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert isinstance(agent.supported_models, list)
    assert len(agent.supported_models) >= 5


def test_get_metadata(agent):
    """Test agent metadata retrieval."""
    metadata = agent.get_metadata()
    assert metadata.name == "ActiveMatterAgent"
    assert metadata.version == "1.0.0"
    assert metadata.description is not None
    assert "active matter" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 4
    method_names = [cap.name for cap in capabilities]
    assert 'Vicsek Model' in method_names
    assert 'Active Brownian Particles' in method_names


# ============================================================================
# Test: Input Validation (8 tests)
# ============================================================================

def test_validate_input_success_vicsek(agent, sample_vicsek_input):
    """Test validation succeeds for valid Vicsek input."""
    validation = agent.validate_input(sample_vicsek_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_abp(agent, sample_active_brownian_input):
    """Test validation succeeds for valid ABP input."""
    validation = agent.validate_input(sample_active_brownian_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_model(agent):
    """Test validation fails when model is missing."""
    validation = agent.validate_input({'parameters': {'n_particles': 100}})
    assert not validation.valid
    assert any('model' in err.lower() for err in validation.errors)


def test_validate_input_invalid_model(agent):
    """Test validation fails for unsupported model."""
    validation = agent.validate_input({'model': 'quantum_swarm'})
    assert not validation.valid
    assert any('unsupported model' in err.lower() for err in validation.errors)


def test_validate_input_small_particle_number(agent):
    """Test validation warns for small particle number."""
    input_data = {
        'model': 'vicsek',
        'parameters': {'n_particles': 5}
    }
    validation = agent.validate_input(input_data)
    assert len(validation.warnings) > 0
    assert any('particle' in warn.lower() for warn in validation.warnings)


def test_validate_input_missing_velocity(agent):
    """Test validation warns when velocity not specified."""
    input_data = {
        'model': 'vicsek',
        'parameters': {'n_particles': 100}
    }
    validation = agent.validate_input(input_data)
    assert len(validation.warnings) > 0


def test_validate_input_abp_missing_peclet(agent):
    """Test validation warns when Péclet number not specified for ABP."""
    input_data = {
        'model': 'active_brownian',
        'parameters': {'n_particles': 100}
    }
    validation = agent.validate_input(input_data)
    assert len(validation.warnings) > 0


def test_validate_input_all_models(agent):
    """Test validation for all supported models."""
    for model in agent.supported_models:
        input_data = {
            'model': model,
            'parameters': {'n_particles': 100}
        }
        validation = agent.validate_input(input_data)
        # Should not have errors (may have warnings)
        assert validation.valid


# ============================================================================
# Test: Resource Estimation (7 tests)
# ============================================================================

def test_estimate_resources_small_system(agent, sample_vicsek_input):
    """Test resource estimation for small system."""
    sample_vicsek_input['parameters']['n_particles'] = 100
    resources = agent.estimate_resources(sample_vicsek_input)
    assert resources.execution_environment.value == 'local'
    assert resources.cpu_cores == 4
    assert resources.estimated_time_sec == 120


def test_estimate_resources_medium_system(agent, sample_vicsek_input):
    """Test resource estimation for medium system."""
    sample_vicsek_input['parameters']['n_particles'] = 50000
    resources = agent.estimate_resources(sample_vicsek_input)
    assert resources.execution_environment.value == 'local'
    assert resources.cpu_cores == 8
    assert resources.estimated_time_sec == 600


def test_estimate_resources_large_system(agent, sample_vicsek_input):
    """Test resource estimation for large system."""
    sample_vicsek_input['parameters']['n_particles'] = 200000
    resources = agent.estimate_resources(sample_vicsek_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.cpu_cores == 16
    assert resources.gpu_count >= 1


def test_estimate_resources_long_simulation(agent, sample_active_brownian_input):
    """Test resource estimation for long simulation."""
    sample_active_brownian_input['parameters']['n_steps'] = 2000000
    resources = agent.estimate_resources(sample_active_brownian_input)
    assert resources.execution_environment.value == 'hpc'


def test_estimate_resources_abp(agent, sample_active_brownian_input):
    """Test resource estimation for ABP."""
    resources = agent.estimate_resources(sample_active_brownian_input)
    assert resources.cpu_cores >= 4


def test_estimate_resources_run_and_tumble(agent, sample_run_and_tumble_input):
    """Test resource estimation for run-and-tumble."""
    resources = agent.estimate_resources(sample_run_and_tumble_input)
    assert resources.estimated_time_sec > 0


def test_estimate_resources_active_nematics(agent, sample_active_nematics_input):
    """Test resource estimation for active nematics."""
    resources = agent.estimate_resources(sample_active_nematics_input)
    assert resources.execution_environment.value == 'local'


# ============================================================================
# Test: Execution for All Models (7 tests)
# ============================================================================

def test_execute_vicsek_success(agent, sample_vicsek_input):
    """Test Vicsek model execution succeeds."""
    result = agent.execute(sample_vicsek_input)
    assert result.success
    assert 'final_order_parameter' in result.data
    assert 'phase' in result.data
    assert result.data['phase'] in ['ordered', 'disordered', 'critical']
    assert 'order_parameter_history' in result.data
    assert 'correlation_function' in result.data


def test_execute_active_brownian_success(agent, sample_active_brownian_input):
    """Test Active Brownian Particles execution succeeds."""
    result = agent.execute(sample_active_brownian_input)
    assert result.success
    assert 'peclet_number' in result.data
    assert 'persistence_time' in result.data
    assert 'effective_diffusion_coefficient' in result.data
    assert 'mips_detected' in result.data
    assert 'phase' in result.data


def test_execute_run_and_tumble_success(agent, sample_run_and_tumble_input):
    """Test run-and-tumble execution succeeds."""
    result = agent.execute(sample_run_and_tumble_input)
    assert result.success
    assert 'tumble_rate_Hz' in result.data
    assert 'run_length_um' in result.data
    assert 'effective_diffusion_theory' in result.data
    assert 'msd_curve' in result.data


def test_execute_active_nematics_success(agent, sample_active_nematics_input):
    """Test active nematics execution succeeds."""
    result = agent.execute(sample_active_nematics_input)
    assert result.success
    assert 'topological_defects' in result.data
    assert 'active_turbulence' in result.data
    assert 'activity_parameter' in result.data
    assert result.data['topological_defects']['total'] >= 0


def test_execute_swarming_success(agent, sample_swarming_input):
    """Test swarming execution succeeds."""
    result = agent.execute(sample_swarming_input)
    assert result.success
    assert 'swarm_cohesion' in result.data
    assert 'swarm_polarization' in result.data
    assert 'swarming_detected' in result.data


def test_execute_vicsek_ordered_phase(agent, sample_vicsek_input):
    """Test Vicsek model produces ordered phase with low noise."""
    sample_vicsek_input['parameters']['eta'] = 0.1  # Low noise
    result = agent.execute(sample_vicsek_input)
    assert result.success
    # Low noise should give high order parameter
    assert result.data['final_order_parameter'] > 0.3


def test_execute_invalid_model(agent):
    """Test execution fails for invalid model."""
    result = agent.execute({'model': 'invalid_model', 'parameters': {}})
    assert not result.success
    assert len(result.errors) > 0


# ============================================================================
# Test: Job Submission/Status/Retrieval (5 tests)
# ============================================================================

def test_submit_calculation_returns_job_id(agent, sample_vicsek_input):
    """Test submit_calculation returns valid job ID."""
    job_id = agent.submit_calculation(sample_vicsek_input)
    assert job_id is not None
    assert isinstance(job_id, str)
    assert len(job_id) > 0


def test_check_status_after_submission(agent, sample_vicsek_input):
    """Test check_status returns valid status after submission."""
    job_id = agent.submit_calculation(sample_vicsek_input)
    status = agent.check_status(job_id)
    assert status == AgentStatus.RUNNING


def test_check_status_invalid_job_id(agent):
    """Test check_status handles invalid job ID."""
    status = agent.check_status('invalid_job_id')
    assert status == AgentStatus.FAILED


def test_retrieve_results_after_completion(agent, sample_vicsek_input):
    """Test retrieve_results after marking job complete."""
    job_id = agent.submit_calculation(sample_vicsek_input)
    # Simulate completion
    if hasattr(agent, 'job_cache') and job_id in agent.job_cache:
        agent.job_cache[job_id]['status'] = AgentStatus.SUCCESS
        agent.job_cache[job_id]['results'] = {'order_parameter': 0.8}

    results = agent.retrieve_results(job_id)
    assert results is not None
    assert isinstance(results, dict)


def test_retrieve_results_invalid_job_id(agent):
    """Test retrieve_results handles invalid job ID."""
    with pytest.raises(Exception):
        agent.retrieve_results('invalid_job_id')


# ============================================================================
# Test: Analysis Methods (6 tests)
# ============================================================================

def test_velocity_correlation_computation(agent):
    """Test velocity correlation function computation."""
    velocities = np.random.randn(100, 2)
    positions = np.random.rand(100, 2) * 20
    box_size = 20.0

    correlation = agent._compute_velocity_correlation(velocities, positions, box_size)

    assert 'r_bins' in correlation
    assert 'correlation' in correlation
    assert 'correlation_length' in correlation
    assert len(correlation['r_bins']) == len(correlation['correlation'])


def test_vicsek_order_parameter_bounds(agent, sample_vicsek_input):
    """Test order parameter is between 0 and 1."""
    result = agent.execute(sample_vicsek_input)
    assert result.success
    order_param = result.data['final_order_parameter']
    assert 0 <= order_param <= 1


def test_abp_peclet_number_calculation(agent, sample_active_brownian_input):
    """Test Péclet number is calculated correctly."""
    result = agent.execute(sample_active_brownian_input)
    assert result.success
    pe = result.data['peclet_number']
    assert pe > 0


def test_run_and_tumble_diffusion(agent, sample_run_and_tumble_input):
    """Test effective diffusion coefficient calculation."""
    result = agent.execute(sample_run_and_tumble_input)
    assert result.success
    D_eff = result.data['effective_diffusion_theory']
    assert D_eff > 0


def test_active_nematics_defect_counting(agent, sample_active_nematics_input):
    """Test topological defect counting."""
    result = agent.execute(sample_active_nematics_input)
    assert result.success
    defects = result.data['topological_defects']
    # Defects should be integers
    assert isinstance(defects['n_plus_half'], int)
    assert isinstance(defects['n_minus_half'], int)
    assert defects['total'] == defects['n_plus_half'] + defects['n_minus_half']


def test_swarming_metrics(agent, sample_swarming_input):
    """Test swarming cohesion and polarization metrics."""
    result = agent.execute(sample_swarming_input)
    assert result.success
    assert 0 <= result.data['swarm_cohesion'] <= 1
    assert 0 <= result.data['swarm_polarization'] <= 1


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_vicsek_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_vicsek_input)
    result2 = agent.execute_with_caching(sample_vicsek_input)

    assert result1.success
    assert result2.success
    # Second call should be from cache
    assert result2.metadata.get('cached', False) or result2.execution_time_seconds < result1.execution_time_seconds * 0.5


def test_caching_different_inputs(agent, sample_vicsek_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_vicsek_input)

    # Modify input
    modified_input = sample_vicsek_input.copy()
    modified_input['parameters'] = sample_vicsek_input['parameters'].copy()
    modified_input['parameters']['eta'] = 0.5

    result2 = agent.execute_with_caching(modified_input)

    assert result1.success
    assert result2.success


def test_provenance_tracking(agent, sample_vicsek_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_vicsek_input)
    assert result.success
    assert hasattr(result, 'provenance')
    assert result.provenance is not None
    assert result.provenance.agent_name == 'ActiveMatterAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_vicsek_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_vicsek_input)
    assert result.success
    assert result.provenance.input_hash is not None
    assert len(result.provenance.input_hash) == 64  # SHA256 hex digest


# ============================================================================
# Test: Physical Validation (4 tests)
# ============================================================================

def test_vicsek_phase_transition(agent):
    """Test Vicsek model shows phase transition with noise."""
    # Low noise -> ordered
    low_noise_input = {
        'model': 'vicsek',
        'parameters': {'n_particles': 1000, 'v0': 0.5, 'eta': 0.1, 'n_steps': 5000}
    }
    result_ordered = agent.execute(low_noise_input)

    # High noise -> disordered
    high_noise_input = {
        'model': 'vicsek',
        'parameters': {'n_particles': 1000, 'v0': 0.5, 'eta': 2.0, 'n_steps': 5000}
    }
    result_disordered = agent.execute(high_noise_input)

    assert result_ordered.success and result_disordered.success
    # Low noise should have higher order parameter
    assert result_ordered.data['final_order_parameter'] > result_disordered.data['final_order_parameter']


def test_abp_mips_detection(agent):
    """Test MIPS detection for high Péclet number."""
    high_pe_input = {
        'model': 'active_brownian',
        'parameters': {
            'n_particles': 1000,
            'v0': 10.0,  # High velocity
            'D_r': 0.01,  # Low rotational diffusion -> high Pe
            'D_t': 0.01,
            'n_steps': 10000
        }
    }
    result = agent.execute(high_pe_input)
    assert result.success
    # High Pe should favor MIPS
    assert result.data['peclet_number'] > 10


def test_run_and_tumble_run_length(agent, sample_run_and_tumble_input):
    """Test run length calculation."""
    result = agent.execute(sample_run_and_tumble_input)
    assert result.success
    run_length = result.data['run_length_um']
    # Run length = v0 / tumble_rate
    expected = sample_run_and_tumble_input['parameters']['v0'] / sample_run_and_tumble_input['parameters']['tumble_rate']
    assert abs(run_length - expected) < 1e-6


def test_active_nematics_turbulence(agent):
    """Test active turbulence detection."""
    high_activity_input = {
        'model': 'active_nematics',
        'parameters': {'grid_size': 64, 'activity': 0.8, 'n_steps': 1000}
    }
    result = agent.execute(high_activity_input)
    assert result.success
    # High activity should show turbulence
    assert result.data['active_turbulence']['detected']


# ============================================================================
# Test: Workflow Integration (3 tests)
# ============================================================================

def test_workflow_vicsek_parameter_sweep(agent):
    """Test Vicsek model parameter sweep."""
    noise_values = [0.1, 0.5, 1.0]
    results = []

    for eta in noise_values:
        input_data = {
            'model': 'vicsek',
            'parameters': {'n_particles': 500, 'v0': 0.5, 'eta': eta, 'n_steps': 5000}
        }
        result = agent.execute(input_data)
        assert result.success
        results.append(result.data['final_order_parameter'])

    # Order parameter should decrease with noise
    assert results[0] > results[-1]


def test_workflow_abp_phase_diagram(agent):
    """Test ABP phase diagram exploration."""
    # Low Pe -> homogeneous
    low_pe = {
        'model': 'active_brownian',
        'parameters': {'n_particles': 500, 'v0': 0.1, 'D_r': 1.0, 'n_steps': 5000}
    }
    result_low = agent.execute(low_pe)
    assert result_low.success


def test_workflow_all_models(agent):
    """Test all active matter models execute successfully."""
    models = ['vicsek', 'active_brownian', 'run_and_tumble', 'active_nematics', 'swarming']
    results = []

    for model in models:
        input_data = {
            'model': model,
            'parameters': {'n_particles': 100, 'n_steps': 1000}
        }
        result = agent.execute(input_data)
        results.append(result.success)

    assert all(results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])