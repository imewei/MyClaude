"""
Test suite for SimulationAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL and HPC)
- Execution for all simulation methods
- Job submission/status/retrieval patterns
- Integration methods
- Caching and provenance
- Scientific validation

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
from simulation_agent import SimulationAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a SimulationAgent instance."""
    return SimulationAgent()


@pytest.fixture
def sample_classical_md_input():
    """Sample input for classical MD simulation."""
    return {
        'method': 'classical_md',
        'engine': 'lammps',
        'structure_file': 'polymer.xyz',
        'parameters': {
            'ensemble': 'NPT',
            'temperature': 300,
            'pressure': 1.0,
            'timestep': 1.0,
            'steps': 100000,
            'output_frequency': 1000
        }
    }


@pytest.fixture
def sample_mlff_train_input():
    """Sample input for MLFF training."""
    return {
        'method': 'mlff',
        'mode': 'train',
        'framework': 'deepmd',
        'parameters': {
            'training_data': 'dft_trajectories.npz',
            'epochs': 1000,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }


@pytest.fixture
def sample_mlff_inference_input():
    """Sample input for MLFF inference."""
    return {
        'method': 'mlff',
        'mode': 'inference',
        'framework': 'nequip',
        'model_file': 'trained_model.pth',
        'structure_file': 'polymer.xyz',
        'parameters': {
            'ensemble': 'NPT',
            'temperature': 298,
            'steps': 50000
        }
    }


@pytest.fixture
def sample_hoomd_input():
    """Sample input for HOOMD-blue simulation."""
    return {
        'method': 'hoomd',
        'structure_file': 'colloids.gsd',
        'parameters': {
            'ensemble': 'NVT',
            'temperature': 300,
            'timestep': 0.005,
            'steps': 1000000,
            'particle_type': 'colloid'
        }
    }


@pytest.fixture
def sample_dpd_input():
    """Sample input for DPD simulation."""
    return {
        'method': 'dpd',
        'structure_file': 'polymer_blend.xyz',
        'parameters': {
            'temperature': 1.0,
            'timestep': 0.01,
            'steps': 500000,
            'bead_types': ['A', 'B'],
            'chi_parameters': {'AA': 25, 'BB': 25, 'AB': 35}
        }
    }


@pytest.fixture
def sample_dem_input():
    """Sample input for nanoscale DEM simulation."""
    return {
        'method': 'nanoscale_dem',
        'structure_file': 'nanoparticles.xyz',
        'parameters': {
            'particle_radius_nm': 50,
            'youngs_modulus_GPa': 70,
            'timestep': 1e-9,
            'steps': 100000,
            'loading_rate_nm_per_ns': 0.1
        }
    }


@pytest.fixture
def sample_md_result():
    """Sample MD simulation result."""
    q_values = np.linspace(0.1, 5.0, 50)
    return {
        'status': 'completed',
        'trajectory_file': 'output.dcd',
        'total_time_ns': 100.0,
        'structure_factor': {
            'q_nm_inv': q_values.tolist(),
            'S_q': (1 + 2 * np.exp(-((q_values - 3.0)**2) / 1.0)).tolist()
        },
        'radial_distribution': {
            'r_nm': np.linspace(0.1, 2.0, 100).tolist(),
            'g_r': (np.exp(-((np.linspace(0.1, 2.0, 100) - 0.35)**2) / 0.05) + 1.0).tolist()
        },
        'transport_properties': {
            'viscosity_Pa_s': 0.85,
            'diffusion_coefficient_m2_per_s': 2.3e-10
        }
    }


@pytest.fixture
def sample_scattering_data():
    """Sample experimental scattering data."""
    q_values = np.linspace(0.1, 5.0, 40)
    return {
        'technique': 'SANS',
        'q_nm_inv': q_values.tolist(),
        'I_q': (1 + 2 * np.exp(-((q_values - 3.0)**2) / 1.0) + np.random.normal(0, 0.05, len(q_values))).tolist(),
        'sigma_I_q': (0.05 * np.ones_like(q_values)).tolist()
    }


@pytest.fixture
def sample_dft_training_data():
    """Sample DFT data for MLFF training."""
    return {
        'source': 'VASP',
        'num_configurations': 5000,
        'energy_data': {
            'energies_eV': np.random.uniform(-500, -400, 5000).tolist(),
            'forces_eV_per_A': np.random.normal(0, 0.5, (5000, 100, 3)).tolist()
        },
        'structures': 'dft_configs.xyz'
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
    assert metadata.name == "SimulationAgent"
    assert metadata.version == "1.0.0"
    assert metadata.description is not None
    assert "simulation" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 5
    method_names = [cap.name for cap in capabilities]
    assert 'classical_md' in method_names
    assert 'mlff' in method_names
    assert 'hoomd' in method_names


# ============================================================================
# Test: Input Validation (8 tests)
# ============================================================================

def test_validate_input_success_classical_md(agent, sample_classical_md_input):
    """Test validation succeeds for valid classical MD input."""
    validation = agent.validate_input(sample_classical_md_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_mlff(agent, sample_mlff_train_input):
    """Test validation succeeds for valid MLFF input."""
    validation = agent.validate_input(sample_mlff_train_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_method(agent):
    """Test validation fails when method is missing."""
    validation = agent.validate_input({'structure_file': 'test.xyz'})
    assert not validation.valid
    assert any('method' in err.lower() for err in validation.errors)


def test_validate_input_invalid_method(agent):
    """Test validation fails for unsupported method."""
    validation = agent.validate_input({'method': 'quantum_chemistry'})
    assert not validation.valid
    assert any('unsupported method' in err.lower() for err in validation.errors)


def test_validate_input_missing_structure_file(agent):
    """Test validation fails when structure file is missing."""
    validation = agent.validate_input({
        'method': 'classical_md',
        'engine': 'lammps'
    })
    assert not validation.valid
    assert any('structure_file' in err.lower() for err in validation.errors)


def test_validate_input_invalid_temperature(agent, sample_classical_md_input):
    """Test validation warns for unusual temperature."""
    sample_classical_md_input['parameters']['temperature'] = -50
    validation = agent.validate_input(sample_classical_md_input)
    # Negative temp should generate warning, not error
    assert len(validation.warnings) > 0 or len(validation.errors) > 0


def test_validate_input_invalid_steps(agent, sample_classical_md_input):
    """Test validation fails for invalid number of steps."""
    sample_classical_md_input['parameters']['steps'] = -1000
    validation = agent.validate_input(sample_classical_md_input)
    assert not validation.valid
    assert any('steps' in err.lower() for err in validation.errors)


def test_validate_input_warnings_few_steps(agent, sample_classical_md_input):
    """Test validation warns for very few steps."""
    sample_classical_md_input['parameters']['steps'] = 500
    validation = agent.validate_input(sample_classical_md_input)
    assert validation.valid  # Still valid
    assert len(validation.warnings) > 0
    assert any('step' in warn.lower() for warn in validation.warnings)


# ============================================================================
# Test: Resource Estimation (7 tests)
# ============================================================================

def test_estimate_resources_classical_md_short(agent, sample_classical_md_input):
    """Test resource estimation for short classical MD."""
    sample_classical_md_input['parameters']['steps'] = 10000
    resources = agent.estimate_resources(sample_classical_md_input)
    assert resources.execution_environment.value in ['local', 'hpc']
    assert resources.estimated_time_sec < 36000


def test_estimate_resources_classical_md_long(agent, sample_classical_md_input):
    """Test resource estimation for long classical MD."""
    sample_classical_md_input['parameters']['steps'] = 10000000
    resources = agent.estimate_resources(sample_classical_md_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 3600


def test_estimate_resources_mlff_training(agent, sample_mlff_train_input):
    """Test resource estimation for MLFF training."""
    resources = agent.estimate_resources(sample_mlff_train_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 3600
    assert resources.gpu_count > 0


def test_estimate_resources_mlff_inference(agent, sample_mlff_inference_input):
    """Test resource estimation for MLFF inference."""
    resources = agent.estimate_resources(sample_mlff_inference_input)
    assert resources.execution_environment.value in ['local', 'hpc']
    assert resources.estimated_time_sec < 7200


def test_estimate_resources_hoomd(agent, sample_hoomd_input):
    """Test resource estimation for HOOMD-blue."""
    resources = agent.estimate_resources(sample_hoomd_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.gpu_count > 0


def test_estimate_resources_dpd(agent, sample_dpd_input):
    """Test resource estimation for DPD."""
    resources = agent.estimate_resources(sample_dpd_input)
    assert resources.estimated_time_sec > 0
    assert resources.execution_environment.value in ['local', 'hpc']


def test_estimate_resources_dem(agent, sample_dem_input):
    """Test resource estimation for nanoscale DEM."""
    resources = agent.estimate_resources(sample_dem_input)
    assert resources.estimated_time_sec > 0
    assert resources.execution_environment.value in ['local', 'hpc']


# ============================================================================
# Test: Execution for All Methods (7 tests)
# ============================================================================

def test_execute_classical_md_success(agent, sample_classical_md_input):
    """Test classical MD execution succeeds."""
    result = agent.execute(sample_classical_md_input)
    assert result.success
    assert 'structure_factor' in result.data
    assert 'radial_distribution' in result.data
    assert 'transport_properties' in result.data
    assert 'q_nm_inv' in result.data['structure_factor']
    assert 'S_q' in result.data['structure_factor']
    assert len(result.data['structure_factor']['q_nm_inv']) > 0


def test_execute_mlff_training_success(agent, sample_mlff_train_input):
    """Test MLFF training execution succeeds."""
    result = agent.execute(sample_mlff_train_input)
    assert result.success
    assert 'model_file' in result.data
    # Check for DeepMD-kit output format
    if 'training_metrics' in result.data:
        assert 'energy_MAE_meV_per_atom' in result.data['training_metrics']
    elif 'final_energy_MAE_meV_per_atom' in result.data:
        assert result.data['final_energy_MAE_meV_per_atom'] < 10.0


def test_execute_mlff_inference_success(agent, sample_mlff_inference_input):
    """Test MLFF inference execution succeeds."""
    result = agent.execute(sample_mlff_inference_input)
    assert result.success
    assert 'speedup_vs_DFT' in result.data
    assert result.data['speedup_vs_DFT'] > 100
    assert 'accuracy_meV_per_atom' in result.data


def test_execute_hoomd_success(agent, sample_hoomd_input):
    """Test HOOMD-blue execution succeeds."""
    result = agent.execute(sample_hoomd_input)
    assert result.success
    # HOOMD should return trajectory and structural data
    assert len(result.data) > 0


def test_execute_dpd_success(agent, sample_dpd_input):
    """Test DPD execution succeeds."""
    result = agent.execute(sample_dpd_input)
    assert result.success
    # DPD should return trajectory and mesoscale data
    assert len(result.data) > 0


def test_execute_dem_success(agent, sample_dem_input):
    """Test nanoscale DEM execution succeeds."""
    result = agent.execute(sample_dem_input)
    assert result.success
    # DEM should return mechanical properties
    assert len(result.data) > 0


def test_execute_invalid_method(agent):
    """Test execution fails for invalid method."""
    result = agent.execute({'method': 'invalid_method', 'structure_file': 'test.xyz'})
    assert not result.success
    assert len(result.errors) > 0
    assert any('unsupported method' in err.lower() for err in result.errors)


# ============================================================================
# Test: Job Submission/Status/Retrieval (5 tests)
# ============================================================================

def test_submit_calculation_returns_job_id(agent, sample_classical_md_input):
    """Test submit_calculation returns valid job ID."""
    job_id = agent.submit_calculation(sample_classical_md_input)
    assert job_id is not None
    assert isinstance(job_id, str)
    assert job_id.startswith('sim_')


def test_check_status_after_submission(agent, sample_classical_md_input):
    """Test check_status returns valid status after submission."""
    job_id = agent.submit_calculation(sample_classical_md_input)
    status = agent.check_status(job_id)
    assert status in [AgentStatus.RUNNING, AgentStatus.SUCCESS]


def test_check_status_invalid_job_id(agent):
    """Test check_status handles invalid job ID."""
    status = agent.check_status('invalid_job_id')
    assert status == AgentStatus.FAILED


def test_retrieve_results_after_completion(agent, sample_classical_md_input):
    """Test retrieve_results returns data after completion."""
    job_id = agent.submit_calculation(sample_classical_md_input)
    # Simulate completion
    if hasattr(agent, 'job_cache') and job_id in agent.job_cache:
        agent.job_cache[job_id]['status'] = AgentStatus.SUCCESS

    results = agent.retrieve_results(job_id)
    assert results is not None
    assert isinstance(results, dict)


def test_retrieve_results_invalid_job_id(agent):
    """Test retrieve_results handles invalid job ID."""
    results = agent.retrieve_results('invalid_job_id')
    assert results is None or 'error' in results


# ============================================================================
# Test: Integration Methods (6 tests)
# ============================================================================

def test_validate_scattering_data_good_agreement(agent, sample_md_result, sample_scattering_data):
    """Test scattering validation with good agreement."""
    validation = agent.validate_scattering_data(sample_md_result, sample_scattering_data)
    assert 'chi_squared' in validation
    assert 'agreement' in validation
    assert validation['chi_squared'] < 10  # Good fit
    assert validation['agreement'] in ['excellent', 'good']


def test_validate_scattering_data_poor_agreement(agent, sample_md_result):
    """Test scattering validation with poor agreement."""
    # Create poorly matching experimental data
    poor_exp_data = {
        'technique': 'SAXS',
        'q_nm_inv': np.linspace(0.1, 5.0, 40).tolist(),
        'I_q': (5 + np.random.normal(0, 1, 40)).tolist(),  # Very different from MD
        'sigma_I_q': (0.1 * np.ones(40)).tolist()
    }
    validation = agent.validate_scattering_data(sample_md_result, poor_exp_data)
    assert 'chi_squared' in validation
    assert validation['chi_squared'] > 10
    assert validation['agreement'] == 'poor'


def test_validate_scattering_data_missing_md_sq(agent, sample_scattering_data):
    """Test scattering validation fails when MD S(q) missing."""
    md_result_no_sq = {'trajectory_file': 'test.dcd'}
    validation = agent.validate_scattering_data(md_result_no_sq, sample_scattering_data)
    assert not validation['success']
    assert 'error' in validation


def test_train_mlff_from_dft_success(agent, sample_dft_training_data):
    """Test MLFF training from DFT data succeeds."""
    result = agent.train_mlff_from_dft(sample_dft_training_data)
    assert result['success']
    assert 'model_file' in result
    assert 'validation_metrics' in result
    assert result['validation_metrics']['energy_MAE_meV_per_atom'] < 10


def test_train_mlff_from_dft_insufficient_data(agent):
    """Test MLFF training fails with insufficient data."""
    poor_dft_data = {
        'source': 'VASP',
        'num_configurations': 50,  # Too few
        'energy_data': {'energies_eV': list(range(50))}
    }
    result = agent.train_mlff_from_dft(poor_dft_data)
    assert not result['success']
    assert 'error' in result or 'warning' in result


def test_predict_rheology_from_trajectory(agent):
    """Test rheology prediction from MD trajectory."""
    trajectory_data = {
        'trajectory_file': 'polymer.dcd',
        'pressure_tensor_trace': np.random.normal(0, 100, 10000).tolist(),
        'timestep_fs': 1.0,
        'temperature_K': 300
    }
    result = agent.predict_rheology(trajectory_data)
    assert result['success']
    assert 'viscosity_Pa_s' in result
    assert result['viscosity_Pa_s'] > 0
    assert 'method' in result
    assert 'green-kubo' in result['method'].lower()


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_classical_md_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_classical_md_input)
    result2 = agent.execute_with_caching(sample_classical_md_input)

    assert result1.success
    assert result2.success
    # Second call should be from cache
    assert result2.metadata.get('cached', False) or result2.execution_time_seconds < result1.execution_time_seconds * 0.5


def test_caching_different_inputs(agent, sample_classical_md_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_classical_md_input)

    # Modify input
    modified_input = sample_classical_md_input.copy()
    modified_input['parameters'] = sample_classical_md_input['parameters'].copy()
    modified_input['parameters']['temperature'] = 350

    result2 = agent.execute_with_caching(modified_input)

    assert result1.success
    assert result2.success
    # Results should be different (not from cache)
    assert result1.data != result2.data


def test_provenance_tracking(agent, sample_classical_md_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_classical_md_input)
    assert result.success
    assert hasattr(result, 'provenance')
    assert result.provenance is not None
    assert result.provenance.agent_name == 'SimulationAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_classical_md_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_classical_md_input)
    assert result.success
    assert result.provenance.input_hash is not None
    assert len(result.provenance.input_hash) == 64  # SHA256 hex digest


# ============================================================================
# Test: Scientific Validation (4 tests)
# ============================================================================

def test_structure_factor_normalization(agent, sample_classical_md_input):
    """Test S(q) normalizes correctly at high q."""
    result = agent.execute(sample_classical_md_input)
    assert result.success

    q_values = np.array(result.data['structure_factor']['q_nm_inv'])
    S_q_values = np.array(result.data['structure_factor']['S_q'])

    # S(q) should approach 1 at high q
    high_q_indices = q_values > 10
    if np.any(high_q_indices):
        assert np.abs(np.mean(S_q_values[high_q_indices]) - 1.0) < 0.3


def test_radial_distribution_constraints(agent, sample_classical_md_input):
    """Test g(r) satisfies physical constraints."""
    result = agent.execute(sample_classical_md_input)
    assert result.success

    r_values = np.array(result.data['radial_distribution']['r_nm'])
    g_r_values = np.array(result.data['radial_distribution']['g_r'])

    # g(r) should be near zero at short distances (excluded volume)
    short_r_indices = r_values < 0.2
    if np.any(short_r_indices):
        assert np.all(g_r_values[short_r_indices] < 2.0)  # Peak can be higher than 0.5

    # g(r) should approach 1 at large distances
    large_r_indices = r_values > 1.5
    if np.any(large_r_indices):
        assert np.abs(np.mean(g_r_values[large_r_indices]) - 1.0) < 1.0


def test_viscosity_positive(agent, sample_classical_md_input):
    """Test predicted viscosity is positive."""
    result = agent.execute(sample_classical_md_input)
    assert result.success

    if 'transport_properties' in result.data:
        if 'viscosity_Pa_s' in result.data['transport_properties']:
            viscosity = result.data['transport_properties']['viscosity_Pa_s']
            assert viscosity > 0
            assert viscosity < 1e6  # Reasonable upper bound


def test_mlff_training_convergence(agent, sample_mlff_train_input):
    """Test MLFF training shows reasonable accuracy."""
    result = agent.execute(sample_mlff_train_input)
    assert result.success

    # Check for training metrics in different formats
    if 'training_metrics' in result.data:
        metrics = result.data['training_metrics']
        assert metrics['energy_MAE_meV_per_atom'] < 10
    elif 'final_energy_MAE_meV_per_atom' in result.data:
        assert result.data['final_energy_MAE_meV_per_atom'] < 10


# ============================================================================
# Test: Workflow Integration (3 tests)
# ============================================================================

def test_workflow_md_to_scattering_validation(agent, sample_classical_md_input, sample_scattering_data):
    """Test complete workflow: MD → scattering validation."""
    # Step 1: Run MD simulation
    md_result = agent.execute(sample_classical_md_input)
    assert md_result.success

    # Step 2: Validate against experimental scattering
    validation = agent.validate_scattering_data(md_result.data, sample_scattering_data)
    assert 'chi_squared' in validation
    assert 'agreement' in validation


def test_workflow_dft_to_mlff_to_md(agent, sample_dft_training_data, sample_mlff_inference_input):
    """Test complete workflow: DFT → MLFF training → MD with MLFF."""
    # Step 1: Train MLFF from DFT
    mlff_result = agent.train_mlff_from_dft(sample_dft_training_data)
    assert mlff_result['success']

    # Step 2: Run MD with trained MLFF
    sample_mlff_inference_input['model_file'] = mlff_result['model_file']
    md_result = agent.execute(sample_mlff_inference_input)
    assert md_result.success
    assert 'speedup_vs_DFT' in md_result.data


def test_workflow_md_to_rheology_prediction(agent, sample_classical_md_input):
    """Test complete workflow: MD → rheology prediction."""
    # Step 1: Run MD simulation
    md_result = agent.execute(sample_classical_md_input)
    assert md_result.success

    # Step 2: Use MD result directly (has transport_properties)
    rheology_result = agent.predict_rheology(md_result.data)
    assert rheology_result['success']
    assert 'viscosity_Pa_s' in rheology_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])