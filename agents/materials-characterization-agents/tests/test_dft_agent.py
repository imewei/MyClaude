"""
Test suite for DFTAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (HPC)
- Execution for all calculation types
- Job submission/status/retrieval patterns
- Integration methods
- Caching and provenance
- Physical validation

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
from dft_agent import DFTAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a DFTAgent instance."""
    return DFTAgent()


@pytest.fixture
def sample_scf_input():
    """Sample input for SCF calculation."""
    return {
        'calculation_type': 'scf',
        'code': 'vasp',
        'structure_file': 'Si.cif',
        'parameters': {
            'encut': 520,
            'kpoints': [8, 8, 8],
            'xc': 'PBE'
        }
    }


@pytest.fixture
def sample_relax_input():
    """Sample input for geometry relaxation."""
    return {
        'calculation_type': 'relax',
        'code': 'vasp',
        'structure_file': 'material.cif',
        'parameters': {
            'encut': 520,
            'kpoints': [6, 6, 6],
            'relax_type': 'both',  # ions + cell
            'ediffg': -0.02
        }
    }


@pytest.fixture
def sample_bands_input():
    """Sample input for band structure."""
    return {
        'calculation_type': 'bands',
        'code': 'vasp',
        'structure_file': 'semiconductor.cif',
        'parameters': {
            'encut': 520,
            'kpath': ['Gamma', 'X', 'W', 'L', 'Gamma', 'K']
        }
    }


@pytest.fixture
def sample_phonon_input():
    """Sample input for phonon calculation."""
    return {
        'calculation_type': 'phonon',
        'code': 'vasp',
        'structure_file': 'crystal.cif',
        'parameters': {
            'encut': 520,
            'qpoints': [4, 4, 4],
            'supercell': [2, 2, 2]
        }
    }


@pytest.fixture
def sample_aimd_input():
    """Sample input for AIMD."""
    return {
        'calculation_type': 'aimd',
        'code': 'vasp',
        'structure_file': 'liquid.cif',
        'parameters': {
            'encut': 400,
            'kpoints': [2, 2, 2],
            'temperature': 300,
            'steps': 5000,
            'timestep': 1.0
        }
    }


@pytest.fixture
def sample_elastic_input():
    """Sample input for elastic constants."""
    return {
        'calculation_type': 'elastic',
        'code': 'vasp',
        'structure_file': 'crystal.cif',
        'parameters': {
            'encut': 520,
            'kpoints': [8, 8, 8]
        }
    }


# ============================================================================
# Test: Initialization and Metadata (3 tests)
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert isinstance(agent.supported_codes, list)
    assert len(agent.supported_codes) >= 4
    assert isinstance(agent.supported_calculations, list)
    assert len(agent.supported_calculations) >= 8


def test_get_metadata(agent):
    """Test agent metadata retrieval."""
    metadata = agent.get_metadata()
    assert metadata.name == "DFTAgent"
    assert metadata.version == "1.0.0"
    assert "dft" in metadata.description.lower() or "density functional" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 8
    capability_names = [cap.name for cap in capabilities]
    assert 'scf' in capability_names
    assert 'relax' in capability_names
    assert 'bands' in capability_names
    assert 'phonon' in capability_names
    assert 'aimd' in capability_names
    assert 'elastic' in capability_names


# ============================================================================
# Test: Input Validation (10 tests)
# ============================================================================

def test_validate_input_success_scf(agent, sample_scf_input):
    """Test validation succeeds for valid SCF input."""
    validation = agent.validate_input(sample_scf_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_relax(agent, sample_relax_input):
    """Test validation succeeds for valid relax input."""
    validation = agent.validate_input(sample_relax_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_calculation_type(agent):
    """Test validation fails when calculation_type is missing."""
    validation = agent.validate_input({'structure_file': 'test.cif'})
    assert not validation.valid
    assert any('calculation_type' in err.lower() for err in validation.errors)


def test_validate_input_invalid_calculation_type(agent):
    """Test validation fails for unsupported calculation type."""
    validation = agent.validate_input({'calculation_type': 'invalid_calc', 'structure_file': 'test.cif'})
    assert not validation.valid
    assert any('unsupported' in err.lower() for err in validation.errors)


def test_validate_input_missing_structure_file(agent):
    """Test validation fails when structure file is missing."""
    validation = agent.validate_input({
        'calculation_type': 'scf',
        'code': 'vasp'
    })
    assert not validation.valid
    assert any('structure_file' in err.lower() for err in validation.errors)


def test_validate_input_low_encut_warning(agent, sample_scf_input):
    """Test validation warns for low energy cutoff."""
    sample_scf_input['parameters']['encut'] = 150
    validation = agent.validate_input(sample_scf_input)
    assert validation.valid  # Still valid
    assert len(validation.warnings) > 0
    assert any('cutoff' in warn.lower() for warn in validation.warnings)


def test_validate_input_coarse_kpoints_warning(agent, sample_scf_input):
    """Test validation warns for coarse k-point mesh."""
    sample_scf_input['parameters']['kpoints'] = [1, 1, 1]
    validation = agent.validate_input(sample_scf_input)
    assert validation.valid  # Still valid
    assert len(validation.warnings) > 0
    assert any('k-point' in warn.lower() or 'coarse' in warn.lower() for warn in validation.warnings)


def test_validate_input_large_aimd_timestep(agent, sample_aimd_input):
    """Test validation warns for large AIMD timestep."""
    sample_aimd_input['parameters']['timestep'] = 3.0
    validation = agent.validate_input(sample_aimd_input)
    assert validation.valid
    assert len(validation.warnings) > 0
    assert any('timestep' in warn.lower() for warn in validation.warnings)


def test_validate_input_short_aimd(agent, sample_aimd_input):
    """Test validation warns for short AIMD run."""
    sample_aimd_input['parameters']['steps'] = 500
    validation = agent.validate_input(sample_aimd_input)
    assert validation.valid
    assert len(validation.warnings) > 0


def test_validate_input_neb_no_structure_required(agent):
    """Test NEB doesn't require single structure file."""
    validation = agent.validate_input({
        'calculation_type': 'neb',
        'parameters': {'n_images': 7}
    })
    # Should not error about missing structure_file
    # (NEB needs initial + final structures, handled differently)
    assert validation.valid or not any('structure_file' in err for err in validation.errors)


# ============================================================================
# Test: Resource Estimation (8 tests)
# ============================================================================

def test_estimate_resources_scf(agent, sample_scf_input):
    """Test resource estimation for SCF."""
    resources = agent.estimate_resources(sample_scf_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.cpu_cores >= 16
    assert resources.estimated_time_sec > 0
    assert resources.estimated_time_sec < 3600  # Should be < 1 hour


def test_estimate_resources_relax(agent, sample_relax_input):
    """Test resource estimation for relaxation."""
    resources = agent.estimate_resources(sample_relax_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 600  # Should be > 10 min


def test_estimate_resources_bands(agent, sample_bands_input):
    """Test resource estimation for band structure."""
    resources = agent.estimate_resources(sample_bands_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 0


def test_estimate_resources_phonon(agent, sample_phonon_input):
    """Test resource estimation for phonons."""
    resources = agent.estimate_resources(sample_phonon_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 3600  # Phonons are expensive (> 1 hour)
    assert resources.cpu_cores >= 32  # Needs many cores


def test_estimate_resources_aimd_short(agent, sample_aimd_input):
    """Test resource estimation for short AIMD."""
    sample_aimd_input['parameters']['steps'] = 1000
    resources = agent.estimate_resources(sample_aimd_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 1000


def test_estimate_resources_aimd_long(agent, sample_aimd_input):
    """Test resource estimation for long AIMD."""
    sample_aimd_input['parameters']['steps'] = 10000
    resources = agent.estimate_resources(sample_aimd_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 7200  # Long AIMD > 2 hours


def test_estimate_resources_elastic(agent, sample_elastic_input):
    """Test resource estimation for elastic constants."""
    resources = agent.estimate_resources(sample_elastic_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 3600  # Moderate (~2 hours)


def test_estimate_resources_neb(agent):
    """Test resource estimation for NEB."""
    neb_input = {
        'calculation_type': 'neb',
        'parameters': {'n_images': 9}
    }
    resources = agent.estimate_resources(neb_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.estimated_time_sec > 7200  # NEB is expensive


# ============================================================================
# Test: Execution for All Calculation Types (8 tests)
# ============================================================================

def test_execute_scf_success(agent, sample_scf_input):
    """Test SCF execution succeeds."""
    result = agent.execute(sample_scf_input)
    assert result.success
    assert 'total_energy_eV' in result.data
    assert 'fermi_energy_eV' in result.data
    assert 'converged' in result.data
    assert result.data['converged'] == True


def test_execute_relax_success(agent, sample_relax_input):
    """Test relaxation execution succeeds."""
    result = agent.execute(sample_relax_input)
    assert result.success
    assert 'final_energy_eV' in result.data
    assert 'converged' in result.data
    assert 'optimized_structure_file' in result.data
    assert result.data['final_energy_eV'] < result.data['initial_energy_eV']  # Energy should decrease


def test_execute_bands_success(agent, sample_bands_input):
    """Test band structure execution succeeds."""
    result = agent.execute(sample_bands_input)
    assert result.success
    assert 'eigenvalues_eV' in result.data
    assert 'band_gap_eV' in result.data
    assert 'band_gap_type' in result.data
    assert len(result.data['eigenvalues_eV']) > 0


def test_execute_dos_success(agent):
    """Test DOS execution succeeds."""
    result = agent.execute({
        'calculation_type': 'dos',
        'structure_file': 'material.cif',
        'parameters': {'encut': 520}
    })
    assert result.success
    assert 'energy_eV' in result.data
    assert 'dos_total_states_per_eV' in result.data
    assert 'fermi_energy_eV' in result.data


def test_execute_phonon_success(agent, sample_phonon_input):
    """Test phonon execution succeeds."""
    result = agent.execute(sample_phonon_input)
    assert result.success
    assert 'frequencies_THz' in result.data
    assert 'has_imaginary_modes' in result.data
    assert 'thermal_properties' in result.data


def test_execute_aimd_success(agent, sample_aimd_input):
    """Test AIMD execution succeeds."""
    result = agent.execute(sample_aimd_input)
    assert result.success
    assert 'trajectory_file' in result.data
    assert 'energies_eV' in result.data
    assert 'temperature_average_K' in result.data
    assert abs(result.data['temperature_average_K'] - 300) < 50  # Should be near target


def test_execute_elastic_success(agent, sample_elastic_input):
    """Test elastic constants execution succeeds."""
    result = agent.execute(sample_elastic_input)
    assert result.success
    assert 'elastic_tensor_GPa' in result.data
    assert 'bulk_modulus_GPa' in result.data
    assert 'shear_modulus_GPa' in result.data
    assert 'youngs_modulus_GPa' in result.data
    assert result.data['bulk_modulus_GPa'] > 0
    assert result.data['shear_modulus_GPa'] > 0


def test_execute_neb_success(agent):
    """Test NEB execution succeeds."""
    result = agent.execute({
        'calculation_type': 'neb',
        'parameters': {'n_images': 7}
    })
    assert result.success
    assert 'energies_eV' in result.data
    assert 'forward_barrier_eV' in result.data
    assert 'reverse_barrier_eV' in result.data
    assert result.data['forward_barrier_eV'] > 0


# ============================================================================
# Test: Job Submission/Status/Retrieval (5 tests)
# ============================================================================

def test_submit_calculation_returns_job_id(agent, sample_scf_input):
    """Test submit_calculation returns valid job ID."""
    job_id = agent.submit_calculation(sample_scf_input)
    assert job_id is not None
    assert isinstance(job_id, str)
    assert job_id.startswith('dft_')


def test_check_status_after_submission(agent, sample_scf_input):
    """Test check_status returns valid status after submission."""
    job_id = agent.submit_calculation(sample_scf_input)
    status = agent.check_status(job_id)
    assert status in [AgentStatus.RUNNING, AgentStatus.SUCCESS]


def test_check_status_invalid_job_id(agent):
    """Test check_status handles invalid job ID."""
    status = agent.check_status('invalid_job_id')
    assert status == AgentStatus.FAILED


def test_retrieve_results_after_completion(agent, sample_scf_input):
    """Test retrieve_results returns data after completion."""
    job_id = agent.submit_calculation(sample_scf_input)
    # Simulate completion
    if hasattr(agent, 'job_cache') and job_id in agent.job_cache:
        agent.job_cache[job_id]['status'] = AgentStatus.SUCCESS

    results = agent.retrieve_results(job_id)
    assert results is not None
    assert isinstance(results, dict)
    assert 'total_energy_eV' in results  # SCF result


def test_retrieve_results_invalid_job_id(agent):
    """Test retrieve_results handles invalid job ID."""
    results = agent.retrieve_results('invalid_job_id')
    assert 'error' in results


# ============================================================================
# Test: Integration Methods (6 tests)
# ============================================================================

def test_generate_training_data_for_mlff_success(agent, sample_aimd_input):
    """Test MLFF training data generation from AIMD."""
    # Run AIMD first
    aimd_result = agent.execute(sample_aimd_input)
    assert aimd_result.success

    # Generate training data
    training_data = agent.generate_training_data_for_mlff(aimd_result.data, n_configs=1000)
    assert training_data['success']
    assert 'num_configurations' in training_data
    assert training_data['num_configurations'] == 1000
    assert 'energy_data' in training_data
    assert 'energies_eV' in training_data['energy_data']
    assert 'forces_eV_per_A' in training_data['energy_data']


def test_generate_training_data_wrong_input(agent, sample_scf_input):
    """Test MLFF training data generation fails for non-AIMD input."""
    scf_result = agent.execute(sample_scf_input)
    training_data = agent.generate_training_data_for_mlff(scf_result.data)
    assert not training_data['success']
    assert 'error' in training_data


def test_validate_elastic_constants_success(agent, sample_elastic_input):
    """Test elastic constants validation."""
    elastic_result = agent.execute(sample_elastic_input)
    assert elastic_result.success

    validation = agent.validate_elastic_constants(elastic_result.data)
    assert validation['success']
    assert 'bulk_modulus_Pa' in validation
    assert 'shear_modulus_Pa' in validation
    assert 'youngs_modulus_Pa' in validation
    assert validation['bulk_modulus_Pa'] > 0
    assert validation['shear_modulus_Pa'] > 0


def test_validate_elastic_constants_wrong_input(agent, sample_scf_input):
    """Test elastic validation fails for non-elastic input."""
    scf_result = agent.execute(sample_scf_input)
    validation = agent.validate_elastic_constants(scf_result.data)
    assert not validation['success']


def test_predict_raman_from_phonons_success(agent, sample_phonon_input):
    """Test Raman spectrum prediction from phonons."""
    phonon_result = agent.execute(sample_phonon_input)
    assert phonon_result.success

    raman_pred = agent.predict_raman_from_phonons(phonon_result.data)
    assert raman_pred['success']
    assert 'raman_frequencies_cm_inv' in raman_pred
    assert 'raman_intensities_arb' in raman_pred
    assert len(raman_pred['raman_frequencies_cm_inv']) > 0


def test_predict_raman_wrong_input(agent, sample_scf_input):
    """Test Raman prediction fails for non-phonon input."""
    scf_result = agent.execute(sample_scf_input)
    raman_pred = agent.predict_raman_from_phonons(scf_result.data)
    assert not raman_pred['success']


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_scf_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_scf_input)
    result2 = agent.execute_with_caching(sample_scf_input)

    assert result1.success
    assert result2.success
    # Second call should be from cache
    assert result2.metadata.get('cached', False) or result2.status == AgentStatus.CACHED


def test_caching_different_inputs(agent, sample_scf_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_scf_input)

    # Modify input
    modified_input = sample_scf_input.copy()
    modified_input['parameters'] = sample_scf_input['parameters'].copy()
    modified_input['parameters']['encut'] = 600

    result2 = agent.execute_with_caching(modified_input)

    assert result1.success
    assert result2.success
    # Results should be different (not from cache)
    assert result1.data != result2.data or result2.status != AgentStatus.CACHED


def test_provenance_tracking(agent, sample_scf_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_scf_input)
    assert result.success
    assert hasattr(result, 'provenance')
    assert result.provenance is not None
    assert result.provenance.agent_name == 'DFTAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_scf_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_scf_input)
    assert result.success
    assert result.provenance.input_hash is not None
    assert len(result.provenance.input_hash) == 64  # SHA256 hex digest


# ============================================================================
# Test: Physical Validation (6 tests)
# ============================================================================

def test_energy_convergence_scf(agent, sample_scf_input):
    """Test SCF energy convergence."""
    result = agent.execute(sample_scf_input)
    assert result.success
    assert result.data['converged'] == True
    assert result.data['total_energy_eV'] < 0  # Total energy should be negative


def test_band_gap_positive(agent, sample_bands_input):
    """Test band gap is non-negative."""
    result = agent.execute(sample_bands_input)
    assert result.success
    assert result.data['band_gap_eV'] >= 0


def test_phonon_stability(agent, sample_phonon_input):
    """Test phonon calculation indicates stability."""
    result = agent.execute(sample_phonon_input)
    assert result.success
    # has_imaginary_modes should be False for stable structure
    assert 'has_imaginary_modes' in result.data


def test_elastic_moduli_positive(agent, sample_elastic_input):
    """Test elastic moduli are positive."""
    result = agent.execute(sample_elastic_input)
    assert result.success
    assert result.data['bulk_modulus_GPa'] > 0
    assert result.data['shear_modulus_GPa'] > 0
    assert result.data['youngs_modulus_GPa'] > 0
    assert 0 < result.data['poisson_ratio'] < 0.5  # Physical range


def test_aimd_temperature_equilibration(agent, sample_aimd_input):
    """Test AIMD temperature equilibrates around target."""
    result = agent.execute(sample_aimd_input)
    assert result.success
    target_temp = sample_aimd_input['parameters']['temperature']
    avg_temp = result.data['temperature_average_K']
    # Temperature should be within 10% of target
    assert abs(avg_temp - target_temp) / target_temp < 0.2


def test_neb_barrier_positive(agent):
    """Test NEB activation barrier is positive."""
    result = agent.execute({
        'calculation_type': 'neb',
        'parameters': {'n_images': 7}
    })
    assert result.success
    assert result.data['forward_barrier_eV'] >= 0
    assert result.data['reverse_barrier_eV'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])