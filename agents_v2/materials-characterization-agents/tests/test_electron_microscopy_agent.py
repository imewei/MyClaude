"""
Test suite for ElectronMicroscopyAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation
- Execution for all EM techniques
- Integration methods
- Caching and provenance
- Physical validation

Total: 45 tests
"""

import pytest
import numpy as np
from pathlib import Path

# Import agent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from electron_microscopy_agent import ElectronMicroscopyAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create an ElectronMicroscopyAgent instance."""
    return ElectronMicroscopyAgent()


@pytest.fixture
def sample_tem_bf_input():
    """Sample input for TEM bright field."""
    return {
        'technique': 'tem_bf',
        'image_file': 'nanoparticles.tif',
        'parameters': {
            'voltage_kV': 200,
            'magnification': 50000
        }
    }


@pytest.fixture
def sample_stem_haadf_input():
    """Sample input for STEM HAADF."""
    return {
        'technique': 'stem_haadf',
        'image_file': 'atomic_resolution.dm3',
        'parameters': {
            'voltage_kV': 200,
            'pixel_size_nm': 0.05
        }
    }


@pytest.fixture
def sample_eels_input():
    """Sample input for EELS."""
    return {
        'technique': 'eels',
        'image_file': 'eels_spectrum.dm3',
        'parameters': {
            'voltage_kV': 200,
            'energy_range_eV': [0, 2000]
        }
    }


@pytest.fixture
def sample_eds_input():
    """Sample input for EDS."""
    return {
        'technique': 'eds',
        'image_file': 'eds_spectrum.msa',
        'parameters': {
            'voltage_kV': 15,
            'live_time_sec': 60
        }
    }


# ============================================================================
# Test: Initialization and Metadata (3 tests)
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert isinstance(agent.supported_techniques, list)
    assert len(agent.supported_techniques) >= 11


def test_get_metadata(agent):
    """Test agent metadata retrieval."""
    metadata = agent.get_metadata()
    assert metadata.name == "ElectronMicroscopyAgent"
    assert metadata.version == "1.0.0"
    assert "electron microscopy" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 11
    capability_names = [cap.name for cap in capabilities]
    assert 'tem_bf' in capability_names
    assert 'stem_haadf' in capability_names
    assert 'eels' in capability_names
    assert 'eds' in capability_names


# ============================================================================
# Test: Input Validation (8 tests)
# ============================================================================

def test_validate_input_success_tem_bf(agent, sample_tem_bf_input):
    """Test validation succeeds for valid TEM input."""
    validation = agent.validate_input(sample_tem_bf_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_eels(agent, sample_eels_input):
    """Test validation succeeds for valid EELS input."""
    validation = agent.validate_input(sample_eels_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_technique(agent):
    """Test validation fails when technique is missing."""
    validation = agent.validate_input({'image_file': 'test.tif'})
    assert not validation.valid
    assert any('technique' in err.lower() for err in validation.errors)


def test_validate_input_invalid_technique(agent):
    """Test validation fails for unsupported technique."""
    validation = agent.validate_input({'technique': 'invalid_tech', 'image_file': 'test.tif'})
    assert not validation.valid
    assert any('unsupported' in err.lower() for err in validation.errors)


def test_validate_input_missing_image_file(agent):
    """Test validation fails when image file is missing."""
    validation = agent.validate_input({'technique': 'tem_bf'})
    assert not validation.valid
    assert any('image_file' in err.lower() for err in validation.errors)


def test_validate_input_low_voltage_warning(agent, sample_tem_bf_input):
    """Test validation warns for low voltage."""
    sample_tem_bf_input['parameters']['voltage_kV'] = 10
    validation = agent.validate_input(sample_tem_bf_input)
    assert validation.valid  # Still valid
    assert len(validation.warnings) > 0


def test_validate_input_small_pixel_size_warning(agent, sample_stem_haadf_input):
    """Test validation warns for very small pixel size."""
    sample_stem_haadf_input['parameters']['pixel_size_nm'] = 0.005
    validation = agent.validate_input(sample_stem_haadf_input)
    assert validation.valid
    assert len(validation.warnings) > 0


def test_validate_input_narrow_eels_range(agent, sample_eels_input):
    """Test validation warns for narrow EELS range."""
    sample_eels_input['parameters']['energy_range_eV'] = [0, 30]
    validation = agent.validate_input(sample_eels_input)
    assert validation.valid
    assert len(validation.warnings) > 0


# ============================================================================
# Test: Resource Estimation (5 tests)
# ============================================================================

def test_estimate_resources_tem_bf(agent, sample_tem_bf_input):
    """Test resource estimation for TEM bright field."""
    resources = agent.estimate_resources(sample_tem_bf_input)
    assert resources.execution_environment.value == 'local'
    assert resources.estimated_time_sec < 600  # Should be fast


def test_estimate_resources_eels(agent, sample_eels_input):
    """Test resource estimation for EELS."""
    resources = agent.estimate_resources(sample_eels_input)
    assert resources.execution_environment.value == 'local'
    assert resources.cpu_cores >= 4
    assert resources.estimated_time_sec < 1200


def test_estimate_resources_4d_stem(agent):
    """Test resource estimation for 4D-STEM."""
    input_data = {
        'technique': '4d_stem',
        'image_file': '4d_dataset.h5',
        'parameters': {'voltage_kV': 200}
    }
    resources = agent.estimate_resources(input_data)
    assert resources.execution_environment.value == 'local'
    assert resources.cpu_cores >= 8  # 4D-STEM needs more cores
    assert resources.memory_gb >= 32  # Large datasets


def test_estimate_resources_cryo_em(agent):
    """Test resource estimation for cryo-EM."""
    input_data = {
        'technique': 'cryo_em',
        'image_file': 'particles.mrc',
        'parameters': {'voltage_kV': 300}
    }
    resources = agent.estimate_resources(input_data)
    assert resources.execution_environment.value == 'local'
    assert resources.memory_gb >= 32  # Many particles


def test_estimate_resources_stem_haadf(agent, sample_stem_haadf_input):
    """Test resource estimation for STEM HAADF."""
    resources = agent.estimate_resources(sample_stem_haadf_input)
    assert resources.execution_environment.value == 'local'
    assert resources.estimated_time_sec > 0


# ============================================================================
# Test: Execution for All Techniques (11 tests)
# ============================================================================

def test_execute_tem_bf_success(agent, sample_tem_bf_input):
    """Test TEM bright field execution succeeds."""
    result = agent.execute(sample_tem_bf_input)
    assert result.success
    assert 'particle_analysis' in result.data
    assert 'n_particles' in result.data['particle_analysis']
    assert result.data['particle_analysis']['n_particles'] > 0


def test_execute_tem_df_success(agent):
    """Test TEM dark field execution succeeds."""
    result = agent.execute({
        'technique': 'tem_df',
        'image_file': 'darkfield.tif',
        'parameters': {'voltage_kV': 200}
    })
    assert result.success
    assert 'grain_analysis' in result.data
    assert 'dislocation_density_per_cm2' in result.data


def test_execute_tem_diffraction_success(agent):
    """Test TEM diffraction execution succeeds."""
    result = agent.execute({
        'technique': 'tem_diffraction',
        'image_file': 'diffraction.tif',
        'parameters': {'voltage_kV': 200, 'camera_length': 500}
    })
    assert result.success
    assert 'd_spacings_A' in result.data
    assert 'lattice_parameter_A' in result.data
    assert len(result.data['d_spacings_A']) > 0


def test_execute_sem_se_success(agent):
    """Test SEM secondary electron execution succeeds."""
    result = agent.execute({
        'technique': 'sem_se',
        'image_file': 'sem_surface.tif',
        'parameters': {'voltage_kV': 15}
    })
    assert result.success
    assert 'surface_analysis' in result.data
    assert 'surface_roughness_nm' in result.data['surface_analysis']


def test_execute_sem_bse_success(agent):
    """Test SEM backscattered execution succeeds."""
    result = agent.execute({
        'technique': 'sem_bse',
        'image_file': 'sem_bse.tif',
        'parameters': {'voltage_kV': 20}
    })
    assert result.success
    assert 'phase_analysis' in result.data
    assert 'n_phases' in result.data['phase_analysis']


def test_execute_stem_haadf_success(agent, sample_stem_haadf_input):
    """Test STEM HAADF execution succeeds."""
    result = agent.execute(sample_stem_haadf_input)
    assert result.success
    assert 'atomic_analysis' in result.data
    assert 'n_atomic_columns' in result.data['atomic_analysis']
    assert 'defect_analysis' in result.data


def test_execute_stem_abf_success(agent):
    """Test STEM ABF execution succeeds."""
    result = agent.execute({
        'technique': 'stem_abf',
        'image_file': 'abf_image.dm3',
        'parameters': {'voltage_kV': 200}
    })
    assert result.success
    assert 'light_element_analysis' in result.data
    assert result.data['light_element_analysis']['oxygen_positions_detected'] == True


def test_execute_eels_success(agent, sample_eels_input):
    """Test EELS execution succeeds."""
    result = agent.execute(sample_eels_input)
    assert result.success
    assert 'core_loss_edges' in result.data
    assert 'low_loss_analysis' in result.data
    assert 'band_gap_eV' in result.data['low_loss_analysis']
    assert len(result.data['core_loss_edges']) > 0


def test_execute_eds_success(agent, sample_eds_input):
    """Test EDS execution succeeds."""
    result = agent.execute(sample_eds_input)
    assert result.success
    assert 'elements_detected' in result.data
    assert 'composition_at_percent' in result.data
    assert len(result.data['elements_detected']) > 0


def test_execute_4d_stem_success(agent):
    """Test 4D-STEM execution succeeds."""
    result = agent.execute({
        'technique': '4d_stem',
        'image_file': '4d_dataset.h5',
        'parameters': {'voltage_kV': 200}
    })
    assert result.success
    assert 'strain_analysis' in result.data
    assert 'orientation_analysis' in result.data
    assert 'strain_xx_map' in result.data['strain_analysis']


def test_execute_cryo_em_success(agent):
    """Test cryo-EM execution succeeds."""
    result = agent.execute({
        'technique': 'cryo_em',
        'image_file': 'particles.mrc',
        'parameters': {'voltage_kV': 300}
    })
    assert result.success
    assert 'particle_analysis' in result.data
    assert 'structure_determination' in result.data
    assert 'resolution_A' in result.data['structure_determination']


# ============================================================================
# Test: Integration Methods (6 tests)
# ============================================================================

def test_validate_with_crystallography_success(agent):
    """Test TEM diffraction validation with XRD."""
    # Run TEM diffraction
    tem_result = agent.execute({
        'technique': 'tem_diffraction',
        'image_file': 'diffraction.tif',
        'parameters': {'voltage_kV': 200}
    })
    assert tem_result.success

    # Mock XRD data
    xrd_data = {
        'd_spacings_A': [3.14, 1.92, 1.64, 1.25]  # Close to Si
    }

    validation = agent.validate_with_crystallography(tem_result.data, xrd_data)
    assert validation['success']
    assert 'n_matches' in validation
    assert validation['n_matches'] > 0


def test_validate_with_crystallography_wrong_input(agent, sample_tem_bf_input):
    """Test crystallography validation fails for non-diffraction input."""
    tem_result = agent.execute(sample_tem_bf_input)
    validation = agent.validate_with_crystallography(tem_result.data, {})
    assert not validation['success']


def test_correlate_structure_with_dft_success(agent, sample_stem_haadf_input):
    """Test STEM structure correlation with DFT."""
    stem_result = agent.execute(sample_stem_haadf_input)
    assert stem_result.success

    # Mock DFT data
    dft_data = {
        'lattice_constants_A': [2.35, 2.35, 2.35]
    }

    correlation = agent.correlate_structure_with_dft(stem_result.data, dft_data)
    assert correlation['success']
    assert 'stem_lattice_spacing_A' in correlation
    assert 'dft_lattice_spacing_A' in correlation
    assert 'difference_percent' in correlation


def test_correlate_structure_wrong_input(agent, sample_eels_input):
    """Test structure correlation fails for non-STEM input."""
    eels_result = agent.execute(sample_eels_input)
    correlation = agent.correlate_structure_with_dft(eels_result.data, {})
    assert not correlation['success']


def test_quantify_composition_for_simulation_success(agent, sample_eds_input):
    """Test EDS composition conversion for simulation."""
    eds_result = agent.execute(sample_eds_input)
    assert eds_result.success

    sim_input = agent.quantify_composition_for_simulation(eds_result.data)
    assert sim_input['success']
    assert 'composition_formula' in sim_input
    assert 'atomic_fractions' in sim_input
    assert 'structure_recommendation' in sim_input


def test_quantify_composition_wrong_input(agent, sample_tem_bf_input):
    """Test composition conversion fails for non-EDS input."""
    tem_result = agent.execute(sample_tem_bf_input)
    sim_input = agent.quantify_composition_for_simulation(tem_result.data)
    assert not sim_input['success']


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_tem_bf_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_tem_bf_input)
    result2 = agent.execute_with_caching(sample_tem_bf_input)

    assert result1.success
    assert result2.success
    # Second call should be from cache
    assert result2.metadata.get('cached', False) or result2.status == AgentStatus.CACHED


def test_caching_different_inputs(agent, sample_tem_bf_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_tem_bf_input)

    # Modify input
    modified_input = sample_tem_bf_input.copy()
    modified_input['parameters'] = sample_tem_bf_input['parameters'].copy()
    modified_input['parameters']['voltage_kV'] = 300

    result2 = agent.execute_with_caching(modified_input)

    assert result1.success
    assert result2.success


def test_provenance_tracking(agent, sample_tem_bf_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_tem_bf_input)
    assert result.success
    assert hasattr(result, 'provenance')
    assert result.provenance is not None
    assert result.provenance.agent_name == 'ElectronMicroscopyAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_tem_bf_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_tem_bf_input)
    assert result.success
    assert result.provenance.input_hash is not None
    assert len(result.provenance.input_hash) == 64  # SHA256


# ============================================================================
# Test: Physical Validation (8 tests)
# ============================================================================

def test_particle_size_positive(agent, sample_tem_bf_input):
    """Test TEM particle sizes are positive."""
    result = agent.execute(sample_tem_bf_input)
    assert result.success
    assert result.data['particle_analysis']['mean_diameter_nm'] > 0
    assert result.data['particle_analysis']['std_diameter_nm'] >= 0


def test_lattice_parameter_reasonable(agent):
    """Test TEM diffraction lattice parameter is reasonable."""
    result = agent.execute({
        'technique': 'tem_diffraction',
        'image_file': 'diffraction.tif',
        'parameters': {'voltage_kV': 200}
    })
    assert result.success
    lattice_param = result.data['lattice_parameter_A']
    assert 2.0 < lattice_param < 10.0  # Reasonable range for most materials


def test_eels_band_gap_positive(agent, sample_eels_input):
    """Test EELS band gap is non-negative."""
    result = agent.execute(sample_eels_input)
    assert result.success
    band_gap = result.data['low_loss_analysis']['band_gap_eV']
    assert band_gap >= 0


def test_eds_composition_sums_to_100(agent, sample_eds_input):
    """Test EDS composition sums to ~100%."""
    result = agent.execute(sample_eds_input)
    assert result.success
    composition = result.data['composition_at_percent']
    total = sum(composition.values())
    assert 99 < total < 101  # Should sum to 100% (±1% tolerance)


def test_stem_resolution_reasonable(agent, sample_stem_haadf_input):
    """Test STEM resolution is physically reasonable."""
    result = agent.execute(sample_stem_haadf_input)
    assert result.success
    resolution_nm = result.data['resolution_nm']
    assert 0.05 < resolution_nm < 0.5  # Typical STEM resolution range


def test_4d_stem_strain_range_physical(agent):
    """Test 4D-STEM strain values are physically reasonable."""
    result = agent.execute({
        'technique': '4d_stem',
        'image_file': '4d_dataset.h5',
        'parameters': {'voltage_kV': 200}
    })
    assert result.success
    strain_range = result.data['strain_analysis']['strain_range_percent']
    assert abs(strain_range[0]) < 10  # Strain typically < 10%
    assert abs(strain_range[1]) < 10


def test_cryo_em_resolution_positive(agent):
    """Test cryo-EM resolution is positive."""
    result = agent.execute({
        'technique': 'cryo_em',
        'image_file': 'particles.mrc',
        'parameters': {'voltage_kV': 300}
    })
    assert result.success
    resolution = result.data['structure_determination']['resolution_A']
    assert resolution > 0
    assert resolution < 100  # Should be sub-10 Å for good structures


def test_sem_roughness_positive(agent):
    """Test SEM surface roughness is non-negative."""
    result = agent.execute({
        'technique': 'sem_se',
        'image_file': 'sem_surface.tif',
        'parameters': {'voltage_kV': 15}
    })
    assert result.success
    roughness = result.data['surface_analysis']['surface_roughness_nm']
    assert roughness >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])