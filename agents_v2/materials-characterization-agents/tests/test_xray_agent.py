"""Test suite for X-ray Scattering Agent.

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation
- All X-ray techniques (SAXS, WAXS, GISAXS, RSoXS, XPCS, XAS, time-resolved)
- Integration methods
- Caching and provenance
- Physical validation
"""

import pytest
import numpy as np
from xray_agent import XRayAgent
from base_agent import AgentStatus, ExecutionEnvironment


class TestXRayAgentInitialization:
    """Test agent initialization and metadata."""

    def test_initialization_default(self):
        """Test default initialization."""
        agent = XRayAgent()
        assert agent.beamline == 'generic'
        assert agent.energy_kev == 10.0
        assert agent.detector == 'pilatus'
        assert agent.sample_detector_distance == 1000.0

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            'beamline': 'ALS_7.3.3',
            'energy': 285.0,
            'detector': 'princeton',
            'sample_detector_distance': 5000.0
        }
        agent = XRayAgent(config=config)
        assert agent.beamline == 'ALS_7.3.3'
        assert agent.energy_kev == 285.0
        assert agent.detector == 'princeton'
        assert agent.sample_detector_distance == 5000.0

    def test_get_metadata(self):
        """Test metadata retrieval."""
        agent = XRayAgent()
        metadata = agent.get_metadata()
        assert metadata.name == "XRayAgent"
        assert metadata.version == "1.0.0"
        assert len(metadata.capabilities) == 6  # 6 techniques
        assert 'numpy' in metadata.dependencies


class TestXRayAgentValidation:
    """Test input validation."""

    def test_validate_valid_saxs_input(self):
        """Test validation of valid SAXS input."""
        agent = XRayAgent()
        input_data = {
            'technique': 'SAXS',
            'data_file': 'sample.dat'
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert len(validation.errors) == 0

    def test_validate_missing_technique(self):
        """Test validation fails when technique missing."""
        agent = XRayAgent()
        input_data = {'data_file': 'sample.dat'}
        validation = agent.validate_input(input_data)
        assert not validation.valid
        assert len(validation.errors) == 1

    def test_validate_unsupported_technique(self):
        """Test validation fails for unsupported technique."""
        agent = XRayAgent()
        input_data = {
            'technique': 'invalid_technique',
            'data_file': 'sample.dat'
        }
        validation = agent.validate_input(input_data)
        assert not validation.valid
        assert 'Unsupported technique' in validation.errors[0]

    def test_validate_missing_data_warning(self):
        """Test warning when no data provided."""
        agent = XRayAgent()
        input_data = {'technique': 'SAXS'}
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert len(validation.warnings) == 1
        assert 'No data provided' in validation.warnings[0]

    def test_validate_xpcs_missing_q(self):
        """Test XPCS validation warns when q_value missing."""
        agent = XRayAgent()
        input_data = {
            'technique': 'XPCS',
            'parameters': {}
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert any('q_value' in w for w in validation.warnings)

    def test_validate_xas_missing_element(self):
        """Test XAS validation warns when element missing."""
        agent = XRayAgent()
        input_data = {
            'technique': 'XAS',
            'parameters': {}
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert any('element' in w for w in validation.warnings)

    def test_validate_rsoxs_missing_energy(self):
        """Test RSoXS validation warns when energy missing."""
        agent = XRayAgent()
        input_data = {
            'technique': 'RSoXS',
            'parameters': {}
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert any('energy' in w for w in validation.warnings)

    def test_validate_case_insensitive(self):
        """Test technique validation is case-insensitive."""
        agent = XRayAgent()
        for technique in ['SAXS', 'saxs', 'SaXs']:
            input_data = {'technique': technique, 'data_file': 'test.dat'}
            validation = agent.validate_input(input_data)
            assert validation.valid


class TestXRayAgentResourceEstimation:
    """Test resource estimation."""

    def test_estimate_resources_saxs(self):
        """Test resource estimation for SAXS."""
        agent = XRayAgent()
        input_data = {'technique': 'SAXS'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 1
        assert resources.memory_gb == 2.0
        assert resources.execution_environment == ExecutionEnvironment.LOCAL

    def test_estimate_resources_xpcs(self):
        """Test resource estimation for XPCS (more intensive)."""
        agent = XRayAgent()
        input_data = {'technique': 'XPCS'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 4
        assert resources.memory_gb == 8.0
        assert resources.estimated_time_sec == 300.0

    def test_estimate_resources_gisaxs(self):
        """Test resource estimation for GISAXS (2D analysis)."""
        agent = XRayAgent()
        input_data = {'technique': 'GISAXS'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 2
        assert resources.memory_gb == 4.0
        assert resources.estimated_time_sec == 120.0

    def test_estimate_resources_time_resolved(self):
        """Test resource estimation for time-resolved (many frames)."""
        agent = XRayAgent()
        input_data = {
            'technique': 'time_resolved',
            'duration_sec': 1000,
            'time_resolution_ms': 10.0
        }
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 4
        assert resources.memory_gb >= 4.0
        assert resources.estimated_time_sec == 180.0


class TestXRayAgentExecution:
    """Test execution of all X-ray techniques."""

    def test_execute_saxs(self):
        """Test SAXS execution."""
        agent = XRayAgent()
        input_data = {
            'technique': 'SAXS',
            'q_range': [0.001, 0.5],
            'n_points': 100
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'SAXS'
        assert 'scattering_vector_inv_angstrom' in result.data
        assert 'guinier_analysis' in result.data
        assert 'porod_analysis' in result.data
        assert 'form_factor_fit' in result.data

        # Check Guinier analysis
        guinier = result.data['guinier_analysis']
        assert 'radius_of_gyration_nm' in guinier
        assert guinier['radius_of_gyration_nm'] > 0

        # Check physical properties
        props = result.data['physical_properties']
        assert 'particle_size_nm' in props
        assert props['aggregation_state'] in ['dispersed', 'aggregated']

    def test_execute_waxs(self):
        """Test WAXS execution."""
        agent = XRayAgent()
        input_data = {
            'technique': 'WAXS',
            'q_range': [0.5, 3.0],
            'n_points': 200
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'WAXS'
        assert 'crystalline_peaks' in result.data
        assert 'crystallinity_analysis' in result.data
        assert 'orientation' in result.data

        # Check crystallinity
        cryst = result.data['crystallinity_analysis']
        assert 'crystallinity_percent' in cryst
        assert 0 <= cryst['crystallinity_percent'] <= 100

        # Check peaks
        peaks = result.data['crystalline_peaks']
        assert len(peaks) > 0
        for peak in peaks:
            assert 'q_position_inv_angstrom' in peak
            assert 'd_spacing_angstrom' in peak

    def test_execute_gisaxs(self):
        """Test GISAXS execution."""
        agent = XRayAgent()
        input_data = {
            'technique': 'GISAXS',
            'qxy_range': [0.001, 0.1],
            'qz_range': [0.001, 0.2]
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'GISAXS'
        assert 'in_plane_structure' in result.data
        assert 'out_of_plane_structure' in result.data
        assert 'morphology' in result.data

        # Check structure
        in_plane = result.data['in_plane_structure']
        assert 'domain_spacing_nm' in in_plane
        assert in_plane['domain_spacing_nm'] > 0

        # Check morphology
        morph = result.data['morphology']
        assert 'structure_type' in morph
        assert 'orientation' in morph

    def test_execute_rsoxs(self):
        """Test RSoXS execution."""
        agent = XRayAgent()
        input_data = {
            'technique': 'RSoXS',
            'parameters': {'energy_ev': 284.0}
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'RSoXS'
        assert 'chemical_contrast' in result.data
        assert 'energy_scan' in result.data
        assert 'phase_separation' in result.data

        # Check chemical contrast
        contrast = result.data['chemical_contrast']
        assert 'domain_purity' in contrast
        assert 0 <= contrast['domain_purity'] <= 1

        # Check energy scan
        energy = result.data['energy_scan']
        assert 'resonant_peaks_ev' in energy
        assert len(energy['resonant_peaks_ev']) > 0

    def test_execute_xpcs(self):
        """Test XPCS execution."""
        agent = XRayAgent()
        input_data = {
            'technique': 'XPCS',
            'parameters': {
                'q_value': 0.01,
                'max_time_sec': 100.0
            }
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'XPCS'
        assert 'correlation_function' in result.data
        assert 'dynamics_analysis' in result.data

        # Check correlation function
        corr = result.data['correlation_function']
        assert 'delay_times_sec' in corr
        assert 'g2_minus_1' in corr
        assert len(corr['delay_times_sec']) == len(corr['g2_minus_1'])

        # Check dynamics
        dynamics = result.data['dynamics_analysis']
        assert 'relaxation_time_sec' in dynamics
        assert dynamics['relaxation_time_sec'] > 0

    def test_execute_xas(self):
        """Test XAS execution."""
        agent = XRayAgent()
        input_data = {
            'technique': 'XAS',
            'parameters': {
                'element': 'Fe',
                'edge': 'K'
            }
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'XAS'
        assert 'xanes_analysis' in result.data
        assert 'exafs_analysis' in result.data
        assert 'chemical_state' in result.data

        # Check XANES
        xanes = result.data['xanes_analysis']
        assert 'oxidation_state' in xanes
        assert 'coordination_geometry' in xanes

        # Check EXAFS
        exafs = result.data['exafs_analysis']
        assert 'first_shell_distance_angstrom' in exafs
        assert exafs['coordination_number'] > 0

    def test_execute_time_resolved(self):
        """Test time-resolved execution."""
        agent = XRayAgent()
        input_data = {
            'technique': 'time_resolved',
            'parameters': {
                'time_resolution_ms': 1.0,
                'duration_sec': 100.0
            }
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'Time-Resolved X-ray'
        assert 'kinetics' in result.data
        assert 'structural_changes' in result.data

        # Check kinetics
        kinetics = result.data['kinetics']
        assert 'times_sec' in kinetics
        assert 'crystallinity_evolution' in kinetics
        assert 'rate_constant' in kinetics

        # Check structural changes
        changes = result.data['structural_changes']
        assert 'initial_phase' in changes
        assert 'final_phase' in changes

    def test_execute_invalid_technique(self):
        """Test execution fails for invalid technique."""
        agent = XRayAgent()
        input_data = {'technique': 'invalid'}
        result = agent.execute(input_data)
        assert not result.success
        assert len(result.errors) > 0


class TestXRayAgentProvenance:
    """Test provenance tracking."""

    def test_provenance_created(self):
        """Test that provenance is created."""
        agent = XRayAgent()
        input_data = {'technique': 'SAXS'}
        result = agent.execute(input_data)
        assert result.provenance is not None
        assert result.provenance.agent_name == "XRayAgent"
        assert result.provenance.agent_version == "1.0.0"

    def test_provenance_parameters(self):
        """Test that parameters are recorded in provenance."""
        agent = XRayAgent(config={'beamline': 'test_beamline'})
        input_data = {
            'technique': 'SAXS',
            'parameters': {'q_range': [0.001, 0.5]}
        }
        result = agent.execute(input_data)
        assert result.provenance.parameters['technique'] == 'saxs'
        assert result.provenance.parameters['beamline'] == 'test_beamline'

    def test_provenance_input_hash(self):
        """Test that input hash is unique."""
        agent = XRayAgent()
        input_data1 = {'technique': 'SAXS', 'q_range': [0.001, 0.5]}
        input_data2 = {'technique': 'SAXS', 'q_range': [0.001, 0.6]}

        result1 = agent.execute(input_data1)
        result2 = agent.execute(input_data2)

        assert result1.provenance.input_hash != result2.provenance.input_hash


class TestXRayAgentCaching:
    """Test result caching."""

    def test_caching_works(self):
        """Test that results are cached."""
        # Create a fresh agent instance for this test
        import uuid
        agent = XRayAgent()
        agent.clear_cache()  # Start with clean cache
        # Use unique UUID to avoid cross-test contamination
        input_data = {'technique': 'SAXS', 'cache_test': str(uuid.uuid4())}

        result1 = agent.execute_with_caching(input_data)
        result2 = agent.execute_with_caching(input_data)

        # First result should be computed
        assert result1.status in [AgentStatus.SUCCESS, AgentStatus.CACHED]
        # Second should be cached
        if result1.status == AgentStatus.SUCCESS:
            assert result2.status == AgentStatus.CACHED
            assert result2.metadata['cached'] is True

    def test_cache_invalidated_on_different_input(self):
        """Test cache is not used for different inputs."""
        agent = XRayAgent()

        result1 = agent.execute_with_caching({'technique': 'SAXS'})
        result2 = agent.execute_with_caching({'technique': 'WAXS'})

        assert result1.status == AgentStatus.SUCCESS
        assert result2.status == AgentStatus.SUCCESS  # Not cached

    def test_clear_cache(self):
        """Test cache clearing."""
        agent = XRayAgent()
        input_data = {'technique': 'SAXS'}

        result1 = agent.execute_with_caching(input_data)
        agent.clear_cache()
        result2 = agent.execute_with_caching(input_data)

        assert result1.status == AgentStatus.SUCCESS
        assert result2.status == AgentStatus.SUCCESS  # Not cached


class TestXRayAgentIntegration:
    """Test integration methods with other agents."""

    def test_validate_with_neutron_sans(self):
        """Test SAXS/SANS cross-validation."""
        saxs_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 10.0
            }
        }
        sans_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 10.5
            }
        }

        validation = XRayAgent.validate_with_neutron_sans(saxs_result, sans_result)
        assert 'validation_type' in validation
        assert validation['validation_type'] == 'SAXS_SANS_cross_check'
        assert 'rg_agreement_percent' in validation
        assert validation['consistent'] is True

    def test_validate_with_neutron_sans_inconsistent(self):
        """Test SAXS/SANS validation detects inconsistency."""
        saxs_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 10.0
            }
        }
        sans_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 20.0
            }
        }

        validation = XRayAgent.validate_with_neutron_sans(saxs_result, sans_result)
        assert validation['consistent'] is False

    def test_correlate_with_microscopy(self):
        """Test GISAXS/TEM correlation."""
        gisaxs_result = {
            'in_plane_structure': {
                'domain_spacing_nm': 25.0
            }
        }
        tem_result = {
            'particle_analysis': {
                'mean_spacing_nm': 26.0
            }
        }

        correlation = XRayAgent.correlate_with_microscopy(gisaxs_result, tem_result)
        assert 'correlation_type' in correlation
        assert correlation['correlation_type'] == 'GISAXS_TEM'
        assert 'reciprocal_vs_real_space' in correlation
        assert 'agreement_percent' in correlation['reciprocal_vs_real_space']
        assert correlation['consistent'] is True

    def test_extract_structure_for_simulation(self):
        """Test extracting structure parameters for MD."""
        saxs_result = {
            'form_factor_fit': {
                'radius_nm': 8.0,
                'polydispersity': 0.15
            },
            'structure_factor': {
                'structure_factor_model': 'hard_sphere',
                'correlation_length_nm': 12.0
            }
        }

        sim_params = XRayAgent.extract_structure_for_simulation(saxs_result)
        assert 'particle_size_nm' in sim_params
        assert sim_params['particle_size_nm'] == 16.0  # 2 * radius
        assert 'polydispersity' in sim_params
        assert 'simulation_suggestions' in sim_params


class TestXRayAgentPhysicalValidation:
    """Test physical validation of results."""

    def test_saxs_guinier_regime_valid(self):
        """Test SAXS Guinier region is physically valid."""
        agent = XRayAgent()
        input_data = {'technique': 'SAXS'}
        result = agent.execute(input_data)

        guinier = result.data['guinier_analysis']
        rg = guinier['radius_of_gyration_nm']

        # Rg should be positive and reasonable
        assert 0 < rg < 1000

    def test_waxs_crystallinity_valid(self):
        """Test WAXS crystallinity is in valid range."""
        agent = XRayAgent()
        input_data = {'technique': 'WAXS'}
        result = agent.execute(input_data)

        cryst = result.data['crystallinity_analysis']['crystallinity_percent']
        assert 0 <= cryst <= 100

    def test_waxs_d_spacing_valid(self):
        """Test WAXS d-spacings are physically reasonable."""
        agent = XRayAgent()
        input_data = {'technique': 'WAXS'}
        result = agent.execute(input_data)

        for peak in result.data['crystalline_peaks']:
            d_spacing = peak['d_spacing_angstrom']
            # Typical d-spacings for polymers: 2-10 Å
            assert 1.0 < d_spacing < 20.0

    def test_xpcs_correlation_decay(self):
        """Test XPCS correlation function decays properly."""
        agent = XRayAgent()
        input_data = {'technique': 'XPCS'}
        result = agent.execute(input_data)

        g2 = np.array(result.data['correlation_function']['g2_minus_1'])
        times = np.array(result.data['correlation_function']['delay_times_sec'])

        # g2-1 should decay with time
        assert g2[0] > g2[-1]
        # g2-1 should be non-negative
        assert np.all(g2 >= 0)

    def test_gisaxs_domain_spacing_valid(self):
        """Test GISAXS domain spacing is reasonable."""
        agent = XRayAgent()
        input_data = {'technique': 'GISAXS'}
        result = agent.execute(input_data)

        spacing = result.data['in_plane_structure']['domain_spacing_nm']
        # Block copolymer domains typically 10-100 nm
        assert 5.0 < spacing < 500.0

    def test_rsoxs_domain_purity_valid(self):
        """Test RSoXS domain purity is in valid range."""
        agent = XRayAgent()
        input_data = {'technique': 'RSoXS'}
        result = agent.execute(input_data)

        purity = result.data['chemical_contrast']['domain_purity']
        assert 0 <= purity <= 1

    def test_xas_coordination_valid(self):
        """Test XAS coordination number is reasonable."""
        agent = XRayAgent()
        input_data = {'technique': 'XAS'}
        result = agent.execute(input_data)

        cn = result.data['exafs_analysis']['coordination_number']
        # Typical coordination numbers: 4-12
        assert 2 <= cn <= 14

    def test_time_resolved_kinetics_monotonic(self):
        """Test time-resolved crystallinity increases monotonically."""
        agent = XRayAgent()
        input_data = {'technique': 'time_resolved'}
        result = agent.execute(input_data)

        cryst = np.array(result.data['kinetics']['crystallinity_evolution'])
        # Crystallinity should be monotonically increasing
        assert np.all(np.diff(cryst) >= -0.01)  # Allow small numerical errors
        # Crystallinity should be in [0, 1]
        assert np.all((cryst >= 0) & (cryst <= 1))


class TestXRayAgentCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self):
        """Test that capabilities are correctly reported."""
        agent = XRayAgent()
        capabilities = agent.get_capabilities()
        assert len(capabilities) == 6

        # Check that all techniques are covered
        technique_names = [cap.name for cap in capabilities]
        assert "SAXS Analysis" in technique_names
        assert "GISAXS Analysis" in technique_names
        assert "RSoXS Analysis" in technique_names
        assert "XPCS Analysis" in technique_names
        assert "XAS Analysis" in technique_names
        assert "Time-Resolved X-ray" in technique_names

    def test_capability_structure(self):
        """Test that capabilities have required fields."""
        agent = XRayAgent()
        capabilities = agent.get_capabilities()

        for cap in capabilities:
            assert hasattr(cap, 'name')
            assert hasattr(cap, 'description')
            assert hasattr(cap, 'input_types')
            assert hasattr(cap, 'output_types')
            assert hasattr(cap, 'typical_use_cases')


class TestXRayAgentInstrumentConnection:
    """Test instrument connection methods."""

    def test_connect_instrument(self):
        """Test instrument connection."""
        agent = XRayAgent()
        connected = agent.connect_instrument()
        assert connected is True

    def test_process_experimental_data(self):
        """Test experimental data processing."""
        agent = XRayAgent()
        raw_data = {'intensity': [1, 2, 3], 'q': [0.01, 0.02, 0.03]}
        processed = agent.process_experimental_data(raw_data)
        # Placeholder returns same data
        assert processed == raw_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])