"""Test suite for Neutron Scattering Agent.

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation
- All neutron techniques (SANS, NSE, QENS, NR, INS, contrast variation)
- Integration methods
- Caching and provenance
- Physical validation
"""

import pytest
import numpy as np
from neutron_agent import NeutronAgent
from base_agent import AgentStatus, ExecutionEnvironment


class TestNeutronAgentInitialization:
    """Test agent initialization and metadata."""

    def test_initialization_default(self):
        """Test default initialization."""
        agent = NeutronAgent()
        assert agent.facility == 'generic'
        assert agent.instrument == 'generic'
        assert agent.wavelength_angstrom == 6.0
        assert agent.temperature_k == 298.0

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            'facility': 'NIST_NCNR',
            'instrument': 'NG7_SANS',
            'wavelength': 5.0,
            'temperature': 323.0
        }
        agent = NeutronAgent(config=config)
        assert agent.facility == 'NIST_NCNR'
        assert agent.instrument == 'NG7_SANS'
        assert agent.wavelength_angstrom == 5.0
        assert agent.temperature_k == 323.0

    def test_get_metadata(self):
        """Test metadata retrieval."""
        agent = NeutronAgent()
        metadata = agent.get_metadata()
        assert metadata.name == "NeutronAgent"
        assert metadata.version == "1.0.0"
        assert len(metadata.capabilities) == 6  # 6 techniques
        assert 'numpy' in metadata.dependencies


class TestNeutronAgentValidation:
    """Test input validation."""

    def test_validate_valid_sans_input(self):
        """Test validation of valid SANS input."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'SANS',
            'data_file': 'sample.nxs'
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert len(validation.errors) == 0

    def test_validate_missing_technique(self):
        """Test validation fails when technique missing."""
        agent = NeutronAgent()
        input_data = {'data_file': 'sample.nxs'}
        validation = agent.validate_input(input_data)
        assert not validation.valid
        assert len(validation.errors) == 1

    def test_validate_unsupported_technique(self):
        """Test validation fails for unsupported technique."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'invalid_technique',
            'data_file': 'sample.nxs'
        }
        validation = agent.validate_input(input_data)
        assert not validation.valid
        assert 'Unsupported technique' in validation.errors[0]

    def test_validate_missing_data_warning(self):
        """Test warning when no data provided."""
        agent = NeutronAgent()
        input_data = {'technique': 'SANS'}
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert len(validation.warnings) >= 1
        assert any('No data provided' in w for w in validation.warnings)

    def test_validate_sans_missing_deuteration(self):
        """Test SANS validation warns when deuteration not specified."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'SANS',
            'parameters': {}
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert any('deuteration' in w for w in validation.warnings)

    def test_validate_nse_missing_fourier_times(self):
        """Test NSE validation warns when Fourier times missing."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'NSE',
            'parameters': {}
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert any('Fourier times' in w for w in validation.warnings)

    def test_validate_qens_missing_temperature(self):
        """Test QENS validation warns when temperature missing."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'QENS',
            'parameters': {}
        }
        validation = agent.validate_input(input_data)
        assert validation.valid
        assert any('temperature' in w for w in validation.warnings)

    def test_validate_case_insensitive(self):
        """Test technique validation is case-insensitive."""
        agent = NeutronAgent()
        for technique in ['SANS', 'sans', 'SaNs']:
            input_data = {'technique': technique, 'data_file': 'test.nxs'}
            validation = agent.validate_input(input_data)
            assert validation.valid


class TestNeutronAgentResourceEstimation:
    """Test resource estimation."""

    def test_estimate_resources_sans(self):
        """Test resource estimation for SANS."""
        agent = NeutronAgent()
        input_data = {'technique': 'SANS'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 1
        assert resources.memory_gb == 2.0
        assert resources.execution_environment == ExecutionEnvironment.LOCAL

    def test_estimate_resources_nse(self):
        """Test resource estimation for NSE (correlation analysis)."""
        agent = NeutronAgent()
        input_data = {'technique': 'NSE'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 2
        assert resources.memory_gb == 4.0
        assert resources.estimated_time_sec == 300.0

    def test_estimate_resources_qens(self):
        """Test resource estimation for QENS (model fitting)."""
        agent = NeutronAgent()
        input_data = {'technique': 'QENS'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 4
        assert resources.memory_gb == 4.0
        assert resources.estimated_time_sec == 240.0

    def test_estimate_resources_contrast_variation(self):
        """Test resource estimation for contrast variation (multiple contrasts)."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'sans_contrast',
            'd2o_fractions': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 2
        assert resources.memory_gb == 3.0
        assert resources.estimated_time_sec >= 180.0 * 6


class TestNeutronAgentExecution:
    """Test execution of all neutron techniques."""

    def test_execute_sans(self):
        """Test SANS execution."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'SANS',
            'q_range': [0.001, 0.5],
            'n_points': 100,
            'h_d_fraction': 0.5
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'SANS'
        assert 'scattering_vector_inv_angstrom' in result.data
        assert 'guinier_analysis' in result.data
        assert 'contrast_analysis' in result.data

        # Check deuteration
        assert result.data['deuteration_level'] == 0.5

        # Check contrast analysis
        contrast = result.data['contrast_analysis']
        assert 'scattering_length_density_contrast' in contrast
        assert 'h_fraction' in contrast
        assert 'd_fraction' in contrast

        # Check physical properties
        props = result.data['physical_properties']
        assert 'hydrodynamic_radius_nm' in props

    def test_execute_nse(self):
        """Test NSE (Neutron Spin Echo) execution."""
        agent = NeutronAgent()
        fourier_times = [0.1, 1.0, 10.0, 100.0]
        input_data = {
            'technique': 'NSE',
            'parameters': {
                'q_value': 0.05,
                'fourier_times_ns': fourier_times
            }
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'NSE'
        assert 'intermediate_scattering_function' in result.data
        assert 'dynamics_analysis' in result.data

        # Check I(q,t)
        iqt = result.data['intermediate_scattering_function']
        assert len(iqt) == 4  # 4 Fourier times

        # Check dynamics
        dynamics = result.data['dynamics_analysis']
        assert 'relaxation_time_ns' in dynamics
        assert dynamics['relaxation_time_ns'] > 0

        # Check physical interpretation
        phys = result.data['physical_interpretation']
        assert 'diffusion_coefficient_cm2_s' in phys

    def test_execute_qens(self):
        """Test QENS (Quasi-Elastic Neutron Scattering) execution."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'QENS',
            'parameters': {
                'q_range': [0.3, 2.0],
                'energy_range_mev': [-2.0, 2.0]
            }
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'QENS'
        assert 'jump_diffusion_analysis' in result.data
        assert 'spectral_analysis' in result.data
        assert 'hydrogen_dynamics' in result.data

        # Check jump diffusion
        jump = result.data['jump_diffusion_analysis']
        assert 'diffusion_coefficient_cm2_s' in jump
        assert 'residence_time_ps' in jump
        assert 'jump_length_angstrom' in jump

        # Check hydrogen dynamics
        h_dynamics = result.data['hydrogen_dynamics']
        assert 'diffusion_type' in h_dynamics

    def test_execute_nr(self):
        """Test NR (Neutron Reflectometry) execution."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'NR',
            'parameters': {
                'q_range': [0.01, 0.3]
            }
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'Neutron Reflectometry'
        assert 'layer_structure' in result.data
        assert 'interface_analysis' in result.data

        # Check layers
        layers = result.data['layer_structure']
        assert len(layers) > 0
        for layer in layers:
            assert 'thickness_nm' in layer
            assert 'sld_e_minus_6_angstrom_minus_2' in layer

        # Check interface
        interface = result.data['interface_analysis']
        assert 'total_thickness_nm' in interface
        assert 'interface_roughness_angstrom' in interface

    def test_execute_ins(self):
        """Test INS (Inelastic Neutron Scattering) execution."""
        agent = NeutronAgent()
        input_data = {
            'technique': 'INS',
            'parameters': {
                'energy_range_mev': [0, 500]
            }
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'INS'
        assert 'vibrational_modes' in result.data
        assert 'phonon_analysis' in result.data
        assert 'hydrogen_specific' in result.data

        # Check vibrational modes
        modes = result.data['vibrational_modes']
        assert len(modes) > 0
        for mode in modes:
            assert 'energy_mev' in mode
            assert 'frequency_cm_inv' in mode
            assert 'assignment' in mode

        # Check hydrogen specificity
        h_spec = result.data['hydrogen_specific']
        assert 'h_scattering_dominates' in h_spec

    def test_execute_sans_contrast(self):
        """Test SANS contrast variation execution."""
        agent = NeutronAgent()
        d2o_fracs = [0.0, 0.4, 0.42, 1.0]
        input_data = {
            'technique': 'sans_contrast',
            'd2o_fractions': d2o_fracs,  # Can be at top level or in parameters
            'parameters': {}
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['technique'] == 'SANS Contrast Variation'
        assert 'contrast_series' in result.data
        assert 'match_point_analysis' in result.data

        # Check contrast series
        series = result.data['contrast_series']
        assert len(series) == 4

        # Check match point
        match = result.data['match_point_analysis']
        assert 'd2o_fraction_match' in match
        assert 0 <= match['d2o_fraction_match'] <= 1

    def test_execute_invalid_technique(self):
        """Test execution fails for invalid technique."""
        agent = NeutronAgent()
        input_data = {'technique': 'invalid'}
        result = agent.execute(input_data)
        assert not result.success
        assert len(result.errors) > 0


class TestNeutronAgentProvenance:
    """Test provenance tracking."""

    def test_provenance_created(self):
        """Test that provenance is created."""
        agent = NeutronAgent()
        input_data = {'technique': 'SANS'}
        result = agent.execute(input_data)
        assert result.provenance is not None
        assert result.provenance.agent_name == "NeutronAgent"
        assert result.provenance.agent_version == "1.0.0"

    def test_provenance_parameters(self):
        """Test that parameters are recorded in provenance."""
        agent = NeutronAgent(config={'facility': 'test_facility'})
        input_data = {
            'technique': 'SANS',
            'parameters': {'q_range': [0.001, 0.5]}
        }
        result = agent.execute(input_data)
        assert result.provenance.parameters['technique'] == 'sans'
        assert result.provenance.parameters['facility'] == 'test_facility'

    def test_provenance_input_hash(self):
        """Test that input hash is unique."""
        agent = NeutronAgent()
        input_data1 = {'technique': 'SANS', 'h_d_fraction': 0.0}
        input_data2 = {'technique': 'SANS', 'h_d_fraction': 1.0}

        result1 = agent.execute(input_data1)
        result2 = agent.execute(input_data2)

        assert result1.provenance.input_hash != result2.provenance.input_hash


class TestNeutronAgentCaching:
    """Test result caching."""

    def test_caching_works(self):
        """Test that results are cached."""
        # Create a fresh agent instance for this test
        import uuid
        agent = NeutronAgent()
        agent.clear_cache()  # Start with clean cache
        # Use unique UUID to avoid cross-test contamination
        input_data = {'technique': 'SANS', 'cache_test': str(uuid.uuid4())}

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
        agent = NeutronAgent()

        result1 = agent.execute_with_caching({'technique': 'SANS'})
        result2 = agent.execute_with_caching({'technique': 'NSE'})

        assert result1.status == AgentStatus.SUCCESS
        assert result2.status == AgentStatus.SUCCESS  # Not cached

    def test_clear_cache(self):
        """Test cache clearing."""
        agent = NeutronAgent()
        input_data = {'technique': 'SANS'}

        result1 = agent.execute_with_caching(input_data)
        agent.clear_cache()
        result2 = agent.execute_with_caching(input_data)

        assert result1.status == AgentStatus.SUCCESS
        assert result2.status == AgentStatus.SUCCESS  # Not cached


class TestNeutronAgentIntegration:
    """Test integration methods with other agents."""

    def test_validate_with_xray_saxs(self):
        """Test SANS/SAXS cross-validation."""
        sans_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 10.0
            }
        }
        saxs_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 10.5
            }
        }

        validation = NeutronAgent.validate_with_xray_saxs(sans_result, saxs_result)
        assert 'validation_type' in validation
        assert validation['validation_type'] == 'SANS_SAXS_cross_check'
        assert 'rg_agreement_percent' in validation
        assert validation['consistent'] is True

    def test_validate_with_xray_saxs_inconsistent(self):
        """Test SANS/SAXS validation detects inconsistency."""
        sans_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 10.0
            }
        }
        saxs_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 20.0
            }
        }

        validation = NeutronAgent.validate_with_xray_saxs(sans_result, saxs_result)
        assert validation['consistent'] is False

    def test_extract_dynamics_for_simulation(self):
        """Test extracting dynamics parameters for MD."""
        nse_result = {
            'physical_interpretation': {
                'diffusion_coefficient_cm2_s': 5e-8
            },
            'dynamics_analysis': {
                'relaxation_time_ns': 50.0
            },
            'q_value_inv_angstrom': 0.05,
            'experimental_details': {
                'temperature_k': 323.0
            }
        }

        sim_params = NeutronAgent.extract_dynamics_for_simulation(nse_result)
        assert 'dynamics_type' in sim_params
        assert 'diffusion_coefficient_cm2_s' in sim_params
        assert sim_params['diffusion_coefficient_cm2_s'] == 5e-8
        assert 'simulation_suggestions' in sim_params

    def test_design_deuteration_strategy(self):
        """Test deuteration strategy design."""
        sample_composition = {
            'components': [
                {'name': 'polymer_A', 'hydrogen_content': 0.6, 'common': True},
                {'name': 'polymer_B', 'hydrogen_content': 0.3, 'common': False}
            ]
        }

        strategy = NeutronAgent.design_deuteration_strategy(sample_composition)
        assert 'deuteration_strategy' in strategy
        assert len(strategy['deuteration_strategy']) == 2
        assert 'optimal_contrast' in strategy
        assert 'measurement_priority' in strategy


class TestNeutronAgentPhysicalValidation:
    """Test physical validation of results."""

    def test_sans_guinier_regime_valid(self):
        """Test SANS Guinier region is physically valid."""
        agent = NeutronAgent()
        input_data = {'technique': 'SANS'}
        result = agent.execute(input_data)

        guinier = result.data['guinier_analysis']
        rg = guinier['radius_of_gyration_nm']

        # Rg should be positive and reasonable
        assert 0 < rg < 1000

    def test_nse_iqt_decay(self):
        """Test NSE I(q,t) decays properly."""
        agent = NeutronAgent()
        input_data = {'technique': 'NSE'}
        result = agent.execute(input_data)

        iqt = np.array(result.data['intermediate_scattering_function'])

        # I(q,t) should decay with time
        assert iqt[0] >= iqt[-1]
        # I(q,t) should be between 0 and 1
        assert np.all((iqt >= 0) & (iqt <= 1.01))  # Small tolerance

    def test_qens_diffusion_coefficient_valid(self):
        """Test QENS diffusion coefficient is reasonable."""
        agent = NeutronAgent()
        input_data = {'technique': 'QENS'}
        result = agent.execute(input_data)

        D = result.data['jump_diffusion_analysis']['diffusion_coefficient_cm2_s']
        # Typical for hydrogen in polymers: 1e-7 to 1e-4 cm²/s
        assert 1e-10 < D < 1e-2

    def test_qens_jump_length_valid(self):
        """Test QENS jump length is physically reasonable."""
        agent = NeutronAgent()
        input_data = {'technique': 'QENS'}
        result = agent.execute(input_data)

        jump = result.data['jump_diffusion_analysis']['jump_length_angstrom']
        # Typical jump lengths: 2-10 Å
        assert 1.0 < jump < 20.0

    def test_nr_layer_thickness_valid(self):
        """Test NR layer thicknesses are reasonable."""
        agent = NeutronAgent()
        input_data = {'technique': 'NR'}
        result = agent.execute(input_data)

        for layer in result.data['layer_structure']:
            thickness = layer['thickness_nm']
            # Typical layer thicknesses: 1-1000 nm
            assert 0.1 < thickness < 10000.0

    def test_ins_vibrational_energies_valid(self):
        """Test INS vibrational energies are reasonable."""
        agent = NeutronAgent()
        input_data = {'technique': 'INS'}
        result = agent.execute(input_data)

        for mode in result.data['vibrational_modes']:
            energy = mode['energy_mev']
            # Typical molecular vibrations: 10-500 meV
            assert 1.0 < energy < 1000.0

    def test_contrast_series_consistency(self):
        """Test contrast variation series is self-consistent."""
        agent = NeutronAgent()
        input_data = {'technique': 'sans_contrast'}
        result = agent.execute(input_data)

        series = result.data['contrast_series']
        for point in series:
            assert 0 <= point['d2o_fraction'] <= 1
            assert 'contrast_squared' in point
            assert point['contrast_squared'] >= 0

    def test_sans_contrast_factor_varies_with_deuteration(self):
        """Test SANS contrast increases with deuteration."""
        agent = NeutronAgent()

        result_h = agent.execute({
            'technique': 'SANS',
            'h_d_fraction': 0.0
        })
        result_d = agent.execute({
            'technique': 'SANS',
            'h_d_fraction': 1.0
        })

        i0_h = result_h.data['guinier_analysis']['forward_scattering_i0']
        i0_d = result_d.data['guinier_analysis']['forward_scattering_i0']

        # Deuteration should increase scattering
        assert i0_d > i0_h


class TestNeutronAgentCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self):
        """Test that capabilities are correctly reported."""
        agent = NeutronAgent()
        capabilities = agent.get_capabilities()
        assert len(capabilities) == 6

        # Check that all techniques are covered
        technique_names = [cap.name for cap in capabilities]
        assert "SANS Analysis" in technique_names
        assert "NSE Analysis" in technique_names
        assert "QENS Analysis" in technique_names
        assert "Neutron Reflectometry" in technique_names
        assert "INS Analysis" in technique_names
        assert "Contrast Variation" in technique_names

    def test_capability_structure(self):
        """Test that capabilities have required fields."""
        agent = NeutronAgent()
        capabilities = agent.get_capabilities()

        for cap in capabilities:
            assert hasattr(cap, 'name')
            assert hasattr(cap, 'description')
            assert hasattr(cap, 'input_types')
            assert hasattr(cap, 'output_types')
            assert hasattr(cap, 'typical_use_cases')


class TestNeutronAgentInstrumentConnection:
    """Test instrument connection methods."""

    def test_connect_instrument(self):
        """Test instrument connection."""
        agent = NeutronAgent()
        connected = agent.connect_instrument()
        assert connected is True

    def test_process_experimental_data(self):
        """Test experimental data processing."""
        agent = NeutronAgent()
        raw_data = {'counts': [100, 200, 300], 'monitor': 1e6}
        processed = agent.process_experimental_data(raw_data)
        # Placeholder returns same data
        assert processed == raw_data


class TestNeutronAgentHydrogenSensitivity:
    """Test hydrogen-specific capabilities."""

    def test_sans_h_d_contrast(self):
        """Test SANS provides H/D contrast information."""
        agent = NeutronAgent()
        input_data = {'technique': 'SANS', 'h_d_fraction': 0.7}
        result = agent.execute(input_data)

        assert 'contrast_analysis' in result.data
        contrast = result.data['contrast_analysis']
        assert abs(contrast['h_fraction'] - 0.3) < 0.01  # Allow for float precision
        assert abs(contrast['d_fraction'] - 0.7) < 0.01

    def test_qens_hydrogen_dynamics(self):
        """Test QENS provides hydrogen-specific dynamics."""
        agent = NeutronAgent()
        input_data = {'technique': 'QENS'}
        result = agent.execute(input_data)

        assert 'hydrogen_dynamics' in result.data
        h_dyn = result.data['hydrogen_dynamics']
        assert 'diffusion_type' in h_dyn

    def test_ins_hydrogen_dominance(self):
        """Test INS highlights hydrogen scattering."""
        agent = NeutronAgent()
        input_data = {'technique': 'INS'}
        result = agent.execute(input_data)

        assert 'hydrogen_specific' in result.data
        h_spec = result.data['hydrogen_specific']
        assert h_spec.get('h_scattering_dominates', False) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])