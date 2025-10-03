"""
Comprehensive tests for CrystallographyAgent.

Test categories:
1. Initialization
2. Input validation
3. Resource estimation
4. Execution (all 6 techniques)
5. Integration methods
6. Provenance tracking
7. Physical validation
"""

import pytest
import numpy as np
import uuid
from crystallography_agent import CrystallographyAgent


class TestCrystallographyAgentBasics:
    """Test basic agent functionality."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = CrystallographyAgent()
        assert agent.VERSION == "1.0.0"
        assert agent.metadata.name == "CrystallographyAgent"

    def test_supported_techniques(self):
        """Test that all crystallography techniques are supported."""
        agent = CrystallographyAgent()
        techniques = agent.SUPPORTED_TECHNIQUES
        expected = ['xrd_powder', 'xrd_single_crystal', 'pdf', 'rietveld', 'texture', 'phase_id']
        assert set(techniques) == set(expected)

    def test_configuration(self):
        """Test agent configuration."""
        config = {'wavelength': 0.71, 'beamline': 'APS_11-ID-B'}
        agent = CrystallographyAgent(config=config)
        assert agent.wavelength_angstrom == 0.71
        assert agent.beamline == 'APS_11-ID-B'


class TestCrystallographyAgentValidation:
    """Test input validation."""

    def test_validation_missing_technique(self):
        """Test validation with missing technique."""
        agent = CrystallographyAgent()
        result = agent.validate_input({})
        assert not result.valid
        assert len(result.errors) > 0

    def test_validation_invalid_technique(self):
        """Test validation with invalid technique."""
        agent = CrystallographyAgent()
        result = agent.validate_input({'technique': 'invalid_xrd'})
        assert not result.valid
        assert 'unsupported' in result.errors[0].lower()

    def test_validation_xrd_powder_valid(self):
        """Test validation for valid powder XRD input."""
        agent = CrystallographyAgent()
        input_data = {
            'technique': 'XRD_POWDER',
            'two_theta_range': [10, 90]
        }
        result = agent.validate_input(input_data)
        assert result.valid

    def test_validation_rietveld_warning(self):
        """Test validation warning for Rietveld without initial structure."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'RIETVELD'}
        result = agent.validate_input(input_data)
        assert result.valid
        assert len(result.warnings) > 0


class TestCrystallographyAgentResourceEstimation:
    """Test resource requirement estimation."""

    def test_estimate_powder_xrd_resources(self):
        """Test resource estimation for powder XRD."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'XRD_POWDER'}
        resources = agent.estimate_resources(input_data)
        assert resources.estimated_time_sec > 0
        assert resources.memory_gb > 0

    def test_estimate_single_crystal_resources(self):
        """Test resource estimation for single crystal (more intensive)."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'XRD_SINGLE_CRYSTAL'}
        resources = agent.estimate_resources(input_data)
        # Single crystal should require more resources
        assert resources.estimated_time_sec > 100
        assert resources.cpu_cores >= 4

    def test_estimate_rietveld_resources(self):
        """Test resource estimation for Rietveld refinement."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'RIETVELD', 'n_phases': 3}
        resources = agent.estimate_resources(input_data)
        assert resources.estimated_time_sec > 0


class TestCrystallographyAgentExecution:
    """Test execution of all crystallography techniques."""

    def test_execute_xrd_powder(self):
        """Test powder XRD execution."""
        agent = CrystallographyAgent()
        input_data = {
            'technique': 'XRD_POWDER',
            'two_theta_range': [10, 90]
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'two_theta_deg' in result.data
        assert 'intensity_counts' in result.data
        assert 'peak_analysis' in result.data
        assert 'lattice_parameters' in result.data
        assert result.data['lattice_parameters']['system'] == 'cubic'

    def test_execute_xrd_single_crystal(self):
        """Test single crystal XRD execution."""
        agent = CrystallographyAgent()
        input_data = {
            'technique': 'XRD_SINGLE_CRYSTAL'
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'lattice_parameters' in result.data
        assert 'atomic_structure' in result.data
        assert 'refinement_statistics' in result.data
        assert result.data['refinement_statistics']['r_factor'] < 0.1

    def test_execute_pdf(self):
        """Test PDF execution."""
        agent = CrystallographyAgent()
        input_data = {
            'technique': 'PDF',
            'r_range': [0, 30]
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'r_angstrom' in result.data
        assert 'g_r' in result.data
        assert 'peak_analysis' in result.data
        assert 'local_structure' in result.data

    def test_execute_rietveld(self):
        """Test Rietveld refinement execution."""
        agent = CrystallographyAgent()
        input_data = {
            'technique': 'RIETVELD'
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'phases' in result.data
        assert 'refinement_quality' in result.data
        assert len(result.data['phases']) >= 1
        # Check that weight fractions sum to ~1
        total_fraction = sum(p['weight_fraction'] for p in result.data['phases'])
        assert abs(total_fraction - 1.0) < 0.01

    def test_execute_texture(self):
        """Test texture analysis execution."""
        agent = CrystallographyAgent()
        input_data = {
            'technique': 'TEXTURE'
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'texture_index' in result.data
        assert 'preferred_orientation' in result.data
        assert 'pole_figures' in result.data

    def test_execute_phase_id(self):
        """Test phase identification execution."""
        agent = CrystallographyAgent()
        input_data = {
            'technique': 'PHASE_ID'
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'identified_phases' in result.data
        assert len(result.data['identified_phases']) >= 1
        assert all(0 <= p['match_quality'] <= 1.0 for p in result.data['identified_phases'])


class TestCrystallographyAgentIntegration:
    """Test integration methods with other agents."""

    def test_validate_with_dft(self):
        """Test XRD-DFT cross-validation."""
        # Mock XRD result
        xrd_result = {
            'lattice_parameters': {
                'a_angstrom': 4.05
            }
        }

        # Mock DFT result
        dft_result = {
            'lattice_parameters': {
                'a_angstrom': 4.03
            }
        }

        validation = CrystallographyAgent.validate_with_dft(xrd_result, dft_result)

        assert 'lattice_agreement' in validation
        assert 'error_percent' in validation['lattice_agreement']
        assert validation['lattice_agreement']['error_percent'] < 1.0  # Should be ~0.5%

    def test_extract_structure_for_dft(self):
        """Test structure extraction for DFT input."""
        # Mock XRD result
        xrd_result = {
            'lattice_parameters': {'a_angstrom': 5.43},
            'space_group': 'Fm-3m',
            'atomic_structure': {'atoms': []}
        }

        structure = CrystallographyAgent.extract_structure_for_dft(xrd_result)

        assert 'lattice_parameters' in structure
        assert 'dft_recommendations' in structure
        assert structure['space_group'] == 'Fm-3m'

    def test_correlate_with_scattering(self):
        """Test correlation between XRD and SAXS."""
        # Mock XRD result
        xrd_result = {
            'crystallite_size_nm': 5.0
        }

        # Mock SAXS result
        saxs_result = {
            'guinier_analysis': {
                'radius_of_gyration_nm': 10.0
            }
        }

        correlation = CrystallographyAgent.correlate_with_scattering(xrd_result, saxs_result)

        assert 'length_scales' in correlation
        assert 'structural_model' in correlation
        assert correlation['length_scales']['xrd_crystallite_size_nm'] == 5.0


class TestCrystallographyAgentProvenance:
    """Test provenance and caching."""

    def test_provenance_tracking(self):
        """Test that execution metadata is tracked."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'XRD_POWDER', 'test_id': str(uuid.uuid4())}
        result = agent.execute(input_data)

        assert result.success
        assert result.provenance is not None
        assert result.provenance.agent_version == '1.0.0'

    def test_caching_works(self):
        """Test that results are cached."""
        agent = CrystallographyAgent()
        agent.clear_cache()

        # Use unique ID to avoid cross-test contamination
        test_id = str(uuid.uuid4())
        input_data = {'technique': 'XRD_POWDER', 'cache_test': test_id}

        # First execution
        result1 = agent.execute(input_data)
        assert result1.success

        # Second execution (should be cached)
        result2 = agent.execute(input_data)
        assert result2.success

    def test_cache_clearing(self):
        """Test cache clearing."""
        agent = CrystallographyAgent()
        test_id = str(uuid.uuid4())
        input_data = {'technique': 'XRD_POWDER', 'clear_test': test_id}

        # Execute and cache
        result1 = agent.execute(input_data)
        assert result1.success

        # Clear cache
        agent.clear_cache()

        # Should execute again
        result2 = agent.execute(input_data)
        assert result2.success


class TestCrystallographyAgentPhysicalValidation:
    """Test physical validation of results."""

    def test_xrd_two_theta_range_valid(self):
        """Test that XRD 2θ values are in valid range."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'XRD_POWDER'}
        result = agent.execute(input_data)

        two_theta = np.array(result.data['two_theta_deg'])
        assert np.all(two_theta >= 0)
        assert np.all(two_theta <= 180)

    def test_xrd_intensities_positive(self):
        """Test that XRD intensities are non-negative."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'XRD_POWDER'}
        result = agent.execute(input_data)

        intensity = np.array(result.data['intensity_counts'])
        assert np.all(intensity >= 0)

    def test_lattice_parameters_positive(self):
        """Test that lattice parameters are positive."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'XRD_POWDER'}
        result = agent.execute(input_data)

        lattice = result.data['lattice_parameters']
        assert lattice['a_angstrom'] > 0
        assert lattice['volume_angstrom3'] > 0

    def test_pdf_correlation_decay(self):
        """Test that PDF G(r) shows proper behavior."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'PDF'}
        result = agent.execute(input_data)

        g_r = np.array(result.data['g_r'])
        # PDF should show oscillations
        assert g_r.max() > 0
        assert g_r.min() < 0.5  # Some regions below average

    def test_rietveld_weight_fractions_sum_to_one(self):
        """Test that Rietveld weight fractions sum to 1."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'RIETVELD'}
        result = agent.execute(input_data)

        phases = result.data['phases']
        total_fraction = sum(p['weight_fraction'] for p in phases)
        assert abs(total_fraction - 1.0) < 0.01

    def test_rietveld_r_factors_valid(self):
        """Test that Rietveld R-factors are reasonable."""
        agent = CrystallographyAgent()
        input_data = {'technique': 'RIETVELD'}
        result = agent.execute(input_data)

        quality = result.data['refinement_quality']
        # R-factors should be between 0 and 1
        assert 0 < quality['r_profile'] < 1
        assert 0 < quality['r_weighted'] < 1
        assert quality['goodness_of_fit'] > 0


class TestCrystallographyAgentCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self):
        """Test capability listing."""
        agent = CrystallographyAgent()
        capabilities = agent.get_capabilities()

        assert len(capabilities) == 6
        # Check that all techniques are covered
        cap_names = [c.name for c in capabilities]
        assert 'xrd_powder' in cap_names
        assert 'rietveld' in cap_names

    def test_capabilities_have_descriptions(self):
        """Test that all capabilities have descriptions."""
        agent = CrystallographyAgent()
        capabilities = agent.get_capabilities()

        for cap in capabilities:
            assert len(cap.description) > 0
            assert len(cap.input_types) > 0
            assert len(cap.output_types) > 0


class TestCrystallographyAgentHPCIntegration:
    """Test HPC job submission (ComputationalAgent interface)."""

    def test_submit_job(self):
        """Test job submission."""
        agent = CrystallographyAgent()
        job_id = agent.submit_job({'technique': 'RIETVELD'})
        assert job_id.startswith('xrd_job_')

    def test_check_job_status(self):
        """Test job status checking."""
        agent = CrystallographyAgent()
        job_id = "xrd_job_12345"
        status = agent.check_job_status(job_id)
        assert status in ['queued', 'running', 'completed', 'failed']

    def test_retrieve_results(self):
        """Test result retrieval."""
        agent = CrystallographyAgent()
        job_id = "xrd_job_12345"
        results = agent.retrieve_results(job_id)
        assert 'status' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])