"""
Tests for SurfaceScienceAgent.

Test categories:
1. Initialization
2. Input validation
3. QCM-D execution
4. SPR execution
5. Contact angle execution
6. Adsorption isotherm execution
"""

import pytest
import numpy as np
import uuid
from surface_science_agent import SurfaceScienceAgent


class TestSurfaceScienceBasics:
    """Test basic agent functionality."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = SurfaceScienceAgent()
        assert agent.VERSION == "1.0.0"
        assert agent.metadata.name == "SurfaceScienceAgent"

    def test_supported_techniques(self):
        """Test that all expected techniques are supported."""
        agent = SurfaceScienceAgent()
        techniques = agent.SUPPORTED_TECHNIQUES
        assert 'qcm_d' in techniques
        assert 'spr' in techniques
        assert 'contact_angle' in techniques
        assert len(techniques) == 6


class TestSurfaceScienceValidation:
    """Test input validation."""

    def test_validation_missing_technique(self):
        """Test validation with missing technique."""
        agent = SurfaceScienceAgent()
        result = agent.validate_input({})
        assert not result.valid

    def test_validation_invalid_technique(self):
        """Test validation with invalid technique."""
        agent = SurfaceScienceAgent()
        result = agent.validate_input({'technique': 'invalid_surface'})
        assert not result.valid

    def test_validation_qcmd_valid(self):
        """Test validation for valid QCM-D input."""
        agent = SurfaceScienceAgent()
        input_data = {
            'technique': 'QCM_D',
            'harmonics': [3, 5, 7]
        }
        result = agent.validate_input(input_data)
        assert result.valid


class TestSurfaceScienceQCMD:
    """Test QCM-D functionality."""

    def test_execute_qcmd(self):
        """Test QCM-D execution."""
        agent = SurfaceScienceAgent()
        input_data = {
            'technique': 'QCM_D',
            'time_range': [0, 1800]
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'frequency_shift_hz' in result.data
        assert 'dissipation_shift' in result.data
        assert 'mass_analysis' in result.data

    def test_qcmd_mass_positive(self):
        """Test that QCM-D reports positive mass."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'QCM_D'}
        result = agent.execute(input_data)

        final_mass = result.data['mass_analysis']['final_mass_ng_cm2']
        # Adsorption should give positive mass
        assert final_mass > 0

    def test_qcmd_has_kinetics(self):
        """Test that QCM-D includes kinetic analysis."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'QCM_D'}
        result = agent.execute(input_data)

        assert 'kinetics' in result.data
        assert 'adsorption_rate_constant_s' in result.data['kinetics']


class TestSurfaceScienceSPR:
    """Test SPR functionality."""

    def test_execute_spr(self):
        """Test SPR execution."""
        agent = SurfaceScienceAgent()
        input_data = {
            'technique': 'SPR',
            'concentrations': [10, 50, 100]
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'sensorgrams_RU' in result.data
        assert 'kinetic_analysis' in result.data
        assert len(result.data['sensorgrams_RU']) == 3

    def test_spr_kinetic_parameters(self):
        """Test that SPR returns kinetic parameters."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'SPR'}
        result = agent.execute(input_data)

        kinetics = result.data['kinetic_analysis']
        assert 'kon_M_inv_s' in kinetics
        assert 'koff_s_inv' in kinetics
        assert 'KD_nM' in kinetics

        # KD should equal koff/kon (converted to nM)
        expected_KD = kinetics['koff_s_inv'] / kinetics['kon_M_inv_s'] * 1e9
        assert abs(kinetics['KD_nM'] - expected_KD) < 1.0

    def test_spr_affinity_classification(self):
        """Test that SPR classifies affinity."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'SPR'}
        result = agent.execute(input_data)

        assert 'affinity_classification' in result.data
        assert isinstance(result.data['affinity_classification'], str)


class TestSurfaceScienceContactAngle:
    """Test contact angle functionality."""

    def test_execute_contact_angle(self):
        """Test contact angle execution."""
        agent = SurfaceScienceAgent()
        input_data = {
            'technique': 'CONTACT_ANGLE',
            'n_measurements': 5
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'static_measurements' in result.data
        assert 'dynamic_measurements' in result.data
        assert 'wettability' in result.data

    def test_contact_angle_range_valid(self):
        """Test that contact angles are in valid range."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'CONTACT_ANGLE'}
        result = agent.execute(input_data)

        theta_mean = result.data['static_measurements']['theta_mean_deg']
        # Contact angles are 0-180 degrees
        assert 0 <= theta_mean <= 180

    def test_contact_angle_hysteresis(self):
        """Test that hysteresis is calculated."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'CONTACT_ANGLE'}
        result = agent.execute(input_data)

        dynamic = result.data['dynamic_measurements']
        assert 'hysteresis_deg' in dynamic
        # Hysteresis should be positive
        assert dynamic['hysteresis_deg'] >= 0


class TestSurfaceScienceAdsorption:
    """Test adsorption isotherm functionality."""

    def test_execute_adsorption(self):
        """Test adsorption isotherm execution."""
        agent = SurfaceScienceAgent()
        input_data = {
            'technique': 'ADSORPTION_ISOTHERM',
            'concentrations': [1, 5, 10, 50, 100]
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'adsorbed_amount_mg_g' in result.data
        assert 'isotherm_model' in result.data
        assert 'model_parameters' in result.data

    def test_adsorption_amounts_positive(self):
        """Test that adsorbed amounts are non-negative."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'ADSORPTION_ISOTHERM'}
        result = agent.execute(input_data)

        amounts = result.data['adsorbed_amount_mg_g']
        assert all(a >= 0 for a in amounts)

    def test_adsorption_model_parameters(self):
        """Test that model parameters are provided."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'ADSORPTION_ISOTHERM'}
        result = agent.execute(input_data)

        params = result.data['model_parameters']
        assert 'q_max_mg_g' in params
        assert 'K_L_L_mg' in params
        assert params['q_max_mg_g'] > 0


class TestSurfaceScienceResourceEstimation:
    """Test resource estimation."""

    def test_estimate_resources(self):
        """Test resource estimation."""
        agent = SurfaceScienceAgent()
        input_data = {'technique': 'QCM_D'}
        resources = agent.estimate_resources(input_data)

        assert resources.estimated_time_sec > 0
        assert resources.memory_gb > 0


class TestSurfaceScienceCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self):
        """Test capability listing."""
        agent = SurfaceScienceAgent()
        capabilities = agent.get_capabilities()

        assert len(capabilities) >= 4
        cap_names = [c.name for c in capabilities]
        assert 'qcm_d' in cap_names
        assert 'spr' in cap_names

    def test_capabilities_have_use_cases(self):
        """Test that capabilities have use cases."""
        agent = SurfaceScienceAgent()
        capabilities = agent.get_capabilities()

        for cap in capabilities:
            assert len(cap.typical_use_cases) > 0


class TestSurfaceScienceProvenance:
    """Test provenance tracking."""

    def test_provenance_tracking(self):
        """Test that execution metadata is tracked."""
        agent = SurfaceScienceAgent()
        input_data = {
            'technique': 'SPR',
            'test_id': str(uuid.uuid4())
        }
        result = agent.execute(input_data)

        assert result.success
        assert result.provenance is not None
        assert result.provenance.agent_version == '1.0.0'


class TestSurfaceScienceInstrumentConnection:
    """Test instrument connection."""

    def test_connect_instrument(self):
        """Test instrument connection."""
        agent = SurfaceScienceAgent()
        success = agent.connect_instrument()
        assert success is True

    def test_process_experimental_data(self):
        """Test experimental data processing."""
        agent = SurfaceScienceAgent()
        raw_data = {'measurement': [1, 2, 3]}
        processed = agent.process_experimental_data(raw_data)
        assert processed is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])