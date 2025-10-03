"""Tests for Light Scattering Agent.

Run with: pytest tests/test_light_scattering_agent.py
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from light_scattering_agent import LightScatteringAgent
from base_agent import AgentStatus, ExecutionEnvironment


class TestLightScatteringAgent:
    """Test suite for LightScatteringAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for tests."""
        config = {
            'instrument': {'mode': 'simulated', 'model': 'test_instrument'}
        }
        return LightScatteringAgent(config)

    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.metadata.name == "LightScatteringAgent"
        assert agent.VERSION == "1.0.0"
        assert len(agent.supported_techniques) == 5

    def test_capabilities(self, agent):
        """Test agent reports correct capabilities."""
        capabilities = agent.get_capabilities()
        assert len(capabilities) == 5
        cap_names = [c.name for c in capabilities]
        assert 'DLS' in cap_names
        assert 'SLS' in cap_names
        assert 'Raman' in cap_names

    def test_metadata(self, agent):
        """Test agent metadata."""
        metadata = agent.get_metadata()
        assert metadata.name == "LightScatteringAgent"
        assert metadata.version == "1.0.0"
        assert 'numpy' in metadata.dependencies

    # Input validation tests

    def test_validate_input_valid_dls(self, agent):
        """Test validation accepts valid DLS input."""
        input_data = {
            'technique': 'DLS',
            'sample_file': 'test_sample.dat',
            'parameters': {'temperature': 298, 'angle': 90}
        }
        result = agent.validate_input(input_data)
        assert result.valid
        assert len(result.errors) == 0

    def test_validate_input_missing_technique(self, agent):
        """Test validation catches missing technique."""
        input_data = {
            'sample_file': 'test.dat'
        }
        result = agent.validate_input(input_data)
        assert not result.valid
        assert any('technique' in err.lower() for err in result.errors)

    def test_validate_input_unsupported_technique(self, agent):
        """Test validation catches unsupported technique."""
        input_data = {
            'technique': 'INVALID_TECHNIQUE',
            'sample_file': 'test.dat'
        }
        result = agent.validate_input(input_data)
        assert not result.valid
        assert any('unsupported' in err.lower() for err in result.errors)

    def test_validate_input_missing_data_source(self, agent):
        """Test validation catches missing data source."""
        input_data = {
            'technique': 'DLS'
        }
        result = agent.validate_input(input_data)
        assert not result.valid
        assert any('sample' in err.lower() for err in result.errors)

    def test_validate_input_temperature_warning(self, agent):
        """Test validation warns about unusual temperature."""
        input_data = {
            'technique': 'DLS',
            'sample_file': 'test.dat',
            'parameters': {'temperature': 400}  # Unusual
        }
        result = agent.validate_input(input_data)
        assert result.valid  # Still valid, just warning
        assert len(result.warnings) > 0
        assert any('temperature' in warn.lower() for warn in result.warnings)

    # Resource estimation tests

    def test_estimate_resources_dls(self, agent):
        """Test resource estimation for DLS."""
        input_data = {'technique': 'DLS', 'sample_file': 'test.dat'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 1
        assert resources.memory_gb == 0.5
        assert resources.gpu_count == 0
        assert resources.estimated_time_sec == 60
        assert resources.execution_environment == ExecutionEnvironment.LOCAL

    def test_estimate_resources_raman(self, agent):
        """Test resource estimation for Raman."""
        input_data = {'technique': 'Raman', 'sample_file': 'test.dat'}
        resources = agent.estimate_resources(input_data)
        assert resources.cpu_cores == 2
        assert resources.memory_gb == 1.0
        assert resources.estimated_time_sec == 180

    def test_estimate_resources_multispeckle(self, agent):
        """Test resource estimation for multi-speckle (fast)."""
        input_data = {'technique': 'multi-speckle', 'sample_file': 'test.dat'}
        resources = agent.estimate_resources(input_data)
        assert resources.estimated_time_sec == 30  # Faster than DLS

    # Execution tests

    def test_execute_dls_success(self, agent):
        """Test successful DLS execution."""
        input_data = {
            'technique': 'DLS',
            'sample_file': 'test_polymer.dat',
            'parameters': {'temperature': 298, 'angle': 90}
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.status == AgentStatus.SUCCESS
        assert 'size_distribution' in result.data
        assert 'hydrodynamic_radius_nm' in result.data
        assert result.data['technique'] == 'DLS'
        assert result.provenance is not None

    def test_execute_sls_success(self, agent):
        """Test successful SLS execution."""
        input_data = {
            'technique': 'SLS',
            'sample_file': 'test_polymer.dat',
            'parameters': {}
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'molecular_weight_dalton' in result.data
        assert 'radius_of_gyration_nm' in result.data

    def test_execute_raman_success(self, agent):
        """Test successful Raman execution."""
        input_data = {
            'technique': 'Raman',
            'sample_file': 'test_sample.dat',
            'parameters': {'laser_wavelength_nm': 532}
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'identified_peaks' in result.data
        assert 'molecular_identification' in result.data

    def test_execute_3d_dls_success(self, agent):
        """Test successful 3D-DLS execution."""
        input_data = {
            'technique': '3D-DLS',
            'sample_file': 'turbid_sample.dat',
            'parameters': {}
        }
        result = agent.execute(input_data)
        assert result.success
        assert result.data['multiple_scattering_suppression'] is True

    def test_execute_multispeckle_success(self, agent):
        """Test successful multi-speckle execution."""
        input_data = {
            'technique': 'multi-speckle',
            'sample_file': 'aggregation.dat',
            'parameters': {}
        }
        result = agent.execute(input_data)
        assert result.success
        assert 'time_resolved_data' in result.data
        assert 'aggregation_rate_nm_per_s' in result.data['time_resolved_data']

    def test_execute_invalid_input(self, agent):
        """Test execution fails with invalid input."""
        input_data = {
            'technique': 'INVALID'
        }
        result = agent.execute(input_data)
        assert not result.success
        assert result.status == AgentStatus.FAILED
        assert len(result.errors) > 0

    def test_execute_unsupported_technique(self, agent):
        """Test execution fails with unsupported technique."""
        input_data = {
            'technique': 'UNSUPPORTED',
            'sample_file': 'test.dat'
        }
        result = agent.execute(input_data)
        assert not result.success
        assert any('unsupported' in err.lower() for err in result.errors)

    # Caching tests

    def test_caching_works(self, agent):
        """Test result caching."""
        input_data = {
            'technique': 'DLS',
            'sample_file': 'test.dat',
            'parameters': {'temperature': 298}
        }

        # First execution
        result1 = agent.execute_with_caching(input_data)
        assert result1.success
        assert result1.status == AgentStatus.SUCCESS

        # Second execution (should be cached)
        result2 = agent.execute_with_caching(input_data)
        assert result2.success
        assert result2.status == AgentStatus.CACHED
        assert result2.metadata.get('cached') is True

    def test_cache_key_different_inputs(self, agent):
        """Test different inputs generate different cache keys."""
        input1 = {'technique': 'DLS', 'sample_file': 'test1.dat'}
        input2 = {'technique': 'DLS', 'sample_file': 'test2.dat'}

        key1 = agent._compute_cache_key(input1)
        key2 = agent._compute_cache_key(input2)

        assert key1 != key2

    def test_clear_cache(self, agent):
        """Test cache clearing."""
        input_data = {
            'technique': 'DLS',
            'sample_file': 'test.dat'
        }

        # Execute and cache
        agent.execute_with_caching(input_data)
        assert len(agent._cache) > 0

        # Clear cache
        agent.clear_cache()
        assert len(agent._cache) == 0

    # Integration tests

    def test_validate_with_sans_saxs_sphere(self, agent):
        """Test validation with SANS/SAXS for spherical particles."""
        dls_result = {'hydrodynamic_radius_nm': 50}
        sans_result = {'radius_of_gyration_nm': 38.75}  # Rg/Rh = 0.775 (sphere)

        validation = agent.validate_with_sans_saxs(dls_result, sans_result)

        assert validation['consistent']
        assert validation['inferred_shape'] == 'sphere'
        assert abs(validation['shape_factor_Rg_Rh'] - 0.775) < 0.05

    def test_validate_with_sans_saxs_random_coil(self, agent):
        """Test validation with SANS/SAXS for random coil."""
        dls_result = {'hydrodynamic_radius_nm': 50}
        sans_result = {'radius_of_gyration_nm': 75}  # Rg/Rh = 1.5 (random coil)

        validation = agent.validate_with_sans_saxs(dls_result, sans_result)

        assert validation['consistent']
        assert validation['inferred_shape'] == 'random_coil'

    def test_compare_with_md_good_agreement(self, agent):
        """Test comparison with MD simulation (good agreement)."""
        dls_result = {'size_distribution': {'mean_diameter_nm': 100}}
        md_result = {'predicted_size_nm': 105}  # 5% difference

        comparison = agent.compare_with_md_simulation(dls_result, md_result)

        assert comparison['agreement'] == 'good'
        assert comparison['percent_difference'] == 5.0

    def test_compare_with_md_poor_agreement(self, agent):
        """Test comparison with MD simulation (poor agreement)."""
        dls_result = {'size_distribution': {'mean_diameter_nm': 100}}
        md_result = {'predicted_size_nm': 150}  # 50% difference

        comparison = agent.compare_with_md_simulation(dls_result, md_result)

        assert comparison['agreement'] == 'poor'
        assert comparison['percent_difference'] == 50.0

    def test_compare_with_md_missing_data(self, agent):
        """Test comparison with MD when data missing."""
        dls_result = {'size_distribution': {'mean_diameter_nm': 100}}
        md_result = {}  # No size data

        comparison = agent.compare_with_md_simulation(dls_result, md_result)

        assert comparison['agreement'] == 'unknown'
        assert 'error' in comparison

    # Provenance tests

    def test_provenance_recorded(self, agent):
        """Test execution records provenance."""
        input_data = {
            'technique': 'DLS',
            'sample_file': 'test.dat',
            'parameters': {'temperature': 298}
        }
        result = agent.execute(input_data)

        assert result.provenance is not None
        assert result.provenance.agent_name == "LightScatteringAgent"
        assert result.provenance.agent_version == agent.VERSION
        assert result.provenance.execution_time_sec > 0
        assert 'temperature' in result.provenance.environment

    def test_result_serialization(self, agent):
        """Test AgentResult can be serialized to dict."""
        input_data = {
            'technique': 'DLS',
            'sample_file': 'test.dat'
        }
        result = agent.execute(input_data)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict['agent_name'] == "LightScatteringAgent"
        assert result_dict['status'] == 'success'
        assert 'provenance' in result_dict


# Fixtures for integration testing

@pytest.fixture
def sample_dls_data():
    """Sample DLS data for testing."""
    return {
        'technique': 'DLS',
        'sample_file': 'polymer_nanoparticles.dat',
        'parameters': {
            'temperature': 298,
            'viscosity': 0.89e-3,
            'angle': 90
        }
    }


@pytest.fixture
def sample_sans_data():
    """Sample SANS data for cross-validation."""
    return {
        'radius_of_gyration_nm': 35.5,
        'structure_factor': [1.0, 0.95, 0.88],  # S(q) at different q
        'q_nm-1': [0.1, 0.2, 0.3]
    }


def test_integration_workflow(sample_dls_data, sample_sans_data):
    """Test integrated workflow: DLS → SANS validation."""
    agent = LightScatteringAgent()

    # Execute DLS
    dls_result = agent.execute(sample_dls_data)
    assert dls_result.success

    # Validate with SANS
    validation = agent.validate_with_sans_saxs(dls_result.data, sample_sans_data)
    assert validation['consistent']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])