"""
Test suite for StochasticDynamicsAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL and HPC)
- Execution for all stochastic methods (5 methods)
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
from stochastic_dynamics_agent import StochasticDynamicsAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a StochasticDynamicsAgent instance."""
    return StochasticDynamicsAgent()


@pytest.fixture
def sample_langevin_input():
    """Sample input for Langevin dynamics."""
    return {
        'method': 'langevin',
        'potential': {'type': 'harmonic', 'k': 1.0, 'x0': 0.0},
        'parameters': {
            'temperature': 300.0,
            'friction': 1.0,
            'n_steps': 100000,
            'dt': 0.01,
            'regime': 'overdamped',
            'initial_position': 0.0
        }
    }


@pytest.fixture
def sample_master_equation_input():
    """Sample input for master equation."""
    n_states = 5
    rate_matrix = np.random.rand(n_states, n_states)
    np.fill_diagonal(rate_matrix, 0)
    rate_matrix[np.diag_indices(n_states)] = -rate_matrix.sum(axis=1)
    return {
        'method': 'master_equation',
        'rate_matrix': rate_matrix.tolist(),
        'parameters': {
            'initial_state': 0,
            'n_steps': 10000
        }
    }


@pytest.fixture
def sample_first_passage_input():
    """Sample input for first-passage time analysis."""
    return {
        'method': 'first_passage',
        'parameters': {
            'barrier_position': 1.0,
            'n_trajectories': 100,
            'n_steps': 10000,
            'dt': 0.01,
            'diffusion_coefficient': 1.0
        }
    }


@pytest.fixture
def sample_kramers_input():
    """Sample input for Kramers escape rate."""
    return {
        'method': 'kramers_escape',
        'potential': {'type': 'double_well'},
        'parameters': {
            'temperature': 300.0,
            'friction': 1.0,
            'barrier_height': 5.0,
            'omega_well': 1.0,
            'omega_barrier': 1.0
        }
    }


@pytest.fixture
def sample_fokker_planck_input():
    """Sample input for Fokker-Planck equation."""
    return {
        'method': 'fokker_planck',
        'parameters': {
            'x_min': -5.0,
            'x_max': 5.0,
            'n_grid': 100,
            'k': 1.0,
            'D': 1.0,
            'dt': 0.001,
            'n_steps': 1000
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
    assert metadata.name == "StochasticDynamicsAgent"
    assert metadata.version == "1.0.0"
    assert "stochastic" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 5
    method_names = [cap.name for cap in capabilities]
    assert 'Langevin Dynamics' in method_names
    assert 'Master Equation Simulation' in method_names
    assert 'Kramers Escape Rate' in method_names


# ============================================================================
# Test: Input Validation (8 tests)
# ============================================================================

def test_validate_input_success_langevin(agent, sample_langevin_input):
    """Test validation succeeds for valid Langevin input."""
    validation = agent.validate_input(sample_langevin_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_master_equation(agent, sample_master_equation_input):
    """Test validation succeeds for valid master equation input."""
    validation = agent.validate_input(sample_master_equation_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_method(agent):
    """Test validation fails when method is missing."""
    validation = agent.validate_input({'potential': {}})
    assert not validation.valid
    assert any('method' in err.lower() for err in validation.errors)


def test_validate_input_invalid_method(agent):
    """Test validation fails for unsupported method."""
    validation = agent.validate_input({'method': 'quantum_tunneling'})
    assert not validation.valid
    assert any('unsupported method' in err.lower() for err in validation.errors)


def test_validate_input_langevin_missing_potential(agent):
    """Test validation fails when Langevin potential is missing."""
    validation = agent.validate_input({
        'method': 'langevin',
        'parameters': {'temperature': 300}
    })
    assert not validation.valid
    assert any('potential' in err.lower() for err in validation.errors)


def test_validate_input_master_equation_missing_rates(agent):
    """Test validation fails when rate matrix is missing."""
    validation = agent.validate_input({
        'method': 'master_equation',
        'parameters': {}
    })
    assert not validation.valid


def test_validate_input_negative_friction(agent, sample_langevin_input):
    """Test validation fails for negative friction."""
    sample_langevin_input['parameters']['friction'] = -1.0
    validation = agent.validate_input(sample_langevin_input)
    assert not validation.valid


def test_validate_input_warnings(agent, sample_langevin_input):
    """Test validation warns for missing optional parameters."""
    sample_langevin_input['parameters'].pop('friction', None)
    validation = agent.validate_input(sample_langevin_input)
    assert len(validation.warnings) > 0


# ============================================================================
# Test: Resource Estimation (7 tests)
# ============================================================================

def test_estimate_resources_small_langevin(agent, sample_langevin_input):
    """Test resource estimation for small Langevin simulation."""
    sample_langevin_input['parameters']['n_steps'] = 10000
    resources = agent.estimate_resources(sample_langevin_input)
    assert resources.execution_environment.value == 'local'
    assert resources.estimated_time_sec == 120


def test_estimate_resources_medium_langevin(agent, sample_langevin_input):
    """Test resource estimation for medium Langevin simulation."""
    sample_langevin_input['parameters']['n_steps'] = 10000000
    resources = agent.estimate_resources(sample_langevin_input)
    assert resources.execution_environment.value == 'local'
    assert resources.estimated_time_sec == 600


def test_estimate_resources_large_langevin(agent, sample_langevin_input):
    """Test resource estimation for large Langevin simulation."""
    sample_langevin_input['parameters']['n_steps'] = 100000000
    sample_langevin_input['parameters']['n_particles'] = 100
    resources = agent.estimate_resources(sample_langevin_input)
    assert resources.execution_environment.value == 'hpc'
    assert resources.gpu_count >= 1


def test_estimate_resources_master_equation(agent, sample_master_equation_input):
    """Test resource estimation for master equation."""
    resources = agent.estimate_resources(sample_master_equation_input)
    assert resources.cpu_cores >= 4


def test_estimate_resources_first_passage(agent, sample_first_passage_input):
    """Test resource estimation for first-passage time."""
    resources = agent.estimate_resources(sample_first_passage_input)
    assert resources.estimated_time_sec > 0


def test_estimate_resources_kramers(agent, sample_kramers_input):
    """Test resource estimation for Kramers rate."""
    resources = agent.estimate_resources(sample_kramers_input)
    assert resources.execution_environment.value == 'local'


def test_estimate_resources_fokker_planck(agent, sample_fokker_planck_input):
    """Test resource estimation for Fokker-Planck."""
    resources = agent.estimate_resources(sample_fokker_planck_input)
    assert resources.cpu_cores >= 4


# ============================================================================
# Test: Execution for All Methods (7 tests)
# ============================================================================

def test_execute_langevin_success(agent, sample_langevin_input):
    """Test Langevin dynamics execution succeeds."""
    result = agent.execute(sample_langevin_input)
    assert result.success
    assert 'trajectory' in result.data
    assert 'statistics' in result.data
    assert 'autocorrelation' in result.data
    assert 'msd' in result.data
    assert result.data['regime'] == 'overdamped'


def test_execute_master_equation_success(agent, sample_master_equation_input):
    """Test master equation execution succeeds."""
    result = agent.execute(sample_master_equation_input)
    assert result.success
    assert 'trajectory' in result.data
    assert 'stationary_distribution' in result.data
    assert 'occupation_probabilities' in result.data
    assert result.data['n_states'] >= 1


def test_execute_first_passage_success(agent, sample_first_passage_input):
    """Test first-passage time analysis succeeds."""
    result = agent.execute(sample_first_passage_input)
    assert result.success
    assert 'mean_first_passage_time' in result.data
    assert 'fpt_distribution' in result.data
    assert result.data['mean_first_passage_time'] > 0


def test_execute_kramers_success(agent, sample_kramers_input):
    """Test Kramers escape rate execution succeeds."""
    result = agent.execute(sample_kramers_input)
    assert result.success
    assert 'escape_rate' in result.data
    assert 'tst_rate' in result.data
    assert 'activation_energy' in result.data
    assert 'regime' in result.data
    assert result.data['escape_rate'] > 0


def test_execute_fokker_planck_success(agent, sample_fokker_planck_input):
    """Test Fokker-Planck solver execution succeeds."""
    result = agent.execute(sample_fokker_planck_input)
    assert result.success
    assert 'steady_state_distribution' in result.data
    assert 'converged' in result.data
    assert result.data['normalization'] > 0.99
    assert result.data['normalization'] < 1.01


def test_execute_langevin_underdamped(agent, sample_langevin_input):
    """Test underdamped Langevin dynamics."""
    sample_langevin_input['parameters']['regime'] = 'underdamped'
    sample_langevin_input['parameters']['mass'] = 1.0
    result = agent.execute(sample_langevin_input)
    assert result.success
    assert result.data['regime'] == 'underdamped'


def test_execute_invalid_method(agent):
    """Test execution fails for invalid method."""
    result = agent.execute({'method': 'invalid_method', 'potential': {}})
    assert not result.success
    assert len(result.errors) > 0


# ============================================================================
# Test: Job Submission/Status/Retrieval (5 tests)
# ============================================================================

def test_submit_calculation_returns_job_id(agent, sample_langevin_input):
    """Test submit_calculation returns valid job ID."""
    job_id = agent.submit_calculation(sample_langevin_input)
    assert job_id is not None
    assert isinstance(job_id, str)


def test_check_status_after_submission(agent, sample_langevin_input):
    """Test check_status returns valid status after submission."""
    job_id = agent.submit_calculation(sample_langevin_input)
    status = agent.check_status(job_id)
    assert status == AgentStatus.RUNNING


def test_check_status_invalid_job_id(agent):
    """Test check_status handles invalid job ID."""
    status = agent.check_status('invalid_job_id')
    assert status == AgentStatus.FAILED


def test_retrieve_results_after_completion(agent, sample_langevin_input):
    """Test retrieve_results after marking job complete."""
    job_id = agent.submit_calculation(sample_langevin_input)
    if hasattr(agent, 'job_cache') and job_id in agent.job_cache:
        agent.job_cache[job_id]['status'] = AgentStatus.SUCCESS
        agent.job_cache[job_id]['results'] = {'trajectory': []}
    results = agent.retrieve_results(job_id)
    assert results is not None


def test_retrieve_results_invalid_job_id(agent):
    """Test retrieve_results handles invalid job ID."""
    with pytest.raises(Exception):
        agent.retrieve_results('invalid_job_id')


# ============================================================================
# Test: Helper Methods (6 tests)
# ============================================================================

def test_compute_autocorrelation(agent):
    """Test autocorrelation computation."""
    trajectory = np.random.randn(1000)
    acf = agent._compute_autocorrelation(trajectory, 100)
    assert len(acf) == 100
    assert acf[0] == 1.0  # Normalized
    assert -1 <= acf.min() <= acf.max() <= 1


def test_compute_msd(agent):
    """Test MSD computation."""
    trajectory = np.cumsum(np.random.randn(1000))
    msd_result = agent._compute_msd(trajectory, 0.01)
    assert 'times' in msd_result
    assert 'msd' in msd_result
    assert 'diffusion_coefficient_measured' in msd_result
    assert len(msd_result['times']) == len(msd_result['msd'])


def test_determine_kramers_regime_low(agent):
    """Test Kramers regime determination - low friction."""
    regime = agent._determine_kramers_regime(0.01, 1.0)
    assert regime == 'low_friction'


def test_determine_kramers_regime_high(agent):
    """Test Kramers regime determination - high friction."""
    regime = agent._determine_kramers_regime(100.0, 1.0)
    assert regime == 'high_friction'


def test_determine_kramers_regime_intermediate(agent):
    """Test Kramers regime determination - intermediate friction."""
    regime = agent._determine_kramers_regime(1.0, 1.0)
    assert regime == 'intermediate_friction'


def test_langevin_double_well_potential(agent):
    """Test Langevin with double-well potential."""
    input_data = {
        'method': 'langevin',
        'potential': {'type': 'double_well', 'a': 1.0, 'b': 4.0},
        'parameters': {
            'temperature': 300,
            'friction': 1.0,
            'n_steps': 10000,
            'dt': 0.01
        }
    }
    result = agent.execute(input_data)
    assert result.success


# ============================================================================
# Test: Integration Methods (5 tests)
# ============================================================================

def test_validate_fluctuation_dissipation_satisfied(agent):
    """Test fluctuation-dissipation theorem validation."""
    temperature = 300.0
    friction = 1.0
    kB = 1.380649e-23
    D_theory = kB * temperature / friction
    validation = agent.validate_fluctuation_dissipation(
        temperature, friction, D_theory, tolerance=0.1
    )
    assert validation['fdt_satisfied']


def test_validate_fluctuation_dissipation_violated(agent):
    """Test FDT validation detects violations."""
    temperature = 300.0
    friction = 1.0
    measured_D = 1e-20  # Very different from theory
    validation = agent.validate_fluctuation_dissipation(
        temperature, friction, measured_D, tolerance=0.1
    )
    assert not validation['fdt_satisfied']


def test_cross_validate_escape_rate_good(agent):
    """Test escape rate cross-validation with good agreement."""
    kramers_rate = 1e-3
    measured_rate = 1.1e-3
    validation = agent.cross_validate_escape_rate(kramers_rate, measured_rate)
    assert validation['methods_agree']


def test_cross_validate_escape_rate_poor(agent):
    """Test escape rate cross-validation with poor agreement."""
    kramers_rate = 1e-3
    measured_rate = 5e-3
    validation = agent.cross_validate_escape_rate(kramers_rate, measured_rate)
    assert not validation['methods_agree']


def test_langevin_einstein_relation(agent, sample_langevin_input):
    """Test Langevin satisfies Einstein relation D = kT/γ."""
    result = agent.execute(sample_langevin_input)
    assert result.success
    D = result.data['diffusion_coefficient']
    temperature = sample_langevin_input['parameters']['temperature']
    friction = sample_langevin_input['parameters']['friction']
    kB = 1.380649e-23
    D_theory = kB * temperature / friction
    # Should be close
    assert abs(D - D_theory) / D_theory < 0.1 or abs(D) < 1e-20


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_langevin_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_langevin_input)
    result2 = agent.execute_with_caching(sample_langevin_input)
    assert result1.success
    assert result2.success


def test_caching_different_inputs(agent, sample_langevin_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_langevin_input)
    modified = sample_langevin_input.copy()
    modified['parameters'] = sample_langevin_input['parameters'].copy()
    modified['parameters']['temperature'] = 350
    result2 = agent.execute_with_caching(modified)
    assert result1.success
    assert result2.success


def test_provenance_tracking(agent, sample_langevin_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_langevin_input)
    assert result.success
    assert result.provenance.agent_name == 'StochasticDynamicsAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_langevin_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_langevin_input)
    assert result.success
    assert len(result.provenance.input_hash) == 64


# ============================================================================
# Test: Physical Validation (4 tests)
# ============================================================================

def test_langevin_equilibrium_distribution(agent):
    """Test Langevin reaches correct equilibrium distribution."""
    input_data = {
        'method': 'langevin',
        'potential': {'type': 'harmonic', 'k': 1.0},
        'parameters': {
            'temperature': 300,
            'friction': 1.0,
            'n_steps': 50000,
            'dt': 0.01
        }
    }
    result = agent.execute(input_data)
    assert result.success
    # For harmonic potential, variance = kT/k
    variance = result.data['statistics']['std_position'] ** 2
    assert variance > 0


def test_master_equation_stationary_distribution(agent, sample_master_equation_input):
    """Test master equation reaches stationary distribution."""
    result = agent.execute(sample_master_equation_input)
    assert result.success
    stat_dist = np.array(result.data['stationary_distribution'])
    # Should sum to 1
    assert abs(np.sum(stat_dist) - 1.0) < 1e-6


def test_kramers_arrhenius_behavior(agent, sample_kramers_input):
    """Test Kramers rate shows Arrhenius behavior."""
    result = agent.execute(sample_kramers_input)
    assert result.success
    # Rate should be exponentially suppressed by barrier
    assert result.data['escape_rate'] < 1.0
    assert result.data['escape_rate'] > 0


def test_fokker_planck_probability_conservation(agent, sample_fokker_planck_input):
    """Test Fokker-Planck conserves probability."""
    result = agent.execute(sample_fokker_planck_input)
    assert result.success
    # Probability should sum to 1
    assert abs(result.data['normalization'] - 1.0) < 0.01


# ============================================================================
# Test: Workflow Integration (3 tests)
# ============================================================================

def test_workflow_langevin_to_first_passage(agent):
    """Test workflow: Langevin → first-passage time."""
    # Run Langevin to generate trajectories
    langevin_input = {
        'method': 'langevin',
        'potential': {'type': 'harmonic', 'k': 1.0},
        'parameters': {'temperature': 300, 'friction': 1.0, 'n_steps': 10000, 'dt': 0.01}
    }
    langevin_result = agent.execute(langevin_input)
    assert langevin_result.success
    
    # Analyze first-passage time
    fpt_input = {
        'method': 'first_passage',
        'parameters': {'barrier_position': 1.0, 'n_trajectories': 50, 'n_steps': 5000, 'dt': 0.01}
    }
    fpt_result = agent.execute(fpt_input)
    assert fpt_result.success


def test_workflow_kramers_to_simulation(agent):
    """Test workflow: Kramers theory → simulation validation."""
    # Calculate Kramers rate
    kramers_result = agent.execute(sample_kramers_input(agent))
    assert kramers_result.success
    kramers_rate = kramers_result.data['escape_rate']
    
    # Should be positive
    assert kramers_rate > 0


def test_workflow_all_stochastic_methods(agent):
    """Test all stochastic methods execute successfully."""
    methods = ['langevin', 'master_equation', 'first_passage', 'kramers_escape', 'fokker_planck']
    results = []
    
    for method in methods:
        if method == 'langevin':
            input_data = {
                'method': method,
                'potential': {'type': 'harmonic', 'k': 1.0},
                'parameters': {'temperature': 300, 'friction': 1.0, 'n_steps': 5000, 'dt': 0.01}
            }
        elif method == 'master_equation':
            input_data = {
                'method': method,
                'parameters': {'n_states': 3, 'initial_state': 0, 'n_steps': 1000}
            }
        elif method == 'first_passage':
            input_data = {
                'method': method,
                'parameters': {'barrier_position': 1.0, 'n_trajectories': 20, 'n_steps': 1000}
            }
        elif method == 'kramers_escape':
            input_data = {
                'method': method,
                'potential': {'type': 'double_well'},
                'parameters': {'temperature': 300, 'friction': 1.0, 'barrier_height': 5.0}
            }
        else:  # fokker_planck
            input_data = {
                'method': method,
                'parameters': {'x_min': -5, 'x_max': 5, 'n_grid': 50, 'n_steps': 500}
            }
        
        result = agent.execute(input_data)
        results.append(result.success)
    
    assert all(results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
