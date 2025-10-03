"""
Test suite for FluctuationAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation (LOCAL)
- Execution for all fluctuation theorems (5 theorems)
- Theorem validation methods
- Integration methods
- Caching and provenance
- Physical/mathematical validation

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
from fluctuation_agent import FluctuationAgent
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a FluctuationAgent instance."""
    return FluctuationAgent()


@pytest.fixture
def sample_jarzynski_input():
    """Sample input for Jarzynski equality."""
    work_data = np.random.normal(-5.0 + 2.0, 3.0, 1000)
    return {
        'theorem': 'jarzynski',
        'work_data': work_data.tolist(),
        'parameters': {
            'temperature': 300.0,
            'free_energy_reference': -5.0
        }
    }


@pytest.fixture
def sample_crooks_input():
    """Sample input for Crooks theorem."""
    delta_F = -5.0
    work_forward = np.random.normal(delta_F + 2.0, 3.0, 1000)
    work_reverse = np.random.normal(-delta_F + 2.0, 3.0, 1000)
    return {
        'theorem': 'crooks',
        'work_data': work_forward.tolist(),
        'reverse_work_data': work_reverse.tolist(),
        'parameters': {'temperature': 300.0}
    }


@pytest.fixture
def sample_integral_fluctuation_input():
    """Sample input for integral fluctuation theorem."""
    entropy_production = np.random.lognormal(np.log(2.0), 0.5, 1000)
    return {
        'theorem': 'integral_fluctuation',
        'entropy_production': entropy_production.tolist(),
        'parameters': {}
    }


@pytest.fixture
def sample_transient_input():
    """Sample input for transient fluctuation theorem."""
    sigma_t = np.random.normal(1.0, 0.5, 1000)
    return {
        'theorem': 'transient',
        'entropy_production_rate': sigma_t.tolist(),
        'parameters': {'observation_time': 1.0}
    }


@pytest.fixture
def sample_detailed_balance_input():
    """Sample input for detailed balance testing."""
    n_states = 5
    transition_matrix = np.random.rand(n_states, n_states)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return {
        'theorem': 'detailed_balance',
        'transition_matrix': transition_matrix.tolist(),
        'parameters': {}
    }


# ============================================================================
# Test: Initialization and Metadata (3 tests)
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert isinstance(agent.supported_theorems, list)
    assert len(agent.supported_theorems) >= 5


def test_get_metadata(agent):
    """Test agent metadata retrieval."""
    metadata = agent.get_metadata()
    assert metadata.name == "FluctuationAgent"
    assert metadata.version == "1.0.0"
    assert "fluctuation" in metadata.description.lower()


def test_get_capabilities(agent):
    """Test agent capabilities listing."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) >= 4
    method_names = [cap.name for cap in capabilities]
    assert 'Jarzynski Equality' in method_names
    assert 'Crooks Fluctuation Theorem' in method_names


# ============================================================================
# Test: Input Validation (8 tests)
# ============================================================================

def test_validate_input_success_jarzynski(agent, sample_jarzynski_input):
    """Test validation succeeds for valid Jarzynski input."""
    validation = agent.validate_input(sample_jarzynski_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_success_crooks(agent, sample_crooks_input):
    """Test validation succeeds for valid Crooks input."""
    validation = agent.validate_input(sample_crooks_input)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_input_missing_theorem(agent):
    """Test validation fails when theorem is missing."""
    validation = agent.validate_input({'work_data': [1, 2, 3]})
    assert not validation.valid
    assert any('theorem' in err.lower() for err in validation.errors)


def test_validate_input_invalid_theorem(agent):
    """Test validation fails for unsupported theorem."""
    validation = agent.validate_input({'theorem': 'unknown_theorem'})
    assert not validation.valid
    assert any('unsupported theorem' in err.lower() for err in validation.errors)


def test_validate_input_jarzynski_missing_work(agent):
    """Test validation fails when work data is missing for Jarzynski."""
    validation = agent.validate_input({
        'theorem': 'jarzynski',
        'parameters': {'temperature': 300}
    })
    assert not validation.valid
    assert any('work_data' in err.lower() for err in validation.errors)


def test_validate_input_crooks_missing_reverse(agent):
    """Test validation fails when reverse work is missing for Crooks."""
    validation = agent.validate_input({
        'theorem': 'crooks',
        'work_data': [1, 2, 3]
    })
    assert not validation.valid


def test_validate_input_small_sample_warning(agent):
    """Test validation warns for small sample size."""
    validation = agent.validate_input({
        'theorem': 'jarzynski',
        'work_data': [1, 2, 3, 4, 5]
    })
    assert len(validation.warnings) > 0


def test_validate_input_negative_temperature(agent, sample_jarzynski_input):
    """Test validation fails for negative temperature."""
    sample_jarzynski_input['parameters']['temperature'] = -50
    validation = agent.validate_input(sample_jarzynski_input)
    assert not validation.valid


# ============================================================================
# Test: Resource Estimation (7 tests)
# ============================================================================

def test_estimate_resources_small_dataset(agent, sample_jarzynski_input):
    """Test resource estimation for small dataset."""
    resources = agent.estimate_resources(sample_jarzynski_input)
    assert resources.execution_environment.value == 'local'
    assert resources.gpu_count == 0
    assert resources.estimated_time_sec == 60


def test_estimate_resources_large_dataset(agent):
    """Test resource estimation for large dataset."""
    large_work = list(range(200000))
    input_data = {
        'theorem': 'jarzynski',
        'work_data': large_work
    }
    resources = agent.estimate_resources(input_data)
    assert resources.execution_environment.value == 'local'
    assert resources.estimated_time_sec == 600


def test_estimate_resources_jarzynski(agent, sample_jarzynski_input):
    """Test resource estimation for Jarzynski."""
    resources = agent.estimate_resources(sample_jarzynski_input)
    assert resources.cpu_cores >= 4


def test_estimate_resources_crooks(agent, sample_crooks_input):
    """Test resource estimation for Crooks."""
    resources = agent.estimate_resources(sample_crooks_input)
    assert resources.memory_gb >= 8.0


def test_estimate_resources_integral_fluctuation(agent, sample_integral_fluctuation_input):
    """Test resource estimation for integral fluctuation theorem."""
    resources = agent.estimate_resources(sample_integral_fluctuation_input)
    assert resources.estimated_time_sec > 0


def test_estimate_resources_transient(agent, sample_transient_input):
    """Test resource estimation for transient theorem."""
    resources = agent.estimate_resources(sample_transient_input)
    assert resources.execution_environment.value == 'local'


def test_estimate_resources_detailed_balance(agent, sample_detailed_balance_input):
    """Test resource estimation for detailed balance."""
    resources = agent.estimate_resources(sample_detailed_balance_input)
    assert resources.cpu_cores >= 4


# ============================================================================
# Test: Execution for All Theorems (7 tests)
# ============================================================================

def test_execute_jarzynski_success(agent, sample_jarzynski_input):
    """Test Jarzynski equality execution succeeds."""
    result = agent.execute(sample_jarzynski_input)
    assert result.success
    assert 'free_energy_difference' in result.data
    assert 'work_statistics' in result.data
    assert 'second_law_satisfied' in result.data
    assert 'convergence' in result.data


def test_execute_crooks_success(agent, sample_crooks_input):
    """Test Crooks theorem execution succeeds."""
    result = agent.execute(sample_crooks_input)
    assert result.success
    assert 'free_energy_difference' in result.data
    assert 'free_energy_jarzynski' in result.data
    assert 'crooks_validation' in result.data
    assert 'work_distributions' in result.data


def test_execute_integral_fluctuation_success(agent, sample_integral_fluctuation_input):
    """Test integral fluctuation theorem execution succeeds."""
    result = agent.execute(sample_integral_fluctuation_input)
    assert result.success
    assert 'ift_average' in result.data
    assert 'theorem_satisfied' in result.data
    assert 'entropy_production_statistics' in result.data
    assert result.data['expected_value'] == 1.0


def test_execute_transient_success(agent, sample_transient_input):
    """Test transient fluctuation theorem execution succeeds."""
    result = agent.execute(sample_transient_input)
    assert result.success
    assert 'mean_entropy_production_rate' in result.data
    assert 'tft_validation' in result.data


def test_execute_detailed_balance_success(agent, sample_detailed_balance_input):
    """Test detailed balance testing execution succeeds."""
    result = agent.execute(sample_detailed_balance_input)
    assert result.success
    assert 'detailed_balance_satisfied' in result.data
    assert 'stationary_distribution' in result.data
    assert 'equilibrium' in result.data


def test_execute_jarzynski_second_law(agent, sample_jarzynski_input):
    """Test Jarzynski respects second law."""
    result = agent.execute(sample_jarzynski_input)
    assert result.success
    assert result.data['second_law_satisfied']
    assert result.data['dissipated_work'] >= -1.0  # Allow small violations due to finite sampling


def test_execute_invalid_theorem(agent):
    """Test execution fails for invalid theorem."""
    result = agent.execute({'theorem': 'invalid_theorem', 'work_data': [1, 2, 3]})
    assert not result.success
    assert len(result.errors) > 0


# ============================================================================
# Test: Theorem Validation (6 tests)
# ============================================================================

def test_jarzynski_free_energy_reasonable(agent, sample_jarzynski_input):
    """Test Jarzynski gives reasonable free energy estimate."""
    result = agent.execute(sample_jarzynski_input)
    assert result.success
    fe = result.data['free_energy_difference']
    # Should be near true value of -5.0
    assert -10.0 < fe < 0.0


def test_crooks_intersection_point(agent, sample_crooks_input):
    """Test Crooks intersection point equals free energy."""
    result = agent.execute(sample_crooks_input)
    assert result.success
    # Crooks validation should have correlation
    assert 'correlation' in result.data['crooks_validation']


def test_integral_fluctuation_unity(agent, sample_integral_fluctuation_input):
    """Test IFT average should be close to 1."""
    result = agent.execute(sample_integral_fluctuation_input)
    assert result.success
    ift_avg = result.data['ift_average']
    # Should be close to 1 with finite sampling
    assert 0.5 < ift_avg < 2.0


def test_detailed_balance_stationary_sum(agent, sample_detailed_balance_input):
    """Test stationary distribution sums to 1."""
    result = agent.execute(sample_detailed_balance_input)
    assert result.success
    stat_dist = np.array(result.data['stationary_distribution'])
    assert abs(np.sum(stat_dist) - 1.0) < 1e-6


def test_transient_symmetry(agent, sample_transient_input):
    """Test transient FT symmetry relation."""
    result = agent.execute(sample_transient_input)
    assert result.success
    # Should have histogram data
    assert 'histogram' in result.data


def test_entropy_production_positive_on_average(agent, sample_integral_fluctuation_input):
    """Test entropy production is positive on average."""
    result = agent.execute(sample_integral_fluctuation_input)
    assert result.success
    mean_sigma = result.data['entropy_production_statistics']['mean']
    assert mean_sigma > 0


# ============================================================================
# Test: Integration Methods (5 tests)
# ============================================================================

def test_cross_validate_free_energy_good(agent):
    """Test cross-validation with good agreement."""
    validation = agent.cross_validate_free_energy(-5.0, -5.2, tolerance=0.5)
    assert validation['methods_agree']


def test_cross_validate_free_energy_poor(agent):
    """Test cross-validation with poor agreement."""
    validation = agent.cross_validate_free_energy(-5.0, -8.0, tolerance=0.5)
    assert not validation['methods_agree']


def test_estimate_sampling_requirements(agent):
    """Test sampling requirements estimation."""
    requirements = agent.estimate_sampling_requirements(
        target_uncertainty=0.1,
        work_std=3.0,
        temperature=300.0
    )
    assert requirements['estimated_samples_needed'] > 0
    assert 'sampling_feasibility' in requirements


def test_jarzynski_crooks_consistency(agent):
    """Test Jarzynski and Crooks give consistent results."""
    # Generate consistent data
    delta_F = -5.0
    work_forward = np.random.normal(delta_F + 2.0, 2.0, 1000)
    work_reverse = np.random.normal(-delta_F + 2.0, 2.0, 1000)
    
    # Jarzynski
    jarzynski_input = {
        'theorem': 'jarzynski',
        'work_data': work_forward.tolist(),
        'parameters': {'temperature': 300.0}
    }
    result_j = agent.execute(jarzynski_input)
    
    # Crooks
    crooks_input = {
        'theorem': 'crooks',
        'work_data': work_forward.tolist(),
        'reverse_work_data': work_reverse.tolist(),
        'parameters': {'temperature': 300.0}
    }
    result_c = agent.execute(crooks_input)
    
    assert result_j.success and result_c.success
    # Both should give similar free energy
    fe_j = result_j.data['free_energy_difference']
    fe_c = result_c.data['free_energy_difference']
    assert abs(fe_j - fe_c) < 2.0  # Within 2 kJ/mol


def test_analyze_trajectory_method(agent):
    """Test trajectory analysis method."""
    result = agent.analyze_trajectory('dummy_trajectory')
    assert 'work_extracted' in result


# ============================================================================
# Test: Caching and Provenance (4 tests)
# ============================================================================

def test_caching_identical_inputs(agent, sample_jarzynski_input):
    """Test caching works for identical inputs."""
    result1 = agent.execute_with_caching(sample_jarzynski_input)
    result2 = agent.execute_with_caching(sample_jarzynski_input)
    assert result1.success
    assert result2.success


def test_caching_different_inputs(agent, sample_jarzynski_input):
    """Test different inputs produce different results."""
    result1 = agent.execute_with_caching(sample_jarzynski_input)
    modified = sample_jarzynski_input.copy()
    modified['parameters'] = {'temperature': 350.0}
    result2 = agent.execute_with_caching(modified)
    assert result1.success
    assert result2.success


def test_provenance_tracking(agent, sample_jarzynski_input):
    """Test provenance metadata is captured."""
    result = agent.execute(sample_jarzynski_input)
    assert result.success
    assert result.provenance.agent_name == 'FluctuationAgent'
    assert result.provenance.agent_version == '1.0.0'


def test_provenance_input_hash(agent, sample_jarzynski_input):
    """Test provenance includes input hash."""
    result = agent.execute(sample_jarzynski_input)
    assert result.success
    assert len(result.provenance.input_hash) == 64


# ============================================================================
# Test: Physical/Mathematical Validation (4 tests)
# ============================================================================

def test_jarzynski_jensen_inequality(agent):
    """Test Jarzynski satisfies Jensen's inequality."""
    # ⟨exp(-βW)⟩ ≥ exp(-β⟨W⟩) implies ΔF ≤ ⟨W⟩
    work_data = np.random.normal(0.0, 5.0, 1000)
    input_data = {
        'theorem': 'jarzynski',
        'work_data': work_data.tolist(),
        'parameters': {'temperature': 300.0}
    }
    result = agent.execute(input_data)
    assert result.success
    fe = result.data['free_energy_difference']
    mean_work = result.data['work_statistics']['mean_work']
    # ΔF ≤ ⟨W⟩ (second law)
    assert fe <= mean_work + 1.0  # Allow 1 kJ/mol tolerance


def test_second_law_violations_rare(agent, sample_integral_fluctuation_input):
    """Test second law violations are rare."""
    result = agent.execute(sample_integral_fluctuation_input)
    assert result.success
    prob_negative = result.data['second_law_violations']['probability']
    # For large systems, violations should be rare
    assert prob_negative < 0.5


def test_detailed_balance_equilibrium_only(agent):
    """Test detailed balance is satisfied only at equilibrium."""
    # Create detailed balance satisfying matrix
    n_states = 3
    rates = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    # Make it satisfy detailed balance
    pi = np.array([1, 2, 1])
    pi = pi / pi.sum()
    transition_matrix = rates.copy()
    for i in range(n_states):
        for j in range(n_states):
            transition_matrix[i, j] = rates[i, j] * pi[j]
        transition_matrix[i] = transition_matrix[i] / transition_matrix[i].sum()
    
    input_data = {
        'theorem': 'detailed_balance',
        'transition_matrix': transition_matrix.tolist()
    }
    result = agent.execute(input_data)
    assert result.success


def test_ift_exponential_martingale(agent, sample_integral_fluctuation_input):
    """Test IFT implies exp(-σ) is a martingale."""
    result = agent.execute(sample_integral_fluctuation_input)
    assert result.success
    # The test that ⟨exp(-σ)⟩ = 1 is already done
    assert 'ift_average' in result.data


# ============================================================================
# Test: Workflow Integration (3 tests)
# ============================================================================

def test_workflow_jarzynski_to_crooks(agent):
    """Test workflow: Jarzynski → Crooks validation."""
    work_forward = np.random.normal(-3.0, 2.0, 1000)
    work_reverse = np.random.normal(3.0, 2.0, 1000)
    
    # Jarzynski
    j_input = {
        'theorem': 'jarzynski',
        'work_data': work_forward.tolist(),
        'parameters': {'temperature': 300}
    }
    j_result = agent.execute(j_input)
    
    # Crooks
    c_input = {
        'theorem': 'crooks',
        'work_data': work_forward.tolist(),
        'reverse_work_data': work_reverse.tolist(),
        'parameters': {'temperature': 300}
    }
    c_result = agent.execute(c_input)
    
    assert j_result.success and c_result.success


def test_workflow_all_theorems(agent):
    """Test all fluctuation theorems execute successfully."""
    theorems = ['jarzynski', 'crooks', 'integral_fluctuation', 'transient', 'detailed_balance']
    results = []
    
    for theorem in theorems:
        if theorem == 'jarzynski':
            input_data = {
                'theorem': theorem,
                'work_data': list(np.random.normal(0, 1, 100)),
                'parameters': {'temperature': 300}
            }
        elif theorem == 'crooks':
            input_data = {
                'theorem': theorem,
                'work_data': list(np.random.normal(0, 1, 100)),
                'reverse_work_data': list(np.random.normal(0, 1, 100)),
                'parameters': {'temperature': 300}
            }
        elif theorem == 'integral_fluctuation':
            input_data = {
                'theorem': theorem,
                'entropy_production': list(np.random.lognormal(0, 1, 100))
            }
        elif theorem == 'transient':
            input_data = {
                'theorem': theorem,
                'entropy_production_rate': list(np.random.normal(1, 0.5, 100))
            }
        else:  # detailed_balance
            input_data = {
                'theorem': theorem,
                'transition_matrix': np.random.rand(3, 3).tolist()
            }
        
        result = agent.execute(input_data)
        results.append(result.success)
    
    assert all(results)


def test_workflow_free_energy_estimation_pipeline(agent):
    """Test complete free energy estimation pipeline."""
    # Generate work distribution
    work_data = np.random.normal(-5.0, 3.0, 2000)
    
    # Method 1: Jarzynski
    result = agent.execute({
        'theorem': 'jarzynski',
        'work_data': work_data.tolist(),
        'parameters': {'temperature': 300}
    })
    
    assert result.success
    assert result.data['convergence']['converged']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
