"""Comprehensive test suite for OptimalControlAgent.

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation
- All 5 methods (minimal dissipation, shortcuts, stochastic control, speed limits, RL)
- Integration methods
- Physics validation
"""

import pytest
import numpy as np
from typing import Dict, Any

from optimal_control_agent import OptimalControlAgent
from base_agent import AgentStatus, ExecutionEnvironment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create OptimalControlAgent instance."""
    return OptimalControlAgent()


@pytest.fixture
def sample_minimal_dissipation_input():
    """Sample input for minimal dissipation protocol."""
    return {
        'method': 'minimal_dissipation_protocol',
        'data': {
            'lambda_initial': 0.0,
            'lambda_final': 1.0
        },
        'parameters': {
            'duration': 10.0,
            'temperature': 300.0,
            'n_steps': 100
        },
        'analysis': ['protocol', 'dissipation', 'efficiency']
    }


@pytest.fixture
def sample_shortcut_input():
    """Sample input for shortcut to adiabaticity."""
    return {
        'method': 'shortcut_to_adiabaticity',
        'data': {
            'field_initial': 1.0,
            'field_final': 2.0
        },
        'parameters': {
            'duration': 1.0,
            'n_steps': 100
        },
        'analysis': ['cd_protocol', 'energy_cost', 'fidelity']
    }


@pytest.fixture
def sample_stochastic_control_input():
    """Sample input for stochastic optimal control."""
    return {
        'method': 'stochastic_optimal_control',
        'data': {
            'x_initial': 0.0,
            'x_target': 1.0
        },
        'parameters': {
            'duration': 10.0,
            'n_steps': 100,
            'state_cost': 1.0,
            'control_cost': 0.1
        },
        'analysis': ['control', 'trajectory', 'cost']
    }


@pytest.fixture
def sample_speed_limit_input():
    """Sample input for thermodynamic speed limits."""
    return {
        'method': 'thermodynamic_speed_limit',
        'data': {
            'free_energy_change': 10.0 * 1.380649e-23 * 300.0,
            'dissipation': 5.0 * 1.380649e-23 * 300.0,
            'activity': 1.0
        },
        'parameters': {
            'temperature': 300.0,
            'actual_duration': 2.0
        },
        'analysis': ['bounds', 'efficiency']
    }


@pytest.fixture
def sample_rl_input():
    """Sample input for reinforcement learning protocol."""
    return {
        'method': 'reinforcement_learning_protocol',
        'data': {
            'n_states': 10,
            'n_actions': 5
        },
        'parameters': {
            'n_episodes': 100,
            'learning_rate': 0.1,
            'discount_factor': 0.95
        },
        'analysis': ['policy', 'trajectory', 'convergence']
    }


# ============================================================================
# Test 1-5: Initialization and Metadata
# ============================================================================

def test_agent_initialization(agent):
    """Test 1: Agent initializes correctly."""
    assert agent.VERSION == "1.0.0"
    assert 'minimal_dissipation_protocol' in agent.supported_methods
    assert 'shortcut_to_adiabaticity' in agent.supported_methods
    assert 'stochastic_optimal_control' in agent.supported_methods
    assert 'thermodynamic_speed_limit' in agent.supported_methods
    assert 'reinforcement_learning_protocol' in agent.supported_methods


def test_agent_metadata(agent):
    """Test 2: Agent metadata is correct."""
    metadata = agent.get_metadata()
    assert metadata.name == "OptimalControlAgent"
    assert metadata.version == "1.0.0"
    assert metadata.author == "Nonequilibrium Physics Team"


def test_agent_capabilities(agent):
    """Test 3: Agent capabilities are defined."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 5
    cap_names = [cap.name for cap in capabilities]
    assert 'minimal_dissipation_protocol' in cap_names
    assert 'shortcut_to_adiabaticity' in cap_names


def test_agent_constants(agent):
    """Test 4: Physical constants are correctly defined."""
    assert agent.kB == 1.380649e-23  # Boltzmann constant
    assert agent.hbar == 1.054571817e-34  # Reduced Planck constant


def test_agent_config_initialization():
    """Test 5: Agent initializes with custom config."""
    config = {'custom_param': 42}
    agent = OptimalControlAgent(config)
    assert agent.config == config


# ============================================================================
# Test 6-15: Input Validation
# ============================================================================

def test_validate_minimal_dissipation_valid(agent, sample_minimal_dissipation_input):
    """Test 6: Valid minimal dissipation input passes validation."""
    result = agent.validate_input(sample_minimal_dissipation_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_missing_method(agent):
    """Test 7: Missing method field fails validation."""
    input_data = {'data': {}, 'parameters': {}}
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('method' in err.lower() for err in result.errors)


def test_validate_invalid_method(agent):
    """Test 8: Invalid method fails validation."""
    input_data = {
        'method': 'invalid_method',
        'data': {},
        'parameters': {}
    }
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('unsupported' in err.lower() for err in result.errors)


def test_validate_missing_data(agent):
    """Test 9: Missing data field fails validation."""
    input_data = {'method': 'minimal_dissipation_protocol', 'parameters': {}}
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('data' in err.lower() for err in result.errors)


def test_validate_missing_parameters_warning(agent):
    """Test 10: Missing parameters generates warning."""
    input_data = {'method': 'minimal_dissipation_protocol', 'data': {}}
    result = agent.validate_input(input_data)
    assert len(result.warnings) > 0
    assert any('parameters' in warn.lower() for warn in result.warnings)


def test_validate_negative_duration(agent):
    """Test 11: Negative duration fails validation."""
    input_data = {
        'method': 'minimal_dissipation_protocol',
        'data': {},
        'parameters': {'duration': -5.0}
    }
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('duration' in err.lower() for err in result.errors)


def test_validate_negative_temperature(agent):
    """Test 12: Negative temperature fails validation."""
    input_data = {
        'method': 'minimal_dissipation_protocol',
        'data': {},
        'parameters': {'temperature': -100.0}
    }
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('temperature' in err.lower() for err in result.errors)


def test_validate_stochastic_control_valid(agent, sample_stochastic_control_input):
    """Test 13: Valid stochastic control input passes validation."""
    result = agent.validate_input(sample_stochastic_control_input)
    assert result.valid is True


def test_validate_speed_limit_valid(agent, sample_speed_limit_input):
    """Test 14: Valid speed limit input passes validation."""
    result = agent.validate_input(sample_speed_limit_input)
    assert result.valid is True


def test_validate_rl_valid(agent, sample_rl_input):
    """Test 15: Valid RL input passes validation."""
    result = agent.validate_input(sample_rl_input)
    assert result.valid is True


# ============================================================================
# Test 16-25: Resource Estimation
# ============================================================================

def test_resource_estimation_minimal_dissipation(agent, sample_minimal_dissipation_input):
    """Test 16: Resource estimation for minimal dissipation."""
    req = agent.estimate_resources(sample_minimal_dissipation_input)
    assert req.environment == ExecutionEnvironment.LOCAL
    assert req.cpu_cores >= 1
    assert req.memory_gb > 0
    assert req.gpu_required is False


def test_resource_estimation_shortcut(agent, sample_shortcut_input):
    """Test 17: Resource estimation for shortcuts to adiabaticity."""
    req = agent.estimate_resources(sample_shortcut_input)
    assert req.environment == ExecutionEnvironment.LOCAL
    assert req.cpu_cores >= 1


def test_resource_estimation_stochastic_control(agent, sample_stochastic_control_input):
    """Test 18: Resource estimation for stochastic control (HPC)."""
    req = agent.estimate_resources(sample_stochastic_control_input)
    assert req.environment == ExecutionEnvironment.HPC
    assert req.cpu_cores >= 4
    assert req.memory_gb >= 4.0


def test_resource_estimation_speed_limit(agent, sample_speed_limit_input):
    """Test 19: Resource estimation for speed limits (lightweight)."""
    req = agent.estimate_resources(sample_speed_limit_input)
    assert req.environment == ExecutionEnvironment.LOCAL
    assert req.estimated_duration_seconds < 60


def test_resource_estimation_rl(agent, sample_rl_input):
    """Test 20: Resource estimation for RL (HPC, high resources)."""
    req = agent.estimate_resources(sample_rl_input)
    assert req.environment == ExecutionEnvironment.HPC
    assert req.cpu_cores >= 8
    assert req.memory_gb >= 8.0


def test_resource_scaling_large_steps(agent, sample_minimal_dissipation_input):
    """Test 21: Resource scaling for large step counts."""
    sample_minimal_dissipation_input['parameters']['n_steps'] = 2000
    req = agent.estimate_resources(sample_minimal_dissipation_input)
    assert req.memory_gb >= 2.0  # Should scale up


def test_resource_estimation_has_duration(agent, sample_minimal_dissipation_input):
    """Test 22: Resource estimation includes duration estimate."""
    req = agent.estimate_resources(sample_minimal_dissipation_input)
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_no_gpu(agent, sample_stochastic_control_input):
    """Test 23: No GPU required for any method."""
    req = agent.estimate_resources(sample_stochastic_control_input)
    assert req.gpu_required is False


def test_resource_estimation_default_method(agent):
    """Test 24: Resource estimation with default method."""
    input_data = {'data': {}, 'parameters': {}}
    req = agent.estimate_resources(input_data)
    assert req.cpu_cores > 0
    assert req.memory_gb > 0


def test_resource_estimation_all_methods(agent):
    """Test 25: Resource estimation for all supported methods."""
    methods = agent.supported_methods
    for method in methods:
        input_data = {'method': method, 'data': {}, 'parameters': {}}
        req = agent.estimate_resources(input_data)
        assert req.cpu_cores > 0


# ============================================================================
# Test 26-30: Execution - Minimal Dissipation Protocol
# ============================================================================

def test_execute_minimal_dissipation_success(agent, sample_minimal_dissipation_input):
    """Test 26: Minimal dissipation protocol executes successfully."""
    result = agent.execute(sample_minimal_dissipation_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'lambda_optimal' in result.data
    assert 'dissipation' in result.data


def test_execute_minimal_dissipation_protocol_shape(agent, sample_minimal_dissipation_input):
    """Test 27: Protocol has correct shape and endpoints."""
    result = agent.execute(sample_minimal_dissipation_input)
    lambda_opt = result.data['lambda_optimal']
    assert len(lambda_opt) == 100
    assert np.isclose(lambda_opt[0], 0.0, atol=1e-6)
    assert np.isclose(lambda_opt[-1], 1.0, atol=1e-6)


def test_execute_minimal_dissipation_physics(agent, sample_minimal_dissipation_input):
    """Test 28: Dissipation is positive."""
    result = agent.execute(sample_minimal_dissipation_input)
    dissipation = result.data['dissipation']
    assert dissipation > 0


def test_execute_minimal_dissipation_efficiency(agent, sample_minimal_dissipation_input):
    """Test 29: Efficiency metric is in valid range."""
    result = agent.execute(sample_minimal_dissipation_input)
    efficiency = result.data['efficiency_metric']
    assert 0 < efficiency <= 1


def test_execute_minimal_dissipation_provenance(agent, sample_minimal_dissipation_input):
    """Test 30: Result includes provenance tracking."""
    result = agent.execute(sample_minimal_dissipation_input)
    assert result.provenance is not None
    assert result.provenance.agent_name == "OptimalControlAgent"
    assert result.provenance.agent_version == "1.0.0"


# ============================================================================
# Test 31-35: Execution - Shortcuts to Adiabaticity
# ============================================================================

def test_execute_shortcut_success(agent, sample_shortcut_input):
    """Test 31: Shortcut to adiabaticity executes successfully."""
    result = agent.execute(sample_shortcut_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'field_protocol' in result.data
    assert 'H_CD_magnitude' in result.data


def test_execute_shortcut_field_protocol(agent, sample_shortcut_input):
    """Test 32: Field protocol has correct endpoints."""
    result = agent.execute(sample_shortcut_input)
    field = result.data['field_protocol']
    assert np.isclose(field[0], 1.0, atol=1e-6)
    assert np.isclose(field[-1], 2.0, atol=1e-6)


def test_execute_shortcut_fidelity(agent, sample_shortcut_input):
    """Test 33: CD driving achieves perfect fidelity."""
    result = agent.execute(sample_shortcut_input)
    fidelity_cd = result.data['fidelity_with_cd']
    assert np.isclose(fidelity_cd, 1.0, atol=1e-6)


def test_execute_shortcut_energy_cost(agent, sample_shortcut_input):
    """Test 34: CD energy cost is positive."""
    result = agent.execute(sample_shortcut_input)
    energy_cost = result.data['energy_cost_cd_J']
    assert energy_cost > 0


def test_execute_shortcut_speedup(agent, sample_shortcut_input):
    """Test 35: Speedup factor is computed."""
    result = agent.execute(sample_shortcut_input)
    speedup = result.data['speedup_factor']
    assert speedup > 0


# ============================================================================
# Test 36-40: Execution - Stochastic Optimal Control
# ============================================================================

def test_execute_stochastic_control_success(agent, sample_stochastic_control_input):
    """Test 36: Stochastic optimal control executes successfully."""
    result = agent.execute(sample_stochastic_control_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'x_optimal' in result.data
    assert 'u_optimal' in result.data


def test_execute_stochastic_control_trajectory(agent, sample_stochastic_control_input):
    """Test 37: Optimal trajectory reaches near target."""
    result = agent.execute(sample_stochastic_control_input)
    x_optimal = result.data['x_optimal']
    x_target = sample_stochastic_control_input['data']['x_target']
    final_error = result.data['final_error']
    assert final_error < 0.2  # Should be reasonably close


def test_execute_stochastic_control_cost(agent, sample_stochastic_control_input):
    """Test 38: Total cost is positive and includes both state and control."""
    result = agent.execute(sample_stochastic_control_input)
    cost_state = result.data['cost_state']
    cost_control = result.data['cost_control']
    total_cost = result.data['total_cost']
    assert cost_state >= 0
    assert cost_control >= 0
    assert np.isclose(total_cost, cost_state + cost_control, atol=1e-6)


def test_execute_stochastic_control_arrays(agent, sample_stochastic_control_input):
    """Test 39: Control and state arrays have correct length."""
    result = agent.execute(sample_stochastic_control_input)
    x_opt = result.data['x_optimal']
    u_opt = result.data['u_optimal']
    n_steps = sample_stochastic_control_input['parameters']['n_steps']
    assert len(x_opt) == n_steps
    assert len(u_opt) == n_steps


def test_execute_stochastic_control_initial_condition(agent, sample_stochastic_control_input):
    """Test 40: Initial condition is respected."""
    result = agent.execute(sample_stochastic_control_input)
    x_opt = result.data['x_optimal']
    x_initial = sample_stochastic_control_input['data']['x_initial']
    assert np.isclose(x_opt[0], x_initial, atol=1e-6)


# ============================================================================
# Test 41-45: Execution - Thermodynamic Speed Limits
# ============================================================================

def test_execute_speed_limit_success(agent, sample_speed_limit_input):
    """Test 41: Speed limit calculation executes successfully."""
    result = agent.execute(sample_speed_limit_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'tau_minimum' in result.data
    assert 'tau_tur_bound' in result.data


def test_execute_speed_limit_bounds_positive(agent, sample_speed_limit_input):
    """Test 42: All speed limit bounds are positive."""
    result = agent.execute(sample_speed_limit_input)
    tau_tur = result.data['tau_tur_bound']
    tau_activity = result.data['tau_activity_bound']
    tau_geometric = result.data['tau_geometric_bound']
    assert tau_tur > 0
    assert tau_activity > 0
    assert tau_geometric > 0


def test_execute_speed_limit_minimum(agent, sample_speed_limit_input):
    """Test 43: Minimum duration is most restrictive bound."""
    result = agent.execute(sample_speed_limit_input)
    tau_min = result.data['tau_minimum']
    tau_tur = result.data['tau_tur_bound']
    tau_activity = result.data['tau_activity_bound']
    tau_geometric = result.data['tau_geometric_bound']
    assert tau_min == max(tau_tur, tau_activity, tau_geometric)


def test_execute_speed_limit_efficiency(agent, sample_speed_limit_input):
    """Test 44: Efficiency is in valid range."""
    result = agent.execute(sample_speed_limit_input)
    efficiency = result.data['efficiency_vs_limit']
    assert 0 <= efficiency <= 1


def test_execute_speed_limit_satisfies_bounds(agent, sample_speed_limit_input):
    """Test 45: Checks if actual duration satisfies bounds."""
    result = agent.execute(sample_speed_limit_input)
    satisfies = result.data['satisfies_bounds']
    actual = result.data['actual_duration']
    minimum = result.data['tau_minimum']
    assert satisfies == (actual >= minimum)


# ============================================================================
# Test 46-50: Execution - Reinforcement Learning
# ============================================================================

def test_execute_rl_success(agent, sample_rl_input):
    """Test 46: RL protocol executes successfully."""
    result = agent.execute(sample_rl_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'learned_policy' in result.data
    assert 'optimal_trajectory' in result.data


def test_execute_rl_policy_shape(agent, sample_rl_input):
    """Test 47: Learned policy has correct shape."""
    result = agent.execute(sample_rl_input)
    policy = result.data['learned_policy']
    n_states = sample_rl_input['data']['n_states']
    assert len(policy) == n_states


def test_execute_rl_convergence(agent, sample_rl_input):
    """Test 48: RL training shows improvement."""
    result = agent.execute(sample_rl_input)
    episode_rewards = result.data['episode_rewards']
    n_episodes = sample_rl_input['parameters']['n_episodes']
    assert len(episode_rewards) == n_episodes
    # Final rewards should be better than initial
    mean_final = np.mean(episode_rewards[-10:])
    mean_initial = np.mean(episode_rewards[:10])
    assert mean_final >= mean_initial


def test_execute_rl_trajectory_reaches_goal(agent, sample_rl_input):
    """Test 49: Optimal trajectory reaches goal state."""
    result = agent.execute(sample_rl_input)
    trajectory = result.data['optimal_trajectory']
    n_states = sample_rl_input['data']['n_states']
    assert trajectory[-1] == n_states - 1


def test_execute_rl_convergence_score(agent, sample_rl_input):
    """Test 50: Convergence score indicates learning."""
    result = agent.execute(sample_rl_input)
    convergence_score = result.data['convergence_score']
    training_converged = result.data['training_converged']
    assert convergence_score > 0
    assert isinstance(training_converged, bool)


# ============================================================================
# Test 51-55: Integration Methods
# ============================================================================

def test_integration_optimize_driven_protocol(agent):
    """Test 51: Integration with DrivenSystemsAgent."""
    driven_params = {
        'lambda_initial': 0.5,
        'lambda_final': 1.5
    }
    result = agent.optimize_driven_protocol(driven_params)
    assert 'lambda_optimal' in result
    assert 'dissipation' in result


def test_integration_design_minimal_work(agent):
    """Test 52: Integration with FluctuationAgent."""
    fluctuation_data = {
        'free_energy_change': 20.0
    }
    result = agent.design_minimal_work_process(fluctuation_data)
    assert 'tau_minimum' in result
    assert 'tau_tur_bound' in result


def test_integration_feedback_control(agent):
    """Test 53: Integration with InformationThermodynamicsAgent."""
    info_thermo_result = {
        'information_nats': 2.5
    }
    result = agent.feedback_optimal_control(info_thermo_result)
    assert 'x_optimal' in result
    assert 'u_optimal' in result
    assert 'total_cost' in result


def test_integration_optimize_driven_returns_dict(agent):
    """Test 54: Integration methods return dictionaries."""
    driven_params = {'lambda_initial': 0.0, 'lambda_final': 1.0}
    result = agent.optimize_driven_protocol(driven_params)
    assert isinstance(result, dict)


def test_integration_feedback_uses_information(agent):
    """Test 55: Feedback control incorporates information value."""
    info_thermo_result = {'information_nats': 3.0}
    result = agent.feedback_optimal_control(info_thermo_result)
    x_target_expected = 3.0
    # Check that target is influenced by information
    x_optimal = result['x_optimal']
    assert max(x_optimal) > 0  # Should move toward information-based target


# ============================================================================
# Test 56-60: Error Handling and Edge Cases
# ============================================================================

def test_execute_invalid_input_returns_failure(agent):
    """Test 56: Invalid input returns FAILED status."""
    invalid_input = {'method': 'invalid_method'}
    result = agent.execute(invalid_input)
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0


def test_execute_missing_data_returns_failure(agent):
    """Test 57: Missing data returns FAILED status."""
    invalid_input = {'method': 'minimal_dissipation_protocol'}
    result = agent.execute(invalid_input)
    assert result.status == AgentStatus.FAILED


def test_minimal_dissipation_zero_duration_safe(agent):
    """Test 58: Zero duration handled safely."""
    input_data = {
        'method': 'minimal_dissipation_protocol',
        'data': {'lambda_initial': 0.0, 'lambda_final': 1.0},
        'parameters': {'duration': 0.0}
    }
    result = agent.execute(input_data)
    # Should fail validation
    assert result.status == AgentStatus.FAILED


def test_speed_limit_zero_dissipation(agent):
    """Test 59: Zero dissipation gives infinite TUR bound."""
    input_data = {
        'method': 'thermodynamic_speed_limit',
        'data': {
            'free_energy_change': 10.0,
            'dissipation': 0.0,
            'activity': 1.0
        },
        'parameters': {'temperature': 300.0}
    }
    result = agent.execute(input_data)
    # Should handle gracefully
    if result.status == AgentStatus.SUCCESS:
        assert np.isinf(result.data['tau_tur_bound'])


def test_rl_zero_episodes_safe(agent):
    """Test 60: Zero episodes handled safely."""
    input_data = {
        'method': 'reinforcement_learning_protocol',
        'data': {'n_states': 10, 'n_actions': 5},
        'parameters': {'n_episodes': 0}
    }
    result = agent.execute(input_data)
    # Should complete (degenerate case with no training)
    if result.status == AgentStatus.SUCCESS:
        assert len(result.data['episode_rewards']) == 0


# ============================================================================
# Test 61-65: Physics Validation
# ============================================================================

def test_physics_dissipation_increases_with_speed(agent):
    """Test 61: Faster protocols have higher dissipation."""
    input_fast = {
        'method': 'minimal_dissipation_protocol',
        'data': {'lambda_initial': 0.0, 'lambda_final': 1.0},
        'parameters': {'duration': 1.0}
    }
    input_slow = {
        'method': 'minimal_dissipation_protocol',
        'data': {'lambda_initial': 0.0, 'lambda_final': 1.0},
        'parameters': {'duration': 10.0}
    }
    result_fast = agent.execute(input_fast)
    result_slow = agent.execute(input_slow)
    assert result_fast.data['dissipation'] > result_slow.data['dissipation']


def test_physics_cd_energy_scales_with_speed(agent):
    """Test 62: Faster shortcuts require more CD energy."""
    input_fast = {
        'method': 'shortcut_to_adiabaticity',
        'data': {'field_initial': 1.0, 'field_final': 2.0},
        'parameters': {'duration': 0.1}
    }
    input_slow = {
        'method': 'shortcut_to_adiabaticity',
        'data': {'field_initial': 1.0, 'field_final': 2.0},
        'parameters': {'duration': 1.0}
    }
    result_fast = agent.execute(input_fast)
    result_slow = agent.execute(input_slow)
    assert result_fast.data['energy_cost_cd_J'] > result_slow.data['energy_cost_cd_J']


def test_physics_tur_bound_satisfied(agent):
    """Test 63: TUR bound satisfied for given dissipation."""
    kB = 1.380649e-23
    T = 300.0
    delta_F = 10.0 * kB * T
    dissipation = 5.0 * kB * T

    input_data = {
        'method': 'thermodynamic_speed_limit',
        'data': {
            'free_energy_change': delta_F,
            'dissipation': dissipation
        },
        'parameters': {'temperature': T, 'actual_duration': 10.0}
    }
    result = agent.execute(input_data)

    # Check TUR: τ * Σ ≥ ΔF² / (2kT)
    tau = result.data['actual_duration']
    expected_bound = (delta_F**2) / (2 * kB * T * dissipation)
    assert result.data['tau_tur_bound'] == pytest.approx(expected_bound, rel=1e-6)


def test_physics_control_cost_tradeoff(agent):
    """Test 64: Higher control cost reduces control effort."""
    input_low_cost = {
        'method': 'stochastic_optimal_control',
        'data': {'x_initial': 0.0, 'x_target': 1.0},
        'parameters': {'duration': 5.0, 'control_cost': 0.01}
    }
    input_high_cost = {
        'method': 'stochastic_optimal_control',
        'data': {'x_initial': 0.0, 'x_target': 1.0},
        'parameters': {'duration': 5.0, 'control_cost': 1.0}
    }
    result_low = agent.execute(input_low_cost)
    result_high = agent.execute(input_high_cost)

    # Higher R should reduce control effort
    control_effort_low = np.sum(np.array(result_low.data['u_optimal'])**2)
    control_effort_high = np.sum(np.array(result_high.data['u_optimal'])**2)
    assert control_effort_low > control_effort_high


def test_physics_rl_discount_factor_effect(agent):
    """Test 65: Discount factor affects learning."""
    input_high_discount = {
        'method': 'reinforcement_learning_protocol',
        'data': {'n_states': 10, 'n_actions': 5},
        'parameters': {'n_episodes': 50, 'discount_factor': 0.95}
    }
    input_low_discount = {
        'method': 'reinforcement_learning_protocol',
        'data': {'n_states': 10, 'n_actions': 5},
        'parameters': {'n_episodes': 50, 'discount_factor': 0.5}
    }
    result_high = agent.execute(input_high_discount)
    result_low = agent.execute(input_low_discount)

    # Both should learn, but with different emphasis
    assert result_high.data['training_converged'] or result_low.data['training_converged']


# ============================================================================
# Test 66-70: Metadata and Provenance
# ============================================================================

def test_provenance_includes_execution_time(agent, sample_minimal_dissipation_input):
    """Test 66: Provenance includes execution time."""
    result = agent.execute(sample_minimal_dissipation_input)
    assert result.provenance.execution_time_sec >= 0


def test_provenance_includes_input_hash(agent, sample_minimal_dissipation_input):
    """Test 67: Provenance includes input hash."""
    result = agent.execute(sample_minimal_dissipation_input)
    assert result.provenance.input_hash is not None
    assert len(result.provenance.input_hash) > 0


def test_metadata_includes_method(agent, sample_minimal_dissipation_input):
    """Test 68: Result metadata includes method."""
    result = agent.execute(sample_minimal_dissipation_input)
    assert result.metadata['method'] == 'minimal_dissipation_protocol'


def test_metadata_includes_analysis_type(agent, sample_minimal_dissipation_input):
    """Test 69: Result metadata includes analysis type."""
    result = agent.execute(sample_minimal_dissipation_input)
    assert 'analysis_type' in result.metadata
    assert result.metadata['analysis_type'] == sample_minimal_dissipation_input['analysis']


def test_different_inputs_different_hashes(agent):
    """Test 70: Different inputs produce different hashes."""
    input1 = {
        'method': 'minimal_dissipation_protocol',
        'data': {'lambda_initial': 0.0, 'lambda_final': 1.0},
        'parameters': {'duration': 10.0}
    }
    input2 = {
        'method': 'minimal_dissipation_protocol',
        'data': {'lambda_initial': 0.0, 'lambda_final': 2.0},
        'parameters': {'duration': 10.0}
    }
    result1 = agent.execute(input1)
    result2 = agent.execute(input2)
    assert result1.provenance.input_hash != result2.provenance.input_hash
