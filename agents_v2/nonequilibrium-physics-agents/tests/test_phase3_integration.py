"""Phase 3 Integration Tests - Cross-Agent Workflows.

Tests integration between Phase 3 agents and existing agents:
- Large Deviation + Fluctuation (10 tests)
- Optimal Control + Driven Systems (10 tests)
- Quantum + Transport (10 tests)
"""

import pytest
import numpy as np
from typing import Dict, Any

from large_deviation_agent import LargeDeviationAgent
from optimal_control_agent import OptimalControlAgent
from nonequilibrium_quantum_agent import NonequilibriumQuantumAgent
from base_agent import AgentStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ld_agent():
    """Create LargeDeviationAgent instance."""
    return LargeDeviationAgent()


@pytest.fixture
def oc_agent():
    """Create OptimalControlAgent instance."""
    return OptimalControlAgent()


@pytest.fixture
def quantum_agent():
    """Create NonequilibriumQuantumAgent instance."""
    return NonequilibriumQuantumAgent()


# ============================================================================
# Test 1-10: Large Deviation + Fluctuation Integration
# ============================================================================

def test_ld_fluctuation_rare_work_events(ld_agent):
    """Test 1: Rare event sampling for work distributions."""
    # Simulate work distribution (mock fluctuation data)
    work_samples = np.random.normal(10.0, 2.0, 1000)

    input_data = {
        'method': 'rare_event_sampling',
        'data': {
            'observable': work_samples.tolist(),
            'bias_parameter': 2.0
        },
        'parameters': {
            'n_grid': 50,
            'n_samples': 1000
        },
        'analysis': ['rate_function', 'reweighted']
    }

    result = ld_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'rate_function_I' in result.data
    assert 'theta_s' in result.data


def test_ld_fluctuation_tail_validation(ld_agent):
    """Test 2: Validate large deviation tails match fluctuation theorem."""
    # Work distribution with exponential tails
    work_samples = np.random.exponential(5.0, 2000)

    input_data = {
        'method': 'rare_event_sampling',
        'data': {'observable': work_samples.tolist()},
        'parameters': {},
        'analysis': ['rate_function']
    }

    result = ld_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS

    # Rate function should exist for tail values
    rate_function = result.data['rate_function_I']
    assert len(rate_function) > 0
    assert all(I >= 0 for I in rate_function)  # Rate function non-negative


def test_ld_fluctuation_integration_method(ld_agent):
    """Test 3: Integration method for fluctuation validation."""
    # Mock fluctuation agent result
    fluctuation_result = {
        'work_samples': np.random.normal(15.0, 3.0, 1000).tolist(),
        'free_energy_change': 15.0
    }

    result = ld_agent.validate_fluctuation_tail(fluctuation_result, observable='work')
    assert 'rate_function_I' in result
    assert 'observable_grid' in result


def test_ld_fluctuation_scgf_computation(ld_agent):
    """Test 4: Scaled cumulant generating function for work."""
    work_samples = np.random.normal(8.0, 1.5, 1500)

    input_data = {
        'method': 'rare_event_sampling',
        'data': {'observable': work_samples.tolist()},
        'parameters': {'n_samples': 1500},
        'analysis': ['scgf']
    }

    result = ld_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'theta_s' in result.data


def test_ld_fluctuation_importance_sampling(ld_agent):
    """Test 5: Importance sampling improves rare event statistics."""
    # Observable with rare large values
    observable = np.concatenate([
        np.random.normal(0.0, 1.0, 950),
        np.random.normal(10.0, 1.0, 50)  # Rare events
    ])

    input_data = {
        'method': 'rare_event_sampling',
        'data': {'observable': observable.tolist(), 'bias_parameter': 1.0},
        'parameters': {},
        'analysis': ['reweighted']
    }

    result = ld_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'reweighted_observable' in result.data


def test_ld_fluctuation_driven_rare_events(ld_agent):
    """Test 6: Analyze rare events in driven systems."""
    # Mock driven system result
    driven_result = {
        'work_samples': np.random.gamma(2.0, 3.0, 1000).tolist()
    }

    result = ld_agent.analyze_driven_rare_events(driven_result, observable_key='work')
    assert 'rate_function_I' in result
    assert 'theta_s' in result


def test_ld_fluctuation_transition_rates(ld_agent):
    """Test 7: Compute transition rates from stochastic dynamics."""
    # Mock stochastic result
    stochastic_result = {
        'trajectory': np.random.randn(1000).tolist(),
        'time_grid': np.linspace(0, 100, 1000).tolist()
    }

    result = ld_agent.compute_transition_rates(stochastic_result)
    assert 'transition_rate_AB' in result or 'escape_rate' in result


def test_ld_fluctuation_tps_committor(ld_agent):
    """Test 8: Transition path sampling with committor analysis."""
    # Trajectory crossing barrier
    trajectory = np.concatenate([
        np.full(300, -1.0) + np.random.randn(300) * 0.1,  # State A
        np.linspace(-1.0, 1.0, 200) + np.random.randn(200) * 0.2,  # Transition
        np.full(500, 1.0) + np.random.randn(500) * 0.1  # State B
    ])

    input_data = {
        'method': 'transition_path_sampling',
        'data': {
            'trajectory': trajectory.tolist(),
            'region_A': -1.0,
            'region_B': 1.0
        },
        'parameters': {'threshold': 0.0},
        'analysis': ['committor', 'reactive_flux']
    }

    result = ld_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'committor_values' in result.data


def test_ld_fluctuation_dynamical_phase_transition(ld_agent):
    """Test 9: Dynamical phase transition in activity."""
    # Time series with varying activity
    time_series = np.random.randn(2000)

    input_data = {
        'method': 'dynamical_phase_transition',
        'data': {'time_series': time_series.tolist()},
        'parameters': {'n_s_values': 20},
        'analysis': ['scgf', 'singularity']
    }

    result = ld_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'theta_values' in result.data
    assert 's_values' in result.data


def test_ld_fluctuation_s_ensemble(ld_agent):
    """Test 10: S-ensemble simulation for biased dynamics."""
    time_series = np.random.randn(1000)

    input_data = {
        'method': 's_ensemble_simulation',
        'data': {'time_series': time_series.tolist()},
        'parameters': {'bias_parameter': 1.5, 'n_trajectories': 100},
        'analysis': ['biased_ensemble']
    }

    result = ld_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'theta_s' in result.data


# ============================================================================
# Test 11-20: Optimal Control + Driven Systems Integration
# ============================================================================

def test_oc_driven_minimal_dissipation(oc_agent):
    """Test 11: Optimize driven system protocol for minimal dissipation."""
    driven_params = {
        'lambda_initial': 0.0,
        'lambda_final': 1.0
    }

    result = oc_agent.optimize_driven_protocol(driven_params)
    assert 'lambda_optimal' in result
    assert 'dissipation' in result
    assert result['dissipation'] > 0


def test_oc_driven_shear_rate_optimization(oc_agent):
    """Test 12: Optimize shear rate protocol for NEMD."""
    input_data = {
        'method': 'minimal_dissipation_protocol',
        'data': {
            'lambda_initial': 0.5,
            'lambda_final': 2.0
        },
        'parameters': {
            'duration': 20.0,
            'temperature': 350.0
        },
        'analysis': ['protocol', 'dissipation']
    }

    result = oc_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'protocol_velocity' in result.data


def test_oc_driven_shortcut_to_adiabaticity(oc_agent):
    """Test 13: Design shortcuts for fast driven processes."""
    input_data = {
        'method': 'shortcut_to_adiabaticity',
        'data': {
            'field_initial': 1.0,
            'field_final': 3.0
        },
        'parameters': {
            'duration': 0.5
        },
        'analysis': ['cd_protocol', 'speedup']
    }

    result = oc_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'speedup_factor' in result.data
    assert result.data['speedup_factor'] > 1.0


def test_oc_driven_feedback_control(oc_agent):
    """Test 14: Optimal feedback control for driven systems."""
    input_data = {
        'method': 'stochastic_optimal_control',
        'data': {
            'x_initial': 0.0,
            'x_target': 2.0
        },
        'parameters': {
            'duration': 15.0,
            'state_cost': 2.0,
            'control_cost': 0.5
        },
        'analysis': ['control', 'trajectory']
    }

    result = oc_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'u_optimal' in result.data
    assert 'x_optimal' in result.data


def test_oc_driven_thermodynamic_speed_limits(oc_agent):
    """Test 15: Speed limits for driven protocols."""
    input_data = {
        'method': 'thermodynamic_speed_limit',
        'data': {
            'free_energy_change': 50.0,
            'dissipation': 25.0
        },
        'parameters': {
            'temperature': 300.0
        },
        'analysis': ['bounds']
    }

    result = oc_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'tau_minimum' in result.data
    assert result.data['tau_minimum'] > 0


def test_oc_driven_reinforcement_learning(oc_agent):
    """Test 16: RL-optimized protocols for driven systems."""
    input_data = {
        'method': 'reinforcement_learning_protocol',
        'data': {
            'n_states': 20,
            'n_actions': 10
        },
        'parameters': {
            'n_episodes': 200,
            'learning_rate': 0.15
        },
        'analysis': ['policy', 'convergence']
    }

    result = oc_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'learned_policy' in result.data
    assert len(result.data['learned_policy']) == 20


def test_oc_driven_integration_method(oc_agent):
    """Test 17: Integration method optimizes driven protocols."""
    driven_params = {
        'lambda_initial': 0.2,
        'lambda_final': 1.8,
        'susceptibility': lambda x: 1.0 + 0.1 * x
    }

    result = oc_agent.optimize_driven_protocol(driven_params)
    assert 'lambda_optimal' in result
    assert 'efficiency_metric' in result


def test_oc_driven_minimal_work_design(oc_agent):
    """Test 18: Design minimal work process for fluctuation experiments."""
    fluctuation_data = {
        'free_energy_change': 30.0,
        'dissipation': 10.0
    }

    result = oc_agent.design_minimal_work_process(fluctuation_data)
    assert 'tau_minimum' in result
    assert 'tau_tur_bound' in result


def test_oc_driven_geodesic_protocol(oc_agent):
    """Test 19: Geodesic protocol in thermodynamic space."""
    input_data = {
        'method': 'minimal_dissipation_protocol',
        'data': {
            'lambda_initial': 0.0,
            'lambda_final': 2.0
        },
        'parameters': {
            'duration': 25.0,
            'n_steps': 200
        },
        'analysis': ['protocol']
    }

    result = oc_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'protocol_shape' in result.data
    assert result.data['protocol_shape'] == 'linear_geodesic'


def test_oc_driven_efficiency_optimization(oc_agent):
    """Test 20: Protocol optimization maximizes efficiency."""
    input_data = {
        'method': 'minimal_dissipation_protocol',
        'data': {
            'lambda_initial': 0.5,
            'lambda_final': 1.5
        },
        'parameters': {
            'duration': 30.0,
            'temperature': 400.0
        },
        'analysis': ['efficiency']
    }

    result = oc_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'efficiency_metric' in result.data
    assert 0 < result.data['efficiency_metric'] <= 1


# ============================================================================
# Test 21-30: Quantum + Transport Integration
# ============================================================================

def test_quantum_transport_landauer(quantum_agent):
    """Test 21: Quantum transport via Landauer formula."""
    input_data = {
        'method': 'quantum_transport',
        'data': {
            'mu_left': 0.15 * 1.602176634e-19,
            'mu_right': 0.0
        },
        'parameters': {
            'temperature': 77.0,  # Liquid nitrogen temperature
            'n_energies': 300
        },
        'analysis': ['conductance', 'current']
    }

    result = quantum_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'conductance_G0_units' in result.data
    assert result.data['conductance_G0_units'] >= 0


def test_quantum_transport_coefficients(quantum_agent):
    """Test 22: Quantum transport coefficients integration."""
    transport_data = {
        'mu_left': 0.1 * 1.602176634e-19,
        'mu_right': 0.05 * 1.602176634e-19
    }

    result = quantum_agent.quantum_transport_coefficients(transport_data)
    assert 'conductance_S' in result
    assert 'seebeck_coefficient_V_K' in result


def test_quantum_transport_seebeck(quantum_agent):
    """Test 23: Quantum Seebeck coefficient calculation."""
    input_data = {
        'method': 'quantum_transport',
        'data': {
            'mu_left': 0.08 * 1.602176634e-19,
            'mu_right': 0.02 * 1.602176634e-19
        },
        'parameters': {
            'temperature': 300.0
        },
        'analysis': ['thermoelectric']
    }

    result = quantum_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'seebeck_coefficient_V_K' in result.data


def test_quantum_transport_low_temperature(quantum_agent):
    """Test 24: Quantum transport at low temperature."""
    input_data = {
        'method': 'quantum_transport',
        'data': {
            'mu_left': 0.05 * 1.602176634e-19,
            'mu_right': 0.0
        },
        'parameters': {
            'temperature': 4.2,  # Liquid helium
            'n_energies': 400
        },
        'analysis': ['conductance']
    }

    result = quantum_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    # At low T, conductance should be quantized-like
    G_normalized = result.data['conductance_G0_units']
    assert 0 <= G_normalized < 5


def test_quantum_transport_bias_dependence(quantum_agent):
    """Test 25: Current-voltage characteristics."""
    biases = [0.0, 0.05, 0.1, 0.15]
    currents = []

    for bias in biases:
        input_data = {
            'method': 'quantum_transport',
            'data': {
                'mu_left': bias * 1.602176634e-19,
                'mu_right': 0.0
            },
            'parameters': {'temperature': 300.0},
            'analysis': ['current']
        }

        result = quantum_agent.execute(input_data)
        currents.append(result.data['current_A'])

    # Current should increase with bias
    assert currents[0] <= currents[1] <= currents[2] <= currents[3]


def test_quantum_lindblad_transport(quantum_agent):
    """Test 26: Open quantum system in transport setup."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {
            'n_dim': 2,
            'H': [[-0.5, 0.1], [0.1, 0.5]],  # Coupled system
            'rho0': [[0.5, 0], [0, 0.5]]
        },
        'parameters': {
            'time': 20.0,
            'decay_rate': 0.05
        },
        'analysis': ['evolution']
    }

    result = quantum_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'rho_evolution' in result.data


def test_quantum_driven_system(quantum_agent):
    """Test 27: Quantum driven system integration."""
    driven_params = {
        'n_dim': 2,
        'H': [[-1.0, 0], [0, 1.0]],
        'rho0': [[1, 0], [0, 0]]
    }

    result = quantum_agent.quantum_driven_system(driven_params)
    assert 'rho_evolution' in result
    assert 'entropy' in result


def test_quantum_thermodynamics_transport(quantum_agent):
    """Test 28: Quantum thermodynamics in transport context."""
    input_data = {
        'method': 'quantum_thermodynamics',
        'data': {
            'n_dim': 2
        },
        'parameters': {
            'time': 15.0
        },
        'analysis': ['work', 'heat']
    }

    result = quantum_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'total_work_J' in result.data
    assert 'total_heat_J' in result.data


def test_quantum_fluctuation_transport(quantum_agent):
    """Test 29: Quantum fluctuation theorem in transport."""
    input_data = {
        'method': 'quantum_fluctuation_theorem',
        'data': {
            'n_dim': 2,
            'H_initial': [[-0.5, 0], [0, 0.5]],
            'H_final': [[-1.5, 0], [0, 1.5]]
        },
        'parameters': {
            'temperature': 300.0,
            'n_realizations': 2000
        },
        'analysis': ['jarzynski']
    }

    result = quantum_agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'jarzynski_ratio' in result.data


def test_quantum_information_thermodynamics(quantum_agent):
    """Test 30: Quantum information thermodynamics (Maxwell demon)."""
    info_data = {
        'n_dim': 2
    }

    result = quantum_agent.quantum_information_thermodynamics(info_data)
    assert 'total_work_J' in result
    assert 'entropy' in result
    assert 'first_law_satisfied' in result


# ============================================================================
# Cross-Phase Integration Tests
# ============================================================================

def test_phase3_agents_compatibility(ld_agent, oc_agent, quantum_agent):
    """Bonus Test 31: All Phase 3 agents are compatible."""
    # Verify all agents initialize
    assert ld_agent.VERSION == "1.0.0"
    assert oc_agent.VERSION == "1.0.0"
    assert quantum_agent.VERSION == "1.0.0"


def test_phase3_workflow_ld_to_oc(ld_agent, oc_agent):
    """Bonus Test 32: Workflow from Large Deviation to Optimal Control."""
    # Step 1: Analyze rare events
    work_samples = np.random.exponential(10.0, 1000)
    ld_input = {
        'method': 'rare_event_sampling',
        'data': {'observable': work_samples.tolist()},
        'parameters': {},
        'analysis': ['rate_function']
    }
    ld_result = ld_agent.execute(ld_input)
    assert ld_result.status == AgentStatus.SUCCESS

    # Step 2: Design optimal protocol to avoid rare events
    oc_input = {
        'method': 'minimal_dissipation_protocol',
        'data': {'lambda_initial': 0.0, 'lambda_final': 1.0},
        'parameters': {'duration': 10.0},
        'analysis': ['dissipation']
    }
    oc_result = oc_agent.execute(oc_input)
    assert oc_result.status == AgentStatus.SUCCESS


def test_phase3_workflow_oc_to_quantum(oc_agent, quantum_agent):
    """Bonus Test 33: Workflow from Optimal Control to Quantum."""
    # Step 1: Design optimal classical protocol
    oc_input = {
        'method': 'shortcut_to_adiabaticity',
        'data': {'field_initial': 1.0, 'field_final': 2.0},
        'parameters': {'duration': 1.0},
        'analysis': ['cd_protocol']
    }
    oc_result = oc_agent.execute(oc_input)
    assert oc_result.status == AgentStatus.SUCCESS

    # Step 2: Verify with quantum simulation
    quantum_input = {
        'method': 'lindblad_master_equation',
        'data': {'n_dim': 2},
        'parameters': {'time': 1.0},
        'analysis': ['evolution']
    }
    quantum_result = quantum_agent.execute(quantum_input)
    assert quantum_result.status == AgentStatus.SUCCESS
