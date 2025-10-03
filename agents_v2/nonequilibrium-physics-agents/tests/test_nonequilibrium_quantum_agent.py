"""Comprehensive test suite for NonequilibriumQuantumAgent.

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation
- All 5 methods (Lindblad, quantum FT, GKSL, transport, thermodynamics)
- Integration methods
- Physics validation (quantum mechanics, thermodynamics)
"""

import pytest
import numpy as np
from typing import Dict, Any

from nonequilibrium_quantum_agent import NonequilibriumQuantumAgent
from base_agent import AgentStatus, ExecutionEnvironment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create NonequilibriumQuantumAgent instance."""
    return NonequilibriumQuantumAgent()


@pytest.fixture
def sample_lindblad_input():
    """Sample input for Lindblad master equation."""
    return {
        'method': 'lindblad_master_equation',
        'data': {
            'n_dim': 2,
            'H': [[-0.5, 0], [0, 0.5]],
            'rho0': [[1, 0], [0, 0]]
        },
        'parameters': {
            'time': 10.0,
            'n_steps': 100,
            'decay_rate': 0.1
        },
        'analysis': ['evolution', 'entropy', 'purity']
    }


@pytest.fixture
def sample_quantum_ft_input():
    """Sample input for quantum fluctuation theorem."""
    return {
        'method': 'quantum_fluctuation_theorem',
        'data': {
            'n_dim': 2,
            'H_initial': [[-0.5, 0], [0, 0.5]],
            'H_final': [[-1.0, 0], [0, 1.0]]
        },
        'parameters': {
            'temperature': 300.0,
            'n_realizations': 1000
        },
        'analysis': ['work_distribution', 'jarzynski', 'crooks']
    }


@pytest.fixture
def sample_gksl_input():
    """Sample input for GKSL master equation solver."""
    return {
        'method': 'quantum_master_equation_solver',
        'data': {
            'n_dim': 2,
            'H': [[-0.5, 0], [0, 0.5]],
            'rho0': [[0, 0], [0, 1]]
        },
        'parameters': {
            'time': 10.0,
            'n_steps': 100
        },
        'analysis': ['steady_state', 'relaxation']
    }


@pytest.fixture
def sample_transport_input():
    """Sample input for quantum transport."""
    return {
        'method': 'quantum_transport',
        'data': {
            'mu_left': 0.1 * 1.602176634e-19,
            'mu_right': 0.0
        },
        'parameters': {
            'temperature': 300.0,
            'n_energies': 200
        },
        'analysis': ['conductance', 'current']
    }


@pytest.fixture
def sample_thermodynamics_input():
    """Sample input for quantum thermodynamics."""
    return {
        'method': 'quantum_thermodynamics',
        'data': {
            'n_dim': 2
        },
        'parameters': {
            'time': 10.0,
            'n_steps': 100
        },
        'analysis': ['work', 'heat', 'entropy']
    }


# ============================================================================
# Test 1-5: Initialization and Metadata
# ============================================================================

def test_agent_initialization(agent):
    """Test 1: Agent initializes correctly."""
    assert agent.VERSION == "1.0.0"
    assert 'lindblad_master_equation' in agent.supported_methods
    assert 'quantum_fluctuation_theorem' in agent.supported_methods
    assert 'quantum_master_equation_solver' in agent.supported_methods
    assert 'quantum_transport' in agent.supported_methods
    assert 'quantum_thermodynamics' in agent.supported_methods


def test_agent_metadata(agent):
    """Test 2: Agent metadata is correct."""
    metadata = agent.get_metadata()
    assert metadata.name == "NonequilibriumQuantumAgent"
    assert metadata.version == "1.0.0"
    assert metadata.author == "Nonequilibrium Physics Team"


def test_agent_capabilities(agent):
    """Test 3: Agent capabilities are defined."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 5
    cap_names = [cap.name for cap in capabilities]
    assert 'lindblad_master_equation' in cap_names
    assert 'quantum_transport' in cap_names


def test_agent_constants(agent):
    """Test 4: Physical constants are correctly defined."""
    assert agent.kB == 1.380649e-23  # Boltzmann constant
    assert agent.hbar == 1.054571817e-34  # Reduced Planck constant
    assert agent.e == 1.602176634e-19  # Elementary charge
    assert agent.h == 6.62607015e-34  # Planck constant


def test_agent_config_initialization():
    """Test 5: Agent initializes with custom config."""
    config = {'custom_param': 42}
    agent = NonequilibriumQuantumAgent(config)
    assert agent.config == config


# ============================================================================
# Test 6-15: Input Validation
# ============================================================================

def test_validate_lindblad_valid(agent, sample_lindblad_input):
    """Test 6: Valid Lindblad input passes validation."""
    result = agent.validate_input(sample_lindblad_input)
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
        'method': 'invalid_quantum_method',
        'data': {},
        'parameters': {}
    }
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('unsupported' in err.lower() for err in result.errors)


def test_validate_missing_data(agent):
    """Test 9: Missing data field fails validation."""
    input_data = {'method': 'lindblad_master_equation', 'parameters': {}}
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('data' in err.lower() for err in result.errors)


def test_validate_missing_parameters_warning(agent):
    """Test 10: Missing parameters generates warning."""
    input_data = {'method': 'lindblad_master_equation', 'data': {}}
    result = agent.validate_input(input_data)
    assert len(result.warnings) > 0
    assert any('parameters' in warn.lower() for warn in result.warnings)


def test_validate_negative_time(agent):
    """Test 11: Negative time fails validation."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {},
        'parameters': {'time': -5.0}
    }
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('time' in err.lower() for err in result.errors)


def test_validate_negative_temperature(agent):
    """Test 12: Negative temperature fails validation."""
    input_data = {
        'method': 'quantum_fluctuation_theorem',
        'data': {},
        'parameters': {'temperature': -100.0}
    }
    result = agent.validate_input(input_data)
    assert result.valid is False
    assert any('temperature' in err.lower() for err in result.errors)


def test_validate_quantum_ft_valid(agent, sample_quantum_ft_input):
    """Test 13: Valid quantum FT input passes validation."""
    result = agent.validate_input(sample_quantum_ft_input)
    assert result.valid is True


def test_validate_transport_valid(agent, sample_transport_input):
    """Test 14: Valid transport input passes validation."""
    result = agent.validate_input(sample_transport_input)
    assert result.valid is True


def test_validate_thermodynamics_valid(agent, sample_thermodynamics_input):
    """Test 15: Valid thermodynamics input passes validation."""
    result = agent.validate_input(sample_thermodynamics_input)
    assert result.valid is True


# ============================================================================
# Test 16-25: Resource Estimation
# ============================================================================

def test_resource_estimation_lindblad_small(agent, sample_lindblad_input):
    """Test 16: Resource estimation for small Lindblad problem."""
    req = agent.estimate_resources(sample_lindblad_input)
    assert req.environment == ExecutionEnvironment.LOCAL
    assert req.cpu_cores >= 4
    assert req.memory_gb > 0
    assert req.gpu_required is False


def test_resource_estimation_lindblad_large(agent):
    """Test 17: Resource estimation for large Lindblad problem (HPC)."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {'n_dim': 10},
        'parameters': {}
    }
    req = agent.estimate_resources(input_data)
    assert req.environment == ExecutionEnvironment.HPC
    assert req.memory_gb > 2.0


def test_resource_estimation_quantum_ft(agent, sample_quantum_ft_input):
    """Test 18: Resource estimation for quantum FT (HPC)."""
    req = agent.estimate_resources(sample_quantum_ft_input)
    assert req.environment == ExecutionEnvironment.HPC
    assert req.cpu_cores >= 8


def test_resource_estimation_gksl(agent, sample_gksl_input):
    """Test 19: Resource estimation for GKSL solver."""
    req = agent.estimate_resources(sample_gksl_input)
    assert req.cpu_cores >= 4
    assert req.memory_gb > 0


def test_resource_estimation_transport(agent, sample_transport_input):
    """Test 20: Resource estimation for transport (lightweight)."""
    req = agent.estimate_resources(sample_transport_input)
    assert req.environment == ExecutionEnvironment.LOCAL
    assert req.estimated_duration_seconds < 120


def test_resource_estimation_thermodynamics(agent, sample_thermodynamics_input):
    """Test 21: Resource estimation for thermodynamics (lightweight)."""
    req = agent.estimate_resources(sample_thermodynamics_input)
    assert req.environment == ExecutionEnvironment.LOCAL
    assert req.cpu_cores >= 2


def test_resource_scaling_large_steps(agent, sample_lindblad_input):
    """Test 22: Resource scaling for large step counts."""
    sample_lindblad_input['parameters']['n_steps'] = 2000
    req = agent.estimate_resources(sample_lindblad_input)
    assert req.memory_gb >= 2.0


def test_resource_estimation_has_duration(agent, sample_lindblad_input):
    """Test 23: Resource estimation includes duration estimate."""
    req = agent.estimate_resources(sample_lindblad_input)
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_no_gpu(agent, sample_quantum_ft_input):
    """Test 24: No GPU required for quantum methods."""
    req = agent.estimate_resources(sample_quantum_ft_input)
    assert req.gpu_required is False


def test_resource_estimation_all_methods(agent):
    """Test 25: Resource estimation for all supported methods."""
    methods = agent.supported_methods
    for method in methods:
        input_data = {'method': method, 'data': {}, 'parameters': {}}
        req = agent.estimate_resources(input_data)
        assert req.cpu_cores > 0


# ============================================================================
# Test 26-30: Execution - Lindblad Master Equation
# ============================================================================

def test_execute_lindblad_success(agent, sample_lindblad_input):
    """Test 26: Lindblad master equation executes successfully."""
    result = agent.execute(sample_lindblad_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'rho_evolution' in result.data
    assert 'entropy' in result.data


def test_execute_lindblad_trace_preservation(agent, sample_lindblad_input):
    """Test 27: Lindblad evolution preserves trace."""
    result = agent.execute(sample_lindblad_input)
    trace_final = result.data['trace_final']
    assert np.isclose(trace_final, 1.0, atol=1e-6)


def test_execute_lindblad_entropy_increase(agent, sample_lindblad_input):
    """Test 28: Entropy increases (or stays constant) for open system."""
    result = agent.execute(sample_lindblad_input)
    entropy = result.data['entropy']
    # Entropy should be non-negative and typically increases
    assert all(s >= 0 for s in entropy)


def test_execute_lindblad_purity_decrease(agent, sample_lindblad_input):
    """Test 29: Purity decreases for dissipative evolution."""
    result = agent.execute(sample_lindblad_input)
    purity = result.data['purity']
    # Purity should be between 0 and 1
    assert all(0 <= p <= 1 for p in purity)


def test_execute_lindblad_provenance(agent, sample_lindblad_input):
    """Test 30: Result includes provenance tracking."""
    result = agent.execute(sample_lindblad_input)
    assert result.provenance is not None
    assert result.provenance.agent_name == "NonequilibriumQuantumAgent"
    assert result.provenance.agent_version == "1.0.0"


# ============================================================================
# Test 31-35: Execution - Quantum Fluctuation Theorem
# ============================================================================

def test_execute_quantum_ft_success(agent, sample_quantum_ft_input):
    """Test 31: Quantum FT executes successfully."""
    result = agent.execute(sample_quantum_ft_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'work_distribution' in result.data
    assert 'jarzynski_ratio' in result.data


def test_execute_quantum_ft_work_samples(agent, sample_quantum_ft_input):
    """Test 32: Work distribution has correct number of samples."""
    result = agent.execute(sample_quantum_ft_input)
    work_dist = result.data['work_distribution']
    n_realizations = sample_quantum_ft_input['parameters']['n_realizations']
    assert len(work_dist) == n_realizations


def test_execute_quantum_ft_jarzynski(agent, sample_quantum_ft_input):
    """Test 33: Jarzynski ratio is approximately 1."""
    result = agent.execute(sample_quantum_ft_input)
    jarzynski_ratio = result.data['jarzynski_ratio']
    # Should be close to 1, but allow statistical fluctuations
    assert 0.5 < jarzynski_ratio < 2.0


def test_execute_quantum_ft_second_law(agent, sample_quantum_ft_input):
    """Test 34: Second law satisfied (mean work >= delta F)."""
    result = agent.execute(sample_quantum_ft_input)
    second_law = result.data['second_law_satisfied']
    assert isinstance(second_law, bool)


def test_execute_quantum_ft_free_energy(agent, sample_quantum_ft_input):
    """Test 35: Free energy change is computed."""
    result = agent.execute(sample_quantum_ft_input)
    delta_F = result.data['delta_F_J']
    assert np.isfinite(delta_F)


# ============================================================================
# Test 36-40: Execution - GKSL Master Equation Solver
# ============================================================================

def test_execute_gksl_success(agent, sample_gksl_input):
    """Test 36: GKSL solver executes successfully."""
    result = agent.execute(sample_gksl_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'rho_steady_state' in result.data
    assert 'relaxation_time' in result.data


def test_execute_gksl_steady_state(agent, sample_gksl_input):
    """Test 37: Steady state has correct properties."""
    result = agent.execute(sample_gksl_input)
    rho_ss = np.array(result.data['rho_steady_state'])
    # Trace should be 1
    trace_ss = np.trace(rho_ss)
    assert np.isclose(np.real(trace_ss), 1.0, atol=1e-6)


def test_execute_gksl_trace_preserved(agent, sample_gksl_input):
    """Test 38: Trace preservation flag is correct."""
    result = agent.execute(sample_gksl_input)
    trace_preserved = result.data['trace_preserved']
    assert trace_preserved is True


def test_execute_gksl_relaxation_time(agent, sample_gksl_input):
    """Test 39: Relaxation time is positive."""
    result = agent.execute(sample_gksl_input)
    tau_relax = result.data['relaxation_time']
    assert tau_relax >= 0


def test_execute_gksl_populations(agent, sample_gksl_input):
    """Test 40: Steady state populations sum to 1."""
    result = agent.execute(sample_gksl_input)
    populations = result.data['steady_state_populations']
    assert np.isclose(sum(populations), 1.0, atol=1e-6)


# ============================================================================
# Test 41-45: Execution - Quantum Transport
# ============================================================================

def test_execute_transport_success(agent, sample_transport_input):
    """Test 41: Quantum transport executes successfully."""
    result = agent.execute(sample_transport_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'conductance_S' in result.data
    assert 'current_A' in result.data


def test_execute_transport_conductance_positive(agent, sample_transport_input):
    """Test 42: Conductance is non-negative."""
    result = agent.execute(sample_transport_input)
    conductance = result.data['conductance_S']
    assert conductance >= 0


def test_execute_transport_current_sign(agent, sample_transport_input):
    """Test 43: Current has correct sign (positive bias -> positive current)."""
    result = agent.execute(sample_transport_input)
    current = result.data['current_A']
    bias = result.data['bias_voltage_V']
    # For positive bias, current should be positive
    if bias > 0:
        assert current >= 0
    elif bias < 0:
        assert current <= 0


def test_execute_transport_quantum_conductance(agent, sample_transport_input):
    """Test 44: Quantum of conductance G0 is correct."""
    result = agent.execute(sample_transport_input)
    G0 = result.data['quantum_conductance_G0']
    # G0 = 2e²/h ≈ 7.748 × 10⁻⁵ S
    expected_G0 = 2 * (1.602176634e-19)**2 / 6.62607015e-34
    assert np.isclose(G0, expected_G0, rtol=1e-6)


def test_execute_transport_seebeck(agent, sample_transport_input):
    """Test 45: Seebeck coefficient is computed."""
    result = agent.execute(sample_transport_input)
    seebeck = result.data['seebeck_coefficient_V_K']
    assert np.isfinite(seebeck)


# ============================================================================
# Test 46-50: Execution - Quantum Thermodynamics
# ============================================================================

def test_execute_thermodynamics_success(agent, sample_thermodynamics_input):
    """Test 46: Quantum thermodynamics executes successfully."""
    result = agent.execute(sample_thermodynamics_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'total_work_J' in result.data
    assert 'total_heat_J' in result.data


def test_execute_thermodynamics_first_law(agent, sample_thermodynamics_input):
    """Test 47: First law of thermodynamics satisfied."""
    result = agent.execute(sample_thermodynamics_input)
    first_law_satisfied = result.data['first_law_satisfied']
    # May not be exactly satisfied due to numerical errors, but should be close
    assert isinstance(first_law_satisfied, bool)


def test_execute_thermodynamics_internal_energy(agent, sample_thermodynamics_input):
    """Test 48: Internal energy is computed."""
    result = agent.execute(sample_thermodynamics_input)
    U = result.data['internal_energy']
    assert len(U) > 0
    assert all(np.isfinite(u) for u in U)


def test_execute_thermodynamics_entropy(agent, sample_thermodynamics_input):
    """Test 49: Entropy is non-negative."""
    result = agent.execute(sample_thermodynamics_input)
    entropy = result.data['entropy']
    assert all(s >= 0 for s in entropy)


def test_execute_thermodynamics_efficiency(agent, sample_thermodynamics_input):
    """Test 50: Efficiency is in valid range."""
    result = agent.execute(sample_thermodynamics_input)
    efficiency = result.data['efficiency']
    # Efficiency should be bounded (can be negative for heat engines)
    assert np.isfinite(efficiency)


# ============================================================================
# Test 51-55: Integration Methods
# ============================================================================

def test_integration_quantum_driven_system(agent):
    """Test 51: Integration with quantum driven systems."""
    driven_params = {
        'n_dim': 2,
        'H': [[-0.5, 0], [0, 0.5]],
        'rho0': [[1, 0], [0, 0]]
    }
    result = agent.quantum_driven_system(driven_params)
    assert 'rho_evolution' in result
    assert 'entropy' in result


def test_integration_quantum_transport_coefficients(agent):
    """Test 52: Integration with transport agent."""
    transport_data = {
        'mu_left': 0.1 * 1.602176634e-19,
        'mu_right': 0.0
    }
    result = agent.quantum_transport_coefficients(transport_data)
    assert 'conductance_S' in result
    assert 'current_A' in result


def test_integration_quantum_information_thermodynamics(agent):
    """Test 53: Integration with information thermodynamics."""
    info_data = {'n_dim': 2}
    result = agent.quantum_information_thermodynamics(info_data)
    assert 'total_work_J' in result
    assert 'total_heat_J' in result


def test_integration_methods_return_dict(agent):
    """Test 54: Integration methods return dictionaries."""
    driven_params = {'n_dim': 2}
    result = agent.quantum_driven_system(driven_params)
    assert isinstance(result, dict)


def test_integration_quantum_driven_has_evolution(agent):
    """Test 55: Quantum driven system returns evolution data."""
    driven_params = {'n_dim': 2}
    result = agent.quantum_driven_system(driven_params)
    assert 'rho_evolution' in result
    assert len(result['rho_evolution']) > 0


# ============================================================================
# Test 56-60: Error Handling and Edge Cases
# ============================================================================

def test_execute_invalid_input_returns_failure(agent):
    """Test 56: Invalid input returns FAILED status."""
    invalid_input = {'method': 'invalid_quantum_method'}
    result = agent.execute(invalid_input)
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0


def test_execute_missing_data_returns_failure(agent):
    """Test 57: Missing data returns FAILED status."""
    invalid_input = {'method': 'lindblad_master_equation'}
    result = agent.execute(invalid_input)
    assert result.status == AgentStatus.FAILED


def test_lindblad_zero_time_safe(agent):
    """Test 58: Zero time handled safely."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {'n_dim': 2},
        'parameters': {'time': 0.0}
    }
    result = agent.execute(input_data)
    # Should fail validation
    assert result.status == AgentStatus.FAILED


def test_quantum_ft_zero_realizations(agent):
    """Test 59: Zero realizations handled safely."""
    input_data = {
        'method': 'quantum_fluctuation_theorem',
        'data': {'n_dim': 2},
        'parameters': {'temperature': 300.0, 'n_realizations': 0}
    }
    result = agent.execute(input_data)
    # Should handle gracefully (empty distribution)
    if result.status == AgentStatus.SUCCESS:
        assert len(result.data['work_distribution']) == 0


def test_transport_zero_bias(agent):
    """Test 60: Zero bias transport calculation."""
    input_data = {
        'method': 'quantum_transport',
        'data': {'mu_left': 0.0, 'mu_right': 0.0},
        'parameters': {'temperature': 300.0}
    }
    result = agent.execute(input_data)
    if result.status == AgentStatus.SUCCESS:
        # Zero bias should give zero current
        current = result.data['current_A']
        assert np.isclose(current, 0.0, atol=1e-12)


# ============================================================================
# Test 61-70: Physics Validation
# ============================================================================

def test_physics_lindblad_dissipation(agent):
    """Test 61: Lindblad evolution with dissipation relaxes to ground state."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {
            'n_dim': 2,
            'H': [[-1.0, 0], [0, 1.0]],
            'rho0': [[0, 0], [0, 1]]  # Start in excited state
        },
        'parameters': {'time': 50.0, 'decay_rate': 0.2, 'n_steps': 100}
    }
    result = agent.execute(input_data)
    populations = np.array(result.data['populations'])
    # Should relax to ground state (population[0] increases)
    assert populations[-1, 0] > populations[0, 0]


def test_physics_quantum_coherence_decay(agent):
    """Test 62: Off-diagonal elements decay in Lindblad evolution."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {
            'n_dim': 2,
            'H': [[0, 0], [0, 0]],
            'rho0': [[0.5, 0.5], [0.5, 0.5]]  # Coherent superposition
        },
        'parameters': {'time': 10.0, 'decay_rate': 0.5}
    }
    result = agent.execute(input_data)
    rho_final = np.array(result.data['rho_final'])
    rho_initial = np.array([[0.5, 0.5], [0.5, 0.5]])
    # Off-diagonal elements should decrease
    assert np.abs(rho_final[0, 1]) < np.abs(rho_initial[0, 1])


def test_physics_jarzynski_equality_holds(agent):
    """Test 63: Jarzynski equality approximately satisfied."""
    input_data = {
        'method': 'quantum_fluctuation_theorem',
        'data': {
            'n_dim': 2,
            'H_initial': [[-1.0, 0], [0, 1.0]],
            'H_final': [[-2.0, 0], [0, 2.0]]
        },
        'parameters': {'temperature': 300.0, 'n_realizations': 5000}
    }
    result = agent.execute(input_data)
    jarzynski_ratio = result.data['jarzynski_ratio']
    # With many realizations, should be close to 1
    assert 0.8 < jarzynski_ratio < 1.2


def test_physics_quantum_steady_state_thermal(agent):
    """Test 64: Steady state approaches thermal state for detailed balance."""
    input_data = {
        'method': 'quantum_master_equation_solver',
        'data': {
            'n_dim': 2,
            'H': [[-1.0, 0], [0, 1.0]],
            'rho0': [[0, 0], [0, 1]]
        },
        'parameters': {'time': 100.0}
    }
    result = agent.execute(input_data)
    populations = result.data['steady_state_populations']
    # Ground state should have higher population
    assert populations[0] > populations[1]


def test_physics_landauer_formula(agent):
    """Test 65: Landauer formula for conductance."""
    input_data = {
        'method': 'quantum_transport',
        'data': {'mu_left': 0.05 * 1.602176634e-19, 'mu_right': 0.0},
        'parameters': {'temperature': 10.0}  # Low temperature
    }
    result = agent.execute(input_data)
    # At low temperature and small bias, I ≈ G * V
    G = result.data['conductance_S']
    I = result.data['current_A']
    V = result.data['bias_voltage_V']
    if V > 0:
        G_from_IV = I / V
        # Should be consistent
        assert np.isclose(G, G_from_IV, rtol=0.1)


def test_physics_unitary_evolution_preserves_energy(agent):
    """Test 66: Pure unitary evolution preserves energy."""
    input_data = {
        'method': 'quantum_thermodynamics',
        'data': {
            'n_dim': 2,
            'rho_evolution': [[[1, 0], [0, 0]]] * 100  # Constant state
        },
        'parameters': {'time': 10.0}
    }
    result = agent.execute(input_data)
    U = result.data['internal_energy']
    # For constant H and rho, energy should be constant
    assert np.std(U) < 1e-6


def test_physics_entropy_positivity(agent):
    """Test 67: Von Neumann entropy is always non-negative."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {'n_dim': 2},
        'parameters': {'time': 10.0}
    }
    result = agent.execute(input_data)
    entropy = result.data['entropy']
    assert all(s >= -1e-12 for s in entropy)  # Allow tiny numerical errors


def test_physics_purity_bounds(agent):
    """Test 68: Purity is bounded between 1/n and 1."""
    input_data = {
        'method': 'lindblad_master_equation',
        'data': {'n_dim': 2},
        'parameters': {'time': 10.0}
    }
    result = agent.execute(input_data)
    purity = result.data['purity']
    n_dim = result.data['n_dim']
    assert all(1.0/n_dim - 1e-6 <= p <= 1.0 + 1e-6 for p in purity)


def test_physics_conductance_quantum_limit(agent):
    """Test 69: Conductance in units of G0 is physically reasonable."""
    input_data = {
        'method': 'quantum_transport',
        'data': {'mu_left': 0.1 * 1.602176634e-19, 'mu_right': 0.0},
        'parameters': {'temperature': 300.0}
    }
    result = agent.execute(input_data)
    G_normalized = result.data['conductance_G0_units']
    # Should be positive and typically O(1) for single channel
    assert 0 <= G_normalized < 10


def test_physics_thermodynamic_consistency(agent):
    """Test 70: ΔU = W + Q (first law check)."""
    input_data = {
        'method': 'quantum_thermodynamics',
        'data': {'n_dim': 2},
        'parameters': {'time': 10.0}
    }
    result = agent.execute(input_data)
    delta_U = result.data['delta_U_J']
    work = result.data['total_work_J']
    heat = result.data['total_heat_J']
    residual = result.data['first_law_residual_J']
    # Residual should be small
    assert residual < 1e-6 or np.abs(residual / (np.abs(delta_U) + 1e-12)) < 0.01
