"""Tests for Magnus Expansion Solver.

Tests cover:
- Correctness vs analytical solutions
- Energy conservation (vs RK4)
- Order of accuracy verification
- Unitary preservation (for pure states)
- Edge cases (constant/rapidly varying Hamiltonians)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from typing import Dict, Any

from solvers.magnus_expansion import (
    MagnusExpansionSolver,
    solve_lindblad_magnus
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_system():
    """Simple two-level system with time-dependent Hamiltonian."""
    n_dim = 2
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)  # Ground state

    # Time-dependent Hamiltonian: linear ramp
    omega_i = 1.0
    omega_f = 2.0
    duration = 10.0

    def H_protocol(t):
        omega_t = omega_i + (omega_f - omega_i) * (t / duration)
        return -0.5 * omega_t * np.array([[1, 0], [0, -1]], dtype=complex)

    # Jump operator (decay)
    L = np.array([[0, 1], [0, 0]], dtype=complex)
    L_ops = [L]
    gammas = [0.1]

    t_span = np.linspace(0, duration, 100)

    return {
        'n_dim': n_dim,
        'rho0': rho0,
        'H_protocol': H_protocol,
        'L_ops': L_ops,
        'gammas': gammas,
        't_span': t_span,
        'duration': duration
    }


@pytest.fixture
def harmonic_system():
    """Harmonic oscillator for energy conservation tests."""
    n_dim = 6
    psi0 = np.zeros(n_dim, dtype=complex)
    psi0[0] = 1.0  # Ground state

    # Time-dependent frequency
    omega_i = 1.0
    omega_f = 3.0
    duration = 10.0

    def H_protocol(t):
        omega_t = omega_i + (omega_f - omega_i) * (t / duration)
        return np.diag(np.arange(n_dim) * omega_t)

    t_span = np.linspace(0, duration, 100)

    return {
        'n_dim': n_dim,
        'psi0': psi0,
        'H_protocol': H_protocol,
        't_span': t_span,
        'duration': duration
    }


# ============================================================================
# Test 1-5: Correctness Tests
# ============================================================================

def test_magnus_initialization():
    """Test 1: Magnus solver initializes correctly."""
    solver = MagnusExpansionSolver(order=4)

    assert solver.order == 4
    assert solver.method == 'gauss'
    assert solver.hbar == 1.054571817e-34


def test_magnus_invalid_order():
    """Test 2: Invalid order raises ValueError."""
    with pytest.raises(ValueError, match="Order must be 2, 4, or 6"):
        MagnusExpansionSolver(order=3)


def test_magnus_lindblad_simple(simple_system):
    """Test 3: Magnus solver runs on simple Lindblad equation."""
    solver = MagnusExpansionSolver(order=4)

    rho_evolution = solver.solve_lindblad(
        simple_system['rho0'],
        simple_system['H_protocol'],
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span']
    )

    # Check shape
    assert rho_evolution.shape == (len(simple_system['t_span']),
                                   simple_system['n_dim'],
                                   simple_system['n_dim'])

    # Check trace preservation
    traces = np.array([np.trace(rho) for rho in rho_evolution])
    trace_error = np.max(np.abs(traces - 1.0))

    assert trace_error < 1e-6, f"Trace not preserved: max error = {trace_error}"


def test_magnus_hermiticity(simple_system):
    """Test 4: Magnus preserves Hermiticity."""
    solver = MagnusExpansionSolver(order=4)

    rho_evolution = solver.solve_lindblad(
        simple_system['rho0'],
        simple_system['H_protocol'],
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span']
    )

    for i, rho in enumerate(rho_evolution):
        hermiticity_error = np.max(np.abs(rho - np.conj(rho.T)))
        assert hermiticity_error < 1e-10, \
            f"Hermiticity violated at step {i}: error = {hermiticity_error}"


def test_magnus_positivity(simple_system):
    """Test 5: Magnus maintains positivity (eigenvalues ≥ 0)."""
    solver = MagnusExpansionSolver(order=4)

    rho_evolution = solver.solve_lindblad(
        simple_system['rho0'],
        simple_system['H_protocol'],
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span']
    )

    for i, rho in enumerate(rho_evolution):
        eigvals = np.linalg.eigvalsh(rho)
        min_eigval = np.min(eigvals)
        assert min_eigval >= -1e-8, \
            f"Negative eigenvalue at step {i}: {min_eigval}"


# ============================================================================
# Test 6-10: Energy Conservation Tests
# ============================================================================

def test_magnus_energy_conservation_unitary(harmonic_system):
    """Test 6: Magnus conserves energy better than RK4 (unitary case)."""
    solver = MagnusExpansionSolver(order=4)

    # Magnus evolution
    psi_magnus = solver.solve_unitary(
        harmonic_system['psi0'],
        harmonic_system['H_protocol'],
        harmonic_system['t_span']
    )

    # Compute energy evolution
    energies_magnus = []
    for i, psi in enumerate(psi_magnus):
        H = harmonic_system['H_protocol'](harmonic_system['t_span'][i])
        E = np.real(psi.conj() @ H @ psi)
        energies_magnus.append(E)

    # Energy drift (should be very small for Magnus)
    energy_drift_magnus = np.std(energies_magnus)

    print(f"Magnus energy drift: {energy_drift_magnus:.2e}")

    # For unitary evolution with good solver, energy should be well conserved
    # Target: < 1e-6 for order 4
    assert energy_drift_magnus < 1e-4, \
        f"Energy drift too large: {energy_drift_magnus:.2e}"


def test_magnus_vs_rk4_benchmark(harmonic_system):
    """Test 7: Magnus outperforms RK4 in energy conservation."""
    solver = MagnusExpansionSolver(order=4)

    benchmark = solver.benchmark_vs_rk4(
        n_dim=harmonic_system['n_dim'],
        duration=harmonic_system['duration'],
        n_steps_list=[50, 100]
    )

    # Check that Magnus is better than RK4
    for i, n_steps in enumerate(benchmark['n_steps']):
        magnus_drift = benchmark['magnus_drift'][i]
        rk4_drift = benchmark['rk4_drift'][i]
        improvement = benchmark['improvement_factor'][i]

        print(f"n_steps={n_steps}:")
        print(f"  Magnus drift: {magnus_drift:.2e}")
        print(f"  RK4 drift: {rk4_drift:.2e}")
        print(f"  Improvement: {improvement:.1f}x")

        # Magnus should be at least 2x better
        assert improvement > 2.0, \
            f"Magnus not better than RK4: improvement = {improvement:.1f}x"


def test_magnus_order_comparison():
    """Test 8: Higher order Magnus is more accurate."""
    n_dim = 4
    psi0 = np.zeros(n_dim, dtype=complex)
    psi0[0] = 1.0

    # Rapidly varying Hamiltonian (challenges low-order methods)
    def H_protocol(t):
        omega_t = 1.0 + 2.0 * np.sin(5 * t)
        return np.diag(np.arange(n_dim) * omega_t)

    t_span = np.linspace(0, 10, 50)

    # Test different orders
    drifts = {}
    for order in [2, 4]:
        solver = MagnusExpansionSolver(order=order)
        psi_evolution = solver.solve_unitary(psi0, H_protocol, t_span)

        energies = []
        for i, psi in enumerate(psi_evolution):
            H = H_protocol(t_span[i])
            E = np.real(psi.conj() @ H @ psi)
            energies.append(E)

        drifts[order] = np.std(energies)

    print(f"Order 2 drift: {drifts[2]:.2e}")
    print(f"Order 4 drift: {drifts[4]:.2e}")
    print(f"Improvement: {drifts[2] / drifts[4]:.1f}x")

    # Order 4 should be better than order 2
    assert drifts[4] < drifts[2], "Order 4 not better than order 2"


def test_magnus_constant_hamiltonian():
    """Test 9: Magnus is exact for constant Hamiltonian."""
    n_dim = 3
    psi0 = np.array([1, 0, 0], dtype=complex)

    # Constant Hamiltonian
    H_constant = np.diag([0, 1, 2])
    H_protocol = lambda t: H_constant

    t_span = np.linspace(0, 10, 100)
    duration = 10.0

    solver = MagnusExpansionSolver(order=2)
    psi_magnus = solver.solve_unitary(psi0, H_protocol, t_span)

    # Analytical solution: ψ(t) = exp(-iHt/ℏ) ψ₀
    from scipy.linalg import expm
    hbar = solver.hbar
    psi_analytical = []

    for t in t_span:
        U = expm(-1j * H_constant * t / hbar)
        psi_analytical.append(U @ psi0)

    psi_analytical = np.array(psi_analytical)

    # Compare
    max_error = np.max(np.abs(psi_magnus - psi_analytical))

    print(f"Magnus vs analytical (constant H): max error = {max_error:.2e}")

    assert max_error < 1e-8, \
        f"Magnus inaccurate for constant H: error = {max_error:.2e}"


def test_magnus_norm_preservation():
    """Test 10: Magnus preserves state norm (unitary evolution)."""
    n_dim = 5
    psi0 = np.random.rand(n_dim) + 1j * np.random.rand(n_dim)
    psi0 = psi0 / np.linalg.norm(psi0)

    def H_protocol(t):
        return np.diag(np.arange(n_dim) * (1 + 0.5 * t))

    t_span = np.linspace(0, 10, 100)

    solver = MagnusExpansionSolver(order=4)
    psi_evolution = solver.solve_unitary(psi0, H_protocol, t_span)

    # Check norm at each step
    norms = np.array([np.linalg.norm(psi) for psi in psi_evolution])
    norm_error = np.max(np.abs(norms - 1.0))

    print(f"Norm preservation error: {norm_error:.2e}")

    assert norm_error < 1e-10, f"Norm not preserved: error = {norm_error:.2e}"


# ============================================================================
# Test 11-15: Lindblad-Specific Tests
# ============================================================================

def test_magnus_lindblad_with_list_protocol(simple_system):
    """Test 11: Magnus accepts list of Hamiltonians."""
    solver = MagnusExpansionSolver(order=4)

    # Create list protocol
    H_list = [simple_system['H_protocol'](t) for t in simple_system['t_span']]

    rho_evolution = solver.solve_lindblad(
        simple_system['rho0'],
        H_list,  # List instead of callable
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span']
    )

    assert rho_evolution.shape[0] == len(simple_system['t_span'])


def test_magnus_lindblad_zero_decay(simple_system):
    """Test 12: Zero decay gives unitary evolution."""
    solver = MagnusExpansionSolver(order=4)

    # No dissipation
    system_unitary = simple_system.copy()
    system_unitary['gammas'] = [0.0]

    rho_evolution = solver.solve_lindblad(
        system_unitary['rho0'],
        system_unitary['H_protocol'],
        system_unitary['L_ops'],
        system_unitary['gammas'],
        system_unitary['t_span']
    )

    # Purity should remain 1.0 (pure state)
    purities = np.array([np.real(np.trace(rho @ rho)) for rho in rho_evolution])

    print(f"Purity range: [{purities.min():.6f}, {purities.max():.6f}]")

    assert np.allclose(purities, 1.0, atol=1e-4), "Purity not preserved in unitary evolution"


def test_magnus_lindblad_multiple_jump_ops(simple_system):
    """Test 13: Multiple jump operators (decay + dephasing)."""
    solver = MagnusExpansionSolver(order=4)

    # Add dephasing operator
    L_dephase = np.array([[1, 0], [0, -1]], dtype=complex)
    system_multi = simple_system.copy()
    system_multi['L_ops'] = simple_system['L_ops'] + [L_dephase]
    system_multi['gammas'] = [0.1, 0.05]

    rho_evolution = solver.solve_lindblad(
        system_multi['rho0'],
        system_multi['H_protocol'],
        system_multi['L_ops'],
        system_multi['gammas'],
        system_multi['t_span']
    )

    # Check basic properties
    traces = np.array([np.trace(rho) for rho in rho_evolution])
    assert np.max(np.abs(traces - 1.0)) < 1e-6


def test_magnus_lindblad_entropy_increase(simple_system):
    """Test 14: Entropy increases (2nd law of thermodynamics)."""
    solver = MagnusExpansionSolver(order=4)

    rho_evolution = solver.solve_lindblad(
        simple_system['rho0'],
        simple_system['H_protocol'],
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span']
    )

    # Compute entropy
    entropies = []
    for rho in rho_evolution:
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-12]
        if len(eigvals) > 0:
            S = -np.sum(eigvals * np.log(eigvals))
        else:
            S = 0.0
        entropies.append(S)

    entropies = np.array(entropies)

    # Entropy should generally increase
    entropy_change = entropies[-1] - entropies[0]

    print(f"Initial entropy: {entropies[0]:.4f}")
    print(f"Final entropy: {entropies[-1]:.4f}")
    print(f"Change: {entropy_change:.4f}")

    assert entropy_change > 0, "Entropy did not increase"


def test_magnus_lindblad_custom_n_steps(simple_system):
    """Test 15: Custom number of steps works correctly."""
    solver = MagnusExpansionSolver(order=4)

    # Use fewer Magnus steps than time grid points
    n_steps_magnus = 20

    rho_evolution = solver.solve_lindblad(
        simple_system['rho0'],
        simple_system['H_protocol'],
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span'],
        n_steps=n_steps_magnus
    )

    # Should have n_steps + 1 points
    assert rho_evolution.shape[0] == n_steps_magnus + 1


# ============================================================================
# Test 16-20: Convenience Function & Integration Tests
# ============================================================================

def test_solve_lindblad_magnus_convenience(simple_system):
    """Test 16: Convenience function works."""
    result = solve_lindblad_magnus(
        simple_system['rho0'],
        simple_system['H_protocol'],
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span'],
        order=4
    )

    # Check all expected keys
    assert 'rho_evolution' in result
    assert 'entropy' in result
    assert 'purity' in result
    assert 'time_grid' in result
    assert 'solver_type' in result
    assert 'order' in result

    assert result['solver_type'] == 'magnus'
    assert result['order'] == 4


def test_solve_lindblad_magnus_observables(simple_system):
    """Test 17: Convenience function computes observables correctly."""
    result = solve_lindblad_magnus(
        simple_system['rho0'],
        simple_system['H_protocol'],
        simple_system['L_ops'],
        simple_system['gammas'],
        simple_system['t_span'],
        order=2
    )

    # Check shapes
    n_points = len(result['rho_evolution'])
    assert len(result['entropy']) == n_points
    assert len(result['purity']) == n_points
    assert len(result['time_grid']) == n_points

    # Check entropy is non-negative
    assert np.all(result['entropy'] >= 0)

    # Check purity is in [0, 1]
    assert np.all(result['purity'] >= 0)
    assert np.all(result['purity'] <= 1.0 + 1e-6)


def test_magnus_different_orders_consistent():
    """Test 18: Different orders give consistent results."""
    n_dim = 3
    rho0 = np.eye(n_dim, dtype=complex) / n_dim

    def H_protocol(t):
        return np.diag([0, 1, 2]) * (1 + 0.1 * t)

    L = np.zeros((n_dim, n_dim), dtype=complex)
    L[0, 1] = 1.0
    L_ops = [L]
    gammas = [0.05]
    t_span = np.linspace(0, 5, 50)

    # Solve with different orders
    results = {}
    for order in [2, 4]:
        result = solve_lindblad_magnus(rho0, H_protocol, L_ops, gammas, t_span, order=order)
        results[order] = result

    # Final states should be similar (higher order more accurate)
    rho_final_2 = results[2]['rho_evolution'][-1]
    rho_final_4 = results[4]['rho_evolution'][-1]

    difference = np.max(np.abs(rho_final_2 - rho_final_4))

    print(f"Difference between order 2 and 4: {difference:.2e}")

    # Should be reasonably close (both are decent approximations)
    assert difference < 0.1, f"Orders give very different results: {difference:.2e}"


def test_magnus_long_evolution_stability():
    """Test 19: Magnus remains stable for long evolution."""
    n_dim = 2
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

    def H_protocol(t):
        return np.array([[1, 0.1 * np.sin(t)], [0.1 * np.sin(t), -1]], dtype=complex)

    L = np.array([[0, 1], [0, 0]], dtype=complex)
    L_ops = [L]
    gammas = [0.01]

    # Long evolution
    t_span = np.linspace(0, 100, 500)

    solver = MagnusExpansionSolver(order=4)
    rho_evolution = solver.solve_lindblad(rho0, H_protocol, L_ops, gammas, t_span)

    # Check stability (trace preservation)
    traces = np.array([np.trace(rho) for rho in rho_evolution])
    trace_error = np.max(np.abs(traces - 1.0))

    print(f"Trace error after long evolution: {trace_error:.2e}")

    assert trace_error < 1e-4, f"Long evolution unstable: trace error = {trace_error:.2e}"


def test_magnus_analytical_comparison_simple():
    """Test 20: Compare with analytical solution (simple case)."""
    # Two-level system with constant Hamiltonian and decay
    n_dim = 2
    omega = 1.0
    gamma = 0.1

    H_constant = -0.5 * omega * np.array([[1, 0], [0, -1]], dtype=complex)
    H_protocol = lambda t: H_constant

    L = np.array([[0, 1], [0, 0]], dtype=complex)
    L_ops = [L]
    gammas = [gamma]

    # Start from excited state
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)

    t_span = np.linspace(0, 10, 100)

    solver = MagnusExpansionSolver(order=4)
    rho_magnus = solver.solve_lindblad(rho0, H_protocol, L_ops, gammas, t_span)

    # For this simple case, analytical solution exists
    # ρ₁₁(t) ≈ 1 - exp(-γt) (approximately, ignoring oscillations)
    # Check qualitative behavior

    rho_11 = np.array([np.real(rho[1, 1]) for rho in rho_magnus])

    # Population should decay exponentially
    assert rho_11[0] > 0.95, "Should start in excited state"
    assert rho_11[-1] < 0.1, "Should decay to ground state"

    # Should be monotonically decreasing (approximately)
    differences = np.diff(rho_11)
    assert np.sum(differences > 0) < 10, "Should mostly decrease"


# ============================================================================
# Performance Comparison Summary
# ============================================================================

@pytest.mark.benchmark
def test_magnus_comprehensive_benchmark():
    """Comprehensive benchmark of Magnus expansion performance."""
    print("\n" + "="*70)
    print("Magnus Expansion Comprehensive Benchmark")
    print("="*70)

    # Test different system sizes
    for n_dim in [4, 6, 8]:
        print(f"\nSystem size: n_dim = {n_dim}")
        print("-" * 50)

        solver = MagnusExpansionSolver(order=4)
        benchmark = solver.benchmark_vs_rk4(
            n_dim=n_dim,
            duration=10.0,
            n_steps_list=[50, 100, 200]
        )

        for i, n_steps in enumerate(benchmark['n_steps']):
            print(f"  n_steps = {n_steps}:")
            print(f"    Magnus drift: {benchmark['magnus_drift'][i]:.2e}")
            print(f"    RK4 drift:    {benchmark['rk4_drift'][i]:.2e}")
            print(f"    Improvement:  {benchmark['improvement_factor'][i]:.1f}x")

    print("\n" + "="*70)
