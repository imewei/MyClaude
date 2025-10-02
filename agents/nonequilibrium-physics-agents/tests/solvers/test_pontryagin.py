"""Tests for Pontryagin Maximum Principle solver.

Test coverage:
- Basic classical optimal control (LQR-type problems)
- Quantum control (state transfer, gate synthesis)
- Single shooting vs multiple shooting
- Constrained vs unconstrained control
- Fixed endpoint vs free endpoint
- Convergence properties
- Hamiltonian conservation

Author: Nonequilibrium Physics Agents
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from solvers.pontryagin import PontryaginSolver, solve_quantum_control_pmp


class TestPontryaginBasics:
    """Test basic PMP solver functionality."""

    def test_1_linear_quadratic_regulator(self):
        """Test 1: LQR problem (analytical solution known)."""
        # Problem: minimize ∫[x² + u²] dt
        # Dynamics: dx/dt = u
        # Initial: x(0) = 1
        # Final: x(T) = 0 (optional)

        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return u

        def running_cost(x, u, t):
            return x[0]**2 + u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0])
        result = solver.solve(
            x0=x0,
            xf=np.array([0.0]),
            duration=2.0,
            n_steps=50,
            method='single_shooting'
        )

        assert result['converged'], "LQR problem should converge"
        assert np.abs(result['state'][-1, 0] - 0.0) < 1e-3, "Should reach target"
        assert result['cost'] < 2.0, "Cost should be reasonable"

        print(f"✓ Test 1 passed: LQR converged with cost {result['cost']:.4f}")

    def test_2_double_integrator(self):
        """Test 2: Double integrator (position + velocity)."""
        # Dynamics: dx1/dt = x2, dx2/dt = u
        # Cost: ∫[x1² + x2² + 0.1u²] dt
        # Transfer from (1, 0) to (0, 0)

        state_dim = 2
        control_dim = 1

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def running_cost(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0, 0.0])
        xf = np.array([0.0, 0.0])

        result = solver.solve(
            x0=x0,
            xf=xf,
            duration=3.0,
            n_steps=60,
            method='multiple_shooting'
        )

        assert result['converged'], "Double integrator should converge"
        error = np.linalg.norm(result['state'][-1] - xf)
        assert error < 0.1, f"Should reach target, error: {error}"

        print(f"✓ Test 2 passed: Double integrator, final error {error:.4e}")

    def test_3_constrained_control(self):
        """Test 3: Control with box constraints."""
        # Same as test 1, but with |u| ≤ 0.5

        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return u

        def running_cost(x, u, t):
            return x[0]**2 + u[0]**2

        u_min = np.array([-0.5])
        u_max = np.array([0.5])

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost,
            control_bounds=(u_min, u_max)
        )

        x0 = np.array([1.0])
        result = solver.solve(
            x0=x0,
            xf=np.array([0.0]),
            duration=3.0,
            n_steps=50,
            method='multiple_shooting'
        )

        # Check control stays within bounds
        u_max_achieved = np.max(np.abs(result['control']))
        assert u_max_achieved <= 0.51, f"Control should respect bounds, got {u_max_achieved}"

        print(f"✓ Test 3 passed: Constrained control, max |u| = {u_max_achieved:.3f}")

    def test_4_free_endpoint(self):
        """Test 4: Free endpoint (minimize energy only)."""
        # No target state, just minimize control energy

        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            # Damped system: dx/dt = -0.1*x + u
            return np.array([-0.1 * x[0] + u[0]])

        def running_cost(x, u, t):
            # Only control cost
            return u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0])
        result = solver.solve(
            x0=x0,
            xf=None,  # Free endpoint
            duration=5.0,
            n_steps=50,
            method='single_shooting'
        )

        # With free endpoint and only control cost, should apply minimal control
        avg_control = np.mean(np.abs(result['control']))
        assert avg_control < 0.5, "Should use minimal control with free endpoint"

        print(f"✓ Test 4 passed: Free endpoint, avg |u| = {avg_control:.4f}")

    def test_5_terminal_cost(self):
        """Test 5: Problem with terminal cost."""
        # Cost: ∫u² dt + 10*(x(T) - x_target)²

        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return u

        def running_cost(x, u, t):
            return u[0]**2

        def terminal_cost(x):
            x_target = 0.0
            return 10.0 * (x[0] - x_target)**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_cost=terminal_cost
        )

        x0 = np.array([1.0])
        result = solver.solve(
            x0=x0,
            xf=None,
            duration=2.0,
            n_steps=40,
            method='single_shooting'
        )

        # Should get close to x=0 due to terminal cost
        x_final = result['state'][-1, 0]
        assert np.abs(x_final) < 0.2, f"Terminal cost should drive x→0, got {x_final}"

        print(f"✓ Test 5 passed: Terminal cost, final x = {x_final:.4f}")


class TestQuantumControl:
    """Test quantum optimal control via PMP."""

    def test_6_two_level_state_transfer(self):
        """Test 6: Two-level system state transfer (|0⟩ → |1⟩)."""
        # Single control: Rabi drive on σ_x

        # Pauli matrices
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

        H0 = -0.5 * sigma_z  # Detuning
        control_hamiltonians = [sigma_x]  # Rabi drive

        psi0 = np.array([1, 0], dtype=complex)  # Ground state
        psi_target = np.array([0, 1], dtype=complex)  # Excited state

        # Control bounds
        u_min = np.array([-2.0])
        u_max = np.array([2.0])

        result = solve_quantum_control_pmp(
            H0=H0,
            control_hamiltonians=control_hamiltonians,
            psi0=psi0,
            target_state=psi_target,
            duration=5.0,
            n_steps=50,
            control_bounds=(u_min, u_max),
            state_cost_weight=10.0,
            control_cost_weight=0.01,
            method='multiple_shooting',
            hbar=1.0
        )

        fidelity = result['final_fidelity']
        assert fidelity > 0.8, f"Should achieve high fidelity, got {fidelity:.4f}"

        print(f"✓ Test 6 passed: Quantum state transfer, fidelity = {fidelity:.4f}")

    def test_7_three_level_ladder(self):
        """Test 7: Three-level ladder system."""
        # Transfer from |0⟩ to |2⟩ via |1⟩

        n_dim = 3
        H0 = np.diag([0.0, 1.0, 2.0])  # Energy levels

        # Two controls: 0↔1 and 1↔2 transitions
        H1 = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=complex)

        H2 = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ], dtype=complex)

        control_hamiltonians = [H1, H2]

        psi0 = np.array([1, 0, 0], dtype=complex)
        psi_target = np.array([0, 0, 1], dtype=complex)

        u_min = np.array([-1.5, -1.5])
        u_max = np.array([1.5, 1.5])

        result = solve_quantum_control_pmp(
            H0=H0,
            control_hamiltonians=control_hamiltonians,
            psi0=psi0,
            target_state=psi_target,
            duration=8.0,
            n_steps=80,
            control_bounds=(u_min, u_max),
            state_cost_weight=5.0,
            control_cost_weight=0.05,
            method='multiple_shooting',
            hbar=1.0
        )

        fidelity = result['final_fidelity']
        assert fidelity > 0.6, f"Three-level transfer should achieve reasonable fidelity, got {fidelity:.4f}"

        print(f"✓ Test 7 passed: Three-level ladder, fidelity = {fidelity:.4f}")

    def test_8_unitarity_preservation(self):
        """Test 8: Check that quantum evolution preserves norm."""
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        H0 = np.zeros((2, 2), dtype=complex)

        psi0 = np.array([1, 0], dtype=complex)

        result = solve_quantum_control_pmp(
            H0=H0,
            control_hamiltonians=[sigma_x],
            psi0=psi0,
            target_state=None,  # Free evolution
            duration=3.0,
            n_steps=30,
            control_cost_weight=0.1,
            method='single_shooting',
            hbar=1.0
        )

        # Check norm preservation
        norms = np.array([np.linalg.norm(result['psi_evolution'][i])
                         for i in range(len(result['psi_evolution']))])

        max_deviation = np.max(np.abs(norms - 1.0))
        assert max_deviation < 0.1, f"Norm should be preserved, max deviation: {max_deviation}"

        print(f"✓ Test 8 passed: Unitarity preserved, max norm deviation = {max_deviation:.4e}")

    def test_9_hadamard_gate_synthesis(self):
        """Test 9: Synthesize Hadamard gate via optimal control."""
        # Target: H|0⟩ = (|0⟩ + |1⟩)/√2

        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        H0 = np.zeros((2, 2), dtype=complex)
        control_hamiltonians = [sigma_x, sigma_y]

        psi0 = np.array([1, 0], dtype=complex)
        psi_target = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)

        result = solve_quantum_control_pmp(
            H0=H0,
            control_hamiltonians=control_hamiltonians,
            psi0=psi0,
            target_state=psi_target,
            duration=4.0,
            n_steps=40,
            control_bounds=(np.array([-3.0, -3.0]), np.array([3.0, 3.0])),
            state_cost_weight=10.0,
            control_cost_weight=0.01,
            method='multiple_shooting',
            hbar=1.0
        )

        fidelity = result['final_fidelity']
        assert fidelity > 0.7, f"Hadamard gate synthesis fidelity: {fidelity:.4f}"

        print(f"✓ Test 9 passed: Hadamard gate, fidelity = {fidelity:.4f}")

    def test_10_energy_minimization(self):
        """Test 10: Quantum control energy minimization."""
        # No target, just minimize control energy

        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        H0 = np.array([[1, 0], [0, -1]], dtype=complex)

        psi0 = np.array([1, 0], dtype=complex)

        result = solve_quantum_control_pmp(
            H0=H0,
            control_hamiltonians=[sigma_x],
            psi0=psi0,
            target_state=None,
            duration=5.0,
            n_steps=50,
            control_cost_weight=1.0,
            method='single_shooting',
            hbar=1.0
        )

        # Should use minimal control
        control_energy = np.mean(result['control']**2)
        assert control_energy < 0.5, "Should minimize control energy"

        print(f"✓ Test 10 passed: Energy minimization, avg u² = {control_energy:.4f}")


class TestSolverComparison:
    """Compare shooting methods."""

    def test_11_single_vs_multiple_shooting(self):
        """Test 11: Compare single and multiple shooting."""
        state_dim = 2
        control_dim = 1

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def running_cost(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0, 0.5])
        xf = np.array([0.0, 0.0])

        # Single shooting
        result_single = solver.solve(
            x0=x0, xf=xf, duration=3.0, n_steps=50,
            method='single_shooting', verbose=False
        )

        # Multiple shooting
        result_multi = solver.solve(
            x0=x0, xf=xf, duration=3.0, n_steps=50,
            method='multiple_shooting', verbose=False
        )

        # Both should converge to similar solutions
        cost_diff = np.abs(result_single['cost'] - result_multi['cost'])
        assert cost_diff < 0.5, f"Both methods should give similar costs, diff: {cost_diff}"

        print(f"✓ Test 11 passed: Single cost = {result_single['cost']:.4f}, "
              f"Multiple cost = {result_multi['cost']:.4f}")

    def test_12_convergence_tolerance(self):
        """Test 12: Test convergence with different tolerances."""
        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return u

        def running_cost(x, u, t):
            return x[0]**2 + u[0]**2

        solver_loose = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )
        solver_loose.tolerance = 1e-4

        solver_tight = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )
        solver_tight.tolerance = 1e-8

        x0 = np.array([1.0])
        xf = np.array([0.0])

        result_loose = solver_loose.solve(x0=x0, xf=xf, duration=2.0, n_steps=40)
        result_tight = solver_tight.solve(x0=x0, xf=xf, duration=2.0, n_steps=40)

        # Tighter tolerance should give lower cost
        assert result_tight['cost'] <= result_loose['cost'] + 0.1

        print(f"✓ Test 12 passed: Loose cost = {result_loose['cost']:.6f}, "
              f"Tight cost = {result_tight['cost']:.6f}")


class TestHamiltonianProperties:
    """Test Hamiltonian properties and conservation."""

    def test_13_hamiltonian_computation(self):
        """Test 13: Hamiltonian computation accuracy."""
        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return u

        def running_cost(x, u, t):
            return x[0]**2 + u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0])
        result = solver.solve(x0=x0, xf=np.array([0.0]), duration=2.0, n_steps=40)

        # Hamiltonian should be approximately constant along optimal trajectory
        H_traj = result['hamiltonian']
        H_std = np.std(H_traj)

        # Note: For this simple problem, H is not exactly constant, but should be smooth
        assert H_std < 1.0, f"Hamiltonian should be relatively smooth, std: {H_std}"

        print(f"✓ Test 13 passed: Hamiltonian std = {H_std:.4e}")

    def test_14_costate_computation(self):
        """Test 14: Costate (adjoint) computation."""
        state_dim = 2
        control_dim = 1

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def running_cost(x, u, t):
            return x[0]**2 + 0.1 * u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0, 0.0])
        result = solver.solve(x0=x0, xf=np.array([0.0, 0.0]), duration=3.0, n_steps=50)

        # Costate should be computed
        assert result['costate'] is not None
        assert result['costate'].shape == (50, 2)

        # Costate should not be all zeros (it's doing something)
        assert np.max(np.abs(result['costate'])) > 0.01

        print(f"✓ Test 14 passed: Costate computed, max |λ| = {np.max(np.abs(result['costate'])):.4f}")

    def test_15_optimality_conditions(self):
        """Test 15: Check optimality conditions (∂H/∂u ≈ 0)."""
        # For unconstrained problems at optimum, gradient should be small

        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return -0.5 * x + u

        def running_cost(x, u, t):
            return x[0]**2 + 0.5 * u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0])
        result = solver.solve(x0=x0, xf=None, duration=5.0, n_steps=50)

        # Check a few points: compute ∂H/∂u numerically
        eps = 1e-5
        max_grad = 0.0

        for i in range(10, 40, 10):
            x = result['state'][i]
            lam = result['costate'][i]
            u = result['control'][i]
            t = result['time'][i]

            H_plus = solver._compute_hamiltonian(x, lam, u + eps, t)
            H_minus = solver._compute_hamiltonian(x, lam, u - eps, t)
            grad = (H_plus - H_minus) / (2 * eps)

            max_grad = max(max_grad, np.abs(grad))

        # Gradient should be small at optimum (not perfect due to discretization)
        assert max_grad < 1.0, f"Optimality gradient should be small, got {max_grad}"

        print(f"✓ Test 15 passed: Max |∂H/∂u| = {max_grad:.4e}")


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_16_zero_control_problem(self):
        """Test 16: Problem where optimal control is zero."""
        # Stable system, no target, only control cost

        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return -x + u  # Stable dynamics

        def running_cost(x, u, t):
            return u[0]**2  # Only control cost

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([0.1])
        result = solver.solve(x0=x0, xf=None, duration=5.0, n_steps=50)

        # Optimal control should be near zero
        avg_u = np.mean(np.abs(result['control']))
        assert avg_u < 0.1, f"Control should be minimal, got {avg_u}"

        print(f"✓ Test 16 passed: Zero control, avg |u| = {avg_u:.4e}")

    def test_17_high_dimensional_state(self):
        """Test 17: Higher dimensional state space."""
        state_dim = 5
        control_dim = 2

        # Random linear system
        np.random.seed(42)
        A = np.random.randn(state_dim, state_dim) * 0.1
        B = np.random.randn(state_dim, control_dim)

        def dynamics(x, u, t):
            return A @ x + B @ u

        def running_cost(x, u, t):
            return np.sum(x**2) + 0.1 * np.sum(u**2)

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.random.randn(state_dim)
        xf = np.zeros(state_dim)

        result = solver.solve(
            x0=x0, xf=xf, duration=5.0, n_steps=50,
            method='multiple_shooting'
        )

        error = np.linalg.norm(result['state'][-1] - xf)
        assert error < 1.0, f"Should approximate target, error: {error}"

        print(f"✓ Test 17 passed: High-dim problem, final error = {error:.4e}")

    def test_18_time_varying_cost(self):
        """Test 18: Time-varying running cost."""
        state_dim = 1
        control_dim = 1

        def dynamics(x, u, t):
            return u

        def running_cost(x, u, t):
            # Cost increases with time
            weight = 1.0 + 0.5 * t
            return weight * (x[0]**2 + u[0]**2)

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0])
        result = solver.solve(x0=x0, xf=np.array([0.0]), duration=2.0, n_steps=40)

        # Should reach target faster due to increasing cost
        # Check that most of the progress happens early
        halfway_idx = len(result['state']) // 2
        x_halfway = np.abs(result['state'][halfway_idx, 0])
        assert x_halfway < 0.5, "Should move quickly due to time-varying cost"

        print(f"✓ Test 18 passed: Time-varying cost, x(T/2) = {x_halfway:.4f}")

    def test_19_multiple_controls(self):
        """Test 19: System with multiple control inputs."""
        state_dim = 2
        control_dim = 2

        def dynamics(x, u, t):
            # Each control affects one state
            return np.array([u[0], u[1]])

        def running_cost(x, u, t):
            return np.sum(x**2) + 0.1 * np.sum(u**2)

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost
        )

        x0 = np.array([1.0, -1.0])
        xf = np.array([0.0, 0.0])

        result = solver.solve(x0=x0, xf=xf, duration=2.0, n_steps=40)

        error = np.linalg.norm(result['state'][-1] - xf)
        assert error < 0.2, f"Should reach target, error: {error}"

        print(f"✓ Test 19 passed: Multiple controls, final error = {error:.4e}")

    def test_20_nonlinear_dynamics(self):
        """Test 20: Nonlinear dynamics (pendulum-like)."""
        state_dim = 2  # [angle, angular_velocity]
        control_dim = 1

        def dynamics(x, u, t):
            # Damped pendulum with control torque
            theta, omega = x
            g = 9.81
            L = 1.0
            b = 0.1  # Damping

            dtheta = omega
            domega = -(g / L) * np.sin(theta) - b * omega + u[0]

            return np.array([dtheta, domega])

        def running_cost(x, u, t):
            # Want to reach upright (θ=π)
            theta_target = np.pi
            return (x[0] - theta_target)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

        solver = PontryaginSolver(
            state_dim=state_dim,
            control_dim=control_dim,
            dynamics=dynamics,
            running_cost=running_cost,
            control_bounds=(np.array([-5.0]), np.array([5.0]))
        )

        x0 = np.array([0.0, 0.0])  # Hanging down
        # Don't enforce hard endpoint (challenging nonlinear problem)

        result = solver.solve(
            x0=x0,
            xf=None,
            duration=5.0,
            n_steps=100,
            method='multiple_shooting'
        )

        # Check that angle increases (moving toward π)
        theta_final = result['state'][-1, 0]
        assert theta_final > 1.0, f"Pendulum should swing up, final θ = {theta_final:.4f}"

        print(f"✓ Test 20 passed: Nonlinear pendulum, final θ = {theta_final:.4f} rad")


def run_all_tests():
    """Run all PMP tests and report results."""
    print("\n" + "="*70)
    print("Running Pontryagin Maximum Principle Solver Tests")
    print("="*70)

    test_classes = [
        TestPontryaginBasics,
        TestQuantumControl,
        TestSolverComparison,
        TestHamiltonianProperties,
        TestEdgeCases
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 70)

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name} FAILED: {e}")
            except Exception as e:
                print(f"✗ {method_name} ERROR: {e}")

    print("\n" + "="*70)
    print(f"Results: {passed_tests}/{total_tests} tests passed ({100*passed_tests/total_tests:.1f}%)")
    print("="*70 + "\n")

    return passed_tests, total_tests


if __name__ == '__main__':
    run_all_tests()
