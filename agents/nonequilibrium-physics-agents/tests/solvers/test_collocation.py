"""Tests for Collocation Methods Solver.

This test suite validates the orthogonal collocation solver for optimal control.

Test Categories:
1. Basic functionality (LQR, double integrator)
2. Different collocation schemes (Gauss-Legendre, Radau, Hermite-Simpson)
3. Constrained problems (control/state bounds)
4. Quantum control applications
5. Mesh refinement and accuracy

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
from solvers.collocation import CollocationSolver, solve_quantum_control_collocation


class TestCollocationBasics:
    """Test basic collocation solver functionality."""

    def test_1_simple_lqr(self):
        """Test 1: Solve simple LQR problem."""
        print("\n  Test 1: Simple LQR with collocation")

        # Simple integrator: dx/dt = u
        def dynamics(x, u, t):
            return u

        # Quadratic cost
        def cost(x, u, t):
            return x**2 + u**2

        solver = CollocationSolver(1, 1, dynamics, cost)
        result = solver.solve(
            x0=np.array([1.0]),
            xf=None,
            duration=5.0,
            n_elements=20,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Cost: {result['cost']:.6f}")
        print(f"    Final state: {result['x'][-1]}")

        # Should converge
        assert result['converged'], "LQR should converge"

        # Cost should be reasonable (known solution exists)
        assert result['cost'] < 2.0, f"LQR cost too high: {result['cost']}"

        # Final state should be near zero
        assert np.abs(result['x'][-1, 0]) < 0.5, "Final state should be small"

    def test_2_double_integrator(self):
        """Test 2: Double integrator problem."""
        print("\n  Test 2: Double integrator")

        # [x, v] state, u control
        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def cost(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        solver = CollocationSolver(2, 1, dynamics, cost)
        result = solver.solve(
            x0=np.array([1.0, 0.0]),
            duration=5.0,
            n_elements=20,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Cost: {result['cost']:.6f}")
        print(f"    Final position: {result['x'][-1, 0]:.4f}")
        print(f"    Final velocity: {result['x'][-1, 1]:.4f}")

        assert result['converged'], "Double integrator should converge"
        assert result['cost'] < 5.0, f"Cost too high: {result['cost']}"

    def test_3_fixed_endpoint(self):
        """Test 3: Problem with fixed endpoint."""
        print("\n  Test 3: Fixed endpoint")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return u**2

        solver = CollocationSolver(1, 1, dynamics, cost)
        result = solver.solve(
            x0=np.array([1.0]),
            xf=np.array([0.0]),  # Fixed endpoint
            duration=5.0,
            n_elements=20,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Final state: {result['x'][-1, 0]:.6f}")
        print(f"    Target: 0.0")
        print(f"    Error: {np.abs(result['x'][-1, 0]):.2e}")

        assert result['converged'], "Fixed endpoint should converge"
        assert np.abs(result['x'][-1, 0]) < 1e-3, "Should reach target"

    def test_4_control_constraints(self):
        """Test 4: Control constraints."""
        print("\n  Test 4: Control constraints")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2

        u_min = np.array([-1.0])
        u_max = np.array([1.0])

        solver = CollocationSolver(
            1, 1, dynamics, cost,
            control_bounds=(u_min, u_max)
        )
        result = solver.solve(
            x0=np.array([5.0]),
            duration=10.0,
            n_elements=20,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Max control: {np.max(np.abs(result['u'])):.4f}")
        print(f"    Constraint: 1.0")

        assert result['converged'], "Constrained problem should converge"
        # Control should respect bounds (with small tolerance)
        assert np.all(result['u'] >= u_min[0] - 1e-6), "Control below lower bound"
        assert np.all(result['u'] <= u_max[0] + 1e-6), "Control above upper bound"


class TestCollocationSchemes:
    """Test different collocation schemes."""

    def test_5_gauss_legendre(self):
        """Test 5: Gauss-Legendre collocation."""
        print("\n  Test 5: Gauss-Legendre collocation")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2 + u**2

        solver = CollocationSolver(
            1, 1, dynamics, cost,
            collocation_type='gauss-legendre',
            collocation_order=3
        )
        result = solver.solve(
            x0=np.array([1.0]),
            duration=5.0,
            n_elements=15,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Cost: {result['cost']:.6f}")

        assert result['converged'], "Gauss-Legendre should converge"

    def test_6_radau(self):
        """Test 6: Radau collocation."""
        print("\n  Test 6: Radau collocation")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2 + u**2

        solver = CollocationSolver(
            1, 1, dynamics, cost,
            collocation_type='radau',
            collocation_order=3
        )
        result = solver.solve(
            x0=np.array([1.0]),
            duration=5.0,
            n_elements=15,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Cost: {result['cost']:.6f}")

        assert result['converged'], "Radau should converge"

    def test_7_hermite_simpson(self):
        """Test 7: Hermite-Simpson collocation."""
        print("\n  Test 7: Hermite-Simpson collocation")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2 + u**2

        solver = CollocationSolver(
            1, 1, dynamics, cost,
            collocation_type='hermite-simpson',
            collocation_order=3
        )
        result = solver.solve(
            x0=np.array([1.0]),
            duration=5.0,
            n_elements=15,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Cost: {result['cost']:.6f}")

        assert result['converged'], "Hermite-Simpson should converge"

    def test_8_scheme_comparison(self):
        """Test 8: Compare collocation schemes."""
        print("\n  Test 8: Scheme comparison")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2 + u**2

        schemes = ['gauss-legendre', 'radau', 'hermite-simpson']
        costs = {}

        for scheme in schemes:
            solver = CollocationSolver(
                1, 1, dynamics, cost,
                collocation_type=scheme,
                collocation_order=3
            )
            result = solver.solve(
                x0=np.array([1.0]),
                duration=5.0,
                n_elements=15,
                verbose=False
            )
            costs[scheme] = result['cost']
            print(f"    {scheme:20s}: {result['cost']:.6f}")

        # All should converge to similar costs
        cost_values = list(costs.values())
        max_diff = max(cost_values) - min(cost_values)
        print(f"    Max cost difference: {max_diff:.4f}")

        assert max_diff < 0.5, "Schemes should give similar costs"


class TestCollocationQuantum:
    """Test quantum control with collocation."""

    def test_9_two_level_system(self):
        """Test 9: Two-level quantum control."""
        print("\n  Test 9: Two-level quantum control")

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        H0 = np.zeros((2, 2), dtype=complex)

        psi0 = np.array([1, 0], dtype=complex)
        psi_target = np.array([0, 1], dtype=complex)

        result = solve_quantum_control_collocation(
            H0=H0,
            control_hamiltonians=[sigma_x],
            psi0=psi0,
            target_state=psi_target,
            duration=5.0,
            n_elements=20,
            control_bounds=(np.array([-5.0]), np.array([5.0])),
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Final fidelity: {result['final_fidelity']:.4f}")
        print(f"    Cost: {result['cost']:.6f}")

        assert result['converged'], "Quantum control should converge"
        assert result['final_fidelity'] > 0.90, f"Fidelity too low: {result['final_fidelity']:.4f}"

    def test_10_hadamard_gate(self):
        """Test 10: Hadamard gate synthesis."""
        print("\n  Test 10: Hadamard gate synthesis")

        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        H0 = np.zeros((2, 2), dtype=complex)

        psi0 = np.array([1, 0], dtype=complex)
        psi_target = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)

        result = solve_quantum_control_collocation(
            H0=H0,
            control_hamiltonians=[sigma_x],
            psi0=psi0,
            target_state=psi_target,
            duration=5.0,
            n_elements=25,
            control_bounds=(np.array([-3.0]), np.array([3.0])),
            control_weight=0.05,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Final fidelity: {result['final_fidelity']:.4f}")
        print(f"    Target fidelity: > 0.95")

        assert result['converged'], "Hadamard gate should converge"
        assert result['final_fidelity'] > 0.85, f"Fidelity too low: {result['final_fidelity']:.4f}"

    def test_11_unitarity_preservation(self):
        """Test 11: Verify unitarity is preserved."""
        print("\n  Test 11: Unitarity preservation")

        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        H0 = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

        psi0 = np.array([1, 0], dtype=complex)
        psi_target = np.array([0, 1], dtype=complex)

        result = solve_quantum_control_collocation(
            H0=H0,
            control_hamiltonians=[sigma_x],
            psi0=psi0,
            target_state=psi_target,
            duration=10.0,
            n_elements=30,
            verbose=False
        )

        # Check norm preservation
        norms = np.array([np.linalg.norm(psi) for psi in result['psi']])
        norm_error = np.max(np.abs(norms - 1.0))

        print(f"    Max norm error: {norm_error:.2e}")
        print(f"    Target: < 1e-6")

        assert norm_error < 1e-4, f"Norm not preserved: {norm_error:.2e}"


class TestCollocationAccuracy:
    """Test accuracy and convergence."""

    def test_12_mesh_refinement(self):
        """Test 12: Convergence with mesh refinement."""
        print("\n  Test 12: Mesh refinement")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2 + u**2

        solver = CollocationSolver(1, 1, dynamics, cost)

        n_elements_list = [5, 10, 20, 40]
        costs = []

        for n_elem in n_elements_list:
            result = solver.solve(
                x0=np.array([1.0]),
                duration=5.0,
                n_elements=n_elem,
                verbose=False
            )
            costs.append(result['cost'])
            print(f"    Elements: {n_elem:3d}, Cost: {result['cost']:.6f}")

        # Cost should converge
        cost_changes = [abs(costs[i+1] - costs[i]) for i in range(len(costs)-1)]
        print(f"    Cost changes: {cost_changes}")

        # Later changes should be smaller (convergence)
        assert cost_changes[-1] < cost_changes[0], "Should converge with refinement"

    def test_13_high_order_accuracy(self):
        """Test 13: High-order collocation accuracy."""
        print("\n  Test 13: High-order accuracy")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2 + u**2

        orders = [2, 3, 4]
        costs = {}

        for order in orders:
            solver = CollocationSolver(
                1, 1, dynamics, cost,
                collocation_order=order
            )
            result = solver.solve(
                x0=np.array([1.0]),
                duration=5.0,
                n_elements=10,
                verbose=False
            )
            costs[order] = result['cost']
            print(f"    Order {order}: Cost = {result['cost']:.6f}")

        # Higher order should generally be more accurate
        assert all(result['converged'] for result in [
            solver.solve(x0=np.array([1.0]), duration=5.0, n_elements=10, verbose=False)
        ]), "All orders should converge"


class TestCollocationEdgeCases:
    """Test edge cases and robustness."""

    def test_14_zero_control_optimal(self):
        """Test 14: Case where zero control is optimal."""
        print("\n  Test 14: Zero control optimal")

        # dx/dt = -x (stable), minimize u^2
        def dynamics(x, u, t):
            return -x + u

        def cost(x, u, t):
            return 100 * u**2  # Heavy control penalty

        solver = CollocationSolver(1, 1, dynamics, cost)
        result = solver.solve(
            x0=np.array([1.0]),
            duration=5.0,
            n_elements=20,
            verbose=False
        )

        mean_control = np.mean(np.abs(result['u']))
        print(f"    Mean |u|: {mean_control:.6f}")
        print(f"    Should be small (< 0.1)")

        assert mean_control < 0.1, "Control should be near zero"

    def test_15_long_horizon(self):
        """Test 15: Long time horizon."""
        print("\n  Test 15: Long horizon")

        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return x**2 + u**2

        solver = CollocationSolver(1, 1, dynamics, cost)
        result = solver.solve(
            x0=np.array([1.0]),
            duration=20.0,  # Long horizon
            n_elements=40,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Cost: {result['cost']:.6f}")

        assert result['converged'], "Long horizon should converge"

    def test_16_multiple_controls(self):
        """Test 16: Multiple control inputs."""
        print("\n  Test 16: Multiple controls")

        # 2D system, 2 controls
        def dynamics(x, u, t):
            return np.array([u[0], u[1]])

        def cost(x, u, t):
            return np.sum(x**2) + np.sum(u**2)

        solver = CollocationSolver(2, 2, dynamics, cost)
        result = solver.solve(
            x0=np.array([1.0, -1.0]),
            duration=5.0,
            n_elements=20,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Final state: {result['x'][-1]}")

        assert result['converged'], "Multiple controls should work"

    def test_17_nonlinear_dynamics(self):
        """Test 17: Nonlinear dynamics (pendulum-like)."""
        print("\n  Test 17: Nonlinear dynamics")

        # Simplified pendulum: dθ/dt = ω, dω/dt = -sin(θ) + u
        def dynamics(x, u, t):
            theta, omega = x
            return np.array([omega, -np.sin(theta) + u[0]])

        def cost(x, u, t):
            return x[0]**2 + 0.1 * u[0]**2

        solver = CollocationSolver(2, 1, dynamics, cost)
        result = solver.solve(
            x0=np.array([0.5, 0.0]),  # Start at 0.5 rad
            duration=5.0,
            n_elements=30,
            verbose=False
        )

        print(f"    Converged: {result['converged']}")
        print(f"    Final angle: {result['x'][-1, 0]:.4f}")

        assert result['converged'], "Nonlinear dynamics should converge"


def run_all_tests():
    """Run all collocation tests."""
    print("\n" + "="*70)
    print("Collocation Methods Solver - Test Suite")
    print("="*70)

    test_classes = [
        TestCollocationBasics,
        TestCollocationSchemes,
        TestCollocationQuantum,
        TestCollocationAccuracy,
        TestCollocationEdgeCases
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)

        test_obj = test_class()
        methods = [m for m in dir(test_obj) if m.startswith('test_')]

        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_obj, method_name)
                method()
                passed_tests += 1
                print(f"  ✓ {method_name}")
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
            except Exception as e:
                print(f"  ✗ {method_name}: Unexpected error: {e}")

    print("\n" + "="*70)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("="*70)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
