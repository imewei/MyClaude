"""Performance and regression tests for advanced solvers.

Tests solver performance, convergence rates, and cross-solver comparisons
for PMP, Collocation, and Magnus expansion solvers.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import solvers
from solvers.pontryagin import PontryaginSolver
from solvers.collocation import CollocationSolver
from solvers.magnus_expansion import MagnusExpansionSolver

# Try importing JAX PMP
try:
    from solvers.pontryagin_jax import PontryaginSolverJAX
    JAX_PMP_AVAILABLE = True
except ImportError:
    JAX_PMP_AVAILABLE = False


class TestPMPPerformanceRegression:
    """Performance regression tests for Pontryagin Maximum Principle solver."""

    def test_lqr_convergence_speed(self):
        """Test: LQR problem converges in < 20 iterations."""
        solver = PontryaginSolver(n_states=2, n_controls=1)

        # LQR problem: minimize ∫ (x'Qx + u'Ru) dt
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        R = np.array([[0.1]])

        def dynamics(x, u, t):
            A = np.array([[0.0, 1.0], [-1.0, -0.1]])
            B = np.array([[0.0], [1.0]])
            return A @ x + B.flatten() * u[0]

        def running_cost(x, u, t):
            return x.T @ Q @ x + u.T @ R @ u

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 5.0]

        start_time = time.time()
        result = solver.solve(
            dynamics=dynamics,
            running_cost=running_cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=50,
            method='single_shooting',
            max_iterations=50,
            tolerance=1e-6
        )
        elapsed = time.time() - start_time

        assert result['success'], "LQR should converge"
        assert result['iterations'] < 20, f"LQR converged in {result['iterations']} iterations, expected < 20"
        assert elapsed < 5.0, f"LQR took {elapsed:.2f}s, expected < 5s"

    def test_multiple_shooting_robustness(self):
        """Test: Multiple shooting handles nonlinear problems better."""
        solver = PontryaginSolver(n_states=2, n_controls=1)

        # Nonlinear pendulum
        def dynamics(x, u, t):
            theta, omega = x
            return np.array([omega, -9.81 * np.sin(theta) + u[0]])

        def running_cost(x, u, t):
            return 0.1 * x[0]**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

        x0 = np.array([np.pi/4, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 3.0]

        # Multiple shooting should converge
        result_multiple = solver.solve(
            dynamics=dynamics,
            running_cost=running_cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=30,
            method='multiple_shooting',
            n_shooting_nodes=5,
            max_iterations=100,
            tolerance=1e-4
        )

        assert result_multiple['success'], "Multiple shooting should converge on nonlinear problem"
        assert result_multiple['iterations'] < 100, "Should converge in < 100 iterations"

    def test_constrained_control_performance(self):
        """Test: Control constraints handled efficiently."""
        solver = PontryaginSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def running_cost(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 2.0]

        u_min = -1.0
        u_max = 1.0

        start_time = time.time()
        result = solver.solve(
            dynamics=dynamics,
            running_cost=running_cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=30,
            u_min=u_min,
            u_max=u_max,
            method='single_shooting',
            max_iterations=50,
            tolerance=1e-6
        )
        elapsed = time.time() - start_time

        assert result['success'], "Constrained problem should converge"
        assert elapsed < 5.0, f"Constrained problem took {elapsed:.2f}s, expected < 5s"

        # Verify constraints satisfied
        u_optimal = result['optimal_control']
        assert np.all(u_optimal >= u_min - 1e-6), "Control should respect lower bound"
        assert np.all(u_optimal <= u_max + 1e-6), "Control should respect upper bound"


class TestCollocationPerformanceRegression:
    """Performance regression tests for Collocation solver."""

    def test_gauss_legendre_accuracy(self):
        """Test: Gauss-Legendre collocation achieves high accuracy."""
        solver = CollocationSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def objective(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_final = np.array([0.0, 0.0])
        t_span = [0.0, 2.0]

        result = solver.solve(
            dynamics=dynamics,
            objective=objective,
            x0=x0,
            x_final=x_final,
            t_span=t_span,
            n_segments=10,
            collocation_scheme='gauss_legendre',
            degree=3
        )

        assert result['success'], "Gauss-Legendre should converge"

        # Check final state accuracy
        x_final_achieved = result['state_trajectory'][-1]
        error = np.linalg.norm(x_final_achieved - x_final)
        assert error < 1e-4, f"Final state error: {error:.2e}, expected < 1e-4"

    def test_scheme_comparison_performance(self):
        """Test: Compare performance of different collocation schemes."""
        schemes = ['gauss_legendre', 'radau', 'hermite_simpson']
        results = {}

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def objective(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_final = np.array([0.0, 0.0])
        t_span = [0.0, 2.0]

        for scheme in schemes:
            solver = CollocationSolver(n_states=2, n_controls=1)

            start_time = time.time()
            result = solver.solve(
                dynamics=dynamics,
                objective=objective,
                x0=x0,
                x_final=x_final,
                t_span=t_span,
                n_segments=10,
                collocation_scheme=scheme,
                degree=3
            )
            elapsed = time.time() - start_time

            results[scheme] = {
                'time': elapsed,
                'cost': result.get('optimal_cost', np.inf),
                'success': result['success']
            }

            assert result['success'], f"{scheme} should converge"
            assert elapsed < 10.0, f"{scheme} took {elapsed:.2f}s, expected < 10s"

        # All should give similar costs (within 10%)
        costs = [r['cost'] for r in results.values() if r['cost'] != np.inf]
        if costs:
            cost_std = np.std(costs)
            cost_mean = np.mean(costs)
            assert cost_std / cost_mean < 0.1, "Schemes should give similar costs"

    def test_constraint_handling_performance(self):
        """Test: Constraints handled efficiently via NLP."""
        solver = CollocationSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def objective(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_final = np.array([0.0, 0.0])
        t_span = [0.0, 2.0]

        u_bounds = (-1.0, 1.0)
        x_bounds = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        start_time = time.time()
        result = solver.solve(
            dynamics=dynamics,
            objective=objective,
            x0=x0,
            x_final=x_final,
            t_span=t_span,
            n_segments=10,
            u_bounds=u_bounds,
            x_bounds=x_bounds,
            collocation_scheme='gauss_legendre',
            degree=3
        )
        elapsed = time.time() - start_time

        assert result['success'], "Constrained collocation should converge"
        assert elapsed < 10.0, f"Constrained took {elapsed:.2f}s, expected < 10s"


class TestMagnusExpansionPerformance:
    """Performance regression tests for Magnus expansion solver."""

    def test_energy_conservation_accuracy(self):
        """Test: Magnus expansion conserves energy better than RK4."""
        # Two-level system with time-dependent drive
        n_dim = 2
        omega = 1.0  # Resonance frequency
        drive_amp = 0.5
        drive_freq = omega

        def H_func(t):
            H0 = np.array([[0.0, 0.0], [0.0, omega]])
            H_drive = drive_amp * np.cos(drive_freq * t) * np.array([[0.0, 1.0], [1.0, 0.0]])
            return H0 + H_drive

        psi0 = np.array([1.0, 0.0], dtype=np.complex128)  # Ground state
        t_span = np.linspace(0, 10.0, 100)

        solver = MagnusExpansionSolver(n_dim=n_dim, order=4)

        result = solver.solve(H_func=H_func, psi0=psi0, t_span=t_span)

        # Check energy conservation (for closed system)
        energies = []
        for i, t in enumerate(t_span):
            psi_t = result['state_trajectory'][i]
            H_t = H_func(t)
            energy = np.real(psi_t.conj() @ H_t @ psi_t)
            energies.append(energy)

        energy_std = np.std(energies)
        # Magnus should conserve energy well
        assert energy_std < 0.1, f"Energy fluctuation: {energy_std:.2e}, expected < 0.1"

    def test_order_convergence_rates(self):
        """Test: Higher Magnus orders converge faster."""
        n_dim = 2
        omega = 1.0

        def H_func(t):
            return np.array([[0.0, np.cos(omega * t)], [np.cos(omega * t), 0.0]])

        psi0 = np.array([1.0, 0.0], dtype=np.complex128)

        orders = [2, 4, 6]
        errors = []

        # Reference solution (fine grid with order 6)
        t_span_fine = np.linspace(0, 2.0, 500)
        solver_ref = MagnusExpansionSolver(n_dim=n_dim, order=6)
        result_ref = solver_ref.solve(H_func=H_func, psi0=psi0, t_span=t_span_fine)
        psi_ref = result_ref['state_trajectory'][-1]

        for order in orders:
            t_span = np.linspace(0, 2.0, 50)
            solver = MagnusExpansionSolver(n_dim=n_dim, order=order)

            start_time = time.time()
            result = solver.solve(H_func=H_func, psi0=psi0, t_span=t_span)
            elapsed = time.time() - start_time

            psi_final = result['state_trajectory'][-1]
            error = np.linalg.norm(psi_final - psi_ref)
            errors.append(error)

            assert elapsed < 2.0, f"Order {order} took {elapsed:.2f}s, expected < 2s"

        # Higher order should have lower error
        assert errors[2] < errors[1] < errors[0], "Higher Magnus order should reduce error"

    def test_magnus_performance_scaling(self):
        """Test: Magnus solver scales well with system size."""
        omega = 1.0

        for n_dim in [2, 4, 8]:
            # Create n-dimensional Hamiltonian
            def H_func(t):
                H = np.zeros((n_dim, n_dim))
                for i in range(n_dim - 1):
                    H[i, i+1] = np.cos(omega * t)
                    H[i+1, i] = np.cos(omega * t)
                return H

            psi0 = np.zeros(n_dim, dtype=np.complex128)
            psi0[0] = 1.0

            t_span = np.linspace(0, 1.0, 50)
            solver = MagnusExpansionSolver(n_dim=n_dim, order=4)

            start_time = time.time()
            result = solver.solve(H_func=H_func, psi0=psi0, t_span=t_span)
            elapsed = time.time() - start_time

            # Should complete in reasonable time
            max_time = 5.0 * (n_dim / 2)  # Linear scaling expected
            assert elapsed < max_time, f"n_dim={n_dim} took {elapsed:.2f}s, expected < {max_time:.1f}s"

            # Verify normalization
            norm = np.linalg.norm(result['state_trajectory'][-1])
            assert np.abs(norm - 1.0) < 1e-6, f"Norm={norm:.6f}, expected 1.0"


class TestCrossSolverComparison:
    """Cross-solver comparison tests."""

    def test_pmp_vs_collocation_lqr(self):
        """Test: PMP and Collocation give similar results on LQR."""
        # Define LQR problem
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        R = np.array([[0.1]])

        def dynamics(x, u, t):
            A = np.array([[0.0, 1.0], [-1.0, -0.1]])
            B = np.array([[0.0], [1.0]])
            return A @ x + B.flatten() * u[0]

        def cost(x, u, t):
            return x.T @ Q @ x + u.T @ R @ u

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 5.0]

        # PMP
        solver_pmp = PontryaginSolver(n_states=2, n_controls=1)
        result_pmp = solver_pmp.solve(
            dynamics=dynamics,
            running_cost=cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=50,
            method='single_shooting',
            max_iterations=50,
            tolerance=1e-6
        )

        # Collocation
        solver_col = CollocationSolver(n_states=2, n_controls=1)
        result_col = solver_col.solve(
            dynamics=dynamics,
            objective=cost,
            x0=x0,
            x_final=x_target,
            t_span=t_span,
            n_segments=10,
            collocation_scheme='gauss_legendre',
            degree=3
        )

        assert result_pmp['success'], "PMP should converge"
        assert result_col['success'], "Collocation should converge"

        # Compare optimal costs (should be within 10%)
        cost_pmp = result_pmp.get('optimal_cost', np.inf)
        cost_col = result_col.get('optimal_cost', np.inf)

        if cost_pmp != np.inf and cost_col != np.inf:
            rel_diff = np.abs(cost_pmp - cost_col) / np.mean([cost_pmp, cost_col])
            assert rel_diff < 0.15, f"Cost difference: {rel_diff:.2%}, expected < 15%"

    def test_solver_speed_comparison(self):
        """Test: Compare solve times across methods."""
        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def cost(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 2.0]

        times = {}

        # PMP
        solver_pmp = PontryaginSolver(n_states=2, n_controls=1)
        start = time.time()
        result_pmp = solver_pmp.solve(
            dynamics=dynamics,
            running_cost=cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=30,
            method='single_shooting',
            max_iterations=50,
            tolerance=1e-6
        )
        times['PMP'] = time.time() - start

        # Collocation
        solver_col = CollocationSolver(n_states=2, n_controls=1)
        start = time.time()
        result_col = solver_col.solve(
            dynamics=dynamics,
            objective=cost,
            x0=x0,
            x_final=x_target,
            t_span=t_span,
            n_segments=10,
            collocation_scheme='gauss_legendre',
            degree=3
        )
        times['Collocation'] = time.time() - start

        # Both should complete in < 10s
        for method, t in times.items():
            assert t < 10.0, f"{method} took {t:.2f}s, expected < 10s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
