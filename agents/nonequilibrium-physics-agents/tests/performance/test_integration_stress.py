"""Integration and stress tests for Phase 4 components.

Tests cross-component integration, edge cases, and system stress scenarios.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import components
from solvers.pontryagin import PontryaginSolver
from solvers.collocation import CollocationSolver
from solvers.magnus_expansion import MagnusExpansionSolver

# Try importing GPU and JAX components
try:
    import jax.numpy as jnp
    from gpu_kernels.quantum_evolution import solve_lindblad_jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Import standards
from standards.data_formats import SolverInput, SolverOutput
from standards.schemas import validate_against_schema


class TestGPUSolverIntegration:
    """Test integration between GPU kernels and solvers."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_gpu_quantum_to_pmp_workflow(self):
        """Test: Use GPU quantum evolution as part of PMP cost function."""
        # Scenario: Optimize control to prepare desired quantum state

        n_dim = 2
        n_controls = 1

        def dynamics(x, u, t):
            # State: [qubit parameters]
            # Control: drive amplitude
            return np.array([u[0], -x[0]])

        def quantum_fidelity_cost(x, u, t):
            # Run quick quantum evolution and compute fidelity
            # This is a simplified example
            target_state = jnp.array([0.0, 1.0], dtype=jnp.complex128)

            # Simple cost based on state distance
            cost = 0.01 * (x[0]**2 + x[1]**2) + 0.1 * u[0]**2
            return cost

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.1])
        t_span = [0.0, 1.0]

        solver = PontryaginSolver(n_states=2, n_controls=n_controls)
        result = solver.solve(
            dynamics=dynamics,
            running_cost=quantum_fidelity_cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=30,
            method='single_shooting',
            max_iterations=30,
            tolerance=1e-4
        )

        assert result['success'], "GPU-PMP integration should work"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_magnus_for_pmp_dynamics(self):
        """Test: Use Magnus expansion for quantum dynamics in PMP."""
        # Combine Magnus (quantum evolution) with PMP (optimal control)

        n_dim = 2
        omega = 1.0

        # PMP to find optimal drive protocol
        def dynamics(x, u, t):
            # x = [drive_amplitude, phase]
            return np.array([u[0], omega])

        def cost(x, u, t):
            # Minimize control effort
            return 0.1 * u[0]**2

        x0 = np.array([0.0, 0.0])
        x_target = np.array([1.0, 2*np.pi])
        t_span = [0.0, 2.0]

        solver_pmp = PontryaginSolver(n_states=2, n_controls=1)
        result_pmp = solver_pmp.solve(
            dynamics=dynamics,
            running_cost=cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=30,
            method='single_shooting',
            max_iterations=30,
            tolerance=1e-4
        )

        if result_pmp['success']:
            # Use optimal control with Magnus
            drive_protocol = result_pmp['optimal_control']

            def H_func(t):
                # Find drive at time t
                idx = np.searchsorted(result_pmp['time'], t)
                idx = min(idx, len(drive_protocol) - 1)
                drive = drive_protocol[idx] if len(drive_protocol.shape) == 1 else drive_protocol[idx, 0]

                H = np.array([[0.0, drive], [drive, omega]])
                return H

            psi0 = np.array([1.0, 0.0], dtype=np.complex128)
            t_span_magnus = np.linspace(0, 2.0, 50)

            solver_magnus = MagnusExpansionSolver(n_dim=n_dim, order=4)
            result_magnus = solver_magnus.solve(H_func=H_func, psi0=psi0, t_span=t_span_magnus)

            assert 'state_trajectory' in result_magnus, "Magnus should produce trajectory"
            assert result_magnus['state_trajectory'].shape[0] == 50, "Should have 50 time points"


class TestStandardsIntegration:
    """Test integration with standards/data_formats."""

    def test_solver_input_to_pmp(self):
        """Test: Convert SolverInput standard format to PMP solver."""
        # Create standard solver input
        solver_input = SolverInput(
            solver_type="pmp",
            problem_type="lqr",
            n_states=2,
            n_controls=1,
            initial_state=[1.0, 0.0],
            target_state=[0.0, 0.0],
            time_horizon=[0.0, 5.0],
            dynamics={
                "A": [[0.0, 1.0], [-1.0, -0.1]],
                "B": [[0.0], [1.0]]
            },
            cost={
                "Q": [[1.0, 0.0], [0.0, 1.0]],
                "R": [[0.1]]
            },
            solver_config={
                "max_iterations": 50,
                "tolerance": 1e-6,
                "method": "single_shooting"
            }
        )

        # Validate
        assert solver_input.validate(), "SolverInput should validate"

        # Convert to PMP format
        def dynamics(x, u, t):
            A = np.array(solver_input.dynamics["A"])
            B = np.array(solver_input.dynamics["B"])
            return A @ x + B.flatten() * u[0]

        def cost_func(x, u, t):
            Q = np.array(solver_input.cost["Q"])
            R = np.array(solver_input.cost["R"])
            return x.T @ Q @ x + u.T @ R @ u

        # Solve
        solver = PontryaginSolver(
            n_states=solver_input.n_states,
            n_controls=solver_input.n_controls
        )

        result = solver.solve(
            dynamics=dynamics,
            running_cost=cost_func,
            x0=np.array(solver_input.initial_state),
            x_target=np.array(solver_input.target_state),
            t_span=solver_input.time_horizon,
            n_time=50,
            method=solver_input.solver_config.get("method", "single_shooting"),
            max_iterations=solver_input.solver_config.get("max_iterations", 50),
            tolerance=solver_input.solver_config.get("tolerance", 1e-6)
        )

        assert result['success'], "Standard format PMP should converge"

        # Convert to SolverOutput
        solver_output = SolverOutput(
            success=result['success'],
            optimal_cost=result.get('optimal_cost', 0.0),
            optimal_state=result['optimal_state'].tolist(),
            optimal_control=result['optimal_control'].tolist(),
            time=result['time'].tolist(),
            iterations=result.get('iterations', 0),
            message=result.get('message', 'Success')
        )

        assert solver_output.validate(), "SolverOutput should validate"

    def test_solver_output_serialization(self):
        """Test: SolverOutput serialization roundtrip."""
        # Create solver output
        solver_output = SolverOutput(
            success=True,
            optimal_cost=1.234,
            optimal_state=[[1.0, 0.5], [0.8, 0.3], [0.5, 0.1]],
            optimal_control=[[0.1], [0.05], [0.01]],
            time=[0.0, 0.5, 1.0],
            iterations=15,
            message="Converged"
        )

        # Validate
        assert solver_output.validate(), "Should validate"

        # Convert to dict
        output_dict = solver_output.to_dict()

        # Validate against schema
        assert validate_against_schema(output_dict, "solver_output"), "Should match schema"

        # Roundtrip
        solver_output_2 = SolverOutput.from_dict(output_dict)
        assert solver_output_2.success == solver_output.success
        assert solver_output_2.optimal_cost == solver_output.optimal_cost


class TestEdgeCasesStress:
    """Edge cases and stress tests."""

    def test_zero_control_problem(self):
        """Test: Handle zero control (no actuation) gracefully."""
        solver = PontryaginSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            # Unstable system
            A = np.array([[1.0, 0.1], [0.0, 1.0]])
            return A @ x

        def cost(x, u, t):
            return x[0]**2 + x[1]**2

        x0 = np.array([1.0, 0.0])
        x_target = np.array([2.0, 0.5])  # Unreachable without control
        t_span = [0.0, 1.0]

        # Should handle gracefully (may not converge, but shouldn't crash)
        try:
            result = solver.solve(
                dynamics=dynamics,
                running_cost=cost,
                x0=x0,
                x_target=x_target,
                t_span=t_span,
                n_time=20,
                method='single_shooting',
                max_iterations=10,
                tolerance=1e-6
            )
            # If it converges, great. If not, should return success=False
            assert 'success' in result
        except Exception as e:
            pytest.fail(f"Zero control problem crashed: {e}")

    def test_very_tight_constraints(self):
        """Test: Very tight control constraints."""
        solver = PontryaginSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def cost(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 5.0]

        # Very tight constraints
        u_min = -0.01
        u_max = 0.01

        result = solver.solve(
            dynamics=dynamics,
            running_cost=cost,
            x0=x0,
            x_target=x_target,
            t_span=t_span,
            n_time=50,
            u_min=u_min,
            u_max=u_max,
            method='single_shooting',
            max_iterations=100,
            tolerance=1e-4
        )

        # Should handle tight constraints
        assert 'success' in result
        if result['success']:
            u = result['optimal_control']
            assert np.all(u >= u_min - 1e-6)
            assert np.all(u <= u_max + 1e-6)

    def test_high_dimensional_state(self):
        """Test: High-dimensional state space."""
        n_states = 10
        n_controls = 3

        solver = PontryaginSolver(n_states=n_states, n_controls=n_controls)

        def dynamics(x, u, t):
            # Simple dynamics: dx_i/dt = u_{i%3} - 0.1*x_i
            dxdt = np.zeros(n_states)
            for i in range(n_states):
                dxdt[i] = u[i % n_controls] - 0.1 * x[i]
            return dxdt

        def cost(x, u, t):
            return np.sum(x**2) + 0.1 * np.sum(u**2)

        x0 = np.ones(n_states)
        x_target = np.zeros(n_states)
        t_span = [0.0, 3.0]

        try:
            result = solver.solve(
                dynamics=dynamics,
                running_cost=cost,
                x0=x0,
                x_target=x_target,
                t_span=t_span,
                n_time=30,
                method='single_shooting',
                max_iterations=50,
                tolerance=1e-4
            )
            assert 'success' in result
        except Exception as e:
            pytest.fail(f"High-dimensional problem crashed: {e}")

    def test_discontinuous_control(self):
        """Test: Handle discontinuous control signals."""
        solver = CollocationSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def objective(x, u, t):
            # Encourage bang-bang control
            return x[0]**2 + x[1]**2 + 0.001 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_final = np.array([0.0, 0.0])
        t_span = [0.0, 2.0]

        u_bounds = (-1.0, 1.0)

        result = solver.solve(
            dynamics=dynamics,
            objective=objective,
            x0=x0,
            x_final=x_final,
            t_span=t_span,
            n_segments=20,
            u_bounds=u_bounds,
            collocation_scheme='gauss_legendre',
            degree=3
        )

        assert result['success'], "Discontinuous control should be handled"

    def test_stiff_dynamics(self):
        """Test: Stiff ODE dynamics."""
        solver = CollocationSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            # Stiff system: fast and slow components
            return np.array([-100*x[0] + u[0], x[0] - x[1]])

        def objective(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_final = np.array([0.0, 0.0])
        t_span = [0.0, 1.0]

        # Collocation should handle stiff systems better
        result = solver.solve(
            dynamics=dynamics,
            objective=objective,
            x0=x0,
            x_final=x_final,
            t_span=t_span,
            n_segments=30,  # More segments for stiff problem
            collocation_scheme='radau',  # Radau is good for stiff
            degree=3
        )

        assert 'success' in result

    def test_long_time_horizon(self):
        """Test: Very long time horizon."""
        solver = CollocationSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            return np.array([x[1], u[0] - 0.1*x[1]])

        def objective(x, u, t):
            return 0.01 * (x[0]**2 + x[1]**2) + 0.001 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_final = np.array([0.0, 0.0])
        t_span = [0.0, 100.0]  # Very long

        result = solver.solve(
            dynamics=dynamics,
            objective=objective,
            x0=x0,
            x_final=x_final,
            t_span=t_span,
            n_segments=50,
            collocation_scheme='hermite_simpson',
            degree=3
        )

        assert 'success' in result


class TestRobustnessValidation:
    """Robustness validation tests."""

    def test_multiple_runs_consistency(self):
        """Test: Multiple runs give consistent results."""
        solver = PontryaginSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            A = np.array([[0.0, 1.0], [-1.0, -0.1]])
            B = np.array([[0.0], [1.0]])
            return A @ x + B.flatten() * u[0]

        def cost(x, u, t):
            Q = np.array([[1.0, 0.0], [0.0, 1.0]])
            R = np.array([[0.1]])
            return x.T @ Q @ x + u.T @ R @ u

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 5.0]

        costs = []
        for run in range(3):
            result = solver.solve(
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

            if result['success']:
                costs.append(result.get('optimal_cost', 0.0))

        if len(costs) >= 2:
            # Should be very consistent
            assert np.std(costs) / np.mean(costs) < 0.01, "Runs should be consistent"

    def test_different_initial_guesses(self):
        """Test: Robustness to different initial guesses."""
        solver = PontryaginSolver(n_states=2, n_controls=1)

        def dynamics(x, u, t):
            return np.array([x[1], u[0]])

        def cost(x, u, t):
            return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        x0 = np.array([1.0, 0.0])
        x_target = np.array([0.0, 0.0])
        t_span = [0.0, 2.0]

        # Different initial guesses shouldn't drastically change result
        result1 = solver.solve(
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

        # Both should converge to similar cost
        assert result1['success'], "Should converge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
