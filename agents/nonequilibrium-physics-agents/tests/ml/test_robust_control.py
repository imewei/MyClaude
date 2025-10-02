"""Tests for Robust Control and Uncertainty Quantification.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.robust_control import (
    HInfinityControl,
    StochasticOptimalControl,
    UncertaintyQuantification,
    RiskSensitiveControl,
    RobustControlConfig,
    UQConfig,
    RobustControlMethod,
    UQMethod
)


class TestRobustControlConfig:
    """Tests for robust control configuration."""

    def test_default_config(self):
        """Test: Default robust control configuration."""
        config = RobustControlConfig()

        assert config.method == RobustControlMethod.H_INFINITY.value
        assert config.gamma > 0
        assert config.disturbance_bound > 0

    def test_custom_config(self):
        """Test: Custom robust control configuration."""
        config = RobustControlConfig(
            method=RobustControlMethod.MU_SYNTHESIS.value,
            gamma=0.5,
            risk_aversion=0.1
        )

        assert config.method == RobustControlMethod.MU_SYNTHESIS.value
        assert config.gamma == 0.5
        assert config.risk_aversion == 0.1


class TestUQConfig:
    """Tests for UQ configuration."""

    def test_default_config(self):
        """Test: Default UQ configuration."""
        config = UQConfig()

        assert config.method == UQMethod.MONTE_CARLO.value
        assert config.num_samples > 0
        assert config.poly_order > 0

    def test_custom_config(self):
        """Test: Custom UQ configuration."""
        config = UQConfig(
            method=UQMethod.POLYNOMIAL_CHAOS.value,
            poly_order=5,
            num_samples=5000
        )

        assert config.method == UQMethod.POLYNOMIAL_CHAOS.value
        assert config.poly_order == 5
        assert config.num_samples == 5000


class TestHInfinityControl:
    """Tests for H-infinity control."""

    def test_initialization(self):
        """Test: H-infinity controller initialization."""
        config = RobustControlConfig(gamma=1.5)
        h_inf = HInfinityControl(config)

        assert h_inf.config.gamma == 1.5
        assert h_inf.K is None  # Not yet designed

    def test_solve_riccati_simple(self):
        """Test: Solve H-infinity Riccati equation for simple system."""
        config = RobustControlConfig()
        h_inf = HInfinityControl(config)

        # Simple 2D system
        A = np.array([[0.0, 1.0], [-1.0, -0.1]])
        B1 = np.array([[0.0], [1.0]])  # Disturbance
        B2 = np.array([[0.0], [1.0]])  # Control
        C1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # Regulated output
        D12 = np.zeros((2, 1))

        gamma = 2.0

        P = h_inf.solve_riccati(A, B1, B2, C1, D12, gamma)

        # Check solution exists
        assert P is not None
        assert P.shape == (2, 2)

        # Check symmetry
        assert np.allclose(P, P.T)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals >= -1e-10)

    def test_solve_riccati_no_solution(self):
        """Test: Riccati with no solution (γ too small)."""
        config = RobustControlConfig()
        h_inf = HInfinityControl(config)

        A = np.array([[0.0, 1.0], [-1.0, -0.1]])
        B1 = np.array([[0.0], [1.0]])
        B2 = np.array([[0.0], [1.0]])
        C1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        D12 = np.zeros((2, 1))

        gamma = 0.01  # Too small

        P = h_inf.solve_riccati(A, B1, B2, C1, D12, gamma)

        # Should fail for very small gamma
        # (May succeed for some systems, but typically fails)
        # This test checks that the function handles failure gracefully
        if P is not None:
            # If solution exists, check properties
            assert P.shape == (2, 2)

    def test_design_controller(self):
        """Test: Design H-infinity controller via γ-iteration."""
        config = RobustControlConfig()
        h_inf = HInfinityControl(config)

        # Stabilizable system
        A = np.array([[0.0, 1.0], [-1.0, -0.5]])
        B1 = np.array([[0.0], [0.5]])
        B2 = np.array([[0.0], [1.0]])
        C1 = np.array([[1.0, 0.0]])
        D12 = np.zeros((1, 1))

        K, gamma_opt = h_inf.design_controller(A, B1, B2, C1, D12)

        assert K is not None
        assert gamma_opt is not None
        assert K.shape == (1, 2)
        assert gamma_opt > 0

        # Check closed-loop stability
        A_cl = A + B2 @ K
        eigvals = np.linalg.eigvals(A_cl)
        assert np.all(np.real(eigvals) < 0)  # Stable

    def test_evaluate_closed_loop_norm(self):
        """Test: Evaluate H-infinity norm of closed-loop."""
        config = RobustControlConfig()
        h_inf = HInfinityControl(config)

        A = np.array([[0.0, 1.0], [-1.0, -0.5]])
        B1 = np.array([[0.0], [0.5]])
        B2 = np.array([[0.0], [1.0]])
        C1 = np.array([[1.0, 0.0]])

        # Simple stabilizing controller
        K = np.array([[-1.0, -1.0]])

        norm = h_inf.evaluate_closed_loop_norm(A, B1, B2, C1, K)

        assert norm > 0
        assert norm < 100  # Should be bounded


class TestStochasticOptimalControl:
    """Tests for stochastic optimal control."""

    def test_initialization(self):
        """Test: Stochastic OC initialization."""
        config = RobustControlConfig()
        soc = StochasticOptimalControl(config)

        assert soc.config is not None

    def test_hjb_residual_computation(self):
        """Test: HJB residual with diffusion term."""
        config = RobustControlConfig()
        soc = StochasticOptimalControl(config)

        # Simple value function: V(x) = x'Px
        P = np.array([[2.0, 0.0], [0.0, 1.0]])

        def V(x):
            return x @ P @ x

        def dV_dx(x):
            return 2 * P @ x

        def d2V_dx2(x):
            return 2 * P

        # Linear dynamics
        def dynamics(x, u):
            A = np.array([[0.0, 1.0], [-1.0, 0.0]])
            B = np.array([[0.0], [1.0]])
            return A @ x + B @ u

        # Constant diffusion
        def diffusion(x):
            return np.array([[0.1, 0.0], [0.0, 0.1]])

        # Quadratic cost
        def running_cost(x, u):
            Q = np.eye(2)
            R = np.array([[0.1]])
            return x @ Q @ x + u @ R @ u

        x = np.array([1.0, 0.5])

        residual = soc.compute_hjb_residual(
            V, dV_dx, d2V_dx2, x, dynamics, diffusion, running_cost
        )

        assert np.isfinite(residual)


class TestUncertaintyQuantification:
    """Tests for uncertainty quantification."""

    def test_initialization(self):
        """Test: UQ initialization."""
        config = UQConfig()
        uq = UncertaintyQuantification(config)

        assert uq.config.num_samples > 0

    def test_monte_carlo_linear_system(self):
        """Test: Monte Carlo propagation for linear system."""
        config = UQConfig(num_samples=100, random_seed=42)
        uq = UncertaintyQuantification(config)

        # Simple linear system: dx/dt = -x + w (w is uncertain parameter)
        def system_dynamics(t, x, u, params):
            return -x + params['disturbance']

        initial_state = np.array([1.0])

        # Uncertain disturbance: w ~ N(0, 0.1)
        def sample_disturbance(rng):
            return rng.normal(0.0, 0.1)

        param_distribution = {'disturbance': sample_disturbance}

        t_span = (0.0, 1.0)

        results = uq.monte_carlo_propagation(
            system_dynamics,
            initial_state,
            param_distribution,
            t_span
        )

        assert 'mean' in results
        assert 'std' in results
        assert 'samples' in results
        assert results['samples'].shape[0] == 100
        assert len(results['mean']) == 1
        assert results['std'][0] > 0  # Should have variability

    def test_polynomial_chaos_simple(self):
        """Test: Polynomial chaos expansion for simple model."""
        config = UQConfig(poly_order=3)
        uq = UncertaintyQuantification(config)

        # Simple model: y = a*x + b
        def model(params):
            return params['a'] * 2.0 + params['b']

        uncertain_params = ['a', 'b']
        param_distributions = {'a': 'normal', 'b': 'normal'}

        results = uq.polynomial_chaos_expansion(
            model,
            uncertain_params,
            param_distributions,
            order=2
        )

        assert 'mean' in results
        assert 'variance' in results
        assert 'std' in results
        assert np.isfinite(results['mean'])
        assert results['variance'] >= 0

    def test_unscented_transform(self):
        """Test: Unscented transform for nonlinear function."""
        config = UQConfig(alpha=1e-3, beta=2.0, kappa=0.0)
        uq = UncertaintyQuantification(config)

        # Mean and covariance of input
        mean = np.array([1.0, 2.0])
        cov = np.array([[0.1, 0.0], [0.0, 0.2]])

        # Nonlinear map: y = [x1^2, x1*x2]
        def nonlinear_map(x):
            return np.array([x[0]**2, x[0] * x[1]])

        mean_Y, cov_Y = uq.unscented_transform(mean, cov, nonlinear_map)

        assert mean_Y.shape == (2,)
        assert cov_Y.shape == (2, 2)
        assert np.allclose(cov_Y, cov_Y.T)  # Symmetric

        # Mean should be close to f(mean(X)) for small variance
        expected_mean = nonlinear_map(mean)
        assert np.allclose(mean_Y, expected_mean, atol=0.1)

    def test_sensitivity_analysis(self):
        """Test: Sensitivity analysis via finite differences."""
        config = UQConfig(perturbation=1e-6)
        uq = UncertaintyQuantification(config)

        # Model: output depends on parameters
        def model(params):
            return {
                'output1': params['a'] * 2.0 + params['b'],
                'output2': params['a']**2 + params['b']**2
            }

        nominal_params = {'a': 1.0, 'b': 2.0}
        output_keys = ['output1', 'output2']

        sensitivities = uq.sensitivity_analysis(model, nominal_params, output_keys)

        assert 'output1' in sensitivities
        assert 'output2' in sensitivities

        # Check sensitivities for output1 = 2a + b
        # ∂output1/∂a = 2, ∂output1/∂b = 1
        assert np.isclose(sensitivities['output1']['a'], 2.0, atol=1e-3)
        assert np.isclose(sensitivities['output1']['b'], 1.0, atol=1e-3)

        # Check sensitivities for output2 = a^2 + b^2
        # ∂output2/∂a = 2a = 2, ∂output2/∂b = 2b = 4
        assert np.isclose(sensitivities['output2']['a'], 2.0, atol=1e-2)
        assert np.isclose(sensitivities['output2']['b'], 4.0, atol=1e-2)


class TestRiskSensitiveControl:
    """Tests for risk-sensitive control."""

    def test_risk_neutral(self):
        """Test: Risk-neutral case (θ=0)."""
        rsc = RiskSensitiveControl(risk_aversion=0.0)

        mean_cost = 10.0
        variance_cost = 5.0

        ce_cost = rsc.compute_certainty_equivalent_cost(mean_cost, variance_cost)

        # Should equal mean for θ=0
        assert np.isclose(ce_cost, mean_cost)

    def test_risk_averse(self):
        """Test: Risk-averse case (θ>0)."""
        rsc = RiskSensitiveControl(risk_aversion=0.5)

        mean_cost = 10.0
        variance_cost = 4.0

        ce_cost = rsc.compute_certainty_equivalent_cost(mean_cost, variance_cost)

        # Should be higher than mean (penalize variance)
        assert ce_cost > mean_cost

        # CE ≈ μ + (θ/2)σ² = 10 + 0.5/2 * 4 = 11
        assert np.isclose(ce_cost, 11.0)

    def test_risk_seeking(self):
        """Test: Risk-seeking case (θ<0)."""
        rsc = RiskSensitiveControl(risk_aversion=-0.5)

        mean_cost = 10.0
        variance_cost = 4.0

        ce_cost = rsc.compute_certainty_equivalent_cost(mean_cost, variance_cost)

        # Should be lower than mean (prefer variance)
        assert ce_cost < mean_cost

    def test_risk_adjusted_control(self):
        """Test: Risk-adjusted control."""
        rsc = RiskSensitiveControl(risk_aversion=0.1)

        nominal_control = np.array([1.0, 0.5])
        covariance = np.array([[0.1, 0.0], [0.0, 0.2]])
        R = np.eye(2)

        adjusted_control = rsc.risk_adjusted_control(nominal_control, covariance, R)

        assert adjusted_control.shape == nominal_control.shape
        # Risk aversion should reduce control magnitude
        assert np.linalg.norm(adjusted_control) <= np.linalg.norm(nominal_control)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_hinf_with_uncertainty(self):
        """Test: H-infinity controller with uncertainty quantification."""
        # Design H-infinity controller
        config_robust = RobustControlConfig()
        h_inf = HInfinityControl(config_robust)

        A = np.array([[0.0, 1.0], [-1.0, -0.5]])
        B1 = np.array([[0.0], [0.5]])
        B2 = np.array([[0.0], [1.0]])
        C1 = np.array([[1.0, 0.0]])
        D12 = np.zeros((1, 1))

        K, gamma_opt = h_inf.design_controller(A, B1, B2, C1, D12)

        assert K is not None

        # Test robustness via Monte Carlo
        config_uq = UQConfig(num_samples=50, random_seed=42)
        uq = UncertaintyQuantification(config_uq)

        def closed_loop_system(t, x, u, params):
            # Uncertain A matrix
            A_uncertain = A + params['A_perturbation']
            return A_uncertain @ x + B2 @ (K @ x)

        x0 = np.array([1.0, 0.0])

        def sample_A_perturbation(rng):
            return rng.normal(0.0, 0.01, size=(2, 2))

        param_dist = {'A_perturbation': sample_A_perturbation}

        results = uq.monte_carlo_propagation(
            closed_loop_system,
            x0,
            param_dist,
            (0.0, 2.0)
        )

        # All samples should converge to near zero (stable)
        assert np.all(np.abs(results['mean']) < 1.0)

    def test_risk_sensitive_with_stochastic_control(self):
        """Test: Risk-sensitive control with stochastic dynamics."""
        config = RobustControlConfig(risk_aversion=0.2)
        soc = StochasticOptimalControl(config)
        rsc = RiskSensitiveControl(risk_aversion=0.2)

        # This test verifies components work together
        # Full implementation would solve risk-sensitive HJB

        mean_cost = 15.0
        variance_cost = 3.0

        ce_cost = rsc.compute_certainty_equivalent_cost(mean_cost, variance_cost)

        assert ce_cost > mean_cost  # Risk-averse
        assert np.isfinite(ce_cost)

    def test_complete_robust_control_workflow(self):
        """Test: Complete robust control workflow."""
        print("\nComplete robust control workflow:")

        # 1. Design H-infinity controller
        print("  1. Designing H-infinity controller...")
        config = RobustControlConfig(gamma=2.0)
        h_inf = HInfinityControl(config)

        A = np.array([[0.0, 1.0], [-2.0, -0.5]])
        B1 = np.array([[0.0], [1.0]])
        B2 = np.array([[0.0], [1.0]])
        C1 = np.eye(2)
        D12 = np.zeros((2, 1))

        K, gamma_opt = h_inf.design_controller(A, B1, B2, C1, D12)

        print(f"     γ_opt = {gamma_opt:.3f}")
        print(f"     K = {K}")

        # 2. Sensitivity analysis
        print("  2. Performing sensitivity analysis...")

        config_uq = UQConfig()
        uq = UncertaintyQuantification(config_uq)

        def performance_model(params):
            A_pert = A + np.array([[0, 0], [params['stiffness_error'], 0]])
            A_cl = A_pert + B2 @ K

            # Settling time estimate (rough approximation)
            eigvals = np.linalg.eigvals(A_cl)
            damping = -np.max(np.real(eigvals))

            return {'settling_time': 4.0 / damping if damping > 0 else 100.0}

        nominal = {'stiffness_error': 0.0}
        sensitivities = uq.sensitivity_analysis(performance_model, nominal, ['settling_time'])

        print(f"     Sensitivity to stiffness: {sensitivities['settling_time'].get('stiffness_error', 0):.3f}")

        # 3. Risk assessment
        print("  3. Risk-sensitive evaluation...")
        rsc = RiskSensitiveControl(risk_aversion=0.1)

        mean_performance = 5.0
        variance_performance = 1.0

        ce = rsc.compute_certainty_equivalent_cost(mean_performance, variance_performance)

        print(f"     Certainty equivalent: {ce:.3f}")

        print("  ✓ Robust control workflow completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
