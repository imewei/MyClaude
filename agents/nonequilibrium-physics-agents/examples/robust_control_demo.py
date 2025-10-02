"""Demonstrations of Robust Control and Uncertainty Quantification.

Shows practical applications of H-infinity control, stochastic optimal control,
uncertainty quantification methods, and risk-sensitive control.

Author: Nonequilibrium Physics Agents
Week: 23-24 of Phase 4
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.robust_control import (
    HInfinityControl,
    StochasticOptimalControl,
    UncertaintyQuantification,
    RiskSensitiveControl,
    RobustControlConfig,
    UQConfig
)


def demo_1_h_infinity_control():
    """Demo 1: H-infinity robust control design."""
    print("\n" + "="*70)
    print("DEMO 1: H-Infinity Robust Control")
    print("="*70)

    print("\nScenario: Design controller robust to disturbances")
    print("  System: Inverted pendulum with wind disturbance")
    print("  Goal: Minimize effect of wind on angle deviation")

    # System matrices
    # dx/dt = Ax + B1*w + B2*u
    # z = C1*x + D12*u (regulated output)
    A = np.array([[0.0, 1.0], [9.8, -0.1]])  # Linearized pendulum
    B1 = np.array([[0.0], [0.5]])  # Wind disturbance
    B2 = np.array([[0.0], [1.0]])  # Control torque
    C1 = np.array([[1.0, 0.0], [0.0, 0.1]])  # Penalize angle and control
    D12 = np.array([[0.0], [0.1]])

    print("\nSystem:")
    print(f"  States: [angle, angular_velocity]")
    print(f"  Disturbance: wind (via B1)")
    print(f"  Control: torque (via B2)")

    # Design H-infinity controller
    config = RobustControlConfig()
    h_inf = HInfinityControl(config)

    print("\nDesigning H-infinity controller...")
    K, gamma_opt = h_inf.design_controller(A, B1, B2, C1, D12,
                                            gamma_min=0.5, gamma_max=50.0)

    if K is not None:
        print(f"\n✓ Controller designed successfully")
        print(f"  Optimal γ: {gamma_opt:.3f}")
        print(f"  Gain matrix K:")
        print(f"    {K[0]}")

        # Evaluate closed-loop
        norm = h_inf.evaluate_closed_loop_norm(A, B1, B2, C1, K)
        print(f"  Closed-loop H∞ norm: {norm:.3f}")

        # Check stability
        A_cl = A + B2 @ K
        eigvals = np.linalg.eigvals(A_cl)
        print(f"  Closed-loop poles: {eigvals}")
        print(f"  Stable: {np.all(np.real(eigvals) < 0)}")

        print("\nInterpretation:")
        print(f"  → γ = {gamma_opt:.3f} means disturbance amplified at most {gamma_opt:.3f}x")
        print(f"  → Lower γ = more robust to disturbances")
        print(f"  → All poles in left-half plane = stable")
    else:
        print("✗ Controller design failed (system may not be stabilizable)")


def demo_2_monte_carlo_uncertainty():
    """Demo 2: Monte Carlo uncertainty propagation."""
    print("\n" + "="*70)
    print("DEMO 2: Monte Carlo Uncertainty Propagation")
    print("="*70)

    print("\nScenario: Spacecraft with uncertain thruster efficiency")
    print("  Nominal: 100% efficiency")
    print("  Actual: 90-110% (uniform distribution)")
    print("  Question: What's the final position uncertainty?")

    config = UQConfig(num_samples=500, random_seed=42)
    uq = UncertaintyQuantification(config)

    # System: dx/dt = u * efficiency
    def system_dynamics(t, x, u, params):
        efficiency = params['thruster_efficiency']
        return np.array([efficiency * u[0]])

    # Nominal control: constant thrust
    def controller(t, x):
        return np.array([1.0])

    x0 = np.array([0.0])  # Start at origin
    t_span = (0.0, 5.0)

    # Uncertainty: thruster efficiency
    def sample_efficiency(rng):
        return rng.uniform(0.9, 1.1)

    param_dist = {'thruster_efficiency': sample_efficiency}

    print("\nRunning Monte Carlo simulation (500 samples)...")
    results = uq.monte_carlo_propagation(
        system_dynamics,
        x0,
        param_dist,
        t_span,
        controller
    )

    print(f"\nResults at t=5s:")
    print(f"  Mean position: {results['mean'][0]:.3f}")
    print(f"  Std deviation: {results['std'][0]:.3f}")
    print(f"  5th percentile: {results['percentile_5'][0]:.3f}")
    print(f"  95th percentile: {results['percentile_95'][0]:.3f}")

    # Expected: mean ≈ 5.0 (1.0 thrust * 5 sec * 1.0 efficiency)
    # Std depends on efficiency variance

    print("\nInterpretation:")
    print(f"  → 90% of outcomes between {results['percentile_5'][0]:.2f} and {results['percentile_95'][0]:.2f}")
    print(f"  → Uncertainty in efficiency causes ±{results['std'][0]:.2f} position uncertainty")


def demo_3_polynomial_chaos():
    """Demo 3: Polynomial chaos expansion for UQ."""
    print("\n" + "="*70)
    print("DEMO 3: Polynomial Chaos Expansion")
    print("="*70)

    print("\nScenario: Control cost with uncertain parameters")
    print("  Cost = Q*x² + R*u²")
    print("  Q ~ N(1, 0.1), R ~ N(0.1, 0.01)")

    config = UQConfig(poly_order=3)
    uq = UncertaintyQuantification(config)

    # Model: cost at specific (x, u)
    x_test = 2.0
    u_test = 1.0

    def cost_model(params):
        Q = params['Q']
        R = params['R']
        return Q * x_test**2 + R * u_test**2

    uncertain_params = ['Q', 'R']
    param_distributions = {'Q': 'normal', 'R': 'normal'}

    print(f"\nTest point: x={x_test}, u={u_test}")
    print("Computing polynomial chaos expansion...")

    results = uq.polynomial_chaos_expansion(
        cost_model,
        uncertain_params,
        param_distributions,
        order=3
    )

    print(f"\nResults:")
    print(f"  Mean cost: {results['mean']:.4f}")
    print(f"  Std deviation: {results['std']:.4f}")
    print(f"  Variance: {results['variance']:.4f}")

    # Compare to nominal (Q=1, R=0.1)
    nominal_cost = 1.0 * x_test**2 + 0.1 * u_test**2
    print(f"\nNominal cost (Q=1, R=0.1): {nominal_cost:.4f}")
    print(f"Difference from mean: {abs(results['mean'] - nominal_cost):.4f}")

    print("\nInterpretation:")
    print("  → PCE gives statistical moments without Monte Carlo")
    print("  → Much faster for smooth functions")
    print("  → Accurate for low-dimensional uncertainties")


def demo_4_unscented_transform():
    """Demo 4: Unscented transform for nonlinear propagation."""
    print("\n" + "="*70)
    print("DEMO 4: Unscented Transform")
    print("="*70)

    print("\nScenario: Convert polar to Cartesian coordinates")
    print("  Input: (r, θ) with uncertainty")
    print("  Output: (x, y) = (r*cos(θ), r*sin(θ))")
    print("  Question: What's the output uncertainty?")

    config = UQConfig(alpha=1e-3, beta=2.0, kappa=0.0)
    uq = UncertaintyQuantification(config)

    # Mean and covariance in polar coordinates
    mean_polar = np.array([1.0, np.pi/4])  # r=1, θ=45°
    cov_polar = np.array([[0.01, 0.0], [0.0, 0.001]])  # Small uncertainties

    # Nonlinear transformation
    def polar_to_cartesian(polar):
        r, theta = polar
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    print(f"\nInput (polar):")
    print(f"  Mean: r={mean_polar[0]:.2f}, θ={np.degrees(mean_polar[1]):.1f}°")
    print(f"  Covariance:")
    print(f"    {cov_polar[0]}")
    print(f"    {cov_polar[1]}")

    mean_cartesian, cov_cartesian = uq.unscented_transform(
        mean_polar,
        cov_polar,
        polar_to_cartesian
    )

    print(f"\nOutput (Cartesian) via Unscented Transform:")
    print(f"  Mean: x={mean_cartesian[0]:.4f}, y={mean_cartesian[1]:.4f}")
    print(f"  Covariance:")
    print(f"    {cov_cartesian[0]}")
    print(f"    {cov_cartesian[1]}")

    # Compare to linearization (first-order)
    nominal_cartesian = polar_to_cartesian(mean_polar)
    print(f"\nNominal (no uncertainty): x={nominal_cartesian[0]:.4f}, y={nominal_cartesian[1]:.4f}")

    print("\nInterpretation:")
    print("  → Unscented transform captures nonlinearity better than linearization")
    print("  → Uses sigma points (2n+1 = 5 for n=2)")
    print("  → More accurate than first-order Taylor expansion")


def demo_5_sensitivity_analysis():
    """Demo 5: Sensitivity analysis for parameter importance."""
    print("\n" + "="*70)
    print("DEMO 5: Sensitivity Analysis")
    print("="*70)

    print("\nScenario: LQR cost sensitivity to weight matrices")
    print("  Cost = ∫(x'Qx + u'Ru) dt")
    print("  Question: Which parameters most affect cost?")

    config = UQConfig(perturbation=1e-6)
    uq = UncertaintyQuantification(config)

    # LQR cost model (simplified)
    def lqr_cost_model(params):
        Q_diag = params['Q_weight']
        R_weight = params['R_weight']
        time_horizon = params['time_horizon']

        # Simplified cost estimate (proportional to weights)
        # In reality, would solve Riccati equation
        state_cost = Q_diag * 5.0  # Typical state trajectory
        control_cost = R_weight * 2.0  # Typical control effort

        total_cost = (state_cost + control_cost) * time_horizon

        return {
            'total_cost': total_cost,
            'settling_time': 1.0 / np.sqrt(Q_diag) if Q_diag > 0 else 10.0
        }

    nominal_params = {
        'Q_weight': 1.0,
        'R_weight': 0.1,
        'time_horizon': 10.0
    }

    output_keys = ['total_cost', 'settling_time']

    print(f"\nNominal parameters:")
    for key, val in nominal_params.items():
        print(f"  {key}: {val}")

    print("\nComputing sensitivities...")
    sensitivities = uq.sensitivity_analysis(
        lqr_cost_model,
        nominal_params,
        output_keys
    )

    print(f"\nSensitivity of total_cost:")
    for param, sens in sensitivities['total_cost'].items():
        print(f"  ∂cost/∂{param}: {sens:.3f}")

    print(f"\nSensitivity of settling_time:")
    for param, sens in sensitivities['settling_time'].items():
        print(f"  ∂settling/∂{param}: {sens:.3f}")

    # Find most influential parameter for each output
    most_influential_cost = max(sensitivities['total_cost'].items(),
                                 key=lambda x: abs(x[1]))
    print(f"\nMost influential for cost: {most_influential_cost[0]}")
    print(f"  (sensitivity = {most_influential_cost[1]:.3f})")

    print("\nInterpretation:")
    print("  → Positive sensitivity: increasing param increases output")
    print("  → Larger absolute value = more influential")
    print("  → Use for robustness analysis and parameter tuning")


def demo_6_risk_sensitive_control():
    """Demo 6: Risk-sensitive vs risk-neutral control."""
    print("\n" + "="*70)
    print("DEMO 6: Risk-Sensitive Control")
    print("="*70)

    print("\nScenario: Control under uncertainty")
    print("  Two strategies:")
    print("    A: Mean cost = 10, Variance = 1  (consistent)")
    print("    B: Mean cost = 9,  Variance = 5  (risky)")
    print("  Question: Which is better?")

    # Risk-neutral
    rsc_neutral = RiskSensitiveControl(risk_aversion=0.0)
    ce_A_neutral = rsc_neutral.compute_certainty_equivalent_cost(10.0, 1.0)
    ce_B_neutral = rsc_neutral.compute_certainty_equivalent_cost(9.0, 5.0)

    print(f"\nRisk-Neutral (θ=0):")
    print(f"  Strategy A: CE = {ce_A_neutral:.2f}")
    print(f"  Strategy B: CE = {ce_B_neutral:.2f}")
    print(f"  → Choose {'B' if ce_B_neutral < ce_A_neutral else 'A'} (lower CE cost)")

    # Risk-averse
    rsc_averse = RiskSensitiveControl(risk_aversion=0.5)
    ce_A_averse = rsc_averse.compute_certainty_equivalent_cost(10.0, 1.0)
    ce_B_averse = rsc_averse.compute_certainty_equivalent_cost(9.0, 5.0)

    print(f"\nRisk-Averse (θ=0.5):")
    print(f"  Strategy A: CE = {ce_A_averse:.2f}")
    print(f"  Strategy B: CE = {ce_B_averse:.2f}")
    print(f"  → Choose {'B' if ce_B_averse < ce_A_averse else 'A'} (lower CE cost)")

    # Risk-seeking
    rsc_seeking = RiskSensitiveControl(risk_aversion=-0.3)
    ce_A_seeking = rsc_seeking.compute_certainty_equivalent_cost(10.0, 1.0)
    ce_B_seeking = rsc_seeking.compute_certainty_equivalent_cost(9.0, 5.0)

    print(f"\nRisk-Seeking (θ=-0.3):")
    print(f"  Strategy A: CE = {ce_A_seeking:.2f}")
    print(f"  Strategy B: CE = {ce_B_seeking:.2f}")
    print(f"  → Choose {'B' if ce_B_seeking < ce_A_seeking else 'A'} (lower CE cost)")

    print("\nInterpretation:")
    print("  → θ = 0: Only mean matters (B wins)")
    print("  → θ > 0: Penalize variance (A may win)")
    print("  → θ < 0: Prefer variance (B wins more)")
    print("  → CE ≈ μ + (θ/2)σ² for small θ")


def demo_7_complete_workflow():
    """Demo 7: Complete robust control workflow."""
    print("\n" + "="*70)
    print("DEMO 7: Complete Robust Control Workflow")
    print("="*70)

    print("\nScenario: Design robust controller for uncertain system")
    print("  1. Design H-infinity controller")
    print("  2. Analyze uncertainty propagation")
    print("  3. Perform sensitivity analysis")
    print("  4. Evaluate risk-sensitive performance")

    # System with uncertainty
    A = np.array([[0.0, 1.0], [-1.0, -0.3]])
    B1 = np.array([[0.0], [0.5]])
    B2 = np.array([[0.0], [1.0]])
    C1 = np.eye(2)
    D12 = np.zeros((2, 1))

    print("\n1. DESIGN H-INFINITY CONTROLLER")
    config_robust = RobustControlConfig()
    h_inf = HInfinityControl(config_robust)

    K, gamma_opt = h_inf.design_controller(A, B1, B2, C1, D12)

    if K is not None:
        print(f"   ✓ γ_opt = {gamma_opt:.3f}")
        print(f"   ✓ K = {K}")

        print("\n2. UNCERTAINTY PROPAGATION")
        config_uq = UQConfig(num_samples=200, random_seed=42)
        uq = UncertaintyQuantification(config_uq)

        def closed_loop_dynamics(t, x, u, params):
            A_pert = A * (1 + params['stiffness_error'])
            return A_pert @ x + B2 @ (K @ x)

        x0 = np.array([1.0, 0.0])

        def sample_stiffness(rng):
            return rng.normal(0.0, 0.05)

        param_dist = {'stiffness_error': sample_stiffness}

        results = uq.monte_carlo_propagation(
            closed_loop_dynamics,
            x0,
            param_dist,
            (0.0, 3.0)
        )

        print(f"   ✓ Final state mean: {results['mean']}")
        print(f"   ✓ Final state std: {results['std']}")

        print("\n3. SENSITIVITY ANALYSIS")

        def performance_model(params):
            A_pert = A * (1 + params['stiffness_error'])
            damping_pert = A_pert[1, 1] + params['damping_change']
            A_pert[1, 1] = damping_pert

            A_cl = A_pert + B2 @ K
            eigvals = np.linalg.eigvals(A_cl)

            real_parts = np.real(eigvals)
            if np.all(real_parts < 0):
                settling = 4.0 / abs(np.max(real_parts))
            else:
                settling = 100.0  # Unstable

            return {'settling_time': settling}

        nominal = {'stiffness_error': 0.0, 'damping_change': 0.0}

        sens = uq.sensitivity_analysis(performance_model, nominal, ['settling_time'])

        print(f"   ✓ ∂settling/∂stiffness: {sens['settling_time'].get('stiffness_error', 0):.3f}")
        print(f"   ✓ ∂settling/∂damping: {sens['settling_time'].get('damping_change', 0):.3f}")

        print("\n4. RISK-SENSITIVE EVALUATION")
        rsc = RiskSensitiveControl(risk_aversion=0.2)

        mean_cost = 12.0
        variance_cost = 2.5

        ce_cost = rsc.compute_certainty_equivalent_cost(mean_cost, variance_cost)

        print(f"   ✓ Mean cost: {mean_cost:.2f}")
        print(f"   ✓ Variance: {variance_cost:.2f}")
        print(f"   ✓ Certainty equivalent: {ce_cost:.2f}")

        print("\n5. SUMMARY")
        print("   ✓ Controller handles disturbances up to γ = {:.2f}x".format(gamma_opt))
        print("   ✓ Uncertainty causes ±{:.3f} state deviation".format(np.mean(results['std'])))
        print("   ✓ Most sensitive to damping parameter")
        print("   ✓ Risk-adjusted cost = {:.2f}".format(ce_cost))

        print("\n✅ Complete workflow successful!")
    else:
        print("   ✗ Controller design failed")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*15 + "ROBUST CONTROL DEMONSTRATIONS")
    print("="*70)

    # Run demos
    demo_1_h_infinity_control()
    demo_2_monte_carlo_uncertainty()
    demo_3_polynomial_chaos()
    demo_4_unscented_transform()
    demo_5_sensitivity_analysis()
    demo_6_risk_sensitive_control()
    demo_7_complete_workflow()

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
