"""Example: Bayesian Inference for Parameter Estimation.

This example demonstrates using the InverseProblemsAgent for Bayesian inference
to estimate parameters from noisy observations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.inverse_problems_agent import InverseProblemsAgent


def example_linear_model():
    """Estimate parameters of a linear model y = ax + b from noisy data."""
    print("="*70)
    print("Bayesian Inference: Linear Model Parameter Estimation")
    print("="*70)

    # True parameters
    a_true = 2.5
    b_true = 1.0

    # Generate noisy observations
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_true = a_true * x + b_true
    noise = np.random.normal(0, 0.5, len(x))
    y_obs = y_true + noise

    print(f"\nTrue parameters: a={a_true}, b={b_true}")
    print(f"Observations: {len(y_obs)} noisy data points")

    # Create agent
    agent = InverseProblemsAgent()

    # Define forward model
    def forward_model(params):
        a, b = params
        return a * x + b

    # Prior: weakly informative
    prior_mean = np.array([1.0, 0.0])  # Weak guess
    prior_cov = np.array([[10.0, 0.0], [0.0, 10.0]])  # High uncertainty

    # Observation noise covariance
    obs_cov = 0.5**2 * np.eye(len(y_obs))

    # Execute Bayesian inference
    result = agent.execute({
        'problem_type': 'bayesian_inference',
        'forward_model': forward_model,
        'observations': y_obs,
        'prior_mean': prior_mean,
        'prior_covariance': prior_cov,
        'observation_covariance': obs_cov
    })

    if result.success:
        # Extract results
        posterior = result.data['solution']
        a_est, b_est = posterior['map_estimate']
        ci_a = posterior['credible_intervals'][0]
        ci_b = posterior['credible_intervals'][1]

        print(f"\n✓ Bayesian Inference Successful!")
        print(f"\nEstimated parameters:")
        print(f"  a = {a_est:.3f} (true: {a_true})")
        print(f"    95% CI: [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")
        print(f"  b = {b_est:.3f} (true: {b_true})")
        print(f"    95% CI: [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")

        # Errors
        error_a = abs(a_est - a_true) / a_true * 100
        error_b = abs(b_est - b_true)
        print(f"\nErrors:")
        print(f"  a: {error_a:.2f}%")
        print(f"  b: {error_b:.3f}")

        # Check if true values in credible intervals
        in_ci_a = ci_a[0] <= a_true <= ci_a[1]
        in_ci_b = ci_b[0] <= b_true <= ci_b[1]
        print(f"\nTrue values in 95% CI:")
        print(f"  a: {'✓' if in_ci_a else '✗'}")
        print(f"  b: {'✓' if in_ci_b else '✗'}")

        # Plot results
        plt.figure(figsize=(12, 5))

        # Subplot 1: Data and fits
        plt.subplot(1, 2, 1)
        plt.scatter(x, y_obs, alpha=0.5, label='Noisy observations')
        plt.plot(x, y_true, 'g--', label=f'True: y = {a_true}x + {b_true}', linewidth=2)
        plt.plot(x, forward_model(posterior['map_estimate']), 'r-',
                label=f'Estimated: y = {a_est:.2f}x + {b_est:.2f}', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Bayesian Parameter Estimation')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Parameter estimates with uncertainties
        plt.subplot(1, 2, 2)
        params = ['a (slope)', 'b (intercept)']
        true_vals = [a_true, b_true]
        estimates = [a_est, b_est]
        cis = [ci_a, ci_b]

        for i, (param, true_val, est, ci) in enumerate(zip(params, true_vals, estimates, cis)):
            plt.errorbar(i, est, yerr=[[est-ci[0]], [ci[1]-est]],
                        fmt='ro', capsize=10, capthick=2, label='Estimate' if i==0 else '')
            plt.plot(i, true_val, 'g*', markersize=15, label='True' if i==0 else '')

        plt.xticks([0, 1], params)
        plt.ylabel('Parameter Value')
        plt.title('Parameter Estimates with 95% CI')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('inverse_bayesian_example.png', dpi=150)
        print(f"\n✓ Plot saved as 'inverse_bayesian_example.png'")

        return True
    else:
        print(f"\n✗ Inference failed: {result.errors}")
        return False


if __name__ == "__main__":
    success = example_linear_model()
    sys.exit(0 if success else 1)
