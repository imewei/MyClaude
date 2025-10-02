"""Example: Ensemble Kalman Filter for Data Assimilation.

This example demonstrates using the InverseProblemsAgent for sequential
data assimilation with the Ensemble Kalman Filter (EnKF).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.inverse_problems_agent import InverseProblemsAgent


def example_state_estimation():
    """Estimate state of a dynamical system from noisy observations."""
    print("="*70)
    print("Ensemble Kalman Filter: State Estimation")
    print("="*70)

    # True state (we want to estimate this)
    true_state = np.array([2.0, -1.0, 3.0])
    n_state = len(true_state)

    print(f"\nTrue state: {true_state}")

    # Generate ensemble (initial guess with uncertainty)
    np.random.seed(42)
    n_ensemble = 50
    ensemble = true_state[:, None] + np.random.normal(0, 1.0, (n_state, n_ensemble))

    print(f"Ensemble size: {n_ensemble} members")
    print(f"Initial ensemble mean: {ensemble.mean(axis=1)}")

    # Observation operator (observe all states with noise)
    def obs_operator(state):
        """H operator: what we actually measure."""
        return state  # Direct observation of all states

    # Generate noisy observations
    obs_noise_std = 0.3
    observations = true_state + np.random.normal(0, obs_noise_std, n_state)
    obs_cov = obs_noise_std**2 * np.eye(n_state)

    print(f"Observations: {observations}")
    print(f"Observation noise std: {obs_noise_std}")

    # Create agent
    agent = InverseProblemsAgent()

    # Execute EnKF
    result = agent.execute({
        'problem_type': 'ensemble_kalman_filter',
        'ensemble': ensemble,
        'observations': observations,
        'observation_operator': obs_operator,
        'observation_covariance': obs_cov
    })

    if result.success:
        # Extract results
        solution = result.data['solution']
        analysis_ensemble = solution['analysis_ensemble']
        analysis_mean = solution['analysis_mean']
        kalman_gain = solution['kalman_gain']

        print(f"\n✓ EnKF Update Successful!")
        print(f"\nAnalysis (updated) state:")
        print(f"  Mean: {analysis_mean}")
        print(f"  Std:  {analysis_ensemble.std(axis=1)}")

        # Compute errors
        forecast_error = np.linalg.norm(ensemble.mean(axis=1) - true_state)
        analysis_error = np.linalg.norm(analysis_mean - true_state)
        reduction = (forecast_error - analysis_error) / forecast_error * 100

        print(f"\nError Reduction:")
        print(f"  Forecast error: {forecast_error:.4f}")
        print(f"  Analysis error: {analysis_error:.4f}")
        print(f"  Reduction: {reduction:.1f}%")

        # Check Kalman gain properties
        print(f"\nKalman Gain:")
        print(f"  Shape: {kalman_gain.shape}")
        print(f"  Norm: {np.linalg.norm(kalman_gain):.3f}")

        # Plot results
        plt.figure(figsize=(14, 5))

        # Subplot 1: State estimates
        plt.subplot(1, 3, 1)
        states = ['State 1', 'State 2', 'State 3']
        x_pos = np.arange(len(states))

        # Forecast
        forecast_mean = ensemble.mean(axis=1)
        forecast_std = ensemble.std(axis=1)
        plt.errorbar(x_pos - 0.2, forecast_mean, yerr=forecast_std,
                    fmt='bo', capsize=5, label='Forecast')

        # Analysis
        analysis_std = analysis_ensemble.std(axis=1)
        plt.errorbar(x_pos, analysis_mean, yerr=analysis_std,
                    fmt='ro', capsize=5, label='Analysis')

        # True
        plt.plot(x_pos + 0.2, true_state, 'g*', markersize=15, label='True')

        plt.xticks(x_pos, states)
        plt.ylabel('State Value')
        plt.title('EnKF State Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Ensemble spread
        plt.subplot(1, 3, 2)
        for i in range(min(20, n_ensemble)):  # Plot subset for clarity
            plt.plot(range(n_state), ensemble[:, i], 'b-', alpha=0.3, linewidth=0.5)
            plt.plot(range(n_state), analysis_ensemble[:, i], 'r-', alpha=0.3, linewidth=0.5)

        plt.plot(range(n_state), forecast_mean, 'b-', linewidth=2, label='Forecast Mean')
        plt.plot(range(n_state), analysis_mean, 'r-', linewidth=2, label='Analysis Mean')
        plt.plot(range(n_state), true_state, 'g*-', markersize=10, linewidth=2, label='True')

        plt.xlabel('State Index')
        plt.ylabel('Value')
        plt.title('Ensemble Members')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 3: Kalman gain matrix
        plt.subplot(1, 3, 3)
        plt.imshow(kalman_gain, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Gain Value')
        plt.xlabel('Observation Index')
        plt.ylabel('State Index')
        plt.title('Kalman Gain Matrix')

        plt.tight_layout()
        plt.savefig('inverse_enkf_example.png', dpi=150)
        print(f"\n✓ Plot saved as 'inverse_enkf_example.png'")

        return True
    else:
        print(f"\n✗ EnKF failed: {result.errors}")
        return False


if __name__ == "__main__":
    success = example_state_estimation()
    sys.exit(0 if success else 1)
