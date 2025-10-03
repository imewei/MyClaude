"""Example: Monte Carlo Uncertainty Propagation.

This example demonstrates using the UncertaintyQuantificationAgent for
propagating uncertainty through a computational model using Monte Carlo sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.uncertainty_quantification_agent import UncertaintyQuantificationAgent


def example_spring_oscillator():
    """Propagate uncertainty in spring-mass system natural frequency."""
    print("="*70)
    print("Monte Carlo Uncertainty Propagation: Spring-Mass System")
    print("="*70)

    # Model: Natural frequency of spring-mass system
    # ω = sqrt(k/m)
    # where k = spring constant, m = mass

    def model(params):
        """Compute natural frequency from uncertain parameters."""
        k, m = params
        omega = np.sqrt(k / m)
        return omega

    # Input distributions (uncertain parameters)
    input_distributions = [
        {'type': 'normal', 'mean': 100.0, 'std': 10.0},  # k ~ N(100, 10²)
        {'type': 'normal', 'mean': 10.0, 'std': 1.0}      # m ~ N(10, 1²)
    ]

    print("\nInput Distributions:")
    print(f"  k (spring constant): N(100, 10²)")
    print(f"  m (mass): N(10, 1²)")

    # Analytical solution for comparison
    k_mean, k_std = 100.0, 10.0
    m_mean, m_std = 10.0, 1.0
    omega_mean_analytical = np.sqrt(k_mean / m_mean)  # Approximate

    # Create agent
    agent = UncertaintyQuantificationAgent()

    # Execute Monte Carlo
    n_samples = 10000
    print(f"\nRunning Monte Carlo with {n_samples} samples...")

    result = agent.execute({
        'problem_type': 'monte_carlo',
        'model': model,
        'input_distributions': input_distributions,
        'n_samples': n_samples,
        'seed': 42
    })

    if result.success:
        # Extract results
        solution = result.data['solution']
        mean = solution['mean']
        std = solution['std']
        variance = solution['variance']
        median = solution['median']
        ci = solution['confidence_interval']
        percentiles = solution['percentiles']

        print(f"\n✓ Monte Carlo Successful!")
        print(f"\nOutput Statistics (ω - natural frequency):")
        print(f"  Mean:     {mean:.3f} rad/s")
        print(f"  Median:   {median:.3f} rad/s")
        print(f"  Std Dev:  {std:.3f} rad/s")
        print(f"  Variance: {variance:.3f}")
        print(f"\n  95% Confidence Interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"\n  Percentiles:")
        print(f"    5%:  {percentiles['5%']:.3f}")
        print(f"    25%: {percentiles['25%']:.3f}")
        print(f"    50%: {percentiles['50%']:.3f}")
        print(f"    75%: {percentiles['75%']:.3f}")
        print(f"    95%: {percentiles['95%']:.3f}")

        # Coefficient of variation
        cv = std / mean * 100
        print(f"\n  Coefficient of Variation: {cv:.2f}%")

        # Higher order moments
        print(f"\n  Skewness: {solution['skewness']:.3f}")
        print(f"  Kurtosis: {solution['kurtosis']:.3f}")

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Subplot 1: Histogram of output
        ax = axes[0, 0]
        samples = solution.get('samples', np.random.normal(mean, std, n_samples))
        ax.hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
        ax.axvline(ci[0], color='orange', linestyle=':', linewidth=2, label='95% CI')
        ax.axvline(ci[1], color='orange', linestyle=':', linewidth=2)
        ax.set_xlabel('Natural Frequency ω (rad/s)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Output Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Subplot 2: Input-Output scatter (k vs ω)
        ax = axes[0, 1]
        k_samples = np.random.normal(k_mean, k_std, 1000)
        m_samples = np.random.normal(m_mean, m_std, 1000)
        omega_samples = np.sqrt(k_samples / m_samples)
        ax.scatter(k_samples, omega_samples, alpha=0.3, s=10)
        ax.set_xlabel('Spring Constant k (N/m)')
        ax.set_ylabel('Natural Frequency ω (rad/s)')
        ax.set_title('Input-Output Relationship: k → ω')
        ax.grid(True, alpha=0.3)

        # Subplot 3: Input-Output scatter (m vs ω)
        ax = axes[1, 0]
        ax.scatter(m_samples, omega_samples, alpha=0.3, s=10)
        ax.set_xlabel('Mass m (kg)')
        ax.set_ylabel('Natural Frequency ω (rad/s)')
        ax.set_title('Input-Output Relationship: m → ω')
        ax.grid(True, alpha=0.3)

        # Subplot 4: Box plot summary
        ax = axes[1, 1]
        box_data = [samples]
        bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Natural Frequency ω (rad/s)')
        ax.set_xticklabels(['ω'])
        ax.set_title('Distribution Summary (Box Plot)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add text with statistics
        stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nCV: {cv:.1f}%"
        ax.text(1.3, mean, stats_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.savefig('uq_monte_carlo_example.png', dpi=150)
        print(f"\n✓ Plot saved as 'uq_monte_carlo_example.png'")

        return True
    else:
        print(f"\n✗ Monte Carlo failed: {result.errors}")
        return False


if __name__ == "__main__":
    success = example_spring_oscillator()
    sys.exit(0 if success else 1)
