"""Example: Sobol Sensitivity Analysis.

This example demonstrates using the UncertaintyQuantificationAgent for
variance-based global sensitivity analysis using Sobol indices.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.uncertainty_quantification_agent import UncertaintyQuantificationAgent


def example_ishigami_function():
    """Sensitivity analysis of the Ishigami function (benchmark test function)."""
    print("="*70)
    print("Sobol Sensitivity Analysis: Ishigami Function")
    print("="*70)

    # Ishigami function: f(x1, x2, x3) = sin(x1) + a*sin²(x2) + b*x3⁴*sin(x1)
    # Standard parameters: a=7, b=0.1
    # Known analytical Sobol indices for validation

    a, b = 7.0, 0.1

    def ishigami_function(x):
        """Ishigami function with known sensitivity indices."""
        x1, x2, x3 = x
        return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

    # Input ranges: all inputs uniform on [-π, π]
    input_ranges = [
        [-np.pi, np.pi],  # x1
        [-np.pi, np.pi],  # x2
        [-np.pi, np.pi]   # x3
    ]

    print("\nModel: Ishigami Function")
    print(f"  f(x1, x2, x3) = sin(x1) + {a}*sin²(x2) + {b}*x3⁴*sin(x1)")
    print(f"  Input ranges: x1, x2, x3 ∈ [-π, π]")

    # Analytical Sobol indices for comparison
    # (from literature, using a=7, b=0.1)
    V = a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18 + 1/2  # Total variance
    analytical_S1 = (b*np.pi**4/5 + b**2*np.pi**8/50 + 1/2) / V
    analytical_S2 = (a**2/8) / V
    analytical_S3 = 0  # x3 has no first-order effect
    analytical_ST1 = (b*np.pi**4/5 + b**2*np.pi**8/18 + 1/2) / V
    analytical_ST2 = analytical_S2  # No interactions for x2
    analytical_ST3 = (b**2*np.pi**8/18) / V  # Only interaction effects

    print("\nAnalytical Sobol Indices (from literature):")
    print(f"  First-order:")
    print(f"    S1 (x1): {analytical_S1:.3f}")
    print(f"    S2 (x2): {analytical_S2:.3f}")
    print(f"    S3 (x3): {analytical_S3:.3f}")
    print(f"  Total-order:")
    print(f"    ST1 (x1): {analytical_ST1:.3f}")
    print(f"    ST2 (x2): {analytical_ST2:.3f}")
    print(f"    ST3 (x3): {analytical_ST3:.3f}")

    # Create agent
    agent = UncertaintyQuantificationAgent()

    # Execute sensitivity analysis
    n_samples = 5000  # Saltelli scheme needs N*(2D+2) evaluations
    print(f"\nRunning Sobol analysis with {n_samples} base samples...")
    print(f"Total model evaluations: {n_samples * (2*3+2)} = {n_samples * 8}")

    result = agent.execute({
        'problem_type': 'sensitivity',
        'model': ishigami_function,
        'input_ranges': input_ranges,
        'n_samples': n_samples,
        'seed': 42
    })

    if result.success:
        # Extract results
        solution = result.data['solution']
        S = solution['first_order_indices']
        ST = solution['total_order_indices']

        print(f"\n✓ Sensitivity Analysis Successful!")
        print(f"\nComputed Sobol Indices:")
        print(f"  First-order:")
        print(f"    S1 (x1): {S['S1']:.3f}  (analytical: {analytical_S1:.3f})")
        print(f"    S2 (x2): {S['S2']:.3f}  (analytical: {analytical_S2:.3f})")
        print(f"    S3 (x3): {S['S3']:.3f}  (analytical: {analytical_S3:.3f})")
        print(f"  Total-order:")
        print(f"    ST1 (x1): {ST['ST1']:.3f}  (analytical: {analytical_ST1:.3f})")
        print(f"    ST2 (x2): {ST['ST2']:.3f}  (analytical: {analytical_ST2:.3f})")
        print(f"    ST3 (x3): {ST['ST3']:.3f}  (analytical: {analytical_ST3:.3f})")

        # Compute errors
        error_S1 = abs(S['S1'] - analytical_S1) / analytical_S1 * 100
        error_S2 = abs(S['S2'] - analytical_S2) / analytical_S2 * 100
        error_ST1 = abs(ST['ST1'] - analytical_ST1) / analytical_ST1 * 100

        print(f"\nRelative Errors:")
        print(f"  S1:  {error_S1:.1f}%")
        print(f"  S2:  {error_S2:.1f}%")
        print(f"  ST1: {error_ST1:.1f}%")

        # Interaction effects
        interaction = {
            'x1': ST['ST1'] - S['S1'],
            'x2': ST['ST2'] - S['S2'],
            'x3': ST['ST3'] - S['S3']
        }

        print(f"\nInteraction Effects (ST - S):")
        print(f"  x1: {interaction['x1']:.3f}  (has interactions)")
        print(f"  x2: {interaction['x2']:.3f}  (no interactions)")
        print(f"  x3: {interaction['x3']:.3f}  (only interactions)")

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Subplot 1: First-order indices comparison
        ax = axes[0, 0]
        variables = ['x1', 'x2', 'x3']
        S_computed = [S['S1'], S['S2'], S['S3']]
        S_analytical = [analytical_S1, analytical_S2, analytical_S3]

        x_pos = np.arange(len(variables))
        width = 0.35

        ax.bar(x_pos - width/2, S_computed, width, label='Computed', alpha=0.8)
        ax.bar(x_pos + width/2, S_analytical, width, label='Analytical', alpha=0.8)

        ax.set_ylabel('Sobol Index')
        ax.set_title('First-Order Sensitivity Indices (S)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Subplot 2: Total-order indices comparison
        ax = axes[0, 1]
        ST_computed = [ST['ST1'], ST['ST2'], ST['ST3']]
        ST_analytical = [analytical_ST1, analytical_ST2, analytical_ST3]

        ax.bar(x_pos - width/2, ST_computed, width, label='Computed', alpha=0.8)
        ax.bar(x_pos + width/2, ST_analytical, width, label='Analytical', alpha=0.8)

        ax.set_ylabel('Sobol Index')
        ax.set_title('Total-Order Sensitivity Indices (ST)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Subplot 3: Interaction effects
        ax = axes[1, 0]
        interactions = [interaction['x1'], interaction['x2'], interaction['x3']]

        ax.bar(x_pos, interactions, alpha=0.8, color='coral')
        ax.set_ylabel('Interaction Effect (ST - S)')
        ax.set_title('Interaction Effects by Variable')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Subplot 4: Pie chart of importance
        ax = axes[1, 1]
        sizes = [S['S1'], S['S2'], S['S3'], 1 - sum([S['S1'], S['S2'], S['S3']])]
        labels = ['x1 (main)', 'x2 (main)', 'x3 (main)', 'Interactions']
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        explode = (0.1, 0.05, 0, 0.05)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Variance Decomposition')

        plt.tight_layout()
        plt.savefig('uq_sensitivity_example.png', dpi=150)
        print(f"\n✓ Plot saved as 'uq_sensitivity_example.png'")

        # Interpretation
        print(f"\nInterpretation:")
        if S['S1'] > 0.3:
            print(f"  • x1 is highly important ({S['S1']:.1%} of variance)")
        if S['S2'] > 0.3:
            print(f"  • x2 is highly important ({S['S2']:.1%} of variance)")
        if S['S3'] < 0.05:
            print(f"  • x3 has negligible direct effect")
        if interaction['x1'] > 0.1:
            print(f"  • x1 has significant interactions")
        if interaction['x3'] > 0.05:
            print(f"  • x3 only affects output through interactions")

        return True
    else:
        print(f"\n✗ Sensitivity analysis failed: {result.errors}")
        return False


if __name__ == "__main__":
    success = example_ishigami_function()
    sys.exit(0 if success else 1)
