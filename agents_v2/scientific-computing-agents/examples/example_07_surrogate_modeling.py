"""Example: Surrogate Modeling Agent

Demonstrates:
1. Gaussian Process Regression for expensive function approximation
2. Polynomial Chaos Expansion for uncertainty quantification
3. Kriging interpolation for spatial data
4. Reduced-Order Models via POD
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.surrogate_modeling_agent import SurrogateModelingAgent


def example_1_gp_regression():
    """Example 1: GP Regression for expensive function surrogate.

    Approximate an expensive function with few samples using GP regression.
    """
    print("\n" + "="*70)
    print("Example 1: Gaussian Process Regression")
    print("="*70)

    agent = SurrogateModelingAgent()

    # "Expensive" function to approximate: Rosenbrock
    def expensive_function(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    # Generate training data (few expensive evaluations)
    np.random.seed(42)
    X_train = np.random.uniform(-2, 2, size=(20, 2))
    y_train = np.array([expensive_function(x) for x in X_train])

    # Test grid
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])

    # Build GP surrogate
    result = agent.execute({
        'problem_type': 'gp_regression',
        'training_x': X_train,
        'training_y': y_train,
        'test_x': X_test,
        'kernel': 'matern',
        'length_scale': 0.5,
        'noise_level': 1e-4
    })

    print(f"\nGP Surrogate Model:")
    print(f"  Kernel: {result.data['metadata']['kernel']}")
    print(f"  Training samples: {result.data['metadata']['n_training']}")
    print(f"  Length scale: {result.data['metadata']['length_scale']}")
    print(f"  Noise level: {result.data['metadata']['noise_level']}")

    predictions = result.data['solution']['predictions']
    uncertainties = result.data['solution']['uncertainties']

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Predictions
    Z_pred = predictions.reshape(X1.shape)
    contour1 = ax1.contourf(X1, X2, np.log10(Z_pred + 1), levels=20, cmap='viridis')
    ax1.scatter(X_train[:, 0], X_train[:, 1], c='red', s=50, edgecolors='white',
               label=f'Training data (n={len(X_train)})', zorder=5)
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_title('GP Predictions (log₁₀)', fontsize=13)
    ax1.legend()
    plt.colorbar(contour1, ax=ax1)

    # Uncertainty
    Z_unc = uncertainties.reshape(X1.shape)
    contour2 = ax2.contourf(X1, X2, Z_unc, levels=20, cmap='plasma')
    ax2.scatter(X_train[:, 0], X_train[:, 1], c='white', s=50, edgecolors='black',
               label='Training data', zorder=5)
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_title('GP Uncertainty (std)', fontsize=13)
    ax2.legend()
    plt.colorbar(contour2, ax=ax2)

    plt.tight_layout()
    plt.savefig('gp_regression_surrogate.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: gp_regression_surrogate.png")

    print(f"\n✓ Surrogate model built with {len(X_train)} samples")
    print(f"  Uncertainty highest far from training data ✓")


def example_2_polynomial_chaos():
    """Example 2: Polynomial Chaos Expansion for uncertainty propagation.

    Propagate input uncertainty through a computational model.
    """
    print("\n" + "="*70)
    print("Example 2: Polynomial Chaos Expansion")
    print("="*70)

    agent = SurrogateModelingAgent()

    # Model: y = x₁² + 2*x₁*x₂ + x₂
    def model(x):
        return x[0]**2 + 2*x[0]*x[1] + x[1]

    # Generate samples from input distribution (standard normal)
    np.random.seed(42)
    n_samples = 200
    samples = np.random.randn(n_samples, 2)

    # Evaluate model
    function_values = np.array([model(x) for x in samples])

    # Build PCE
    result = agent.execute({
        'problem_type': 'polynomial_chaos',
        'samples': samples,
        'function_values': function_values,
        'polynomial_order': 3
    })

    print(f"\nPCE Surrogate:")
    print(f"  Polynomial order: {result.data['metadata']['polynomial_order']}")
    print(f"  Input dimensions: {result.data['metadata']['n_dimensions']}")
    print(f"  PCE coefficients: {result.data['metadata']['n_coefficients']}")

    coefficients = result.data['solution']['coefficients']
    sensitivity_indices = result.data['solution']['sensitivity_indices']

    print(f"\nSensitivity Indices (Sobol):")
    for key, value in sensitivity_indices.items():
        print(f"  {key}: {value:.4f}")

    # Plot PCE reconstruction vs true values
    pce_predictions = result.data['solution']['basis'] @ coefficients

    plt.figure(figsize=(10, 6))
    plt.scatter(function_values, pce_predictions, alpha=0.5, s=20)
    plt.plot([function_values.min(), function_values.max()],
             [function_values.min(), function_values.max()],
             'r--', linewidth=2, label='Perfect fit')
    plt.xlabel('True Model Output', fontsize=12)
    plt.ylabel('PCE Prediction', fontsize=12)
    plt.title('Polynomial Chaos Expansion: Prediction Accuracy', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pce_uncertainty_propagation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: pce_uncertainty_propagation.png")

    # Compute statistics
    mean_pce = coefficients[0]
    variance_pce = np.sum(coefficients[1:]**2)

    print(f"\nOutput Statistics (from PCE):")
    print(f"  Mean: {mean_pce:.4f}")
    print(f"  Variance: {variance_pce:.4f}")
    print(f"  Std Dev: {np.sqrt(variance_pce):.4f}")

    print(f"\n✓ Uncertainty quantified via PCE")


def example_3_kriging():
    """Example 3: Kriging for spatial interpolation.

    Interpolate sparse spatial measurements.
    """
    print("\n" + "="*70)
    print("Example 3: Kriging Interpolation")
    print("="*70)

    agent = SurrogateModelingAgent()

    # Generate sparse measurements on 2D domain
    np.random.seed(42)
    n_measurements = 25
    locations = np.random.uniform(0, 10, size=(n_measurements, 2))

    # True field: f(x, y) = sin(x) * cos(y)
    values = np.sin(locations[:, 0]) * np.cos(locations[:, 1])

    # Add noise
    values += np.random.normal(0, 0.1, size=n_measurements)

    # Create prediction grid
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    prediction_locations = np.column_stack([X.ravel(), Y.ravel()])

    # Kriging interpolation
    result = agent.execute({
        'problem_type': 'kriging',
        'locations': locations,
        'values': values,
        'prediction_locations': prediction_locations,
        'kernel': 'thin_plate_spline'
    })

    print(f"\nKriging Interpolation:")
    print(f"  Kernel: {result.data['metadata']['kernel']}")
    print(f"  Training points: {result.data['metadata']['n_training_points']}")
    print(f"  Spatial dimensions: {result.data['metadata']['spatial_dimensions']}")

    predictions = result.data['solution']['predictions']
    prediction_variance = result.data['solution']['prediction_variance']

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # True field
    Z_true = np.sin(X) * np.cos(Y)
    im1 = ax1.contourf(X, Y, Z_true, levels=20, cmap='RdBu_r')
    ax1.scatter(locations[:, 0], locations[:, 1], c='black', s=30,
               label='Measurements', zorder=5)
    ax1.set_title('True Field: sin(x)·cos(y)', fontsize=13)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.legend()
    plt.colorbar(im1, ax=ax1)

    # Kriging predictions
    Z_pred = predictions.reshape(X.shape)
    im2 = ax2.contourf(X, Y, Z_pred, levels=20, cmap='RdBu_r')
    ax2.scatter(locations[:, 0], locations[:, 1], c='black', s=30,
               label='Measurements', zorder=5)
    ax2.set_title('Kriging Interpolation', fontsize=13)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.legend()
    plt.colorbar(im2, ax=ax2)

    # Prediction variance
    Z_var = prediction_variance.reshape(X.shape)
    im3 = ax3.contourf(X, Y, Z_var, levels=20, cmap='YlOrRd')
    ax3.scatter(locations[:, 0], locations[:, 1], c='blue', s=30,
               edgecolors='white', label='Measurements', zorder=5)
    ax3.set_title('Prediction Uncertainty', fontsize=13)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.legend()
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig('kriging_spatial_interpolation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: kriging_spatial_interpolation.png")

    print(f"\n✓ Spatial field interpolated from {n_measurements} measurements")


def example_4_reduced_order_model():
    """Example 4: Reduced-Order Model via POD.

    Build ROM for time-varying PDE solution.
    """
    print("\n" + "="*70)
    print("Example 4: Reduced-Order Model (POD)")
    print("="*70)

    agent = SurrogateModelingAgent()

    # Generate snapshot matrix: 1D heat equation solutions at different times
    n_spatial = 100
    n_snapshots = 50
    x = np.linspace(0, 1, n_spatial)
    t = np.linspace(0, 1, n_snapshots)

    # Initial condition: u(x, 0) = sin(πx) + 0.5*sin(3πx)
    # Solution: u(x,t) = sin(πx)*exp(-π²t) + 0.5*sin(3πx)*exp(-9π²t)
    snapshots = np.zeros((n_spatial, n_snapshots))
    for i, ti in enumerate(t):
        snapshots[:, i] = (np.sin(np.pi * x) * np.exp(-np.pi**2 * ti) +
                          0.5 * np.sin(3 * np.pi * x) * np.exp(-9 * np.pi**2 * ti))

    # Build ROM
    n_modes = 3
    result = agent.execute({
        'problem_type': 'rom',
        'snapshots': snapshots,
        'n_modes': n_modes
    })

    print(f"\nReduced-Order Model:")
    print(f"  Full DOF: {result.data['metadata']['n_dof']}")
    print(f"  Snapshots: {result.data['metadata']['n_snapshots']}")
    print(f"  ROM modes: {result.data['metadata']['n_modes']}")
    print(f"  Energy captured: {result.data['metadata']['energy_captured']*100:.2f}%")
    print(f"  Reconstruction error: {result.data['metadata']['reconstruction_error']:.6f}")

    reduced_basis = result.data['solution']['reduced_basis']
    singular_values = result.data['solution']['singular_values']
    reconstructed = result.data['solution']['reconstructed']

    # Plot
    fig = plt.figure(figsize=(14, 10))

    # POD modes
    ax1 = plt.subplot(3, 2, 1)
    for i in range(min(3, n_modes)):
        ax1.plot(x, reduced_basis[:, i], label=f'Mode {i+1}', linewidth=2)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('Mode amplitude', fontsize=11)
    ax1.set_title('POD Modes', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Singular values
    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogy(singular_values, 'o-', markersize=8, linewidth=2)
    ax2.set_xlabel('Mode number', fontsize=11)
    ax2.set_ylabel('Singular value', fontsize=11)
    ax2.set_title('Singular Value Spectrum', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Original snapshots
    ax3 = plt.subplot(3, 2, 3)
    im3 = ax3.contourf(t, x, snapshots, levels=30, cmap='RdBu_r')
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Space (x)', fontsize=11)
    ax3.set_title('Original Solution Field', fontsize=13)
    plt.colorbar(im3, ax=ax3)

    # ROM reconstruction
    ax4 = plt.subplot(3, 2, 4)
    im4 = ax4.contourf(t, x, reconstructed, levels=30, cmap='RdBu_r')
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_ylabel('Space (x)', fontsize=11)
    ax4.set_title(f'ROM Reconstruction ({n_modes} modes)', fontsize=13)
    plt.colorbar(im4, ax=ax4)

    # Error
    ax5 = plt.subplot(3, 2, 5)
    error = snapshots - reconstructed
    im5 = ax5.contourf(t, x, error, levels=30, cmap='seismic')
    ax5.set_xlabel('Time', fontsize=11)
    ax5.set_ylabel('Space (x)', fontsize=11)
    ax5.set_title('Reconstruction Error', fontsize=13)
    plt.colorbar(im5, ax=ax5)

    # Snapshots comparison
    ax6 = plt.subplot(3, 2, 6)
    snap_indices = [0, 24, 49]
    for idx in snap_indices:
        ax6.plot(x, snapshots[:, idx], '--', label=f't={t[idx]:.2f} (original)', alpha=0.7)
        ax6.plot(x, reconstructed[:, idx], '-', label=f't={t[idx]:.2f} (ROM)', linewidth=2)
    ax6.set_xlabel('x', fontsize=11)
    ax6.set_ylabel('u(x, t)', fontsize=11)
    ax6.set_title('Snapshot Comparison', fontsize=13)
    ax6.legend(fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rom_pod_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: rom_pod_analysis.png")

    print(f"\n✓ ROM built: {n_spatial} DOF → {n_modes} modes")
    print(f"  Speedup potential: {n_spatial/n_modes:.1f}x")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("SURROGATE MODELING AGENT EXAMPLES")
    print("="*70)

    example_1_gp_regression()
    example_2_polynomial_chaos()
    example_3_kriging()
    example_4_reduced_order_model()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nGenerated plots:")
    print("  1. gp_regression_surrogate.png")
    print("  2. pce_uncertainty_propagation.png")
    print("  3. kriging_spatial_interpolation.png")
    print("  4. rom_pod_analysis.png")


if __name__ == "__main__":
    main()
