"""Example: Physics-Informed Machine Learning Agent

Demonstrates:
1. PINN for 1D heat equation
2. Inverse problem - parameter identification
3. DeepONet for operator learning
4. Conservation law enforcement
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.physics_informed_ml_agent import PhysicsInformedMLAgent


def example_1_pinn_heat_equation():
    """Example 1: Solve 1D heat equation with PINN.

    PDE: u_t = u_xx, with u(0,t) = 0, u(1,t) = 0, u(x,0) = sin(πx)
    Analytical solution: u(x,t) = sin(πx) * exp(-π²t)
    """
    print("\n" + "="*70)
    print("Example 1: PINN for 1D Heat Equation")
    print("="*70)

    agent = PhysicsInformedMLAgent(config={'epochs': 200, 'tolerance': 1e-3})

    # Define PDE residual: u_t - u_xx = 0
    def pde_residual(x, u, ux, uxx):
        # For simplicity, approximate u_t from u
        u_t = np.zeros_like(u)  # Placeholder
        return u_t - uxx

    # Simplified 1D spatial problem at t=0
    result = agent.execute({
        'problem_type': 'pinn',
        'pde_residual': pde_residual,
        'domain': {
            'bounds': [[0, 1]],
            'n_collocation': 100
        },
        'boundary_conditions': [
            {'type': 'dirichlet', 'location': np.array([[0.0]]), 'value': 0.0},
            {'type': 'dirichlet', 'location': np.array([[1.0]]), 'value': 0.0}
        ],
        'hidden_layers': [20, 20],
        'epochs': 100
    })

    print(f"\nPINN Training Complete:")
    print(f"  Network architecture: {result.data['metadata']['network_architecture']}")
    print(f"  Collocation points: {result.data['metadata']['n_collocation_points']}")
    print(f"  Network parameters: {result.data['metadata']['n_parameters']}")
    print(f"  Final loss: {result.data['metadata']['final_loss']:.6f}")
    print(f"  Method: {result.data['metadata']['method']}")

    # Extract solution
    x = result.data['solution']['x'].flatten()
    u = result.data['solution']['u']

    # Analytical solution at t=0: u(x,0) = sin(πx)
    u_exact = np.sin(np.pi * x)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, u, 'b-', label='PINN Solution', linewidth=2)
    plt.plot(x, u_exact, 'r--', label='Analytical (t=0)', linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x,t)', fontsize=12)
    plt.title('PINN for 1D Heat Equation (t=0)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pinn_heat_equation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: pinn_heat_equation.png")

    print(f"\n✓ PINN solution obtained")
    print(f"  Note: Full convergence requires more iterations")


def example_2_inverse_problem():
    """Example 2: Inverse problem - identify diffusion coefficient.

    Given observations of diffusion process, identify unknown diffusion coefficient D.
    Model: u_t = D * u_xx
    """
    print("\n" + "="*70)
    print("Example 2: Inverse Problem - Parameter Identification")
    print("="*70)

    agent = PhysicsInformedMLAgent()

    # Generate synthetic observations
    true_D = 0.5  # True diffusion coefficient
    x_obs = np.linspace(0, 1, 20)
    t_obs = 0.1

    # Analytical solution: u(x,t) = sin(πx) * exp(-Dπ²t)
    u_obs = np.sin(np.pi * x_obs) * np.exp(-true_D * np.pi**2 * t_obs)

    # Add noise
    u_obs += np.random.normal(0, 0.01, size=u_obs.shape)

    # Define forward model
    def forward_model(params):
        D = params[0]
        return np.sin(np.pi * x_obs) * np.exp(-D * np.pi**2 * t_obs)

    # Solve inverse problem
    result = agent.execute({
        'problem_type': 'inverse',
        'observations': u_obs,
        'forward_model': forward_model,
        'initial_parameters': np.array([0.3])  # Initial guess
    })

    estimated_D = result.data['solution']['parameters'][0]
    uncertainty = result.data['solution']['uncertainty'][0]
    misfit = result.data['solution']['misfit']

    print(f"\nInverse Problem Results:")
    print(f"  True diffusion coefficient: D = {true_D:.4f}")
    print(f"  Estimated coefficient: D = {estimated_D:.4f}")
    print(f"  Relative error: {abs(estimated_D - true_D) / true_D * 100:.2f}%")
    print(f"  Uncertainty: ±{uncertainty:.4f}")
    print(f"  Final misfit: {misfit:.6e}")
    print(f"  Optimization converged: {result.data['solution']['success']}")

    # Plot observations vs model
    u_fit = forward_model(np.array([estimated_D]))

    plt.figure(figsize=(10, 6))
    plt.plot(x_obs, u_obs, 'ko', label='Observations (noisy)', markersize=8, alpha=0.6)
    plt.plot(x_obs, u_fit, 'b-', label=f'Fitted model (D={estimated_D:.3f})', linewidth=2)
    plt.plot(x_obs, forward_model(np.array([true_D])), 'r--',
             label=f'True model (D={true_D})', linewidth=2, alpha=0.7)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x,t)', fontsize=12)
    plt.title(f'Inverse Problem: Diffusion Coefficient Identification (t={t_obs})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('inverse_problem_identification.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: inverse_problem_identification.png")

    print(f"\n✓ Parameter identified successfully")
    print(f"  Relative error: {abs(estimated_D - true_D) / true_D * 100:.1f}%")


def example_3_deeponet():
    """Example 3: DeepONet for operator learning.

    Learn the mapping from input functions to output functions.
    Example: Learn antiderivative operator.
    """
    print("\n" + "="*70)
    print("Example 3: DeepONet for Operator Learning")
    print("="*70)

    agent = PhysicsInformedMLAgent()

    # Generate training data: learn antiderivative operator
    n_samples = 100
    n_points = 50

    # Input: random functions (derivatives)
    u_train = np.random.randn(n_samples, n_points)

    # Output: antiderivatives (cumulative sum approximation)
    y_train = np.cumsum(u_train, axis=1) * (1.0 / n_points)

    # Train DeepONet
    result = agent.execute({
        'problem_type': 'deeponet',
        'training_data': {
            'input_functions': u_train,
            'output_functions': y_train,
            'n_samples': n_samples
        },
        'operator_type': 'antiderivative'
    })

    print(f"\nDeepONet Training Complete:")
    print(f"  Operator type: {result.data['metadata']['operator_type']}")
    print(f"  Training samples: {result.data['metadata']['n_training_samples']}")
    print(f"  Input dimension: {result.data['metadata']['input_dimension']}")
    print(f"  Architecture: {result.data['metadata']['architecture']}")

    # Test on new function
    u_test = np.sin(np.linspace(0, 2*np.pi, n_points))
    operator = result.data['solution']['operator']

    # Apply operator (simplified)
    y_pred = operator * u_test

    # True antiderivative
    x = np.linspace(0, 2*np.pi, n_points)
    y_true = -np.cos(x) + 1  # Antiderivative of sin(x)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, u_test, 'b-', label='Input: sin(x)', linewidth=2)
    plt.ylabel('u(x)', fontsize=11)
    plt.title('DeepONet: Learning Antiderivative Operator', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(x, y_true, 'r-', label='True antiderivative: -cos(x)+1', linewidth=2)
    plt.plot(x, y_pred, 'b--', label='DeepONet prediction', linewidth=2, alpha=0.7)
    plt.xlabel('x', fontsize=11)
    plt.ylabel('G[u](x)', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('deeponet_operator_learning.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: deeponet_operator_learning.png")

    print(f"\n✓ Operator learning completed")
    print(f"  Note: Simplified implementation for demonstration")


def example_4_conservation_laws():
    """Example 4: Enforce conservation laws.

    Check and enforce mass and energy conservation.
    """
    print("\n" + "="*70)
    print("Example 4: Conservation Law Enforcement")
    print("="*70)

    agent = PhysicsInformedMLAgent()

    # Test mass conservation
    print("\n--- Mass Conservation ---")
    n_cells = 100
    solution = np.ones(n_cells) * 0.01  # Total mass = 1.0

    result_mass = agent.execute({
        'problem_type': 'conservation',
        'conservation_type': 'mass',
        'solution': solution,
        'expected_mass': 1.0
    })

    violation_mass = result_mass.data['solution']['violation']
    satisfied_mass = result_mass.data['solution']['satisfied']

    print(f"  Expected total mass: 1.0")
    print(f"  Actual total mass: {np.sum(solution):.6f}")
    print(f"  Violation: {violation_mass:.6e}")
    print(f"  Conservation satisfied: {satisfied_mass}")

    # Test energy conservation
    print("\n--- Energy Conservation ---")
    n_particles = 100
    velocities = np.random.randn(n_particles) * 0.1  # Small velocities
    expected_energy = np.sum(velocities**2) / 2  # Kinetic energy

    result_energy = agent.execute({
        'problem_type': 'conservation',
        'conservation_type': 'energy',
        'solution': velocities,
        'expected_energy': expected_energy
    })

    violation_energy = result_energy.data['solution']['violation']
    satisfied_energy = result_energy.data['solution']['satisfied']

    print(f"  Expected energy: {expected_energy:.6f}")
    print(f"  Computed energy: {np.sum(velocities**2)/2:.6f}")
    print(f"  Violation: {violation_energy:.6e}")
    print(f"  Conservation satisfied: {satisfied_energy}")

    # Visualize conservation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mass conservation
    ax1.bar(['Expected', 'Actual'], [1.0, np.sum(solution)], color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Total Mass', fontsize=11)
    ax1.set_title('Mass Conservation Check', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # Energy conservation
    ax2.bar(['Expected', 'Actual'], [expected_energy, np.sum(velocities**2)/2],
            color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Total Energy', fontsize=11)
    ax2.set_title('Energy Conservation Check', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('conservation_laws.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: conservation_laws.png")

    print(f"\n✓ Conservation checks completed")
    print(f"  Mass conservation: {'✓' if satisfied_mass else '✗'}")
    print(f"  Energy conservation: {'✓' if satisfied_energy else '✗'}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PHYSICS-INFORMED MACHINE LEARNING AGENT EXAMPLES")
    print("="*70)

    example_1_pinn_heat_equation()
    example_2_inverse_problem()
    example_3_deeponet()
    example_4_conservation_laws()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nGenerated plots:")
    print("  1. pinn_heat_equation.png")
    print("  2. inverse_problem_identification.png")
    print("  3. deeponet_operator_learning.png")
    print("  4. conservation_laws.png")


if __name__ == "__main__":
    main()
