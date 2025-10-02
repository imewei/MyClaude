"""Collocation Methods Demonstrations.

This script demonstrates the collocation solver for optimal control problems.

Demos:
1. LQR problem - Classic linear-quadratic regulator
2. Double integrator - Position + velocity control
3. Constrained control - Box constraints on control
4. Collocation schemes comparison - Different methods
5. Nonlinear pendulum - Swingdown control

Author: Nonequilibrium Physics Agents
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from solvers.collocation import CollocationSolver


def demo_1_lqr():
    """Demo 1: Simple LQR problem."""
    print("\n" + "="*70)
    print("Demo 1: LQR Problem with Collocation")
    print("="*70)

    # Simple integrator: dx/dt = u
    def dynamics(x, u, t):
        return u

    # Quadratic cost: J = ∫(x² + u²)dt
    def cost(x, u, t):
        return x**2 + u**2

    # Create solver
    solver = CollocationSolver(
        state_dim=1,
        control_dim=1,
        dynamics=dynamics,
        running_cost=cost,
        collocation_type='gauss-legendre',
        collocation_order=3
    )

    # Solve
    print("\nSolving LQR problem...")
    result = solver.solve(
        x0=np.array([1.0]),
        duration=5.0,
        n_elements=20,
        verbose=True
    )

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(result['t'], result['x'], 'b-', linewidth=2, label='State x(t)')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('State')
    axes[0].set_title('LQR: State Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result['t'], result['u'], 'r-', linewidth=2, label='Control u(t)')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Control')
    axes[1].set_title('LQR: Optimal Control')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collocation_demo1_lqr.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: collocation_demo1_lqr.png")

    return result


def demo_2_double_integrator():
    """Demo 2: Double integrator control."""
    print("\n" + "="*70)
    print("Demo 2: Double Integrator")
    print("="*70)

    # State: [position, velocity]
    def dynamics(x, u, t):
        return np.array([x[1], u[0]])

    # Cost: minimize position^2 + velocity^2 + control effort
    def cost(x, u, t):
        return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

    solver = CollocationSolver(
        state_dim=2,
        control_dim=1,
        dynamics=dynamics,
        running_cost=cost,
        collocation_type='gauss-legendre',
        collocation_order=4
    )

    print("\nSolving double integrator...")
    result = solver.solve(
        x0=np.array([1.0, 0.5]),  # Start at position 1, velocity 0.5
        duration=5.0,
        n_elements=25,
        verbose=True
    )

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(result['t'], result['x'][:, 0], 'b-', linewidth=2, label='Position')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Position')
    axes[0].set_title('Double Integrator: Position')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result['t'], result['x'][:, 1], 'g-', linewidth=2, label='Velocity')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Double Integrator: Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result['t'], result['u'], 'r-', linewidth=2, label='Control')
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Control')
    axes[2].set_title('Double Integrator: Optimal Control')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collocation_demo2_double_integrator.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: collocation_demo2_double_integrator.png")

    return result


def demo_3_constrained_control():
    """Demo 3: Control with constraints."""
    print("\n" + "="*70)
    print("Demo 3: Constrained Control")
    print("="*70)

    def dynamics(x, u, t):
        return u

    def cost(x, u, t):
        return x**2

    # Control bounds: -1 ≤ u ≤ 1
    u_min = np.array([-1.0])
    u_max = np.array([1.0])

    solver = CollocationSolver(
        state_dim=1,
        control_dim=1,
        dynamics=dynamics,
        running_cost=cost,
        control_bounds=(u_min, u_max),
        collocation_type='gauss-legendre',
        collocation_order=3
    )

    print("\nSolving constrained problem...")
    print(f"Control bounds: [{u_min[0]}, {u_max[0]}]")

    result = solver.solve(
        x0=np.array([5.0]),  # Large initial state
        duration=10.0,
        n_elements=30,
        verbose=True
    )

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(result['t'], result['x'], 'b-', linewidth=2, label='State')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('State')
    axes[0].set_title('Constrained Control: State Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result['t'], result['u'], 'r-', linewidth=2, label='Control')
    axes[1].axhline(u_max[0], color='k', linestyle='--', alpha=0.5, label='Upper bound')
    axes[1].axhline(u_min[0], color='k', linestyle='--', alpha=0.5, label='Lower bound')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Control')
    axes[1].set_title('Constrained Control: Optimal Control (Bang-Bang)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collocation_demo3_constrained.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: collocation_demo3_constrained.png")

    # Verify constraints
    print(f"\nControl statistics:")
    print(f"  Min control: {np.min(result['u']):.6f} (bound: {u_min[0]})")
    print(f"  Max control: {np.max(result['u']):.6f} (bound: {u_max[0]})")
    print(f"  Constraints satisfied: {np.all(result['u'] >= u_min - 1e-6) and np.all(result['u'] <= u_max + 1e-6)}")

    return result


def demo_4_collocation_schemes():
    """Demo 4: Compare collocation schemes."""
    print("\n" + "="*70)
    print("Demo 4: Collocation Schemes Comparison")
    print("="*70)

    def dynamics(x, u, t):
        return u

    def cost(x, u, t):
        return x**2 + u**2

    schemes = {
        'gauss-legendre': 'Gauss-Legendre',
        'radau': 'Radau IIA',
        'hermite-simpson': 'Hermite-Simpson'
    }

    results = {}
    colors = {'gauss-legendre': 'b', 'radau': 'g', 'hermite-simpson': 'r'}

    print("\nSolving with different schemes...")

    for scheme_key, scheme_name in schemes.items():
        print(f"\n  {scheme_name}...")
        solver = CollocationSolver(
            state_dim=1,
            control_dim=1,
            dynamics=dynamics,
            running_cost=cost,
            collocation_type=scheme_key,
            collocation_order=3
        )

        result = solver.solve(
            x0=np.array([1.0]),
            duration=5.0,
            n_elements=15,
            verbose=False
        )
        results[scheme_key] = result
        print(f"    Cost: {result['cost']:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for scheme_key, scheme_name in schemes.items():
        result = results[scheme_key]
        axes[0].plot(result['t'], result['x'], colors[scheme_key] + '-',
                    linewidth=2, label=scheme_name)
        axes[1].plot(result['t'], result['u'], colors[scheme_key] + '-',
                    linewidth=2, label=scheme_name)

    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('State')
    axes[0].set_title('Collocation Schemes: State Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Control')
    axes[1].set_title('Collocation Schemes: Control Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collocation_demo4_schemes.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: collocation_demo4_schemes.png")

    return results


def demo_5_nonlinear_pendulum():
    """Demo 5: Nonlinear pendulum swingdown."""
    print("\n" + "="*70)
    print("Demo 5: Nonlinear Pendulum Swingdown")
    print("="*70)

    # State: [angle θ, angular velocity ω]
    # Dynamics: dθ/dt = ω, dω/dt = -sin(θ) + u
    def dynamics(x, u, t):
        theta, omega = x
        return np.array([omega, -np.sin(theta) + u[0]])

    # Cost: minimize deviation from downward position + control effort
    def cost(x, u, t):
        theta, omega = x
        # Want θ ≈ 0 (down), ω ≈ 0 (stopped)
        return theta**2 + omega**2 + 0.1 * u[0]**2

    # Control bounds
    u_min = np.array([-2.0])
    u_max = np.array([2.0])

    solver = CollocationSolver(
        state_dim=2,
        control_dim=1,
        dynamics=dynamics,
        running_cost=cost,
        control_bounds=(u_min, u_max),
        collocation_type='gauss-legendre',
        collocation_order=4
    )

    print("\nSolving pendulum swingdown...")
    print(f"Initial angle: 1.0 rad (~57°)")

    result = solver.solve(
        x0=np.array([1.0, 0.0]),  # Start at 57° from vertical
        duration=8.0,
        n_elements=40,
        verbose=True
    )

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(result['t'], result['x'][:, 0] * 180/np.pi, 'b-',
                linewidth=2, label='Angle')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3, label='Target (down)')
    axes[0].set_ylabel('Angle (degrees)')
    axes[0].set_title('Pendulum Swingdown: Angle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result['t'], result['x'][:, 1], 'g-',
                linewidth=2, label='Angular velocity')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_title('Pendulum Swingdown: Angular Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result['t'], result['u'], 'r-', linewidth=2, label='Torque')
    axes[2].axhline(u_max[0], color='k', linestyle='--', alpha=0.5)
    axes[2].axhline(u_min[0], color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Control Torque')
    axes[2].set_title('Pendulum Swingdown: Optimal Control')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collocation_demo5_pendulum.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: collocation_demo5_pendulum.png")

    print(f"\nFinal state:")
    print(f"  Angle: {result['x'][-1, 0] * 180/np.pi:.2f}°")
    print(f"  Angular velocity: {result['x'][-1, 1]:.4f} rad/s")

    return result


def main():
    """Run all collocation demonstrations."""
    print("\n" + "="*70)
    print("COLLOCATION METHODS - COMPREHENSIVE DEMONSTRATIONS")
    print("="*70)
    print("\nThese demos showcase orthogonal collocation for optimal control.")
    print("Collocation is an alternative to shooting methods, often more robust")
    print("for unstable systems or long time horizons.")

    try:
        # Run all demos
        demo_1_lqr()
        demo_2_double_integrator()
        demo_3_constrained_control()
        demo_4_collocation_schemes()
        demo_5_nonlinear_pendulum()

        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nGenerated plots:")
        print("  1. collocation_demo1_lqr.png")
        print("  2. collocation_demo2_double_integrator.png")
        print("  3. collocation_demo3_constrained.png")
        print("  4. collocation_demo4_schemes.png")
        print("  5. collocation_demo5_pendulum.png")

        print("\n" + "="*70)
        print("Key Takeaways:")
        print("="*70)
        print("1. Collocation converts continuous OCP to finite-dimensional NLP")
        print("2. Multiple collocation schemes available (Gauss-Legendre, Radau, Hermite-Simpson)")
        print("3. Handles constraints naturally via NLP bounds")
        print("4. More robust than shooting for unstable systems")
        print("5. Mesh refinement improves accuracy systematically")

    except Exception as e:
        print(f"\n✗ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
