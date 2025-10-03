"""Pontryagin Maximum Principle Solver Demo.

This example demonstrates the PMP solver for optimal control problems
in both classical and quantum systems.

Key Features:
- Two-point boundary value problems (TPBVP)
- Single and multiple shooting methods
- Control constraints handling
- Quantum state transfer
- Optimal trajectories visualization

Run: python3 examples/pontryagin_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')

from solvers.pontryagin import PontryaginSolver, solve_quantum_control_pmp


def demo_1_linear_quadratic_regulator():
    """Demo 1: Classic LQR problem."""
    print("\n" + "="*70)
    print("Demo 1: Linear Quadratic Regulator (LQR)")
    print("="*70)

    # Problem: minimize ∫[x² + u²] dt
    # Dynamics: dx/dt = u
    # Initial: x(0) = 1, Final: x(T) = 0

    state_dim = 1
    control_dim = 1

    def dynamics(x, u, t):
        return u

    def running_cost(x, u, t):
        return x[0]**2 + u[0]**2

    solver = PontryaginSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost
    )

    x0 = np.array([1.0])
    xf = np.array([0.0])

    print("  Problem: Transfer from x=1 to x=0")
    print("  Cost: ∫[x² + u²] dt")

    result = solver.solve(
        x0=x0,
        xf=xf,
        duration=2.0,
        n_steps=50,
        method='single_shooting',
        verbose=True
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # State trajectory
    axes[0].plot(result['time'], result['state'][:, 0], 'b-', linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('State x')
    axes[0].set_title('State Trajectory')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Control trajectory
    axes[1].plot(result['time'], result['control'][:, 0], 'g-', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Control u')
    axes[1].set_title('Optimal Control')
    axes[1].grid(True, alpha=0.3)

    # Hamiltonian
    axes[2].plot(result['time'], result['hamiltonian'], 'r-', linewidth=2)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Hamiltonian H')
    axes[2].set_title('Hamiltonian (should be constant)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pontryagin_demo_1_lqr.png', dpi=150)
    print(f"  Plot saved: pontryagin_demo_1_lqr.png")


def demo_2_double_integrator():
    """Demo 2: Double integrator (position + velocity)."""
    print("\n" + "="*70)
    print("Demo 2: Double Integrator Control")
    print("="*70)

    # Dynamics: dx1/dt = x2, dx2/dt = u
    # Cost: ∫[x1² + x2² + 0.1u²] dt

    state_dim = 2
    control_dim = 1

    def dynamics(x, u, t):
        return np.array([x[1], u[0]])

    def running_cost(x, u, t):
        return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

    solver = PontryaginSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost
    )

    x0 = np.array([1.0, 0.5])
    xf = np.array([0.0, 0.0])

    print("  Problem: Double integrator")
    print(f"  Initial: position={x0[0]}, velocity={x0[1]}")
    print(f"  Target: position={xf[0]}, velocity={xf[1]}")

    result = solver.solve(
        x0=x0,
        xf=xf,
        duration=3.0,
        n_steps=60,
        method='multiple_shooting',
        verbose=True
    )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Phase portrait
    axes[0, 0].plot(result['state'][:, 0], result['state'][:, 1], 'b-', linewidth=2)
    axes[0, 0].plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    axes[0, 0].plot(xf[0], xf[1], 'ro', markersize=10, label='Target')
    axes[0, 0].set_xlabel('Position x₁')
    axes[0, 0].set_ylabel('Velocity x₂')
    axes[0, 0].set_title('Phase Portrait')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Position and velocity
    axes[0, 1].plot(result['time'], result['state'][:, 0], 'b-', linewidth=2, label='Position')
    axes[0, 1].plot(result['time'], result['state'][:, 1], 'r-', linewidth=2, label='Velocity')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('State')
    axes[0, 1].set_title('Position & Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Control
    axes[1, 0].plot(result['time'], result['control'][:, 0], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Control u')
    axes[1, 0].set_title('Optimal Control')
    axes[1, 0].grid(True, alpha=0.3)

    # Costate (adjoint)
    axes[1, 1].plot(result['time'], result['costate'][:, 0], 'b-', linewidth=2, label='λ₁')
    axes[1, 1].plot(result['time'], result['costate'][:, 1], 'r-', linewidth=2, label='λ₂')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Costate λ')
    axes[1, 1].set_title('Costate (Adjoint) Variables')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('pontryagin_demo_2_double_integrator.png', dpi=150)
    print(f"  Plot saved: pontryagin_demo_2_double_integrator.png")


def demo_3_constrained_control():
    """Demo 3: Control with constraints."""
    print("\n" + "="*70)
    print("Demo 3: Constrained Optimal Control")
    print("="*70)

    state_dim = 1
    control_dim = 1

    def dynamics(x, u, t):
        return u

    def running_cost(x, u, t):
        return x[0]**2 + 0.5 * u[0]**2

    # Control bounds: |u| ≤ 0.5
    u_min = np.array([-0.5])
    u_max = np.array([0.5])

    solver_constrained = PontryaginSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost,
        control_bounds=(u_min, u_max)
    )

    solver_unconstrained = PontryaginSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost
    )

    x0 = np.array([1.0])
    xf = np.array([0.0])

    print("  Comparing constrained vs unconstrained control")
    print(f"  Constraint: |u| ≤ {u_max[0]}")

    result_const = solver_constrained.solve(
        x0=x0, xf=xf, duration=3.0, n_steps=60,
        method='multiple_shooting', verbose=False
    )

    result_unconst = solver_unconstrained.solve(
        x0=x0, xf=xf, duration=3.0, n_steps=60,
        method='single_shooting', verbose=False
    )

    print(f"  Constrained cost: {result_const['cost']:.4f}")
    print(f"  Unconstrained cost: {result_unconst['cost']:.4f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # State comparison
    axes[0].plot(result_const['time'], result_const['state'][:, 0],
                'b-', linewidth=2, label='Constrained')
    axes[0].plot(result_unconst['time'], result_unconst['state'][:, 0],
                'r--', linewidth=2, label='Unconstrained')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('State x')
    axes[0].set_title('State Trajectories')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Control comparison
    axes[1].plot(result_const['time'], result_const['control'][:, 0],
                'b-', linewidth=2, label='Constrained')
    axes[1].plot(result_unconst['time'], result_unconst['control'][:, 0],
                'r--', linewidth=2, label='Unconstrained')
    axes[1].axhline(y=u_max[0], color='k', linestyle=':', alpha=0.5, label='Bounds')
    axes[1].axhline(y=u_min[0], color='k', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Control u')
    axes[1].set_title('Control Comparison')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('pontryagin_demo_3_constrained.png', dpi=150)
    print(f"  Plot saved: pontryagin_demo_3_constrained.png")


def demo_4_nonlinear_pendulum():
    """Demo 4: Nonlinear pendulum swing-up."""
    print("\n" + "="*70)
    print("Demo 4: Nonlinear Pendulum Swing-Up")
    print("="*70)

    state_dim = 2  # [angle, angular_velocity]
    control_dim = 1

    def dynamics(x, u, t):
        theta, omega = x
        g = 9.81
        L = 1.0
        b = 0.1  # Damping

        dtheta = omega
        domega = -(g / L) * np.sin(theta) - b * omega + u[0]

        return np.array([dtheta, domega])

    def running_cost(x, u, t):
        # Want to reach upright (θ=π)
        theta_target = np.pi
        return 0.5 * (x[0] - theta_target)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

    solver = PontryaginSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost,
        control_bounds=(np.array([-10.0]), np.array([10.0]))
    )

    x0 = np.array([0.0, 0.0])  # Hanging down

    print("  Problem: Swing pendulum from down (θ=0) toward up (θ=π)")
    print("  Nonlinear dynamics with damping")

    result = solver.solve(
        x0=x0,
        xf=None,  # Free endpoint (challenging to reach π exactly)
        duration=5.0,
        n_steps=100,
        method='multiple_shooting',
        verbose=True
    )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Angle trajectory
    axes[0, 0].plot(result['time'], result['state'][:, 0], 'b-', linewidth=2)
    axes[0, 0].axhline(y=np.pi, color='r', linestyle='--', alpha=0.5, label='Target (π)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Angle θ (rad)')
    axes[0, 0].set_title('Pendulum Angle')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Angular velocity
    axes[0, 1].plot(result['time'], result['state'][:, 1], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Angular velocity ω (rad/s)')
    axes[0, 1].set_title('Angular Velocity')
    axes[0, 1].grid(True, alpha=0.3)

    # Control torque
    axes[1, 0].plot(result['time'], result['control'][:, 0], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Control torque u')
    axes[1, 0].set_title('Control Input')
    axes[1, 0].grid(True, alpha=0.3)

    # Phase portrait
    axes[1, 1].plot(result['state'][:, 0], result['state'][:, 1], 'b-', linewidth=2)
    axes[1, 1].plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    axes[1, 1].plot(result['state'][-1, 0], result['state'][-1, 1], 'ro', markersize=10, label='Final')
    axes[1, 1].set_xlabel('Angle θ')
    axes[1, 1].set_ylabel('Angular velocity ω')
    axes[1, 1].set_title('Phase Portrait')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('pontryagin_demo_4_pendulum.png', dpi=150)
    print(f"  Plot saved: pontryagin_demo_4_pendulum.png")
    print(f"  Final angle: {result['state'][-1, 0]:.4f} rad ({result['state'][-1, 0]*180/np.pi:.1f}°)")


def demo_5_shooting_methods_comparison():
    """Demo 5: Compare single vs multiple shooting."""
    print("\n" + "="*70)
    print("Demo 5: Single vs Multiple Shooting Comparison")
    print("="*70)

    state_dim = 2
    control_dim = 1

    def dynamics(x, u, t):
        return np.array([x[1], u[0]])

    def running_cost(x, u, t):
        return x[0]**2 + x[1]**2 + 0.1 * u[0]**2

    solver = PontryaginSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost
    )

    x0 = np.array([1.0, 0.5])
    xf = np.array([0.0, 0.0])

    print("  Solving same problem with both methods...")

    result_single = solver.solve(
        x0=x0, xf=xf, duration=3.0, n_steps=50,
        method='single_shooting', verbose=False
    )

    result_multi = solver.solve(
        x0=x0, xf=xf, duration=3.0, n_steps=50,
        method='multiple_shooting', verbose=False
    )

    print(f"\n  Single Shooting:")
    print(f"    Converged: {result_single['converged']}")
    print(f"    Iterations: {result_single['iterations']}")
    print(f"    Cost: {result_single['cost']:.6f}")
    print(f"    Final error: {np.linalg.norm(result_single['state'][-1] - xf):.6e}")

    print(f"\n  Multiple Shooting:")
    print(f"    Converged: {result_multi['converged']}")
    print(f"    Iterations: {result_multi['iterations']}")
    print(f"    Cost: {result_multi['cost']:.6f}")
    print(f"    Final error: {np.linalg.norm(result_multi['state'][-1] - xf):.6e}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # State comparison
    axes[0].plot(result_single['time'], result_single['state'][:, 0],
                'b-', linewidth=2, label='Single Shooting')
    axes[0].plot(result_multi['time'], result_multi['state'][:, 0],
                'r--', linewidth=2, label='Multiple Shooting')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Position x₁')
    axes[0].set_title('State Trajectories')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Control comparison
    axes[1].plot(result_single['time'], result_single['control'][:, 0],
                'b-', linewidth=2, label='Single Shooting')
    axes[1].plot(result_multi['time'], result_multi['control'][:, 0],
                'r--', linewidth=2, label='Multiple Shooting')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Control u')
    axes[1].set_title('Control Comparison')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('pontryagin_demo_5_methods.png', dpi=150)
    print(f"  Plot saved: pontryagin_demo_5_methods.png")


def main():
    """Run all demos."""
    print("\n" + "#"*70)
    print("# Pontryagin Maximum Principle - Comprehensive Demo")
    print("#"*70)
    print("\nThis demo showcases the PMP solver for optimal control problems.")
    print("PMP transforms control problems into two-point boundary value problems.")

    # Run demos
    demo_1_linear_quadratic_regulator()
    demo_2_double_integrator()
    demo_3_constrained_control()
    demo_4_nonlinear_pendulum()
    demo_5_shooting_methods_comparison()

    print("\n" + "#"*70)
    print("# All demos complete!")
    print("#"*70)
    print("\nGenerated plots:")
    print("  - pontryagin_demo_1_lqr.png")
    print("  - pontryagin_demo_2_double_integrator.png")
    print("  - pontryagin_demo_3_constrained.png")
    print("  - pontryagin_demo_4_pendulum.png")
    print("  - pontryagin_demo_5_methods.png")
    print("\nKey takeaways:")
    print("  • PMP solves optimal control via costate equations")
    print("  • Multiple shooting more robust for complex problems")
    print("  • Handles nonlinear dynamics and control constraints")
    print("  • Applicable to classical and quantum systems")
    print()


if __name__ == '__main__':
    main()
