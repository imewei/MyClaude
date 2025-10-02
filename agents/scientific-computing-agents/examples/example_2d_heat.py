"""Example: 2D Heat Equation.

Demonstrates solving the 2D heat equation:
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)

With:
- Initial condition: u(x,y,0) = sin(πx)sin(πy)
- Boundary conditions: u = 0 on all boundaries
- Analytical solution: u(x,y,t) = exp(-2απ²t)sin(πx)sin(πy)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent


def main():
    """Solve and visualize 2D heat equation."""

    print("="*70)
    print("2D HEAT EQUATION EXAMPLE")
    print("="*70)
    print("\nProblem: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)")
    print("Domain: [0,1] × [0,1]")
    print("BC: u = 0 on all boundaries")
    print("IC: u(x,y,0) = sin(πx)sin(πy)")
    print()

    # Problem parameters
    alpha = 0.01
    t_final = 0.5

    # Initial condition
    def initial_condition(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Analytical solution
    def analytical_solution(X, Y, t, alpha):
        return np.exp(-2 * alpha * np.pi**2 * t) * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Solve using ODEPDESolverAgent
    print("Solving 2D heat equation...")
    print(f"  α = {alpha}")
    print(f"  t_final = {t_final}")
    print(f"  Grid: 50×50")
    print()

    agent = ODEPDESolverAgent()

    result = agent.solve_pde_2d({
        'pde_type': 'heat',
        'domain': [[0, 1], [0, 1]],
        'nx': 50,
        'ny': 50,
        'alpha': alpha,
        'initial_condition': initial_condition,
        't_span': (0, t_final),
        'boundary_conditions': {'value': 0.0}
    })

    if result.success:
        sol = result.data['solution']
        U_numerical = sol['u']
        X = sol['X']
        Y = sol['Y']
        x = sol['x']
        y = sol['y']

        print("✓ Solution complete!")
        print(f"  Computation time: {result.data.get('execution_time', 0):.3f} s")
        print()

        # Compute analytical solution
        U_exact = analytical_solution(X, Y, t_final, alpha)

        # Compute error
        error = np.linalg.norm(U_numerical - U_exact) / np.linalg.norm(U_exact)

        print(f"Accuracy:")
        print(f"  Relative L2 error: {error:.6e}")
        print(f"  Max absolute error: {np.max(np.abs(U_numerical - U_exact)):.6e}")
        print()

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Numerical solution
        ax = axes[0]
        c = ax.contourf(X, Y, U_numerical, levels=20, cmap='hot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Numerical Solution at t={t_final}')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        # Analytical solution
        ax = axes[1]
        c = ax.contourf(X, Y, U_exact, levels=20, cmap='hot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Analytical Solution')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        # Error
        ax = axes[2]
        error_field = np.abs(U_numerical - U_exact)
        c = ax.contourf(X, Y, error_field, levels=20, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Absolute Error')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        plt.tight_layout()

        output_path = Path(__file__).parent / 'example_2d_heat_output.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

        return 0
    else:
        print(f"✗ Solution failed: {result.errors}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
