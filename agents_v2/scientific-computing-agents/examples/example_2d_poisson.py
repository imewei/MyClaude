"""Example: 2D Poisson Equation (Electrostatics).

Demonstrates solving the 2D Poisson equation:
∇²u = f(x,y)

Problem: Electric potential from a point charge
- Domain: [-1,1] × [-1,1]
- Source: f(x,y) = -4π δ(x,y) (approximated as Gaussian)
- Boundary conditions: u = 0 on all boundaries
- Analytical solution: u(x,y) ≈ -ln(r) for point charge at origin
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent


def main():
    """Solve and visualize 2D Poisson equation."""

    print("="*70)
    print("2D POISSON EQUATION EXAMPLE - ELECTROSTATICS")
    print("="*70)
    print("\nProblem: ∇²u = f(x,y)")
    print("Physical interpretation: Electric potential from charge distribution")
    print("Domain: [-1,1] × [-1,1]")
    print("BC: u = 0 on all boundaries")
    print("Source: Gaussian charge distribution at origin")
    print()

    # Source term: Gaussian approximation of point charge
    def source_term(X, Y):
        """Gaussian charge distribution centered at origin."""
        sigma = 0.1  # Width of Gaussian
        r_squared = X**2 + Y**2
        # Normalized Gaussian: ∫∫ f dA = 1
        return -(1.0 / (2 * np.pi * sigma**2)) * np.exp(-r_squared / (2 * sigma**2))

    # Analytical solution for point charge (approximate)
    def analytical_solution(X, Y):
        """Approximate analytical solution: u ≈ -ln(r)/2π for point charge."""
        r = np.sqrt(X**2 + Y**2)
        # Avoid singularity at origin
        r = np.maximum(r, 0.01)
        return -np.log(r) / (2 * np.pi) * 0.1  # Scaled to match source strength

    # Solve using ODEPDESolverAgent
    print("Solving 2D Poisson equation...")
    print("  Grid: 80×80")
    print("  Method: 5-point stencil finite difference")
    print()

    agent = ODEPDESolverAgent()

    result = agent.solve_pde_2d({
        'pde_type': 'poisson',
        'domain': [[-1, 1], [-1, 1]],
        'nx': 80,
        'ny': 80,
        'source_term': source_term,
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

        # Compute source term on grid
        F = source_term(X, Y)

        # Solution statistics
        print(f"Solution statistics:")
        print(f"  Min potential: {np.min(U_numerical):.6f}")
        print(f"  Max potential: {np.max(U_numerical):.6f}")
        print(f"  Potential at origin: {U_numerical[40, 40]:.6f}")
        print(f"  Total charge (integrated source): {np.sum(F) * (2.0/80)**2:.6f}")
        print()

        # Verification: check if ∇²u ≈ f by computing Laplacian
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        laplacian = np.zeros_like(U_numerical)
        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                d2u_dx2 = (U_numerical[i+1, j] - 2*U_numerical[i, j] + U_numerical[i-1, j]) / dx**2
                d2u_dy2 = (U_numerical[i, j+1] - 2*U_numerical[i, j] + U_numerical[i, j-1]) / dy**2
                laplacian[i, j] = d2u_dx2 + d2u_dy2

        # Compare Laplacian with source (interior points)
        interior_error = np.linalg.norm(laplacian[1:-1, 1:-1] - F[1:-1, 1:-1])
        print(f"Verification:")
        print(f"  ||∇²u - f|| (interior): {interior_error:.6e}")
        print()

        # Visualization
        fig = plt.figure(figsize=(16, 5))

        # Subplot 1: Potential (contour plot)
        ax = fig.add_subplot(131)
        levels = np.linspace(np.min(U_numerical), np.max(U_numerical), 20)
        c = ax.contourf(X, Y, U_numerical, levels=levels, cmap='RdBu_r')
        ax.contour(X, Y, U_numerical, levels=levels, colors='k', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Electric Potential u(x,y)')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax, label='Potential')

        # Subplot 2: Source term (charge distribution)
        ax = fig.add_subplot(132)
        c = ax.contourf(X, Y, F, levels=20, cmap='hot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Charge Distribution f(x,y)')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax, label='Charge density')

        # Subplot 3: 3D surface plot
        ax = fig.add_subplot(133, projection='3d')
        # Downsample for faster plotting
        skip = 4
        surf = ax.plot_surface(X[::skip, ::skip], Y[::skip, ::skip],
                              U_numerical[::skip, ::skip],
                              cmap='viridis', alpha=0.8, antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y)')
        ax.set_title('3D Potential Surface')
        fig.colorbar(surf, ax=ax, shrink=0.5)

        plt.tight_layout()

        output_path = Path(__file__).parent / 'example_2d_poisson_output.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

        # Additional plot: Electric field (gradient of potential)
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Compute electric field: E = -∇u
        Ex = np.zeros_like(U_numerical)
        Ey = np.zeros_like(U_numerical)

        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                Ex[i, j] = -(U_numerical[i+1, j] - U_numerical[i-1, j]) / (2 * dx)
                Ey[i, j] = -(U_numerical[i, j+1] - U_numerical[i, j-1]) / (2 * dy)

        E_magnitude = np.sqrt(Ex**2 + Ey**2)

        # Subplot 1: Electric field magnitude
        ax = axes[0]
        c = ax.contourf(X, Y, E_magnitude, levels=20, cmap='plasma')
        # Add field lines (streamplot)
        skip_stream = 5
        ax.streamplot(x[::skip_stream], y[::skip_stream],
                     Ex[::skip_stream, ::skip_stream].T,
                     Ey[::skip_stream, ::skip_stream].T,
                     color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Electric Field Magnitude |E| = |∇u|')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax, label='Field strength')

        # Subplot 2: Radial profile
        ax = axes[1]
        # Extract radial profile along x-axis (y=0)
        center_idx = len(y) // 2
        r_profile = x[len(x)//2:]  # Right half
        u_profile = U_numerical[len(x)//2:, center_idx]

        ax.plot(r_profile, u_profile, 'b-', linewidth=2, label='Numerical')
        # Theoretical: u(r) ~ -ln(r) for point charge
        r_theory = np.linspace(0.1, 1.0, 50)
        u_theory = -np.log(r_theory) / (2 * np.pi) * 0.1
        ax.plot(r_theory, u_theory, 'r--', linewidth=2, label='Analytical (approx)')

        ax.set_xlabel('Distance from center r')
        ax.set_ylabel('Potential u(r)')
        ax.set_title('Radial Potential Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path2 = Path(__file__).parent / 'example_2d_poisson_field.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"✓ Electric field visualization saved to: {output_path2}")

        return 0
    else:
        print(f"✗ Solution failed: {result.errors}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
