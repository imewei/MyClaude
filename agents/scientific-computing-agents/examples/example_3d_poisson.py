"""Example: 3D Poisson Equation.

Demonstrates solving the 3D Poisson equation:
∇²u = f(x,y,z)

Problem: 3D potential field from a charge distribution
- Domain: [-1,1] × [-1,1] × [-1,1]
- Source: 3D Gaussian charge distribution
- Boundary conditions: u = 0 on all boundaries
- Analytical solution: u(r) ≈ -1/r for point charge
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent


def main():
    """Solve and visualize 3D Poisson equation."""

    print("="*70)
    print("3D POISSON EQUATION EXAMPLE")
    print("="*70)
    print("\nProblem: ∇²u = f(x,y,z)")
    print("Physical interpretation: 3D electrostatic potential")
    print("Domain: [-1,1] × [-1,1] × [-1,1]")
    print("BC: u = 0 on all boundaries")
    print("Source: 3D Gaussian charge distribution at origin")
    print()

    # Source term: 3D Gaussian approximation of point charge
    def source_term(x, y, z):
        """3D Gaussian charge distribution centered at origin."""
        sigma = 0.15  # Width of Gaussian
        r_squared = x**2 + y**2 + z**2
        # 3D normalized Gaussian
        normalization = 1.0 / (sigma**3 * (2*np.pi)**(3/2))
        return -normalization * np.exp(-r_squared / (2 * sigma**2))

    # Solve using ODEPDESolverAgent
    print("Solving 3D Poisson equation...")
    print("  Grid: 30×30×30 (27,000 unknowns)")
    print("  Method: 7-point stencil finite difference")
    print("  Solver: Sparse direct solver")
    print()

    agent = ODEPDESolverAgent()

    result = agent.solve_poisson_3d({
        'domain': [[-1, 1], [-1, 1], [-1, 1]],
        'nx': 30,
        'ny': 30,
        'nz': 30,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    if result.success:
        sol = result.data['solution']
        U = sol['u']
        x = sol['x']
        y = sol['y']
        z = sol['z']

        print("✓ Solution complete!")
        print(f"  Computation time: {result.data.get('execution_time', 0):.3f} s")
        print(f"  Problem size: {U.size:,} unknowns")
        print()

        # Solution statistics
        print(f"Solution statistics:")
        print(f"  Min potential: {np.min(U):.6f}")
        print(f"  Max potential: {np.max(U):.6f}")
        center_idx = len(x) // 2
        print(f"  Potential at origin: {U[center_idx, center_idx, center_idx]:.6f}")
        print()

        # Verification: Compute total charge
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        dV = dx * dy * dz

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        F_grid = source_term(X, Y, Z)
        total_charge = np.sum(F_grid) * dV

        print(f"Verification:")
        print(f"  Total charge (integrated source): {total_charge:.6f}")
        print()

        # Visualization
        fig = plt.figure(figsize=(16, 10))

        # Create slice indices
        mid_idx = len(x) // 2

        # Subplot 1: XY slice (z=0)
        ax = fig.add_subplot(2, 3, 1)
        c = ax.contourf(X[:, :, mid_idx], Y[:, :, mid_idx], U[:, :, mid_idx],
                       levels=20, cmap='RdBu_r')
        ax.contour(X[:, :, mid_idx], Y[:, :, mid_idx], U[:, :, mid_idx],
                  levels=20, colors='k', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('XY Slice (z=0)')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        # Subplot 2: XZ slice (y=0)
        ax = fig.add_subplot(2, 3, 2)
        c = ax.contourf(X[:, mid_idx, :], Z[:, mid_idx, :], U[:, mid_idx, :],
                       levels=20, cmap='RdBu_r')
        ax.contour(X[:, mid_idx, :], Z[:, mid_idx, :], U[:, mid_idx, :],
                  levels=20, colors='k', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title('XZ Slice (y=0)')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        # Subplot 3: YZ slice (x=0)
        ax = fig.add_subplot(2, 3, 3)
        c = ax.contourf(Y[mid_idx, :, :], Z[mid_idx, :, :], U[mid_idx, :, :],
                       levels=20, cmap='RdBu_r')
        ax.contour(Y[mid_idx, :, :], Z[mid_idx, :, :], U[mid_idx, :, :],
                  levels=20, colors='k', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.set_title('YZ Slice (x=0)')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        # Subplot 4: Isosurface visualization (using scatter)
        ax = fig.add_subplot(2, 3, 4, projection='3d')

        # Create isosurface at specific potential level
        iso_level = 0.15
        mask = U > iso_level
        x_iso, y_iso, z_iso = X[mask], Y[mask], Z[mask]
        u_iso = U[mask]

        # Downsample for visualization
        step = max(1, len(x_iso) // 1000)
        scatter = ax.scatter(x_iso[::step], y_iso[::step], z_iso[::step],
                           c=u_iso[::step], cmap='viridis', s=10, alpha=0.6)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'Isosurface (u > {iso_level})')
        plt.colorbar(scatter, ax=ax, shrink=0.5)

        # Subplot 5: Radial profile
        ax = fig.add_subplot(2, 3, 5)

        # Extract radial profile along x-axis
        r_profile = x[mid_idx:]
        u_profile = U[mid_idx:, mid_idx, mid_idx]

        ax.plot(r_profile, u_profile, 'b-', linewidth=2, label='Numerical')

        # Theoretical: u(r) ~ -1/(4πr) for point charge
        r_theory = np.linspace(0.1, 1.0, 50)
        # Scale to match the Gaussian source strength
        u_theory = -0.15 / (r_theory)

        ax.plot(r_theory, u_theory, 'r--', linewidth=2, label='Analytical (1/r)')

        ax.set_xlabel('Distance from center r')
        ax.set_ylabel('Potential u(r)')
        ax.set_title('Radial Potential Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Subplot 6: Source term slice
        ax = fig.add_subplot(2, 3, 6)
        F_slice = F_grid[:, :, mid_idx]
        c = ax.contourf(X[:, :, mid_idx], Y[:, :, mid_idx], F_slice,
                       levels=20, cmap='hot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Source Term f(x,y,0)')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        plt.tight_layout()

        output_path = Path(__file__).parent / 'example_3d_poisson_output.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

        # Additional 3D visualization with multiple isosurfaces
        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111, projection='3d')

        # Plot multiple isosurface levels
        iso_levels = [0.05, 0.10, 0.15, 0.20]
        colors = ['blue', 'cyan', 'yellow', 'red']
        alphas = [0.2, 0.3, 0.4, 0.5]

        for iso_level, color, alpha in zip(iso_levels, colors, alphas):
            mask = (U > iso_level) & (U < iso_level + 0.02)
            if np.any(mask):
                x_iso, y_iso, z_iso = X[mask], Y[mask], Z[mask]
                # Downsample
                step = max(1, len(x_iso) // 300)
                ax.scatter(x_iso[::step], y_iso[::step], z_iso[::step],
                         c=color, s=5, alpha=alpha, label=f'u ≈ {iso_level}')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D Isosurfaces of Potential')
        ax.legend()

        # Set equal aspect ratio
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        output_path2 = Path(__file__).parent / 'example_3d_poisson_isosurfaces.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"✓ Isosurface visualization saved to: {output_path2}")

        return 0
    else:
        print(f"✗ Solution failed: {result.errors}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
