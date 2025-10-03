"""Example: 2D Wave Equation.

Demonstrates solving the 2D wave equation:
∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)

Problem: Wave propagation on a membrane
- Domain: [0,1] × [0,1]
- Initial condition: Gaussian pulse
- Initial velocity: zero
- Boundary conditions: u = 0 (fixed edges)
- Analytical comparison: Energy conservation
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent


def main():
    """Solve and visualize 2D wave equation."""

    print("="*70)
    print("2D WAVE EQUATION EXAMPLE")
    print("="*70)
    print("\nProblem: ∂²u/∂t² = c²∇²u")
    print("Physical interpretation: Wave propagation on membrane")
    print("Domain: [0,1] × [0,1]")
    print("BC: u = 0 on all boundaries (fixed edges)")
    print("IC: Gaussian pulse at center, zero velocity")
    print()

    # Problem parameters
    c = 1.0  # Wave speed

    # Initial condition: Gaussian pulse
    def initial_condition(X, Y):
        """Gaussian pulse centered at (0.5, 0.5)."""
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        r_squared = (X - x0)**2 + (Y - y0)**2
        return np.exp(-r_squared / (2 * sigma**2))

    # Initial velocity: zero
    def initial_velocity(X, Y):
        """Zero initial velocity."""
        return np.zeros_like(X)

    # Solve using ODEPDESolverAgent
    print("Solving 2D wave equation...")
    print(f"  Wave speed c = {c}")
    print(f"  Grid: 60×60")
    print(f"  Time span: [0, 2.0]")
    print()

    agent = ODEPDESolverAgent()

    result = agent.solve_pde_2d({
        'pde_type': 'wave',
        'domain': [[0, 1], [0, 1]],
        'nx': 60,
        'ny': 60,
        'wave_speed': c,
        'initial_condition': initial_condition,
        'initial_velocity': initial_velocity,
        't_span': (0, 2.0),
        'boundary_conditions': {'value': 0.0}
    })

    if result.success:
        sol = result.data['solution']
        U_all = sol['u_all']
        V_all = sol['v_all']
        t = sol['t']
        X = sol['X']
        Y = sol['Y']
        x = sol['x']
        y = sol['y']

        print("✓ Solution complete!")
        print(f"  Computation time: {result.data.get('execution_time', 0):.3f} s")
        print(f"  Time steps: {len(t)}")
        print()

        # Energy analysis (kinetic + potential)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dA = dx * dy

        energies = []
        for k in range(len(t)):
            U_k = U_all[:, :, k]
            V_k = V_all[:, :, k]

            # Kinetic energy: (1/2) ∫ v² dA
            KE = 0.5 * np.sum(V_k**2) * dA

            # Potential energy: (1/2) c² ∫ |∇u|² dA
            grad_u_x = np.zeros_like(U_k)
            grad_u_y = np.zeros_like(U_k)
            for i in range(1, len(x)-1):
                for j in range(1, len(y)-1):
                    grad_u_x[i, j] = (U_k[i+1, j] - U_k[i-1, j]) / (2 * dx)
                    grad_u_y[i, j] = (U_k[i, j+1] - U_k[i, j-1]) / (2 * dy)

            PE = 0.5 * c**2 * np.sum(grad_u_x**2 + grad_u_y**2) * dA

            total_energy = KE + PE
            energies.append([KE, PE, total_energy])

        energies = np.array(energies)

        print(f"Energy conservation:")
        print(f"  Initial energy: {energies[0, 2]:.6f}")
        print(f"  Final energy: {energies[-1, 2]:.6f}")
        print(f"  Relative change: {abs(energies[-1, 2] - energies[0, 2]) / energies[0, 2]:.6e}")
        print()

        # Visualization: Snapshots at different times
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        # Select 8 time snapshots
        n_snapshots = 8
        snapshot_indices = np.linspace(0, len(t)-1, n_snapshots, dtype=int)

        vmax = np.max(np.abs(U_all))
        for idx, snap_idx in enumerate(snapshot_indices):
            ax = axes[idx]
            U_snap = U_all[:, :, snap_idx]
            t_snap = t[snap_idx]

            c = ax.contourf(X, Y, U_snap, levels=20, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
            ax.contour(X, Y, U_snap, levels=10, colors='k',
                      alpha=0.3, linewidths=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f't = {t_snap:.3f}')
            ax.set_aspect('equal')
            plt.colorbar(c, ax=ax)

        plt.tight_layout()

        output_path = Path(__file__).parent / 'example_2d_wave_snapshots.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Snapshots saved to: {output_path}")

        # Energy plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Energy components over time
        ax = axes[0]
        ax.plot(t, energies[:, 0], 'b-', label='Kinetic', linewidth=2)
        ax.plot(t, energies[:, 1], 'r-', label='Potential', linewidth=2)
        ax.plot(t, energies[:, 2], 'k--', label='Total', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Center point time series
        ax = axes[1]
        center_i = len(x) // 2
        center_j = len(y) // 2
        u_center = U_all[center_i, center_j, :]

        ax.plot(t, u_center, 'b-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Displacement u(0.5, 0.5, t)')
        ax.set_title('Center Point Time Series')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1)

        plt.tight_layout()

        output_path2 = Path(__file__).parent / 'example_2d_wave_analysis.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"✓ Analysis plots saved to: {output_path2}")

        # Cross-section view
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Subplot 1: X cross-section at different times
        ax = axes[0, 0]
        mid_j = len(y) // 2
        for snap_idx in snapshot_indices[::2]:  # Every other snapshot
            u_xsection = U_all[:, mid_j, snap_idx]
            ax.plot(x, u_xsection, label=f't={t[snap_idx]:.2f}', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, 0.5, t)')
        ax.set_title('X Cross-Section (y=0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Subplot 2: 3D surface at peak
        ax = axes[0, 1]
        ax.remove()
        ax = fig.add_subplot(222, projection='3d')

        # Find time of maximum displacement
        max_idx = np.argmax(np.abs(U_all))
        max_time_idx = max_idx // (len(x) * len(y))
        U_peak = U_all[:, :, min(max_time_idx, len(t)-1)]

        surf = ax.plot_surface(X[::3, ::3], Y[::3, ::3], U_peak[::3, ::3],
                              cmap='viridis', alpha=0.8, antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y,t)')
        ax.set_title(f'3D Surface at t={t[min(max_time_idx, len(t)-1)]:.2f}')
        fig.colorbar(surf, ax=ax, shrink=0.5)

        # Subplot 3: Velocity field at mid-time
        ax = axes[1, 0]
        mid_time_idx = len(t) // 2
        V_mid = V_all[:, :, mid_time_idx]
        c = ax.contourf(X, Y, V_mid, levels=20, cmap='seismic')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Velocity Field at t={t[mid_time_idx]:.2f}')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

        # Subplot 4: Quiver plot of velocity gradient
        ax = axes[1, 1]
        # Compute gradient of displacement at mid-time
        U_mid = U_all[:, :, mid_time_idx]
        grad_u_x = np.zeros_like(U_mid)
        grad_u_y = np.zeros_like(U_mid)
        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                grad_u_x[i, j] = (U_mid[i+1, j] - U_mid[i-1, j]) / (2 * dx)
                grad_u_y[i, j] = (U_mid[i, j+1] - U_mid[i, j-1]) / (2 * dy)

        # Quiver plot (downsampled)
        skip = 5
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 grad_u_x[::skip, ::skip], grad_u_y[::skip, ::skip])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Displacement Gradient at t={t[mid_time_idx]:.2f}')
        ax.set_aspect('equal')

        plt.tight_layout()

        output_path3 = Path(__file__).parent / 'example_2d_wave_crosssection.png'
        plt.savefig(output_path3, dpi=150, bbox_inches='tight')
        print(f"✓ Cross-section plots saved to: {output_path3}")

        return 0
    else:
        print(f"✗ Solution failed: {result.errors}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
