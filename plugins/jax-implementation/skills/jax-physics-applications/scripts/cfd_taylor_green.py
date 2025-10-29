#!/usr/bin/env python3
"""
Complete JAX-CFD Taylor-Green Vortex Simulation

This script demonstrates a full CFD workflow for 2D incompressible flow:
1. Grid setup
2. Initial conditions (Taylor-Green vortex)
3. Time integration (incompressible Navier-Stokes)
4. Energy decay validation

The Taylor-Green vortex is a benchmark test case with analytical solution
for energy decay: E(t) = E₀ exp(-2νk²t)

Usage:
    python cfd_taylor_green.py

Requirements:
    pip install jax jax-cfd matplotlib numpy
"""

import jax
import jax.numpy as jnp
from jax_cfd import grids, finite_differences
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Configuration
    grid_size = 64
    Re = 100  # Reynolds number
    viscosity = 1.0 / Re
    dt = 0.001
    n_steps = 5000
    sample_every = 100

    print("=" * 60)
    print("JAX-CFD Taylor-Green Vortex Simulation")
    print("=" * 60)
    print(f"Grid size: {grid_size} x {grid_size}")
    print(f"Reynolds number: {Re}")
    print(f"Viscosity: {viscosity:.6f}")
    print(f"Timestep: {dt}")
    print(f"Total steps: {n_steps}")
    print()

    # 1. Define computational grid
    grid = grids.Grid((grid_size, grid_size),
                      domain=((0, 2*jnp.pi), (0, 2*jnp.pi)))

    # 2. Initial conditions (Taylor-Green vortex)
    print("Initializing Taylor-Green vortex...")
    velocity = taylor_green_ic(grid)
    u, v = velocity

    print(f"  Initial velocity field: u.shape={u.shape}, v.shape={v.shape}")
    print(f"  Initial kinetic energy: {compute_kinetic_energy(velocity):.6f}")
    print()

    # 3. Time integration
    print("Starting time integration...")
    times = []
    kinetic_energies = []
    enstrophies = []
    max_divergences = []

    for step in range(n_steps):
        velocity = navier_stokes_step(velocity, grid, viscosity, dt)

        if step % sample_every == 0:
            t = step * dt
            ke = compute_kinetic_energy(velocity)
            ens = compute_enstrophy(velocity, grid)
            div = compute_divergence_max(velocity, grid)

            times.append(t)
            kinetic_energies.append(ke)
            enstrophies.append(ens)
            max_divergences.append(div)

            if (step + 1) % 1000 == 0:
                print(f"  Step {step+1}/{n_steps}: t={t:.3f}, "
                      f"KE={ke:.6f}, Enstrophy={ens:.6f}, max|∇·u|={div:.2e}")

    print("✓ Time integration complete\n")

    # 4. Validation against analytical solution
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    times = jnp.array(times)
    kinetic_energies = jnp.array(kinetic_energies)
    enstrophies = jnp.array(enstrophies)
    max_divergences = jnp.array(max_divergences)

    # Analytical solution for energy decay
    k = 1.0  # Wavenumber
    E0 = kinetic_energies[0]
    analytical_energy = E0 * jnp.exp(-2 * viscosity * k**2 * times)

    # Compute error
    energy_error = jnp.abs(kinetic_energies - analytical_energy) / E0
    max_error = jnp.max(energy_error)
    mean_error = jnp.mean(energy_error)

    print(f"\nEnergy Decay Validation:")
    print(f"  Initial KE: {E0:.6f}")
    print(f"  Final KE: {kinetic_energies[-1]:.6f}")
    print(f"  Analytical final KE: {analytical_energy[-1]:.6f}")
    print(f"  Max relative error: {max_error:.6f} ({max_error*100:.4f}%)")
    print(f"  Mean relative error: {mean_error:.6f} ({mean_error*100:.4f}%)")

    if max_error < 0.01:
        print("  ✓ Energy decay validation: EXCELLENT (error < 1%)")
    elif max_error < 0.05:
        print("  ✓ Energy decay validation: GOOD (error < 5%)")
    else:
        print("  ⚠ Energy decay validation: POOR (consider smaller timestep)")

    # Mass conservation (divergence-free)
    max_div = jnp.max(max_divergences)
    print(f"\nMass Conservation (∇·u = 0):")
    print(f"  Maximum divergence: {max_div:.2e}")

    if max_div < 1e-10:
        print("  ✓ Mass conservation: EXCELLENT (machine precision)")
    elif max_div < 1e-6:
        print("  ✓ Mass conservation: GOOD")
    else:
        print("  ⚠ Mass conservation: POOR (check pressure solver)")

    # 5. Visualization
    print(f"\nGenerating plots...")
    fig = plt.figure(figsize=(14, 10))

    # Energy decay comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(times, kinetic_energies, 'b-', linewidth=2, label='Numerical')
    ax1.plot(times, analytical_energy, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Kinetic Energy')
    ax1.set_title('Energy Decay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative error
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(times, energy_error, 'g-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Energy Decay Error')
    ax2.grid(True, alpha=0.3)

    # Enstrophy
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, enstrophies, 'purple', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Enstrophy')
    ax3.set_title('Enstrophy Evolution')
    ax3.grid(True, alpha=0.3)

    # Velocity field at final time
    u, v = velocity
    x, y = grid.mesh()

    ax4 = plt.subplot(2, 3, 4)
    speed = jnp.sqrt(u**2 + v**2)
    im = ax4.contourf(x, y, speed, levels=20, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Final Velocity Magnitude')
    ax4.set_aspect('equal')
    plt.colorbar(im, ax=ax4)

    # Vorticity field at final time
    ax5 = plt.subplot(2, 3, 5)
    vorticity = compute_vorticity(velocity, grid)
    im = ax5.contourf(x, y, vorticity, levels=20, cmap='RdBu_r')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Final Vorticity Field')
    ax5.set_aspect('equal')
    plt.colorbar(im, ax=ax5)

    # Divergence check
    ax6 = plt.subplot(2, 3, 6)
    ax6.semilogy(times, max_divergences, 'orange', linewidth=2)
    ax6.axhline(1e-10, color='r', linestyle='--', label='Machine precision')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('max|∇·u|')
    ax6.set_title('Mass Conservation Check')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cfd_taylor_green_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: cfd_taylor_green_results.png")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


def taylor_green_ic(grid):
    """Initial conditions for Taylor-Green vortex"""
    x, y = grid.mesh()
    u = jnp.sin(x) * jnp.cos(y)
    v = -jnp.cos(x) * jnp.sin(y)
    return (u, v)


def navier_stokes_step(velocity, grid, viscosity, dt):
    """Single time step for 2D incompressible Navier-Stokes"""
    u, v = velocity

    # Advection term: -u·∇u
    u_advection = -advect_scalar(u, velocity, grid)
    v_advection = -advect_scalar(v, velocity, grid)

    # Diffusion term: ν∇²u
    u_diffusion = viscosity * laplacian(u, grid)
    v_diffusion = viscosity * laplacian(v, grid)

    # Predict velocity (before pressure projection)
    u_star = u + dt * (u_advection + u_diffusion)
    v_star = v + dt * (v_advection + v_diffusion)

    # Pressure projection (enforce incompressibility)
    u_new, v_new = pressure_projection((u_star, v_star), grid, dt)

    return (u_new, v_new)


def advect_scalar(scalar, velocity, grid):
    """Compute advection term u·∇φ using upwind scheme"""
    u, v = velocity
    dx, dy = grid.step

    # Central differences for gradients
    dscalar_dx = (jnp.roll(scalar, -1, axis=0) - jnp.roll(scalar, 1, axis=0)) / (2 * dx[0])
    dscalar_dy = (jnp.roll(scalar, -1, axis=1) - jnp.roll(scalar, 1, axis=1)) / (2 * dy[1])

    return u * dscalar_dx + v * dscalar_dy


def laplacian(scalar, grid):
    """Compute Laplacian ∇²φ using finite differences"""
    dx, dy = grid.step

    d2scalar_dx2 = (jnp.roll(scalar, -1, axis=0) - 2*scalar + jnp.roll(scalar, 1, axis=0)) / dx[0]**2
    d2scalar_dy2 = (jnp.roll(scalar, -1, axis=1) - 2*scalar + jnp.roll(scalar, 1, axis=1)) / dy[1]**2

    return d2scalar_dx2 + d2scalar_dy2


def pressure_projection(velocity, grid, dt):
    """Project velocity onto divergence-free space"""
    u, v = velocity
    dx, dy = grid.step

    # Compute divergence
    div = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dx[0]) + \
          (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dy[1])

    # Solve Poisson equation for pressure: ∇²p = ∇·u / dt
    # Using Jacobi iterations (simple solver)
    pressure = solve_poisson(div / dt, grid, n_iterations=100)

    # Correct velocity: u = u - dt·∇p
    dp_dx = (jnp.roll(pressure, -1, axis=0) - jnp.roll(pressure, 1, axis=0)) / (2 * dx[0])
    dp_dy = (jnp.roll(pressure, -1, axis=1) - jnp.roll(pressure, 1, axis=1)) / (2 * dy[1])

    u_corrected = u - dt * dp_dx
    v_corrected = v - dt * dp_dy

    return (u_corrected, v_corrected)


def solve_poisson(rhs, grid, n_iterations=100):
    """Solve ∇²p = rhs using Jacobi iterations"""
    dx, dy = grid.step
    p = jnp.zeros_like(rhs)

    for _ in range(n_iterations):
        p_new = 0.25 * (
            jnp.roll(p, -1, axis=0) + jnp.roll(p, 1, axis=0) +
            jnp.roll(p, -1, axis=1) + jnp.roll(p, 1, axis=1) -
            dx[0]**2 * rhs
        )
        p = p_new

    return p


def compute_kinetic_energy(velocity):
    """Compute total kinetic energy"""
    u, v = velocity
    return 0.5 * jnp.mean(u**2 + v**2)


def compute_enstrophy(velocity, grid):
    """Compute enstrophy (integral of vorticity squared)"""
    vorticity = compute_vorticity(velocity, grid)
    return 0.5 * jnp.mean(vorticity**2)


def compute_vorticity(velocity, grid):
    """Compute vorticity ω = ∂v/∂x - ∂u/∂y"""
    u, v = velocity
    dx, dy = grid.step

    du_dy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * dy[1])
    dv_dx = (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)) / (2 * dx[0])

    return dv_dx - du_dy


def compute_divergence_max(velocity, grid):
    """Compute maximum absolute divergence"""
    u, v = velocity
    dx, dy = grid.step

    div = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dx[0]) + \
          (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dy[1])

    return jnp.max(jnp.abs(div))


if __name__ == '__main__':
    main()
