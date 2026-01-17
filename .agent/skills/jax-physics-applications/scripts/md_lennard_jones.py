#!/usr/bin/env python3
"""
Complete JAX-MD Lennard-Jones Liquid Simulation

This script demonstrates a full molecular dynamics workflow:
1. System initialization
2. Equilibration phase
3. Production run with observables
4. Analysis (RDF, energy conservation)

Usage:
    python md_lennard_jones.py

Requirements:
    pip install jax jax-md matplotlib
"""

import jax
import jax.numpy as jnp
from jax_md import space, energy, simulate, quantity
import matplotlib.pyplot as plt


def main():
    # Configuration
    N = 500  # Number of particles
    box_size = 10.0
    dt = 0.005
    n_equilibration = 1000
    n_production = 10000
    sample_every = 100

    print("=" * 60)
    print("JAX-MD Lennard-Jones Liquid Simulation")
    print("=" * 60)
    print(f"Particles: {N}")
    print(f"Box size: {box_size}")
    print(f"Timestep: {dt}")
    print()

    # 1. Define simulation space (periodic boundary conditions)
    displacement_fn, shift_fn = space.periodic(box_size=box_size)

    # 2. Define energy function (Lennard-Jones potential)
    energy_fn = energy.lennard_jones_pair(
        displacement_fn,
        species=None,
        sigma=1.0,
        epsilon=1.0
    )

    # 3. Initialize system
    key = jax.random.PRNGKey(0)
    R = space.random_position(key, (N, 3), box_size=box_size)

    # 4. Create integrator (NVE ensemble - constant energy)
    init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, dt=dt)
    state = init_fn(key, R, mass=1.0)

    # JIT compile for performance
    @jax.jit
    def step_fn(state):
        return apply_fn(state)

    # 5. Equilibration phase
    print("Starting equilibration...")
    for step in range(n_equilibration):
        state = step_fn(state)

        if (step + 1) % 200 == 0:
            E = energy_fn(state.position)
            T = quantity.temperature(state, kB=1.0)
            print(f"  Equilibration step {step+1}/{n_equilibration}: "
                  f"E={E:.4f}, T={T:.4f}")

    print("✓ Equilibration complete\n")

    # 6. Production run with observables
    print("Starting production run...")
    positions_trajectory = []
    energies = []
    temperatures = []

    for step in range(n_production):
        state = step_fn(state)

        if step % sample_every == 0:
            # Sample observables
            positions_trajectory.append(state.position)
            E = energy_fn(state.position)
            T = quantity.temperature(state, kB=1.0)
            energies.append(E)
            temperatures.append(T)

            if (step + 1) % 1000 == 0:
                print(f"  Production step {step+1}/{n_production}: "
                      f"E={E:.4f}, T={T:.4f}")

    print("✓ Production run complete\n")

    # 7. Analysis
    print("=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    energies = jnp.array(energies)
    temperatures = jnp.array(temperatures)

    # Energy conservation check
    energy_mean = jnp.mean(energies)
    energy_std = jnp.std(energies)
    energy_drift = jnp.abs(energies[-1] - energies[0]) / jnp.abs(energy_mean)

    print(f"\nEnergy Statistics:")
    print(f"  Mean energy: {energy_mean:.6f}")
    print(f"  Std deviation: {energy_std:.6f}")
    print(f"  Relative drift: {energy_drift:.6f} ({energy_drift*100:.4f}%)")

    if energy_drift < 1e-4:
        print("  ✓ Energy conservation: EXCELLENT (drift < 0.01%)")
    elif energy_drift < 1e-3:
        print("  ✓ Energy conservation: GOOD (drift < 0.1%)")
    else:
        print("  ⚠ Energy conservation: POOR (consider smaller timestep)")

    # Temperature statistics
    temp_mean = jnp.mean(temperatures)
    temp_std = jnp.std(temperatures)

    print(f"\nTemperature Statistics:")
    print(f"  Mean temperature: {temp_mean:.6f}")
    print(f"  Std deviation: {temp_std:.6f}")
    print(f"  Relative fluctuation: {temp_std/temp_mean*100:.2f}%")

    # Compute radial distribution function
    print(f"\nComputing radial distribution function...")
    rdf, r_bins = compute_rdf(positions_trajectory[-1], box_size, displacement_fn)

    first_peak_idx = jnp.argmax(rdf[:20])
    first_peak_position = r_bins[first_peak_idx]
    lj_minimum = 2**(1/6)  # Expected first peak for LJ potential

    print(f"  First peak position: {first_peak_position:.3f}")
    print(f"  Expected (LJ minimum): {lj_minimum:.3f}")
    print(f"  Difference: {jnp.abs(first_peak_position - lj_minimum):.3f}")

    # 8. Visualization
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Energy time series
    axes[0, 0].plot(energies, 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].axhline(energy_mean, color='r', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Total Energy')
    axes[0, 0].set_title('Energy Conservation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Temperature time series
    axes[0, 1].plot(temperatures, 'g-', linewidth=1, alpha=0.7)
    axes[0, 1].axhline(temp_mean, color='r', linestyle='--', label='Mean')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Temperature')
    axes[0, 1].set_title('Temperature Fluctuations')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Radial distribution function
    axes[1, 0].plot(r_bins, rdf, 'k-', linewidth=2)
    axes[1, 0].axvline(lj_minimum, color='r', linestyle='--',
                       label=f'LJ minimum ({lj_minimum:.2f})')
    axes[1, 0].set_xlabel('Distance r')
    axes[1, 0].set_ylabel('g(r)')
    axes[1, 0].set_title('Radial Distribution Function')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 5)

    # Energy histogram
    axes[1, 1].hist(energies, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(energy_mean, color='r', linestyle='--',
                       linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Energy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Energy Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('md_lennard_jones_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: md_lennard_jones_results.png")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


def compute_rdf(positions, box_size, displacement_fn, n_bins=100, r_max=None):
    """Compute radial distribution function g(r)"""
    if r_max is None:
        r_max = box_size / 2

    N = positions.shape[0]
    bins = jnp.linspace(0, r_max, n_bins)
    dr = bins[1] - bins[0]

    # Compute all pairwise distances
    distances = []
    for i in range(N):
        for j in range(i+1, N):
            dr_vec = displacement_fn(positions[i], positions[j])
            r = jnp.sqrt(jnp.sum(dr_vec**2))
            distances.append(r)

    distances = jnp.array(distances)

    # Histogram distances
    hist, _ = jnp.histogram(distances, bins=bins)

    # Normalize by ideal gas
    r_bins = (bins[:-1] + bins[1:]) / 2
    volume_shell = 4 * jnp.pi * r_bins**2 * dr
    number_density = N / box_size**3
    ideal_count = number_density * volume_shell * N

    rdf = hist / ideal_count

    return rdf, r_bins


if __name__ == '__main__':
    main()
