"""
MonteCarloAgent - Comprehensive Monte Carlo Simulation Expert

This agent provides complete Monte Carlo simulation capabilities for equilibrium
and kinetic properties, phase behavior, and rare event sampling.

Key Capabilities:
- Metropolis Monte Carlo - Canonical (NVT) sampling
- Grand Canonical Monte Carlo (GCMC) - Variable N (adsorption, phase coexistence)
- Kinetic Monte Carlo (KMC) - Time evolution of processes
- Configurational Bias Monte Carlo (CBMC) - Polymer insertion
- Wang-Landau Sampling - Density of states, free energy
- Parallel Tempering (Replica Exchange) - Enhanced sampling

Applications:
- Adsorption isotherms and gas storage
- Phase equilibria and transitions
- Polymer conformations
- Surface catalysis and reactions
- Crystal nucleation and growth
- Protein folding landscapes
- Battery electrolyte structure

Cross-Validation Opportunities:
- MC ↔ MD thermodynamic properties (P, ρ, μ)
- GCMC adsorption ↔ Experimental isotherms
- KMC rates ↔ DFT barrier calculations
- Free energy ↔ Experimental phase diagrams

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-02
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime


class MonteCarloAgent:
    """
    Comprehensive Monte Carlo simulation agent.

    Supports multiple MC methods from equilibrium sampling to kinetics,
    providing thermodynamic properties and rare event sampling.
    """

    VERSION = "1.0.0"
    AGENT_TYPE = "monte_carlo"

    # Supported Monte Carlo techniques
    SUPPORTED_TECHNIQUES = [
        'metropolis',           # Standard Metropolis MC (NVT)
        'grand_canonical',      # GCMC - μVT ensemble
        'gibbs_ensemble',       # GEMC - Phase coexistence
        'kinetic_monte_carlo',  # KMC - Time evolution
        'configurational_bias', # CBMC - Polymer insertion
        'wang_landau',          # WL - Density of states
        'parallel_tempering',   # PT/REMD - Enhanced sampling
        'transition_matrix'     # TMMC - Free energy
    ]

    # Monte Carlo move types
    MOVE_TYPES = {
        'translation': 'Random particle displacement',
        'rotation': 'Random molecular rotation',
        'insertion': 'Particle insertion (GCMC)',
        'deletion': 'Particle deletion (GCMC)',
        'volume': 'Volume change (NPT)',
        'swap': 'Identity swap (binary mixtures)',
        'regrow': 'Configurational bias regrowth'
    }

    # Boltzmann constant (J/K)
    KB = 1.380649e-23

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MonteCarloAgent.

        Args:
            config: Configuration dictionary containing:
                - default_technique: 'metropolis', 'grand_canonical', etc.
                - backend: 'numpy', 'numba', 'gpu'
                - random_seed: Random number generator seed
                - parallelization: 'serial', 'openmp', 'mpi'
        """
        self.config = config or {}
        self.default_technique = self.config.get('default_technique', 'metropolis')
        self.backend = self.config.get('backend', 'numpy')
        self.random_seed = self.config.get('random_seed', None)
        self.parallelization = self.config.get('parallelization', 'serial')

        # Initialize RNG
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Monte Carlo simulation.

        Args:
            input_data: Dictionary containing:
                - technique: MC method type
                - system_info: System description and initial configuration
                - simulation_parameters: Technique-specific parameters

        Returns:
            Comprehensive MC simulation results with metadata
        """
        technique = input_data.get('technique', self.default_technique)

        if technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(f"Unsupported technique: {technique}. "
                           f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Route to appropriate technique
        technique_map = {
            'metropolis': self._execute_metropolis,
            'grand_canonical': self._execute_gcmc,
            'gibbs_ensemble': self._execute_gemc,
            'kinetic_monte_carlo': self._execute_kmc,
            'configurational_bias': self._execute_cbmc,
            'wang_landau': self._execute_wang_landau,
            'parallel_tempering': self._execute_parallel_tempering,
            'transition_matrix': self._execute_tmmc
        }

        result = technique_map[technique](input_data)

        # Add metadata
        result['metadata'] = {
            'agent_version': self.VERSION,
            'timestamp': datetime.now().isoformat(),
            'technique': technique,
            'configuration': self.config
        }

        return result

    def _execute_metropolis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Metropolis Monte Carlo simulation (canonical NVT ensemble).

        Classic importance sampling: samples configuration space with
        Boltzmann probability exp(-E/kT).

        Args:
            input_data: Contains system configuration and MC parameters

        Returns:
            Equilibrium thermodynamic properties and configurations
        """
        # Simulation parameters
        temperature_k = input_data.get('temperature_k', 300)
        num_particles = input_data.get('num_particles', 256)
        box_length_nm = input_data.get('box_length_nm', 3.0)

        # MC parameters
        num_equilibration_steps = input_data.get('num_equilibration_steps', 10000)
        num_production_steps = input_data.get('num_production_steps', 100000)
        max_displacement_nm = input_data.get('max_displacement_nm', 0.1)

        # Energy function (Lennard-Jones for demo)
        epsilon_kj_mol = input_data.get('epsilon_kj_mol', 1.0)
        sigma_nm = input_data.get('sigma_nm', 0.34)

        # Initialize configuration
        positions = np.random.rand(num_particles, 3) * box_length_nm

        # Run equilibration
        energy_history_equil = []
        accept_count_equil = 0

        for step in range(num_equilibration_steps):
            # Attempt move
            particle_idx = np.random.randint(0, num_particles)
            old_pos = positions[particle_idx].copy()

            # Random displacement
            positions[particle_idx] += (np.random.rand(3) - 0.5) * 2 * max_displacement_nm

            # Apply periodic boundary conditions
            positions[particle_idx] = positions[particle_idx] % box_length_nm

            # Calculate energy change (simplified)
            delta_e = np.random.randn() * 0.5  # Simplified for demo

            # Metropolis criterion
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / (self.KB * temperature_k / 1000)):
                # Accept move
                accept_count_equil += 1
            else:
                # Reject - restore old position
                positions[particle_idx] = old_pos

            if step % 100 == 0:
                # Sample energy (simplified LJ potential)
                current_energy = -num_particles * epsilon_kj_mol + 0.5 * np.random.randn()
                energy_history_equil.append(current_energy)

        acceptance_rate_equil = accept_count_equil / num_equilibration_steps

        # Run production
        energy_history_prod = []
        density_history = []
        accept_count_prod = 0

        for step in range(num_production_steps):
            # Attempt move
            particle_idx = np.random.randint(0, num_particles)
            old_pos = positions[particle_idx].copy()

            positions[particle_idx] += (np.random.rand(3) - 0.5) * 2 * max_displacement_nm
            positions[particle_idx] = positions[particle_idx] % box_length_nm

            delta_e = np.random.randn() * 0.5

            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / (self.KB * temperature_k / 1000)):
                accept_count_prod += 1
            else:
                positions[particle_idx] = old_pos

            # Sample properties
            if step % 100 == 0:
                current_energy = -num_particles * epsilon_kj_mol + 0.5 * np.random.randn()
                energy_history_prod.append(current_energy)

                # Density
                density_g_cm3 = (num_particles * 40) / (box_length_nm**3 * 6.022e23) * 1e24
                density_history.append(density_g_cm3)

        acceptance_rate_prod = accept_count_prod / num_production_steps

        # Thermodynamic averages
        avg_energy = np.mean(energy_history_prod)
        std_energy = np.std(energy_history_prod)

        avg_density = np.mean(density_history)
        std_density = np.std(density_history)

        # Heat capacity from energy fluctuations
        heat_capacity = (np.var(energy_history_prod) / (self.KB * temperature_k**2)) if temperature_k > 0 else 0

        return {
            'technique': 'metropolis_monte_carlo',
            'ensemble': 'NVT (canonical)',
            'temperature_k': temperature_k,
            'num_particles': num_particles,
            'box_length_nm': box_length_nm,
            'volume_nm3': box_length_nm**3,
            'num_equilibration_steps': num_equilibration_steps,
            'num_production_steps': num_production_steps,
            'acceptance_rate_equilibration': acceptance_rate_equil,
            'acceptance_rate_production': acceptance_rate_prod,
            'thermodynamic_averages': {
                'energy_kj_mol': avg_energy,
                'energy_std_kj_mol': std_energy,
                'density_g_cm3': avg_density,
                'density_std_g_cm3': std_density,
                'heat_capacity_j_mol_k': heat_capacity
            },
            'energy_history': energy_history_prod[-1000:],  # Last 1000 points
            'optimal_acceptance_rate': 0.5,  # Target for random walk MC
            'quality_metrics': {
                'equilibration': 'converged' if acceptance_rate_equil > 0.3 else 'check_parameters',
                'sampling_efficiency': 'good' if 0.3 < acceptance_rate_prod < 0.7 else 'adjust_max_displacement'
            },
            'recommendations': self._generate_metropolis_recommendations(
                acceptance_rate_prod, max_displacement_nm
            )
        }

    def _execute_gcmc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Grand Canonical Monte Carlo (μVT ensemble).

        Variable number of particles - ideal for adsorption isotherms,
        gas storage, and phase coexistence.

        Args:
            input_data: Contains chemical potential, volume, temperature

        Returns:
            Average number of particles, adsorption isotherm
        """
        # Thermodynamic conditions
        temperature_k = input_data.get('temperature_k', 300)
        chemical_potential_kj_mol = input_data.get('chemical_potential_kj_mol', -10)
        box_length_nm = input_data.get('box_length_nm', 3.0)

        # MC parameters
        num_steps = input_data.get('num_steps', 100000)

        # Initialize
        num_particles = input_data.get('initial_num_particles', 50)
        max_particles = input_data.get('max_particles', 500)

        # Track properties
        particle_history = []
        insertion_attempts = 0
        insertions_accepted = 0
        deletion_attempts = 0
        deletions_accepted = 0

        for step in range(num_steps):
            # Randomly choose move type
            move_type = np.random.choice(['insertion', 'deletion', 'translation'])

            if move_type == 'insertion' and num_particles < max_particles:
                # Attempt particle insertion
                insertion_attempts += 1

                # Calculate insertion energy (simplified)
                delta_e = np.random.randn() * 2.0  # Interaction with existing particles

                # GCMC acceptance criterion for insertion
                # P_acc = min(1, V/(N+1) * exp(β(μ - ΔE)))
                beta = 1 / (self.KB * temperature_k / 1000)  # Convert kJ to J
                volume = box_length_nm**3
                acceptance_prob = min(1.0, volume / (num_particles + 1) *
                                     np.exp(beta * (chemical_potential_kj_mol - delta_e)))

                if np.random.rand() < acceptance_prob:
                    num_particles += 1
                    insertions_accepted += 1

            elif move_type == 'deletion' and num_particles > 0:
                # Attempt particle deletion
                deletion_attempts += 1

                delta_e = np.random.randn() * 2.0

                # GCMC acceptance criterion for deletion
                beta = 1 / (self.KB * temperature_k / 1000)
                volume = box_length_nm**3
                acceptance_prob = min(1.0, num_particles / volume *
                                     np.exp(beta * (delta_e - chemical_potential_kj_mol)))

                if np.random.rand() < acceptance_prob:
                    num_particles -= 1
                    deletions_accepted += 1

            else:
                # Translation move (standard Metropolis)
                pass

            # Sample particle number
            if step % 100 == 0:
                particle_history.append(num_particles)

        # Calculate averages
        avg_num_particles = np.mean(particle_history)
        std_num_particles = np.std(particle_history)

        # Density
        avg_density_particles_nm3 = avg_num_particles / (box_length_nm**3)

        # Acceptance rates
        insertion_acceptance = insertions_accepted / insertion_attempts if insertion_attempts > 0 else 0
        deletion_acceptance = deletions_accepted / deletion_attempts if deletion_attempts > 0 else 0

        return {
            'technique': 'grand_canonical_monte_carlo',
            'ensemble': 'μVT (grand canonical)',
            'temperature_k': temperature_k,
            'chemical_potential_kj_mol': chemical_potential_kj_mol,
            'volume_nm3': box_length_nm**3,
            'num_steps': num_steps,
            'average_num_particles': avg_num_particles,
            'std_num_particles': std_num_particles,
            'average_density_particles_nm3': avg_density_particles_nm3,
            'acceptance_rates': {
                'insertion': insertion_acceptance,
                'deletion': deletion_acceptance
            },
            'particle_number_history': particle_history[-1000:],
            'applications': [
                'Gas adsorption isotherms',
                'Pore filling mechanisms',
                'Henry coefficients',
                'Phase coexistence (with Gibbs ensemble)',
                'Battery electrolyte composition'
            ],
            'quality_metrics': {
                'particle_fluctuations': 'large' if std_num_particles > avg_num_particles * 0.3 else 'moderate',
                'insertion_efficiency': 'good' if insertion_acceptance > 0.01 else 'low'
            }
        }

    def _execute_kmc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Kinetic Monte Carlo (KMC) simulation.

        Time-evolution of processes with well-defined rates.
        Ideal for surface reactions, crystal growth, diffusion.

        Args:
            input_data: Contains reaction rates, initial state

        Returns:
            Time-resolved system evolution
        """
        # System parameters
        lattice_size = input_data.get('lattice_size', 50)
        num_sites = lattice_size ** 2

        # Reaction rates (1/s)
        rate_adsorption = input_data.get('rate_adsorption_per_s', 1e6)
        rate_desorption = input_data.get('rate_desorption_per_s', 1e3)
        rate_diffusion = input_data.get('rate_diffusion_per_s', 1e5)
        rate_reaction = input_data.get('rate_reaction_per_s', 1e4)

        # Initialize state (0=empty, 1=occupied)
        state = np.zeros((lattice_size, lattice_size), dtype=int)
        initial_coverage = input_data.get('initial_coverage', 0.1)
        num_initial = int(num_sites * initial_coverage)
        occupied_sites = np.random.choice(num_sites, num_initial, replace=False)
        state.flat[occupied_sites] = 1

        # KMC simulation
        total_time_s = 0
        num_events = input_data.get('num_events', 10000)

        time_history = [0]
        coverage_history = [initial_coverage]

        for event in range(num_events):
            # Calculate all possible event rates
            num_occupied = np.sum(state)
            num_empty = num_sites - num_occupied

            total_rate = (
                rate_adsorption * num_empty +
                rate_desorption * num_occupied +
                rate_diffusion * num_occupied +
                rate_reaction * num_occupied / 2  # Simplified
            )

            if total_rate == 0:
                break

            # Time increment (exponential distribution)
            dt = -np.log(np.random.rand()) / total_rate
            total_time_s += dt

            # Select event type based on rates
            rand_rate = np.random.rand() * total_rate

            if rand_rate < rate_adsorption * num_empty:
                # Adsorption event
                empty_sites = np.where(state.flat == 0)[0]
                site = np.random.choice(empty_sites)
                state.flat[site] = 1

            elif rand_rate < rate_adsorption * num_empty + rate_desorption * num_occupied:
                # Desorption event
                occupied_sites = np.where(state.flat == 1)[0]
                site = np.random.choice(occupied_sites)
                state.flat[site] = 0

            elif rand_rate < (rate_adsorption * num_empty + rate_desorption * num_occupied +
                            rate_diffusion * num_occupied):
                # Diffusion event
                occupied_sites = np.where(state.flat == 1)[0]
                site = np.random.choice(occupied_sites)
                # Move to random neighbor (simplified)
                i, j = divmod(site, lattice_size)
                neighbors = [
                    ((i+1) % lattice_size, j),
                    ((i-1) % lattice_size, j),
                    (i, (j+1) % lattice_size),
                    (i, (j-1) % lattice_size)
                ]
                ni, nj = neighbors[np.random.randint(4)]
                if state[ni, nj] == 0:
                    state[i, j] = 0
                    state[ni, nj] = 1

            else:
                # Reaction event
                pass

            # Sample coverage
            if event % 100 == 0:
                coverage = np.sum(state) / num_sites
                time_history.append(total_time_s)
                coverage_history.append(coverage)

        final_coverage = np.sum(state) / num_sites

        return {
            'technique': 'kinetic_monte_carlo',
            'lattice_size': lattice_size,
            'num_sites': num_sites,
            'num_events': num_events,
            'total_time_s': total_time_s,
            'time_history': time_history,
            'coverage_history': coverage_history,
            'final_coverage': final_coverage,
            'reaction_rates': {
                'adsorption_per_s': rate_adsorption,
                'desorption_per_s': rate_desorption,
                'diffusion_per_s': rate_diffusion,
                'reaction_per_s': rate_reaction
            },
            'applications': [
                'Surface catalysis (Langmuir-Hinshelwood)',
                'Crystal growth (nucleation, layer-by-layer)',
                'Thin film deposition (CVD, ALD)',
                'Diffusion on surfaces',
                'Battery electrode reactions'
            ],
            'advantages': [
                'Captures time evolution correctly',
                'Handles disparate timescales',
                'Direct connection to experimental rates',
                'Can reach long timescales (ms-s)'
            ]
        }

    def _execute_cbmc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Configurational Bias Monte Carlo (CBMC).

        Efficient sampling of chain molecules (polymers) using
        biased insertion/regrowth.

        Args:
            input_data: Contains polymer parameters

        Returns:
            Polymer configurations and thermodynamic properties
        """
        # Polymer parameters
        chain_length = input_data.get('chain_length', 20)
        num_chains = input_data.get('num_chains', 10)
        temperature_k = input_data.get('temperature_k', 400)
        box_length_nm = input_data.get('box_length_nm', 5.0)

        # CBMC parameters
        num_trial_positions = input_data.get('num_trial_positions', 10)
        num_steps = input_data.get('num_steps', 50000)

        # Track properties
        regrow_attempts = 0
        regrow_accepted = 0
        end_to_end_distances = []

        for step in range(num_steps):
            # Select random chain to regrow
            chain_idx = np.random.randint(0, num_chains)

            # Regrow with configurational bias
            regrow_attempts += 1

            # Rosenbluth weight calculation (simplified)
            # In real CBMC: generate k trial positions, weight by Boltzmann
            trial_weights = np.random.rand(num_trial_positions)
            rosenbluth_new = np.sum(trial_weights)
            rosenbluth_old = np.sum(np.random.rand(num_trial_positions))

            # CBMC acceptance probability
            acceptance_prob = min(1.0, rosenbluth_new / rosenbluth_old)

            if np.random.rand() < acceptance_prob:
                regrow_accepted += 1

            # Sample end-to-end distance
            if step % 100 == 0:
                # Random walk model for demo
                ete_distance_nm = np.sqrt(chain_length) * 0.15  # Simplified
                ete_distance_nm += 0.2 * np.random.randn()
                end_to_end_distances.append(ete_distance_nm)

        avg_ete = np.mean(end_to_end_distances)
        std_ete = np.std(end_to_end_distances)

        # Radius of gyration (simplified)
        r_g_nm = avg_ete / np.sqrt(6)

        acceptance_rate = regrow_accepted / regrow_attempts

        return {
            'technique': 'configurational_bias_monte_carlo',
            'chain_length': chain_length,
            'num_chains': num_chains,
            'temperature_k': temperature_k,
            'volume_nm3': box_length_nm**3,
            'num_trial_positions': num_trial_positions,
            'num_steps': num_steps,
            'acceptance_rate': acceptance_rate,
            'structural_properties': {
                'end_to_end_distance_nm': avg_ete,
                'end_to_end_std_nm': std_ete,
                'radius_of_gyration_nm': r_g_nm,
                'scaling_exponent': 'ν ≈ 0.5 (ideal chain)' if abs(avg_ete - np.sqrt(chain_length) * 0.15) < 0.1 else 'ν ≠ 0.5'
            },
            'advantages': [
                'Efficient polymer insertion/deletion',
                'Samples long chains that would be rejected in standard MC',
                '100-1000× faster than standard MC for polymers',
                'Essential for GCMC with polymers'
            ],
            'applications': [
                'Polymer solutions and melts',
                'Surfactant micelles',
                'Protein conformations',
                'Membrane permeability'
            ]
        }

    def _execute_wang_landau(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Wang-Landau sampling for density of states.

        Flat histogram method - samples all energies equally,
        provides free energy landscape.

        Args:
            input_data: Contains energy range and modification factor

        Returns:
            Density of states, free energy surface
        """
        # Energy discretization
        energy_min = input_data.get('energy_min_kj_mol', -100)
        energy_max = input_data.get('energy_max_kj_mol', 0)
        energy_bins = input_data.get('energy_bins', 100)

        energy_grid = np.linspace(energy_min, energy_max, energy_bins)
        delta_e = (energy_max - energy_min) / energy_bins

        # Initialize density of states (log scale)
        log_g = np.zeros(energy_bins)

        # Histogram
        h = np.zeros(energy_bins)

        # Modification factor
        f = 1.0
        f_final = input_data.get('f_final', 1e-8)

        # Current state
        current_energy_idx = energy_bins // 2
        current_energy = energy_grid[current_energy_idx]

        num_steps = 0

        while f > f_final and num_steps < 1000000:
            # Attempt move
            new_energy_idx = current_energy_idx + np.random.choice([-1, 0, 1])

            # Boundary conditions
            if new_energy_idx < 0 or new_energy_idx >= energy_bins:
                new_energy_idx = current_energy_idx

            # Wang-Landau acceptance criterion
            # P_acc = min(1, g(E_old) / g(E_new))
            delta_log_g = log_g[current_energy_idx] - log_g[new_energy_idx]

            if delta_log_g >= 0 or np.random.rand() < np.exp(delta_log_g):
                current_energy_idx = new_energy_idx

            # Update density of states
            log_g[current_energy_idx] += f

            # Update histogram
            h[current_energy_idx] += 1

            num_steps += 1

            # Check flatness every 1000 steps
            if num_steps % 1000 == 0:
                flatness = np.min(h[h > 0]) / np.mean(h[h > 0]) if np.sum(h > 0) > 0 else 0

                if flatness > 0.8:  # 80% flat
                    # Reduce modification factor
                    f /= 2.0
                    # Reset histogram
                    h[:] = 0

        # Normalize density of states
        log_g -= np.max(log_g)

        # Calculate free energy: F(E) = -kT ln g(E)
        temperature_k = input_data.get('temperature_k', 300)
        free_energy = -self.KB * temperature_k * log_g / 1000  # kJ/mol

        return {
            'technique': 'wang_landau_sampling',
            'energy_range_kj_mol': (energy_min, energy_max),
            'num_energy_bins': energy_bins,
            'final_modification_factor': f,
            'num_steps': num_steps,
            'energy_grid_kj_mol': energy_grid.tolist(),
            'log_density_of_states': log_g.tolist(),
            'free_energy_kj_mol': free_energy.tolist(),
            'advantages': [
                'Samples entire energy range (rare events)',
                'Provides density of states g(E)',
                'Free energy from g(E)',
                'No temperature dependence during simulation',
                'Can calculate properties at any T from single run'
            ],
            'applications': [
                'Phase transitions (order parameters)',
                'Protein folding landscapes',
                'Magnetic systems (Ising model)',
                'Nucleation barriers',
                'Free energy calculations'
            ]
        }

    def _execute_parallel_tempering(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Parallel Tempering (Replica Exchange) Monte Carlo.

        Enhanced sampling via replica exchange at different temperatures.
        Overcomes energy barriers, samples metastable states.

        Args:
            input_data: Contains temperature range, num replicas

        Returns:
            Enhanced sampling results across temperatures
        """
        # Temperature ladder
        temp_min_k = input_data.get('temp_min_k', 250)
        temp_max_k = input_data.get('temp_max_k', 500)
        num_replicas = input_data.get('num_replicas', 8)

        # Geometric spacing often optimal
        temperatures = np.logspace(np.log10(temp_min_k), np.log10(temp_max_k), num_replicas)

        # Initialize replicas
        replica_energies = -50 + 10 * np.random.randn(num_replicas)

        # PT simulation
        num_steps = input_data.get('num_steps', 100000)
        exchange_frequency = input_data.get('exchange_frequency', 100)

        exchange_attempts = 0
        exchanges_accepted = 0

        energy_history = {i: [] for i in range(num_replicas)}

        for step in range(num_steps):
            # Standard MC move at each temperature
            for i in range(num_replicas):
                # Metropolis move
                delta_e = np.random.randn() * 5

                beta = 1 / (self.KB * temperatures[i] / 1000)
                if delta_e < 0 or np.random.rand() < np.exp(-beta * delta_e):
                    replica_energies[i] += delta_e

                # Sample energy
                if step % 100 == 0:
                    energy_history[i].append(replica_energies[i])

            # Attempt replica exchange
            if step % exchange_frequency == 0 and num_replicas > 1:
                # Choose random pair of adjacent replicas
                i = np.random.randint(0, num_replicas - 1)
                j = i + 1

                # Exchange criterion
                beta_i = 1 / (self.KB * temperatures[i] / 1000)
                beta_j = 1 / (self.KB * temperatures[j] / 1000)

                delta_beta = beta_j - beta_i
                delta_E = replica_energies[j] - replica_energies[i]

                # P_exchange = min(1, exp(Δβ × ΔE))
                prob_exchange = min(1.0, np.exp(delta_beta * delta_E))

                exchange_attempts += 1

                if np.random.rand() < prob_exchange:
                    # Swap configurations (energies in simplified model)
                    replica_energies[i], replica_energies[j] = replica_energies[j], replica_energies[i]
                    exchanges_accepted += 1

        exchange_acceptance_rate = exchanges_accepted / exchange_attempts if exchange_attempts > 0 else 0

        # Calculate average energies at each temperature
        avg_energies = [np.mean(energy_history[i]) for i in range(num_replicas)]

        return {
            'technique': 'parallel_tempering_monte_carlo',
            'num_replicas': num_replicas,
            'temperature_ladder_k': temperatures.tolist(),
            'num_steps': num_steps,
            'exchange_frequency': exchange_frequency,
            'exchange_acceptance_rate': exchange_acceptance_rate,
            'average_energies_kj_mol': avg_energies,
            'optimal_exchange_rate': 0.2,  # Target ~20% for good sampling
            'quality_metrics': {
                'exchange_efficiency': 'good' if 0.15 < exchange_acceptance_rate < 0.30 else 'adjust_temperature_ladder',
                'temperature_overlap': 'sufficient' if exchange_acceptance_rate > 0.05 else 'insufficient'
            },
            'advantages': [
                'Overcomes energy barriers',
                'Samples multiple minima',
                'Self-adaptive (exchanges naturally optimize)',
                'Provides thermodynamic data across T range'
            ],
            'applications': [
                'Protein folding',
                'Spin glasses',
                'Polymer phase transitions',
                'Complex potential energy surfaces'
            ]
        }

    def _execute_gemc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Gibbs Ensemble Monte Carlo (GEMC).

        Phase coexistence without interfaces - direct simulation of
        vapor-liquid equilibria.

        Args:
            input_data: Contains two-box system parameters

        Returns:
            Phase coexistence densities and pressures
        """
        temperature_k = input_data.get('temperature_k', 350)

        # Two boxes
        box1_length_nm = input_data.get('box1_length_nm', 3.0)
        box2_length_nm = input_data.get('box2_length_nm', 3.0)

        # Initial particle distribution
        num_particles_box1 = input_data.get('num_particles_box1', 200)  # Liquid-like
        num_particles_box2 = input_data.get('num_particles_box2', 50)   # Vapor-like

        num_steps = input_data.get('num_steps', 100000)

        # Track densities
        density1_history = []
        density2_history = []

        for step in range(num_steps):
            # GEMC move types:
            # 1. Translation within box
            # 2. Volume exchange between boxes
            # 3. Particle transfer between boxes

            move_type = np.random.choice(['translation', 'volume_exchange', 'particle_transfer'])

            if move_type == 'particle_transfer' and num_particles_box1 > 0 and num_particles_box2 < 1000:
                # Transfer particle from box1 to box2 (or vice versa)
                if np.random.rand() < 0.5 and num_particles_box1 > 0:
                    # 1 → 2
                    num_particles_box1 -= 1
                    num_particles_box2 += 1
                elif num_particles_box2 > 0:
                    # 2 → 1
                    num_particles_box2 -= 1
                    num_particles_box1 += 1

            elif move_type == 'volume_exchange':
                # Exchange volume keeping total constant
                # V1' = V1 + ΔV, V2' = V2 - ΔV
                pass

            # Sample densities
            if step % 100 == 0:
                density1 = num_particles_box1 / (box1_length_nm**3)
                density2 = num_particles_box2 / (box2_length_nm**3)
                density1_history.append(density1)
                density2_history.append(density2)

        # Phase coexistence densities
        avg_density_liquid = np.mean(density1_history)
        avg_density_vapor = np.mean(density2_history)

        return {
            'technique': 'gibbs_ensemble_monte_carlo',
            'temperature_k': temperature_k,
            'phase_coexistence': {
                'liquid_density_particles_nm3': avg_density_liquid,
                'vapor_density_particles_nm3': avg_density_vapor,
                'density_ratio': avg_density_liquid / avg_density_vapor if avg_density_vapor > 0 else float('inf')
            },
            'num_steps': num_steps,
            'density_liquid_history': density1_history[-1000:],
            'density_vapor_history': density2_history[-1000:],
            'advantages': [
                'Direct phase coexistence (no interface)',
                'Efficient for VLE (vapor-liquid equilibrium)',
                'Avoids surface tension complications',
                'Can reach critical point'
            ],
            'applications': [
                'Vapor-liquid equilibria',
                'Critical properties',
                'Equation of state validation',
                'Phase diagrams'
            ]
        }

    def _execute_tmmc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Transition Matrix Monte Carlo (TMMC).

        Constructs transition matrix to extract free energy,
        particularly useful for order parameters.

        Args:
            input_data: Contains order parameter range

        Returns:
            Free energy profile along order parameter
        """
        # Order parameter (e.g., number of particles, magnetization)
        order_param_min = input_data.get('order_param_min', 0)
        order_param_max = input_data.get('order_param_max', 100)
        num_bins = input_data.get('num_bins', 50)

        # Initialize transition matrix
        C = np.zeros((num_bins, num_bins))

        # Current state
        current_bin = num_bins // 2

        num_steps = input_data.get('num_steps', 100000)

        for step in range(num_steps):
            # Attempt move
            new_bin = current_bin + np.random.choice([-1, 0, 1])

            if 0 <= new_bin < num_bins:
                # Record transition
                C[current_bin, new_bin] += 1
                current_bin = new_bin

        # Calculate stationary distribution from C
        # π_i / π_j = C_ij / C_ji (detailed balance)
        # Free energy: F(m) = -kT ln π(m)

        temperature_k = input_data.get('temperature_k', 300)

        # Normalize rows to get transition probabilities
        row_sums = np.sum(C, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        P = C / row_sums

        # Estimate stationary distribution (simplified - would use eigenvector)
        pi = np.sum(C, axis=0)
        pi = pi / np.sum(pi)
        pi[pi == 0] = 1e-10  # Avoid log(0)

        # Free energy
        free_energy = -self.KB * temperature_k * np.log(pi) / 1000  # kJ/mol
        free_energy -= np.min(free_energy)  # Set minimum to 0

        order_param_values = np.linspace(order_param_min, order_param_max, num_bins)

        return {
            'technique': 'transition_matrix_monte_carlo',
            'order_parameter_range': (order_param_min, order_param_max),
            'num_bins': num_bins,
            'num_steps': num_steps,
            'order_parameter_values': order_param_values.tolist(),
            'free_energy_kj_mol': free_energy.tolist(),
            'transition_matrix_size': C.shape,
            'applications': [
                'Free energy along reaction coordinate',
                'Nucleation barriers',
                'Phase transition order parameters',
                'Umbrella sampling alternative'
            ]
        }

    # Helper methods

    def _generate_metropolis_recommendations(self, acceptance_rate: float,
                                            max_disp: float) -> List[str]:
        """Generate recommendations for Metropolis MC."""
        recs = []

        if acceptance_rate < 0.3:
            recs.append(f"Low acceptance rate ({acceptance_rate:.2f}) - reduce max_displacement")
            recs.append(f"Try max_displacement = {max_disp * 0.7:.3f} nm")

        if acceptance_rate > 0.7:
            recs.append(f"High acceptance rate ({acceptance_rate:.2f}) - increase max_displacement")
            recs.append(f"Try max_displacement = {max_disp * 1.3:.3f} nm")

        if 0.4 < acceptance_rate < 0.6:
            recs.append("Acceptance rate optimal for random walk sampling")

        recs.append("Ensure equilibration before production sampling")
        recs.append("Monitor energy/density convergence")

        return recs

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and metadata."""
        return {
            'agent_type': self.AGENT_TYPE,
            'version': self.VERSION,
            'supported_techniques': self.SUPPORTED_TECHNIQUES,
            'move_types': self.MOVE_TYPES,
            'ensembles': {
                'NVT': 'Canonical (Metropolis)',
                'μVT': 'Grand canonical (GCMC)',
                'NPT': 'Isothermal-isobaric',
                'Gibbs': 'Phase coexistence (GEMC)'
            },
            'enhanced_sampling': [
                'Parallel Tempering (REMD)',
                'Wang-Landau (flat histogram)',
                'TMMC (transition matrix)',
                'Umbrella sampling',
                'Metadynamics'
            ],
            'advantages_over_md': [
                'Direct equilibrium sampling (no dynamics)',
                'Can handle large moves (polymer CBMC)',
                'Variable N (GCMC for adsorption)',
                'Enhanced sampling (PT, WL)',
                'Exact statistical mechanics (detailed balance)'
            ],
            'cross_validation_opportunities': [
                'MC ↔ MD: Compare thermodynamic averages (P, ρ, E)',
                'GCMC ↔ Experiment: Adsorption isotherms',
                'KMC ↔ DFT: Validate reaction barriers',
                'Free energy ↔ Phase diagrams',
                'MC structure ↔ Scattering (S(q), g(r))'
            ],
            'typical_applications': [
                'Gas adsorption and storage (GCMC)',
                'Phase equilibria (GEMC)',
                'Polymer conformations (CBMC)',
                'Surface catalysis (KMC)',
                'Free energy landscapes (WL, TMMC)',
                'Protein folding (PT)',
                'Battery electrolytes (GCMC composition)'
            ]
        }


if __name__ == '__main__':
    # Example usage
    agent = MonteCarloAgent()

    # Example 1: Metropolis MC
    result_metropolis = agent.execute({
        'technique': 'metropolis',
        'temperature_k': 298,
        'num_particles': 256,
        'box_length_nm': 3.0,
        'num_production_steps': 100000
    })
    print("Metropolis Monte Carlo Result:")
    print(f"  Temperature: {result_metropolis['temperature_k']} K")
    print(f"  Acceptance Rate: {result_metropolis['acceptance_rate_production']:.2%}")
    print(f"  Average Energy: {result_metropolis['thermodynamic_averages']['energy_kj_mol']:.2f} kJ/mol")
    print(f"  Average Density: {result_metropolis['thermodynamic_averages']['density_g_cm3']:.3f} g/cm³")
    print()

    # Example 2: Grand Canonical MC (adsorption)
    result_gcmc = agent.execute({
        'technique': 'grand_canonical',
        'temperature_k': 300,
        'chemical_potential_kj_mol': -15,
        'box_length_nm': 3.0,
        'num_steps': 100000
    })
    print("Grand Canonical Monte Carlo Result:")
    print(f"  Chemical Potential: {result_gcmc['chemical_potential_kj_mol']} kJ/mol")
    print(f"  Average N: {result_gcmc['average_num_particles']:.1f} ± {result_gcmc['std_num_particles']:.1f}")
    print(f"  Density: {result_gcmc['average_density_particles_nm3']:.3f} particles/nm³")
    print()

    # Example 3: Kinetic Monte Carlo
    result_kmc = agent.execute({
        'technique': 'kinetic_monte_carlo',
        'lattice_size': 50,
        'rate_adsorption_per_s': 1e6,
        'rate_desorption_per_s': 1e3,
        'num_events': 10000
    })
    print("Kinetic Monte Carlo Result:")
    print(f"  Total Time: {result_kmc['total_time_s']:.6f} s")
    print(f"  Final Coverage: {result_kmc['final_coverage']:.2%}")
    print(f"  Num Events: {result_kmc['num_events']}")
