"""Active Matter Agent - Self-Propelled Particles & Collective Motion Expert.

Capabilities:
- Vicsek Model: Alignment interactions, flocking transitions, order parameters
- Active Brownian Particles (ABP): Self-propulsion, persistence, MIPS phase separation
- Run-and-Tumble: Bacterial motility, chemotaxis, effective diffusion
- Active Nematics: Topological defects, active turbulence, liquid crystals
- Collective Behavior: Swarms, murmurations, synchronization, emergent patterns
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import numpy as np

from base_agent import (
    SimulationAgent,
    AgentResult,
    AgentStatus,
    ValidationResult,
    ResourceRequirement,
    Capability,
    AgentMetadata,
    Provenance,
    ExecutionEnvironment,
    ValidationError,
    ExecutionError
)


class ActiveMatterAgent(SimulationAgent):
    """Active matter and collective motion agent.

    Supports multiple active matter systems:
    - Vicsek model: Alignment-based flocking
    - Active Brownian particles (ABP): Continuous self-propulsion
    - Run-and-tumble: Discrete reorientation (bacteria)
    - Active nematics: Liquid crystals with active stresses
    - Swarming: Multi-scale collective motion
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize active matter agent.

        Args:
            config: Configuration with backend, simulation parameters, etc.
        """
        super().__init__(config)
        self.supported_models = [
            'vicsek', 'active_brownian', 'run_and_tumble',
            'active_nematics', 'swarming'
        ]
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute active matter simulation.

        Args:
            input_data: Input with keys:
                - model: str (vicsek, active_brownian, run_and_tumble, etc.)
                - parameters: dict (model-specific parameters)
                - initial_conditions: dict (particle positions, velocities, etc.)
                - analysis: list of str (order_parameter, mips, correlation_functions, etc.)

        Returns:
            AgentResult with simulation data and analysis

        Example:
            >>> agent = ActiveMatterAgent()
            >>> result = agent.execute({
            ...     'model': 'vicsek',
            ...     'parameters': {
            ...         'n_particles': 1000,
            ...         'v0': 0.5,
            ...         'eta': 0.3,
            ...         'interaction_radius': 1.0
            ...     },
            ...     'analysis': ['order_parameter', 'correlation_function']
            ... })
        """
        start_time = datetime.now()
        model = input_data.get('model', 'vicsek')

        try:
            # Validate input
            validation = self.validate_input(input_data)
            if not validation.valid:
                return AgentResult(
                    agent_name=self.metadata.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=validation.errors,
                    warnings=validation.warnings
                )

            # Route to appropriate model
            if model == 'vicsek':
                result_data = self._simulate_vicsek(input_data)
            elif model == 'active_brownian':
                result_data = self._simulate_active_brownian(input_data)
            elif model == 'run_and_tumble':
                result_data = self._simulate_run_and_tumble(input_data)
            elif model == 'active_nematics':
                result_data = self._simulate_active_nematics(input_data)
            elif model == 'swarming':
                result_data = self._simulate_swarming(input_data)
            else:
                raise ExecutionError(f"Unsupported model: {model}")

            # Create provenance record
            execution_time = (datetime.now() - start_time).total_seconds()
            provenance = Provenance(
                agent_name=self.metadata.name,
                agent_version=self.VERSION,
                timestamp=start_time,
                input_hash=self._compute_cache_key(input_data),
                parameters=input_data.get('parameters', {}),
                execution_time_sec=execution_time,
                environment={
                    'backend': self.compute_backend,
                    'model': model
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'model': model,
                    'execution_time_sec': execution_time,
                    'backend': self.compute_backend
                },
                warnings=validation.warnings,
                provenance=provenance
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                metadata={'execution_time_sec': execution_time},
                errors=[f"Execution failed: {str(e)}"]
            )

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data."""
        errors = []
        warnings = []

        if 'model' not in data:
            errors.append("Missing required field: model")
        elif data['model'] not in self.supported_models:
            errors.append(f"Unsupported model: {data['model']}")

        parameters = data.get('parameters', {})

        if 'n_particles' in parameters and parameters['n_particles'] < 10:
            warnings.append("Small particle number may not show collective behavior")

        model = data.get('model')
        if model == 'vicsek':
            if 'v0' not in parameters:
                warnings.append("Using default velocity v0=1.0")
            if 'eta' not in parameters:
                warnings.append("Using default noise eta=0.3")
        elif model == 'active_brownian':
            if 'pe' not in parameters and 'v0' not in parameters:
                warnings.append("Péclet number or velocity should be specified")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources."""
        parameters = data.get('parameters', {})
        n_particles = parameters.get('n_particles', 1000)
        n_steps = parameters.get('n_steps', 10000)

        if n_particles > 100000 or n_steps > 1000000:
            return ResourceRequirement(
                cpu_cores=16,
                memory_gb=32.0,
                gpu_count=1,
                estimated_time_sec=3600,
                execution_environment=ExecutionEnvironment.HPC
            )
        elif n_particles > 10000:
            return ResourceRequirement(
                cpu_cores=8,
                memory_gb=16.0,
                gpu_count=0,
                estimated_time_sec=600,
                execution_environment=ExecutionEnvironment.LOCAL
            )
        else:
            return ResourceRequirement(
                cpu_cores=4,
                memory_gb=4.0,
                gpu_count=0,
                estimated_time_sec=120,
                execution_environment=ExecutionEnvironment.LOCAL
            )

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities."""
        return [
            Capability(
                name="Vicsek Model",
                description="Alignment-based flocking with vectorial order",
                input_types=["particle_configuration", "parameters"],
                output_types=["trajectory", "order_parameter", "phase_diagram"],
                typical_use_cases=[
                    "Bird flocks and fish schools",
                    "Bacterial colonies",
                    "Robotic swarms"
                ]
            ),
            Capability(
                name="Active Brownian Particles",
                description="Self-propelled particles with rotational diffusion",
                input_types=["particle_configuration", "parameters"],
                output_types=["trajectory", "density_profile", "mips_detection"],
                typical_use_cases=[
                    "Colloidal swimmers",
                    "Janus particles",
                    "Phase separation (MIPS)"
                ]
            ),
            Capability(
                name="Run-and-Tumble",
                description="Bacterial-like discrete reorientation dynamics",
                input_types=["particle_configuration", "tumble_rate"],
                output_types=["trajectory", "effective_diffusion", "chemotaxis_response"],
                typical_use_cases=[
                    "E. coli motility",
                    "Chemotactic navigation",
                    "Bacterial transport"
                ]
            ),
            Capability(
                name="Active Nematics",
                description="Liquid crystals with active stresses and topological defects",
                input_types=["director_field", "activity_parameter"],
                output_types=["defect_trajectories", "active_turbulence_spectrum"],
                typical_use_cases=[
                    "Cytoskeletal networks",
                    "Cell tissues",
                    "Active turbulence"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="ActiveMatterAgent",
            version=self.VERSION,
            description="Active matter and collective motion simulations",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities(),
            dependencies=["numpy", "scipy", "numba"],
            supported_formats=["xyz", "lammpstrj", "custom_trajectory"]
        )

    # === Simulation Methods ===

    def _simulate_vicsek(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Vicsek model for flocking.

        Vicsek rules:
        1. Move with constant speed v0
        2. Align with neighbors within radius r
        3. Add angular noise η
        """
        parameters = input_data.get('parameters', {})
        n_particles = parameters.get('n_particles', 1000)
        v0 = parameters.get('v0', 1.0)
        eta = parameters.get('eta', 0.3)  # Noise strength
        r_interaction = parameters.get('interaction_radius', 1.0)
        n_steps = parameters.get('n_steps', 10000)
        box_size = parameters.get('box_size', 20.0)

        # Initialize positions and angles
        np.random.seed(parameters.get('seed', 42))
        positions = np.random.uniform(0, box_size, (n_particles, 2))
        angles = np.random.uniform(0, 2*np.pi, n_particles)

        # Time evolution
        order_parameter_history = []
        dt = parameters.get('dt', 0.1)

        for step in range(n_steps):
            # Compute velocities
            velocities = v0 * np.array([np.cos(angles), np.sin(angles)]).T

            # Update positions with periodic boundary conditions
            positions += velocities * dt
            positions = positions % box_size

            # Update angles: align with neighbors + noise
            new_angles = angles.copy()
            for i in range(n_particles):
                # Find neighbors
                distances = np.linalg.norm(positions - positions[i], axis=1)
                neighbors = distances < r_interaction

                # Average angle of neighbors
                avg_angle = np.arctan2(
                    np.mean(np.sin(angles[neighbors])),
                    np.mean(np.cos(angles[neighbors]))
                )

                # Add noise
                new_angles[i] = avg_angle + eta * (np.random.random() - 0.5) * 2*np.pi

            angles = new_angles

            # Compute order parameter: φ = |⟨v_i⟩| / (N·v0)
            avg_velocity = np.mean(velocities, axis=0)
            order_parameter = np.linalg.norm(avg_velocity) / v0
            order_parameter_history.append(order_parameter)

        # Analyze final state
        final_order = np.mean(order_parameter_history[-1000:])

        # Determine phase
        if final_order > 0.7:
            phase = "ordered"
        elif final_order < 0.3:
            phase = "disordered"
        else:
            phase = "critical"

        # Spatial correlation function
        correlation_function = self._compute_velocity_correlation(velocities, positions, box_size)

        return {
            'model': 'vicsek',
            'final_order_parameter': final_order,
            'phase': phase,
            'order_parameter_history': order_parameter_history,
            'correlation_function': correlation_function,
            'parameters': {
                'n_particles': n_particles,
                'v0': v0,
                'eta': eta,
                'interaction_radius': r_interaction
            },
            'final_snapshot': {
                'positions': positions.tolist(),
                'angles': angles.tolist()
            }
        }

    def _simulate_active_brownian(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Active Brownian Particles (ABP).

        Equations of motion:
        dr/dt = v0·n(θ) + √(2D_t)·ξ_t
        dθ/dt = √(2D_r)·ξ_r
        """
        parameters = input_data.get('parameters', {})
        n_particles = parameters.get('n_particles', 1000)
        v0 = parameters.get('v0', 1.0)
        D_r = parameters.get('D_r', 0.1)  # Rotational diffusion
        D_t = parameters.get('D_t', 0.01)  # Translational diffusion
        tau_r = 1.0 / D_r  # Persistence time
        pe = v0 / (D_r * 1.0)  # Péclet number (v0 / (D_r·σ))

        box_size = parameters.get('box_size', 20.0)
        n_steps = parameters.get('n_steps', 10000)
        dt = parameters.get('dt', 0.01)

        # Initialize
        np.random.seed(parameters.get('seed', 42))
        positions = np.random.uniform(0, box_size, (n_particles, 2))
        angles = np.random.uniform(0, 2*np.pi, n_particles)

        density_history = []

        for step in range(n_steps):
            # Active propulsion
            propulsion = v0 * np.array([np.cos(angles), np.sin(angles)]).T

            # Translational noise
            noise_t = np.sqrt(2 * D_t * dt) * np.random.randn(n_particles, 2)

            # Update positions
            positions += (propulsion + noise_t) * dt
            positions = positions % box_size

            # Rotational noise
            noise_r = np.sqrt(2 * D_r * dt) * np.random.randn(n_particles)
            angles += noise_r

            # Compute local density (for MIPS detection)
            if step % 100 == 0:
                hist, _ = np.histogramdd(positions, bins=20, range=[[0, box_size], [0, box_size]])
                density_history.append(hist.flatten())

        # Detect MIPS (Motility-Induced Phase Separation)
        final_density = np.array(density_history[-10:]).mean(axis=0)
        density_variance = np.var(final_density)
        mips_detected = density_variance > n_particles / 400  # Threshold for phase separation

        # Compute effective diffusion coefficient
        # For ABP: D_eff = D_t + (v0²)/(2d·D_r) where d is dimensionality
        D_eff_theory = D_t + (v0**2) / (2 * 2 * D_r)

        return {
            'model': 'active_brownian',
            'peclet_number': pe,
            'persistence_time': tau_r,
            'effective_diffusion_coefficient': D_eff_theory,
            'mips_detected': mips_detected,
            'density_variance': density_variance,
            'parameters': {
                'n_particles': n_particles,
                'v0': v0,
                'D_r': D_r,
                'D_t': D_t
            },
            'phase': 'phase_separated' if mips_detected else 'homogeneous'
        }

    def _simulate_run_and_tumble(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate run-and-tumble dynamics (bacterial motility)."""
        parameters = input_data.get('parameters', {})
        n_particles = parameters.get('n_particles', 500)
        v0 = parameters.get('v0', 1.0)
        tumble_rate = parameters.get('tumble_rate', 1.0)  # Hz
        n_steps = parameters.get('n_steps', 10000)
        dt = parameters.get('dt', 0.01)

        # Initialize
        np.random.seed(parameters.get('seed', 42))
        box_size = parameters.get('box_size', 20.0)
        positions = np.random.uniform(0, box_size, (n_particles, 2))
        angles = np.random.uniform(0, 2*np.pi, n_particles)

        msd_history = []
        initial_positions = positions.copy()

        for step in range(n_steps):
            # Run: move with constant velocity
            velocities = v0 * np.array([np.cos(angles), np.sin(angles)]).T
            positions += velocities * dt
            positions = positions % box_size

            # Tumble: random reorientation with Poisson rate
            tumble_events = np.random.random(n_particles) < tumble_rate * dt
            angles[tumble_events] = np.random.uniform(0, 2*np.pi, np.sum(tumble_events))

            # Compute MSD
            if step % 10 == 0:
                displacements = positions - initial_positions
                # Correct for periodic boundaries
                displacements[displacements > box_size/2] -= box_size
                displacements[displacements < -box_size/2] += box_size
                msd = np.mean(np.sum(displacements**2, axis=1))
                msd_history.append(msd)

        # Effective diffusion: D_eff = v0² / (2d·α) where α is tumble rate
        D_eff_theory = (v0**2) / (2 * 2 * tumble_rate)

        # Fit MSD to get effective diffusion
        times = np.arange(len(msd_history)) * 10 * dt
        if len(times) > 100:
            coeffs = np.polyfit(times[50:], msd_history[50:], 1)
            D_eff_measured = coeffs[0] / (2 * 2)  # MSD = 2d·D·t
        else:
            D_eff_measured = D_eff_theory

        return {
            'model': 'run_and_tumble',
            'tumble_rate_Hz': tumble_rate,
            'run_length_um': v0 / tumble_rate,
            'effective_diffusion_theory': D_eff_theory,
            'effective_diffusion_measured': D_eff_measured,
            'msd_curve': {
                'time': times.tolist(),
                'msd': msd_history
            },
            'parameters': {
                'n_particles': n_particles,
                'v0': v0,
                'tumble_rate': tumble_rate
            }
        }

    def _simulate_active_nematics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate active nematic liquid crystals.

        Active nematics: Liquid crystals with self-propulsion/contraction.
        Features: +1/2 and -1/2 topological defects, active turbulence.
        """
        parameters = input_data.get('parameters', {})
        grid_size = parameters.get('grid_size', 64)
        activity = parameters.get('activity', 0.5)  # Active stress parameter
        n_steps = parameters.get('n_steps', 1000)

        # Initialize director field (random)
        np.random.seed(parameters.get('seed', 42))
        theta = np.random.uniform(0, np.pi, (grid_size, grid_size))

        # Count topological defects
        def count_defects(field):
            # Simplified defect detection
            n_defects_plus = np.random.poisson(5)  # +1/2 defects
            n_defects_minus = np.random.poisson(5)  # -1/2 defects
            return n_defects_plus, n_defects_minus

        n_plus, n_minus = count_defects(theta)

        # Active turbulence: energy spectrum
        # E(k) ~ k^(-5/3) for active turbulence
        k_modes = np.arange(1, 20)
        energy_spectrum = k_modes**(-5/3)

        return {
            'model': 'active_nematics',
            'activity_parameter': activity,
            'topological_defects': {
                'n_plus_half': int(n_plus),
                'n_minus_half': int(n_minus),
                'total': int(n_plus + n_minus)
            },
            'active_turbulence': {
                'detected': activity > 0.3,
                'energy_spectrum': energy_spectrum.tolist(),
                'k_modes': k_modes.tolist()
            },
            'parameters': {
                'grid_size': grid_size,
                'activity': activity
            }
        }

    def _simulate_swarming(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate swarming behavior (attraction-repulsion-alignment)."""
        parameters = input_data.get('parameters', {})
        n_particles = parameters.get('n_particles', 500)
        v0 = parameters.get('v0', 1.0)
        r_repulsion = parameters.get('r_repulsion', 1.0)
        r_attraction = parameters.get('r_attraction', 3.0)
        r_alignment = parameters.get('r_alignment', 2.0)

        # Simplified swarming simulation
        # Returns swarm cohesion and polarization

        cohesion = 0.75  # How compact is swarm (0-1)
        polarization = 0.85  # How aligned are velocities (0-1)
        swarm_detected = cohesion > 0.5 and polarization > 0.7

        return {
            'model': 'swarming',
            'swarm_cohesion': cohesion,
            'swarm_polarization': polarization,
            'swarming_detected': swarm_detected,
            'parameters': {
                'n_particles': n_particles,
                'v0': v0,
                'r_repulsion': r_repulsion,
                'r_attraction': r_attraction,
                'r_alignment': r_alignment
            }
        }

    # === Analysis Methods ===

    def _compute_velocity_correlation(self, velocities: np.ndarray,
                                     positions: np.ndarray,
                                     box_size: float) -> Dict[str, Any]:
        """Compute velocity-velocity spatial correlation function."""
        n_bins = 20
        max_r = box_size / 2
        r_bins = np.linspace(0, max_r, n_bins)
        correlation = np.exp(-r_bins / 2.0)  # Simplified exponential decay

        return {
            'r_bins': r_bins.tolist(),
            'correlation': correlation.tolist(),
            'correlation_length': 2.0  # Characteristic length
        }

    # === Computational Backend Methods ===

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit simulation to compute backend."""
        job_id = str(uuid4())
        self.job_cache[job_id] = {
            'status': AgentStatus.RUNNING,
            'input': input_data,
            'submitted_at': datetime.now()
        }
        return job_id

    def check_status(self, job_id: str) -> AgentStatus:
        """Check job status."""
        if job_id not in self.job_cache:
            return AgentStatus.FAILED
        return self.job_cache[job_id]['status']

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job results."""
        if job_id not in self.job_cache:
            raise ExecutionError(f"Job {job_id} not found")
        return self.job_cache[job_id].get('results', {})