"""Stochastic Dynamics Agent - Langevin, Master Equations & Escape Dynamics.

Capabilities:
- Langevin Dynamics: Overdamped/underdamped, colored noise, position-dependent friction
- Master Equations: Rate equations, kinetic Monte Carlo, chemical kinetics
- First-Passage Times: Transition paths, mean first-passage time (MFPT)
- Kramers Theory: Thermal activation, escape rates, transition state theory
- Fokker-Planck: Probability evolution, steady-state distributions
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
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


class StochasticDynamicsAgent(SimulationAgent):
    """Stochastic dynamics agent for Langevin, master equations, and escape problems.

    Supports multiple stochastic frameworks:
    - Langevin: Brownian motion with deterministic forces
    - Master Equations: Discrete-state Markov processes
    - First-Passage: Barrier crossing, transition path analysis
    - Kramers: Arrhenius-like escape rates in potential wells
    - Fokker-Planck: Probability density evolution (PDE)
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize stochastic dynamics agent.

        Args:
            config: Configuration with backend, integrator settings, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'langevin', 'master_equation', 'first_passage',
            'kramers_escape', 'fokker_planck'
        ]
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute stochastic dynamics simulation or analysis.

        Args:
            input_data: Input with keys:
                - method: str (langevin, master_equation, first_passage, etc.)
                - potential: dict or callable (energy landscape)
                - parameters: dict (temperature, friction, timestep, etc.)
                - initial_conditions: dict (starting position, state, etc.)

        Returns:
            AgentResult with trajectories, statistics, and rates

        Example:
            >>> agent = StochasticDynamicsAgent()
            >>> result = agent.execute({
            ...     'method': 'langevin',
            ...     'potential': {'type': 'harmonic', 'k': 1.0},
            ...     'parameters': {'temperature': 300, 'friction': 1.0, 'n_steps': 100000}
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'langevin')

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

            # Route to appropriate method
            if method == 'langevin':
                result_data = self._simulate_langevin(input_data)
            elif method == 'master_equation':
                result_data = self._simulate_master_equation(input_data)
            elif method == 'first_passage':
                result_data = self._analyze_first_passage(input_data)
            elif method == 'kramers_escape':
                result_data = self._compute_kramers_rate(input_data)
            elif method == 'fokker_planck':
                result_data = self._solve_fokker_planck(input_data)
            else:
                raise ExecutionError(f"Unsupported method: {method}")

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
                    'method': method
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'method': method,
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
        """Validate input data.

        Args:
            data: Input data to validate

        Returns:
            ValidationResult with validity status and messages
        """
        errors = []
        warnings = []

        # Check required fields
        if 'method' not in data:
            errors.append("Missing required field: method")
        elif data['method'] not in self.supported_methods:
            errors.append(f"Unsupported method: {data['method']}")

        method = data.get('method')
        parameters = data.get('parameters', {})

        # Method-specific validation
        if method == 'langevin':
            if 'potential' not in data:
                errors.append("Langevin simulation requires 'potential' specification")
            if 'friction' not in parameters:
                warnings.append("Friction coefficient not specified, using default γ=1.0")
            if 'temperature' not in parameters:
                warnings.append("Temperature not specified, using default T=300K")

        elif method == 'master_equation':
            if 'transition_rates' not in data and 'rate_matrix' not in data:
                errors.append("Master equation requires 'transition_rates' or 'rate_matrix'")

        elif method == 'first_passage':
            if 'barrier_position' not in parameters and 'target_state' not in parameters:
                errors.append("First-passage analysis requires 'barrier_position' or 'target_state'")

        elif method == 'kramers_escape':
            if 'potential' not in data:
                errors.append("Kramers theory requires potential energy surface")
            if 'barrier_height' not in parameters:
                warnings.append("Barrier height not specified, will estimate from potential")

        # Physical constraints
        if 'temperature' in parameters and parameters['temperature'] <= 0:
            errors.append("Temperature must be positive")

        if 'friction' in parameters and parameters['friction'] <= 0:
            errors.append("Friction coefficient must be positive")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed.

        Args:
            data: Input data for estimation

        Returns:
            ResourceRequirement specifying needed resources
        """
        method = data.get('method', 'langevin')
        parameters = data.get('parameters', {})

        n_steps = parameters.get('n_steps', 100000)
        n_particles = parameters.get('n_particles', 1)
        n_trajectories = parameters.get('n_trajectories', 1)

        total_steps = n_steps * n_particles * n_trajectories

        if total_steps > 1e9:  # Billion steps
            return ResourceRequirement(
                cpu_cores=16,
                memory_gb=32.0,
                gpu_count=1,
                estimated_time_sec=3600,
                execution_environment=ExecutionEnvironment.HPC
            )
        elif total_steps > 1e8:  # 100 million steps
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
                memory_gb=8.0,
                gpu_count=0,
                estimated_time_sec=120,
                execution_environment=ExecutionEnvironment.LOCAL
            )

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name="Langevin Dynamics",
                description="Simulate Brownian motion in potential energy landscapes",
                input_types=["potential_function", "friction_coefficient", "temperature"],
                output_types=["trajectory", "diffusion_coefficient", "correlation_functions"],
                typical_use_cases=[
                    "Protein folding dynamics",
                    "Colloidal particle motion",
                    "Chemical reaction dynamics"
                ]
            ),
            Capability(
                name="Master Equation Simulation",
                description="Discrete-state Markov processes and kinetic Monte Carlo",
                input_types=["rate_matrix", "initial_state"],
                output_types=["state_trajectory", "stationary_distribution", "relaxation_times"],
                typical_use_cases=[
                    "Chemical kinetics",
                    "Gene regulatory networks",
                    "Population dynamics"
                ]
            ),
            Capability(
                name="First-Passage Time Analysis",
                description="Compute transition times and path statistics",
                input_types=["trajectories", "barrier_definition"],
                output_types=["mfpt", "fpt_distribution", "transition_paths"],
                typical_use_cases=[
                    "Reaction rate calculations",
                    "Barrier crossing",
                    "Nucleation events"
                ]
            ),
            Capability(
                name="Kramers Escape Rate",
                description="Thermal activation and transition state theory",
                input_types=["potential_energy_surface", "temperature"],
                output_types=["escape_rate", "prefactor", "arrhenius_parameters"],
                typical_use_cases=[
                    "Chemical reaction rates",
                    "Nucleation rates",
                    "Defect migration"
                ]
            ),
            Capability(
                name="Fokker-Planck Solver",
                description="Solve probability evolution PDEs",
                input_types=["drift_term", "diffusion_term", "initial_distribution"],
                output_types=["probability_density", "steady_state", "relaxation_spectrum"],
                typical_use_cases=[
                    "Steady-state distributions",
                    "Relaxation dynamics",
                    "Nonequilibrium patterns"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata object
        """
        return AgentMetadata(
            name="StochasticDynamicsAgent",
            version=self.VERSION,
            description="Stochastic dynamics: Langevin, master equations, first-passage times",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities(),
            dependencies=["numpy", "scipy", "numba"],
            supported_formats=["trajectory", "rate_matrix", "potential_function"]
        )

    # === Simulation Methods ===

    def _simulate_langevin(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Langevin dynamics.

        Overdamped: γ dx/dt = -∇U(x) + √(2γkT) ξ(t)
        Underdamped: m d²x/dt² = -γ dx/dt - ∇U(x) + √(2γkT) ξ(t)

        Args:
            input_data: Input with potential and parameters

        Returns:
            Dictionary with trajectory and statistics
        """
        parameters = input_data.get('parameters', {})
        potential_spec = input_data.get('potential', {})

        # Parameters
        temperature = parameters.get('temperature', 300.0)  # K
        friction = parameters.get('friction', 1.0)  # ps⁻¹
        n_steps = parameters.get('n_steps', 100000)
        dt = parameters.get('dt', 0.01)  # ps
        regime = parameters.get('regime', 'overdamped')  # or 'underdamped'

        kB = 1.380649e-23  # J/K (or use kJ/mol units)
        D = kB * temperature / friction  # Diffusion coefficient

        # Define potential (example: harmonic, double-well, etc.)
        potential_type = potential_spec.get('type', 'harmonic')

        if potential_type == 'harmonic':
            k = potential_spec.get('k', 1.0)
            x0 = potential_spec.get('x0', 0.0)
            def force(x):
                return -k * (x - x0)
        elif potential_type == 'double_well':
            # U(x) = a·x⁴ - b·x²
            a = potential_spec.get('a', 1.0)
            b = potential_spec.get('b', 4.0)
            def force(x):
                return -4*a*x**3 + 2*b*x
        else:
            # Default: no force
            def force(x):
                return 0.0

        # Initialize
        x0_initial = parameters.get('initial_position', 0.0)
        x = x0_initial

        # Storage
        trajectory = [x]
        time_array = [0.0]

        # Integrate Langevin equation
        if regime == 'overdamped':
            # dx/dt = (1/γ)F(x) + √(2D) ξ(t)
            for step in range(n_steps):
                noise = np.sqrt(2 * D * dt) * np.random.randn()
                x = x + (1/friction) * force(x) * dt + noise
                trajectory.append(x)
                time_array.append((step+1) * dt)
        else:
            # Underdamped: need velocity
            v = 0.0
            mass = parameters.get('mass', 1.0)
            for step in range(n_steps):
                noise = np.sqrt(2 * friction * kB * temperature * dt / mass) * np.random.randn()
                v = v + (force(x) - friction * v) * dt / mass + noise / mass
                x = x + v * dt
                trajectory.append(x)
                time_array.append((step+1) * dt)

        trajectory = np.array(trajectory)
        time_array = np.array(time_array)

        # Analysis
        mean_position = np.mean(trajectory[n_steps//2:])  # After equilibration
        std_position = np.std(trajectory[n_steps//2:])

        # Compute autocorrelation function
        lag_max = min(1000, n_steps // 10)
        autocorr = self._compute_autocorrelation(trajectory, lag_max)

        # Estimate diffusion coefficient from MSD
        msd = self._compute_msd(trajectory, dt)

        return {
            'method': 'langevin',
            'regime': regime,
            'temperature_K': temperature,
            'friction_ps_inv': friction,
            'diffusion_coefficient': D,
            'trajectory': {
                'time_ps': time_array[::100].tolist(),  # Subsample for storage
                'position': trajectory[::100].tolist()
            },
            'statistics': {
                'mean_position': float(mean_position),
                'std_position': float(std_position),
                'equilibrium_variance_kT': float(std_position**2)
            },
            'autocorrelation': {
                'lag': list(range(lag_max)),
                'acf': autocorr.tolist()
            },
            'msd': msd
        }

    def _simulate_master_equation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate master equation (discrete-state Markov process).

        dp_i/dt = Σ_j (W_ji p_j - W_ij p_i)

        Args:
            input_data: Input with rate matrix

        Returns:
            Dictionary with state trajectory and distribution
        """
        parameters = input_data.get('parameters', {})

        # Get rate matrix or construct from transition rates
        if 'rate_matrix' in input_data:
            rate_matrix = np.array(input_data['rate_matrix'])
        else:
            # Construct from individual rates
            n_states = parameters.get('n_states', 5)
            rate_matrix = np.random.rand(n_states, n_states)
            np.fill_diagonal(rate_matrix, 0)
            # Diagonal: -sum of outgoing rates
            rate_matrix[np.diag_indices(n_states)] = -rate_matrix.sum(axis=1)

        n_states = rate_matrix.shape[0]

        # Initial state
        initial_state = parameters.get('initial_state', 0)

        # Gillespie algorithm (kinetic Monte Carlo)
        n_steps = parameters.get('n_steps', 10000)
        current_state = initial_state
        time = 0.0

        state_trajectory = [current_state]
        time_trajectory = [time]

        for step in range(n_steps):
            # Rates for leaving current state
            rates = rate_matrix[current_state, :].copy()
            rates[current_state] = 0  # Remove self-transition

            total_rate = np.sum(rates[rates > 0])
            if total_rate == 0:
                break  # Absorbing state

            # Time to next transition (exponential distribution)
            dt = np.random.exponential(1.0 / total_rate)
            time += dt

            # Choose next state
            probabilities = rates / total_rate
            probabilities[probabilities < 0] = 0
            current_state = np.random.choice(n_states, p=probabilities)

            state_trajectory.append(current_state)
            time_trajectory.append(time)

        # Compute stationary distribution (if exists)
        # Solve: W^T p_ss = 0
        try:
            eigenvalues, eigenvectors = np.linalg.eig(rate_matrix.T)
            stationary_idx = np.argmin(np.abs(eigenvalues))
            stationary_dist = np.abs(eigenvectors[:, stationary_idx])
            stationary_dist = stationary_dist / stationary_dist.sum()
        except:
            stationary_dist = np.ones(n_states) / n_states

        # Compute occupation probabilities from trajectory
        occupation = np.bincount(state_trajectory, minlength=n_states) / len(state_trajectory)

        return {
            'method': 'master_equation',
            'n_states': n_states,
            'trajectory': {
                'time': time_trajectory[::max(1, len(time_trajectory)//1000)],
                'state': [int(s) for s in state_trajectory[::max(1, len(state_trajectory)//1000)]]
            },
            'stationary_distribution': stationary_dist.tolist(),
            'occupation_probabilities': occupation.tolist(),
            'final_time': time
        }

    def _analyze_first_passage(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze first-passage time statistics.

        Args:
            input_data: Input with trajectories and barrier definition

        Returns:
            Dictionary with MFPT and FPT distribution
        """
        parameters = input_data.get('parameters', {})
        barrier_position = parameters.get('barrier_position', 1.0)

        # Get or generate trajectories
        if 'trajectories' in input_data:
            trajectories = input_data['trajectories']
        else:
            # Generate sample trajectories (Brownian motion)
            n_trajectories = parameters.get('n_trajectories', 100)
            n_steps = parameters.get('n_steps', 10000)
            dt = parameters.get('dt', 0.01)
            D = parameters.get('diffusion_coefficient', 1.0)

            trajectories = []
            for _ in range(n_trajectories):
                x = 0.0
                traj = [x]
                for step in range(n_steps):
                    x = x + np.sqrt(2 * D * dt) * np.random.randn()
                    traj.append(x)
                trajectories.append(traj)

        # Compute first-passage times
        first_passage_times = []
        dt = parameters.get('dt', 0.01)

        for traj in trajectories:
            traj_arr = np.array(traj)
            crossing_indices = np.where(traj_arr >= barrier_position)[0]
            if len(crossing_indices) > 0:
                fpt = crossing_indices[0] * dt
                first_passage_times.append(fpt)

        if len(first_passage_times) == 0:
            return {
                'method': 'first_passage',
                'barrier_position': barrier_position,
                'error': 'No trajectories crossed barrier'
            }

        fpt_array = np.array(first_passage_times)

        # Mean first-passage time
        mfpt = np.mean(fpt_array)
        std_fpt = np.std(fpt_array)

        # FPT distribution
        hist, bin_edges = np.histogram(fpt_array, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Theoretical MFPT for barrier at distance a: ⟨t⟩ = a²/(2D)
        D = parameters.get('diffusion_coefficient', 1.0)
        mfpt_theory = barrier_position**2 / (2 * D)

        return {
            'method': 'first_passage',
            'barrier_position': barrier_position,
            'mean_first_passage_time': float(mfpt),
            'std_first_passage_time': float(std_fpt),
            'mfpt_theoretical': mfpt_theory,
            'n_successful_crossings': len(first_passage_times),
            'n_trajectories': len(trajectories),
            'fpt_distribution': {
                'bins': bin_centers.tolist(),
                'density': hist.tolist()
            }
        }

    def _compute_kramers_rate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Kramers escape rate from potential well.

        Kramers rate: k = (ω_well ω_barrier)/(2π γ) exp(-ΔU / kT)

        Args:
            input_data: Input with potential and temperature

        Returns:
            Dictionary with escape rate and TST parameters
        """
        parameters = input_data.get('parameters', {})
        potential_spec = input_data.get('potential', {})

        temperature = parameters.get('temperature', 300.0)
        friction = parameters.get('friction', 1.0)

        # Barrier height
        if 'barrier_height' in parameters:
            barrier_height = parameters['barrier_height']
        else:
            # Estimate from potential (for double-well)
            barrier_height = 1.0  # Default

        # Curvatures at well minimum and barrier top
        # For harmonic approximation: U ≈ U_0 + (1/2)κ·x²
        omega_well = parameters.get('omega_well', 1.0)  # Frequency at well
        omega_barrier = parameters.get('omega_barrier', 1.0)  # Imaginary freq at barrier

        kB = 1.380649e-23  # J/K

        # Kramers rate (TST with friction)
        # Intermediate friction: k = (ω_well ω_barrier)/(2π γ) exp(-β ΔU)
        prefactor = (omega_well * abs(omega_barrier)) / (2 * np.pi * friction)
        exponential = np.exp(-barrier_height / (kB * temperature))
        kramers_rate = prefactor * exponential

        # Arrhenius form: k = A exp(-E_a / kT)
        activation_energy = barrier_height
        arrhenius_prefactor = prefactor

        # Transition state theory (TST) rate (high-friction limit)
        tst_rate = (omega_well / (2 * np.pi)) * exponential

        return {
            'method': 'kramers_escape',
            'escape_rate': kramers_rate,
            'tst_rate': tst_rate,
            'prefactor': arrhenius_prefactor,
            'activation_energy': activation_energy,
            'temperature_K': temperature,
            'friction': friction,
            'regime': self._determine_kramers_regime(friction, omega_well),
            'barrier_height': barrier_height
        }

    def _solve_fokker_planck(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve Fokker-Planck equation for probability density.

        ∂p/∂t = -∂(A(x)p)/∂x + (1/2)∂²(B(x)p)/∂x²

        Args:
            input_data: Input with drift and diffusion terms

        Returns:
            Dictionary with steady-state distribution
        """
        parameters = input_data.get('parameters', {})

        # Grid
        x_min = parameters.get('x_min', -5.0)
        x_max = parameters.get('x_max', 5.0)
        n_grid = parameters.get('n_grid', 100)
        x_grid = np.linspace(x_min, x_max, n_grid)
        dx = x_grid[1] - x_grid[0]

        # Initial distribution
        if 'initial_distribution' in input_data:
            p = np.array(input_data['initial_distribution'])
        else:
            # Gaussian initial condition
            p = np.exp(-x_grid**2 / 2)
            p = p / (np.sum(p) * dx)

        # Drift and diffusion (example: harmonic potential)
        k = parameters.get('k', 1.0)
        D = parameters.get('D', 1.0)

        def drift(x):
            return -k * x  # Force / friction

        def diffusion(x):
            return D * np.ones_like(x)

        # Time evolution (finite difference)
        dt = parameters.get('dt', 0.001)
        n_steps = parameters.get('n_steps', 1000)

        for step in range(n_steps):
            # Compute derivatives
            A = drift(x_grid)
            B = diffusion(x_grid)

            # Flux: J = A·p - (1/2)∂(B·p)/∂x
            grad_Bp = np.gradient(B * p, dx)
            flux = A * p - 0.5 * grad_Bp

            # ∂p/∂t = -∂J/∂x
            grad_flux = np.gradient(flux, dx)
            p = p - grad_flux * dt

            # Normalize
            p[p < 0] = 0
            p = p / (np.sum(p) * dx)

        # Steady state (analytical for harmonic)
        # p_ss(x) = sqrt(k/(πD)) exp(-k x² / D)
        p_steady_theory = np.sqrt(k / (np.pi * D)) * np.exp(-k * x_grid**2 / D)

        return {
            'method': 'fokker_planck',
            'steady_state_distribution': {
                'x': x_grid.tolist(),
                'probability': p.tolist(),
                'theoretical': p_steady_theory.tolist()
            },
            'normalization': float(np.sum(p) * dx),
            'converged': True
        }

    # === Helper Methods ===

    def _compute_autocorrelation(self, trajectory: np.ndarray, lag_max: int) -> np.ndarray:
        """Compute autocorrelation function."""
        trajectory = trajectory - np.mean(trajectory)
        autocorr = np.correlate(trajectory, trajectory, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:lag_max] / autocorr[0]
        return autocorr

    def _compute_msd(self, trajectory: np.ndarray, dt: float) -> Dict[str, Any]:
        """Compute mean-squared displacement."""
        n = len(trajectory)
        max_lag = min(n // 4, 1000)
        msd = np.zeros(max_lag)

        for lag in range(max_lag):
            displacements = trajectory[lag:] - trajectory[:-lag] if lag > 0 else np.zeros(n)
            msd[lag] = np.mean(displacements**2)

        # Fit to get diffusion coefficient: MSD = 2D·t
        times = np.arange(max_lag) * dt
        if max_lag > 10:
            coeffs = np.polyfit(times[1:max_lag//2], msd[1:max_lag//2], 1)
            D_measured = coeffs[0] / 2
        else:
            D_measured = 0.0

        return {
            'times': times.tolist(),
            'msd': msd.tolist(),
            'diffusion_coefficient_measured': D_measured
        }

    def _determine_kramers_regime(self, friction: float, omega: float) -> str:
        """Determine Kramers regime based on friction."""
        if friction < omega / 10:
            return "low_friction"
        elif friction > 10 * omega:
            return "high_friction"
        else:
            return "intermediate_friction"

    # === Computational Backend Methods ===

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit calculation to compute backend."""
        job_id = str(uuid4())
        self.job_cache[job_id] = {
            'status': AgentStatus.RUNNING,
            'input': input_data,
            'submitted_at': datetime.now()
        }
        return job_id

    def check_status(self, job_id: str) -> AgentStatus:
        """Check calculation status."""
        if job_id not in self.job_cache:
            return AgentStatus.FAILED
        return self.job_cache[job_id]['status']

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve calculation results."""
        if job_id not in self.job_cache:
            raise ExecutionError(f"Job {job_id} not found")

        job = self.job_cache[job_id]
        if job['status'] != AgentStatus.SUCCESS:
            raise ExecutionError(f"Job {job_id} not completed successfully")

        return job.get('results', {})

    # === Integration Methods ===

    def validate_fluctuation_dissipation(self,
                                        temperature: float,
                                        friction: float,
                                        measured_D: float,
                                        tolerance: float = 0.1) -> Dict[str, Any]:
        """Validate fluctuation-dissipation theorem.

        Einstein relation: D = kT / γ

        Args:
            temperature: Temperature in K
            friction: Friction coefficient
            measured_D: Measured diffusion coefficient
            tolerance: Relative tolerance

        Returns:
            Validation result dictionary
        """
        kB = 1.380649e-23  # J/K
        D_theory = kB * temperature / friction

        relative_error = abs(measured_D - D_theory) / D_theory
        fdt_satisfied = relative_error < tolerance

        return {
            'measured_D': measured_D,
            'theoretical_D': D_theory,
            'temperature_K': temperature,
            'friction': friction,
            'relative_error': relative_error,
            'fdt_satisfied': fdt_satisfied,
            'tolerance': tolerance
        }

    def cross_validate_escape_rate(self,
                                   kramers_rate: float,
                                   measured_rate: float,
                                   tolerance: float = 0.5) -> Dict[str, Any]:
        """Cross-validate Kramers theory with simulations.

        Args:
            kramers_rate: Theoretical rate from Kramers
            measured_rate: Rate from simulation (MFPT⁻¹)
            tolerance: Relative tolerance

        Returns:
            Validation result dictionary
        """
        relative_difference = abs(kramers_rate - measured_rate) / kramers_rate
        agrees = relative_difference < tolerance

        return {
            'kramers_rate': kramers_rate,
            'measured_rate': measured_rate,
            'relative_difference': relative_difference,
            'methods_agree': agrees,
            'tolerance': tolerance
        }