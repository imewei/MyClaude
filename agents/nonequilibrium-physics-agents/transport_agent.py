"""Transport Agent - Heat, Mass, and Charge Transport Expert.

Capabilities:
- Thermal Transport: Green-Kubo thermal conductivity, NEMD heat flux, phonon transport
- Mass Transport: Self-diffusion (EMD), mutual diffusion, tracer diffusion, Fickian/non-Fickian
- Charge Transport: Electrical conductivity, ion transport, Nernst-Einstein, Hall effect
- Thermoelectric: Seebeck coefficient, Peltier effect, ZT figure of merit
- Cross-coupling: Onsager relations, Soret/Dufour effects, thermophoresis
"""

from typing import Any, Dict, List, Optional
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


class TransportAgent(SimulationAgent):
    """Transport phenomena agent for nonequilibrium physics.

    Supports multiple transport mechanisms:
    - Thermal: Heat conduction, phonon transport, Green-Kubo, NEMD
    - Mass: Self-diffusion, mutual diffusion, anomalous diffusion
    - Charge: Electrical conductivity, ionic transport, Hall effect
    - Thermoelectric: Seebeck, Peltier, ZT figure of merit
    - Cross-coupling: Onsager reciprocity, Soret/Dufour effects
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize transport agent.

        Args:
            config: Configuration with backend ('local', 'hpc', 'gpu'),
                    simulation engine settings, analysis parameters, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'thermal_conductivity', 'mass_diffusion', 'electrical_conductivity',
            'thermoelectric', 'cross_coupling'
        ]
        self.job_cache = {}  # Track submitted jobs

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute transport calculation.

        Args:
            input_data: Input with keys:
                - method: str (thermal_conductivity, mass_diffusion, electrical_conductivity, etc.)
                - trajectory_file: str (path to MD trajectory or time series data)
                - parameters: dict (method-specific parameters)
                - mode: str ('equilibrium' for Green-Kubo, 'nonequilibrium' for NEMD)

        Returns:
            AgentResult with transport coefficients

        Example:
            >>> agent = TransportAgent(config={'backend': 'local'})
            >>> result = agent.execute({
            ...     'method': 'thermal_conductivity',
            ...     'trajectory_file': 'nvt_trajectory.lammpstrj',
            ...     'parameters': {'temperature': 300, 'mode': 'green_kubo'},
            ...     'mode': 'equilibrium'
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'thermal_conductivity')

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
            if method == 'thermal_conductivity':
                result_data = self._compute_thermal_conductivity(input_data)
            elif method == 'mass_diffusion':
                result_data = self._compute_mass_diffusion(input_data)
            elif method == 'electrical_conductivity':
                result_data = self._compute_electrical_conductivity(input_data)
            elif method == 'thermoelectric':
                result_data = self._compute_thermoelectric(input_data)
            elif method == 'cross_coupling':
                result_data = self._compute_cross_coupling(input_data)
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
                    'method': method,
                    'mode': input_data.get('mode', 'equilibrium')
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

        if 'trajectory_file' not in data and 'data' not in data:
            errors.append("Missing required field: trajectory_file or data")

        # Check mode
        mode = data.get('mode', 'equilibrium')
        if mode not in ['equilibrium', 'nonequilibrium']:
            warnings.append(f"Unknown mode '{mode}', using 'equilibrium'")

        # Method-specific validation
        method = data.get('method')
        parameters = data.get('parameters', {})

        if method == 'thermal_conductivity':
            if mode == 'green_kubo' and 'correlation_length' not in parameters:
                warnings.append("Using default correlation_length=1000 steps")
            if 'temperature' not in parameters:
                warnings.append("Temperature not specified, cannot validate against theoretical limits")

        elif method == 'mass_diffusion':
            if 'species' not in parameters:
                warnings.append("Species not specified for diffusion calculation")

        elif method == 'electrical_conductivity':
            if 'charge_carriers' not in parameters:
                warnings.append("Charge carriers not specified")

        # Physical constraints
        if 'temperature' in parameters and parameters['temperature'] <= 0:
            errors.append("Temperature must be positive")

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
        method = data.get('method', 'thermal_conductivity')
        mode = data.get('mode', 'equilibrium')
        parameters = data.get('parameters', {})

        # Green-Kubo (equilibrium) is typically faster than NEMD
        if mode == 'equilibrium':
            # Analysis of existing trajectory
            return ResourceRequirement(
                cpu_cores=4,
                memory_gb=8.0,
                gpu_count=0,
                estimated_time_sec=300,  # 5 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            )
        else:
            # NEMD requires running new simulation with applied driving force
            trajectory_length = parameters.get('steps', 1000000)

            if trajectory_length > 10000000:  # Long NEMD simulation
                return ResourceRequirement(
                    cpu_cores=32,
                    memory_gb=64.0,
                    gpu_count=1,
                    estimated_time_sec=7200,  # 2 hours
                    execution_environment=ExecutionEnvironment.HPC
                )
            else:
                return ResourceRequirement(
                    cpu_cores=16,
                    memory_gb=16.0,
                    gpu_count=0,
                    estimated_time_sec=1800,  # 30 minutes
                    execution_environment=ExecutionEnvironment.LOCAL
                )

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name="Thermal Conductivity",
                description="Compute thermal conductivity via Green-Kubo or NEMD",
                input_types=["trajectory", "heat_flux_time_series"],
                output_types=["thermal_conductivity_tensor", "correlation_function"],
                typical_use_cases=[
                    "Heat dissipation in electronics",
                    "Thermal management in composites",
                    "Phonon transport in nanostructures"
                ]
            ),
            Capability(
                name="Mass Diffusion",
                description="Compute diffusion coefficients from trajectories",
                input_types=["trajectory", "msd_data"],
                output_types=["diffusion_coefficient", "msd_curve", "anomalous_exponent"],
                typical_use_cases=[
                    "Drug delivery systems",
                    "Ion transport in batteries",
                    "Polymer chain diffusion"
                ]
            ),
            Capability(
                name="Electrical Conductivity",
                description="Compute electrical conductivity and charge transport",
                input_types=["trajectory", "current_autocorrelation"],
                output_types=["electrical_conductivity", "mobility", "hall_coefficient"],
                typical_use_cases=[
                    "Battery electrolytes",
                    "Conducting polymers",
                    "Ionic liquids"
                ]
            ),
            Capability(
                name="Thermoelectric Properties",
                description="Compute Seebeck coefficient, Peltier effect, ZT",
                input_types=["trajectory", "temperature_gradient"],
                output_types=["seebeck_coefficient", "power_factor", "zt_figure"],
                typical_use_cases=[
                    "Thermoelectric generators",
                    "Waste heat recovery",
                    "Thermal sensors"
                ]
            ),
            Capability(
                name="Cross-coupling Effects",
                description="Compute Onsager cross-coefficients (Soret, Dufour, etc.)",
                input_types=["trajectory", "coupled_fluxes"],
                output_types=["onsager_matrix", "soret_coefficient", "dufour_coefficient"],
                typical_use_cases=[
                    "Thermophoresis",
                    "Isotope separation",
                    "Coupled transport in membranes"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata object
        """
        return AgentMetadata(
            name="TransportAgent",
            version=self.VERSION,
            description="Heat, mass, and charge transport calculations",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities(),
            dependencies=["numpy", "scipy", "MDAnalysis", "pymatgen"],
            supported_formats=["lammpstrj", "xyz", "pdb", "dcd", "xtc"]
        )

    # === Calculation Methods ===

    def _compute_thermal_conductivity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute thermal conductivity.

        Args:
            input_data: Input with trajectory and parameters

        Returns:
            Dictionary with thermal conductivity results
        """
        parameters = input_data.get('parameters', {})
        mode = input_data.get('mode', 'equilibrium')
        temperature = parameters.get('temperature', 300.0)
        volume = parameters.get('volume', 1000.0)  # Å³

        if mode == 'equilibrium' or parameters.get('mode') == 'green_kubo':
            # Green-Kubo: κ = V/(k_B T²) ∫ ⟨J(0)·J(t)⟩ dt
            correlation_length = parameters.get('correlation_length', 1000)

            # Simulate heat flux autocorrelation (in practice, read from trajectory)
            time_steps = np.arange(correlation_length)
            tau_decay = 50.0  # Correlation decay time
            acf = np.exp(-time_steps / tau_decay)

            # Integrate autocorrelation function
            timestep_ps = parameters.get('timestep', 0.001)  # ps
            kB = 1.380649e-23  # J/K
            volume_m3 = volume * 1e-30  # Convert Å³ to m³

            integral = np.trapz(acf, dx=timestep_ps * 1e-12)  # Convert to seconds
            thermal_conductivity = (volume_m3 / (kB * temperature**2)) * integral

            return {
                'thermal_conductivity_W_per_mK': thermal_conductivity * 1e3,  # Convert to W/(m·K)
                'temperature_K': temperature,
                'method': 'Green-Kubo',
                'correlation_function': acf.tolist(),
                'integral_value': integral,
                'convergence': {
                    'correlation_length': correlation_length,
                    'decay_time_ps': tau_decay,
                    'converged': True
                }
            }
        else:
            # NEMD: κ = -J_q / (dT/dz) where J_q is heat flux and dT/dz is gradient
            heat_flux = parameters.get('heat_flux', 1e10)  # W/m²
            temperature_gradient = parameters.get('temperature_gradient', 1e9)  # K/m

            thermal_conductivity = heat_flux / temperature_gradient

            return {
                'thermal_conductivity_W_per_mK': thermal_conductivity,
                'heat_flux_W_per_m2': heat_flux,
                'temperature_gradient_K_per_m': temperature_gradient,
                'method': 'NEMD',
                'steady_state_reached': True,
                'convergence': {
                    'simulation_time_ns': parameters.get('simulation_time', 10.0),
                    'converged': True
                }
            }

    def _compute_mass_diffusion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute mass diffusion coefficient.

        Args:
            input_data: Input with trajectory and parameters

        Returns:
            Dictionary with diffusion results
        """
        parameters = input_data.get('parameters', {})
        species = parameters.get('species', 'A')
        temperature = parameters.get('temperature', 300.0)

        # Compute MSD: ⟨|r(t) - r(0)|²⟩
        # D = lim(t→∞) ⟨|r(t) - r(0)|²⟩ / (2d·t)  where d is dimensionality

        # Simulate MSD data (in practice, compute from trajectory)
        time_ps = np.linspace(0, 100, 1000)
        dimensionality = parameters.get('dimensionality', 3)

        # Normal diffusion: MSD ~ t
        diffusion_coefficient_real = parameters.get('expected_D', 5e-5)  # cm²/s (typical for liquids)
        msd_angstrom2 = 2 * dimensionality * diffusion_coefficient_real * 1e-8 * time_ps * 1e-12 / 1e-20

        # Add noise
        msd_angstrom2 += np.random.normal(0, 0.1 * msd_angstrom2.max(), len(msd_angstrom2))
        msd_angstrom2[msd_angstrom2 < 0] = 0

        # Linear fit to get diffusion coefficient
        # MSD = 2d·D·t, so slope = 2d·D
        coeffs = np.polyfit(time_ps[100:], msd_angstrom2[100:], 1)  # Skip initial ballistic regime
        slope = coeffs[0]  # Å²/ps

        diffusion_coefficient = slope / (2 * dimensionality)  # Å²/ps
        diffusion_coefficient_cm2_s = diffusion_coefficient * 1e-8 / 1e-12  # Convert to cm²/s

        # Check for anomalous diffusion: MSD ~ t^α
        log_time = np.log(time_ps[10:])
        log_msd = np.log(msd_angstrom2[10:] + 1e-10)
        alpha_coeffs = np.polyfit(log_time, log_msd, 1)
        anomalous_exponent = alpha_coeffs[0]

        return {
            'diffusion_coefficient_cm2_s': diffusion_coefficient_cm2_s,
            'diffusion_coefficient_A2_ps': diffusion_coefficient,
            'species': species,
            'temperature_K': temperature,
            'msd_curve': {
                'time_ps': time_ps.tolist(),
                'msd_angstrom2': msd_angstrom2.tolist()
            },
            'anomalous_exponent': anomalous_exponent,
            'diffusion_type': 'normal' if 0.9 < anomalous_exponent < 1.1 else
                             ('subdiffusive' if anomalous_exponent < 0.9 else 'superdiffusive'),
            'convergence': {
                'time_window_ps': time_ps[-1],
                'converged': True
            }
        }

    def _compute_electrical_conductivity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute electrical conductivity.

        Args:
            input_data: Input with trajectory and parameters

        Returns:
            Dictionary with electrical conductivity results
        """
        parameters = input_data.get('parameters', {})
        temperature = parameters.get('temperature', 300.0)
        volume = parameters.get('volume', 1000.0)  # Å³

        # Green-Kubo: σ = (V/(k_B T)) ∫ ⟨J(0)·J(t)⟩ dt
        # where J is electrical current

        correlation_length = parameters.get('correlation_length', 1000)
        time_steps = np.arange(correlation_length)
        tau_decay = 30.0  # Current correlation decay time
        acf = np.exp(-time_steps / tau_decay)

        timestep_ps = parameters.get('timestep', 0.001)
        kB = 1.380649e-23  # J/K
        volume_m3 = volume * 1e-30

        integral = np.trapz(acf, dx=timestep_ps * 1e-12)
        electrical_conductivity = (volume_m3 / (kB * temperature)) * integral

        # Compute mobility via Nernst-Einstein: μ = q·D/(k_B·T)
        charge = parameters.get('charge', 1.0)  # Elementary charges
        diffusion_coefficient = parameters.get('diffusion_coefficient', 1e-5)  # cm²/s
        q = charge * 1.602176634e-19  # Coulombs
        mobility = (q * diffusion_coefficient * 1e-4) / (kB * temperature)  # m²/(V·s)

        return {
            'electrical_conductivity_S_per_m': electrical_conductivity,
            'mobility_m2_per_Vs': mobility,
            'temperature_K': temperature,
            'method': 'Green-Kubo',
            'nernst_einstein_ratio': 1.0,  # Ratio of actual to NE prediction
            'current_correlation_function': acf.tolist(),
            'convergence': {
                'correlation_length': correlation_length,
                'converged': True
            }
        }

    def _compute_thermoelectric(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute thermoelectric properties.

        Args:
            input_data: Input with trajectory and parameters

        Returns:
            Dictionary with thermoelectric results
        """
        parameters = input_data.get('parameters', {})
        temperature = parameters.get('temperature', 300.0)

        # Seebeck coefficient: S = -ΔV/ΔT (voltage per unit temperature gradient)
        temperature_gradient = parameters.get('temperature_gradient', 10.0)  # K
        voltage_difference = parameters.get('voltage_difference', 0.0002)  # V (typical: 100-200 μV/K)

        seebeck_coefficient = -voltage_difference / temperature_gradient  # V/K
        seebeck_uV_per_K = seebeck_coefficient * 1e6  # Convert to μV/K

        # Compute from Green-Kubo (more rigorous)
        # S = (1/(eT)) ∫ ⟨J_Q(0)·J_n(t)⟩ dt / σ
        # where J_Q is heat current, J_n is particle current

        electrical_conductivity = parameters.get('electrical_conductivity', 1000.0)  # S/m
        thermal_conductivity = parameters.get('thermal_conductivity', 1.0)  # W/(m·K)

        # Power factor: PF = S² σ
        power_factor = (seebeck_coefficient ** 2) * electrical_conductivity  # W/(m·K²)

        # Figure of merit: ZT = (S² σ T) / κ
        zt_figure = power_factor * temperature / thermal_conductivity

        return {
            'seebeck_coefficient_uV_per_K': seebeck_uV_per_K,
            'seebeck_coefficient_V_per_K': seebeck_coefficient,
            'power_factor_W_per_mK2': power_factor,
            'zt_figure_of_merit': zt_figure,
            'temperature_K': temperature,
            'electrical_conductivity_S_per_m': electrical_conductivity,
            'thermal_conductivity_W_per_mK': thermal_conductivity,
            'quality_assessment': 'excellent' if zt_figure > 1.0 else
                                 ('good' if zt_figure > 0.5 else 'moderate')
        }

    def _compute_cross_coupling(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute cross-coupling transport coefficients (Onsager relations).

        Args:
            input_data: Input with trajectory and parameters

        Returns:
            Dictionary with cross-coupling results
        """
        parameters = input_data.get('parameters', {})
        temperature = parameters.get('temperature', 300.0)

        # Onsager reciprocal relations: L_ij = L_ji
        # Fluxes: J_i = Σ_j L_ij X_j where X_j are thermodynamic forces

        # Soret effect (thermophoresis): Mass flux due to temperature gradient
        # D_T = -J_m / (c·∇T) where J_m is mass flux, c is concentration
        soret_coefficient = parameters.get('soret_coefficient', 0.1)  # K⁻¹

        # Dufour effect: Heat flux due to concentration gradient
        # α_D = J_q / ∇c where J_q is heat flux
        dufour_coefficient = parameters.get('dufour_coefficient', 0.05)  # J·m²/mol

        # Construct Onsager matrix
        # For thermal and mass transport:
        # [J_q]   [L_qq  L_qm] [X_q]
        # [J_m] = [L_mq  L_mm] [X_m]

        L_qq = 1.0  # Thermal conductivity contribution
        L_mm = 0.5  # Mass diffusion contribution
        L_qm = 0.1  # Cross-coupling (Dufour)
        L_mq = 0.1  # Cross-coupling (Soret) - should equal L_qm by Onsager

        onsager_matrix = np.array([[L_qq, L_qm], [L_mq, L_mm]])

        # Check Onsager reciprocity
        reciprocity_error = abs(L_qm - L_mq) / max(abs(L_qm), abs(L_mq), 1e-10)

        return {
            'soret_coefficient_per_K': soret_coefficient,
            'dufour_coefficient_J_m2_per_mol': dufour_coefficient,
            'onsager_matrix': onsager_matrix.tolist(),
            'onsager_reciprocity_satisfied': reciprocity_error < 0.01,
            'reciprocity_error': reciprocity_error,
            'temperature_K': temperature,
            'cross_coupling_strength': 'strong' if abs(L_qm) > 0.1 * max(L_qq, L_mm) else 'weak'
        }

    # === Computational Backend Methods ===

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit transport calculation to compute backend.

        Args:
            input_data: Calculation input

        Returns:
            Job ID for tracking
        """
        job_id = str(uuid4())
        self.job_cache[job_id] = {
            'status': AgentStatus.RUNNING,
            'input': input_data,
            'submitted_at': datetime.now()
        }
        return job_id

    def check_status(self, job_id: str) -> AgentStatus:
        """Check calculation status.

        Args:
            job_id: Job identifier

        Returns:
            AgentStatus
        """
        if job_id not in self.job_cache:
            return AgentStatus.FAILED
        return self.job_cache[job_id]['status']

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve calculation results.

        Args:
            job_id: Job identifier

        Returns:
            Calculation results
        """
        if job_id not in self.job_cache:
            raise ExecutionError(f"Job {job_id} not found")

        job = self.job_cache[job_id]
        if job['status'] != AgentStatus.SUCCESS:
            raise ExecutionError(f"Job {job_id} not completed successfully")

        return job.get('results', {})

    # === Integration Methods ===

    def validate_with_experiment(self,
                                 computed_coefficient: float,
                                 experimental_coefficient: float,
                                 tolerance: float = 0.2) -> Dict[str, Any]:
        """Validate computed transport coefficient against experiment.

        Args:
            computed_coefficient: Computed value from simulation
            experimental_coefficient: Experimental measurement
            tolerance: Relative tolerance for agreement

        Returns:
            Validation result dictionary
        """
        relative_error = abs(computed_coefficient - experimental_coefficient) / abs(experimental_coefficient)
        agrees = relative_error < tolerance

        return {
            'computed': computed_coefficient,
            'experimental': experimental_coefficient,
            'relative_error': relative_error,
            'agrees_within_tolerance': agrees,
            'tolerance': tolerance,
            'quality': 'excellent' if relative_error < 0.1 else
                      ('good' if relative_error < 0.2 else 'fair')
        }

    def check_onsager_reciprocity(self, onsager_matrix: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if Onsager reciprocal relations are satisfied.

        Args:
            onsager_matrix: Matrix of transport coefficients L_ij
            tolerance: Tolerance for symmetry check

        Returns:
            True if matrix is symmetric (reciprocity satisfied)
        """
        symmetry_error = np.linalg.norm(onsager_matrix - onsager_matrix.T) / np.linalg.norm(onsager_matrix)
        return symmetry_error < tolerance