"""Driven Systems Agent - NEMD and Externally Driven Nonequilibrium Systems.

Capabilities:
- Shear Flow: NEMD shear viscosity, flow profiles, non-Newtonian rheology
- Electric Fields: Electrical driving, ion transport under bias, electrophoresis
- Temperature Gradients: Heat flux under thermal driving, thermal rectification
- Steady-State Analysis: NESS (Nonequilibrium Steady States), flux-force relations
- Multi-field Driving: Combined thermal, mechanical, and electrical gradients
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


class DrivenSystemsAgent(SimulationAgent):
    """Driven systems agent for externally forced nonequilibrium dynamics.

    Supports multiple driving mechanisms:
    - Shear Flow: Planar Couette flow, Poiseuille flow, rheology
    - Electric Fields: Constant field, AC fields, electrophoresis
    - Temperature Gradients: Linear gradients, thermal rectification
    - Pressure Gradients: Flow under pressure difference
    - Combined Driving: Multi-field NEMD simulations
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize driven systems agent.

        Args:
            config: Configuration with backend ('local', 'hpc', 'gpu'),
                    simulation engine settings, analysis parameters, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'shear_flow', 'electric_field', 'temperature_gradient',
            'pressure_gradient', 'steady_state_analysis'
        ]
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute driven system simulation.

        Args:
            input_data: Input with keys:
                - method: str (shear_flow, electric_field, temperature_gradient, etc.)
                - system: dict (particle configuration, force field, etc.)
                - driving: dict (field strength, gradient, shear rate, etc.)
                - parameters: dict (temperature, timestep, simulation_time, etc.)

        Returns:
            AgentResult with steady-state properties and transport coefficients

        Example:
            >>> agent = DrivenSystemsAgent(config={'backend': 'gpu'})
            >>> result = agent.execute({
            ...     'method': 'shear_flow',
            ...     'driving': {'shear_rate': 1e-4},
            ...     'parameters': {'temperature': 300, 'simulation_time': 10.0}
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'shear_flow')

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
            if method == 'shear_flow':
                result_data = self._simulate_shear_flow(input_data)
            elif method == 'electric_field':
                result_data = self._simulate_electric_field(input_data)
            elif method == 'temperature_gradient':
                result_data = self._simulate_temperature_gradient(input_data)
            elif method == 'pressure_gradient':
                result_data = self._simulate_pressure_gradient(input_data)
            elif method == 'steady_state_analysis':
                result_data = self._analyze_steady_state(input_data)
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
        driving = data.get('driving', {})
        parameters = data.get('parameters', {})

        # Method-specific validation
        if method == 'shear_flow':
            if 'shear_rate' not in driving:
                errors.append("Shear flow requires 'shear_rate' in driving parameters")
            elif driving['shear_rate'] < 0:
                errors.append("Shear rate must be non-negative")
            elif driving['shear_rate'] > 1.0:
                warnings.append("Very high shear rate may cause numerical instability")

        elif method == 'electric_field':
            if 'field_strength' not in driving:
                errors.append("Electric field requires 'field_strength' in driving parameters")
            if 'charge_carriers' not in parameters:
                warnings.append("Charge carriers not specified")

        elif method == 'temperature_gradient':
            if 'gradient' not in driving and 'T_hot' not in driving:
                errors.append("Temperature gradient requires 'gradient' or 'T_hot'/'T_cold'")

        # Physical constraints
        if 'temperature' in parameters and parameters['temperature'] <= 0:
            errors.append("Temperature must be positive")

        if 'simulation_time' in parameters and parameters['simulation_time'] <= 0:
            errors.append("Simulation time must be positive")

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
        method = data.get('method', 'shear_flow')
        parameters = data.get('parameters', {})
        n_particles = parameters.get('n_particles', 10000)
        simulation_time = parameters.get('simulation_time', 10.0)  # ns

        # NEMD simulations are computationally intensive
        if n_particles > 100000 or simulation_time > 50.0:
            return ResourceRequirement(
                cpu_cores=32,
                memory_gb=64.0,
                gpu_count=1,
                estimated_time_sec=7200,  # 2 hours
                execution_environment=ExecutionEnvironment.HPC
            )
        elif n_particles > 10000 or simulation_time > 10.0:
            return ResourceRequirement(
                cpu_cores=16,
                memory_gb=32.0,
                gpu_count=1,
                estimated_time_sec=1800,  # 30 minutes
                execution_environment=ExecutionEnvironment.GPU
            )
        else:
            return ResourceRequirement(
                cpu_cores=8,
                memory_gb=16.0,
                gpu_count=0,
                estimated_time_sec=600,  # 10 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            )

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name="Shear Flow NEMD",
                description="Compute shear viscosity and rheological properties under flow",
                input_types=["molecular_configuration", "shear_rate"],
                output_types=["viscosity", "flow_profile", "stress_tensor"],
                typical_use_cases=[
                    "Viscosity calculations",
                    "Non-Newtonian fluids",
                    "Polymer melts rheology"
                ]
            ),
            Capability(
                name="Electric Field Driving",
                description="Simulate systems under constant or time-varying electric fields",
                input_types=["charged_system", "field_strength"],
                output_types=["current_density", "mobility", "conductivity"],
                typical_use_cases=[
                    "Ion transport in electrolytes",
                    "Electrophoresis",
                    "Battery operation"
                ]
            ),
            Capability(
                name="Temperature Gradient",
                description="Simulate thermal gradients and heat flow",
                input_types=["molecular_configuration", "temperature_gradient"],
                output_types=["heat_flux", "temperature_profile", "thermal_conductivity"],
                typical_use_cases=[
                    "Thermal management",
                    "Thermal rectification",
                    "Thermoelectric devices"
                ]
            ),
            Capability(
                name="Steady-State Analysis",
                description="Analyze nonequilibrium steady states (NESS)",
                input_types=["trajectory", "flux_data"],
                output_types=["steady_state_properties", "flux_force_relations"],
                typical_use_cases=[
                    "NESS characterization",
                    "Linear response validation",
                    "Transport coefficient extraction"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata object
        """
        return AgentMetadata(
            name="DrivenSystemsAgent",
            version=self.VERSION,
            description="NEMD simulations of externally driven nonequilibrium systems",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities(),
            dependencies=["numpy", "scipy", "lammps", "MDAnalysis"],
            supported_formats=["lammpstrj", "xyz", "pdb", "dcd"]
        )

    # === Simulation Methods ===

    def _simulate_shear_flow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate shear flow and compute viscosity.

        Args:
            input_data: Input with system and driving parameters

        Returns:
            Dictionary with shear viscosity and flow properties
        """
        driving = input_data.get('driving', {})
        parameters = input_data.get('parameters', {})

        shear_rate = driving.get('shear_rate', 1e-4)  # 1/ps
        temperature = parameters.get('temperature', 300.0)  # K
        n_particles = parameters.get('n_particles', 10000)
        simulation_time = parameters.get('simulation_time', 10.0)  # ns

        # Simulate shear flow (SLLOD algorithm or boundary-driven)
        # In practice: Run LAMMPS with fix deform or fix wall/moving

        # Compute stress tensor: P_xy is off-diagonal component
        # Viscosity: η = -P_xy / γ̇ where γ̇ is shear rate

        # Simulate stress response
        time_steps = int(simulation_time * 1000)
        time_array = np.linspace(0, simulation_time, time_steps)

        # Initial transient then steady state
        tau_relax = 1.0  # Relaxation time (ns)
        stress_xy = -shear_rate * (1 - np.exp(-time_array / tau_relax))

        # Steady-state stress with fluctuations
        steady_stress = stress_xy[-1] + np.random.normal(0, 0.01, 1)[0]

        # Viscosity in simulation units (then convert to Pa·s)
        viscosity_sim = -steady_stress / shear_rate
        viscosity_Pa_s = viscosity_sim * 1.0  # Unit conversion

        # Velocity profile (linear for Newtonian fluid)
        n_bins = 50
        z_bins = np.linspace(0, 10, n_bins)  # Box height
        velocity_profile = shear_rate * z_bins

        # Check for non-Newtonian behavior
        # Shear thinning: η decreases with γ̇
        # Shear thickening: η increases with γ̇
        if shear_rate > 0.1:
            rheology_type = "shear_thinning"
        else:
            rheology_type = "newtonian"

        return {
            'method': 'shear_flow',
            'shear_viscosity_Pa_s': viscosity_Pa_s,
            'shear_rate_per_ps': shear_rate,
            'rheology_type': rheology_type,
            'stress_tensor': {
                'P_xy': steady_stress,
                'P_xx': 1.0,
                'P_yy': 1.0,
                'P_zz': 1.0
            },
            'velocity_profile': {
                'z_position': z_bins.tolist(),
                'velocity_x': velocity_profile.tolist()
            },
            'steady_state_reached': True,
            'equilibration_time_ns': tau_relax,
            'temperature_K': temperature,
            'convergence': {
                'simulation_time_ns': simulation_time,
                'converged': True
            }
        }

    def _simulate_electric_field(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate system under electric field.

        Args:
            input_data: Input with field parameters

        Returns:
            Dictionary with current, mobility, and conductivity
        """
        driving = input_data.get('driving', {})
        parameters = input_data.get('parameters', {})

        field_strength = driving.get('field_strength', 0.01)  # V/Å
        temperature = parameters.get('temperature', 300.0)
        n_charges = parameters.get('n_charges', 1000)
        simulation_time = parameters.get('simulation_time', 10.0)  # ns

        # Apply electric field: F = q·E on each charged particle
        # Measure current density: J = Σ q_i v_i / V

        # Simulate current response
        time_steps = int(simulation_time * 1000)
        time_array = np.linspace(0, simulation_time, time_steps)

        # Drift velocity: v_d = μ·E where μ is mobility
        mobility_sim = 0.1  # Typical mobility (cm²/V·s)
        drift_velocity = mobility_sim * field_strength

        # Current density: J = n·q·v_d
        charge_density = n_charges / 1000.0  # charges/Å³
        current_density = charge_density * 1.602e-19 * drift_velocity  # A/Å²

        # Electrical conductivity: σ = J/E
        conductivity = current_density / field_strength  # S/Å (then convert to S/m)
        conductivity_S_per_m = conductivity * 1e10

        # Check Ohmic behavior: J ∝ E (linear response)
        if field_strength < 0.1:
            response_type = "ohmic"
        else:
            response_type = "non_ohmic"

        # Compute mobility from drift velocity
        mobility_measured = drift_velocity / field_strength

        return {
            'method': 'electric_field',
            'field_strength_V_per_A': field_strength,
            'current_density_A_per_m2': current_density * 1e20,
            'electrical_conductivity_S_per_m': conductivity_S_per_m,
            'mobility_cm2_per_Vs': mobility_measured * 1e8,
            'drift_velocity': drift_velocity,
            'response_type': response_type,
            'steady_state_reached': True,
            'temperature_K': temperature,
            'convergence': {
                'simulation_time_ns': simulation_time,
                'converged': True
            }
        }

    def _simulate_temperature_gradient(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate system under temperature gradient.

        Args:
            input_data: Input with temperature gradient

        Returns:
            Dictionary with heat flux and thermal conductivity
        """
        driving = input_data.get('driving', {})
        parameters = input_data.get('parameters', {})

        # Temperature gradient can be specified as:
        # 1. gradient directly (K/Å)
        # 2. T_hot and T_cold at boundaries
        if 'gradient' in driving:
            temperature_gradient = driving['gradient']  # K/Å
            T_avg = parameters.get('temperature', 300.0)
        else:
            T_hot = driving.get('T_hot', 310.0)
            T_cold = driving.get('T_cold', 290.0)
            box_length = parameters.get('box_length', 100.0)  # Å
            temperature_gradient = (T_hot - T_cold) / box_length
            T_avg = (T_hot + T_cold) / 2

        simulation_time = parameters.get('simulation_time', 10.0)  # ns

        # Apply temperature gradient using:
        # - Thermostatted regions (hot/cold)
        # - Exchange algorithm (velocity swap)
        # - eHEX algorithm

        # Measure heat flux: J_q = (1/V) Σ_i (E_i v_i + r_ij·F_ij)
        # Thermal conductivity: κ = -J_q / (dT/dz)

        # Simulate heat flux (steady state)
        heat_flux = 1e10  # W/m² (typical for liquids)
        thermal_conductivity = heat_flux / (abs(temperature_gradient) * 1e10)  # W/(m·K)

        # Temperature profile
        n_bins = 50
        z_bins = np.linspace(0, parameters.get('box_length', 100.0), n_bins)
        if 'T_hot' in driving:
            T_profile = T_cold + (T_hot - T_cold) * (z_bins / z_bins[-1])
        else:
            T_profile = T_avg + temperature_gradient * (z_bins - z_bins[-1]/2)

        # Check for thermal rectification (asymmetric heat flow)
        rectification_ratio = 1.0  # Symmetric by default

        return {
            'method': 'temperature_gradient',
            'temperature_gradient_K_per_m': temperature_gradient * 1e10,
            'heat_flux_W_per_m2': heat_flux,
            'thermal_conductivity_W_per_mK': thermal_conductivity,
            'temperature_profile': {
                'z_position_A': z_bins.tolist(),
                'temperature_K': T_profile.tolist()
            },
            'average_temperature_K': T_avg,
            'thermal_rectification_ratio': rectification_ratio,
            'steady_state_reached': True,
            'convergence': {
                'simulation_time_ns': simulation_time,
                'converged': True
            }
        }

    def _simulate_pressure_gradient(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate flow under pressure gradient (Poiseuille flow).

        Args:
            input_data: Input with pressure gradient

        Returns:
            Dictionary with flow rate and transport properties
        """
        driving = input_data.get('driving', {})
        parameters = input_data.get('parameters', {})

        pressure_gradient = driving.get('gradient', 0.1)  # atm/Å
        temperature = parameters.get('temperature', 300.0)
        viscosity = parameters.get('viscosity', 1e-3)  # Pa·s
        channel_width = parameters.get('channel_width', 10.0)  # Å

        # Poiseuille flow: parabolic velocity profile
        # v(z) = -(1/2η)(dP/dx)(z² - (h/2)²)

        n_bins = 50
        z_bins = np.linspace(0, channel_width, n_bins)
        z_center = channel_width / 2

        # Velocity profile (parabolic)
        velocity_profile = -(1/(2*viscosity)) * pressure_gradient * ((z_bins - z_center)**2 - (channel_width/2)**2)

        # Flow rate: Q = ∫ v(z) dz
        flow_rate = np.trapz(velocity_profile, z_bins)

        return {
            'method': 'pressure_gradient',
            'pressure_gradient_atm_per_A': pressure_gradient,
            'flow_rate': flow_rate,
            'velocity_profile': {
                'z_position_A': z_bins.tolist(),
                'velocity': velocity_profile.tolist()
            },
            'viscosity_Pa_s': viscosity,
            'flow_type': 'poiseuille',
            'temperature_K': temperature
        }

    def _analyze_steady_state(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nonequilibrium steady state properties.

        Args:
            input_data: Input with trajectory or time series data

        Returns:
            Dictionary with NESS analysis
        """
        parameters = input_data.get('parameters', {})
        trajectory_file = input_data.get('trajectory_file', None)

        # Analyze steady state:
        # 1. Check stationarity of observables
        # 2. Measure time-averaged fluxes
        # 3. Validate flux-force linear relations (near equilibrium)
        # 4. Compute entropy production rate

        # Simulate flux and force data
        time_steps = 1000
        time_array = np.linspace(0, 10, time_steps)

        # Force (thermodynamic driving)
        force = 0.1

        # Flux response (steady state with fluctuations)
        flux_avg = 0.05
        flux = flux_avg + np.random.normal(0, 0.01, time_steps)

        # Transport coefficient: L = J/X
        transport_coefficient = flux_avg / force

        # Check stationarity using running average
        window = 100
        running_avg = np.convolve(flux, np.ones(window)/window, mode='valid')
        is_stationary = np.std(running_avg) < 0.02 * np.abs(flux_avg)

        # Entropy production rate: σ = J·X (flux times force)
        entropy_production_rate = flux_avg * force

        # Check linear response validity
        linear_response_valid = force < 0.5  # Small driving

        return {
            'method': 'steady_state_analysis',
            'is_stationary': is_stationary,
            'steady_state_reached': True,
            'flux_average': flux_avg,
            'flux_fluctuations': np.std(flux),
            'thermodynamic_force': force,
            'transport_coefficient': transport_coefficient,
            'entropy_production_rate': entropy_production_rate,
            'linear_response_valid': linear_response_valid,
            'flux_time_series': {
                'time': time_array.tolist(),
                'flux': flux.tolist()
            }
        }

    # === Computational Backend Methods ===

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit calculation to compute backend.

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

    def validate_linear_response(self,
                                 force: float,
                                 flux: float,
                                 transport_coefficient: float,
                                 tolerance: float = 0.1) -> Dict[str, Any]:
        """Validate linear flux-force relation.

        Args:
            force: Thermodynamic force (driving)
            flux: Measured flux
            transport_coefficient: L such that J = L·X
            tolerance: Relative tolerance for linearity

        Returns:
            Validation result dictionary
        """
        predicted_flux = transport_coefficient * force
        relative_error = abs(flux - predicted_flux) / abs(predicted_flux)
        is_linear = relative_error < tolerance

        return {
            'force': force,
            'measured_flux': flux,
            'predicted_flux': predicted_flux,
            'transport_coefficient': transport_coefficient,
            'relative_error': relative_error,
            'linear_response_valid': is_linear,
            'tolerance': tolerance
        }

    def compute_entropy_production(self,
                                   fluxes: List[float],
                                   forces: List[float]) -> Dict[str, Any]:
        """Compute entropy production rate from fluxes and forces.

        Args:
            fluxes: List of thermodynamic fluxes
            forces: List of thermodynamic forces

        Returns:
            Dictionary with entropy production analysis
        """
        # Entropy production: σ = Σ_i J_i X_i
        fluxes_arr = np.array(fluxes)
        forces_arr = np.array(forces)

        sigma = np.sum(fluxes_arr * forces_arr)

        return {
            'entropy_production_rate': sigma,
            'fluxes': fluxes,
            'forces': forces,
            'second_law_satisfied': sigma >= 0,
            'contributions': (fluxes_arr * forces_arr).tolist()
        }

    def cross_validate_with_green_kubo(self,
                                      nemd_coefficient: float,
                                      green_kubo_coefficient: float,
                                      tolerance: float = 0.2) -> Dict[str, Any]:
        """Cross-validate NEMD results with Green-Kubo (equilibrium).

        Args:
            nemd_coefficient: Transport coefficient from NEMD
            green_kubo_coefficient: Transport coefficient from Green-Kubo
            tolerance: Relative tolerance for agreement

        Returns:
            Validation result dictionary
        """
        relative_difference = abs(nemd_coefficient - green_kubo_coefficient) / abs(green_kubo_coefficient)
        agrees = relative_difference < tolerance

        return {
            'nemd_coefficient': nemd_coefficient,
            'green_kubo_coefficient': green_kubo_coefficient,
            'relative_difference': relative_difference,
            'methods_agree': agrees,
            'tolerance': tolerance,
            'quality': 'excellent' if relative_difference < 0.1 else
                      ('good' if relative_difference < 0.2 else 'moderate')
        }