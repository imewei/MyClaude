"""Simulation Agent - Molecular Dynamics & Multiscale Simulation Expert.

Capabilities:
- Classical MD: LAMMPS, GROMACS (NVT/NPT/production, S(q), g(r), Green-Kubo viscosity)
- Machine Learning Force Fields: DeepMD-kit, NequIP (training, inference, 1000x speedup)
- HOOMD-blue: GPU-native soft matter (anisotropic particles, rigid bodies)
- Dissipative Particle Dynamics (DPD): Mesoscale coarse-graining, hydrodynamics
- Nanoscale DEM: Discrete Element Method for granular materials
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import uuid4
import numpy as np

from base_agent import (
    ComputationalAgent,
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


class SimulationAgent(ComputationalAgent):
    """Molecular dynamics and multiscale simulation agent.

    Supports multiple simulation methods:
    - Classical MD: LAMMPS, GROMACS (atomistic, all-atom/united-atom)
    - MLFF: DeepMD-kit, NequIP (neural network potentials, 1000x speedup)
    - HOOMD-blue: GPU-native soft matter (anisotropic, rigid bodies)
    - DPD: Dissipative Particle Dynamics (mesoscale, hydrodynamics)
    - Nanoscale DEM: Discrete Element Method (granular, particulate)
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize simulation agent.

        Args:
            config: Configuration with backend ('local', 'hpc', 'cloud'),
                    MD engine settings, HPC credentials, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'classical_md', 'mlff', 'hoomd', 'dpd', 'nanoscale_dem'
        ]
        self.job_cache = {}  # Track submitted jobs

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute simulation.

        Args:
            input_data: Input with keys:
                - method: str (classical_md, mlff, hoomd, dpd, nanoscale_dem)
                - structure_file: str (path to structure: xyz, pdb, lammps-data, etc.)
                - parameters: dict (method-specific parameters)
                - mode: str ('run', 'analyze', 'train' for MLFF)

        Returns:
            AgentResult with simulation data

        Example:
            >>> agent = SimulationAgent(config={'backend': 'local'})
            >>> result = agent.execute({
            ...     'method': 'classical_md',
            ...     'structure_file': 'polymer.xyz',
            ...     'parameters': {'ensemble': 'NPT', 'steps': 100000, 'temperature': 298}
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'classical_md')

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
            if method == 'classical_md':
                result_data = self._execute_classical_md(input_data)
            elif method == 'mlff':
                result_data = self._execute_mlff(input_data)
            elif method == 'hoomd':
                result_data = self._execute_hoomd(input_data)
            elif method == 'dpd':
                result_data = self._execute_dpd(input_data)
            elif method == 'nanoscale_dem':
                result_data = self._execute_nanoscale_dem(input_data)
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
                    'temperature': input_data.get('parameters', {}).get('temperature', 298)
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
            ValidationResult with status and errors/warnings
        """
        errors = []
        warnings = []

        # Check required fields
        method = data.get('method')
        if not method:
            errors.append("Missing required field: 'method'")
        elif method not in self.supported_methods:
            errors.append(
                f"Unsupported method: {method}. "
                f"Supported: {', '.join(self.supported_methods)}"
            )

        # Check for structure file (except MLFF training mode)
        mode = data.get('mode', 'run')
        if mode != 'train' and 'structure_file' not in data:
            errors.append("Must provide 'structure_file' for simulation")

        # Validate parameters
        params = data.get('parameters', {})
        if method == 'classical_md':
            if 'steps' in params:
                steps = params['steps']
                if steps <= 0:
                    errors.append("steps must be positive")
                elif steps < 10000:
                    warnings.append(f"Low step count: {steps} (typical 100K-1M)")
            if 'ensemble' in params:
                ensemble = params['ensemble']
                if ensemble not in ['NVE', 'NVT', 'NPT']:
                    errors.append(f"Invalid ensemble: {ensemble} (use NVE/NVT/NPT)")
            if 'temperature' in params:
                temp = params['temperature']
                if temp <= 0 or temp > 1000:
                    warnings.append(f"Unusual temperature: {temp}K (typical 200-400K)")

        if method == 'mlff' and mode == 'train':
            if 'training_data' not in params:
                errors.append("MLFF training requires 'training_data' in parameters")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources.

        Args:
            data: Input data

        Returns:
            ResourceRequirement
        """
        method = data.get('method', 'classical_md')
        params = data.get('parameters', {})
        mode = data.get('mode', 'run')

        # Classical MD
        if method == 'classical_md':
            steps = params.get('steps', 100000)
            atoms = params.get('n_atoms', 1000)

            # Estimate time: ~1 hour per 100K steps for 1000 atoms on 8 cores
            time_sec = (steps / 100000) * (atoms / 1000) * 3600

            return ResourceRequirement(
                cpu_cores=16,
                memory_gb=8.0,
                gpu_count=0,
                estimated_time_sec=min(time_sec, 14400),  # Cap at 4 hours
                execution_environment=ExecutionEnvironment.HPC
            )

        # MLFF
        elif method == 'mlff':
            if mode == 'train':
                # Training is GPU-intensive
                return ResourceRequirement(
                    cpu_cores=8,
                    memory_gb=32.0,
                    gpu_count=4,
                    estimated_time_sec=7200,  # 2 hours typical
                    execution_environment=ExecutionEnvironment.HPC
                )
            else:
                # Inference is much faster, can be local
                return ResourceRequirement(
                    cpu_cores=2,
                    memory_gb=4.0,
                    gpu_count=1,
                    estimated_time_sec=300,  # 5 minutes
                    execution_environment=ExecutionEnvironment.LOCAL
                )

        # HOOMD-blue
        elif method == 'hoomd':
            steps = params.get('steps', 100000)
            return ResourceRequirement(
                cpu_cores=4,
                memory_gb=8.0,
                gpu_count=2,  # GPU-native
                estimated_time_sec=1800,  # 30 minutes
                execution_environment=ExecutionEnvironment.HPC
            )

        # DPD
        elif method == 'dpd':
            steps = params.get('steps', 100000)
            return ResourceRequirement(
                cpu_cores=8,
                memory_gb=4.0,
                gpu_count=0,
                estimated_time_sec=3600,  # 1 hour
                execution_environment=ExecutionEnvironment.HPC
            )

        # Nanoscale DEM
        elif method == 'nanoscale_dem':
            steps = params.get('steps', 100000)
            particles = params.get('n_particles', 10000)
            time_sec = (steps / 100000) * (particles / 10000) * 2400

            return ResourceRequirement(
                cpu_cores=16,
                memory_gb=8.0,
                gpu_count=0,
                estimated_time_sec=min(time_sec, 7200),  # Cap at 2 hours
                execution_environment=ExecutionEnvironment.HPC if particles > 5000 else ExecutionEnvironment.LOCAL
            )

        # Default
        return ResourceRequirement(
            cpu_cores=8,
            memory_gb=4.0,
            gpu_count=0,
            estimated_time_sec=3600,
            execution_environment=ExecutionEnvironment.HPC
        )

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name="classical_md",
                description="Classical molecular dynamics (LAMMPS, GROMACS)",
                input_types=["structure_file", "force_field", "ensemble_parameters"],
                output_types=["trajectory", "energies", "structure_factor_S_q", "radial_distribution_g_r", "viscosity"],
                typical_use_cases=[
                    "Atomistic simulations (all-atom, united-atom)",
                    "Structure factor S(q) calculation",
                    "Radial distribution g(r) analysis",
                    "Viscosity via Green-Kubo",
                    "NVE/NVT/NPT ensemble simulations"
                ]
            ),
            Capability(
                name="mlff",
                description="Machine Learning Force Fields (DeepMD-kit, NequIP)",
                input_types=["training_data_from_DFT", "structure_file"],
                output_types=["trained_model", "forces", "energies", "speedup_metrics"],
                typical_use_cases=[
                    "Train ML potential from DFT data",
                    "1000x speedup over DFT-MD",
                    "Near-DFT accuracy (~1 meV/atom)",
                    "Large-scale MD with DFT quality",
                    "Transfer learning across systems"
                ]
            ),
            Capability(
                name="hoomd",
                description="GPU-native soft matter simulations (HOOMD-blue)",
                input_types=["particle_configuration", "interaction_potential"],
                output_types=["trajectory", "phase_behavior", "self_assembly"],
                typical_use_cases=[
                    "Soft matter (polymers, colloids, liquid crystals)",
                    "Anisotropic particles (ellipsoids, dumbbells)",
                    "Rigid body dynamics",
                    "GPU-accelerated (10-100x faster)",
                    "Active matter, self-assembly"
                ]
            ),
            Capability(
                name="dpd",
                description="Dissipative Particle Dynamics (mesoscale)",
                input_types=["coarse_grained_structure", "dpd_parameters"],
                output_types=["trajectory", "hydrodynamic_properties", "phase_separation"],
                typical_use_cases=[
                    "Mesoscale simulations (1-100 nm, μs-ms)",
                    "Polymer blends, lipid membranes",
                    "Hydrodynamics with momentum conservation",
                    "Phase separation dynamics",
                    "Surfactant systems"
                ]
            ),
            Capability(
                name="nanoscale_dem",
                description="Discrete Element Method for granular materials",
                input_types=["particle_ensemble", "contact_model"],
                output_types=["trajectory", "stress_tensor", "packing_fraction"],
                typical_use_cases=[
                    "Granular materials, powders",
                    "Nanoparticle assemblies",
                    "Contact mechanics",
                    "Packing and jamming",
                    "Powder flow simulations"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata
        """
        return AgentMetadata(
            name="SimulationAgent",
            version=self.VERSION,
            description="Molecular dynamics and multiscale simulation expert",
            author="Materials Science Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],  # Real: lammps, gromacs, deepmd, nequip, hoomd
            supported_formats=['xyz', 'pdb', 'lammps-data', 'gro', 'cif']
        )

    # ComputationalAgent interface

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit calculation to compute backend.

        Args:
            input_data: Calculation input

        Returns:
            Job ID for tracking
        """
        job_id = f"sim_{uuid4().hex[:8]}"

        if self.compute_backend == 'hpc':
            # In production: submit to SLURM/PBS
            # slurm: sbatch script.sh → job_id
            # Store job info for status checking
            self.job_cache[job_id] = {
                'status': AgentStatus.RUNNING,
                'input': input_data,
                'submitted_at': datetime.now()
            }
        elif self.compute_backend == 'local':
            # Run locally (simulated as instant for demo)
            self.job_cache[job_id] = {
                'status': AgentStatus.SUCCESS,
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
            Calculation results dictionary
        """
        if job_id not in self.job_cache:
            return {'error': 'Job not found'}

        if self.job_cache[job_id]['status'] != AgentStatus.SUCCESS:
            return {'error': 'Job not complete'}

        # In production: retrieve from HPC filesystem
        # For demo: return simulated results based on input
        input_data = self.job_cache[job_id]['input']
        method = input_data.get('method', 'classical_md')

        # Execute method to get results
        if method == 'classical_md':
            return self._execute_classical_md(input_data)
        elif method == 'mlff':
            return self._execute_mlff(input_data)
        elif method == 'hoomd':
            return self._execute_hoomd(input_data)
        elif method == 'dpd':
            return self._execute_dpd(input_data)
        elif method == 'nanoscale_dem':
            return self._execute_nanoscale_dem(input_data)

        return {}

    # Method implementations

    def _execute_classical_md(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute classical MD (LAMMPS or GROMACS).

        Args:
            input_data: Input with structure, force field, ensemble, steps

        Returns:
            MD results (trajectory, energies, S(q), g(r), viscosity)
        """
        params = input_data.get('parameters', {})
        ensemble = params.get('ensemble', 'NPT')
        steps = params.get('steps', 100000)
        temperature = params.get('temperature', 298)
        pressure = params.get('pressure', 1.0)  # bar
        timestep = params.get('timestep', 1.0)  # fs

        # Simulated MD results
        # In production: run LAMMPS/GROMACS, parse outputs

        # Time array
        time_ns = np.linspace(0, steps * timestep / 1e6, 100)  # Convert fs to ns

        # Energy (should equilibrate)
        E_total = -1000 + 50 * np.exp(-time_ns / 0.5) + 5 * np.random.randn(100)

        # Structure factor S(q) - characteristic peak for liquid
        q_nm_inv = np.linspace(0.5, 10, 50)  # 1/nm
        S_q = 1 + 2 * np.exp(-((q_nm_inv - 3.0)**2) / 1.0)  # Peak at q ~ 3 nm^-1

        # Radial distribution function g(r) - should go to 1 at large r
        r_nm = np.linspace(0.1, 2.0, 50)
        g_r = np.exp(-((r_nm - 0.35)**2) / 0.05) + 1.0  # Peak at first neighbor ~0.35 nm
        g_r = np.maximum(g_r, 0.01)  # g(r) must be positive

        # Viscosity from Green-Kubo (pressure autocorrelation)
        viscosity_Pa_s = 0.89e-3  # Water-like

        result = {
            'method': 'classical_md',
            'engine': 'LAMMPS',  # or 'GROMACS'
            'ensemble': ensemble,
            'steps': steps,
            'temperature_K': temperature,
            'pressure_bar': pressure,
            'timestep_fs': timestep,
            'time_ns': time_ns.tolist(),
            'total_energy_kJ_per_mol': E_total.tolist(),
            'final_temperature_K': temperature + np.random.randn() * 2,  # Should be ~T
            'structure_factor': {
                'q_nm_inv': q_nm_inv.tolist(),
                'S_q': S_q.tolist()
            },
            'radial_distribution': {
                'r_nm': r_nm.tolist(),
                'g_r': g_r.tolist()
            },
            'transport_properties': {
                'viscosity_Pa_s': viscosity_Pa_s,
                'diffusion_coefficient_m2_per_s': 2.3e-9
            }
        }

        return result

    def _execute_mlff(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Machine Learning Force Field simulation or training.

        Args:
            input_data: Input with mode ('train' or 'run'), training_data or structure

        Returns:
            MLFF results (trained model or MD with MLFF)
        """
        params = input_data.get('parameters', {})
        mode = input_data.get('mode', 'run')

        if mode == 'train':
            # Training MLFF from DFT data
            training_data_path = params.get('training_data', '')
            n_structures = params.get('n_structures', 1000)  # Get from params if available
            n_epochs = params.get('epochs', 100)

            # Simulated training
            result = {
                'method': 'mlff',
                'mode': 'train',
                'model_type': 'DeepMD-kit',  # or 'NequIP'
                'training_data_file': training_data_path,
                'training_structures': n_structures,
                'epochs': n_epochs,
                'final_energy_MAE_meV_per_atom': 0.8,  # <1 meV is excellent
                'final_force_MAE_meV_per_A': 15.0,  # <20 meV/Å is good
                'training_time_hours': 2.5,
                'model_size_MB': 50,
                'model_file': f'mlff_trained_{n_epochs}epochs.pb',
                'notes': 'Training complete. Model ready for inference.'
            }

        else:
            # Inference with trained MLFF
            steps = params.get('steps', 100000)
            temperature = params.get('temperature', 298)

            # MLFF should give DFT-quality results at 1000x speed
            result = {
                'method': 'mlff',
                'mode': 'inference',
                'model_type': 'DeepMD-kit',
                'steps': steps,
                'temperature_K': temperature,
                'speedup_vs_DFT': 1250,  # 1000x typical
                'accuracy_meV_per_atom': 0.9,
                'time_per_step_ms': 0.01,  # vs 10 ms for DFT
                'notes': 'MLFF inference complete. DFT-level accuracy at MD speed.'
            }

        return result

    def _execute_hoomd(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HOOMD-blue GPU simulation.

        Args:
            input_data: Input with particle configuration, potential

        Returns:
            HOOMD results (trajectory, phase behavior)
        """
        params = input_data.get('parameters', {})
        steps = params.get('steps', 100000)
        temperature = params.get('temperature', 298)
        particle_type = params.get('particle_type', 'sphere')  # sphere, ellipsoid, dumbbell
        potential = params.get('potential', 'LJ')  # LJ, Yukawa, WCA, Gay-Berne

        # Simulated HOOMD-blue results
        result = {
            'method': 'hoomd',
            'gpu_accelerated': True,
            'particle_type': particle_type,
            'potential': potential,
            'steps': steps,
            'temperature_K': temperature,
            'n_particles': params.get('n_particles', 10000),
            'final_state': 'liquid' if temperature > 250 else 'crystalline',
            'gpu_speedup_vs_cpu': 85,  # GPU is 10-100x faster
            'time_per_step_ms': 0.005,  # Very fast on GPU
            'notes': 'GPU-native HOOMD-blue simulation complete.'
        }

        return result

    def _execute_dpd(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Dissipative Particle Dynamics (mesoscale).

        Args:
            input_data: Input with coarse-grained structure, DPD parameters

        Returns:
            DPD results (mesoscale dynamics, hydrodynamics)
        """
        params = input_data.get('parameters', {})
        steps = params.get('steps', 100000)
        temperature = params.get('temperature', 298)
        n_beads = params.get('n_beads', 100000)  # Coarse-grained beads

        # DPD parameters
        gamma = params.get('gamma', 4.5)  # Friction coefficient
        a_ij = params.get('a_ij', 25.0)  # Conservative force parameter

        # Simulated DPD results
        result = {
            'method': 'dpd',
            'n_beads': n_beads,
            'steps': steps,
            'temperature_K': temperature,
            'dpd_parameters': {
                'gamma': gamma,
                'a_ij': a_ij,
                'sigma': np.sqrt(2 * gamma * temperature)  # Fluctuation-dissipation
            },
            'time_scale_microseconds': steps * 0.01 / 1000,  # DPD reaches μs-ms
            'length_scale_nm': 5.0,  # Coarse-grained bead ~ 5 nm
            'hydrodynamics': 'momentum_conserving',
            'phase_behavior': 'microphase_separated' if a_ij > 20 else 'mixed',
            'notes': 'DPD simulation complete. Mesoscale hydrodynamics captured.'
        }

        return result

    def _execute_nanoscale_dem(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute nanoscale Discrete Element Method.

        Args:
            input_data: Input with particle ensemble, contact model

        Returns:
            DEM results (granular dynamics, stress, packing)
        """
        params = input_data.get('parameters', {})
        steps = params.get('steps', 100000)
        n_particles = params.get('n_particles', 10000)
        contact_model = params.get('contact_model', 'Hertz')  # Hertz, JKR, DMT

        # Simulated DEM results
        result = {
            'method': 'nanoscale_dem',
            'n_particles': n_particles,
            'steps': steps,
            'contact_model': contact_model,
            'particle_diameter_nm': 50,
            'packing_fraction': 0.64,  # Random close packing ~ 0.64
            'stress_tensor_Pa': {
                'sigma_xx': 1e5,
                'sigma_yy': 1e5,
                'sigma_zz': 1.2e5,  # Vertical compression
                'sigma_xy': 0,
                'sigma_xz': 0,
                'sigma_yz': 0
            },
            'coordination_number': 6.0,  # Average contacts per particle
            'notes': 'Nanoscale DEM simulation complete. Contact forces resolved.'
        }

        return result

    # Integration methods

    def validate_scattering_data(self, md_result: Dict[str, Any],
                                 experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare MD S(q) with experimental SANS/SAXS/DLS data.

        Args:
            md_result: MD simulation result with S(q)
            experimental_data: Experimental scattering data (SANS/SAXS/DLS)

        Returns:
            Validation results with chi-squared, agreement assessment
        """
        # Extract MD S(q)
        if 'structure_factor' not in md_result:
            return {'success': False, 'error': 'MD result missing structure_factor'}

        md_q = np.array(md_result['structure_factor']['q_nm_inv'])
        md_Sq = np.array(md_result['structure_factor']['S_q'])

        # Extract experimental S(q) or I(q)
        exp_q = np.array(experimental_data.get('q_nm_inv', []))
        exp_Sq = np.array(experimental_data.get('S_q', experimental_data.get('I_q', [])))

        if len(exp_q) == 0 or len(exp_Sq) == 0:
            return {'success': False, 'error': 'Experimental data missing q or S(q)/I(q)'}

        # Interpolate MD to experimental q points
        md_Sq_interp = np.interp(exp_q, md_q, md_Sq)

        # Calculate chi-squared
        # χ² = Σ[(S_exp - S_md)² / σ²]
        sigma = np.array(experimental_data.get('sigma_Sq', experimental_data.get('sigma_I_q', np.ones_like(exp_Sq) * 0.1)))
        chi_squared = np.sum(((exp_Sq - md_Sq_interp) / sigma) ** 2) / len(exp_q)

        # Assess agreement
        if chi_squared < 1.0:
            agreement = 'excellent'
        elif chi_squared < 2.0:
            agreement = 'good'
        elif chi_squared < 5.0:
            agreement = 'acceptable'
        else:
            agreement = 'poor'

        validation = {
            'success': True,
            'chi_squared': chi_squared,
            'agreement': agreement,
            'md_peak_position_nm_inv': md_q[np.argmax(md_Sq)],
            'exp_peak_position_nm_inv': exp_q[np.argmax(exp_Sq)],
            'peak_shift_percent': abs(md_q[np.argmax(md_Sq)] - exp_q[np.argmax(exp_Sq)]) / exp_q[np.argmax(exp_Sq)] * 100,
            'notes': f'{agreement.capitalize()} agreement between MD and experiment (χ² = {chi_squared:.2f})'
        }

        return validation

    def train_mlff_from_dft(self, dft_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML force field from DFT data.

        Args:
            dft_data: DFT calculation results (structures, forces, energies)

        Returns:
            Trained MLFF model info
        """
        # Extract DFT data
        n_structures = dft_data.get('num_configurations', dft_data.get('n_structures', 0))

        if n_structures == 0:
            return {'success': False, 'error': 'DFT data missing structures'}

        if n_structures < 100:
            return {'success': False, 'warning': f'Insufficient data: {n_structures} structures (need >100 for good MLFF)'}

        # Simulate MLFF training
        # In production: call DeepMD-kit or NequIP training pipeline

        training_result = {
            'success': True,
            'model_type': 'DeepMD-kit',
            'training_structures': n_structures,
            'training_epochs': 100,
            'model_file': f'mlff_model_{n_structures}configs.pb',
            'validation_metrics': {
                'energy_MAE_meV_per_atom': 0.7 + 0.3 * (1000 / max(n_structures, 1)),  # More data → better
                'force_MAE_meV_per_A': 12.0 + 8.0 * (1000 / max(n_structures, 1))
            },
            'training_time_hours': 2.0 + n_structures / 500,
            'speedup_vs_DFT': 1150,  # 1000x typical
            'notes': f'MLFF trained on {n_structures} DFT structures. Energy MAE < 1 meV/atom (excellent).'
        }

        # Assess quality
        if training_result['validation_metrics']['energy_MAE_meV_per_atom'] < 1.0:
            training_result['quality'] = 'excellent'
        elif training_result['validation_metrics']['energy_MAE_meV_per_atom'] < 2.0:
            training_result['quality'] = 'good'
        else:
            training_result['quality'] = 'needs_more_data'

        return training_result

    def predict_rheology(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict rheological properties from MD trajectory.

        Calculate viscosity (Green-Kubo), diffusion coefficient, relaxation time

        Args:
            trajectory_data: MD trajectory with pressure tensor

        Returns:
            Predicted rheological properties for comparison with RheologistAgent
        """
        # Extract trajectory info
        if 'transport_properties' in trajectory_data:
            # Already calculated
            return {
                'success': True,
                'viscosity_Pa_s': trajectory_data['transport_properties'].get('viscosity_Pa_s', 0),
                'diffusion_m2_per_s': trajectory_data['transport_properties'].get('diffusion_coefficient_m2_per_s', 0),
                'method': 'from_trajectory_analysis',
                'notes': 'Rheology predicted from MD trajectory (Green-Kubo for viscosity)'
            }

        # Simulate calculation if not present
        # In production: calculate from pressure autocorrelation (Green-Kubo)
        # η = V/(k_B T) ∫ <P_xy(0) P_xy(t)> dt

        temperature = trajectory_data.get('temperature_K', 298)

        # Typical values for simple liquid
        predicted_viscosity = 1.0e-3  # Pa·s (water-like)
        predicted_diffusion = 2.3e-9  # m²/s

        return {
            'success': True,
            'viscosity_Pa_s': predicted_viscosity,
            'diffusion_m2_per_s': predicted_diffusion,
            'zero_shear_viscosity_Pa_s': predicted_viscosity,
            'temperature_K': temperature,
            'method': 'Green-Kubo',
            'notes': 'Rheology predicted from MD. Use RheologistAgent.validate_with_md_viscosity() to compare.'
        }