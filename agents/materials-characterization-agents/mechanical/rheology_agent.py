"""Rheologist Agent - Rheology Expert (Refactored).

VERSION 2.0.0 - DMA and tensile testing extracted to specialized agents.

Capabilities:
- Oscillatory Rheology: G', G'', frequency/temperature sweeps, SAOS, LAOS
- Steady Shear: Viscosity curves, flow curves, thixotropy, yield stress
- Extensional Rheology: FiSER, CaBER, Hencky strain, strain-hardening
- Microrheology: Passive (DWS, particle tracking), Active (optical tweezers, AFM), GSER
- Peel Testing: 90°, 180°, T-peel configurations, adhesion characterization

DEPRECATED TECHNIQUES (moved to specialized agents):
- DMA → Use DMAAgent
- tensile/compression/flexural → Use TensileTestingAgent
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from base_agent import (
    ExperimentalAgent,
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


class RheologistAgent(ExperimentalAgent):
    """Rheology characterization agent (refactored).

    VERSION 2.0.0 - Focused on fluid rheology only.

    Supports rheology techniques:
    - Oscillatory: G', G'', frequency sweeps, SAOS, LAOS
    - Steady Shear: Viscosity curves, flow curves, yield stress
    - Extensional: FiSER, CaBER, Hencky strain
    - Microrheology: Passive/active, GSER, local properties
    - Peel: 90°, 180°, T-peel adhesion testing

    DEPRECATED (use specialized agents):
    - DMA → DMAAgent
    - tensile/compression/flexural → TensileTestingAgent
    """

    VERSION = "2.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize rheologist agent.

        Args:
            config: Configuration with instrument settings, calibration, etc.
        """
        super().__init__(config)
        self.supported_techniques = [
            'oscillatory', 'steady_shear',
            'extensional', 'microrheology', 'peel'
        ]

        # Deprecated techniques (redirected to specialized agents)
        self.deprecated_techniques = {
            'DMA': 'Use DMAAgent for dynamic mechanical analysis',
            'tensile': 'Use TensileTestingAgent for tensile testing',
            'compression': 'Use TensileTestingAgent for compression testing',
            'flexural': 'Use TensileTestingAgent for flexural testing'
        }

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute rheology measurement/analysis.

        Args:
            input_data: Input with keys:
                - technique: str (oscillatory, steady_shear, DMA, tensile, etc.)
                - sample_file: str (path to data file) OR sample_description: dict
                - parameters: dict (technique-specific parameters)
                - mode: str ('measure' or 'analyze', default='analyze')

        Returns:
            AgentResult with rheology data and analysis

        Example:
            >>> agent = RheologistAgent()
            >>> result = agent.execute({
            ...     'technique': 'oscillatory',
            ...     'sample_file': 'polymer_gel.dat',
            ...     'parameters': {'temperature': 298, 'freq_range': [0.1, 100]}
            ... })
        """
        start_time = datetime.now()
        technique = input_data.get('technique', 'oscillatory')

        try:
            # Check for deprecated techniques
            if technique in self.deprecated_techniques:
                return AgentResult(
                    agent_name=self.metadata.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=[f"Technique '{technique}' is deprecated. {self.deprecated_techniques[technique]}"],
                    warnings=[]
                )

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

            # Route to appropriate technique
            if technique == 'oscillatory':
                result_data = self._execute_oscillatory(input_data)
            elif technique == 'steady_shear':
                result_data = self._execute_steady_shear(input_data)
            elif technique == 'extensional':
                result_data = self._execute_extensional(input_data)
            elif technique == 'microrheology':
                result_data = self._execute_microrheology(input_data)
            elif technique == 'peel':
                result_data = self._execute_peel(input_data)
            else:
                raise ExecutionError(f"Unsupported technique: {technique}")

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
                    'instrument': self.instrument_config.get('model', 'simulated'),
                    'temperature': input_data.get('parameters', {}).get('temperature', 298)
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'technique': technique,
                    'execution_time_sec': execution_time
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
        technique = data.get('technique')
        if not technique:
            errors.append("Missing required field: 'technique'")
        elif technique not in self.supported_techniques:
            errors.append(
                f"Unsupported technique: {technique}. "
                f"Supported: {', '.join(self.supported_techniques)}"
            )

        # Check for data source
        if 'sample_file' not in data and 'sample_description' not in data:
            errors.append("Must provide either 'sample_file' or 'sample_description'")

        # Validate parameters
        params = data.get('parameters', {})
        if technique == 'oscillatory':
            if 'freq_range' in params:
                freq_range = params['freq_range']
                if not isinstance(freq_range, list) or len(freq_range) != 2:
                    errors.append("freq_range must be [min_freq, max_freq]")
                elif freq_range[0] >= freq_range[1]:
                    errors.append("freq_range: min must be < max")
            if 'strain_percent' in params:
                strain = params['strain_percent']
                if strain <= 0 or strain > 100:
                    errors.append(f"Invalid strain: {strain}% (must be 0-100%)")
                elif strain < 0.1 or strain > 10:
                    warnings.append(f"Unusual strain: {strain}% (typical 0.1-10%)")

        if technique == 'steady_shear':
            if 'shear_rate_range' in params:
                rate_range = params['shear_rate_range']
                if not isinstance(rate_range, list) or len(rate_range) != 2:
                    errors.append("shear_rate_range must be [min_rate, max_rate]")

        if technique in ['tensile', 'compression', 'flexural']:
            if 'strain_rate' in params:
                if params['strain_rate'] <= 0:
                    errors.append("strain_rate must be positive")

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
        technique = data.get('technique', 'oscillatory')

        # Resource requirements by technique
        resource_map = {
            'oscillatory': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=600,  # 10 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'steady_shear': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=900,  # 15 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'extensional': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.5,
                gpu_count=0,
                estimated_time_sec=1200,  # 20 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'microrheology': ResourceRequirement(
                cpu_cores=4,
                memory_gb=2.0,
                gpu_count=0,
                estimated_time_sec=1800,  # 30 minutes (computationally intensive)
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'peel': ResourceRequirement(
                cpu_cores=1,
                memory_gb=0.5,
                gpu_count=0,
                estimated_time_sec=300,  # 5 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            )
        }

        return resource_map.get(technique, ResourceRequirement(
            cpu_cores=2,
            memory_gb=1.0,
            gpu_count=0,
            estimated_time_sec=900,
            execution_environment=ExecutionEnvironment.LOCAL
        ))

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name="oscillatory",
                description="Oscillatory rheology for viscoelastic properties",
                input_types=["frequency_sweep", "temperature_sweep", "amplitude_sweep"],
                output_types=["storage_modulus_G_prime", "loss_modulus_G_double_prime", "complex_viscosity", "tan_delta"],
                typical_use_cases=[
                    "Viscoelastic characterization",
                    "Gel point determination",
                    "Polymer melt rheology",
                    "SAOS and LAOS analysis"
                ]
            ),
            Capability(
                name="steady_shear",
                description="Steady shear for viscosity and flow behavior",
                input_types=["shear_rate_sweep", "stress_sweep"],
                output_types=["viscosity", "shear_stress", "flow_curve", "yield_stress"],
                typical_use_cases=[
                    "Viscosity measurement",
                    "Shear thinning/thickening",
                    "Yield stress determination",
                    "Thixotropy analysis"
                ]
            ),
            Capability(
                name="extensional",
                description="Extensional rheology for elongational behavior",
                input_types=["filament_stretching", "capillary_breakup"],
                output_types=["extensional_viscosity", "relaxation_time", "strain_hardening"],
                typical_use_cases=[
                    "Fiber spinning",
                    "Film blowing",
                    "Coating flows",
                    "Strain-hardening assessment"
                ]
            ),
            Capability(
                name="microrheology",
                description="Microrheology for local viscoelastic properties",
                input_types=["particle_tracking", "DWS", "optical_tweezers"],
                output_types=["local_modulus", "spatial_heterogeneity", "high_freq_rheology"],
                typical_use_cases=[
                    "Local vs. bulk properties",
                    "High-frequency rheology (MHz)",
                    "Small sample volumes",
                    "Spatial heterogeneity mapping"
                ]
            ),
            Capability(
                name="peel",
                description="Peel testing for adhesion characterization",
                input_types=["peel_curve"],
                output_types=["peel_strength", "adhesion_energy", "failure_mode"],
                typical_use_cases=[
                    "Adhesive strength",
                    "Laminate bonding",
                    "Coating adhesion",
                    "Tape characterization"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata
        """
        return AgentMetadata(
            name="RheologistAgent",
            version=self.VERSION,
            description="Rheology and mechanical properties characterization expert",
            author="Materials Science Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dat', 'txt', 'csv', 'xlsx', 'rheo']
        )

    # Instrument connection

    def connect_instrument(self) -> bool:
        """Connect to rheometer or mechanical testing instrument.

        Returns:
            True if connected, False otherwise
        """
        instrument_mode = self.instrument_config.get('mode', 'simulated')
        if instrument_mode == 'simulated':
            return True
        # TODO: Implement real instrument connection (TA Instruments, Anton Paar, Instron, etc.)
        return False

    def process_experimental_data(self, raw_data: Any) -> Dict[str, Any]:
        """Process raw experimental data.

        Args:
            raw_data: Raw data from instrument

        Returns:
            Processed data dictionary
        """
        # In real implementation: instrument-specific data processing
        return {'raw': raw_data}

    # Technique implementations

    def _execute_oscillatory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute oscillatory rheology (SAOS/LAOS).

        Measures viscoelastic properties: G' (storage modulus), G'' (loss modulus)

        Args:
            input_data: Input with parameters (freq_range, strain, temperature)

        Returns:
            Oscillatory rheology results
        """
        params = input_data.get('parameters', {})
        freq_range = params.get('freq_range', [0.1, 100])  # Hz
        strain = params.get('strain_percent', 1.0)
        temperature = params.get('temperature', 298)  # K

        # In production: perform frequency sweep, extract G', G''
        # Simulated result for demo
        frequencies = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 20)

        # Simulate polymer gel behavior (G' > G'' at low freq)
        G_prime = 1000 * (frequencies ** 0.1)  # Pa
        G_double_prime = 500 * (frequencies ** 0.3)  # Pa

        result = {
            'technique': 'oscillatory',
            'frequency_Hz': frequencies.tolist(),
            'storage_modulus_G_prime_Pa': G_prime.tolist(),
            'loss_modulus_G_double_prime_Pa': G_double_prime.tolist(),
            'complex_viscosity_Pa_s': (np.sqrt(G_prime**2 + G_double_prime**2) / (2 * np.pi * frequencies)).tolist(),
            'tan_delta': (G_double_prime / G_prime).tolist(),
            'crossover_frequency_Hz': 2.5,  # Where G' = G''
            'strain_percent': strain,
            'temperature_K': temperature,
            'test_type': 'SAOS' if strain <= 5 else 'LAOS'
        }

        return result

    def _execute_steady_shear(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute steady shear rheology.

        Measures viscosity as function of shear rate

        Args:
            input_data: Input with shear_rate_range, temperature

        Returns:
            Steady shear results (flow curves, viscosity)
        """
        params = input_data.get('parameters', {})
        shear_rate_range = params.get('shear_rate_range', [0.1, 1000])  # 1/s
        temperature = params.get('temperature', 298)

        # Simulate shear-thinning polymer (power-law fluid)
        shear_rates = np.logspace(np.log10(shear_rate_range[0]), np.log10(shear_rate_range[1]), 20)

        # Power-law: η = K * γ̇^(n-1), n < 1 for shear-thinning
        K = 10.0  # Pa·s^n
        n = 0.6  # Power-law index (shear-thinning)
        viscosity = K * (shear_rates ** (n - 1))
        shear_stress = viscosity * shear_rates

        result = {
            'technique': 'steady_shear',
            'shear_rate_1_per_s': shear_rates.tolist(),
            'viscosity_Pa_s': viscosity.tolist(),
            'shear_stress_Pa': shear_stress.tolist(),
            'zero_shear_viscosity_Pa_s': viscosity[0],  # At lowest shear rate
            'power_law_index_n': n,
            'flow_behavior': 'shear-thinning' if n < 1 else 'Newtonian' if n == 1 else 'shear-thickening',
            'temperature_K': temperature
        }

        return result


    def _execute_extensional(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute extensional rheology (FiSER, CaBER).

        Measures elongational viscosity, relaxation time, strain-hardening

        Args:
            input_data: Input with method (FiSER/CaBER), strain_rate

        Returns:
            Extensional rheology results
        """
        params = input_data.get('parameters', {})
        method = params.get('method', 'FiSER')  # FiSER or CaBER
        strain_rate = params.get('strain_rate', 1.0)  # 1/s (Hencky)
        temperature = params.get('temperature', 298)

        # Simulate strain-hardening polymer
        time = np.linspace(0, 5, 100)  # seconds
        hencky_strain = strain_rate * time

        # Extensional viscosity with strain-hardening
        eta_0 = 1000  # Pa·s (zero-shear viscosity)
        strain_hardening_factor = 1 + 2 * hencky_strain  # Linear strain-hardening
        eta_E = 3 * eta_0 * strain_hardening_factor  # Trouton ratio = 3 for Newtonian

        result = {
            'technique': 'extensional',
            'method': method,
            'time_s': time.tolist(),
            'hencky_strain': hencky_strain.tolist(),
            'extensional_viscosity_eta_E_Pa_s': eta_E.tolist(),
            'strain_rate_1_per_s': strain_rate,
            'strain_hardening_parameter': eta_E[-1] / (3 * eta_0),  # ηE / 3η0
            'relaxation_time_s': 2.5 if method == 'CaBER' else None,  # CaBER measures relaxation
            'temperature_K': temperature
        }

        return result

    def _execute_microrheology(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute microrheology (passive or active).

        Measures local viscoelastic properties at microscale

        Args:
            input_data: Input with method (passive/active)

        Returns:
            Microrheology results (local G', G'')
        """
        params = input_data.get('parameters', {})
        method = params.get('method', 'passive')  # passive (DWS, tracking) or active (tweezers, AFM)
        temperature = params.get('temperature', 298)

        # Simulate frequency-dependent local moduli
        frequencies = np.logspace(-1, 6, 50)  # 0.1 Hz to 1 MHz

        # Local moduli (typically higher frequency than bulk rheology)
        G_prime_local = 100 * (frequencies ** 0.5)
        G_double_prime_local = 50 * (frequencies ** 0.6)

        result = {
            'technique': 'microrheology',
            'method': method,
            'frequency_Hz': frequencies.tolist(),
            'local_storage_modulus_G_prime_Pa': G_prime_local.tolist(),
            'local_loss_modulus_G_double_prime_Pa': G_double_prime_local.tolist(),
            'frequency_range': 'MHz' if method == 'active' else 'kHz',
            'spatial_resolution_um': 0.5 if method == 'active' else 1.0,
            'temperature_K': temperature,
            'notes': 'Generalized Stokes-Einstein relation (GSER) applied'
        }

        return result

    def _execute_peel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute peel testing (90°, 180°, T-peel).

        Measures adhesion strength

        Args:
            input_data: Input with peel_angle (90, 180, T)

        Returns:
            Peel test results
        """
        params = input_data.get('parameters', {})
        peel_angle = params.get('peel_angle', 180)  # degrees
        peel_rate = params.get('peel_rate', 300)  # mm/min
        temperature = params.get('temperature', 298)

        # Simulate peel curve
        displacement = np.linspace(0, 100, 200)  # mm

        # Peel force (relatively constant after initial rise)
        peel_force = 50 + 10 * np.sin(displacement / 5)  # N/m (force per unit width)

        result = {
            'technique': 'peel',
            'peel_angle_deg': peel_angle,
            'displacement_mm': displacement.tolist(),
            'peel_force_N_per_m': peel_force.tolist(),
            'average_peel_strength_N_per_m': np.mean(peel_force[50:]),  # After steady state
            'adhesion_energy_J_per_m2': np.mean(peel_force[50:]) * (1 - np.cos(np.radians(peel_angle))),  # Mode I fracture
            'peel_rate_mm_per_min': peel_rate,
            'temperature_K': temperature,
            'failure_mode': 'adhesive'  # or 'cohesive' or 'mixed'
        }

        return result

    # Integration methods

    def validate_with_md_viscosity(self, rheology_result: Dict[str, Any],
                                   md_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare experimental viscosity with MD simulation prediction.

        Args:
            rheology_result: Experimental rheology data (steady_shear or oscillatory)
            md_result: MD simulation result with predicted viscosity

        Returns:
            Validation results with agreement assessment
        """
        # Extract experimental viscosity
        if rheology_result.get('technique') == 'steady_shear':
            exp_viscosity = rheology_result.get('zero_shear_viscosity_Pa_s', 0)
        elif rheology_result.get('technique') == 'oscillatory':
            # Zero-shear viscosity from G', G'' at low frequency: η0 ≈ √(G'² + G''²) / ω
            complex_visc = rheology_result.get('complex_viscosity_Pa_s', [0])
            exp_viscosity = complex_visc[0] if isinstance(complex_visc, list) else complex_visc
        else:
            return {'error': 'Rheology technique must be steady_shear or oscillatory'}

        # Extract MD viscosity
        md_viscosity = md_result.get('predicted_viscosity_Pa_s', 0)

        if md_viscosity > 0 and exp_viscosity > 0:
            percent_diff = abs(exp_viscosity - md_viscosity) / exp_viscosity * 100
            agreement = percent_diff < 20  # Within 20% considered good

            validation = {
                'experimental_viscosity_Pa_s': exp_viscosity,
                'md_predicted_viscosity_Pa_s': md_viscosity,
                'percent_difference': percent_diff,
                'agreement': 'excellent' if percent_diff < 10 else 'good' if percent_diff < 20 else 'poor',
                'notes': f"{'Excellent' if percent_diff < 10 else 'Good' if percent_diff < 20 else 'Poor'} agreement between experiment and MD ({percent_diff:.1f}% difference)"
            }
        else:
            validation = {
                'agreement': 'unknown',
                'error': 'Viscosity data missing or invalid'
            }

        return validation

