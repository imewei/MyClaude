"""DMA Agent - Dynamic Mechanical Analysis Expert.

Capabilities:
- Temperature Sweeps: E', E'', tan δ vs. temperature for glass transition (Tg) determination
- Frequency Sweeps: Master curves, time-temperature superposition (TTS)
- Isothermal: Stress relaxation, creep compliance, aging studies
- Multi-frequency: Multiple frequencies simultaneously for broadband characterization
- Stress/Strain Controlled: Constant stress or strain amplitude
- Characterization: Polymer Tg, β-relaxation, crystallinity effects, damping
- Cross-validation: With DSC (Tg), BDS (α-relaxation), rheology (G' comparison)
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


class DMAAgent(ExperimentalAgent):
    """Dynamic Mechanical Analysis (DMA) characterization agent.

    Supports multiple DMA techniques:
    - temperature_sweep: E', E'', tan δ vs. T for glass transition
    - frequency_sweep: Master curves and time-temperature superposition
    - isothermal: Stress relaxation and creep compliance
    - multi_frequency: Broadband characterization
    - stress_controlled: Constant stress amplitude
    - strain_controlled: Constant strain amplitude
    - creep_recovery: Creep and recovery for viscoelasticity
    - dynamic_strain_sweep: Amplitude sweeps for LVE determination
    """

    NAME = "DMAAgent"
    VERSION = "1.0.0"

    SUPPORTED_TECHNIQUES = [
        'temperature_sweep',
        'frequency_sweep',
        'isothermal',
        'multi_frequency',
        'stress_controlled',
        'strain_controlled',
        'creep_recovery',
        'dynamic_strain_sweep'
    ]

    # Typical DMA parameters
    FREQUENCY_RANGE_HZ = (0.01, 100)  # Typical DMA frequency range
    TEMPERATURE_RANGE_K = (100, 500)  # Typical temperature range
    STRAIN_AMPLITUDE_PERCENT = (0.01, 5.0)  # Linear viscoelastic range

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DMA agent.

        Args:
            config: Configuration with instrument settings, calibration, etc.
        """
        super().__init__(config)
        self.instrument_config = config or {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute DMA measurement/analysis.

        Args:
            input_data: Input with keys:
                - technique: str (temperature_sweep, frequency_sweep, etc.)
                - sample_file: str (path to data file) OR sample_description: dict
                - parameters: dict (technique-specific parameters)
                - mode: str ('measure' or 'analyze', default='analyze')

        Returns:
            AgentResult with DMA data and analysis

        Example:
            >>> agent = DMAAgent()
            >>> result = agent.execute({
            ...     'technique': 'temperature_sweep',
            ...     'sample_description': {'material': 'polystyrene'},
            ...     'parameters': {
            ...         'temp_range': [200, 400],
            ...         'heating_rate': 3,
            ...         'frequency': 1.0
            ...     }
            ... })
        """
        start_time = datetime.now()
        technique = input_data.get('technique', 'temperature_sweep')

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

            # Route to appropriate technique
            if technique == 'temperature_sweep':
                result_data = self._execute_temperature_sweep(input_data)
            elif technique == 'frequency_sweep':
                result_data = self._execute_frequency_sweep(input_data)
            elif technique == 'isothermal':
                result_data = self._execute_isothermal(input_data)
            elif technique == 'multi_frequency':
                result_data = self._execute_multi_frequency(input_data)
            elif technique == 'stress_controlled':
                result_data = self._execute_stress_controlled(input_data)
            elif technique == 'strain_controlled':
                result_data = self._execute_strain_controlled(input_data)
            elif technique == 'creep_recovery':
                result_data = self._execute_creep_recovery(input_data)
            elif technique == 'dynamic_strain_sweep':
                result_data = self._execute_dynamic_strain_sweep(input_data)
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
                    'technique': technique
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
        elif technique not in self.SUPPORTED_TECHNIQUES:
            errors.append(
                f"Unsupported technique: {technique}. "
                f"Supported: {', '.join(self.SUPPORTED_TECHNIQUES)}"
            )

        # Check for data source
        if 'sample_file' not in data and 'sample_description' not in data:
            errors.append("Must provide either 'sample_file' or 'sample_description'")

        # Validate parameters
        params = data.get('parameters', {})

        if technique == 'temperature_sweep':
            if 'temp_range' in params:
                temp_range = params['temp_range']
                if not isinstance(temp_range, list) or len(temp_range) != 2:
                    errors.append("temp_range must be [min_temp, max_temp]")
                elif temp_range[0] >= temp_range[1]:
                    errors.append("temp_range: min must be < max")
                elif temp_range[0] < 100 or temp_range[1] > 600:
                    warnings.append(f"Temperature range {temp_range} K outside typical DMA range (100-600 K)")

            if 'heating_rate' in params:
                rate = params['heating_rate']
                if rate <= 0 or rate > 20:
                    warnings.append(f"Heating rate {rate} K/min is unusual (typical: 1-10 K/min)")

        if technique == 'frequency_sweep':
            if 'freq_range' in params:
                freq_range = params['freq_range']
                if not isinstance(freq_range, list) or len(freq_range) != 2:
                    errors.append("freq_range must be [min_freq, max_freq]")
                elif freq_range[0] >= freq_range[1]:
                    errors.append("freq_range: min must be < max")

        if technique == 'dynamic_strain_sweep':
            if 'strain_range' in params:
                strain_range = params['strain_range']
                if not isinstance(strain_range, list) or len(strain_range) != 2:
                    errors.append("strain_range must be [min_strain, max_strain]")
                elif strain_range[0] >= strain_range[1]:
                    errors.append("strain_range: min must be < max")

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
        technique = data.get('technique', 'temperature_sweep')

        # Resource requirements by technique
        resource_map = {
            'temperature_sweep': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=1200,  # 20 minutes (full T-sweep)
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'frequency_sweep': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=900,  # 15 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'isothermal': ResourceRequirement(
                cpu_cores=2,
                memory_gb=0.8,
                gpu_count=0,
                estimated_time_sec=600,  # 10 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'multi_frequency': ResourceRequirement(
                cpu_cores=4,
                memory_gb=2.0,
                gpu_count=0,
                estimated_time_sec=1800,  # 30 minutes (multiple frequencies)
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'creep_recovery': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=1500,  # 25 minutes
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'dynamic_strain_sweep': ResourceRequirement(
                cpu_cores=2,
                memory_gb=0.8,
                gpu_count=0,
                estimated_time_sec=600,  # 10 minutes
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
                name="temperature_sweep",
                description="Temperature sweep for glass transition (Tg) determination",
                input_types=["temp_range", "heating_rate", "frequency"],
                output_types=["storage_modulus_E_prime", "loss_modulus_E_double_prime", "tan_delta", "Tg"],
                typical_use_cases=[
                    "Glass transition temperature (Tg)",
                    "β-relaxation identification",
                    "Polymer characterization",
                    "Temperature-dependent stiffness"
                ]
            ),
            Capability(
                name="frequency_sweep",
                description="Frequency sweep for master curve construction",
                input_types=["freq_range", "temperature"],
                output_types=["storage_modulus_E_prime", "loss_modulus_E_double_prime", "master_curve"],
                typical_use_cases=[
                    "Time-temperature superposition (TTS)",
                    "Master curve construction",
                    "WLF equation fitting",
                    "Broadband viscoelastic spectrum"
                ]
            ),
            Capability(
                name="isothermal",
                description="Isothermal stress relaxation and creep",
                input_types=["temperature", "duration"],
                output_types=["relaxation_modulus", "creep_compliance", "relaxation_time"],
                typical_use_cases=[
                    "Stress relaxation",
                    "Creep compliance",
                    "Long-term stability",
                    "Viscoelastic relaxation times"
                ]
            ),
            Capability(
                name="multi_frequency",
                description="Multi-frequency DMA for broadband characterization",
                input_types=["frequency_list", "temperature"],
                output_types=["E_prime_multi", "E_double_prime_multi", "tan_delta_multi"],
                typical_use_cases=[
                    "Broadband viscoelastic properties",
                    "Multiple relaxation processes",
                    "Frequency-dependent damping",
                    "Complex modulus mapping"
                ]
            ),
            Capability(
                name="creep_recovery",
                description="Creep and recovery for elastic/viscous separation",
                input_types=["stress_amplitude", "creep_duration", "recovery_duration"],
                output_types=["creep_compliance", "recovery_compliance", "viscous_component"],
                typical_use_cases=[
                    "Elastic vs. viscous response",
                    "Permanent deformation",
                    "Recovery behavior",
                    "Viscoelastic modeling"
                ]
            ),
            Capability(
                name="dynamic_strain_sweep",
                description="Strain amplitude sweep for LVE determination",
                input_types=["strain_range", "frequency", "temperature"],
                output_types=["E_prime_vs_strain", "E_double_prime_vs_strain", "LVE_limit"],
                typical_use_cases=[
                    "Linear viscoelastic (LVE) regime",
                    "Yield strain determination",
                    "Amplitude-dependent behavior",
                    "Nonlinear viscoelasticity"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata
        """
        return AgentMetadata(
            name=self.NAME,
            version=self.VERSION,
            description="Dynamic Mechanical Analysis (DMA) characterization expert",
            author="Materials Science Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dat', 'txt', 'csv', 'xlsx', 'dma']
        )

    # Technique implementations

    def _execute_temperature_sweep(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DMA temperature sweep.

        Measures E', E'', tan δ vs. temperature for glass transition determination

        Args:
            input_data: Input with parameters (temp_range, heating_rate, frequency)

        Returns:
            DMA temperature sweep results
        """
        params = input_data.get('parameters', {})
        temp_range = params.get('temp_range', [200, 400])  # K
        heating_rate = params.get('heating_rate', 3.0)  # K/min
        frequency = params.get('frequency', 1.0)  # Hz
        strain_amplitude = params.get('strain_amplitude', 0.1)  # % (in LVE)

        # Simulate polymer DMA (glass transition)
        temperatures = np.linspace(temp_range[0], temp_range[1], 100)

        # Glass transition parameters
        Tg = params.get('Tg_expected', 300)  # K
        E_glassy = 3e9  # Pa (glassy state, ~3 GPa)
        E_rubbery = 1e6  # Pa (rubbery state, ~1 MPa)

        # Simulate glass transition using sigmoidal function
        # Storage modulus E' drops from glassy to rubbery plateau
        transition_width = 20  # K
        E_prime = E_rubbery + (E_glassy - E_rubbery) / (1 + np.exp((temperatures - Tg) / transition_width))

        # Loss modulus E'' peaks at Tg (maximum energy dissipation)
        E_double_prime = 0.15 * E_glassy * np.exp(-((temperatures - Tg)**2) / (2 * transition_width**2))

        # tan δ = E''/E' (peak slightly above Tg)
        tan_delta = E_double_prime / E_prime

        # Identify Tg from different criteria
        Tg_peak_E_double_prime = temperatures[np.argmax(E_double_prime)]
        Tg_peak_tan_delta = temperatures[np.argmax(tan_delta)]
        Tg_onset = temperatures[np.argmax(np.gradient(E_prime))]  # Onset of E' drop

        result = {
            'technique': 'temperature_sweep',
            'temperature_K': temperatures.tolist(),
            'storage_modulus_E_prime_Pa': E_prime.tolist(),
            'loss_modulus_E_double_prime_Pa': E_double_prime.tolist(),
            'tan_delta': tan_delta.tolist(),
            'glass_transition_Tg_K': {
                'E_double_prime_peak': float(Tg_peak_E_double_prime),
                'tan_delta_peak': float(Tg_peak_tan_delta),
                'onset': float(Tg_onset),
                'midpoint': float(Tg)
            },
            'E_glassy_Pa': float(E_glassy),
            'E_rubbery_Pa': float(E_rubbery),
            'frequency_Hz': frequency,
            'heating_rate_K_per_min': heating_rate,
            'strain_amplitude_percent': strain_amplitude,
            'measurements': {
                'number_of_points': len(temperatures),
                'temperature_resolution_K': float(np.mean(np.diff(temperatures)))
            }
        }

        return result

    def _execute_frequency_sweep(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DMA frequency sweep.

        Measures E', E'' vs. frequency for master curve construction

        Args:
            input_data: Input with parameters (freq_range, temperature)

        Returns:
            DMA frequency sweep results
        """
        params = input_data.get('parameters', {})
        freq_range = params.get('freq_range', [0.01, 100])  # Hz
        temperature = params.get('temperature', 298)  # K
        strain_amplitude = params.get('strain_amplitude', 0.1)  # %

        # Simulate frequency-dependent moduli
        frequencies = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 30)

        # Simulate polymer viscoelastic behavior (power-law)
        # E' increases with frequency (material becomes stiffer)
        # E'' also increases but with different scaling
        E_0 = 1e6  # Pa (low-frequency modulus)

        E_prime = E_0 * (1 + (frequencies / 1.0) ** 0.7)  # Viscoelastic solid
        E_double_prime = E_0 * 0.3 * (frequencies ** 0.5)  # Loss increases with freq

        tan_delta = E_double_prime / E_prime

        result = {
            'technique': 'frequency_sweep',
            'frequency_Hz': frequencies.tolist(),
            'storage_modulus_E_prime_Pa': E_prime.tolist(),
            'loss_modulus_E_double_prime_Pa': E_double_prime.tolist(),
            'tan_delta': tan_delta.tolist(),
            'temperature_K': temperature,
            'strain_amplitude_percent': strain_amplitude,
            'measurements': {
                'number_of_points': len(frequencies),
                'frequency_decade_coverage': float(np.log10(freq_range[1] / freq_range[0]))
            },
            'analysis': {
                'low_freq_E_prime_Pa': float(E_prime[0]),
                'high_freq_E_prime_Pa': float(E_prime[-1]),
                'frequency_dependence': 'solid-like (E\' > E\'\')' if E_prime[-1] > E_double_prime[-1] else 'liquid-like'
            }
        }

        return result

    def _execute_isothermal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute isothermal DMA (stress relaxation or creep).

        Measures time-dependent modulus or compliance

        Args:
            input_data: Input with parameters (temperature, duration, test_type)

        Returns:
            Isothermal DMA results
        """
        params = input_data.get('parameters', {})
        temperature = params.get('temperature', 298)  # K
        duration = params.get('duration', 1000)  # seconds
        test_type = params.get('test_type', 'relaxation')  # 'relaxation' or 'creep'

        time = np.logspace(-2, np.log10(duration), 100)

        if test_type == 'relaxation':
            # Stress relaxation: E(t) = E_∞ + (E_0 - E_∞) exp(-t/τ)
            E_0 = 1e9  # Pa (instantaneous modulus)
            E_inf = 1e6  # Pa (equilibrium modulus)
            tau = 100  # seconds (relaxation time)

            E_t = E_inf + (E_0 - E_inf) * np.exp(-time / tau)

            result = {
                'technique': 'isothermal',
                'test_type': 'stress_relaxation',
                'time_s': time.tolist(),
                'relaxation_modulus_E_t_Pa': E_t.tolist(),
                'instantaneous_modulus_E0_Pa': float(E_0),
                'equilibrium_modulus_Einf_Pa': float(E_inf),
                'relaxation_time_tau_s': float(tau),
                'temperature_K': temperature
            }
        else:  # creep
            # Creep compliance: J(t) = J_0 + (J_max - J_0)(1 - exp(-t/τ))
            J_0 = 1e-9  # Pa^-1 (instantaneous compliance)
            J_max = 1e-6  # Pa^-1 (maximum compliance)
            tau = 100  # seconds

            J_t = J_0 + (J_max - J_0) * (1 - np.exp(-time / tau))

            result = {
                'technique': 'isothermal',
                'test_type': 'creep',
                'time_s': time.tolist(),
                'creep_compliance_J_t_Pa_inv': J_t.tolist(),
                'instantaneous_compliance_J0_Pa_inv': float(J_0),
                'equilibrium_compliance_Jmax_Pa_inv': float(J_max),
                'retardation_time_tau_s': float(tau),
                'temperature_K': temperature
            }

        return result

    def _execute_multi_frequency(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-frequency DMA.

        Measures E', E'' at multiple frequencies simultaneously

        Args:
            input_data: Input with parameters (frequency_list, temperature)

        Returns:
            Multi-frequency DMA results
        """
        params = input_data.get('parameters', {})
        frequency_list = params.get('frequency_list', [0.1, 1.0, 10.0, 100.0])  # Hz
        temperature = params.get('temperature', 298)  # K
        strain_amplitude = params.get('strain_amplitude', 0.1)  # %

        frequencies = np.array(frequency_list)

        # Simulate viscoelastic response at each frequency
        E_0 = 1e6  # Pa
        E_prime = E_0 * (1 + (frequencies / 1.0) ** 0.7)
        E_double_prime = E_0 * 0.3 * (frequencies ** 0.5)
        tan_delta = E_double_prime / E_prime

        result = {
            'technique': 'multi_frequency',
            'frequency_Hz': frequencies.tolist(),
            'storage_modulus_E_prime_Pa': E_prime.tolist(),
            'loss_modulus_E_double_prime_Pa': E_double_prime.tolist(),
            'tan_delta': tan_delta.tolist(),
            'temperature_K': temperature,
            'strain_amplitude_percent': strain_amplitude,
            'measurements': {
                'number_of_frequencies': len(frequencies)
            }
        }

        return result

    def _execute_stress_controlled(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stress-controlled DMA.

        Args:
            input_data: Input with parameters (stress_amplitude, frequency, temperature)

        Returns:
            Stress-controlled DMA results
        """
        params = input_data.get('parameters', {})
        stress_amplitude = params.get('stress_amplitude', 1e5)  # Pa
        frequency = params.get('frequency', 1.0)  # Hz
        temperature = params.get('temperature', 298)  # K

        # Simulate response
        time = np.linspace(0, 10, 200)
        omega = 2 * np.pi * frequency

        # Applied stress (sinusoidal)
        stress = stress_amplitude * np.sin(omega * time)

        # Resulting strain (with phase lag δ)
        E_star = 1e9  # Complex modulus magnitude
        delta = 0.1  # Phase angle (radians)
        strain = (stress_amplitude / E_star) * np.sin(omega * time - delta)

        E_prime = E_star * np.cos(delta)
        E_double_prime = E_star * np.sin(delta)

        result = {
            'technique': 'stress_controlled',
            'time_s': time.tolist(),
            'stress_Pa': stress.tolist(),
            'strain': strain.tolist(),
            'storage_modulus_E_prime_Pa': float(E_prime),
            'loss_modulus_E_double_prime_Pa': float(E_double_prime),
            'phase_angle_deg': float(np.degrees(delta)),
            'stress_amplitude_Pa': stress_amplitude,
            'frequency_Hz': frequency,
            'temperature_K': temperature
        }

        return result

    def _execute_strain_controlled(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strain-controlled DMA.

        Args:
            input_data: Input with parameters (strain_amplitude, frequency, temperature)

        Returns:
            Strain-controlled DMA results
        """
        params = input_data.get('parameters', {})
        strain_amplitude = params.get('strain_amplitude', 0.001)  # Dimensionless
        frequency = params.get('frequency', 1.0)  # Hz
        temperature = params.get('temperature', 298)  # K

        # Simulate response
        time = np.linspace(0, 10, 200)
        omega = 2 * np.pi * frequency

        # Applied strain (sinusoidal)
        strain = strain_amplitude * np.sin(omega * time)

        # Resulting stress (with phase lag δ)
        E_star = 1e9  # Complex modulus
        delta = 0.1  # Phase angle
        stress = E_star * strain_amplitude * np.sin(omega * time + delta)

        E_prime = E_star * np.cos(delta)
        E_double_prime = E_star * np.sin(delta)

        result = {
            'technique': 'strain_controlled',
            'time_s': time.tolist(),
            'strain': strain.tolist(),
            'stress_Pa': stress.tolist(),
            'storage_modulus_E_prime_Pa': float(E_prime),
            'loss_modulus_E_double_prime_Pa': float(E_double_prime),
            'phase_angle_deg': float(np.degrees(delta)),
            'strain_amplitude': strain_amplitude,
            'frequency_Hz': frequency,
            'temperature_K': temperature
        }

        return result

    def _execute_creep_recovery(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute creep-recovery test.

        Args:
            input_data: Input with parameters (stress, creep_duration, recovery_duration)

        Returns:
            Creep-recovery results
        """
        params = input_data.get('parameters', {})
        stress_amplitude = params.get('stress_amplitude', 1e5)  # Pa
        creep_duration = params.get('creep_duration', 600)  # seconds
        recovery_duration = params.get('recovery_duration', 600)  # seconds
        temperature = params.get('temperature', 298)  # K

        # Creep phase
        time_creep = np.linspace(0, creep_duration, 100)
        J_0 = 1e-9  # Instantaneous compliance
        J_e = 5e-9  # Elastic compliance
        J_v = 2e-9  # Viscous compliance
        tau = 100  # Retardation time

        J_creep = J_0 + J_e * (1 - np.exp(-time_creep / tau)) + J_v * (time_creep / tau)
        strain_creep = stress_amplitude * J_creep

        # Recovery phase (stress = 0)
        time_recovery = np.linspace(0, recovery_duration, 100)
        strain_recovery = stress_amplitude * (J_e * np.exp(-time_recovery / tau) + J_v * (creep_duration / tau))

        result = {
            'technique': 'creep_recovery',
            'creep_phase': {
                'time_s': time_creep.tolist(),
                'strain': strain_creep.tolist(),
                'compliance_Pa_inv': J_creep.tolist()
            },
            'recovery_phase': {
                'time_s': time_recovery.tolist(),
                'strain': strain_recovery.tolist()
            },
            'analysis': {
                'instantaneous_compliance_J0_Pa_inv': float(J_0),
                'elastic_compliance_Je_Pa_inv': float(J_e),
                'viscous_compliance_Jv_Pa_inv': float(J_v),
                'retardation_time_tau_s': float(tau),
                'permanent_strain': float(strain_recovery[-1])
            },
            'stress_amplitude_Pa': stress_amplitude,
            'temperature_K': temperature
        }

        return result

    def _execute_dynamic_strain_sweep(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic strain amplitude sweep.

        Determines linear viscoelastic (LVE) limit

        Args:
            input_data: Input with parameters (strain_range, frequency, temperature)

        Returns:
            Strain sweep results with LVE limit
        """
        params = input_data.get('parameters', {})
        strain_range = params.get('strain_range', [0.01, 10.0])  # % strain
        frequency = params.get('frequency', 1.0)  # Hz
        temperature = params.get('temperature', 298)  # K

        # Simulate strain-dependent moduli
        strain_percent = np.logspace(np.log10(strain_range[0]), np.log10(strain_range[1]), 30)

        # LVE region: E' constant
        # Non-linear region: E' decreases (strain softening)
        LVE_limit = 1.0  # % strain
        E_prime_LVE = 1e9  # Pa

        E_prime = np.where(
            strain_percent <= LVE_limit,
            E_prime_LVE,
            E_prime_LVE * (LVE_limit / strain_percent) ** 0.5  # Strain softening
        )

        E_double_prime = 0.1 * E_prime  # Loss modulus
        tan_delta = E_double_prime / E_prime

        result = {
            'technique': 'dynamic_strain_sweep',
            'strain_percent': strain_percent.tolist(),
            'storage_modulus_E_prime_Pa': E_prime.tolist(),
            'loss_modulus_E_double_prime_Pa': E_double_prime.tolist(),
            'tan_delta': tan_delta.tolist(),
            'LVE_limit_strain_percent': float(LVE_limit),
            'E_prime_LVE_Pa': float(E_prime_LVE),
            'frequency_Hz': frequency,
            'temperature_K': temperature,
            'analysis': {
                'linear_viscoelastic_range': f'< {LVE_limit}% strain',
                'nonlinear_behavior': 'strain softening'
            }
        }

        return result

    # Cross-validation methods

    @staticmethod
    def cross_validate_with_dsc(dma_result: Dict[str, Any],
                                 dsc_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate DMA glass transition with DSC.

        Args:
            dma_result: DMA temperature sweep result
            dsc_result: DSC result with Tg

        Returns:
            Cross-validation results
        """
        if dma_result.get('technique') != 'temperature_sweep':
            return {'error': 'DMA result must be temperature_sweep'}

        # Extract Tg values
        dma_tg = dma_result.get('glass_transition_Tg_K', {}).get('midpoint', 0)
        dsc_tg = dsc_result.get('glass_transition_Tg_K', 0)

        if dma_tg > 0 and dsc_tg > 0:
            delta_tg = abs(dma_tg - dsc_tg)
            agreement = delta_tg < 10  # Within 10 K is good agreement

            return {
                'DMA_Tg_K': dma_tg,
                'DSC_Tg_K': dsc_tg,
                'delta_Tg_K': delta_tg,
                'agreement': 'excellent' if delta_tg < 5 else 'good' if delta_tg < 10 else 'poor',
                'notes': f"DMA Tg often ~5-10 K higher than DSC (frequency-dependent). Δ = {delta_tg:.1f} K"
            }
        else:
            return {'error': 'Tg data missing or invalid'}

    @staticmethod
    def cross_validate_with_bds(dma_result: Dict[str, Any],
                                 bds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate DMA α-relaxation with BDS.

        Args:
            dma_result: DMA temperature sweep result
            bds_result: BDS temperature sweep result

        Returns:
            Cross-validation results
        """
        if dma_result.get('technique') != 'temperature_sweep':
            return {'error': 'DMA result must be temperature_sweep'}

        # Extract α-relaxation temperatures
        dma_tg = dma_result.get('glass_transition_Tg_K', {}).get('tan_delta_peak', 0)

        # BDS α-relaxation from loss peak
        bds_temps = bds_result.get('temperature_K', [])
        bds_loss = bds_result.get('epsilon_double_prime', [])

        if dma_tg > 0 and len(bds_temps) > 0 and len(bds_loss) > 0:
            bds_alpha_temp = bds_temps[np.argmax(bds_loss)]
            delta_T = abs(dma_tg - bds_alpha_temp)

            return {
                'DMA_Tg_tan_delta_peak_K': dma_tg,
                'BDS_alpha_relaxation_K': bds_alpha_temp,
                'delta_T_K': delta_T,
                'agreement': 'excellent' if delta_T < 5 else 'good' if delta_T < 15 else 'poor',
                'notes': f"α-relaxation temperatures should be similar. Δ = {delta_T:.1f} K"
            }
        else:
            return {'error': 'Relaxation data missing or invalid'}

    @staticmethod
    def cross_validate_with_rheology(dma_result: Dict[str, Any],
                                      rheology_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate DMA modulus with oscillatory rheology.

        DMA measures E' (tensile), rheology measures G' (shear)
        For isotropic materials: E ≈ 3G (for Poisson's ratio ν ≈ 0.5)

        Args:
            dma_result: DMA result with E'
            rheology_result: Oscillatory rheology result with G'

        Returns:
            Cross-validation results
        """
        # Extract moduli at similar conditions
        dma_E_prime = None
        if dma_result.get('technique') == 'temperature_sweep':
            E_prime_list = dma_result.get('storage_modulus_E_prime_Pa', [])
            if E_prime_list:
                dma_E_prime = E_prime_list[0]  # Low temperature (glassy)
        elif dma_result.get('technique') == 'frequency_sweep':
            E_prime_list = dma_result.get('storage_modulus_E_prime_Pa', [])
            if E_prime_list:
                dma_E_prime = E_prime_list[-1]  # High frequency

        # Extract G' from rheology
        G_prime_list = rheology_result.get('storage_modulus_G_prime_Pa', [])
        rheo_G_prime = G_prime_list[0] if G_prime_list else None

        if dma_E_prime and rheo_G_prime:
            # E ≈ 3G for incompressible materials (ν = 0.5)
            # E ≈ 2.6G for typical polymers (ν = 0.3-0.4)
            ratio = dma_E_prime / rheo_G_prime
            expected_ratio = 2.6  # Typical for polymers

            agreement = 2.0 < ratio < 3.5  # Reasonable range

            return {
                'DMA_E_prime_Pa': dma_E_prime,
                'Rheology_G_prime_Pa': rheo_G_prime,
                'E_to_G_ratio': ratio,
                'expected_ratio': expected_ratio,
                'agreement': 'good' if agreement else 'poor',
                'notes': f"E/G ratio = {ratio:.2f} (expected ~2.6 for typical polymers)"
            }
        else:
            return {'error': 'Modulus data missing or invalid'}
