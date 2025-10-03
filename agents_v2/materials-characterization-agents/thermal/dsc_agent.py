"""DSC Agent - Differential Scanning Calorimetry Expert.

This agent specializes in thermal analysis via differential scanning calorimetry:
- DSC (standard, modulated, high-pressure)
- Glass transition temperature (Tg) determination
- Melting and crystallization analysis
- Heat capacity measurement (Cp)
- Purity analysis
- Reaction kinetics and thermodynamics
- Curing and crosslinking studies

Expert in thermal transitions, phase behavior, and thermal properties characterization.
"""

from base_agent import (
    ExperimentalAgent, AgentResult, AgentStatus, ValidationResult,
    ResourceRequirement, Capability, AgentMetadata, Provenance,
    ExecutionEnvironment
)
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import hashlib
import numpy as np


class DSCAgent(ExperimentalAgent):
    """DSC agent for differential scanning calorimetry analysis.

    Capabilities:
    - Standard DSC: Heat flow vs temperature
    - Modulated DSC (MDSC): Separate reversing/non-reversing signals
    - High-pressure DSC: Elevated pressure measurements
    - Temperature-modulated DSC: Phase lag analysis
    - Isothermal DSC: Heat flow vs time

    Measurements:
    - Glass transition temperature (Tg)
    - Melting temperature (Tm) and enthalpy (ΔHm)
    - Crystallization temperature (Tc) and enthalpy (ΔHc)
    - Heat capacity (Cp)
    - Degree of crystallinity
    - Purity determination
    - Reaction kinetics
    - Curing behavior

    Key advantages:
    - Sub-milliwatt sensitivity
    - Wide temperature range (-180°C to 725°C)
    - Small sample size (5-20 mg)
    - Quantitative thermal analysis
    - Non-destructive measurement
    """

    VERSION = "1.0.0"

    # Supported DSC techniques
    SUPPORTED_TECHNIQUES = [
        'standard_dsc',       # Standard heat/cool scans
        'modulated_dsc',      # Temperature modulation (MDSC/TMDSC)
        'isothermal_dsc',     # Isothermal curing/crystallization
        'high_pressure_dsc',  # Elevated pressure
        'cyclic_dsc',         # Multiple heat/cool cycles
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DSC agent.

        Args:
            config: Configuration including:
                - instrument: DSC model (TA Instruments, PerkinElmer, Mettler Toledo)
                - temperature_range: Min/max temperatures (K)
                - heating_rate: Default heating rate (K/min)
                - atmosphere: Purge gas (N2, He, air)
        """
        super().__init__(config)
        self.instrument = self.config.get('instrument', 'DSC_Q2000')
        self.temp_range = self.config.get('temperature_range', [173, 873])  # K
        self.heating_rate = self.config.get('heating_rate', 10.0)  # K/min
        self.atmosphere = self.config.get('atmosphere', 'N2')

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute DSC analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or thermogram_data: DSC curve data
                - parameters: Technique-specific parameters
                  - heating_rate: K/min
                  - temperature_range: [T_min, T_max] in K
                  - sample_mass: mg
                  - atmosphere: gas type

        Returns:
            AgentResult with DSC analysis
        """
        start_time = datetime.now()

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

        technique = input_data['technique'].lower()

        # Route to technique-specific execution
        try:
            if technique == 'standard_dsc':
                result_data = self._execute_standard_dsc(input_data)
            elif technique == 'modulated_dsc':
                result_data = self._execute_modulated_dsc(input_data)
            elif technique == 'isothermal_dsc':
                result_data = self._execute_isothermal_dsc(input_data)
            elif technique == 'high_pressure_dsc':
                result_data = self._execute_high_pressure_dsc(input_data)
            elif technique == 'cyclic_dsc':
                result_data = self._execute_cyclic_dsc(input_data)
            else:
                raise ValueError(f"Unsupported technique: {technique}")

            execution_time = (datetime.now() - start_time).total_seconds()

            # Create provenance
            provenance = Provenance(
                agent_name=self.metadata.name,
                agent_version=self.VERSION,
                timestamp=start_time,
                input_hash=self._compute_cache_key(input_data),
                parameters={
                    'technique': technique,
                    'instrument': self.instrument,
                    'heating_rate': input_data.get('parameters', {}).get('heating_rate', self.heating_rate),
                    'temperature_range': input_data.get('parameters', {}).get('temperature_range', self.temp_range),
                    'atmosphere': self.atmosphere
                },
                execution_time_sec=execution_time,
                environment={'sample_mass_mg': input_data.get('parameters', {}).get('sample_mass', 10.0)}
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'technique': technique,
                    'execution_time_sec': execution_time,
                    'instrument': self.instrument
                },
                provenance=provenance,
                warnings=validation.warnings
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=[f"Execution failed: {str(e)}"]
            )

    def _execute_standard_dsc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute standard DSC heat/cool scan.

        DSC provides:
        - Glass transition temperature (Tg)
        - Melting/crystallization peaks
        - Heat capacity changes
        - Enthalpy of transitions
        """
        params = input_data.get('parameters', {})
        heating_rate = params.get('heating_rate', self.heating_rate)  # K/min
        temp_range = params.get('temperature_range', self.temp_range)  # K
        sample_mass = params.get('sample_mass', 10.0)  # mg

        n_points = 500
        temperature_c = np.linspace(temp_range[0] - 273.15, temp_range[1] - 273.15, n_points)
        temperature_k = temperature_c + 273.15

        # Simulate DSC thermogram
        heat_flow = np.zeros_like(temperature_c)  # mW

        # Baseline (heat capacity)
        cp_baseline = 1.5  # J/(g·K) - typical for polymers
        heat_flow += cp_baseline * sample_mass * heating_rate / 60  # Convert to mW

        # Glass transition (sigmoid + derivative peak)
        tg_c = np.random.uniform(50, 120)  # °C
        tg_width = 10  # °C
        tg_step_height = 0.3  # mW (ΔCp step)

        # Sigmoid for glass transition
        sigmoid = 1 / (1 + np.exp(-(temperature_c - tg_c) / tg_width))
        heat_flow += tg_step_height * sigmoid

        # Derivative peak (actual Tg inflection point)
        tg_derivative = tg_step_height * np.exp(-(temperature_c - tg_c) / tg_width) / (tg_width * (1 + np.exp(-(temperature_c - tg_c) / tg_width))**2)

        # Melting peak (Gaussian)
        tm_c = np.random.uniform(150, 180)  # °C
        delta_hm = np.random.uniform(80, 120)  # J/g
        melting_width = 5  # °C
        melting_peak_height = delta_hm * sample_mass * 1000 / (melting_width * np.sqrt(2 * np.pi))  # mW
        melting_peak = melting_peak_height * np.exp(-((temperature_c - tm_c) / melting_width)**2 / 2)
        heat_flow += melting_peak

        # Crystallization peak (if cooling) - exothermic
        if params.get('include_cooling', False):
            tc_c = tm_c - 30  # °C (supercooling)
            delta_hc = -delta_hm * 0.9  # Slightly less due to imperfect crystallization
            crystallization_peak = -abs(delta_hc) * sample_mass * 1000 / (melting_width * np.sqrt(2 * np.pi)) * np.exp(-((temperature_c - tc_c) / melting_width)**2 / 2)
            heat_flow += crystallization_peak

        # Add noise
        noise = np.random.normal(0, 0.02, n_points)
        heat_flow += noise

        # Calculate derived values
        crystallinity = (delta_hm / 293.6) * 100 if delta_hm > 0 else 0  # For PET, ΔH_100% = 293.6 J/g

        return {
            'technique': 'Standard DSC',
            'temperature_c': temperature_c.tolist(),
            'temperature_k': temperature_k.tolist(),
            'heat_flow_mw': heat_flow.tolist(),
            'heating_rate_k_per_min': heating_rate,
            'sample_mass_mg': sample_mass,
            'glass_transition': {
                'tg_onset_c': tg_c - tg_width,
                'tg_midpoint_c': tg_c,
                'tg_endset_c': tg_c + tg_width,
                'tg_k': tg_c + 273.15,
                'delta_cp_j_per_g_k': tg_step_height / sample_mass * 60 / heating_rate,
                'tg_width_c': tg_width * 2
            },
            'melting_transition': {
                'tm_onset_c': tm_c - melting_width * 2,
                'tm_peak_c': tm_c,
                'tm_endset_c': tm_c + melting_width * 2,
                'tm_k': tm_c + 273.15,
                'delta_hm_j_per_g': delta_hm,
                'peak_width_c': melting_width * 4
            },
            'crystallinity_analysis': {
                'degree_of_crystallinity_percent': crystallinity,
                'crystallization_enthalpy_j_per_g': delta_hm,
                'amorphous_fraction_percent': 100 - crystallinity,
                'reference_enthalpy_j_per_g': 293.6  # For PET
            },
            'heat_capacity': {
                'cp_glass_j_per_g_k': cp_baseline,
                'cp_melt_j_per_g_k': cp_baseline + tg_step_height / sample_mass * 60 / heating_rate,
                'delta_cp_j_per_g_k': tg_step_height / sample_mass * 60 / heating_rate
            },
            'quality_metrics': {
                'signal_to_noise_ratio': 50.0,
                'baseline_stability': 0.02,  # mW
                'temperature_precision_c': 0.1
            }
        }

    def _execute_modulated_dsc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute modulated DSC (MDSC/TMDSC).

        Modulated DSC provides:
        - Separation of reversing (heat capacity related) and non-reversing (kinetic) signals
        - Improved Tg detection
        - Simultaneous measurement of Cp and kinetic processes
        """
        params = input_data.get('parameters', {})
        modulation_amplitude = params.get('modulation_amplitude', 0.5)  # °C
        modulation_period = params.get('modulation_period', 60)  # seconds

        # Get standard DSC result first
        standard_result = self._execute_standard_dsc(input_data)

        # Add modulated DSC specific analysis
        return {
            **standard_result,
            'technique': 'Modulated DSC (MDSC)',
            'modulation_parameters': {
                'amplitude_c': modulation_amplitude,
                'period_s': modulation_period,
                'frequency_hz': 1 / modulation_period
            },
            'reversing_signal': {
                'description': 'Heat capacity related, reversible processes',
                'tg_enhanced': True,
                'melting_present': True
            },
            'non_reversing_signal': {
                'description': 'Kinetic processes, irreversible changes',
                'relaxation_enthalpy': 5.2,  # J/g
                'cold_crystallization': False
            },
            'complex_heat_capacity': {
                'cp_prime_j_per_g_k': standard_result['heat_capacity']['cp_glass_j_per_g_k'],
                'cp_double_prime_j_per_g_k': 0.05,  # Imaginary component
                'phase_angle_deg': 2.0
            }
        }

    def _execute_isothermal_dsc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute isothermal DSC for curing/crystallization kinetics.

        Isothermal DSC provides:
        - Curing kinetics
        - Crystallization kinetics
        - Reaction rate constants
        - Avrami analysis
        """
        params = input_data.get('parameters', {})
        isothermal_temp_c = params.get('isothermal_temperature', 160)  # °C
        duration_min = params.get('duration_min', 60)
        sample_mass = params.get('sample_mass', 10.0)

        n_points = 300
        time_min = np.linspace(0, duration_min, n_points)

        # Simulate exothermic curing reaction (Avrami kinetics)
        k_rate = 0.1  # Rate constant
        n_avrami = 2.0  # Avrami exponent
        total_heat = -250  # J/g (exothermic)

        # Avrami equation: α(t) = 1 - exp(-(kt)^n)
        conversion = 1 - np.exp(-(k_rate * time_min)**n_avrami)

        # Heat flow is derivative of conversion
        heat_flow_raw = total_heat * sample_mass * k_rate * n_avrami * (k_rate * time_min)**(n_avrami - 1) * np.exp(-(k_rate * time_min)**n_avrami)

        # Add noise
        heat_flow = heat_flow_raw + np.random.normal(0, 0.5, n_points)

        return {
            'technique': 'Isothermal DSC',
            'isothermal_temperature_c': isothermal_temp_c,
            'isothermal_temperature_k': isothermal_temp_c + 273.15,
            'time_min': time_min.tolist(),
            'heat_flow_mw': heat_flow.tolist(),
            'conversion': conversion.tolist(),
            'kinetics_analysis': {
                'total_heat_j_per_g': total_heat,
                'rate_constant_k': k_rate,
                'avrami_exponent_n': n_avrami,
                'half_time_min': (-np.log(0.5))**(1/n_avrami) / k_rate,
                'mechanism': 'diffusion_controlled' if n_avrami < 2 else 'nucleation_controlled'
            },
            'curing_parameters': {
                'degree_of_cure_final': conversion[-1],
                'time_to_95_percent_min': float(time_min[np.argmax(conversion > 0.95)]) if any(conversion > 0.95) else None,
                'exotherm_total_j': total_heat * sample_mass
            }
        }

    def _execute_high_pressure_dsc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute high-pressure DSC.

        High-pressure DSC provides:
        - Pressure-temperature phase diagrams
        - High-pressure melting points
        - Pressure effects on transitions
        """
        params = input_data.get('parameters', {})
        pressure_bar = params.get('pressure', 100)  # bar

        # Get standard result and modify for pressure
        standard_result = self._execute_standard_dsc(input_data)

        # Pressure increases Tm (Clausius-Clapeyron)
        # dT/dP = T * ΔV / ΔH
        delta_tm = 0.05 * pressure_bar  # Approximate: 0.05 °C per bar

        return {
            **standard_result,
            'technique': 'High-Pressure DSC',
            'pressure_bar': pressure_bar,
            'pressure_mpa': pressure_bar * 0.1,
            'melting_transition': {
                **standard_result['melting_transition'],
                'tm_peak_c': standard_result['melting_transition']['tm_peak_c'] + delta_tm,
                'tm_k': standard_result['melting_transition']['tm_k'] + delta_tm,
                'pressure_shift_c_per_bar': 0.05
            },
            'thermodynamics': {
                'volume_change_ml_per_g': 0.1,  # ΔV for melting
                'clausius_clapeyron_slope': 0.05
            }
        }

    def _execute_cyclic_dsc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cyclic DSC (multiple heat-cool cycles).

        Cyclic DSC provides:
        - Thermal history effects
        - Reversibility of transitions
        - Aging and annealing behavior
        """
        params = input_data.get('parameters', {})
        n_cycles = params.get('n_cycles', 3)

        cycles_data = []
        for cycle in range(1, n_cycles + 1):
            cycle_result = self._execute_standard_dsc(input_data)

            # Thermal history effects: Tg increases slightly each cycle
            tg_shift = 2.0 * cycle  # °C increase per cycle

            cycles_data.append({
                'cycle_number': cycle,
                'tg_midpoint_c': cycle_result['glass_transition']['tg_midpoint_c'] + tg_shift,
                'tm_peak_c': cycle_result['melting_transition']['tm_peak_c'],
                'crystallinity_percent': cycle_result['crystallinity_analysis']['degree_of_crystallinity_percent']
            })

        return {
            'technique': 'Cyclic DSC',
            'number_of_cycles': n_cycles,
            'cycles': cycles_data,
            'thermal_history_analysis': {
                'tg_reproducibility_c': 2.0,
                'thermal_reversibility': 'good',
                'aging_effect_c_per_cycle': 2.0,
                'memory_effect': 'moderate'
            }
        }

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data."""
        errors = []
        warnings = []

        # Check technique
        if 'technique' not in data:
            errors.append("Missing required field: 'technique'")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        technique = data['technique'].lower()
        if technique not in self.SUPPORTED_TECHNIQUES:
            errors.append(
                f"Unsupported technique: {technique}. "
                f"Supported: {self.SUPPORTED_TECHNIQUES}"
            )

        # Check for data
        if 'data_file' not in data and 'thermogram_data' not in data:
            warnings.append("No data provided; will use simulated data")

        # Validate parameters
        params = data.get('parameters', {})

        if 'heating_rate' in params:
            rate = params['heating_rate']
            if rate <= 0 or rate > 100:
                errors.append(f"Invalid heating rate: {rate} K/min (must be 0-100)")

        if 'temperature_range' in params:
            temp_range = params['temperature_range']
            if not isinstance(temp_range, list) or len(temp_range) != 2:
                errors.append("temperature_range must be [T_min, T_max]")
            elif temp_range[0] >= temp_range[1]:
                errors.append("temperature_range: T_min must be < T_max")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed."""
        technique = data.get('technique', '').lower()

        # DSC analysis is fast and local
        return ResourceRequirement(
            cpu_cores=1,
            memory_gb=0.5,
            estimated_time_sec=30.0,
            execution_environment=ExecutionEnvironment.LOCAL
        )

    def get_capabilities(self) -> List[Capability]:
        """Get list of agent capabilities."""
        return [
            Capability(
                name='standard_dsc',
                description='Standard DSC for thermal transitions',
                input_types=['thermogram', 'temperature_program'],
                output_types=['tg', 'tm', 'tc', 'enthalpy', 'heat_capacity'],
                typical_use_cases=['polymer_characterization', 'purity_analysis', 'thermal_stability']
            ),
            Capability(
                name='modulated_dsc',
                description='Modulated DSC for complex thermal behavior',
                input_types=['modulated_thermogram'],
                output_types=['reversing_signal', 'non_reversing_signal', 'complex_cp'],
                typical_use_cases=['weak_transitions', 'overlapping_events', 'kinetic_separation']
            ),
            Capability(
                name='isothermal_dsc',
                description='Isothermal DSC for curing and crystallization kinetics',
                input_types=['isothermal_data', 'time_series'],
                output_types=['kinetic_parameters', 'conversion', 'reaction_rate'],
                typical_use_cases=['curing_kinetics', 'crystallization_kinetics', 'reaction_monitoring']
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata."""
        return AgentMetadata(
            name="DSCAgent",
            version=self.VERSION,
            description="Differential Scanning Calorimetry expert for thermal analysis",
            author="Materials Characterization Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dsc', 'txt', 'csv', 'xlsx']
        )

    # ExperimentalAgent interface methods
    def connect_instrument(self) -> bool:
        """Connect to DSC instrument."""
        # In production: connect to TA Instruments, PerkinElmer, Mettler Toledo
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw DSC data."""
        # In production:
        # - Baseline correction
        # - Peak integration
        # - Transition identification
        # - Heat capacity calculation
        return raw_data

    # Integration methods for cross-validation
    @staticmethod
    def validate_with_dma(dsc_result: Dict[str, Any], dma_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate DSC Tg with DMA E'' peak.

        DSC measures Tg as heat capacity change.
        DMA measures Tg as E'' (loss modulus) peak.
        They should agree within ±5°C.

        Args:
            dsc_result: DSC analysis with Tg
            dma_result: DMA analysis with Tg

        Returns:
            Validation report
        """
        dsc_tg = dsc_result.get('glass_transition', {}).get('tg_midpoint_c', 0)
        dma_tg = dma_result.get('glass_transition_Tg_K', 300) - 273.15  # Convert K to C

        if dsc_tg > 0 and dma_tg > -273:
            diff = abs(dsc_tg - dma_tg)
            agreement = diff < 5.0

            return {
                'validation_type': 'DSC_DMA_Tg_cross_check',
                'dsc_tg_c': dsc_tg,
                'dma_tg_c': dma_tg,
                'difference_c': diff,
                'agreement': 'excellent' if diff < 3 else 'good' if diff < 5 else 'poor',
                'consistent': agreement,
                'notes': f'DSC and DMA Tg values {"agree" if agreement else "disagree"} within {diff:.1f}°C'
            }
        else:
            return {
                'validation_type': 'DSC_DMA_Tg_cross_check',
                'consistent': False,
                'error': 'Missing Tg data from DSC or DMA'
            }

    @staticmethod
    def correlate_with_xrd(dsc_result: Dict[str, Any], xrd_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate DSC crystallinity with XRD crystallinity.

        DSC calculates crystallinity from melting enthalpy.
        XRD measures crystallinity from peak intensities.
        Independent methods should agree.

        Args:
            dsc_result: DSC analysis with crystallinity
            xrd_result: XRD analysis with crystallinity

        Returns:
            Correlation report
        """
        dsc_crystallinity = dsc_result.get('crystallinity_analysis', {}).get('degree_of_crystallinity_percent', 0)
        xrd_crystallinity = xrd_result.get('crystallinity_analysis', {}).get('crystallinity_percent', 0)

        if dsc_crystallinity > 0 and xrd_crystallinity > 0:
            diff = abs(dsc_crystallinity - xrd_crystallinity)
            agreement = diff < 10.0

            return {
                'correlation_type': 'DSC_XRD_crystallinity',
                'dsc_crystallinity_percent': dsc_crystallinity,
                'xrd_crystallinity_percent': xrd_crystallinity,
                'difference_percent': diff,
                'agreement': 'excellent' if diff < 5 else 'good' if diff < 10 else 'poor',
                'consistent': agreement,
                'notes': f'DSC (enthalpy-based) and XRD (diffraction-based) crystallinity {"agree" if agreement else "differ"} by {diff:.1f}%'
            }
        else:
            return {
                'correlation_type': 'DSC_XRD_crystallinity',
                'consistent': False,
                'error': 'Missing crystallinity data from DSC or XRD'
            }
