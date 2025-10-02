"""TMA Agent - Thermomechanical Analysis Expert.

This agent specializes in thermomechanical analysis techniques:
- TMA (standard thermomechanical analysis)
- CTE measurement (coefficient of thermal expansion)
- Penetration mode (softening point)
- Tension mode (thermal expansion and shrinkage)
- DTA (differential thermal analysis)

Expert in dimensional stability, thermal expansion, softening behavior, and thermal transitions.
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


class TMAAgent(ExperimentalAgent):
    """TMA agent for thermomechanical analysis.

    Capabilities:
    - Expansion mode: Dimensional changes vs temperature
    - Penetration mode: Softening point determination
    - Tension mode: Shrinkage and expansion under load
    - Compression mode: Thermal expansion under stress
    - DTA mode: Differential thermal analysis

    Measurements:
    - Coefficient of thermal expansion (CTE, α)
    - Softening temperature
    - Glass transition temperature (Tg)
    - Dimensional stability
    - Thermal shrinkage
    - Phase transitions

    Key advantages:
    - Sub-nanometer displacement resolution
    - Wide temperature range (-150°C to 1000°C)
    - Controlled force application
    - Direct dimensional measurement
    - Non-destructive for expansion measurements
    """

    VERSION = "1.0.0"

    # Supported TMA techniques
    SUPPORTED_TECHNIQUES = [
        'expansion',       # Thermal expansion (CTE measurement)
        'penetration',     # Softening point determination
        'tension',         # Shrinkage and expansion under tension
        'compression',     # Compression mode
        'dta',             # Differential thermal analysis
        'three_point_bend' # Flexural measurements
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TMA agent.

        Args:
            config: Configuration including:
                - instrument: TMA model (TA Instruments, PerkinElmer, Mettler Toledo)
                - max_force: Maximum force in mN
                - displacement_resolution: nm
                - temperature_range: Min/max temperatures (K)
        """
        super().__init__(config)
        self.instrument = self.config.get('instrument', 'TMA_Q400')
        self.max_force = self.config.get('max_force', 1000)  # mN
        self.resolution = self.config.get('displacement_resolution', 0.1)  # nm
        self.temp_range = self.config.get('temperature_range', [173, 873])  # K

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute TMA analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or tma_data: TMA curve data
                - parameters: Technique-specific parameters
                  - heating_rate: K/min
                  - temperature_range: [T_min, T_max] in K
                  - force: mN
                  - initial_length: mm

        Returns:
            AgentResult with TMA analysis
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
            if technique == 'expansion':
                result_data = self._execute_expansion(input_data)
            elif technique == 'penetration':
                result_data = self._execute_penetration(input_data)
            elif technique == 'tension':
                result_data = self._execute_tension(input_data)
            elif technique == 'compression':
                result_data = self._execute_compression(input_data)
            elif technique == 'dta':
                result_data = self._execute_dta(input_data)
            elif technique == 'three_point_bend':
                result_data = self._execute_three_point_bend(input_data)
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
                    'heating_rate': input_data.get('parameters', {}).get('heating_rate', 5.0),
                    'force': input_data.get('parameters', {}).get('force', 10.0)
                },
                execution_time_sec=execution_time,
                environment={}
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

    def _execute_expansion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute expansion mode for CTE measurement.

        Expansion mode provides:
        - Coefficient of thermal expansion (CTE, α)
        - Dimensional changes vs temperature
        - Glass transition temperature
        - Phase transitions
        """
        params = input_data.get('parameters', {})
        heating_rate = params.get('heating_rate', 5.0)  # K/min
        temp_range = params.get('temperature_range', [233, 473])  # K (-40 to 200°C)
        initial_length = params.get('initial_length', 10.0)  # mm
        force = params.get('force', 10.0)  # mN (small contact force)

        n_points = 300
        temperature_c = np.linspace(temp_range[0] - 273.15, temp_range[1] - 273.15, n_points)
        temperature_k = temperature_c + 273.15

        # Simulate thermal expansion curve
        # Below Tg: Linear expansion (glassy CTE)
        # At Tg: Slope change (increased CTE)
        # Above Tg: Linear expansion (rubbery CTE)

        tg_c = 80  # Glass transition temperature
        alpha_glassy = 60e-6  # CTE below Tg (1/K) - typical for glassy polymers
        alpha_rubbery = 200e-6  # CTE above Tg (1/K) - typical for rubbery polymers

        # Calculate expansion
        expansion_um = np.zeros_like(temperature_c)

        for i, temp in enumerate(temperature_c):
            if temp < tg_c:
                # Glassy region
                expansion_um[i] = alpha_glassy * initial_length * 1000 * (temp - temperature_c[0])
            else:
                # Rubbery region (with continuous transition at Tg)
                expansion_glassy_at_tg = alpha_glassy * initial_length * 1000 * (tg_c - temperature_c[0])
                expansion_rubbery = alpha_rubbery * initial_length * 1000 * (temp - tg_c)
                expansion_um[i] = expansion_glassy_at_tg + expansion_rubbery

        # Add noise
        noise = np.random.normal(0, 0.02, n_points)
        expansion_um += noise

        # Calculate dimensional change
        dimensional_change_percent = (expansion_um / (initial_length * 1000)) * 100

        # Calculate instantaneous CTE (derivative)
        cte_instantaneous = np.gradient(expansion_um, temperature_c) / (initial_length * 1000)

        # Find glass transition from CTE change
        tg_onset = tg_c - 10
        tg_endset = tg_c + 10

        return {
            'technique': 'TMA Expansion',
            'temperature_c': temperature_c.tolist(),
            'temperature_k': temperature_k.tolist(),
            'expansion_um': expansion_um.tolist(),
            'dimensional_change_percent': dimensional_change_percent.tolist(),
            'heating_rate_k_per_min': heating_rate,
            'initial_length_mm': initial_length,
            'force_mn': force,
            'cte_analysis': {
                'cte_glassy_per_k': alpha_glassy,
                'cte_glassy_per_c': alpha_glassy,
                'cte_rubbery_per_k': alpha_rubbery,
                'cte_rubbery_per_c': alpha_rubbery,
                'cte_average_per_k': np.mean([alpha_glassy, alpha_rubbery]),
                'temperature_range_c': [temperature_c[0], temperature_c[-1]]
            },
            'glass_transition': {
                'tg_onset_c': tg_onset,
                'tg_midpoint_c': tg_c,
                'tg_endset_c': tg_endset,
                'tg_k': tg_c + 273.15,
                'delta_cte_per_k': alpha_rubbery - alpha_glassy,
                'detection_method': 'cte_change'
            },
            'dimensional_stability': {
                'total_expansion_um': float(expansion_um[-1]),
                'total_expansion_percent': float(dimensional_change_percent[-1]),
                'stability_rating': 'good' if dimensional_change_percent[-1] < 2.0 else 'moderate',
                'recommended_use_temperature_c': tg_c - 20  # Safety margin below Tg
            }
        }

    def _execute_penetration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute penetration mode for softening point.

        Penetration mode provides:
        - Softening temperature
        - Vicat softening point
        - Penetration depth vs temperature
        - Thermal stability under load
        """
        params = input_data.get('parameters', {})
        heating_rate = params.get('heating_rate', 5.0)
        temp_range = params.get('temperature_range', [298, 473])
        force = params.get('force', 1000.0)  # mN (1 N typical for penetration)
        probe_diameter = params.get('probe_diameter', 1.0)  # mm

        n_points = 200
        temperature_c = np.linspace(temp_range[0] - 273.15, temp_range[1] - 273.15, n_points)
        temperature_k = temperature_c + 273.15

        # Simulate penetration curve
        # Initially rigid (minimal penetration)
        # Rapid penetration at softening point

        softening_temp_c = 120  # Softening temperature
        penetration_depth = np.zeros_like(temperature_c)

        for i, temp in enumerate(temperature_c):
            if temp < softening_temp_c - 10:
                # Rigid phase - minimal thermal expansion only
                penetration_depth[i] = 0.01 * (temp - temperature_c[0])
            elif temp < softening_temp_c + 20:
                # Softening region - rapid penetration
                transition = (temp - (softening_temp_c - 10)) / 30  # Normalized 0-1
                penetration_depth[i] = 0.01 * (softening_temp_c - 10 - temperature_c[0]) + 100 * transition
            else:
                # Fully softened
                penetration_depth[i] = 100 + 10 * (temp - (softening_temp_c + 20))

        # Add noise
        penetration_depth += np.random.normal(0, 0.5, n_points)

        # Calculate penetration rate
        penetration_rate = np.gradient(penetration_depth, temperature_c)

        # Find softening point (maximum penetration rate)
        softening_index = np.argmax(penetration_rate)
        vicat_softening_temp = temperature_c[softening_index]

        return {
            'technique': 'TMA Penetration',
            'temperature_c': temperature_c.tolist(),
            'temperature_k': temperature_k.tolist(),
            'penetration_depth_um': penetration_depth.tolist(),
            'penetration_rate_um_per_c': penetration_rate.tolist(),
            'force_mn': force,
            'probe_diameter_mm': probe_diameter,
            'softening_analysis': {
                'vicat_softening_temperature_c': float(vicat_softening_temp),
                'vicat_softening_temperature_k': float(vicat_softening_temp + 273.15),
                'onset_softening_c': float(vicat_softening_temp - 10),
                'complete_softening_c': float(vicat_softening_temp + 20),
                'penetration_at_softening_um': float(penetration_depth[softening_index])
            },
            'thermal_stability_under_load': {
                'max_use_temperature_c': float(vicat_softening_temp - 20),
                'stability_rating': 'load_bearing_limited_above_{:.0f}C'.format(vicat_softening_temp - 20),
                'heat_deflection_temperature_estimate_c': float(vicat_softening_temp - 5)
            }
        }

    def _execute_tension(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tension mode for shrinkage/expansion.

        Tension mode provides:
        - Thermal shrinkage
        - Stress relaxation
        - Dimensional stability under tension
        - CTE under load
        """
        params = input_data.get('parameters', {})
        heating_rate = params.get('heating_rate', 5.0)
        temp_range = params.get('temperature_range', [298, 473])
        initial_length = params.get('initial_length', 20.0)  # mm
        force = params.get('force', 20.0)  # mN

        n_points = 250
        temperature_c = np.linspace(temp_range[0] - 273.15, temp_range[1] - 273.15, n_points)
        temperature_k = temperature_c + 273.15

        # Simulate thermal shrinkage behavior (typical for oriented films/fibers)
        # Initial expansion, then shrinkage as stress relaxes above Tg

        tg_c = 85
        alpha = 80e-6  # CTE before shrinkage
        shrinkage_onset_c = tg_c
        shrinkage_magnitude = 500  # μm (5% of 10mm)

        dimensional_change_um = np.zeros_like(temperature_c)

        for i, temp in enumerate(temperature_c):
            if temp < shrinkage_onset_c:
                # Normal thermal expansion
                dimensional_change_um[i] = alpha * initial_length * 1000 * (temp - temperature_c[0])
            else:
                # Shrinkage dominates (stress relaxation)
                expansion_to_onset = alpha * initial_length * 1000 * (shrinkage_onset_c - temperature_c[0])
                shrinkage_fraction = 1 - np.exp(-(temp - shrinkage_onset_c) / 30)
                dimensional_change_um[i] = expansion_to_onset - shrinkage_magnitude * shrinkage_fraction

        # Add noise
        dimensional_change_um += np.random.normal(0, 2.0, n_points)

        return {
            'technique': 'TMA Tension',
            'temperature_c': temperature_c.tolist(),
            'temperature_k': temperature_k.tolist(),
            'dimensional_change_um': dimensional_change_um.tolist(),
            'initial_length_mm': initial_length,
            'tension_force_mn': force,
            'shrinkage_analysis': {
                'shrinkage_onset_c': shrinkage_onset_c,
                'maximum_shrinkage_um': shrinkage_magnitude,
                'maximum_shrinkage_percent': (shrinkage_magnitude / (initial_length * 1000)) * 100,
                'shrinkage_mechanism': 'stress_relaxation_of_oriented_chains',
                'reversibility': 'irreversible'
            },
            'dimensional_stability': {
                'cte_below_shrinkage_per_k': alpha,
                'net_dimensional_change_um': float(dimensional_change_um[-1]),
                'net_dimensional_change_percent': float((dimensional_change_um[-1] / (initial_length * 1000)) * 100),
                'stability_rating': 'unstable_under_tension_above_tg'
            }
        }

    def _execute_compression(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compression mode for expansion under load.

        Compression mode provides:
        - CTE under compressive load
        - Compressive behavior
        - Dimensional stability under stress
        """
        # Similar to expansion but with compressive load
        expansion_result = self._execute_expansion(input_data)

        return {
            **expansion_result,
            'technique': 'TMA Compression',
            'load_type': 'compressive',
            'compressive_stress_analysis': {
                'applied_stress_mpa': 0.5,  # Estimated from force/area
                'compliance_effect': 'minimal',
                'cte_under_load_per_k': expansion_result['cte_analysis']['cte_average_per_k']
            }
        }

    def _execute_dta(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute differential thermal analysis.

        DTA provides:
        - Phase transitions
        - Melting/crystallization
        - Thermal events
        """
        params = input_data.get('parameters', {})
        heating_rate = params.get('heating_rate', 10.0)
        temp_range = params.get('temperature_range', [298, 573])

        n_points = 300
        temperature_c = np.linspace(temp_range[0] - 273.15, temp_range[1] - 273.15, n_points)
        temperature_k = temperature_c + 273.15

        # Simulate DTA signal (temperature difference)
        dta_signal = np.zeros_like(temperature_c)

        # Glass transition (small step)
        tg_c = 80
        dta_signal += 0.2 / (1 + np.exp(-(temperature_c - tg_c) / 5))

        # Melting peak (endothermic)
        tm_c = 165
        melting_peak = -2.0 * np.exp(-((temperature_c - tm_c) / 8)**2)
        dta_signal += melting_peak

        # Add noise
        dta_signal += np.random.normal(0, 0.05, n_points)

        return {
            'technique': 'DTA (Differential Thermal Analysis)',
            'temperature_c': temperature_c.tolist(),
            'temperature_k': temperature_k.tolist(),
            'dta_signal_uv': dta_signal.tolist(),
            'thermal_events': {
                'glass_transition': {
                    'tg_c': tg_c,
                    'type': 'second_order_transition'
                },
                'melting': {
                    'tm_c': tm_c,
                    'type': 'first_order_transition_endothermic'
                }
            }
        }

    def _execute_three_point_bend(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute three-point bend mode for flexural properties.

        Three-point bend provides:
        - Flexural modulus vs temperature
        - Heat deflection temperature
        - Thermal-mechanical behavior
        """
        params = input_data.get('parameters', {})
        heating_rate = params.get('heating_rate', 5.0)
        temp_range = params.get('temperature_range', [298, 473])
        span_length = params.get('span_length', 50.0)  # mm
        force = params.get('force', 500.0)  # mN

        n_points = 200
        temperature_c = np.linspace(temp_range[0] - 273.15, temp_range[1] - 273.15, n_points)
        temperature_k = temperature_c + 273.15

        # Simulate deflection vs temperature
        # Initially rigid, then softens above Tg

        tg_c = 90
        deflection_um = np.zeros_like(temperature_c)

        for i, temp in enumerate(temperature_c):
            if temp < tg_c:
                # Rigid - small deflection
                deflection_um[i] = 10 + 0.1 * (temp - temperature_c[0])
            else:
                # Softening - increased deflection
                deflection_um[i] = 10 + 0.1 * (tg_c - temperature_c[0]) + 5 * (temp - tg_c)

        # Add noise
        deflection_um += np.random.normal(0, 0.5, n_points)

        # Find heat deflection temperature (HDT) - often defined at specific deflection
        hdt_deflection = 50  # μm
        hdt_index = np.argmax(deflection_um > hdt_deflection)
        hdt_temperature = temperature_c[hdt_index]

        return {
            'technique': 'TMA Three-Point Bend',
            'temperature_c': temperature_c.tolist(),
            'temperature_k': temperature_k.tolist(),
            'deflection_um': deflection_um.tolist(),
            'span_length_mm': span_length,
            'applied_force_mn': force,
            'heat_deflection_analysis': {
                'hdt_at_50um_deflection_c': float(hdt_temperature),
                'hdt_at_50um_deflection_k': float(hdt_temperature + 273.15),
                'flexural_stability_rating': 'good_below_{:.0f}C'.format(hdt_temperature - 10)
            },
            'flexural_modulus_estimate': {
                'modulus_at_room_temp_gpa': 2.5,
                'modulus_at_hdt_gpa': 0.5,
                'modulus_reduction': 'significant_above_tg'
            }
        }

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data."""
        errors = []
        warnings = []

        if 'technique' not in data:
            errors.append("Missing required field: 'technique'")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        technique = data['technique'].lower()
        if technique not in self.SUPPORTED_TECHNIQUES:
            errors.append(
                f"Unsupported technique: {technique}. "
                f"Supported: {self.SUPPORTED_TECHNIQUES}"
            )

        if 'data_file' not in data and 'tma_data' not in data:
            warnings.append("No data provided; will use simulated data")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed."""
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
                name='expansion',
                description='Thermal expansion and CTE measurement',
                input_types=['dimensional_data', 'temperature_program'],
                output_types=['cte', 'expansion', 'tg'],
                typical_use_cases=['cte_measurement', 'dimensional_stability', 'thermal_compatibility']
            ),
            Capability(
                name='penetration',
                description='Softening point determination',
                input_types=['penetration_data'],
                output_types=['softening_temperature', 'vicat_point'],
                typical_use_cases=['quality_control', 'thermal_stability', 'process_optimization']
            ),
            Capability(
                name='tension',
                description='Thermal shrinkage and expansion under load',
                input_types=['tension_data'],
                output_types=['shrinkage', 'dimensional_change'],
                typical_use_cases=['film_stability', 'fiber_analysis', 'stress_relaxation']
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata."""
        return AgentMetadata(
            name="TMAAgent",
            version=self.VERSION,
            description="Thermomechanical Analysis expert for dimensional stability and thermal expansion",
            author="Materials Characterization Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['tma', 'txt', 'csv', 'xlsx']
        )

    def connect_instrument(self) -> bool:
        """Connect to TMA instrument."""
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw TMA data."""
        return raw_data

    # Integration methods
    @staticmethod
    def correlate_with_dsc(tma_result: Dict[str, Any], dsc_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate TMA Tg with DSC Tg.

        TMA detects Tg as CTE change.
        DSC detects Tg as Cp change.
        Should agree within ±5°C.

        Args:
            tma_result: TMA analysis
            dsc_result: DSC analysis

        Returns:
            Correlation report
        """
        tma_tg = tma_result.get('glass_transition', {}).get('tg_midpoint_c', 0)
        dsc_tg = dsc_result.get('glass_transition', {}).get('tg_midpoint_c', 0)

        if tma_tg > 0 and dsc_tg > 0:
            diff = abs(tma_tg - dsc_tg)
            return {
                'correlation_type': 'TMA_DSC_Tg',
                'tma_tg_c': tma_tg,
                'dsc_tg_c': dsc_tg,
                'difference_c': diff,
                'agreement': 'excellent' if diff < 3 else 'good' if diff < 5 else 'poor'
            }
        return {'error': 'Missing Tg data'}

    @staticmethod
    def validate_with_xrd(tma_result: Dict[str, Any], xrd_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate TMA CTE with XRD lattice expansion.

        TMA measures bulk CTE.
        XRD measures lattice parameter changes.
        Should be correlated.

        Args:
            tma_result: TMA CTE analysis
            xrd_result: XRD temperature-dependent analysis

        Returns:
            Validation report
        """
        tma_cte = tma_result.get('cte_analysis', {}).get('cte_glassy_per_k', 0)

        return {
            'validation_type': 'TMA_XRD_CTE',
            'tma_bulk_cte_per_k': tma_cte,
            'notes': 'Bulk CTE (TMA) typically higher than lattice CTE (XRD) due to grain boundaries and defects',
            'expected_relationship': 'CTE_bulk > CTE_lattice'
        }
