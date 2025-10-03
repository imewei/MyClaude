"""TGA Agent - Thermogravimetric Analysis Expert.

This agent specializes in thermogravimetric analysis techniques:
- TGA (standard thermogravimetric analysis)
- TGA-FTIR (evolved gas analysis via infrared)
- TGA-MS (evolved gas analysis via mass spectrometry)
- High-resolution TGA (Hi-Res TGA)
- Isothermal TGA (constant temperature weight loss)
- Decomposition kinetics analysis

Expert in thermal degradation, composition analysis, and thermal stability characterization.
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


class TGAAgent(ExperimentalAgent):
    """TGA agent for thermogravimetric analysis.

    Capabilities:
    - Standard TGA: Weight loss vs temperature
    - TGA-FTIR: Coupled with FTIR for gas identification
    - TGA-MS: Coupled with mass spec for evolved gas analysis
    - Hi-Res TGA: High-resolution dynamic heating
    - Isothermal TGA: Weight loss at constant temperature
    - Derivative TGA (DTG): Rate of weight loss

    Measurements:
    - Decomposition temperatures (T_onset, T_peak, T_endset)
    - Mass loss percentages
    - Residue (ash content)
    - Thermal stability
    - Composition analysis
    - Degradation kinetics
    - Activation energies

    Key advantages:
    - Microgram sensitivity (0.1 μg)
    - Wide temperature range (RT to 1600°C)
    - Small sample size (5-20 mg)
    - Quantitative composition analysis
    - Kinetic parameter extraction
    """

    VERSION = "1.0.0"

    # Supported TGA techniques
    SUPPORTED_TECHNIQUES = [
        'standard_tga',      # Standard temperature ramp
        'isothermal_tga',    # Constant temperature
        'hi_res_tga',        # High-resolution dynamic heating
        'tga_ftir',          # Coupled with FTIR
        'tga_ms',            # Coupled with mass spectrometry
        'multi_ramp_tga',    # Multiple heating rates
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TGA agent.

        Args:
            config: Configuration including:
                - instrument: TGA model (TA Instruments, PerkinElmer, Mettler Toledo)
                - max_temperature: Maximum temperature (K)
                - balance_sensitivity: μg
                - atmosphere: Purge gas (N2, air, O2, He)
        """
        super().__init__(config)
        self.instrument = self.config.get('instrument', 'TGA_Q500')
        self.max_temp = self.config.get('max_temperature', 1273)  # K (1000°C)
        self.sensitivity = self.config.get('balance_sensitivity', 0.1)  # μg
        self.atmosphere = self.config.get('atmosphere', 'N2')

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute TGA analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or tga_data: TGA curve data
                - parameters: Technique-specific parameters
                  - heating_rate: K/min
                  - temperature_range: [T_min, T_max] in K
                  - sample_mass: mg
                  - atmosphere: gas type

        Returns:
            AgentResult with TGA analysis
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
            if technique == 'standard_tga':
                result_data = self._execute_standard_tga(input_data)
            elif technique == 'isothermal_tga':
                result_data = self._execute_isothermal_tga(input_data)
            elif technique == 'hi_res_tga':
                result_data = self._execute_hi_res_tga(input_data)
            elif technique == 'tga_ftir':
                result_data = self._execute_tga_ftir(input_data)
            elif technique == 'tga_ms':
                result_data = self._execute_tga_ms(input_data)
            elif technique == 'multi_ramp_tga':
                result_data = self._execute_multi_ramp_tga(input_data)
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
                    'heating_rate': input_data.get('parameters', {}).get('heating_rate', 10.0),
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

    def _execute_standard_tga(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute standard TGA temperature ramp.

        TGA provides:
        - Weight loss as function of temperature
        - Decomposition temperatures
        - Residue/ash content
        - Multi-step degradation analysis
        """
        params = input_data.get('parameters', {})
        heating_rate = params.get('heating_rate', 10.0)  # K/min
        temp_range = params.get('temperature_range', [298, 1073])  # K (25-800°C)
        sample_mass = params.get('sample_mass', 10.0)  # mg
        atmosphere = params.get('atmosphere', self.atmosphere)

        n_points = 500
        temperature_c = np.linspace(temp_range[0] - 273.15, temp_range[1] - 273.15, n_points)
        temperature_k = temperature_c + 273.15

        # Simulate TGA curve with multiple decomposition steps
        weight_percent = np.ones(n_points) * 100.0

        # Example: Polymer with multiple degradation steps
        # Step 1: Moisture loss (~100°C)
        t1_c = 100
        width1 = 20
        loss1 = 2.0  # 2% moisture
        weight_percent -= loss1 / (1 + np.exp(-(temperature_c - t1_c) / width1))

        # Step 2: Side chain degradation (~250°C)
        t2_c = 250
        width2 = 30
        loss2 = 15.0  # 15% side chains
        weight_percent -= loss2 / (1 + np.exp(-(temperature_c - t2_c) / width2))

        # Step 3: Main chain decomposition (~400°C)
        t3_c = 400
        width3 = 40
        loss3 = 60.0  # 60% main chain
        weight_percent -= loss3 / (1 + np.exp(-(temperature_c - t3_c) / width3))

        # Final residue: 23%
        residue_percent = 100 - loss1 - loss2 - loss3

        # Calculate derivative (DTG)
        dtg = -np.gradient(weight_percent, temperature_c)  # %/°C

        # Add noise
        noise = np.random.normal(0, 0.1, n_points)
        weight_percent += noise
        dtg += np.random.normal(0, 0.01, n_points)

        # Identify decomposition steps
        decomposition_steps = [
            {
                'step_number': 1,
                'description': 'Moisture evolution',
                'onset_temperature_c': t1_c - width1 * 2,
                'peak_temperature_c': t1_c,
                'endset_temperature_c': t1_c + width1 * 2,
                'weight_loss_percent': loss1,
                'temperature_range_c': [t1_c - width1 * 2, t1_c + width1 * 2]
            },
            {
                'step_number': 2,
                'description': 'Side chain degradation',
                'onset_temperature_c': t2_c - width2 * 2,
                'peak_temperature_c': t2_c,
                'endset_temperature_c': t2_c + width2 * 2,
                'weight_loss_percent': loss2,
                'temperature_range_c': [t2_c - width2 * 2, t2_c + width2 * 2]
            },
            {
                'step_number': 3,
                'description': 'Main chain decomposition',
                'onset_temperature_c': t3_c - width3 * 2,
                'peak_temperature_c': t3_c,
                'endset_temperature_c': t3_c + width3 * 2,
                'weight_loss_percent': loss3,
                'temperature_range_c': [t3_c - width3 * 2, t3_c + width3 * 2]
            }
        ]

        return {
            'technique': 'Standard TGA',
            'temperature_c': temperature_c.tolist(),
            'temperature_k': temperature_k.tolist(),
            'weight_percent': weight_percent.tolist(),
            'weight_mg': (weight_percent / 100 * sample_mass).tolist(),
            'derivative_dtg_percent_per_c': dtg.tolist(),
            'heating_rate_k_per_min': heating_rate,
            'sample_mass_mg': sample_mass,
            'atmosphere': atmosphere,
            'decomposition_analysis': {
                'total_weight_loss_percent': loss1 + loss2 + loss3,
                'number_of_steps': 3,
                'decomposition_steps': decomposition_steps
            },
            'residue_analysis': {
                'final_residue_percent': residue_percent,
                'residue_mass_mg': residue_percent / 100 * sample_mass,
                'residue_composition': 'inorganic_ash_or_char'
            },
            'thermal_stability': {
                'onset_decomposition_c': t1_c - width1 * 2,
                'temperature_at_5_percent_loss': float(temperature_c[np.argmax(weight_percent < 95)]),
                'temperature_at_50_percent_loss': float(temperature_c[np.argmax(weight_percent < 50)]),
                'max_decomposition_rate_c': float(temperature_c[np.argmax(dtg)])
            },
            'composition_estimate': {
                'volatile_content_percent': loss1,
                'organic_content_percent': loss2 + loss3,
                'inorganic_residue_percent': residue_percent
            }
        }

    def _execute_isothermal_tga(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute isothermal TGA.

        Isothermal TGA provides:
        - Weight loss kinetics at constant temperature
        - Degradation rate constants
        - Shelf life predictions
        """
        params = input_data.get('parameters', {})
        isothermal_temp_c = params.get('isothermal_temperature', 300)
        duration_min = params.get('duration_min', 120)
        sample_mass = params.get('sample_mass', 10.0)

        n_points = 240
        time_min = np.linspace(0, duration_min, n_points)

        # First-order degradation kinetics
        k_degradation = 0.01  # Rate constant (1/min)
        initial_weight = 100.0
        weight_percent = initial_weight * np.exp(-k_degradation * time_min)

        # Calculate half-life
        t_half = np.log(2) / k_degradation

        # Add noise
        weight_percent += np.random.normal(0, 0.2, n_points)

        return {
            'technique': 'Isothermal TGA',
            'isothermal_temperature_c': isothermal_temp_c,
            'isothermal_temperature_k': isothermal_temp_c + 273.15,
            'time_min': time_min.tolist(),
            'weight_percent': weight_percent.tolist(),
            'weight_mg': (weight_percent / 100 * sample_mass).tolist(),
            'kinetics_analysis': {
                'degradation_order': 1,
                'rate_constant_per_min': k_degradation,
                'half_life_min': t_half,
                'activation_energy_kj_per_mol': 150.0,  # Typical for polymers
                'degradation_mechanism': 'random_chain_scission'
            },
            'stability_prediction': {
                'time_to_10_percent_loss_min': -np.log(0.9) / k_degradation,
                'time_to_50_percent_loss_min': t_half,
                'long_term_stability': 'moderate'
            }
        }

    def _execute_hi_res_tga(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute high-resolution TGA.

        Hi-Res TGA provides:
        - Enhanced separation of decomposition steps
        - Better resolution of overlapping events
        - Dynamic heating rate adjustment
        """
        # Get standard TGA result
        standard_result = self._execute_standard_tga(input_data)

        return {
            **standard_result,
            'technique': 'Hi-Res TGA',
            'resolution_enhancement': {
                'dynamic_heating_rate': True,
                'heating_rate_range': [0.1, 20.0],  # K/min
                'resolution_factor': 5,
                'separation_improvement': 'excellent'
            },
            'advantages': [
                'Better separation of overlapping decompositions',
                'Sharper DTG peaks',
                'Improved quantification',
                'Reduced peak overlap'
            ]
        }

    def _execute_tga_ftir(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TGA-FTIR (evolved gas analysis).

        TGA-FTIR provides:
        - Identification of evolved gases
        - Correlation of weight loss with gas evolution
        - Degradation mechanism elucidation
        """
        # Get standard TGA result
        standard_result = self._execute_standard_tga(input_data)

        # Simulate evolved gas identification
        evolved_gases = {
            'step_1': {
                'temperature_range_c': [80, 120],
                'primary_gas': 'H2O (water)',
                'ftir_peaks': [3600, 1600],  # cm^-1
                'mechanism': 'moisture_evaporation'
            },
            'step_2': {
                'temperature_range_c': [220, 280],
                'primary_gas': 'HCl (hydrogen chloride)',
                'secondary_gas': 'CO2 (carbon dioxide)',
                'ftir_peaks': [2886, 2349],  # cm^-1
                'mechanism': 'dehydrochlorination'
            },
            'step_3': {
                'temperature_range_c': [360, 440],
                'primary_gas': 'CO2 (carbon dioxide)',
                'secondary_gas': 'CO (carbon monoxide)',
                'tertiary_gas': 'hydrocarbons',
                'ftir_peaks': [2349, 2143, 2900],  # cm^-1
                'mechanism': 'main_chain_scission'
            }
        }

        return {
            **standard_result,
            'technique': 'TGA-FTIR',
            'evolved_gas_analysis': evolved_gases,
            'degradation_mechanism': {
                'step_1': 'Physical desorption of moisture',
                'step_2': 'Chemical degradation of side groups (HCl elimination)',
                'step_3': 'Main chain thermal cracking producing CO2, CO, hydrocarbons'
            },
            'ftir_correlation': {
                'time_resolved_ftir': True,
                'gas_identification_confidence': 'high',
                'transfer_line_temperature_c': 250
            }
        }

    def _execute_tga_ms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TGA-MS (mass spectrometry coupling).

        TGA-MS provides:
        - Molecular weight of evolved species
        - Fragmentation patterns
        - Quantitative gas analysis
        """
        # Get standard TGA result
        standard_result = self._execute_standard_tga(input_data)

        # Simulate mass spec data
        ms_data = {
            'step_1': {
                'temperature_range_c': [80, 120],
                'ion_fragments': {
                    'm/z_18': {'species': 'H2O+', 'intensity': 'very_high'},
                    'm/z_17': {'species': 'OH+', 'intensity': 'medium'}
                },
                'primary_species': 'Water (H2O)'
            },
            'step_2': {
                'temperature_range_c': [220, 280],
                'ion_fragments': {
                    'm/z_36': {'species': 'HCl+', 'intensity': 'high'},
                    'm/z_44': {'species': 'CO2+', 'intensity': 'medium'}
                },
                'primary_species': 'Hydrogen chloride (HCl)'
            },
            'step_3': {
                'temperature_range_c': [360, 440],
                'ion_fragments': {
                    'm/z_44': {'species': 'CO2+', 'intensity': 'very_high'},
                    'm/z_28': {'species': 'CO+', 'intensity': 'high'},
                    'm/z_15-50': {'species': 'hydrocarbon fragments', 'intensity': 'medium'}
                },
                'primary_species': 'Carbon dioxide (CO2) and hydrocarbons'
            }
        }

        return {
            **standard_result,
            'technique': 'TGA-MS',
            'mass_spec_analysis': ms_data,
            'evolved_gas_quantification': {
                'h2o_evolution_mg': 0.2,
                'hcl_evolution_mg': 1.5,
                'co2_evolution_mg': 4.8,
                'total_gas_evolved_mg': 6.5
            },
            'ionization_method': 'electron_impact_70eV',
            'transfer_line_temperature_c': 280
        }

    def _execute_multi_ramp_tga(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-ramp TGA for kinetics.

        Multi-ramp TGA provides:
        - Activation energy via Kissinger/Ozawa methods
        - Heating rate effects
        - Kinetic parameter extraction
        """
        params = input_data.get('parameters', {})
        heating_rates = params.get('heating_rates', [5, 10, 20, 40])  # K/min

        results_by_rate = {}
        peak_temperatures = []

        for rate in heating_rates:
            # Modify input for each heating rate
            rate_input = input_data.copy()
            rate_input['parameters'] = {**params, 'heating_rate': rate}
            result = self._execute_standard_tga(rate_input)
            results_by_rate[f'{rate}_K_per_min'] = result

            # Extract peak temperature (max DTG)
            dtg = np.array(result['derivative_dtg_percent_per_c'])
            temp_c = np.array(result['temperature_c'])
            peak_temp = temp_c[np.argmax(dtg)]
            peak_temperatures.append(peak_temp)

        # Kissinger analysis for activation energy
        # ln(β/Tp^2) = -Ea/(R*Tp) + const
        # where β = heating rate, Tp = peak temperature, Ea = activation energy, R = gas constant

        peak_temperatures_k = np.array(peak_temperatures) + 273.15
        heating_rates_arr = np.array(heating_rates) / 60  # Convert to K/s
        R = 8.314  # J/(mol·K)

        # Linear fit
        x = 1000 / peak_temperatures_k  # 1000/T for better numerics
        y = np.log(heating_rates_arr / peak_temperatures_k**2)

        # Calculate activation energy from slope
        slope = np.polyfit(x, y, 1)[0]
        Ea_kj_mol = -slope * R / 1000  # Convert to kJ/mol

        return {
            'technique': 'Multi-Ramp TGA',
            'heating_rates_k_per_min': heating_rates,
            'results_by_heating_rate': results_by_rate,
            'kinetic_analysis': {
                'method': 'Kissinger',
                'activation_energy_kj_per_mol': Ea_kj_mol,
                'pre_exponential_factor': 1e10,  # s^-1 (typical)
                'reaction_order': 1,
                'peak_temperatures_c': peak_temperatures
            },
            'heating_rate_dependence': {
                'peak_shift': 'increases_with_heating_rate',
                'peak_broadening': 'increases_at_high_rates',
                'recommended_rate': 10.0  # K/min for standard analysis
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
        if 'data_file' not in data and 'tga_data' not in data:
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
                name='standard_tga',
                description='Standard TGA for thermal degradation',
                input_types=['thermogram', 'temperature_program'],
                output_types=['weight_loss', 'decomposition_temp', 'residue'],
                typical_use_cases=['thermal_stability', 'composition_analysis', 'degradation_study']
            ),
            Capability(
                name='isothermal_tga',
                description='Isothermal TGA for degradation kinetics',
                input_types=['isothermal_data'],
                output_types=['kinetic_parameters', 'half_life'],
                typical_use_cases=['stability_testing', 'shelf_life_prediction']
            ),
            Capability(
                name='tga_ftir',
                description='TGA-FTIR for evolved gas identification',
                input_types=['coupled_tga_ftir'],
                output_types=['gas_identification', 'degradation_mechanism'],
                typical_use_cases=['mechanism_elucidation', 'gas_analysis']
            ),
            Capability(
                name='tga_ms',
                description='TGA-MS for molecular identification',
                input_types=['coupled_tga_ms'],
                output_types=['molecular_weight', 'fragmentation_pattern'],
                typical_use_cases=['species_identification', 'quantitative_gas_analysis']
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata."""
        return AgentMetadata(
            name="TGAAgent",
            version=self.VERSION,
            description="Thermogravimetric Analysis expert for thermal decomposition",
            author="Materials Characterization Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['tga', 'txt', 'csv', 'xlsx']
        )

    # ExperimentalAgent interface methods
    def connect_instrument(self) -> bool:
        """Connect to TGA instrument."""
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw TGA data."""
        # In production:
        # - Baseline correction
        # - Smoothing
        # - Derivative calculation
        # - Peak finding
        return raw_data

    # Integration methods for cross-validation
    @staticmethod
    def correlate_with_dsc(tga_result: Dict[str, Any], dsc_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate TGA decomposition with DSC thermal events.

        TGA measures mass loss.
        DSC measures heat flow.
        Decompositions should be exo/endothermic in DSC.

        Args:
            tga_result: TGA analysis
            dsc_result: DSC analysis

        Returns:
            Correlation report
        """
        tga_decomp_temp = tga_result.get('thermal_stability', {}).get('onset_decomposition_c', 0)
        dsc_peaks = dsc_result.get('melting_transition', {})

        return {
            'correlation_type': 'TGA_DSC_thermal_events',
            'tga_decomposition_c': tga_decomp_temp,
            'dsc_events': dsc_peaks,
            'notes': 'TGA decomposition should correlate with DSC exothermic/endothermic events'
        }

    @staticmethod
    def validate_with_eds(tga_result: Dict[str, Any], eds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate TGA composition with EDS elemental analysis.

        TGA provides weight percentages of components.
        EDS provides elemental composition.
        Should be consistent.

        Args:
            tga_result: TGA composition analysis
            eds_result: EDS elemental analysis

        Returns:
            Validation report
        """
        tga_residue = tga_result.get('residue_analysis', {}).get('final_residue_percent', 0)
        tga_organic = tga_result.get('composition_estimate', {}).get('organic_content_percent', 0)

        return {
            'validation_type': 'TGA_EDS_composition',
            'tga_inorganic_percent': tga_residue,
            'tga_organic_percent': tga_organic,
            'notes': 'TGA residue should match inorganic content from EDS',
            'recommendation': 'Correlate TGA residue with metal/inorganic elements from EDS'
        }
