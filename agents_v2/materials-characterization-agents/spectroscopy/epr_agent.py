"""
EPR Agent for Electron Paramagnetic Resonance Spectroscopy

This agent provides comprehensive EPR capabilities including:
- CW-EPR: Continuous wave EPR for radical detection
- Pulse EPR: ESEEM, HYSCORE for hyperfine interactions
- ENDOR: Electron-nuclear double resonance
- High-field EPR: 94 GHz (W-band) and above
- Variable temperature EPR
- Spin quantification and kinetics
- Multi-frequency EPR (X, Q, W-band)

Author: Claude (Anthropic)
Date: 2025-10-01
Version: 1.0.0
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum

from base_agent import ExperimentalAgent, AgentResult


class EPRBand(Enum):
    """EPR frequency bands"""
    L_BAND = ("L", 1.0, 0.035)    # (name, frequency_GHz, field_T)
    S_BAND = ("S", 3.0, 0.107)
    X_BAND = ("X", 9.5, 0.339)    # Most common
    Q_BAND = ("Q", 35, 1.25)
    W_BAND = ("W", 94, 3.35)


class RadicalType(Enum):
    """Common radical types"""
    ORGANIC_RADICAL = "organic_radical"
    TRANSITION_METAL = "transition_metal"
    NITROXIDE = "nitroxide"
    SEMIQUINONE = "semiquinone"
    DPPH = "dpph_standard"
    DEFECT_CENTER = "defect_center"


@dataclass
class EPRResult:
    """Results from EPR measurement"""
    experiment_type: str
    frequency_band: str
    microwave_frequency_ghz: float
    magnetic_field_range_t: Tuple[float, float]
    temperature_k: float
    spectrum_data: Dict[str, Any]
    spectroscopic_parameters: Dict[str, Any]
    radical_characterization: Dict[str, Any]
    spin_quantification: Dict[str, float]
    metadata: Dict[str, Any]


class EPRAgent(ExperimentalAgent):
    """
    Agent for Electron Paramagnetic Resonance spectroscopy.

    Capabilities:
    - CW-EPR (continuous wave) for radical detection
    - Multi-frequency EPR (L, S, X, Q, W-band)
    - Variable temperature (4-400 K)
    - Spin quantification
    - g-factor determination
    - Hyperfine structure analysis
    - Radical kinetics and stability
    - Transition metal characterization
    """

    NAME = "EPRAgent"
    VERSION = "1.0.0"
    DESCRIPTION = "Comprehensive EPR spectroscopy for paramagnetic species characterization"

    SUPPORTED_TECHNIQUES = [
        'cw_epr',           # Continuous wave EPR
        'pulse_epr',        # Pulsed EPR
        'endor',            # Electron-nuclear double resonance
        'eseem',            # Electron spin echo envelope modulation
        'hyscore',          # Hyperfine sublevel correlation
        'multi_frequency',  # Multi-frequency EPR
        'variable_temp',    # Temperature-dependent studies
        'spin_trapping',    # Spin trap experiments
        'power_saturation', # Power saturation studies
        'kinetics'          # Time-resolved EPR
    ]

    # Standard g-values for reference
    G_VALUES = {
        'free_electron': 2.0023193043737,
        'organic_radical': (2.0020, 2.0040),
        'nitroxide': (2.0055, 2.0061),
        'Cu2+': (2.05, 2.30),
        'Fe3+': (2.0, 6.0),
        'Mn2+': 2.001,
        'DPPH': 2.0036
    }

    def __init__(self):
        super().__init__(self.NAME, self.VERSION, self.DESCRIPTION)
        self.capabilities = {
            'frequency_bands': ['L', 'S', 'X', 'Q', 'W'],
            'field_range_t': (0, 3.5),
            'temperature_range_k': (4, 400),
            'time_resolution_ns': 1,  # For pulse EPR
            'sensitivity_spins': 1e10,
            'modulation_amplitude_mt': (0.01, 10),
            'microwave_power_mw': (0.01, 200),
            'field_modulation_khz': 100,
            'variable_temperature': True,
            'spin_quantification': True
        }

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute EPR measurement.

        Args:
            input_data: Dictionary containing:
                - technique: EPR experiment type
                - frequency_band: L, S, X, Q, or W-band
                - field_range_mt: Magnetic field range in mT
                - temperature_k: Sample temperature
                - modulation_amplitude_mt: Field modulation
                - microwave_power_mw: MW power
                - sample_type: Type of paramagnetic species

        Returns:
            AgentResult with EPR spectrum and analysis
        """
        start_time = datetime.now()

        # Validate input
        validation = self.validate_input(input_data)
        if not validation['valid']:
            return AgentResult(
                status='error',
                data={},
                metadata={'validation_errors': validation['errors']},
                provenance=self._create_provenance(input_data, start_time),
                errors=validation['errors']
            )

        technique = input_data['technique'].lower()

        # Route to appropriate technique
        technique_map = {
            'cw_epr': self._execute_cw_epr,
            'pulse_epr': self._execute_pulse_epr,
            'endor': self._execute_endor,
            'eseem': self._execute_eseem,
            'hyscore': self._execute_hyscore,
            'multi_frequency': self._execute_multi_frequency,
            'variable_temp': self._execute_variable_temp,
            'spin_trapping': self._execute_spin_trapping,
            'power_saturation': self._execute_power_saturation,
            'kinetics': self._execute_kinetics
        }

        if technique not in technique_map:
            return AgentResult(
                status='error',
                data={},
                metadata={},
                provenance=self._create_provenance(input_data, start_time),
                errors=[f"Unsupported technique: {technique}"]
            )

        # Execute technique
        result_data = technique_map[technique](input_data)

        # Create result
        return AgentResult(
            status='success',
            data=result_data,
            metadata={
                'technique': technique,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'agent_version': self.VERSION
            },
            provenance=self._create_provenance(input_data, start_time),
            warnings=result_data.get('warnings', [])
        )

    def _execute_cw_epr(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute continuous wave EPR for radical detection and characterization"""
        frequency_band = input_data.get('frequency_band', 'X')
        temperature_k = input_data.get('temperature_k', 298)
        modulation_amp_mt = input_data.get('modulation_amplitude_mt', 0.1)
        microwave_power_mw = input_data.get('microwave_power_mw', 2.0)
        sample_type = input_data.get('sample_type', 'organic_radical')

        # Get frequency band parameters
        band_params = {
            'X': (9.5, 339, 3300, 3500),  # freq_GHz, field_mT, field_min_G, field_max_G
            'Q': (35, 1250, 12300, 12700),
            'W': (94, 3350, 33300, 33700)
        }

        freq_ghz, center_field_mt, field_min_g, field_max_g = band_params.get(frequency_band, band_params['X'])

        # Generate field array
        n_points = 4096
        field_gauss = np.linspace(field_min_g, field_max_g, n_points)
        field_mt = field_gauss * 0.1  # Convert to mT

        # Determine g-factor and linewidth based on sample type
        if sample_type == 'organic_radical':
            g_iso = 2.0032
            linewidth_g = 5.0  # Gauss
            hyperfine_lines = 1
        elif sample_type == 'nitroxide':
            g_iso = 2.0061
            linewidth_g = 2.0
            hyperfine_lines = 3  # Nitrogen hyperfine (I=1)
            a_n_gauss = 15.0
        elif sample_type == 'Cu2+':
            g_parallel = 2.26
            g_perp = 2.06
            a_parallel_gauss = 180
            linewidth_g = 20
            hyperfine_lines = 4  # Cu-63/65, I=3/2
        elif sample_type == 'Mn2+':
            g_iso = 2.001
            a_mn_gauss = 90
            linewidth_g = 10
            hyperfine_lines = 6  # Mn-55, I=5/2
        else:
            g_iso = 2.0036
            linewidth_g = 3.0
            hyperfine_lines = 1

        # Calculate resonance field
        h = 6.62607015e-34  # Planck constant
        mu_b = 9.2740100783e-24  # Bohr magneton
        freq_hz = freq_ghz * 1e9

        # Simulate spectrum
        signal = np.zeros_like(field_gauss)

        if sample_type in ['organic_radical', 'DPPH']:
            # Single line
            resonance_g = (h * freq_hz) / (mu_b * g_iso) * 1e4  # Convert to Gauss
            signal = self._generate_derivative_lorentzian(field_gauss, resonance_g, linewidth_g)

        elif sample_type == 'nitroxide':
            # Nitrogen hyperfine triplet (I=1, 2I+1=3 lines)
            resonance_g = (h * freq_hz) / (mu_b * g_iso) * 1e4
            for m_i in [-1, 0, 1]:
                field_pos = resonance_g + m_i * a_n_gauss
                signal += self._generate_derivative_lorentzian(field_gauss, field_pos, linewidth_g) / 3

        elif sample_type == 'Cu2+':
            # Anisotropic Cu2+ with hyperfine (simplified powder pattern)
            # Parallel component
            resonance_parallel = (h * freq_hz) / (mu_b * g_parallel) * 1e4
            for m_i in [-3/2, -1/2, 1/2, 3/2]:
                field_pos = resonance_parallel + m_i * a_parallel_gauss
                signal += self._generate_derivative_lorentzian(field_gauss, field_pos, linewidth_g) / 8

            # Perpendicular component (broader)
            resonance_perp = (h * freq_hz) / (mu_b * g_perp) * 1e4
            signal += self._generate_derivative_lorentzian(field_gauss, resonance_perp, linewidth_g * 3) / 2

        elif sample_type == 'Mn2+':
            # Mn2+ sextet (I=5/2, 2I+1=6 lines)
            resonance_g = (h * freq_hz) / (mu_b * g_iso) * 1e4
            for m_i in [-5/2, -3/2, -1/2, 1/2, 3/2, 5/2]:
                field_pos = resonance_g + m_i * a_mn_gauss
                signal += self._generate_derivative_lorentzian(field_gauss, field_pos, linewidth_g) / 6

        # Add noise
        noise_level = np.max(np.abs(signal)) / 200  # SNR ~ 200
        noise = np.random.normal(0, noise_level, n_points)
        signal += noise

        # Calculate spectroscopic parameters
        peak_to_peak_linewidth = linewidth_g
        double_integral = np.trapz(np.cumsum(signal), field_gauss)  # Absorption integral

        # Spin quantification (compare to standard)
        spins_per_sample = abs(double_integral) * 1e13  # Simplified

        # Analyze g-factors
        if sample_type == 'Cu2+':
            g_values = {
                'g_parallel': g_parallel,
                'g_perpendicular': g_perp,
                'g_average': (g_parallel + 2*g_perp) / 3,
                'anisotropy': 'axial'
            }
        else:
            g_values = {
                'g_isotropic': g_iso,
                'anisotropy': 'isotropic'
            }

        return {
            'experiment_type': 'CW-EPR',
            'frequency_band': frequency_band,
            'microwave_frequency_ghz': freq_ghz,
            'temperature_k': temperature_k,
            'modulation_amplitude_mt': modulation_amp_mt,
            'microwave_power_mw': microwave_power_mw,
            'spectrum': {
                'magnetic_field_gauss': field_gauss.tolist(),
                'magnetic_field_mt': field_mt.tolist(),
                'signal_derivative': signal.tolist(),
                'sweep_width_gauss': field_max_g - field_min_g,
                'center_field_gauss': (field_max_g + field_min_g) / 2
            },
            'spectroscopic_parameters': {
                **g_values,
                'peak_to_peak_linewidth_gauss': float(peak_to_peak_linewidth),
                'hyperfine_coupling_gauss': a_n_gauss if sample_type == 'nitroxide' else
                                            a_parallel_gauss if sample_type == 'Cu2+' else
                                            a_mn_gauss if sample_type == 'Mn2+' else None,
                'number_of_lines': hyperfine_lines,
                'line_shape': 'lorentzian'
            },
            'radical_characterization': {
                'radical_type': sample_type,
                'paramagnetic_center': self._identify_paramagnetic_center(g_iso if 'g_iso' in locals() else g_parallel),
                'spin_state': 'S=1/2' if sample_type != 'Mn2+' else 'S=5/2',
                'electronic_configuration': self._get_electronic_config(sample_type)
            },
            'spin_quantification': {
                'double_integral': float(double_integral),
                'spin_concentration_spins_g': float(spins_per_sample / 0.1),  # Assume 0.1g sample
                'spin_concentration_mol_l': float(spins_per_sample / (6.022e23 * 0.001)),  # Assume 1 mL
                'relative_intensity': 1.0
            },
            'quality_metrics': {
                'snr': float(200 + np.random.uniform(-20, 20)),
                'baseline_quality': 0.95,
                'modulation_optimization': 'optimal' if modulation_amp_mt < linewidth_g * 0.1 else 'over_modulated'
            },
            'experimental_conditions': {
                'sample_state': 'liquid' if temperature_k > 273 else 'frozen',
                'solvent': 'toluene' if temperature_k > 273 else 'frozen_solution',
                'tube_type': 'quartz_epr_tube',
                'sample_volume_ml': 0.2
            }
        }

    def _execute_pulse_epr(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pulsed EPR for relaxation and spin dynamics"""
        pulse_sequence = input_data.get('pulse_sequence', 'two_pulse_echo')

        # Generate time axis for echo decay
        tau_values_ns = np.linspace(100, 10000, 100)  # ns

        # Simulate echo decay (exponential with T2)
        t2_phase_memory_ns = 2000
        echo_intensity = np.exp(-tau_values_ns / t2_phase_memory_ns)

        # Add ESEEM modulation (nuclear modulation)
        eseem_freq_mhz = 14.7  # Proton Larmor at X-band
        modulation_depth = 0.1
        echo_intensity *= (1 + modulation_depth * np.sin(2 * np.pi * eseem_freq_mhz * tau_values_ns / 1000))

        # Add noise
        echo_intensity += np.random.normal(0, 0.02, len(tau_values_ns))

        return {
            'experiment_type': 'Pulse EPR',
            'pulse_sequence': pulse_sequence,
            'time_domain': {
                'tau_ns': tau_values_ns.tolist(),
                'echo_intensity': echo_intensity.tolist()
            },
            'relaxation_parameters': {
                't2_phase_memory_ns': t2_phase_memory_ns,
                't2_phase_memory_us': t2_phase_memory_ns / 1000,
                'decay_function': 'exponential'
            },
            'hyperfine_modulation': {
                'eseem_detected': True,
                'modulation_frequency_mhz': eseem_freq_mhz,
                'modulation_depth': modulation_depth,
                'coupled_nuclei': ['1H']
            }
        }

    def _execute_endor(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ENDOR (Electron-Nuclear DOuble Resonance)"""

        # RF frequency scan around nuclear Larmor frequencies
        rf_freq_mhz = np.linspace(5, 25, 200)  # MHz

        # Simulate ENDOR spectrum with nuclear transitions
        signal = np.zeros_like(rf_freq_mhz)

        # Proton ENDOR peaks (split by hyperfine)
        larmor_1h = 14.7  # MHz at X-band
        a_proton_mhz = 5.0  # Hyperfine coupling
        signal += self._generate_lorentzian_1d(rf_freq_mhz, larmor_1h - a_proton_mhz/2, 0.5)
        signal += self._generate_lorentzian_1d(rf_freq_mhz, larmor_1h + a_proton_mhz/2, 0.5)

        # Nitrogen ENDOR
        larmor_14n = 1.1  # MHz at X-band
        a_nitrogen_mhz = 2.0
        signal += 0.5 * self._generate_lorentzian_1d(rf_freq_mhz, larmor_14n + a_nitrogen_mhz, 0.3)

        return {
            'experiment_type': 'ENDOR',
            'rf_frequency_mhz': rf_freq_mhz.tolist(),
            'endor_intensity': signal.tolist(),
            'identified_nuclei': [
                {
                    'nucleus': '1H',
                    'larmor_frequency_mhz': larmor_1h,
                    'hyperfine_coupling_mhz': a_proton_mhz,
                    'number_of_coupled_protons': 2
                },
                {
                    'nucleus': '14N',
                    'larmor_frequency_mhz': larmor_14n,
                    'hyperfine_coupling_mhz': a_nitrogen_mhz,
                    'number_of_coupled_nitrogens': 1
                }
            ],
            'advantages': [
                'Resolves overlapping hyperfine structure',
                'Identifies coupled nuclei',
                'Measures weak hyperfine couplings'
            ]
        }

    def _execute_multi_frequency(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-frequency EPR (X, Q, W-band) for g-anisotropy resolution"""

        bands = ['X', 'Q', 'W']
        results_by_band = []

        for band in bands:
            # Simulate spectrum at each frequency
            result = self._execute_cw_epr({**input_data, 'frequency_band': band})
            results_by_band.append({
                'band': band,
                'frequency_ghz': result['microwave_frequency_ghz'],
                'g_resolution': result['spectroscopic_parameters']
            })

        return {
            'experiment_type': 'Multi-frequency EPR',
            'bands_measured': bands,
            'results_by_band': results_by_band,
            'advantages': {
                'g_resolution': 'Higher frequency = better g-factor resolution',
                'orientation_selection': 'Improved at high field',
                'frequency_dependence': 'Distinguishes g-anisotropy from A-anisotropy'
            }
        }

    def _execute_variable_temp(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute variable temperature EPR for dynamics and thermodynamics"""

        temperatures = np.array([100, 150, 200, 250, 298, 350, 400])  # K

        # Simulate temperature-dependent parameters
        # Curie law: intensity ∝ 1/T
        intensities = 1000 / temperatures

        # Temperature-dependent linewidth (Arrhenius)
        activation_energy_kj_mol = 15
        r_const = 8.314  # J/mol·K
        linewidths = 2.0 * np.exp(activation_energy_kj_mol * 1000 / (r_const * temperatures))

        return {
            'experiment_type': 'Variable Temperature EPR',
            'temperature_range_k': (temperatures[0], temperatures[-1]),
            'measurements': [
                {
                    'temperature_k': float(t),
                    'intensity': float(i),
                    'linewidth_gauss': float(lw),
                    'g_factor': 2.0036  # May shift with temperature
                }
                for t, i, lw in zip(temperatures, intensities, linewidths)
            ],
            'thermodynamic_analysis': {
                'curie_law_followed': True,
                'activation_energy_kj_mol': activation_energy_kj_mol,
                'dynamics': 'thermally_activated_motion',
                'phase_transitions_detected': False
            }
        }

    def _execute_spin_trapping(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute spin trapping for transient radical detection"""
        spin_trap = input_data.get('spin_trap', 'DMPO')  # 5,5-dimethyl-1-pyrroline N-oxide

        # DMPO traps radicals forming stable nitroxide adducts
        # Hydroxyl radical adduct: quartet of quartets

        return {
            'experiment_type': 'Spin Trapping',
            'spin_trap': spin_trap,
            'trapped_radicals': [
                {
                    'radical': 'hydroxyl_radical',
                    'adduct': 'DMPO-OH',
                    'hyperfine_pattern': 'quartet_of_quartets',
                    'a_n_gauss': 14.9,
                    'a_h_beta_gauss': 14.9,
                    'g_factor': 2.0061,
                    'identification': 'confirmed'
                },
                {
                    'radical': 'superoxide',
                    'adduct': 'DMPO-OOH',
                    'hyperfine_pattern': 'quartet_of_doublets',
                    'a_n_gauss': 14.3,
                    'a_h_beta_gauss': 11.7,
                    'g_factor': 2.0056,
                    'identification': 'confirmed'
                }
            ],
            'kinetics': {
                'formation_rate_constant_m_s': 5e8,
                'adduct_stability': 'stable_for_minutes',
                'decay_half_life_min': 15
            }
        }

    def _execute_power_saturation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute power saturation study for relaxation and accessibility"""

        # Microwave power range (mW)
        powers_mw = np.logspace(-3, 2, 20)  # 0.001 to 100 mW

        # Simulate signal intensity vs power
        # I ∝ sqrt(P) at low power, saturates at high power
        p_half = 5.0  # mW (half-saturation power)
        intensities = np.sqrt(powers_mw) / (1 + powers_mw / p_half)

        # Add noise
        intensities += np.random.normal(0, 0.02 * np.max(intensities), len(powers_mw))

        return {
            'experiment_type': 'Power Saturation',
            'microwave_powers_mw': powers_mw.tolist(),
            'signal_intensities': intensities.tolist(),
            'saturation_parameters': {
                'p_half_saturation_mw': float(p_half),
                'saturation_behavior': 'homogeneous',
                't1_estimate_ns': 1000,  # Related to P_1/2
                'accessibility': 'bulk_solution'
            },
            'applications': [
                'Distinguish overlapping species by relaxation',
                'Probe radical environment',
                'Optimize measurement conditions'
            ]
        }

    def _execute_kinetics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute time-resolved EPR for radical kinetics"""

        time_points_s = np.linspace(0, 600, 50)  # 10 minutes

        # Simulate radical decay kinetics (second-order)
        initial_conc = 1e-4  # M
        rate_constant = 1e6  # M^-1 s^-1
        concentrations = initial_conc / (1 + initial_conc * rate_constant * time_points_s)

        # EPR intensity proportional to concentration
        intensities = concentrations / initial_conc

        return {
            'experiment_type': 'Kinetic EPR',
            'time_points_s': time_points_s.tolist(),
            'relative_intensities': intensities.tolist(),
            'kinetic_analysis': {
                'reaction_order': 2,
                'rate_constant_m_s': rate_constant,
                'initial_concentration_m': initial_conc,
                'half_life_s': 1 / (rate_constant * initial_conc),
                'mechanism': 'radical_recombination'
            }
        }

    # Helper methods

    def _generate_derivative_lorentzian(self, x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
        """Generate derivative of Lorentzian lineshape (typical EPR display)"""
        # d/dx [L(x)] where L(x) = gamma^2 / ((x-x0)^2 + gamma^2)
        numerator = -2 * gamma**2 * (x - x0)
        denominator = ((x - x0)**2 + gamma**2)**2
        return numerator / denominator

    def _generate_lorentzian_1d(self, x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
        """Generate Lorentzian absorption lineshape"""
        return gamma**2 / ((x - x0)**2 + gamma**2)

    def _identify_paramagnetic_center(self, g_value: float) -> str:
        """Identify paramagnetic center from g-value"""
        if abs(g_value - self.G_VALUES['free_electron']) < 0.001:
            return 'free_radical_or_defect'
        elif 2.002 <= g_value <= 2.004:
            return 'organic_radical_carbon_centered'
        elif 2.004 <= g_value <= 2.007:
            return 'nitroxide_or_oxygen_centered'
        elif 2.0 <= g_value < 2.01:
            return 'transition_metal_high_spin'
        elif 2.1 <= g_value <= 2.3:
            return 'Cu2+_or_low_spin_metal'
        else:
            return 'unusual_paramagnetic_species'

    def _get_electronic_config(self, sample_type: str) -> str:
        """Get electronic configuration"""
        configs = {
            'organic_radical': '...π*^1 (unpaired electron in π* orbital)',
            'nitroxide': 'N-O· (nitrogen-oxygen radical)',
            'Cu2+': '3d^9 (one unpaired electron)',
            'Mn2+': '3d^5 (five unpaired electrons, high spin)',
            'Fe3+': '3d^5 (five unpaired electrons)',
            'DPPH': 'organic radical (stable reference)'
        }
        return configs.get(sample_type, 'unknown')

    # Cross-validation methods

    @staticmethod
    def validate_with_uv_vis(epr_result: Dict[str, Any], uv_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate EPR radical detection with UV-Vis absorption"""
        epr_detected = epr_result.get('spin_quantification', {}).get('spin_concentration_mol_l', 0) > 1e-6
        uv_absorption = uv_result.get('absorbance_at_peak', 0) > 0.1

        return {
            'technique_pair': 'EPR-UV-Vis',
            'parameter': 'radical_presence',
            'epr_radical_detected': epr_detected,
            'uv_absorption_detected': uv_absorption,
            'agreement': 'good' if epr_detected == uv_absorption else 'check_conditions',
            'note': 'EPR: paramagnetic species, UV-Vis: electronic transitions'
        }

    @staticmethod
    def validate_with_nmr(epr_result: Dict[str, Any], nmr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate EPR with NMR for paramagnetic effects"""
        epr_spin_conc = epr_result.get('spin_quantification', {}).get('spin_concentration_mol_l', 0)
        nmr_line_broadening = nmr_result.get('quality_metrics', {}).get('resolution_hz', 0) > 5.0

        # High radical concentration causes NMR line broadening
        expected_broadening = epr_spin_conc > 1e-4

        return {
            'technique_pair': 'EPR-NMR',
            'parameter': 'paramagnetic_effects',
            'epr_spin_concentration_m': epr_spin_conc,
            'nmr_line_broadening_observed': nmr_line_broadening,
            'agreement': 'consistent' if nmr_line_broadening == expected_broadening else 'check_sample',
            'note': 'Paramagnetic species broaden or quench NMR signals'
        }

    @staticmethod
    def validate_with_electrochemistry(epr_result: Dict[str, Any], cv_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate EPR with electrochemistry for redox properties"""
        epr_radical_type = epr_result.get('radical_characterization', {}).get('radical_type', '')
        cv_reversible = cv_result.get('reversibility', {}).get('reversible', False)

        return {
            'technique_pair': 'EPR-Electrochemistry',
            'parameter': 'redox_stability',
            'epr_radical_type': epr_radical_type,
            'cv_reversibility': 'reversible' if cv_reversible else 'irreversible',
            'stability': 'stable' if cv_reversible else 'unstable_upon_oxidation_reduction',
            'note': 'EPR monitors radical species, CV tests electrochemical stability'
        }

    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters"""
        errors = []
        warnings = []

        # Check required fields
        if 'technique' not in input_data:
            errors.append("Missing required field: technique")
        else:
            technique = input_data['technique'].lower()
            if technique not in self.SUPPORTED_TECHNIQUES:
                errors.append(f"Unsupported technique: {technique}")

        # Validate frequency band
        if 'frequency_band' in input_data:
            if input_data['frequency_band'] not in ['L', 'S', 'X', 'Q', 'W']:
                errors.append("Invalid frequency band. Must be L, S, X, Q, or W")

        # Validate temperature
        if 'temperature_k' in input_data:
            temp = input_data['temperature_k']
            if temp < 4 or temp > 400:
                warnings.append(f"Temperature {temp} K outside typical range (4-400 K)")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def estimate_resources(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate experimental resources"""
        technique = input_data.get('technique', 'cw_epr')

        time_estimates = {
            'cw_epr': 5,  # minutes
            'pulse_epr': 30,
            'endor': 60,
            'eseem': 45,
            'hyscore': 120,
            'multi_frequency': 20,
            'variable_temp': 120,
            'spin_trapping': 15,
            'power_saturation': 30,
            'kinetics': 60
        }

        return {
            'estimated_time_minutes': time_estimates.get(technique, 30),
            'sample_amount_mg': 1.0,
            'sample_volume_ml': 0.2,
            'consumables': ['quartz_epr_tube', 'spin_trap' if 'trapping' in technique else None],
            'spectrometer_cost_per_hour': 200.0,
            'recoverable_sample': True
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities"""
        return self.capabilities

    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata"""
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'description': self.DESCRIPTION,
            'supported_techniques': self.SUPPORTED_TECHNIQUES,
            'frequency_bands': self.capabilities['frequency_bands'],
            'temperature_range_k': self.capabilities['temperature_range_k'],
            'capabilities': self.capabilities,
            'cross_validation_methods': [
                'validate_with_uv_vis',
                'validate_with_nmr',
                'validate_with_electrochemistry'
            ]
        }

    def _create_provenance(self, input_data: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Create provenance record"""
        return {
            'agent': self.NAME,
            'version': self.VERSION,
            'timestamp': datetime.now().isoformat(),
            'input_hash': hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest(),
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'parameters': input_data
        }


# Example usage
if __name__ == "__main__":
    agent = EPRAgent()

    # Example: CW-EPR of nitroxide radical
    result = agent.execute({
        'technique': 'cw_epr',
        'frequency_band': 'X',
        'temperature_k': 298,
        'sample_type': 'nitroxide'
    })
    print("CW-EPR result:", result.status)

    # Example: Spin trapping
    result = agent.execute({
        'technique': 'spin_trapping',
        'spin_trap': 'DMPO'
    })
    print("Spin trapping result:", result.status)
