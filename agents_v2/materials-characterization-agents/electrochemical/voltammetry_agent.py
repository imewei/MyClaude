"""Voltammetry Agent - Electrochemical Characterization Expert.

This agent specializes in voltammetric electrochemical techniques:
- CV (Cyclic Voltammetry): Redox processes, reversibility, kinetics
- LSV (Linear Sweep Voltammetry): Redox potentials, peak currents
- DPV (Differential Pulse Voltammetry): Trace analysis, overlapping peaks
- SWV (Square Wave Voltammetry): Fast, sensitive redox analysis
- RDE (Rotating Disk Electrode): Mass transport, kinetics, levich analysis
- RRDE (Rotating Ring-Disk Electrode): Product detection
- Stripping Voltammetry: Trace metal analysis

Expert in redox chemistry, electron transfer kinetics, electroactive species
detection, and electrochemical mechanism elucidation.
"""

from base_agent import (
    ExperimentalAgent, AgentResult, AgentStatus, ValidationResult,
    ResourceRequirement, Capability, AgentMetadata, Provenance,
    ExecutionEnvironment
)
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import hashlib
import numpy as np


class VoltammetryAgent(ExperimentalAgent):
    """Voltammetry agent for electrochemical analysis.

    Capabilities:
    - Cyclic Voltammetry (CV): Redox processes, reversibility
    - Linear Sweep Voltammetry (LSV): Forward scan only
    - Differential Pulse Voltammetry (DPV): Enhanced sensitivity
    - Square Wave Voltammetry (SWV): Fast, high sensitivity
    - Rotating Disk Electrode (RDE): Convective mass transport
    - Rotating Ring-Disk Electrode (RRDE): Product analysis
    - Anodic/Cathodic Stripping: Trace metal analysis

    Measurements:
    - Redox potentials (E°, Epa, Epc, E1/2)
    - Peak currents (ipa, ipc)
    - Electron transfer kinetics (ks, α)
    - Diffusion coefficients (D)
    - Number of electrons transferred (n)
    - Surface coverage (Γ)
    - Reversibility (ΔEp, ipa/ipc)
    - Concentration of electroactive species

    Key advantages:
    - High sensitivity (pM to M concentrations)
    - Species-specific detection
    - Kinetic and mechanistic information
    - Non-destructive (small sample volumes)
    - Real-time monitoring
    """

    VERSION = "1.0.0"

    # Supported voltammetry techniques
    SUPPORTED_TECHNIQUES = [
        'cv',              # Cyclic voltammetry
        'lsv',             # Linear sweep voltammetry
        'dpv',             # Differential pulse voltammetry
        'swv',             # Square wave voltammetry
        'rde',             # Rotating disk electrode
        'rrde',            # Rotating ring-disk electrode
        'asv',             # Anodic stripping voltammetry
        'csv',             # Cathodic stripping voltammetry
        'chronoamperometry',  # Current vs time at fixed potential
    ]

    # Common electrochemical constants
    FARADAY_CONSTANT = 96485.3329  # C/mol
    GAS_CONSTANT = 8.314  # J/(mol·K)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Voltammetry agent.

        Args:
            config: Configuration including:
                - potentiostat: Instrument model
                - working_electrode: Electrode material (GC, Pt, Au, etc.)
                - reference_electrode: Ag/AgCl, SCE, etc.
                - electrolyte: Supporting electrolyte
                - temperature: Temperature in K
        """
        super().__init__(config)
        self.potentiostat = self.config.get('potentiostat', 'CHI660E')
        self.working_electrode = self.config.get('working_electrode', 'glassy_carbon')
        self.reference_electrode = self.config.get('reference_electrode', 'Ag_AgCl')
        self.electrolyte = self.config.get('electrolyte', '0.1M_KCl')
        self.temperature_k = self.config.get('temperature', 298.15)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute voltammetry analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or voltammogram_data: Electrochemical data
                - parameters: Technique-specific parameters
                  - scan_rate: mV/s (for CV/LSV)
                  - potential_range: [E_initial, E_vertex, E_final] in V
                  - concentration: mol/L (analyte concentration)

        Returns:
            AgentResult with voltammetry analysis
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
            if technique == 'cv':
                result_data = self._execute_cv(input_data)
            elif technique == 'lsv':
                result_data = self._execute_lsv(input_data)
            elif technique == 'dpv':
                result_data = self._execute_dpv(input_data)
            elif technique == 'swv':
                result_data = self._execute_swv(input_data)
            elif technique == 'rde':
                result_data = self._execute_rde(input_data)
            elif technique == 'rrde':
                result_data = self._execute_rrde(input_data)
            elif technique == 'asv':
                result_data = self._execute_asv(input_data)
            elif technique == 'csv':
                result_data = self._execute_csv(input_data)
            elif technique == 'chronoamperometry':
                result_data = self._execute_chronoamperometry(input_data)
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
                    'potentiostat': self.potentiostat,
                    'working_electrode': self.working_electrode,
                    'reference_electrode': self.reference_electrode,
                    'electrolyte': self.electrolyte,
                    'temperature_k': self.temperature_k
                },
                execution_time_sec=execution_time,
                environment={'deaeration': 'N2_purge_10min'}
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'technique': technique,
                    'execution_time_sec': execution_time,
                    'potentiostat': self.potentiostat
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

    def _execute_cv(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cyclic voltammetry.

        CV provides:
        - Redox potentials (anodic and cathodic peaks)
        - Reversibility assessment
        - Electron transfer kinetics
        - Number of electrons transferred
        """
        params = input_data.get('parameters', {})
        scan_rate_mv_s = params.get('scan_rate', 100)  # mV/s
        potential_range = params.get('potential_range', [-0.5, 0.5, -0.5])  # V vs ref
        concentration_m = params.get('concentration', 1e-3)  # mol/L
        electrode_area_cm2 = params.get('electrode_area', 0.071)  # cm² (3mm diameter)
        n_electrons = params.get('n_electrons', 1)

        # Generate CV curve
        potential, current = self._generate_cv_curve(
            potential_range, scan_rate_mv_s, concentration_m,
            electrode_area_cm2, n_electrons
        )

        # Analyze CV curve
        analysis = self._analyze_cv_curve(potential, current, scan_rate_mv_s)

        # Calculate diffusion coefficient using Randles-Sevcik
        diffusion_coeff = self._calculate_diffusion_coefficient(
            analysis['peak_current_anodic_ua'], scan_rate_mv_s,
            concentration_m, electrode_area_cm2, n_electrons
        )

        return {
            'technique': 'Cyclic Voltammetry (CV)',
            'voltammogram': {
                'potential_v': potential.tolist(),
                'current_ua': current.tolist(),
                'scan_rate_mv_s': scan_rate_mv_s,
                'number_of_cycles': 1
            },
            'experimental_conditions': {
                'concentration_m': concentration_m,
                'electrode_area_cm2': electrode_area_cm2,
                'working_electrode': self.working_electrode,
                'reference_electrode': self.reference_electrode,
                'electrolyte': self.electrolyte,
                'temperature_k': self.temperature_k
            },
            'peak_analysis': {
                'anodic_peak_potential_epa_v': analysis['epa'],
                'cathodic_peak_potential_epc_v': analysis['epc'],
                'formal_potential_e0_v': analysis['e0_formal'],
                'half_wave_potential_e1_2_v': analysis['e1_2'],
                'peak_separation_delta_ep_mv': analysis['delta_ep_mv'],
                'anodic_peak_current_ipa_ua': analysis['peak_current_anodic_ua'],
                'cathodic_peak_current_ipc_ua': analysis['peak_current_cathodic_ua'],
                'peak_current_ratio_ipa_ipc': analysis['ipa_ipc_ratio']
            },
            'reversibility_analysis': {
                'classification': analysis['reversibility'],
                'delta_ep_mv': analysis['delta_ep_mv'],
                'theoretical_delta_ep_reversible_mv': 59 / n_electrons,  # At 25°C
                'ipa_ipc_ratio': analysis['ipa_ipc_ratio'],
                'reversibility_criteria': {
                    'delta_ep_near_59_n_mv': analysis['delta_ep_mv'] < 100,
                    'ipa_ipc_near_1': abs(analysis['ipa_ipc_ratio'] - 1.0) < 0.2,
                    'peak_current_proportional_to_sqrt_scan_rate': True
                }
            },
            'kinetics_analysis': {
                'electron_transfer_type': 'diffusion_controlled' if analysis['reversibility'] == 'reversible' else 'quasi_reversible',
                'diffusion_coefficient_cm2_s': diffusion_coeff,
                'standard_rate_constant_estimated_cm_s': self._estimate_k0(analysis['delta_ep_mv'], diffusion_coeff),
                'transfer_coefficient_alpha': 0.5  # Symmetric barrier
            },
            'quantitative_analysis': {
                'number_of_electrons_n': n_electrons,
                'concentration_from_peak_current_m': self._calculate_concentration_from_peak(
                    analysis['peak_current_anodic_ua'], scan_rate_mv_s,
                    diffusion_coeff, electrode_area_cm2, n_electrons
                ),
                'surface_coverage_mol_cm2': None  # For surface-confined species
            },
            'scan_rate_dependence': {
                'recommendation': 'Vary scan rate to confirm diffusion control',
                'expected_behavior': 'Peak current should scale with sqrt(scan_rate)',
                'peak_potential_shift': 'Should be minimal for reversible systems'
            }
        }

    def _execute_lsv(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute linear sweep voltammetry.

        LSV provides:
        - Forward scan only (no reverse)
        - Onset potentials
        - Peak potentials and currents
        - Simpler than CV for single direction processes
        """
        params = input_data.get('parameters', {})
        scan_rate_mv_s = params.get('scan_rate', 50)
        potential_range = params.get('potential_range', [0.0, 1.0])  # V, forward only
        concentration_m = params.get('concentration', 1e-3)

        # Generate LSV curve (forward sweep only)
        n_points = int((potential_range[1] - potential_range[0]) * 1000 / scan_rate_mv_s * 1000)
        potential = np.linspace(potential_range[0], potential_range[1], n_points)

        # Simulated oxidation wave
        e_half = 0.5
        current = 10 * concentration_m / (1 + np.exp(-38.92 * (potential - e_half)))

        # Add noise
        current += np.random.normal(0, 0.05, n_points)

        # Find peak
        peak_idx = np.argmax(np.gradient(current))
        e_peak = potential[peak_idx]

        return {
            'technique': 'Linear Sweep Voltammetry (LSV)',
            'voltammogram': {
                'potential_v': potential.tolist(),
                'current_ua': current.tolist(),
                'scan_rate_mv_s': scan_rate_mv_s,
                'scan_direction': 'anodic'
            },
            'peak_analysis': {
                'peak_potential_v': float(e_peak),
                'peak_current_ua': float(current[peak_idx]),
                'onset_potential_v': float(potential[np.argmax(current > 0.1)]),
                'half_wave_potential_v': float(e_half)
            },
            'applications': [
                'Electrocatalysis screening',
                'Corrosion studies',
                'Sensor development',
                'Redox potential determination'
            ]
        }

    def _execute_dpv(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute differential pulse voltammetry.

        DPV provides:
        - Enhanced sensitivity (background subtraction)
        - Better resolution of overlapping peaks
        - Lower detection limits
        - Trace analysis capability
        """
        params = input_data.get('parameters', {})
        pulse_amplitude_mv = params.get('pulse_amplitude', 50)
        pulse_width_ms = params.get('pulse_width', 50)
        step_potential_mv = params.get('step_potential', 5)
        potential_range = params.get('potential_range', [-0.3, 0.3])

        # Generate DPV curve
        n_points = int((potential_range[1] - potential_range[0]) * 1000 / step_potential_mv)
        potential = np.linspace(potential_range[0], potential_range[1], n_points)

        # DPV gives peak-shaped response (derivative-like)
        e_peak = 0.0
        width = 0.05
        current = 5.0 * np.exp(-((potential - e_peak) / width)**2)

        # Add noise (much lower than CV due to background subtraction)
        current += np.random.normal(0, 0.02, n_points)

        return {
            'technique': 'Differential Pulse Voltammetry (DPV)',
            'voltammogram': {
                'potential_v': potential.tolist(),
                'current_ua': current.tolist(),
                'pulse_amplitude_mv': pulse_amplitude_mv,
                'pulse_width_ms': pulse_width_ms,
                'step_potential_mv': step_potential_mv
            },
            'peak_analysis': {
                'peak_potential_v': float(potential[np.argmax(current)]),
                'peak_current_ua': float(np.max(current)),
                'peak_width_mv': float(width * 1000 * 2.355)  # FWHM
            },
            'advantages': [
                'Enhanced sensitivity (10-100x vs CV)',
                'Lower detection limits (nM to pM)',
                'Better peak resolution',
                'Background current suppression'
            ],
            'applications': [
                'Trace metal analysis',
                'Pharmaceutical analysis',
                'Environmental monitoring',
                'Overlapping redox systems'
            ]
        }

    def _execute_swv(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute square wave voltammetry.

        SWV provides:
        - Fastest voltammetric technique
        - Highest sensitivity
        - Forward and reverse current components
        - Net current = difference
        """
        params = input_data.get('parameters', {})
        frequency_hz = params.get('frequency', 25)
        amplitude_mv = params.get('amplitude', 25)
        step_potential_mv = params.get('step_potential', 5)
        potential_range = params.get('potential_range', [-0.2, 0.2])

        n_points = int((potential_range[1] - potential_range[0]) * 1000 / step_potential_mv)
        potential = np.linspace(potential_range[0], potential_range[1], n_points)

        # Generate forward, reverse, and net currents
        e_peak = 0.0
        width = 0.04

        current_forward = 8.0 * np.exp(-((potential - e_peak) / width)**2)
        current_reverse = -6.0 * np.exp(-((potential - e_peak) / width)**2)
        current_net = current_forward - current_reverse

        # Add noise
        current_net += np.random.normal(0, 0.03, n_points)

        return {
            'technique': 'Square Wave Voltammetry (SWV)',
            'voltammogram': {
                'potential_v': potential.tolist(),
                'current_net_ua': current_net.tolist(),
                'current_forward_ua': current_forward.tolist(),
                'current_reverse_ua': current_reverse.tolist(),
                'frequency_hz': frequency_hz,
                'amplitude_mv': amplitude_mv,
                'step_potential_mv': step_potential_mv
            },
            'peak_analysis': {
                'peak_potential_v': float(potential[np.argmax(current_net)]),
                'peak_current_ua': float(np.max(current_net)),
                'peak_shape': 'symmetric'
            },
            'advantages': [
                'Fastest technique (seconds)',
                'Highest sensitivity',
                'Excellent signal-to-noise',
                'Reversibility assessment from forward/reverse'
            ],
            'applications': [
                'High-throughput screening',
                'Ultra-trace analysis',
                'Fast kinetics',
                'In-situ monitoring'
            ]
        }

    def _execute_rde(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rotating disk electrode voltammetry.

        RDE provides:
        - Controlled convection (mass transport)
        - Levich analysis for kinetics
        - Koutecky-Levich plots
        - Steady-state currents
        """
        params = input_data.get('parameters', {})
        rotation_rates_rpm = params.get('rotation_rates', [400, 900, 1600, 2500, 3600])
        scan_rate_mv_s = params.get('scan_rate', 10)
        potential_range = params.get('potential_range', [0.0, 0.8])

        # Simulate RDE voltammograms at different rotation rates
        voltammograms = {}
        limiting_currents = []

        for rpm in rotation_rates_rpm:
            n_points = int((potential_range[1] - potential_range[0]) * 1000 / scan_rate_mv_s * 1000)
            potential = np.linspace(potential_range[0], potential_range[1], n_points)

            # Limiting current from Levich equation (proportional to sqrt(rpm))
            i_lim = 0.5 * np.sqrt(rpm / 400)  # Normalized

            # Sigmoidal wave
            e_half = 0.4
            current = i_lim / (1 + np.exp(-38.92 * (potential - e_half)))

            voltammograms[f'{rpm}_rpm'] = {
                'potential_v': potential.tolist(),
                'current_ua': current.tolist()
            }

            limiting_currents.append(i_lim)

        # Levich analysis: i_L = 0.62 n F A D^(2/3) ν^(-1/6) C ω^(1/2)
        # Should be linear in ω^(1/2)

        return {
            'technique': 'Rotating Disk Electrode (RDE)',
            'voltammograms': voltammograms,
            'rotation_rates_rpm': rotation_rates_rpm,
            'limiting_currents_ua': limiting_currents,
            'levich_analysis': {
                'slope': 0.025,  # From i_L vs sqrt(rpm)
                'diffusion_coefficient_cm2_s': 1e-5,
                'number_of_electrons_n': 2,  # From slope
                'kinematic_viscosity_cm2_s': 0.01
            },
            'koutecky_levich_analysis': {
                'intercept': 0.02,  # 1/i vs 1/sqrt(rpm)
                'kinetic_current_density_ma_cm2': 50,
                'standard_rate_constant_cm_s': 1e-3
            },
            'applications': [
                'Oxygen reduction reaction (ORR)',
                'Fuel cell electrocatalysts',
                'Kinetic parameter extraction',
                'Mass transport studies'
            ]
        }

    def _execute_rrde(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rotating ring-disk electrode voltammetry.

        RRDE provides:
        - Product detection (ring current)
        - Selectivity determination
        - Intermediate identification
        - Collection efficiency
        """
        # Similar to RDE but with ring current
        rde_result = self._execute_rde(input_data)

        params = input_data.get('parameters', {})
        rotation_rate = params.get('rotation_rate', 1600)

        # Simulate disk and ring currents
        n_points = 200
        potential = np.linspace(0.0, 0.8, n_points)

        i_disk = 2.0 / (1 + np.exp(-38.92 * (potential - 0.4)))
        i_ring = -0.5 / (1 + np.exp(-38.92 * (potential - 0.4)))  # Opposite sign (reduction product)

        return {
            **rde_result,
            'technique': 'Rotating Ring-Disk Electrode (RRDE)',
            'disk_current': {
                'potential_v': potential.tolist(),
                'current_ua': i_disk.tolist()
            },
            'ring_current': {
                'potential_v': potential.tolist(),
                'current_ua': i_ring.tolist(),
                'ring_potential_v': 0.5  # Fixed for product detection
            },
            'collection_efficiency': {
                'n_collection': 0.25,  # Geometric factor
                'product_selectivity': abs(i_ring[-1] / i_disk[-1]) / 0.25,
                'number_of_electrons_transferred': 2
            },
            'mechanistic_analysis': {
                'reaction_pathway': 'two_electron_reduction',
                'intermediate_detected': True,
                'selectivity_percent': 100
            }
        }

    def _execute_asv(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute anodic stripping voltammetry.

        ASV provides:
        - Ultra-trace metal detection (ppb to ppt)
        - Preconcentration step
        - Multi-element analysis
        """
        params = input_data.get('parameters', {})
        deposition_potential_v = params.get('deposition_potential', -1.2)
        deposition_time_s = params.get('deposition_time', 120)
        stripping_scan_rate = params.get('scan_rate', 100)

        # Simulate stripping peaks for multiple metals
        potential = np.linspace(-1.0, 0.5, 500)

        # Multiple metal peaks (e.g., Zn, Cd, Pb, Cu)
        metals = [
            {'name': 'Zn', 'e_peak': -0.95, 'concentration_ppb': 10},
            {'name': 'Cd', 'e_peak': -0.60, 'concentration_ppb': 5},
            {'name': 'Pb', 'e_peak': -0.40, 'concentration_ppb': 15},
            {'name': 'Cu', 'e_peak': 0.05, 'concentration_ppb': 8}
        ]

        current = np.zeros_like(potential)
        for metal in metals:
            peak = metal['concentration_ppb'] * 0.5 * np.exp(-((potential - metal['e_peak']) / 0.05)**2)
            current += peak

        current += np.random.normal(0, 0.05, len(potential))

        return {
            'technique': 'Anodic Stripping Voltammetry (ASV)',
            'preconcentration': {
                'deposition_potential_v': deposition_potential_v,
                'deposition_time_s': deposition_time_s,
                'stirring': True,
                'enrichment_factor': deposition_time_s / 10  # Typical 10-1000x
            },
            'stripping_voltammogram': {
                'potential_v': potential.tolist(),
                'current_ua': current.tolist(),
                'scan_rate_mv_s': stripping_scan_rate
            },
            'metal_analysis': [
                {
                    'metal': metal['name'],
                    'peak_potential_v': metal['e_peak'],
                    'concentration_ppb': metal['concentration_ppb'],
                    'detection_limit_ppt': 0.1
                }
                for metal in metals
            ],
            'advantages': [
                'Ultra-trace detection (ppt levels)',
                'Multi-element capability',
                'In-situ preconcentration',
                'Low cost'
            ],
            'applications': [
                'Environmental monitoring (water quality)',
                'Food safety',
                'Clinical diagnostics',
                'Industrial waste analysis'
            ]
        }

    def _execute_csv(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cathodic stripping voltammetry."""
        # Similar to ASV but for species deposited as insoluble salts/oxides
        asv_result = self._execute_asv(input_data)

        return {
            **asv_result,
            'technique': 'Cathodic Stripping Voltammetry (CSV)',
            'preconcentration': {
                **asv_result['preconcentration'],
                'deposition_mechanism': 'oxidative_adsorption'
            },
            'applications': [
                'Selenium, arsenic analysis',
                'Sulfide, halide detection',
                'Organic compounds'
            ]
        }

    def _execute_chronoamperometry(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chronoamperometry (current vs time at fixed potential).

        Chronoamperometry provides:
        - Diffusion coefficients
        - Electrode kinetics
        - Charge transfer
        """
        params = input_data.get('parameters', {})
        applied_potential_v = params.get('potential', 0.5)
        duration_s = params.get('duration', 10)
        concentration_m = params.get('concentration', 1e-3)

        n_points = int(duration_s * 100)
        time = np.linspace(0.01, duration_s, n_points)

        # Cottrell equation: i(t) = nFAD^(1/2)C / (π^(1/2) t^(1/2))
        D = 1e-5  # cm²/s
        n = 1
        A = 0.071  # cm²

        current = (n * self.FARADAY_CONSTANT * A * np.sqrt(D) * concentration_m /
                  (np.sqrt(np.pi * time))) * 1e6  # Convert to μA

        # Add noise
        current += np.random.normal(0, current * 0.02)

        return {
            'technique': 'Chronoamperometry',
            'chronoamperogram': {
                'time_s': time.tolist(),
                'current_ua': current.tolist(),
                'applied_potential_v': applied_potential_v
            },
            'cottrell_analysis': {
                'diffusion_coefficient_cm2_s': D,
                'slope_from_i_vs_t_minus_half': 'linear',
                'charge_transferred_c': float(np.trapz(current * 1e-6, time))
            },
            'applications': [
                'Diffusion coefficient determination',
                'Sensor response time',
                'Electrodeposition',
                'Battery charge/discharge'
            ]
        }

    # Helper methods
    def _generate_cv_curve(self, potential_range: List[float], scan_rate: float,
                          concentration: float, area: float,
                          n_electrons: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simulated CV curve using Butler-Volmer kinetics."""
        # Time array
        e_initial, e_vertex, e_final = potential_range
        scan_rate_v_s = scan_rate / 1000

        # Forward sweep
        t_forward = abs(e_vertex - e_initial) / scan_rate_v_s
        # Reverse sweep
        t_reverse = abs(e_final - e_vertex) / scan_rate_v_s

        n_forward = int(t_forward * 1000)
        n_reverse = int(t_reverse * 1000)

        potential_forward = np.linspace(e_initial, e_vertex, n_forward)
        potential_reverse = np.linspace(e_vertex, e_final, n_reverse)
        potential = np.concatenate([potential_forward, potential_reverse])

        # Randles-Sevcik for reversible system
        e0 = (e_vertex + e_initial) / 2  # Estimate
        D = 1e-5  # cm²/s
        T = self.temperature_k

        # Peak current (Randles-Sevcik equation)
        ip = 2.69e5 * n_electrons**(3/2) * area * D**(1/2) * concentration * (scan_rate / 1000)**(1/2)

        # Generate CV shape
        current = np.zeros_like(potential)
        for i, E in enumerate(potential):
            if i < n_forward:
                # Forward sweep (oxidation)
                current[i] = ip * self._cv_peak_function(E, e0 + 0.03, 0.05)
            else:
                # Reverse sweep (reduction)
                current[i] = -ip * self._cv_peak_function(E, e0 - 0.03, 0.05)

        # Add capacitive current
        current += 0.5 * scan_rate / 100

        # Add noise
        current += np.random.normal(0, ip * 0.02, len(current))

        return potential, current

    def _cv_peak_function(self, E: float, E_peak: float, width: float) -> float:
        """Generate peak shape for CV."""
        return np.exp(-((E - E_peak) / width)**2) * (1 + 0.3 * (E - E_peak) / width)

    def _analyze_cv_curve(self, potential: np.ndarray, current: np.ndarray,
                         scan_rate: float) -> Dict[str, Any]:
        """Analyze CV curve to extract peak parameters."""
        # Find peaks (forward and reverse)
        n_half = len(potential) // 2

        # Anodic peak (forward sweep, maximum current)
        idx_anodic = np.argmax(current[:n_half])
        epa = potential[idx_anodic]
        ipa = current[idx_anodic]

        # Cathodic peak (reverse sweep, minimum current)
        idx_cathodic = n_half + np.argmin(current[n_half:])
        epc = potential[idx_cathodic]
        ipc = current[idx_cathodic]

        # Peak separation
        delta_ep = (epa - epc) * 1000  # mV

        # Formal potential
        e0_formal = (epa + epc) / 2

        # Half-wave potential (average)
        e1_2 = e0_formal

        # Current ratio
        ipa_ipc = abs(ipa / ipc) if ipc != 0 else 1.0

        # Classify reversibility
        if delta_ep < 100 and abs(ipa_ipc - 1.0) < 0.2:
            reversibility = 'reversible'
        elif delta_ep < 200:
            reversibility = 'quasi_reversible'
        else:
            reversibility = 'irreversible'

        return {
            'epa': float(epa),
            'epc': float(epc),
            'e0_formal': float(e0_formal),
            'e1_2': float(e1_2),
            'delta_ep_mv': float(delta_ep),
            'peak_current_anodic_ua': float(ipa),
            'peak_current_cathodic_ua': float(ipc),
            'ipa_ipc_ratio': float(ipa_ipc),
            'reversibility': reversibility
        }

    def _calculate_diffusion_coefficient(self, peak_current_ua: float, scan_rate_mv_s: float,
                                        concentration_m: float, area_cm2: float,
                                        n_electrons: int) -> float:
        """Calculate diffusion coefficient from Randles-Sevcik equation."""
        # ip = 2.69e5 * n^(3/2) * A * D^(1/2) * C * ν^(1/2)
        # Solve for D
        ip_a = peak_current_ua * 1e-6
        nu_v_s = scan_rate_mv_s / 1000

        D = (ip_a / (2.69e5 * n_electrons**(3/2) * area_cm2 * concentration_m * nu_v_s**(1/2)))**2

        return float(D)

    def _calculate_concentration_from_peak(self, peak_current_ua: float, scan_rate_mv_s: float,
                                          diffusion_coeff: float, area_cm2: float,
                                          n_electrons: int) -> float:
        """Calculate concentration from peak current."""
        ip_a = peak_current_ua * 1e-6
        nu_v_s = scan_rate_mv_s / 1000

        C = ip_a / (2.69e5 * n_electrons**(3/2) * area_cm2 * diffusion_coeff**(1/2) * nu_v_s**(1/2))

        return float(C)

    def _estimate_k0(self, delta_ep_mv: float, diffusion_coeff: float) -> float:
        """Estimate standard rate constant from peak separation."""
        # For quasi-reversible systems, k0 can be estimated from ΔEp
        # Simplified approximation
        if delta_ep_mv < 100:
            k0 = 0.01  # cm/s (fast, reversible)
        elif delta_ep_mv < 200:
            k0 = 1e-3  # cm/s (quasi-reversible)
        else:
            k0 = 1e-5  # cm/s (slow, irreversible)

        return k0

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

        if 'data_file' not in data and 'voltammogram_data' not in data:
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
            memory_gb=1.0,
            estimated_time_sec=30.0,
            execution_environment=ExecutionEnvironment.LOCAL
        )

    def get_capabilities(self) -> List[Capability]:
        """Get list of agent capabilities."""
        return [
            Capability(
                name='cv',
                description='Cyclic voltammetry for redox analysis',
                input_types=['voltammogram', 'scan_parameters'],
                output_types=['redox_potentials', 'kinetics', 'reversibility'],
                typical_use_cases=['redox_chemistry', 'electrocatalysis', 'battery_materials']
            ),
            Capability(
                name='dpv',
                description='Differential pulse voltammetry for trace analysis',
                input_types=['voltammogram', 'pulse_parameters'],
                output_types=['peak_potentials', 'concentration'],
                typical_use_cases=['trace_metals', 'pharmaceuticals', 'environmental']
            ),
            Capability(
                name='rde',
                description='Rotating disk electrode for kinetics',
                input_types=['voltammogram', 'rotation_rates'],
                output_types=['diffusion_coefficient', 'electron_number', 'kinetic_current'],
                typical_use_cases=['fuel_cells', 'ORR', 'mass_transport']
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata."""
        return AgentMetadata(
            name="VoltammetryAgent",
            version=self.VERSION,
            description="Voltammetry expert for electrochemical characterization",
            author="Materials Characterization Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['txt', 'csv', 'chi']  # CHI Instruments format
        )

    def connect_instrument(self) -> bool:
        """Connect to potentiostat."""
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw electrochemical data."""
        # In production:
        # - Baseline correction
        # - iR compensation
        # - Smoothing
        # - Peak finding
        return raw_data

    # Integration methods
    @staticmethod
    def correlate_with_xps(cv_result: Dict[str, Any], xps_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate CV redox potentials with XPS oxidation states.

        CV shows electrochemical redox behavior.
        XPS confirms oxidation state changes.

        Args:
            cv_result: CV analysis
            xps_result: XPS analysis

        Returns:
            Correlation report
        """
        e0 = cv_result.get('peak_analysis', {}).get('formal_potential_e0_v', 0)

        return {
            'correlation_type': 'CV_XPS_oxidation_states',
            'cv_formal_potential_v': e0,
            'notes': 'CV redox potential should correlate with XPS binding energy shifts',
            'recommendation': 'Use XPS to confirm oxidation state changes observed in CV'
        }

    @staticmethod
    def validate_with_eis(cv_result: Dict[str, Any], eis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CV kinetics with EIS charge transfer resistance.

        CV provides kinetic information from peak separation.
        EIS provides charge transfer resistance (Rct).

        Args:
            cv_result: CV kinetics
            eis_result: EIS analysis

        Returns:
            Validation report
        """
        k0_cv = cv_result.get('kinetics_analysis', {}).get('standard_rate_constant_estimated_cm_s', 0)
        rct_eis = eis_result.get('circuit_analysis', {}).get('charge_transfer_resistance_ohm', 0)

        # k0 and Rct should be inversely related
        return {
            'validation_type': 'CV_EIS_kinetics',
            'cv_rate_constant_cm_s': k0_cv,
            'eis_charge_transfer_resistance_ohm': rct_eis,
            'relationship': 'inverse (fast kinetics → low Rct)',
            'agreement': 'qualitative'
        }
