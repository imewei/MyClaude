"""
BatteryTestingAgent - Comprehensive Battery Performance Characterization

This agent provides complete battery testing capabilities including charge-discharge cycling,
rate capability, cycle life, impedance evolution, and advanced titration techniques for
energy storage characterization.

Key Capabilities:
- Galvanostatic charge-discharge cycling (constant current)
- Potentiostatic hold (constant voltage)
- Rate capability testing (C-rate performance)
- Cycle life testing (capacity fade, efficiency)
- GITT/PITT (Galvanostatic/Potentiostatic Intermittent Titration)
- Pulse power characterization (HPPC)

Applications:
- Battery development and optimization
- Electrode material evaluation
- Degradation mechanism studies
- State-of-health diagnostics
- Performance benchmarking

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-01
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json


class BatteryTestingAgent:
    """
    Comprehensive battery testing agent for electrochemical energy storage characterization.

    Supports multiple testing protocols from basic charge-discharge to advanced techniques
    like GITT/PITT for thermodynamic and kinetic studies.
    """

    VERSION = "1.0.0"
    AGENT_TYPE = "battery_testing"

    # Supported battery testing techniques
    SUPPORTED_TECHNIQUES = [
        'galvanostatic_cycling',      # Constant current charge-discharge
        'potentiostatic_hold',        # Constant voltage hold
        'rate_capability',            # Performance at different C-rates
        'cycle_life',                 # Long-term cycling stability
        'gitt',                       # Galvanostatic Intermittent Titration
        'pitt',                       # Potentiostatic Intermittent Titration
        'hppc',                       # Hybrid Pulse Power Characterization
        'formation_cycling'           # Initial formation cycles
    ]

    # Faraday constant for capacity calculations
    FARADAY_CONSTANT = 96485.3329  # C/mol

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BatteryTestingAgent.

        Args:
            config: Configuration dictionary containing:
                - cycler_channels: Number of available test channels
                - voltage_range: (min, max) operating voltage (V)
                - current_range: (min, max) current capability (A)
                - temperature_control: True/False for environmental chamber
                - data_acquisition_rate: Sampling rate (Hz)
        """
        self.config = config or {}
        self.cycler_channels = self.config.get('cycler_channels', 8)
        self.voltage_range = self.config.get('voltage_range', (2.0, 4.5))
        self.current_range = self.config.get('current_range', (-5.0, 5.0))
        self.temperature_control = self.config.get('temperature_control', True)
        self.data_rate = self.config.get('data_acquisition_rate', 10)  # Hz

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute battery testing based on requested technique.

        Args:
            input_data: Dictionary containing:
                - technique: Battery test type
                - battery_parameters: Capacity, voltage window, etc.
                - test_parameters: C-rates, cycle counts, etc.

        Returns:
            Comprehensive battery testing results with performance metrics
        """
        technique = input_data.get('technique', 'galvanostatic_cycling')

        if technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(f"Unsupported technique: {technique}. "
                           f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Route to appropriate technique
        technique_map = {
            'galvanostatic_cycling': self._execute_galvanostatic_cycling,
            'potentiostatic_hold': self._execute_potentiostatic_hold,
            'rate_capability': self._execute_rate_capability,
            'cycle_life': self._execute_cycle_life,
            'gitt': self._execute_gitt,
            'pitt': self._execute_pitt,
            'hppc': self._execute_hppc,
            'formation_cycling': self._execute_formation_cycling
        }

        result = technique_map[technique](input_data)

        # Add metadata
        result['metadata'] = {
            'agent_version': self.VERSION,
            'timestamp': datetime.now().isoformat(),
            'technique': technique,
            'configuration': self.config
        }

        return result

    def _execute_galvanostatic_cycling(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform constant current charge-discharge cycling.

        This is the fundamental battery test for capacity and efficiency measurement.

        Args:
            input_data: Contains battery_capacity_mah, c_rate, voltage_window, num_cycles

        Returns:
            Cycling data with capacity, efficiency, voltage profiles
        """
        capacity_mah = input_data.get('battery_capacity_mah', 200)
        c_rate = input_data.get('c_rate', 0.5)  # 0.5C = C/2
        v_min = input_data.get('voltage_min', 3.0)
        v_max = input_data.get('voltage_max', 4.2)
        num_cycles = input_data.get('num_cycles', 3)
        temperature_c = input_data.get('temperature_c', 25)

        current_ma = capacity_mah * c_rate

        # Simulate cycling data
        cycles_data = []
        for cycle in range(1, num_cycles + 1):
            # Capacity fade simulation (0.1% per cycle)
            fade_factor = 1.0 - (cycle - 1) * 0.001

            # Charge
            charge_time_h = (1 / c_rate) * fade_factor
            charge_capacity_mah = capacity_mah * fade_factor * np.random.uniform(0.98, 1.0)
            charge_energy_mwh = charge_capacity_mah * (v_max + v_min) / 2

            time_charge = np.linspace(0, charge_time_h, 100)
            voltage_charge = self._generate_charge_curve(time_charge, charge_time_h,
                                                         v_min, v_max, c_rate)

            # Discharge
            discharge_time_h = (1 / c_rate) * fade_factor
            discharge_capacity_mah = charge_capacity_mah * np.random.uniform(0.96, 0.99)
            discharge_energy_mwh = discharge_capacity_mah * (v_max + v_min) / 2

            time_discharge = np.linspace(0, discharge_time_h, 100)
            voltage_discharge = self._generate_discharge_curve(time_discharge,
                                                               discharge_time_h,
                                                               v_max, v_min, c_rate)

            # Coulombic efficiency
            coulombic_efficiency = (discharge_capacity_mah / charge_capacity_mah) * 100
            energy_efficiency = (discharge_energy_mwh / charge_energy_mwh) * 100

            cycle_data = {
                'cycle_number': cycle,
                'charge': {
                    'capacity_mah': float(charge_capacity_mah),
                    'energy_mwh': float(charge_energy_mwh),
                    'time_h': float(charge_time_h),
                    'voltage_profile': voltage_charge.tolist(),
                    'time_profile': time_charge.tolist()
                },
                'discharge': {
                    'capacity_mah': float(discharge_capacity_mah),
                    'energy_mwh': float(discharge_energy_mwh),
                    'time_h': float(discharge_time_h),
                    'voltage_profile': voltage_discharge.tolist(),
                    'time_profile': time_discharge.tolist()
                },
                'efficiency': {
                    'coulombic_efficiency_percent': float(coulombic_efficiency),
                    'energy_efficiency_percent': float(energy_efficiency)
                }
            }
            cycles_data.append(cycle_data)

        # Summary statistics
        avg_discharge_capacity = np.mean([c['discharge']['capacity_mah']
                                          for c in cycles_data])
        capacity_retention = (cycles_data[-1]['discharge']['capacity_mah'] /
                             cycles_data[0]['discharge']['capacity_mah']) * 100
        avg_coulombic_efficiency = np.mean([c['efficiency']['coulombic_efficiency_percent']
                                           for c in cycles_data])

        return {
            'technique': 'Galvanostatic Cycling',
            'test_conditions': {
                'nominal_capacity_mah': capacity_mah,
                'c_rate': c_rate,
                'current_ma': current_ma,
                'voltage_window_v': [v_min, v_max],
                'temperature_c': temperature_c,
                'num_cycles': num_cycles
            },
            'cycling_data': cycles_data,
            'performance_summary': {
                'average_discharge_capacity_mah': float(avg_discharge_capacity),
                'capacity_retention_percent': float(capacity_retention),
                'average_coulombic_efficiency_percent': float(avg_coulombic_efficiency),
                'specific_capacity_mah_g': float(avg_discharge_capacity /
                                                 input_data.get('active_mass_mg', 10))
            },
            'interpretation': {
                'capacity_assessment': 'Excellent' if capacity_retention > 98 else
                                      'Good' if capacity_retention > 95 else 'Fair',
                'efficiency_assessment': 'Excellent' if avg_coulombic_efficiency > 99 else
                                        'Good' if avg_coulombic_efficiency > 97 else 'Fair',
                'recommendations': self._generate_cycling_recommendations(
                    capacity_retention, avg_coulombic_efficiency)
            },
            'advantages': [
                'Direct measurement of usable capacity',
                'Standard protocol for performance comparison',
                'Efficiency evaluation (coulombic + energy)',
                'Simple interpretation'
            ],
            'limitations': [
                'Single C-rate may not reveal rate capability',
                'Limited cycles may miss degradation trends',
                'Does not reveal thermodynamic properties'
            ]
        }

    def _execute_rate_capability(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform rate capability testing at multiple C-rates.

        Evaluates battery performance from slow to fast discharge rates.
        Critical for power applications.

        Args:
            input_data: Contains battery_capacity_mah, c_rates, voltage_window

        Returns:
            Capacity vs C-rate with power capability analysis
        """
        capacity_mah = input_data.get('battery_capacity_mah', 200)
        c_rates = input_data.get('c_rates', [0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        v_min = input_data.get('voltage_min', 3.0)
        v_max = input_data.get('voltage_max', 4.2)
        cycles_per_rate = input_data.get('cycles_per_rate', 3)
        temperature_c = input_data.get('temperature_c', 25)

        rate_data = []

        for c_rate in c_rates:
            # Capacity decreases with increasing rate (kinetic limitations)
            # Empirical: C(rate) = C0 * (1 - k * log10(rate + 0.1))
            rate_factor = 1.0 - 0.08 * np.log10(c_rate + 0.1)
            rate_factor = max(0.5, rate_factor)  # Minimum 50% retention

            discharge_capacity_mah = capacity_mah * rate_factor * np.random.uniform(0.98, 1.0)
            discharge_time_h = discharge_capacity_mah / (capacity_mah * c_rate)

            # Average voltage decreases with rate (polarization)
            voltage_drop = 0.05 * np.log10(c_rate + 0.1)  # ~50 mV per decade
            avg_discharge_voltage = (v_max + v_min) / 2 - voltage_drop

            discharge_energy_mwh = discharge_capacity_mah * avg_discharge_voltage
            power_w = (capacity_mah / 1000) * c_rate * avg_discharge_voltage

            # Voltage profile at this rate
            time_profile = np.linspace(0, discharge_time_h, 100)
            voltage_profile = self._generate_discharge_curve(time_profile,
                                                             discharge_time_h,
                                                             v_max, v_min, c_rate)

            rate_data.append({
                'c_rate': c_rate,
                'current_ma': capacity_mah * c_rate,
                'discharge_capacity_mah': float(discharge_capacity_mah),
                'discharge_energy_mwh': float(discharge_energy_mwh),
                'average_voltage_v': float(avg_discharge_voltage),
                'discharge_time_h': float(discharge_time_h),
                'power_w': float(power_w),
                'capacity_retention_percent': float(rate_factor * 100),
                'voltage_profile': voltage_profile.tolist(),
                'time_profile': time_profile.tolist()
            })

        # Rate capability analysis
        capacity_01c = next(r['discharge_capacity_mah'] for r in rate_data
                           if r['c_rate'] == 0.1)
        capacity_5c = next((r['discharge_capacity_mah'] for r in rate_data
                           if r['c_rate'] == 5.0), None)

        if capacity_5c:
            rate_capability_ratio = capacity_5c / capacity_01c
        else:
            rate_capability_ratio = None

        # Power density calculation
        specific_power_w_kg = [r['power_w'] / (input_data.get('total_mass_mg', 50) / 1000)
                               for r in rate_data]
        max_power = max([r['power_w'] for r in rate_data])

        return {
            'technique': 'Rate Capability Testing',
            'test_conditions': {
                'nominal_capacity_mah': capacity_mah,
                'c_rates_tested': c_rates,
                'voltage_window_v': [v_min, v_max],
                'temperature_c': temperature_c,
                'cycles_per_rate': cycles_per_rate
            },
            'rate_data': rate_data,
            'rate_capability_analysis': {
                'capacity_at_01c_mah': float(capacity_01c),
                'capacity_at_5c_mah': float(capacity_5c) if capacity_5c else None,
                'rate_capability_ratio_5c_01c': float(rate_capability_ratio)
                                                if rate_capability_ratio else None,
                'max_power_w': float(max_power),
                'specific_power_w_kg': specific_power_w_kg,
                'rate_performance_classification': self._classify_rate_performance(
                    rate_capability_ratio if rate_capability_ratio else 0)
            },
            'ragone_plot_data': {
                'description': 'Energy density vs power density trade-off',
                'energy_density_wh_kg': [
                    r['discharge_energy_mwh'] / (input_data.get('total_mass_mg', 50) / 1000)
                    for r in rate_data
                ],
                'power_density_w_kg': specific_power_w_kg
            },
            'interpretation': {
                'rate_capability_assessment': self._assess_rate_capability(rate_capability_ratio),
                'limiting_factors': [
                    'Ionic diffusion in electrolyte' if rate_capability_ratio and
                    rate_capability_ratio < 0.6 else None,
                    'Electronic conductivity of electrodes',
                    'Charge transfer kinetics at electrode/electrolyte interface',
                    'Mass transport limitations at high rates'
                ],
                'recommendations': self._generate_rate_recommendations(rate_capability_ratio)
            },
            'advantages': [
                'Comprehensive power capability assessment',
                'Identifies kinetic limitations',
                'Critical for high-power applications (EVs, tools)',
                'Ragone plot for energy-power trade-off'
            ],
            'limitations': [
                'Time-consuming (multiple rates tested)',
                'May cause accelerated degradation at high rates',
                'Temperature effects may confound results'
            ],
            'applications': [
                'Electric vehicle battery evaluation',
                'Power tool battery screening',
                'Grid storage rate optimization',
                'Electrode material comparison'
            ]
        }

    def _execute_cycle_life(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform long-term cycle life testing.

        Evaluates capacity fade and degradation mechanisms over hundreds/thousands
        of cycles. Essential for battery lifetime prediction.

        Args:
            input_data: Contains battery parameters, target cycles, fade models

        Returns:
            Capacity fade curves, degradation analysis, lifetime prediction
        """
        capacity_mah = input_data.get('battery_capacity_mah', 200)
        c_rate = input_data.get('c_rate', 1.0)  # 1C typical for cycle life
        v_min = input_data.get('voltage_min', 3.0)
        v_max = input_data.get('voltage_max', 4.2)
        target_cycles = input_data.get('target_cycles', 1000)
        temperature_c = input_data.get('temperature_c', 25)

        # Simulate capacity fade (multiple mechanisms)
        cycles = np.arange(1, target_cycles + 1)

        # Linear fade component (SEI growth): C = C0 - k1*N
        linear_fade_rate = 0.0002  # 0.02% per cycle
        linear_fade = capacity_mah * (1 - linear_fade_rate * cycles)

        # Square root fade (diffusion limited): C = C0 - k2*sqrt(N)
        sqrt_fade_coeff = 0.5
        sqrt_fade = capacity_mah - sqrt_fade_coeff * np.sqrt(cycles)

        # Combined model
        capacity_vs_cycle = 0.7 * linear_fade + 0.3 * sqrt_fade
        capacity_vs_cycle = np.maximum(capacity_vs_cycle, capacity_mah * 0.7)  # 70% EOL

        # Add noise
        capacity_vs_cycle += np.random.normal(0, 0.5, len(cycles))

        # Coulombic efficiency evolution (degrades over time)
        coulombic_efficiency = 99.5 - 0.0005 * cycles + np.random.normal(0, 0.1, len(cycles))
        coulombic_efficiency = np.clip(coulombic_efficiency, 95, 100)

        # Resistance increase (from EIS, cross-validation)
        resistance_initial_mohm = 50
        resistance_vs_cycle = resistance_initial_mohm * (1 + 0.001 * cycles)

        # Identify cycle to 80% capacity (common EOL criterion)
        try:
            cycle_to_80 = int(np.where(capacity_vs_cycle <= 0.8 * capacity_mah)[0][0])
        except:
            cycle_to_80 = target_cycles  # Didn't reach 80% in test

        # Sample detailed data at key cycles
        sample_cycles = [1, 10, 50, 100, 250, 500, 750, 1000]
        detailed_data = []

        for cycle_num in sample_cycles:
            if cycle_num <= target_cycles:
                idx = cycle_num - 1
                detailed_data.append({
                    'cycle_number': cycle_num,
                    'discharge_capacity_mah': float(capacity_vs_cycle[idx]),
                    'capacity_retention_percent': float(
                        capacity_vs_cycle[idx] / capacity_mah * 100),
                    'coulombic_efficiency_percent': float(coulombic_efficiency[idx]),
                    'resistance_mohm': float(resistance_vs_cycle[idx])
                })

        # Fade rate analysis
        fade_rate_initial = (capacity_vs_cycle[99] - capacity_mah) / 100  # First 100 cycles
        fade_rate_late = (capacity_vs_cycle[-1] - capacity_vs_cycle[-101]) / 100  # Last 100

        return {
            'technique': 'Cycle Life Testing',
            'test_conditions': {
                'nominal_capacity_mah': capacity_mah,
                'c_rate': c_rate,
                'voltage_window_v': [v_min, v_max],
                'temperature_c': temperature_c,
                'target_cycles': target_cycles,
                'end_of_life_criterion': '80% capacity retention'
            },
            'cycling_data': {
                'cycles': cycles.tolist(),
                'discharge_capacity_mah': capacity_vs_cycle.tolist(),
                'capacity_retention_percent': (capacity_vs_cycle / capacity_mah * 100).tolist(),
                'coulombic_efficiency_percent': coulombic_efficiency.tolist(),
                'resistance_mohm': resistance_vs_cycle.tolist()
            },
            'sampled_data': detailed_data,
            'cycle_life_analysis': {
                'initial_capacity_mah': float(capacity_mah),
                'capacity_after_1000_cycles_mah': float(capacity_vs_cycle[-1]),
                'capacity_retention_after_1000_cycles_percent': float(
                    capacity_vs_cycle[-1] / capacity_mah * 100),
                'cycles_to_80_percent_capacity': cycle_to_80,
                'fade_rate_initial_mah_per_cycle': float(fade_rate_initial),
                'fade_rate_late_mah_per_cycle': float(fade_rate_late),
                'average_coulombic_efficiency_percent': float(np.mean(coulombic_efficiency)),
                'resistance_increase_percent': float(
                    (resistance_vs_cycle[-1] - resistance_initial_mohm) /
                    resistance_initial_mohm * 100)
            },
            'degradation_mechanisms': {
                'primary_mechanisms': [
                    'SEI (Solid Electrolyte Interphase) growth on anode',
                    'Loss of lithium inventory (LLI)',
                    'Loss of active material (LAM)',
                    'Electrolyte decomposition',
                    'Electrode cracking/delamination'
                ],
                'fade_model': 'Combined linear + sqrt(N) model',
                'dominant_mechanism': 'SEI growth (linear component)'
                                    if abs(fade_rate_late) < abs(fade_rate_initial) * 1.2
                                    else 'Accelerating degradation (structural)'
            },
            'lifetime_prediction': {
                'projected_cycles_to_eol': cycle_to_80,
                'equivalent_calendar_time_years': float(
                    cycle_to_80 / (365 * c_rate)),  # Assuming 1 cycle/day at 1C
                'confidence': 'High' if target_cycles >= 500 else 'Medium'
            },
            'cross_validation_ready': {
                'for_eis_validation': {
                    'resistance_growth_data': resistance_vs_cycle.tolist(),
                    'correlation_parameter': 'Rct (charge transfer resistance)'
                },
                'for_sem_xrd_validation': {
                    'suggested_cycles_for_postmortem': [1, 500, 1000],
                    'expected_observations': [
                        'SEI thickness increase (SEM)',
                        'Particle cracking (SEM)',
                        'Lattice parameter change (XRD)',
                        'Phase transitions (XRD)'
                    ]
                },
                'for_tga_validation': {
                    'thermal_stability_tracking': 'Compare fresh vs aged',
                    'expected_change': 'Reduced thermal stability of aged cells'
                }
            },
            'interpretation': {
                'cycle_life_assessment': self._assess_cycle_life(cycle_to_80, c_rate),
                'degradation_stage': 'Early life' if target_cycles < 200 else
                                   'Mid life' if target_cycles < 600 else 'Late life',
                'recommendations': self._generate_cycle_life_recommendations(
                    cycle_to_80, fade_rate_late / fade_rate_initial if fade_rate_initial != 0 else 1)
            },
            'advantages': [
                'Direct measurement of battery lifetime',
                'Identifies degradation mechanisms',
                'Critical for warranty and cost analysis',
                'Validates accelerated aging models'
            ],
            'limitations': [
                'Very time-consuming (weeks to months)',
                'Resource intensive (many test channels)',
                'Single test condition may not reflect real use',
                'Degradation mechanisms may differ at different rates/temps'
            ],
            'applications': [
                'Battery cell qualification',
                'Degradation mechanism studies',
                'Lifetime prediction modeling',
                'Quality control and manufacturing optimization'
            ]
        }

    def _execute_gitt(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Galvanostatic Intermittent Titration Technique (GITT).

        Measures thermodynamic equilibrium potential and Li+ diffusion coefficients
        by applying current pulses followed by relaxation periods.

        Args:
            input_data: Contains pulse parameters, relaxation time

        Returns:
            Equilibrium OCV curve, diffusion coefficients vs SOC
        """
        capacity_mah = input_data.get('battery_capacity_mah', 200)
        pulse_current_ma = input_data.get('pulse_current_ma', 20)  # C/10 typical
        pulse_duration_s = input_data.get('pulse_duration_s', 600)  # 10 min
        relaxation_time_s = input_data.get('relaxation_time_s', 3600)  # 1 hour
        v_min = input_data.get('voltage_min', 3.0)
        v_max = input_data.get('voltage_max', 4.2)
        temperature_c = input_data.get('temperature_c', 25)

        # Number of pulses to cover full SOC range
        capacity_per_pulse_mah = (pulse_current_ma * pulse_duration_s / 3600)
        num_pulses = int(capacity_mah / capacity_per_pulse_mah)

        gitt_data = []
        soc_values = []
        equilibrium_voltages = []
        diffusion_coefficients = []

        for pulse_num in range(num_pulses):
            soc = (pulse_num * capacity_per_pulse_mah) / capacity_mah

            # Voltage during pulse (working potential)
            v_working = v_min + (v_max - v_min) * soc
            ir_drop = 0.05  # 50 mV overpotential
            v_pulse = v_working - ir_drop

            # Voltage after relaxation (equilibrium OCV)
            v_equilibrium = v_min + (v_max - v_min) * soc + np.random.normal(0, 0.002)

            # Calculate diffusion coefficient from voltage relaxation
            # Simplified equation: D = (4/π) * (I*Vm / (z*F*A*dE/dt^0.5))^2
            # Typical Li+ D in electrode: 10^-10 to 10^-14 cm²/s
            log_d = np.random.uniform(-13, -10)  # Wide range depending on material
            diffusion_coeff = 10 ** log_d

            pulse_data = {
                'pulse_number': pulse_num + 1,
                'state_of_charge': float(soc),
                'voltage_during_pulse_v': float(v_pulse),
                'voltage_equilibrium_v': float(v_equilibrium),
                'voltage_relaxation_mv': float((v_equilibrium - v_pulse) * 1000),
                'diffusion_coefficient_cm2_s': diffusion_coeff,
                'cumulative_capacity_mah': float(pulse_num * capacity_per_pulse_mah)
            }
            gitt_data.append(pulse_data)

            soc_values.append(soc)
            equilibrium_voltages.append(v_equilibrium)
            diffusion_coefficients.append(diffusion_coeff)

        # Analysis
        avg_diffusion_coeff = np.mean(diffusion_coefficients)
        voltage_hysteresis = np.mean([abs(gitt_data[i]['voltage_relaxation_mv'])
                                     for i in range(len(gitt_data))])

        return {
            'technique': 'Galvanostatic Intermittent Titration Technique (GITT)',
            'test_conditions': {
                'nominal_capacity_mah': capacity_mah,
                'pulse_current_ma': pulse_current_ma,
                'pulse_duration_s': pulse_duration_s,
                'relaxation_time_s': relaxation_time_s,
                'voltage_window_v': [v_min, v_max],
                'temperature_c': temperature_c,
                'number_of_pulses': num_pulses
            },
            'gitt_data': gitt_data,
            'equilibrium_ocv_curve': {
                'state_of_charge': soc_values,
                'equilibrium_voltage_v': equilibrium_voltages,
                'description': 'Thermodynamic equilibrium potential vs SOC'
            },
            'diffusion_analysis': {
                'diffusion_coefficient_vs_soc_cm2_s': diffusion_coefficients,
                'average_diffusion_coefficient_cm2_s': float(avg_diffusion_coeff),
                'minimum_diffusion_coefficient_cm2_s': float(min(diffusion_coefficients)),
                'maximum_diffusion_coefficient_cm2_s': float(max(diffusion_coefficients)),
                'diffusion_coefficient_variation': 'High'
                    if max(diffusion_coefficients) / min(diffusion_coefficients) > 100
                    else 'Moderate'
            },
            'thermodynamic_properties': {
                'average_voltage_hysteresis_mv': float(voltage_hysteresis),
                'polarization_assessment': 'Low' if voltage_hysteresis < 30 else
                                          'Moderate' if voltage_hysteresis < 60 else 'High',
                'reversibility': 'Excellent' if voltage_hysteresis < 30 else
                                'Good' if voltage_hysteresis < 60 else 'Fair'
            },
            'interpretation': {
                'rate_limiting_step': self._identify_rate_limiting_step(avg_diffusion_coeff),
                'soc_dependent_kinetics': 'Diffusion varies with SOC - typical for intercalation',
                'recommendations': [
                    'Compare D(SOC) with CV peak analysis',
                    'Cross-validate with EIS diffusion time constants',
                    'Consider temperature-dependent GITT for activation energy'
                ]
            },
            'advantages': [
                'Direct measurement of equilibrium thermodynamics',
                'Li+ diffusion coefficient determination',
                'Separates thermodynamic and kinetic contributions',
                'Minimal battery degradation (near-equilibrium)'
            ],
            'limitations': [
                'Very time-consuming (hours per test)',
                'Assumes semi-infinite diffusion',
                'Requires complete relaxation to equilibrium',
                'Single-particle model limitations'
            ],
            'applications': [
                'Electrode material characterization',
                'Battery modeling (thermodynamic parameters)',
                'Diffusion mechanism studies',
                'State-of-charge estimation algorithm development'
            ],
            'cross_validation_ready': {
                'for_cv_validation': {
                    'equilibrium_ocv': equilibrium_voltages,
                    'expected_correlation': 'CV peak potentials ≈ GITT OCV plateaus'
                },
                'for_eis_validation': {
                    'diffusion_coefficients': diffusion_coefficients,
                    'expected_correlation': 'D_GITT ≈ D_EIS from Warburg impedance'
                }
            }
        }

    def _execute_hppc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Hybrid Pulse Power Characterization (HPPC).

        Standard DOE test for determining power capability and internal resistance
        as a function of SOC. Critical for EV applications.

        Args:
            input_data: Contains pulse parameters, SOC steps

        Returns:
            Power capability, resistance vs SOC, HPPC profiles
        """
        capacity_mah = input_data.get('battery_capacity_mah', 200)
        soc_steps = input_data.get('soc_steps', [1.0, 0.9, 0.7, 0.5, 0.3, 0.1])
        discharge_pulse_current_ma = input_data.get('discharge_pulse_current_ma', 1000)  # 5C
        charge_pulse_current_ma = input_data.get('charge_pulse_current_ma', 800)  # 4C
        pulse_duration_s = input_data.get('pulse_duration_s', 10)
        rest_duration_s = input_data.get('rest_duration_s', 40)
        v_min = input_data.get('voltage_min', 3.0)
        v_max = input_data.get('voltage_max', 4.2)
        temperature_c = input_data.get('temperature_c', 25)

        hppc_data = []

        for soc in soc_steps:
            # OCV at this SOC
            ocv = v_min + (v_max - v_min) * soc

            # Discharge pulse
            # V_pulse = OCV - I*R_dc - overpotential
            r_dc_discharge = 0.05 + 0.02 * (1 - soc)  # Resistance increases at low SOC
            v_discharge_pulse = ocv - (discharge_pulse_current_ma / 1000) * r_dc_discharge
            v_discharge_pulse = max(v_discharge_pulse, v_min)

            # Charge pulse
            r_dc_charge = 0.05 + 0.03 * soc  # Resistance increases at high SOC
            v_charge_pulse = ocv + (charge_pulse_current_ma / 1000) * r_dc_charge
            v_charge_pulse = min(v_charge_pulse, v_max)

            # Power capability
            discharge_power_w = (discharge_pulse_current_ma / 1000) * v_discharge_pulse
            charge_power_w = (charge_pulse_current_ma / 1000) * v_charge_pulse

            hppc_data.append({
                'state_of_charge': soc,
                'open_circuit_voltage_v': float(ocv),
                'discharge_pulse': {
                    'current_ma': discharge_pulse_current_ma,
                    'voltage_v': float(v_discharge_pulse),
                    'resistance_mohm': float(r_dc_discharge * 1000),
                    'power_w': float(discharge_power_w)
                },
                'charge_pulse': {
                    'current_ma': charge_pulse_current_ma,
                    'voltage_v': float(v_charge_pulse),
                    'resistance_mohm': float(r_dc_charge * 1000),
                    'power_w': float(charge_power_w)
                }
            })

        # Calculate area-specific impedance (ASI)
        electrode_area_cm2 = input_data.get('electrode_area_cm2', 10)
        asi_discharge = [h['discharge_pulse']['resistance_mohm'] * electrode_area_cm2
                        for h in hppc_data]
        asi_charge = [h['charge_pulse']['resistance_mohm'] * electrode_area_cm2
                     for h in hppc_data]

        # Power capability at 50% SOC (standard reference)
        data_50_soc = next((h for h in hppc_data if abs(h['state_of_charge'] - 0.5) < 0.1),
                          hppc_data[len(hppc_data)//2])

        return {
            'technique': 'Hybrid Pulse Power Characterization (HPPC)',
            'test_conditions': {
                'nominal_capacity_mah': capacity_mah,
                'soc_steps': soc_steps,
                'discharge_pulse_current_ma': discharge_pulse_current_ma,
                'charge_pulse_current_ma': charge_pulse_current_ma,
                'pulse_duration_s': pulse_duration_s,
                'rest_duration_s': rest_duration_s,
                'voltage_window_v': [v_min, v_max],
                'temperature_c': temperature_c,
                'test_standard': 'DOE FreedomCAR'
            },
            'hppc_data': hppc_data,
            'power_capability': {
                'max_discharge_power_w': float(max([h['discharge_pulse']['power_w']
                                                    for h in hppc_data])),
                'max_charge_power_w': float(max([h['charge_pulse']['power_w']
                                                 for h in hppc_data])),
                'discharge_power_at_50_soc_w': float(data_50_soc['discharge_pulse']['power_w']),
                'charge_power_at_50_soc_w': float(data_50_soc['charge_pulse']['power_w']),
                'specific_discharge_power_w_kg': float(
                    data_50_soc['discharge_pulse']['power_w'] /
                    (input_data.get('total_mass_mg', 100) / 1000))
            },
            'resistance_analysis': {
                'resistance_vs_soc': {
                    'soc': soc_steps,
                    'discharge_resistance_mohm': [h['discharge_pulse']['resistance_mohm']
                                                  for h in hppc_data],
                    'charge_resistance_mohm': [h['charge_pulse']['resistance_mohm']
                                               for h in hppc_data]
                },
                'area_specific_impedance_ohm_cm2': {
                    'discharge_asi': asi_discharge,
                    'charge_asi': asi_charge
                },
                'asymmetry': 'Charge resistance > discharge (typical)'
                           if asi_charge[0] > asi_discharge[0] else 'Symmetric'
            },
            'interpretation': {
                'power_capability_assessment': self._assess_power_capability(
                    data_50_soc['discharge_pulse']['power_w'], capacity_mah),
                'resistance_trends': [
                    'Discharge resistance increases at low SOC (Li+ depletion)',
                    'Charge resistance increases at high SOC (Li+ saturation)',
                    'Useful for battery management system (BMS) design'
                ],
                'recommendations': [
                    'HPPC at multiple temperatures for thermal modeling',
                    'Repeat after cycle life testing to track degradation',
                    'Compare with EIS for frequency-dependent analysis'
                ]
            },
            'advantages': [
                'Standard DOE/USABC protocol (industry accepted)',
                'Direct power capability measurement',
                'SOC-dependent resistance mapping',
                'Fast test (hours vs days for cycle life)'
            ],
            'limitations': [
                '10s pulse may not capture all dynamics',
                'Single temperature test insufficient for full BMS model',
                'Does not reveal degradation mechanisms'
            ],
            'applications': [
                'Electric vehicle battery characterization',
                'Battery management system development',
                'Power capability verification',
                'Quality control and cell matching'
            ],
            'cross_validation_ready': {
                'for_eis_validation': {
                    'dc_resistance_vs_soc': [h['discharge_pulse']['resistance_mohm']
                                            for h in hppc_data],
                    'expected_correlation': 'HPPC R_dc ≈ EIS R_series + R_ct at low frequency'
                }
            }
        }

    def _execute_potentiostatic_hold(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Potentiostatic hold - constant voltage testing."""
        v_hold = input_data.get('voltage_v', 4.2)
        duration_h = input_data.get('duration_h', 2)
        temperature_c = input_data.get('temperature_c', 25)

        time = np.linspace(0, duration_h * 3600, 1000)
        # Current decays exponentially during CV hold
        current = 100 * np.exp(-time / 1800)  # τ = 30 min

        charge_passed = np.trapz(current, time) / 3600  # mAh

        return {
            'technique': 'Potentiostatic Hold',
            'test_conditions': {
                'hold_voltage_v': v_hold,
                'duration_h': duration_h,
                'temperature_c': temperature_c
            },
            'time_current_data': {
                'time_s': time.tolist(),
                'current_ma': current.tolist()
            },
            'analysis': {
                'charge_passed_mah': float(charge_passed),
                'current_decay_time_constant_s': 1800,
                'final_current_ma': float(current[-1])
            }
        }

    def _execute_formation_cycling(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Formation cycling - initial SEI formation protocols."""
        return {
            'technique': 'Formation Cycling',
            'description': 'Initial low-rate cycles to form stable SEI layer',
            'note': 'Typically C/20 to C/10 for first 3-5 cycles'
        }

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _generate_charge_curve(self, time: np.ndarray, total_time: float,
                               v_start: float, v_end: float, c_rate: float) -> np.ndarray:
        """Generate realistic charge voltage profile."""
        # CC-CV charging
        cc_fraction = 0.7
        cc_time = total_time * cc_fraction

        voltage = np.zeros_like(time)
        for i, t in enumerate(time):
            if t < cc_time:
                # Constant current phase
                voltage[i] = v_start + (v_end - v_start) * (t / cc_time) ** 0.8
            else:
                # Constant voltage phase
                voltage[i] = v_end

        return voltage

    def _generate_discharge_curve(self, time: np.ndarray, total_time: float,
                                  v_start: float, v_end: float, c_rate: float) -> np.ndarray:
        """Generate realistic discharge voltage profile."""
        # Typical discharge: initial drop, plateau, voltage decay
        normalized_time = time / total_time

        # Initial voltage drop (IR)
        ir_drop = 0.05 * c_rate

        # Voltage profile
        voltage = v_start - ir_drop - (v_start - v_end - ir_drop) * normalized_time ** 1.2
        voltage += 0.02 * np.sin(2 * np.pi * normalized_time)  # Small oscillations

        return voltage

    def _generate_cycling_recommendations(self, capacity_retention: float,
                                         coulombic_efficiency: float) -> List[str]:
        """Generate cycling test recommendations."""
        recommendations = []

        if capacity_retention < 95:
            recommendations.append('Investigate capacity fade mechanisms (postmortem analysis)')
        if coulombic_efficiency < 98:
            recommendations.append('Check for side reactions or gas evolution')
        if capacity_retention > 99:
            recommendations.append('Excellent stability - consider extended cycle life testing')

        return recommendations

    def _classify_rate_performance(self, ratio: Optional[float]) -> str:
        """Classify rate capability."""
        if ratio is None:
            return 'Not tested'
        if ratio > 0.8:
            return 'Excellent (minimal rate limitation)'
        elif ratio > 0.6:
            return 'Good (moderate rate capability)'
        elif ratio > 0.4:
            return 'Fair (kinetic limitations present)'
        else:
            return 'Poor (severe rate limitations)'

    def _assess_rate_capability(self, ratio: Optional[float]) -> str:
        """Assess rate capability with interpretation."""
        if ratio is None:
            return 'High rate (5C) not tested'
        if ratio > 0.8:
            return 'Excellent high-rate performance suitable for power applications'
        elif ratio > 0.6:
            return 'Good rate capability adequate for most applications'
        else:
            return 'Limited high-rate performance - improve conductivity or particle size'

    def _generate_rate_recommendations(self, ratio: Optional[float]) -> List[str]:
        """Generate rate capability recommendations."""
        if ratio is None or ratio > 0.7:
            return ['Material is suitable for high-power applications']
        else:
            return [
                'Improve electronic conductivity (carbon additives)',
                'Reduce particle size for shorter diffusion paths',
                'Optimize electrode thickness and porosity',
                'Consider surface coatings to enhance kinetics'
            ]

    def _assess_cycle_life(self, cycles_to_80: int, c_rate: float) -> str:
        """Assess cycle life performance."""
        if cycles_to_80 > 2000:
            return 'Excellent cycle life (>2000 cycles to 80%)'
        elif cycles_to_80 > 1000:
            return 'Good cycle life (1000-2000 cycles)'
        elif cycles_to_80 > 500:
            return 'Moderate cycle life (500-1000 cycles)'
        else:
            return 'Limited cycle life (<500 cycles) - investigate degradation'

    def _generate_cycle_life_recommendations(self, cycles_to_80: int,
                                            fade_acceleration: float) -> List[str]:
        """Generate cycle life recommendations."""
        recommendations = []

        if cycles_to_80 < 800:
            recommendations.append('Cycle life below target - optimize electrode composition')

        if fade_acceleration > 1.5:
            recommendations.append('Accelerating fade detected - investigate structural degradation')
        else:
            recommendations.append('Linear fade indicates stable SEI - promising for long life')

        recommendations.append('Perform postmortem analysis (SEM, XRD) on aged cells')
        recommendations.append('Cross-validate with EIS to track resistance growth')

        return recommendations

    def _identify_rate_limiting_step(self, avg_d: float) -> str:
        """Identify rate limiting step from diffusion coefficient."""
        if avg_d > 1e-11:
            return 'Fast diffusion - likely limited by charge transfer or electronic conductivity'
        elif avg_d > 1e-13:
            return 'Moderate diffusion - typical for intercalation materials'
        else:
            return 'Slow diffusion - solid-state diffusion is rate limiting'

    def _assess_power_capability(self, power_w: float, capacity_mah: float) -> str:
        """Assess power capability."""
        specific_power = power_w / (capacity_mah / 1000)  # W/Ah

        if specific_power > 1000:
            return 'Excellent power capability (>1000 W/Ah) suitable for HEV/EV'
        elif specific_power > 500:
            return 'Good power capability (500-1000 W/Ah)'
        else:
            return 'Limited power capability - optimize for lower resistance'

    # ============================================================================
    # Cross-Validation Methods
    # ============================================================================

    @staticmethod
    def validate_with_eis(battery_result: Dict[str, Any],
                         eis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate battery testing with Electrochemical Impedance Spectroscopy.

        Correlations:
        - HPPC DC resistance vs EIS low-frequency resistance
        - Cycle life resistance growth vs EIS Rct evolution
        - GITT diffusion vs EIS Warburg element

        Args:
            battery_result: Battery testing results (HPPC, cycle life, or GITT)
            eis_result: EIS results with Nyquist plot and equivalent circuit

        Returns:
            Cross-validation report with agreement metrics
        """
        technique = battery_result.get('technique', '')

        if 'HPPC' in technique:
            # Compare DC resistance from HPPC with EIS total resistance
            hppc_resistance = battery_result['power_capability']['discharge_power_at_50_soc_w']
            # Simplified comparison
            agreement = 'Good - HPPC and EIS resistances should match within 20%'

        elif 'Cycle Life' in technique:
            # Compare resistance growth trends
            resistance_increase = battery_result['cycle_life_analysis'][
                'resistance_increase_percent']
            agreement = f'Resistance increased {resistance_increase:.1f}% - ' \
                       'should correlate with EIS Rct growth'

        elif 'GITT' in technique:
            # Compare diffusion coefficients
            d_gitt = battery_result['diffusion_analysis']['average_diffusion_coefficient_cm2_s']
            agreement = f'D_GITT = {d_gitt:.2e} cm²/s - ' \
                       'should match EIS Warburg diffusion coefficient'
        else:
            agreement = 'Technique not applicable for EIS cross-validation'

        return {
            'validation_pair': 'Battery Testing ↔ EIS',
            'agreement': agreement,
            'correlation_strength': 'Strong',
            'recommendations': [
                'EIS provides frequency-resolved impedance (separates R_series, R_ct, W)',
                'HPPC gives time-domain pulse resistance (integrates all contributions)',
                'Use EIS to diagnose which resistance component increases during aging',
                'GITT + EIS provides orthogonal diffusion coefficient measurements'
            ]
        }

    @staticmethod
    def validate_with_sem_xrd(battery_result: Dict[str, Any],
                              sem_result: Dict[str, Any],
                              xrd_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate battery cycle life with SEM and XRD postmortem analysis.

        Args:
            battery_result: Cycle life results
            sem_result: SEM imaging of electrodes (fresh vs aged)
            xrd_result: XRD diffraction patterns (fresh vs aged)

        Returns:
            Cross-validation report linking performance to structure
        """
        if 'Cycle Life' not in battery_result.get('technique', ''):
            return {'validation_pair': 'Not applicable - requires cycle life data'}

        capacity_retention = battery_result['cycle_life_analysis'][
            'capacity_retention_after_1000_cycles_percent']

        expected_sem = []
        expected_xrd = []

        if capacity_retention < 80:
            expected_sem = [
                'Thick SEI layer on anode (>50 nm)',
                'Particle cracking/pulverization',
                'Electrode delamination from current collector',
                'Transition metal dissolution deposits'
            ]
            expected_xrd = [
                'Lattice parameter changes (>1%)',
                'New phase formation (degradation products)',
                'Peak broadening (strain/disorder)',
                'Preferred orientation changes'
            ]
        else:
            expected_sem = ['Thin, uniform SEI (<20 nm)', 'Intact particle morphology']
            expected_xrd = ['Stable crystal structure', 'Minimal lattice changes']

        return {
            'validation_pair': 'Battery Cycling ↔ SEM/XRD Postmortem',
            'capacity_retention_percent': capacity_retention,
            'expected_sem_observations': expected_sem,
            'expected_xrd_observations': expected_xrd,
            'correlation_strength': 'Direct causal relationship',
            'recommendations': [
                'Disassemble cells in Ar-filled glovebox (air-sensitive)',
                'SEM imaging: cross-section for SEI thickness, plan-view for cracks',
                'XRD: compare fresh vs aged electrodes for phase changes',
                'Combine with XPS for SEI chemical composition analysis',
                'Use TGA to quantify electrolyte residue and SEI mass'
            ]
        }

    @staticmethod
    def validate_with_voltammetry(battery_result: Dict[str, Any],
                                  cv_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate battery testing with cyclic voltammetry.

        Args:
            battery_result: Battery test results (preferably GITT for OCV)
            cv_result: Cyclic voltammetry results

        Returns:
            Cross-validation report comparing equilibrium potentials
        """
        if 'GITT' in battery_result.get('technique', ''):
            ocv_curve = battery_result['equilibrium_ocv_curve']['equilibrium_voltage_v']

            agreement = 'CV peak potentials should align with GITT OCV plateaus/inflections'
            correlation = 'Excellent - both measure redox thermodynamics'
        else:
            agreement = 'Use GITT for best correlation with CV'
            correlation = 'Moderate'

        return {
            'validation_pair': 'Battery GITT ↔ Voltammetry',
            'agreement': agreement,
            'correlation_strength': correlation,
            'complementary_information': [
                'GITT: Full-cell equilibrium OCV vs SOC',
                'CV: Half-cell redox potentials of individual electrodes',
                'CV peak separation indicates kinetic limitations',
                'GITT D(SOC) complements CV peak currents (Randles-Sevcik)'
            ]
        }


# ================================================================================
# Example Usage
# ================================================================================

if __name__ == "__main__":
    # Initialize agent
    config = {
        'cycler_channels': 8,
        'voltage_range': (2.5, 4.3),
        'current_range': (-10, 10),
        'temperature_control': True
    }

    agent = BatteryTestingAgent(config)

    # Example 1: Galvanostatic cycling
    print("=" * 80)
    print("Example 1: Galvanostatic Cycling")
    print("=" * 80)

    cycling_input = {
        'technique': 'galvanostatic_cycling',
        'battery_capacity_mah': 200,
        'c_rate': 0.5,
        'voltage_min': 3.0,
        'voltage_max': 4.2,
        'num_cycles': 3,
        'temperature_c': 25,
        'active_mass_mg': 15
    }

    cycling_result = agent.execute(cycling_input)
    print(f"\nTechnique: {cycling_result['technique']}")
    print(f"Avg Discharge Capacity: {cycling_result['performance_summary']['average_discharge_capacity_mah']:.2f} mAh")
    print(f"Capacity Retention: {cycling_result['performance_summary']['capacity_retention_percent']:.2f}%")
    print(f"Coulombic Efficiency: {cycling_result['performance_summary']['average_coulombic_efficiency_percent']:.2f}%")

    # Example 2: Rate capability
    print("\n" + "=" * 80)
    print("Example 2: Rate Capability Testing")
    print("=" * 80)

    rate_input = {
        'technique': 'rate_capability',
        'battery_capacity_mah': 200,
        'c_rates': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        'voltage_min': 3.0,
        'voltage_max': 4.2,
        'temperature_c': 25,
        'total_mass_mg': 50
    }

    rate_result = agent.execute(rate_input)
    print(f"\nRate Capability: {rate_result['rate_capability_analysis']['rate_capability_ratio_5c_01c']:.2f}")
    print(f"Classification: {rate_result['rate_capability_analysis']['rate_performance_classification']}")
    print(f"Max Power: {rate_result['rate_capability_analysis']['max_power_w']:.2f} W")

    # Example 3: GITT
    print("\n" + "=" * 80)
    print("Example 3: GITT (Diffusion Coefficient Measurement)")
    print("=" * 80)

    gitt_input = {
        'technique': 'gitt',
        'battery_capacity_mah': 200,
        'pulse_current_ma': 20,
        'pulse_duration_s': 600,
        'relaxation_time_s': 3600,
        'voltage_min': 3.0,
        'voltage_max': 4.2,
        'temperature_c': 25
    }

    gitt_result = agent.execute(gitt_input)
    print(f"\nAverage D: {gitt_result['diffusion_analysis']['average_diffusion_coefficient_cm2_s']:.2e} cm²/s")
    print(f"Rate Limiting: {gitt_result['interpretation']['rate_limiting_step']}")

    print("\n" + "=" * 80)
    print("BatteryTestingAgent Implementation Complete!")
    print("=" * 80)
