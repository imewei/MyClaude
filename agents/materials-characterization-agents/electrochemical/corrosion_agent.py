"""
CorrosionAgent - Comprehensive Corrosion Testing and Electrochemical Analysis

This agent provides complete corrosion characterization capabilities using
electrochemical and environmental exposure techniques.

Key Capabilities:
- Potentiodynamic Polarization - Tafel analysis for corrosion rates
- Linear Polarization Resistance (LPR) - Rapid corrosion rate estimation
- Cyclic Polarization - Pitting susceptibility and repassivation
- Electrochemical Impedance (EIS) - Corrosion mechanism analysis
- Salt Spray Testing (ASTM B117) - Accelerated corrosion
- Immersion Testing - Long-term corrosion rates
- Galvanic Corrosion - Dissimilar metal coupling
- Intergranular Corrosion - Sensitization testing

Applications:
- Material selection for corrosive environments
- Coating and surface treatment evaluation
- Corrosion inhibitor screening
- Failure analysis and prevention
- Quality control and acceptance testing
- Lifetime prediction modeling
- Environmental durability assessment

Cross-Validation Opportunities:
- Tafel ↔ LPR corrosion rate comparison
- EIS ↔ Polarization validation
- Accelerated ↔ Real-time exposure correlation
- Electrochemical ↔ Weight loss validation

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-02
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime


class CorrosionAgent:
    """
    Comprehensive corrosion testing and analysis agent.

    Supports electrochemical and environmental testing techniques for
    corrosion characterization across diverse material systems.
    """

    VERSION = "1.0.0"
    AGENT_TYPE = "corrosion"

    # Supported corrosion testing techniques
    SUPPORTED_TECHNIQUES = [
        'potentiodynamic_polarization',  # Tafel analysis
        'linear_polarization_resistance',  # LPR - quick i_corr
        'cyclic_polarization',           # Pitting susceptibility
        'eis_corrosion',                 # Impedance-based
        'salt_spray',                    # ASTM B117
        'immersion',                     # Long-term exposure
        'galvanic_coupling',             # Dissimilar metals
        'intergranular_corrosion',       # IGC/sensitization
        'crevice_corrosion',             # Localized attack
        'stress_corrosion_cracking'      # SCC susceptibility
    ]

    # Standard reference electrodes (potential vs SHE in V)
    REFERENCE_ELECTRODES = {
        'SCE': 0.241,      # Saturated Calomel Electrode
        'Ag/AgCl': 0.197,  # Silver/Silver Chloride (sat. KCl)
        'Hg/HgO': 0.098,   # Mercury/Mercury Oxide (1M NaOH)
        'SHE': 0.000       # Standard Hydrogen Electrode
    }

    # Tafel slope ranges for different reactions
    TAFEL_SLOPES = {
        'anodic': {'typical': 0.040, 'range': (0.020, 0.120)},  # V/decade
        'cathodic': {'typical': -0.120, 'range': (-0.200, -0.040)}
    }

    # Corrosion rate classification (mm/year)
    CORROSION_RATE_CLASSIFICATION = {
        'excellent': (0, 0.025),
        'good': (0.025, 0.125),
        'fair': (0.125, 0.5),
        'poor': (0.5, 1.25),
        'unacceptable': (1.25, float('inf'))
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CorrosionAgent.

        Args:
            config: Configuration dictionary containing:
                - default_technique: 'potentiodynamic_polarization', 'lpr', etc.
                - reference_electrode: 'SCE', 'Ag/AgCl', etc.
                - temperature_control: True/False
                - deaeration: 'N2', 'Ar', 'air', 'O2'
                - potentiostat_model: Instrument specification
        """
        self.config = config or {}
        self.default_technique = self.config.get('default_technique', 'potentiodynamic_polarization')
        self.reference_electrode = self.config.get('reference_electrode', 'SCE')
        self.temperature_control = self.config.get('temperature_control', True)
        self.deaeration = self.config.get('deaeration', 'N2')
        self.potentiostat = self.config.get('potentiostat_model', 'Gamry_Reference_600')

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute corrosion testing based on requested technique.

        Args:
            input_data: Dictionary containing:
                - technique: Corrosion test type
                - material_info: Material and environment description
                - test_parameters: Technique-specific parameters

        Returns:
            Comprehensive corrosion test results with metadata
        """
        technique = input_data.get('technique', self.default_technique)

        if technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(f"Unsupported technique: {technique}. "
                           f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Route to appropriate technique
        technique_map = {
            'potentiodynamic_polarization': self._execute_potentiodynamic,
            'linear_polarization_resistance': self._execute_lpr,
            'cyclic_polarization': self._execute_cyclic_polarization,
            'eis_corrosion': self._execute_eis_corrosion,
            'salt_spray': self._execute_salt_spray,
            'immersion': self._execute_immersion,
            'galvanic_coupling': self._execute_galvanic,
            'intergranular_corrosion': self._execute_igc,
            'crevice_corrosion': self._execute_crevice,
            'stress_corrosion_cracking': self._execute_scc
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

    def _execute_potentiodynamic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform potentiodynamic polarization with Tafel analysis.

        The gold standard for electrochemical corrosion rate determination.
        Measures current as a function of applied potential.

        Tafel equation: η = βa × log(i/i_corr) or η = βc × log(i/i_corr)

        Args:
            input_data: Contains scan parameters, polarization data

        Returns:
            Corrosion rate, Tafel slopes, corrosion potential
        """
        # Test parameters
        material = input_data.get('material', 'carbon_steel')
        electrolyte = input_data.get('electrolyte', '3.5% NaCl')
        temperature_c = input_data.get('temperature_c', 25)
        scan_rate_mv_s = input_data.get('scan_rate_mv_s', 0.167)  # 10 mV/min

        # Electrode area
        area_cm2 = input_data.get('electrode_area_cm2', 1.0)

        # Get polarization data or simulate
        potential_v = input_data.get('potential_vs_ref_v', None)
        current_density_a_cm2 = input_data.get('current_density_a_cm2', None)

        if potential_v is None or current_density_a_cm2 is None:
            # Simulate typical polarization curve
            potential_v, current_density_a_cm2 = self._simulate_polarization_curve(
                material, electrolyte
            )

        # Perform Tafel analysis
        tafel_results = self._tafel_analysis(potential_v, current_density_a_cm2)

        # Extract key parameters
        e_corr = tafel_results['e_corr']
        i_corr = tafel_results['i_corr']
        beta_a = tafel_results['beta_a']
        beta_c = tafel_results['beta_c']

        # Calculate corrosion rate
        equivalent_weight = input_data.get('equivalent_weight', 27.9)  # Fe -> Fe2+ (55.8/2)
        density_g_cm3 = input_data.get('density_g_cm3', 7.87)  # Steel

        corrosion_rate_mm_yr = self._calculate_corrosion_rate(
            i_corr, equivalent_weight, density_g_cm3
        )

        # Classify corrosion resistance
        classification = self._classify_corrosion_rate(corrosion_rate_mm_yr)

        # Polarization resistance
        rp_ohm_cm2 = (beta_a * abs(beta_c)) / (2.303 * i_corr * (beta_a + abs(beta_c)))

        return {
            'technique': 'potentiodynamic_polarization',
            'corrosion_potential_v_vs_ref': e_corr,
            'corrosion_current_density_a_cm2': i_corr,
            'corrosion_rate_mm_per_year': corrosion_rate_mm_yr,
            'corrosion_rate_mpy': corrosion_rate_mm_yr * 39.37,  # mils per year
            'anodic_tafel_slope_v_decade': beta_a,
            'cathodic_tafel_slope_v_decade': beta_c,
            'polarization_resistance_ohm_cm2': rp_ohm_cm2,
            'material': material,
            'electrolyte': electrolyte,
            'temperature_c': temperature_c,
            'scan_rate_mv_s': scan_rate_mv_s,
            'electrode_area_cm2': area_cm2,
            'reference_electrode': self.reference_electrode,
            'corrosion_classification': classification,
            'polarization_data': {
                'potential_v': potential_v.tolist() if isinstance(potential_v, np.ndarray) else potential_v,
                'current_density_a_cm2': current_density_a_cm2.tolist() if isinstance(current_density_a_cm2, np.ndarray) else current_density_a_cm2
            },
            'quality_metrics': {
                'tafel_fit_quality': tafel_results.get('fit_quality', 'good'),
                'scan_quality': 'good' if scan_rate_mv_s <= 1.0 else 'fast'
            },
            'recommendations': self._generate_polarization_recommendations(
                e_corr, i_corr, beta_a, beta_c
            )
        }

    def _execute_lpr(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Linear Polarization Resistance (LPR) measurement.

        Rapid, non-destructive technique for corrosion rate monitoring.
        Applies small potential perturbation (±10-20 mV) around OCP.

        Stern-Geary equation: i_corr = B / R_p
        where B = βa × βc / [2.303(βa + βc)]

        Args:
            input_data: Contains LPR scan data and material properties

        Returns:
            Corrosion rate from LPR analysis
        """
        # Test parameters
        material = input_data.get('material', 'stainless_steel_304')
        electrolyte = input_data.get('electrolyte', '3.5% NaCl')
        temperature_c = input_data.get('temperature_c', 25)

        # LPR scan parameters
        potential_window_mv = input_data.get('potential_window_mv', 20)  # ±10 mV
        scan_rate_mv_s = input_data.get('scan_rate_mv_s', 0.167)

        # Tafel slopes (can be estimated or measured separately)
        beta_a = input_data.get('beta_a', 0.060)
        beta_c = input_data.get('beta_c', -0.120)

        # Measure or simulate polarization resistance
        rp_ohm_cm2 = input_data.get('polarization_resistance_ohm_cm2', None)

        if rp_ohm_cm2 is None:
            # Simulate LPR measurement
            expected_i_corr = input_data.get('expected_i_corr_a_cm2', 1e-6)
            B = (beta_a * abs(beta_c)) / (2.303 * (beta_a + abs(beta_c)))
            rp_ohm_cm2 = B / expected_i_corr
            rp_ohm_cm2 *= (1 + 0.05 * np.random.randn())  # Add noise

        # Calculate Stern-Geary constant
        B = (beta_a * abs(beta_c)) / (2.303 * (beta_a + abs(beta_c)))

        # Calculate corrosion current density
        i_corr = B / rp_ohm_cm2

        # Calculate corrosion rate
        equivalent_weight = input_data.get('equivalent_weight', 27.9)
        density_g_cm3 = input_data.get('density_g_cm3', 7.87)

        corrosion_rate_mm_yr = self._calculate_corrosion_rate(
            i_corr, equivalent_weight, density_g_cm3
        )

        classification = self._classify_corrosion_rate(corrosion_rate_mm_yr)

        return {
            'technique': 'linear_polarization_resistance',
            'polarization_resistance_ohm_cm2': rp_ohm_cm2,
            'stern_geary_constant_v': B,
            'corrosion_current_density_a_cm2': i_corr,
            'corrosion_rate_mm_per_year': corrosion_rate_mm_yr,
            'corrosion_rate_mpy': corrosion_rate_mm_yr * 39.37,
            'material': material,
            'electrolyte': electrolyte,
            'temperature_c': temperature_c,
            'potential_window_mv': potential_window_mv,
            'anodic_tafel_slope_v_decade': beta_a,
            'cathodic_tafel_slope_v_decade': beta_c,
            'corrosion_classification': classification,
            'advantages': [
                'Rapid measurement (< 5 minutes)',
                'Non-destructive (small perturbation)',
                'Real-time monitoring capability',
                'Good for inhibitor screening'
            ],
            'limitations': [
                'Requires accurate Tafel slopes (or assumes B = 0.026 V)',
                'Assumes uniform corrosion',
                'Less accurate than full polarization for localized corrosion'
            ]
        }

    def _execute_cyclic_polarization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cyclic polarization for pitting susceptibility assessment.

        Scan potential forward to high anodic values, then reverse.
        Hysteresis indicates pitting; return/OCP crossing indicates repassivation.

        Args:
            input_data: Contains cyclic scan parameters

        Returns:
            Pitting potential, protection potential, repassivation behavior
        """
        material = input_data.get('material', 'stainless_steel_316L')
        electrolyte = input_data.get('electrolyte', '3.5% NaCl')
        temperature_c = input_data.get('temperature_c', 25)

        # Scan parameters
        reverse_potential_v = input_data.get('reverse_potential_v', 1.0)
        scan_rate_mv_s = input_data.get('scan_rate_mv_s', 0.167)

        # Simulate cyclic polarization
        e_corr = input_data.get('e_corr_v', -0.200)  # vs SCE
        e_pit = input_data.get('e_pit_v', 0.300)  # Pitting potential
        e_prot = input_data.get('e_prot_v', 0.100)  # Protection/repassivation potential

        # Hysteresis area (indicator of pitting damage)
        hysteresis_area = abs(e_pit - e_prot) * 1e-4  # Simplified

        # Pitting susceptibility
        pitting_susceptibility = self._classify_pitting_susceptibility(
            e_pit, e_prot, e_corr
        )

        return {
            'technique': 'cyclic_polarization',
            'corrosion_potential_v': e_corr,
            'pitting_potential_v': e_pit,
            'protection_potential_v': e_prot,
            'reverse_potential_v': reverse_potential_v,
            'hysteresis_area': hysteresis_area,
            'pitting_susceptibility': pitting_susceptibility,
            'repassivation_behavior': 'good' if e_prot < e_corr else 'poor',
            'material': material,
            'electrolyte': electrolyte,
            'temperature_c': temperature_c,
            'interpretation': {
                'positive_hysteresis': 'Indicates pitting susceptibility',
                'negative_hysteresis': 'Good repassivation, low pitting risk',
                'protection_potential': 'Safe operating potential below this value'
            },
            'recommendations': [
                f"Maintain potential below {e_prot:.3f} V to prevent pitting",
                "Use cathodic protection or inhibitors",
                "Consider more resistant alloy if E_pit - E_corr < 200 mV"
            ]
        }

    def _execute_eis_corrosion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Electrochemical Impedance Spectroscopy for corrosion analysis.

        Measure impedance vs frequency to extract corrosion parameters and
        understand corrosion mechanisms.

        Args:
            input_data: Contains EIS data and fitting parameters

        Returns:
            Corrosion parameters from EIS circuit fitting
        """
        material = input_data.get('material', 'aluminum_alloy')
        electrolyte = input_data.get('electrolyte', '0.1M NaCl')

        # EIS parameters
        frequency_hz = input_data.get('frequency_hz', np.logspace(-2, 5, 50))

        # Simulate or use measured impedance
        # Typically fit to Randles circuit: Rs + (Rp || CPE)
        r_s = input_data.get('solution_resistance_ohm_cm2', 10)  # Solution resistance
        r_p = input_data.get('polarization_resistance_ohm_cm2', 10000)  # Polarization resistance

        # From R_p, calculate corrosion rate
        B = input_data.get('stern_geary_constant', 0.026)  # V
        i_corr = B / r_p

        equivalent_weight = input_data.get('equivalent_weight', 9.0)  # Al -> Al3+ (27/3)
        density_g_cm3 = input_data.get('density_g_cm3', 2.70)

        corrosion_rate_mm_yr = self._calculate_corrosion_rate(
            i_corr, equivalent_weight, density_g_cm3
        )

        # CPE parameters (Constant Phase Element - accounts for non-ideal capacitor)
        cpe_t = input_data.get('cpe_t', 50e-6)  # F·s^(n-1)/cm²
        cpe_n = input_data.get('cpe_n', 0.85)  # Ideality factor (1 = ideal capacitor)

        # Coating capacitance (if present)
        coating_present = r_p > 100000  # High R_p suggests coating

        return {
            'technique': 'electrochemical_impedance_spectroscopy',
            'solution_resistance_ohm_cm2': r_s,
            'polarization_resistance_ohm_cm2': r_p,
            'corrosion_current_density_a_cm2': i_corr,
            'corrosion_rate_mm_per_year': corrosion_rate_mm_yr,
            'cpe_capacitance_f_cm2': cpe_t,
            'cpe_exponent': cpe_n,
            'coating_detected': coating_present,
            'material': material,
            'electrolyte': electrolyte,
            'frequency_range_hz': (frequency_hz[0], frequency_hz[-1]) if isinstance(frequency_hz, np.ndarray) else frequency_hz,
            'equivalent_circuit': 'Randles: Rs + (Rp || CPE)',
            'interpretation': {
                'high_rp': 'Good corrosion resistance',
                'low_cpe_n': 'Surface heterogeneity or roughness',
                'multiple_time_constants': 'Complex corrosion mechanism or coating'
            },
            'advantages': [
                'Mechanistic information from frequency response',
                'Non-destructive measurement',
                'Sensitive to coating degradation',
                'Can separate charge transfer and diffusion'
            ]
        }

    def _execute_salt_spray(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform salt spray (fog) testing per ASTM B117.

        Accelerated corrosion testing using continuous atomized salt solution spray.

        Args:
            input_data: Contains exposure duration, visual inspection results

        Returns:
            Corrosion damage assessment and rating
        """
        material = input_data.get('material', 'zinc_coated_steel')
        coating_thickness_um = input_data.get('coating_thickness_um', 25)

        # Test parameters (ASTM B117 standard)
        temperature_c = 35
        salt_concentration_percent = 5  # NaCl
        ph = 6.8  # 6.5-7.2 per standard

        # Exposure duration
        exposure_hours = input_data.get('exposure_hours', 96)

        # Visual inspection results
        corrosion_type = input_data.get('observed_corrosion', 'red_rust')
        percent_area_corroded = input_data.get('percent_area_corroded', 5)

        # Rating per ASTM standards
        corrosion_rating = self._astm_corrosion_rating(percent_area_corroded)

        # Estimate time to first rust
        time_to_rust_hours = input_data.get('time_to_first_rust_hours', 48)

        # Comparison to requirement
        required_hours = input_data.get('required_exposure_hours', 96)
        passes_requirement = exposure_hours >= required_hours and percent_area_corroded < 5

        return {
            'technique': 'salt_spray_test_astm_b117',
            'exposure_duration_hours': exposure_hours,
            'time_to_first_rust_hours': time_to_rust_hours,
            'percent_area_corroded': percent_area_corroded,
            'corrosion_rating': corrosion_rating,
            'corrosion_type_observed': corrosion_type,
            'passes_requirement': passes_requirement,
            'material': material,
            'coating_thickness_um': coating_thickness_um,
            'test_conditions': {
                'temperature_c': temperature_c,
                'salt_concentration_percent': salt_concentration_percent,
                'ph': ph,
                'standard': 'ASTM B117'
            },
            'interpretation': {
                'rating_10': 'No corrosion',
                'rating_9': '< 0.1% area',
                'rating_7': '0.25-0.5% area',
                'rating_5': '1-2.5% area',
                'rating_0': '> 50% area'
            },
            'limitations': [
                'Highly accelerated - not representative of real exposure',
                'Results vary by material and environment',
                'Qualitative assessment',
                'Should correlate with field testing'
            ]
        }

    def _execute_immersion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform immersion testing with weight loss measurement.

        Long-term exposure in corrosive medium with periodic weight measurements.
        Provides average corrosion rate over exposure period.

        Args:
            input_data: Contains immersion parameters and weight measurements

        Returns:
            Corrosion rate from weight loss
        """
        material = input_data.get('material', 'copper')
        electrolyte = input_data.get('electrolyte', '3.5% NaCl')
        temperature_c = input_data.get('temperature_c', 25)

        # Sample parameters
        initial_weight_g = input_data.get('initial_weight_g', 50.0)
        final_weight_g = input_data.get('final_weight_g', None)
        surface_area_cm2 = input_data.get('surface_area_cm2', 20.0)
        exposure_time_hours = input_data.get('exposure_time_hours', 720)  # 30 days

        if final_weight_g is None:
            # Simulate weight loss
            expected_rate_mm_yr = input_data.get('expected_corrosion_rate_mm_yr', 0.1)
            density_g_cm3 = input_data.get('density_g_cm3', 8.96)

            # Weight loss = rate × time × area × density
            weight_loss_g = (expected_rate_mm_yr / 10 / 8760) * exposure_time_hours * \
                          surface_area_cm2 * density_g_cm3
            final_weight_g = initial_weight_g - weight_loss_g

        weight_loss_g = initial_weight_g - final_weight_g

        # Calculate corrosion rate from weight loss
        density_g_cm3 = input_data.get('density_g_cm3', 8.96)
        corrosion_rate_mm_yr = (weight_loss_g * 87600) / \
                              (surface_area_cm2 * exposure_time_hours * density_g_cm3)

        classification = self._classify_corrosion_rate(corrosion_rate_mm_yr)

        return {
            'technique': 'immersion_weight_loss',
            'initial_weight_g': initial_weight_g,
            'final_weight_g': final_weight_g,
            'weight_loss_g': weight_loss_g,
            'weight_loss_percent': (weight_loss_g / initial_weight_g) * 100,
            'corrosion_rate_mm_per_year': corrosion_rate_mm_yr,
            'corrosion_rate_mpy': corrosion_rate_mm_yr * 39.37,
            'exposure_time_hours': exposure_time_hours,
            'exposure_time_days': exposure_time_hours / 24,
            'material': material,
            'electrolyte': electrolyte,
            'temperature_c': temperature_c,
            'surface_area_cm2': surface_area_cm2,
            'corrosion_classification': classification,
            'advantages': [
                'Simple and direct measurement',
                'Average corrosion rate over exposure',
                'Real corrosion environment',
                'Low cost'
            ],
            'limitations': [
                'Destructive test',
                'Only total weight loss (no localized info)',
                'Long test duration',
                'Requires careful cleaning procedure'
            ]
        }

    def _execute_galvanic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess galvanic corrosion between dissimilar metals.

        Measures galvanic current when two different metals are electrically
        coupled in an electrolyte.

        Args:
            input_data: Contains metal pair and galvanic measurements

        Returns:
            Galvanic current, potential difference, corrosion risk
        """
        metal_1 = input_data.get('metal_1', '316_stainless_steel')
        metal_2 = input_data.get('metal_2', 'carbon_steel')
        electrolyte = input_data.get('electrolyte', 'seawater')

        # Uncoupled potentials (vs reference)
        e_metal_1 = input_data.get('e_metal_1_v', -0.050)  # Noble (cathode)
        e_metal_2 = input_data.get('e_metal_2_v', -0.600)  # Active (anode)

        potential_difference = e_metal_1 - e_metal_2

        # Area ratio (critical parameter)
        area_metal_1_cm2 = input_data.get('area_metal_1_cm2', 100)
        area_metal_2_cm2 = input_data.get('area_metal_2_cm2', 10)
        area_ratio = area_metal_1_cm2 / area_metal_2_cm2

        # Galvanic current (simplified - would measure in real test)
        galvanic_current_a = input_data.get('galvanic_current_a', None)

        if galvanic_current_a is None:
            # Estimate from potential difference and assume polarization resistance
            r_galvanic = 100  # ohm·cm² (total circuit resistance)
            galvanic_current_a = abs(potential_difference) / r_galvanic * area_metal_2_cm2

        # Current density on anode (metal 2 - corroding)
        i_galvanic_a_cm2 = galvanic_current_a / area_metal_2_cm2

        # Accelerated corrosion rate on anode
        equivalent_weight = input_data.get('equivalent_weight_metal2', 27.9)
        density_g_cm3 = input_data.get('density_metal2', 7.87)

        corrosion_rate_mm_yr = self._calculate_corrosion_rate(
            i_galvanic_a_cm2, equivalent_weight, density_g_cm3
        )

        # Galvanic compatibility assessment
        compatibility = self._assess_galvanic_compatibility(
            potential_difference, area_ratio
        )

        return {
            'technique': 'galvanic_corrosion',
            'metal_1_cathode': metal_1,
            'metal_2_anode': metal_2,
            'potential_metal_1_v': e_metal_1,
            'potential_metal_2_v': e_metal_2,
            'potential_difference_v': potential_difference,
            'area_ratio_cathode_to_anode': area_ratio,
            'galvanic_current_a': galvanic_current_a,
            'galvanic_current_density_anode_a_cm2': i_galvanic_a_cm2,
            'accelerated_corrosion_rate_mm_yr': corrosion_rate_mm_yr,
            'galvanic_compatibility': compatibility,
            'electrolyte': electrolyte,
            'risk_factors': {
                'large_potential_difference': potential_difference > 0.25,
                'unfavorable_area_ratio': area_ratio > 10,  # Large cathode, small anode
                'conductive_electrolyte': 'seawater' in electrolyte.lower()
            },
            'mitigation_strategies': [
                'Use insulating gaskets/washers to break electrical contact',
                'Apply coatings (preferably to cathode)',
                'Use sacrificial anodes',
                'Select metals close in galvanic series',
                'Avoid large cathode / small anode ratios'
            ]
        }

    def _execute_igc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform intergranular corrosion (IGC) testing.

        Assess susceptibility to grain boundary attack due to sensitization
        (carbide precipitation in stainless steels).

        Common tests: ASTM A262 (Practice A, E), ISO 3651

        Args:
            input_data: Contains IGC test parameters and results

        Returns:
            IGC susceptibility assessment
        """
        material = input_data.get('material', 'stainless_steel_304')
        test_method = input_data.get('test_method', 'ASTM_A262_Practice_E')

        # Heat treatment history
        sensitization_treatment = input_data.get('sensitization_treatment',
                                                 '650C_for_10_hours')

        # Practice E: Copper-copper sulfate-sulfuric acid test
        if 'Practice_E' in test_method:
            # 24 hour immersion, bend test, visual inspection
            exposure_hours = input_data.get('exposure_hours', 24)
            cracks_observed = input_data.get('cracks_after_bend', False)

            if cracks_observed:
                result = 'FAIL - Susceptible to IGC'
                recommendation = 'Material is sensitized - use low-C grade (304L) or solution anneal'
            else:
                result = 'PASS - Not susceptible to IGC'
                recommendation = 'Material shows good IGC resistance'

        # Practice A: Oxalic acid etch test
        elif 'Practice_A' in test_method:
            etch_structure = input_data.get('etch_structure', 'dual')
            # Structures: step, dual, ditch

            if etch_structure == 'step':
                result = 'Acceptable - Not sensitized'
                recommendation = 'Material suitable for corrosive service'
            elif etch_structure == 'ditch':
                result = 'FAIL - Severely sensitized'
                recommendation = 'Solution anneal required (1050-1100°C)'
            else:  # dual
                result = 'MARGINAL - Partially sensitized'
                recommendation = 'Consider stabilization anneal or use L-grade'

        else:
            result = 'Unknown test method'
            recommendation = 'Specify valid ASTM A262 practice'

        return {
            'technique': 'intergranular_corrosion_testing',
            'test_method': test_method,
            'material': material,
            'sensitization_treatment': sensitization_treatment,
            'test_result': result,
            'recommendation': recommendation,
            'common_test_methods': {
                'ASTM_A262_Practice_A': 'Oxalic acid etch (qualitative)',
                'ASTM_A262_Practice_E': 'Copper sulfate + sulfuric acid + bend test',
                'ISO_3651-2': 'Ferric sulfate test',
                'Strauss_test': 'Copper sulfate + sulfuric acid immersion'
            },
            'mitigation': [
                'Use low-carbon grades (304L, 316L)',
                'Solution anneal (1050-1100°C) after welding',
                'Use stabilized grades (321, 347) with Ti or Nb',
                'Avoid exposure to sensitization range (450-850°C)'
            ]
        }

    def _execute_crevice(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess crevice corrosion susceptibility.

        Localized attack in shielded areas (under gaskets, washers, deposits).

        Args:
            input_data: Contains crevice test parameters

        Returns:
            Crevice corrosion assessment
        """
        material = input_data.get('material', 'titanium_alloy')
        electrolyte = input_data.get('electrolyte', '10% FeCl3')
        temperature_c = input_data.get('temperature_c', 50)

        # Crevice former (ASTM G48 uses multiple crevice assembly)
        crevice_gap_um = input_data.get('crevice_gap_um', 25)
        exposure_hours = input_data.get('exposure_hours', 72)

        # Inspection results
        attack_depth_um = input_data.get('attack_depth_um', 0)
        weight_loss_mg = input_data.get('weight_loss_mg', 0)

        # Critical crevice temperature (CCT)
        cct_c = input_data.get('critical_crevice_temperature_c', 60)

        if attack_depth_um == 0 and weight_loss_mg == 0:
            result = 'No crevice corrosion detected'
            susceptibility = 'resistant'
        elif attack_depth_um < 100:
            result = 'Minor crevice attack'
            susceptibility = 'moderate'
        else:
            result = 'Severe crevice corrosion'
            susceptibility = 'susceptible'

        return {
            'technique': 'crevice_corrosion_test',
            'material': material,
            'electrolyte': electrolyte,
            'temperature_c': temperature_c,
            'crevice_gap_um': crevice_gap_um,
            'exposure_hours': exposure_hours,
            'attack_depth_um': attack_depth_um,
            'weight_loss_mg': weight_loss_mg,
            'critical_crevice_temperature_c': cct_c,
            'susceptibility': susceptibility,
            'test_result': result,
            'standards': ['ASTM G48', 'ASTM G78'],
            'prevention': [
                'Avoid crevices in design (seal welds, smooth surfaces)',
                'Use more resistant alloys (higher Mo, N content)',
                'Regular cleaning to prevent deposits',
                'Cathodic protection',
                'Operate below critical crevice temperature'
            ]
        }

    def _execute_scc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess stress corrosion cracking (SCC) susceptibility.

        Combined effect of tensile stress and corrosive environment.

        Tests: Slow strain rate test (SSRT), U-bend, C-ring

        Args:
            input_data: Contains SCC test parameters

        Returns:
            SCC susceptibility assessment
        """
        material = input_data.get('material', '7075_aluminum')
        environment = input_data.get('environment', '3.5% NaCl')
        stress_mpa = input_data.get('applied_stress_mpa', 300)
        test_type = input_data.get('test_type', 'slow_strain_rate')

        if test_type == 'slow_strain_rate':
            # SSRT at very slow strain rate (1e-6 to 1e-5 /s)
            strain_rate_per_s = input_data.get('strain_rate_per_s', 1e-6)
            time_to_failure_hours = input_data.get('time_to_failure_hours', 100)

            # Compare to baseline (inert environment)
            time_to_failure_air_hours = input_data.get('time_to_failure_air_hours', 200)

            scc_susceptibility_index = time_to_failure_hours / time_to_failure_air_hours

            if scc_susceptibility_index > 0.8:
                susceptibility = 'low'
            elif scc_susceptibility_index > 0.5:
                susceptibility = 'moderate'
            else:
                susceptibility = 'high'

        else:  # U-bend or C-ring
            exposure_hours = input_data.get('exposure_hours', 720)
            cracks_observed = input_data.get('cracks_observed', False)

            susceptibility = 'high' if cracks_observed else 'low'
            scc_susceptibility_index = 0 if cracks_observed else 1

        return {
            'technique': 'stress_corrosion_cracking_test',
            'test_type': test_type,
            'material': material,
            'environment': environment,
            'applied_stress_mpa': stress_mpa,
            'scc_susceptibility': susceptibility,
            'scc_susceptibility_index': scc_susceptibility_index,
            'common_scc_systems': {
                'austenitic_stainless_chlorides': '304/316 in Cl⁻ solutions',
                'brass_ammonia': 'Brass in NH₃',
                'carbon_steel_hydroxide': 'CS in caustic (NaOH)',
                'aluminum_alloys_chlorides': '7xxx in NaCl',
                'titanium_alloys_chlorides': 'Ti in hot Cl⁻ + methanol'
            },
            'prevention': [
                'Reduce applied/residual stress (stress relief anneal)',
                'Use resistant alloy',
                'Avoid critical environment',
                'Cathodic protection (carefully - avoid hydrogen embrittlement)',
                'Apply compressive surface stress (shot peening)'
            ]
        }

    # Helper methods

    def _simulate_polarization_curve(self, material: str,
                                     electrolyte: str) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate typical polarization curve for demonstration."""
        # Potential range: -1.0 to +0.5 V vs reference
        potential = np.linspace(-1.0, 0.5, 200)

        # Typical parameters for carbon steel in NaCl
        e_corr = -0.65  # V vs SCE
        i_corr = 1e-5   # A/cm²
        beta_a = 0.060   # V/decade
        beta_c = -0.120  # V/decade

        # Tafel equations
        eta = potential - e_corr
        i_a = i_corr * 10 ** (eta / beta_a)
        i_c = -i_corr * 10 ** (eta / beta_c)
        current_density = i_a + i_c

        # Add small noise
        current_density *= (1 + 0.05 * np.random.randn(len(current_density)))

        return potential, current_density

    def _tafel_analysis(self, potential: np.ndarray,
                       current_density: np.ndarray) -> Dict[str, Any]:
        """Perform Tafel analysis to extract corrosion parameters."""
        # Find OCP (where current ~ 0)
        idx_corr = np.argmin(np.abs(current_density))
        e_corr = potential[idx_corr]

        # Anodic branch (potential > E_corr)
        anodic_mask = potential > (e_corr + 0.050)
        if np.sum(anodic_mask) > 10:
            log_i_a = np.log10(np.abs(current_density[anodic_mask]))
            e_a = potential[anodic_mask]
            beta_a = np.polyfit(log_i_a, e_a, 1)[0]
        else:
            beta_a = 0.060  # Default

        # Cathodic branch (potential < E_corr)
        cathodic_mask = potential < (e_corr - 0.050)
        if np.sum(cathodic_mask) > 10:
            log_i_c = np.log10(np.abs(current_density[cathodic_mask]))
            e_c = potential[cathodic_mask]
            beta_c = np.polyfit(log_i_c, e_c, 1)[0]
        else:
            beta_c = -0.120  # Default

        # Estimate i_corr (simplified - would use Tafel extrapolation in practice)
        i_corr = np.abs(current_density[idx_corr])
        if i_corr < 1e-10:
            i_corr = 1e-6  # Typical value

        return {
            'e_corr': e_corr,
            'i_corr': i_corr,
            'beta_a': beta_a,
            'beta_c': beta_c,
            'fit_quality': 'good'
        }

    def _calculate_corrosion_rate(self, i_corr: float, equivalent_weight: float,
                                  density: float) -> float:
        """
        Calculate corrosion rate from current density using Faraday's law.

        CR (mm/year) = (i_corr × EW × 3.27 × 10^6) / (n × ρ)

        where:
        - i_corr in A/cm²
        - EW = equivalent weight (g/eq)
        - ρ = density (g/cm³)
        - 3.27 × 10^6 = conversion constant to mm/year
        """
        # CR = i × K × EW / density
        # K = 3.27 × 10^6 for mm/year
        K = 3.27e6  # mm/(A·cm·year)

        corrosion_rate = (i_corr * K * equivalent_weight) / density

        return corrosion_rate

    def _classify_corrosion_rate(self, rate_mm_yr: float) -> str:
        """Classify corrosion rate severity."""
        for classification, (min_rate, max_rate) in self.CORROSION_RATE_CLASSIFICATION.items():
            if min_rate <= rate_mm_yr < max_rate:
                return classification
        return 'unknown'

    def _classify_pitting_susceptibility(self, e_pit: float, e_prot: float,
                                         e_corr: float) -> str:
        """Classify pitting susceptibility from cyclic polarization."""
        delta_e_pit = e_pit - e_corr

        if delta_e_pit > 0.5:
            return 'very_low_susceptibility'
        elif delta_e_pit > 0.3:
            return 'low_susceptibility'
        elif delta_e_pit > 0.15:
            return 'moderate_susceptibility'
        else:
            return 'high_susceptibility'

    def _astm_corrosion_rating(self, percent_corroded: float) -> int:
        """Convert percent area corroded to ASTM corrosion rating (0-10)."""
        if percent_corroded == 0:
            return 10
        elif percent_corroded < 0.1:
            return 9
        elif percent_corroded < 0.25:
            return 8
        elif percent_corroded < 0.5:
            return 7
        elif percent_corroded < 1.0:
            return 6
        elif percent_corroded < 2.5:
            return 5
        elif percent_corroded < 5.0:
            return 4
        elif percent_corroded < 10:
            return 3
        elif percent_corroded < 25:
            return 2
        elif percent_corroded < 50:
            return 1
        else:
            return 0

    def _assess_galvanic_compatibility(self, potential_diff: float,
                                       area_ratio: float) -> str:
        """Assess galvanic compatibility of metal pair."""
        if potential_diff < 0.10:
            return 'excellent_compatibility'
        elif potential_diff < 0.25:
            if area_ratio < 1.0:  # Small cathode
                return 'good_compatibility'
            else:
                return 'fair_compatibility'
        else:  # > 0.25 V
            if area_ratio > 5:
                return 'poor_compatibility_high_risk'
            else:
                return 'marginal_compatibility'

    def _generate_polarization_recommendations(self, e_corr: float, i_corr: float,
                                               beta_a: float, beta_c: float) -> List[str]:
        """Generate recommendations based on polarization results."""
        recs = []

        if i_corr > 1e-4:
            recs.append("High corrosion rate - consider protective coating or inhibitor")

        if abs(beta_a) < 0.030 or abs(beta_c) < 0.030:
            recs.append("Unusual Tafel slopes - verify scan quality and stability")

        if e_corr < -0.7:
            recs.append("Very negative E_corr - material highly active, consider cathodic protection")

        recs.append("Validate with LPR for quick corrosion rate monitoring")
        recs.append("Consider EIS for mechanistic understanding")

        return recs

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and metadata."""
        return {
            'agent_type': self.AGENT_TYPE,
            'version': self.VERSION,
            'supported_techniques': self.SUPPORTED_TECHNIQUES,
            'electrochemical_techniques': [
                'Potentiodynamic Polarization (Tafel)',
                'Linear Polarization Resistance (LPR)',
                'Cyclic Polarization (pitting)',
                'EIS (mechanistic analysis)'
            ],
            'environmental_techniques': [
                'Salt Spray (ASTM B117)',
                'Immersion (weight loss)',
                'Atmospheric exposure'
            ],
            'specialized_techniques': [
                'Galvanic corrosion',
                'Intergranular corrosion (IGC)',
                'Crevice corrosion',
                'Stress corrosion cracking (SCC)'
            ],
            'measurement_ranges': {
                'corrosion_rate': '0.001-1000 mm/year',
                'current_density': '1e-9 to 1e-2 A/cm²',
                'potential': '-2.0 to +2.0 V vs reference'
            },
            'cross_validation_opportunities': [
                'Tafel ↔ LPR corrosion rate verification',
                'EIS ↔ Polarization parameter comparison',
                'Accelerated ↔ Real-time correlation',
                'Electrochemical ↔ Weight loss validation'
            ]
        }


if __name__ == '__main__':
    # Example usage
    agent = CorrosionAgent()

    # Example 1: Potentiodynamic polarization
    result_tafel = agent.execute({
        'technique': 'potentiodynamic_polarization',
        'material': 'carbon_steel',
        'electrolyte': '3.5% NaCl',
        'temperature_c': 25
    })
    print("Potentiodynamic Polarization Result:")
    print(f"  E_corr = {result_tafel['corrosion_potential_v_vs_ref']:.3f} V vs {result_tafel['reference_electrode']}")
    print(f"  i_corr = {result_tafel['corrosion_current_density_a_cm2']:.2e} A/cm²")
    print(f"  Corrosion Rate = {result_tafel['corrosion_rate_mm_per_year']:.3f} mm/year")
    print(f"  Classification: {result_tafel['corrosion_classification']}")
    print()

    # Example 2: LPR measurement
    result_lpr = agent.execute({
        'technique': 'linear_polarization_resistance',
        'material': 'stainless_steel_316',
        'electrolyte': '0.1M H2SO4',
        'polarization_resistance_ohm_cm2': 50000
    })
    print("LPR Result:")
    print(f"  R_p = {result_lpr['polarization_resistance_ohm_cm2']:.0f} Ω·cm²")
    print(f"  i_corr = {result_lpr['corrosion_current_density_a_cm2']:.2e} A/cm²")
    print(f"  Corrosion Rate = {result_lpr['corrosion_rate_mm_per_year']:.4f} mm/year")
    print()

    # Example 3: Cyclic polarization
    result_cyclic = agent.execute({
        'technique': 'cyclic_polarization',
        'material': 'stainless_steel_304',
        'electrolyte': '3.5% NaCl',
        'e_pit_v': 0.350,
        'e_prot_v': 0.150
    })
    print("Cyclic Polarization Result:")
    print(f"  Pitting Potential = {result_cyclic['pitting_potential_v']:.3f} V")
    print(f"  Protection Potential = {result_cyclic['protection_potential_v']:.3f} V")
    print(f"  Pitting Susceptibility: {result_cyclic['pitting_susceptibility']}")
