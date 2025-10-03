"""
HardnessTestingAgent - Comprehensive Macro and Microhardness Testing

This agent provides complete hardness testing capabilities across multiple scales
and materials, from soft polymers to ultra-hard ceramics.

Key Capabilities:
- Vickers Hardness (HV) - Universal microhardness
- Rockwell Hardness (HR) - Multiple scales for different materials
- Brinell Hardness (HB) - Large-scale bulk hardness
- Knoop Hardness (HK) - Elongated indentation for thin layers
- Shore Hardness - Polymers and elastomers (A, D, 00 scales)
- Mohs Hardness - Mineralogical scratch resistance

Applications:
- Quality control and material acceptance
- Heat treatment verification
- Weld zone characterization
- Coating and surface treatment evaluation
- Polymer and rubber hardness testing
- Mineralogical classification
- Hardness-strength correlations

Cross-Validation Opportunities:
- Vickers ↔ Rockwell conversions (scale-dependent)
- Brinell ↔ Vickers correlations
- Hardness ↔ Tensile strength relationships
- Micro vs macro hardness comparisons
- Shore ↔ Tensile modulus for polymers

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-02
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime


class HardnessTestingAgent:
    """
    Comprehensive hardness testing agent for materials characterization.

    Supports multiple hardness scales from Shore 00 (soft elastomers) to
    Vickers HV30000 (ultra-hard ceramics).
    """

    VERSION = "1.0.0"
    AGENT_TYPE = "hardness_testing"

    # Supported hardness testing techniques
    SUPPORTED_TECHNIQUES = [
        'vickers',              # HV - Universal microhardness
        'rockwell',             # HR - Multiple scales
        'brinell',              # HB - Bulk hardness
        'knoop',                # HK - Thin sections/coatings
        'shore',                # Shore A/D/00 - Polymers/elastomers
        'mohs',                 # Mineralogical scratch
        'hardness_profile',     # Depth profiling
        'micro_mapping',        # Spatial hardness maps
        'conversion'            # Between hardness scales
    ]

    # Vickers indenter geometry (diamond pyramid)
    VICKERS_INDENTER = {
        'angle': 136,  # degrees
        'shape': 'square_pyramid',
        'geometry_factor': 1854.4  # For HV calculation
    }

    # Rockwell scales and their specifications
    ROCKWELL_SCALES = {
        'HRC': {
            'indenter': 'diamond_cone',
            'major_load_kgf': 150,
            'minor_load_kgf': 10,
            'typical_range': (20, 70),
            'material': 'hardened_steel_carbides'
        },
        'HRB': {
            'indenter': '1.588mm_ball',
            'major_load_kgf': 100,
            'minor_load_kgf': 10,
            'typical_range': (0, 100),
            'material': 'soft_steel_aluminum_brass'
        },
        'HRA': {
            'indenter': 'diamond_cone',
            'major_load_kgf': 60,
            'minor_load_kgf': 10,
            'typical_range': (20, 88),
            'material': 'carbides_thin_steel'
        },
        'HRD': {
            'indenter': 'diamond_cone',
            'major_load_kgf': 100,
            'minor_load_kgf': 10,
            'typical_range': (40, 77),
            'material': 'thin_hard_case_hardened'
        },
        'HRE': {
            'indenter': '3.175mm_ball',
            'major_load_kgf': 100,
            'minor_load_kgf': 10,
            'typical_range': (0, 100),
            'material': 'soft_materials_plastics'
        },
        'HRF': {
            'indenter': '1.588mm_ball',
            'major_load_kgf': 60,
            'minor_load_kgf': 10,
            'typical_range': (0, 100),
            'material': 'annealed_copper_thin_soft'
        }
    }

    # Brinell ball diameters (mm)
    BRINELL_BALLS = {
        'standard': 10.0,
        'small': 5.0,
        'micro': 2.5,
        'mini': 1.0
    }

    # Knoop indenter geometry (elongated pyramid)
    KNOOP_INDENTER = {
        'long_to_short_ratio': 7.11,
        'geometry_factor': 14.229,  # For HK calculation
        'shape': 'elongated_pyramid'
    }

    # Shore durometer scales
    SHORE_SCALES = {
        'Shore_00': {
            'range': (0, 100),
            'typical_materials': 'very_soft_rubbers_gels',
            'spring_force': 'very_light'
        },
        'Shore_A': {
            'range': (0, 100),
            'typical_materials': 'soft_rubbers_elastomers',
            'spring_force': 'light'
        },
        'Shore_D': {
            'range': (0, 100),
            'typical_materials': 'hard_rubbers_rigid_plastics',
            'spring_force': 'heavy'
        }
    }

    # Mohs hardness reference minerals
    MOHS_SCALE = {
        1: 'Talc',
        2: 'Gypsum',
        3: 'Calcite',
        4: 'Fluorite',
        5: 'Apatite',
        6: 'Orthoclase',
        7: 'Quartz',
        8: 'Topaz',
        9: 'Corundum',
        10: 'Diamond'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the HardnessTestingAgent.

        Args:
            config: Configuration dictionary containing:
                - default_technique: 'vickers', 'rockwell', 'brinell', etc.
                - temperature_control: True/False for environmental control
                - automated: True/False for automated testing
                - load_accuracy: ±% load accuracy
                - measurement_resolution: µm for optical measurement
        """
        self.config = config or {}
        self.default_technique = self.config.get('default_technique', 'vickers')
        self.temperature_control = self.config.get('temperature_control', True)
        self.automated = self.config.get('automated', True)
        self.load_accuracy = self.config.get('load_accuracy', 0.5)  # ±0.5%
        self.measurement_resolution = self.config.get('measurement_resolution', 0.1)  # µm

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hardness testing based on requested technique.

        Args:
            input_data: Dictionary containing:
                - technique: Hardness test type
                - material_info: Material description
                - test_parameters: Technique-specific parameters

        Returns:
            Comprehensive hardness test results with metadata
        """
        technique = input_data.get('technique', self.default_technique)

        if technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(f"Unsupported technique: {technique}. "
                           f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Route to appropriate technique
        technique_map = {
            'vickers': self._execute_vickers,
            'rockwell': self._execute_rockwell,
            'brinell': self._execute_brinell,
            'knoop': self._execute_knoop,
            'shore': self._execute_shore,
            'mohs': self._execute_mohs,
            'hardness_profile': self._execute_hardness_profile,
            'micro_mapping': self._execute_micro_mapping,
            'conversion': self._execute_conversion
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

    def _execute_vickers(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Vickers microhardness testing.

        Universal hardness scale using diamond pyramid indenter. Suitable for
        all materials from soft to very hard.

        Formula: HV = 1854.4 × F / d²
        where F = load (N), d = diagonal length (µm)

        Args:
            input_data: Contains load, indentation measurements, material info

        Returns:
            Vickers hardness with statistical analysis
        """
        load_n = input_data.get('load_n', 9.807)  # Default 1 kgf
        load_kgf = load_n / 9.807

        # Get diagonal measurements (multiple indents for statistics)
        diagonals_um = input_data.get('diagonal_measurements_um', [])

        if not diagonals_um:
            # Simulate typical measurements with 2% variability
            expected_hardness = input_data.get('expected_hardness_hv', 200)
            expected_diagonal = np.sqrt(1854.4 * load_n / expected_hardness)
            diagonals_um = expected_diagonal * (1 + 0.02 * np.random.randn(5))

        # Calculate hardness for each indent
        hardness_values = []
        for d in diagonals_um:
            hv = self._calculate_vickers_hardness(load_n, d)
            hardness_values.append(hv)

        hardness_values = np.array(hardness_values)

        # Statistical analysis
        mean_hv = np.mean(hardness_values)
        std_hv = np.std(hardness_values, ddof=1)
        uncertainty_hv = std_hv / np.sqrt(len(hardness_values))
        cv_percent = (std_hv / mean_hv) * 100 if mean_hv > 0 else 0

        # Determine load designation (HV1, HV0.5, etc.)
        load_designation = f"HV{load_kgf:.2g}"

        # Material classification based on hardness
        material_class = self._classify_material_by_vickers(mean_hv)

        # Estimate tensile strength (empirical correlation for steels)
        estimated_tensile_mpa = self._vickers_to_tensile_strength(mean_hv)

        return {
            'technique': 'vickers_microhardness',
            'hardness_value': mean_hv,
            'hardness_std': std_hv,
            'hardness_uncertainty': uncertainty_hv,
            'coefficient_of_variation_percent': cv_percent,
            'unit': load_designation,
            'load_kgf': load_kgf,
            'load_n': load_n,
            'number_of_indents': len(hardness_values),
            'individual_measurements': hardness_values.tolist(),
            'diagonal_measurements_um': diagonals_um,
            'material_classification': material_class,
            'estimated_tensile_strength_mpa': estimated_tensile_mpa,
            'indenter': self.VICKERS_INDENTER,
            'quality_metrics': {
                'measurement_uniformity': 'excellent' if cv_percent < 3 else 'good' if cv_percent < 5 else 'acceptable',
                'statistical_confidence': '95%' if len(hardness_values) >= 5 else '68%'
            },
            'recommendations': self._generate_vickers_recommendations(mean_hv, cv_percent)
        }

    def _calculate_vickers_hardness(self, load_n: float, diagonal_um: float) -> float:
        """
        Calculate Vickers hardness from load and diagonal measurement.

        Args:
            load_n: Applied load in Newtons
            diagonal_um: Average diagonal length in micrometers

        Returns:
            Vickers hardness number (HV)
        """
        if diagonal_um <= 0:
            raise ValueError("Diagonal measurement must be positive")

        # HV = 1854.4 × F / d²  (F in N, d in µm)
        hv = self.VICKERS_INDENTER['geometry_factor'] * load_n / (diagonal_um ** 2)
        return hv

    def _execute_rockwell(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Rockwell hardness testing.

        Rapid hardness testing with direct readout, multiple scales for different
        material ranges.

        Args:
            input_data: Contains scale type, material info, test parameters

        Returns:
            Rockwell hardness results with scale information
        """
        scale = input_data.get('scale', 'HRC')

        if scale not in self.ROCKWELL_SCALES:
            raise ValueError(f"Invalid Rockwell scale: {scale}. "
                           f"Available: {list(self.ROCKWELL_SCALES.keys())}")

        scale_info = self.ROCKWELL_SCALES[scale]

        # Get measurements (multiple readings for statistics)
        measurements = input_data.get('measurements', [])

        if not measurements:
            # Simulate typical measurements
            expected_value = input_data.get('expected_hardness', 50)
            measurements = expected_value + 1.0 * np.random.randn(5)

        measurements = np.array(measurements)

        # Statistical analysis
        mean_hr = np.mean(measurements)
        std_hr = np.std(measurements, ddof=1)
        uncertainty_hr = std_hr / np.sqrt(len(measurements))

        # Check if in valid range
        valid_range = scale_info['typical_range']
        in_range = valid_range[0] <= mean_hr <= valid_range[1]

        # Convert to Vickers for comparison
        equivalent_hv = self._rockwell_to_vickers(mean_hr, scale)

        return {
            'technique': 'rockwell_hardness',
            'hardness_value': mean_hr,
            'hardness_std': std_hr,
            'hardness_uncertainty': uncertainty_hr,
            'unit': scale,
            'scale_information': scale_info,
            'number_of_measurements': len(measurements),
            'individual_measurements': measurements.tolist(),
            'in_valid_range': in_range,
            'valid_range': valid_range,
            'equivalent_vickers_hv': equivalent_hv,
            'quality_metrics': {
                'repeatability': 'excellent' if std_hr < 1 else 'good' if std_hr < 2 else 'acceptable',
                'range_validity': 'valid' if in_range else 'out_of_range'
            },
            'recommendations': self._generate_rockwell_recommendations(mean_hr, scale, in_range)
        }

    def _execute_brinell(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Brinell hardness testing.

        Large indentation for bulk hardness measurement, particularly suitable
        for heterogeneous materials (castings, forgings).

        Formula: HB = 2F / (πD(D - √(D² - d²)))
        where F = load (kgf), D = ball diameter (mm), d = impression diameter (mm)

        Args:
            input_data: Contains load, ball diameter, impression measurements

        Returns:
            Brinell hardness results
        """
        load_kgf = input_data.get('load_kgf', 3000)
        ball_diameter_mm = input_data.get('ball_diameter_mm', 10.0)

        # Get impression diameter measurements
        impression_diameters_mm = input_data.get('impression_diameters_mm', [])

        if not impression_diameters_mm:
            # Simulate based on expected hardness
            expected_hb = input_data.get('expected_hardness_hb', 200)
            # Approximate d from HB (iterative, simplified here)
            d_approx = ball_diameter_mm * 0.4  # Typical ~40% of ball diameter
            impression_diameters_mm = d_approx * (1 + 0.02 * np.random.randn(3))

        # Calculate hardness for each impression
        hardness_values = []
        for d in impression_diameters_mm:
            hb = self._calculate_brinell_hardness(load_kgf, ball_diameter_mm, d)
            hardness_values.append(hb)

        hardness_values = np.array(hardness_values)

        # Statistical analysis
        mean_hb = np.mean(hardness_values)
        std_hb = np.std(hardness_values, ddof=1)
        uncertainty_hb = std_hb / np.sqrt(len(hardness_values))

        # Designation: HB(ball/load/time) e.g., HB10/3000/15
        designation = f"HB{ball_diameter_mm:.0f}/{load_kgf:.0f}"

        # Convert to Vickers
        equivalent_hv = mean_hb * 0.95  # Approximate conversion

        return {
            'technique': 'brinell_hardness',
            'hardness_value': mean_hb,
            'hardness_std': std_hb,
            'hardness_uncertainty': uncertainty_hb,
            'unit': designation,
            'load_kgf': load_kgf,
            'ball_diameter_mm': ball_diameter_mm,
            'impression_diameters_mm': impression_diameters_mm,
            'number_of_impressions': len(hardness_values),
            'individual_measurements': hardness_values.tolist(),
            'equivalent_vickers_hv': equivalent_hv,
            'quality_metrics': {
                'measurement_consistency': 'excellent' if std_hb < 5 else 'good' if std_hb < 10 else 'acceptable'
            },
            'recommendations': self._generate_brinell_recommendations(ball_diameter_mm, load_kgf)
        }

    def _calculate_brinell_hardness(self, load_kgf: float, ball_diameter_mm: float,
                                   impression_diameter_mm: float) -> float:
        """
        Calculate Brinell hardness from test parameters.

        Args:
            load_kgf: Applied load in kgf
            ball_diameter_mm: Ball diameter in mm
            impression_diameter_mm: Impression diameter in mm

        Returns:
            Brinell hardness number (HB)
        """
        D = ball_diameter_mm
        d = impression_diameter_mm
        F = load_kgf

        if d >= D:
            raise ValueError("Impression diameter cannot exceed ball diameter")

        # HB = 2F / (πD(D - √(D² - d²)))
        denominator = np.pi * D * (D - np.sqrt(D**2 - d**2))
        hb = 2 * F / denominator

        return hb

    def _execute_knoop(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Knoop microhardness testing.

        Elongated indentation ideal for thin layers, coatings, and measuring
        hardness directionality.

        Formula: HK = 14.229 × F / L²
        where F = load (N), L = long diagonal (µm)

        Args:
            input_data: Contains load, long diagonal measurements

        Returns:
            Knoop hardness results
        """
        load_n = input_data.get('load_n', 0.9807)  # Default 100 gf
        load_kgf = load_n / 9.807

        # Get long diagonal measurements
        long_diagonals_um = input_data.get('long_diagonal_measurements_um', [])

        if not long_diagonals_um:
            # Simulate based on expected hardness
            expected_hk = input_data.get('expected_hardness_hk', 200)
            expected_diagonal = np.sqrt(self.KNOOP_INDENTER['geometry_factor'] * load_n / expected_hk)
            long_diagonals_um = expected_diagonal * (1 + 0.02 * np.random.randn(5))

        # Calculate hardness for each indent
        hardness_values = []
        for L in long_diagonals_um:
            hk = self._calculate_knoop_hardness(load_n, L)
            hardness_values.append(hk)

        hardness_values = np.array(hardness_values)

        # Statistical analysis
        mean_hk = np.mean(hardness_values)
        std_hk = np.std(hardness_values, ddof=1)
        uncertainty_hk = std_hk / np.sqrt(len(hardness_values))

        # Load designation
        load_designation = f"HK{load_kgf:.3g}"

        return {
            'technique': 'knoop_microhardness',
            'hardness_value': mean_hk,
            'hardness_std': std_hk,
            'hardness_uncertainty': uncertainty_hk,
            'unit': load_designation,
            'load_kgf': load_kgf,
            'load_n': load_n,
            'long_diagonal_measurements_um': long_diagonals_um,
            'number_of_indents': len(hardness_values),
            'individual_measurements': hardness_values.tolist(),
            'indenter_geometry': self.KNOOP_INDENTER,
            'quality_metrics': {
                'measurement_precision': 'excellent' if std_hk < mean_hk * 0.03 else 'good'
            },
            'applications': [
                'Thin coating hardness',
                'Anisotropic materials',
                'Brittle materials (less cracking than Vickers)',
                'Case depth profiling'
            ]
        }

    def _calculate_knoop_hardness(self, load_n: float, long_diagonal_um: float) -> float:
        """
        Calculate Knoop hardness from load and long diagonal.

        Args:
            load_n: Applied load in Newtons
            long_diagonal_um: Long diagonal length in micrometers

        Returns:
            Knoop hardness number (HK)
        """
        if long_diagonal_um <= 0:
            raise ValueError("Diagonal measurement must be positive")

        # HK = 14.229 × F / L²
        hk = self.KNOOP_INDENTER['geometry_factor'] * load_n / (long_diagonal_um ** 2)
        return hk

    def _execute_shore(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Shore durometer testing for polymers and elastomers.

        Rapid indentation depth measurement for soft materials.

        Args:
            input_data: Contains scale type, measurements

        Returns:
            Shore hardness results
        """
        scale = input_data.get('scale', 'Shore_A')

        if scale not in self.SHORE_SCALES:
            raise ValueError(f"Invalid Shore scale: {scale}. "
                           f"Available: {list(self.SHORE_SCALES.keys())}")

        scale_info = self.SHORE_SCALES[scale]

        # Get measurements
        measurements = input_data.get('measurements', [])

        if not measurements:
            # Simulate typical measurements
            expected_value = input_data.get('expected_hardness', 60)
            measurements = expected_value + 2.0 * np.random.randn(5)
            measurements = np.clip(measurements, 0, 100)

        measurements = np.array(measurements)

        # Statistical analysis
        mean_shore = np.mean(measurements)
        std_shore = np.std(measurements, ddof=1)
        uncertainty_shore = std_shore / np.sqrt(len(measurements))

        # Estimate tensile modulus for Shore A (empirical correlation)
        if scale == 'Shore_A':
            modulus_mpa = self._shore_a_to_modulus(mean_shore)
        else:
            modulus_mpa = None

        return {
            'technique': 'shore_durometer',
            'hardness_value': mean_shore,
            'hardness_std': std_shore,
            'hardness_uncertainty': uncertainty_shore,
            'unit': scale,
            'scale_information': scale_info,
            'number_of_measurements': len(measurements),
            'individual_measurements': measurements.tolist(),
            'estimated_tensile_modulus_mpa': modulus_mpa,
            'quality_metrics': {
                'repeatability': 'excellent' if std_shore < 2 else 'good' if std_shore < 5 else 'acceptable'
            },
            'test_conditions': {
                'instantaneous_reading': 'Use for quick testing',
                'after_delay': 'Standard is 15 seconds after application'
            },
            'recommendations': self._generate_shore_recommendations(mean_shore, scale)
        }

    def _execute_mohs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Mohs scratch hardness testing.

        Mineralogical scratch resistance scale (1-10).

        Args:
            input_data: Contains scratch test results

        Returns:
            Mohs hardness classification
        """
        # Determine hardness from scratch tests
        scratched_by = input_data.get('scratched_by', [])
        scratches = input_data.get('scratches', [])

        if not scratched_by and not scratches:
            # Need to specify
            raise ValueError("Must provide either 'scratched_by' (minerals that scratch sample) "
                           "or 'scratches' (minerals sample scratches)")

        # Determine Mohs hardness range
        if scratched_by:
            mohs_max = min(scratched_by) - 0.5
        else:
            mohs_max = 10

        if scratches:
            mohs_min = max(scratches) + 0.5
        else:
            mohs_min = 1

        mohs_value = (mohs_min + mohs_max) / 2

        # Find reference minerals
        reference_minerals = {
            'lower_bound': self.MOHS_SCALE.get(int(np.floor(mohs_min)), 'Unknown'),
            'upper_bound': self.MOHS_SCALE.get(int(np.ceil(mohs_max)), 'Unknown'),
            'closest': self.MOHS_SCALE.get(round(mohs_value), 'Unknown')
        }

        return {
            'technique': 'mohs_hardness',
            'mohs_hardness': mohs_value,
            'mohs_range': (mohs_min, mohs_max),
            'reference_minerals': reference_minerals,
            'mohs_scale': self.MOHS_SCALE,
            'scratched_by_minerals': scratched_by,
            'scratches_minerals': scratches,
            'applications': [
                'Mineralogical identification',
                'Gemstone classification',
                'Abrasive selection',
                'Qualitative hardness ranking'
            ],
            'note': 'Mohs is a relative, non-linear scale. '
                   'Use Vickers/Knoop for quantitative hardness.'
        }

    def _execute_hardness_profile(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform depth-resolved hardness profiling.

        Map hardness as a function of depth for case hardening, coatings, welds.

        Args:
            input_data: Contains depth positions and hardness technique

        Returns:
            Hardness vs depth profile
        """
        technique = input_data.get('base_technique', 'vickers')
        depths_um = input_data.get('depths_um', np.linspace(0, 1000, 11))

        # Perform hardness testing at each depth
        profile_data = []
        for depth in depths_um:
            # Create input for individual measurement
            test_input = input_data.copy()
            test_input['technique'] = technique
            test_input['position_um'] = depth

            # Execute individual test
            result = self.execute(test_input)

            profile_data.append({
                'depth_um': depth,
                'hardness': result.get('hardness_value', 0),
                'hardness_uncertainty': result.get('hardness_uncertainty', 0)
            })

        # Analyze profile
        hardness_values = np.array([p['hardness'] for p in profile_data])
        surface_hardness = hardness_values[0]
        core_hardness = hardness_values[-1]
        max_hardness = np.max(hardness_values)

        # Estimate case depth (depth where hardness drops to 90% of surface)
        threshold = surface_hardness * 0.9
        case_depth_idx = np.where(hardness_values < threshold)[0]
        case_depth_um = depths_um[case_depth_idx[0]] if len(case_depth_idx) > 0 else depths_um[-1]

        return {
            'technique': 'hardness_depth_profile',
            'base_technique': technique,
            'profile_data': profile_data,
            'depths_um': depths_um.tolist(),
            'hardness_values': hardness_values.tolist(),
            'surface_hardness': surface_hardness,
            'core_hardness': core_hardness,
            'maximum_hardness': max_hardness,
            'estimated_case_depth_um': case_depth_um,
            'hardness_gradient': (surface_hardness - core_hardness) / (depths_um[-1] - depths_um[0]),
            'applications': [
                'Case hardening verification',
                'Carburizing depth',
                'Nitriding effectiveness',
                'Coating adhesion quality',
                'Weld heat-affected zone (HAZ) characterization'
            ]
        }

    def _execute_micro_mapping(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform 2D hardness mapping for spatial heterogeneity.

        Create hardness maps across sample surface.

        Args:
            input_data: Contains grid specification and technique

        Returns:
            2D hardness map data
        """
        technique = input_data.get('base_technique', 'vickers')

        # Grid specification
        x_range_um = input_data.get('x_range_um', (0, 1000))
        y_range_um = input_data.get('y_range_um', (0, 1000))
        grid_spacing_um = input_data.get('grid_spacing_um', 100)

        # Create grid
        x_positions = np.arange(x_range_um[0], x_range_um[1] + grid_spacing_um, grid_spacing_um)
        y_positions = np.arange(y_range_um[0], y_range_um[1] + grid_spacing_um, grid_spacing_um)

        # Simulate hardness map (in reality, would perform actual measurements)
        hardness_map = []
        for y in y_positions:
            row = []
            for x in x_positions:
                # Simulate spatial variation (example: gradient + noise)
                base_hardness = input_data.get('base_hardness', 200)
                spatial_variation = 20 * np.sin(2 * np.pi * x / 500) * np.cos(2 * np.pi * y / 500)
                noise = 5 * np.random.randn()
                hardness = base_hardness + spatial_variation + noise
                row.append(hardness)
            hardness_map.append(row)

        hardness_map = np.array(hardness_map)

        # Statistical analysis
        mean_hardness = np.mean(hardness_map)
        std_hardness = np.std(hardness_map)
        min_hardness = np.min(hardness_map)
        max_hardness = np.max(hardness_map)

        return {
            'technique': 'hardness_microstructure_mapping',
            'base_technique': technique,
            'x_positions_um': x_positions.tolist(),
            'y_positions_um': y_positions.tolist(),
            'hardness_map': hardness_map.tolist(),
            'map_shape': hardness_map.shape,
            'grid_spacing_um': grid_spacing_um,
            'statistics': {
                'mean_hardness': mean_hardness,
                'std_hardness': std_hardness,
                'min_hardness': min_hardness,
                'max_hardness': max_hardness,
                'coefficient_of_variation_percent': (std_hardness / mean_hardness) * 100
            },
            'applications': [
                'Microstructure correlation',
                'Phase distribution mapping',
                'Weld zone characterization',
                'Quality control of heat treatment',
                'Composite material characterization'
            ]
        }

    def _execute_conversion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert between hardness scales.

        Args:
            input_data: Contains source scale, value, target scale

        Returns:
            Converted hardness values
        """
        source_scale = input_data.get('source_scale', 'HV')
        source_value = input_data.get('source_value')
        target_scales = input_data.get('target_scales', ['HRC', 'HB'])

        if source_value is None:
            raise ValueError("Must provide source_value for conversion")

        conversions = {}

        for target in target_scales:
            if source_scale == 'HV' and target == 'HRC':
                conversions['HRC'] = self._vickers_to_rockwell_c(source_value)
            elif source_scale == 'HV' and target == 'HB':
                conversions['HB'] = source_value / 0.95
            elif source_scale == 'HRC' and target == 'HV':
                conversions['HV'] = self._rockwell_to_vickers(source_value, 'HRC')
            elif source_scale == 'HB' and target == 'HV':
                conversions['HV'] = source_value * 0.95
            else:
                conversions[target] = None

        return {
            'technique': 'hardness_scale_conversion',
            'source_scale': source_scale,
            'source_value': source_value,
            'conversions': conversions,
            'warning': 'Conversions are approximate and material-dependent. '
                      'Use with caution, especially across different hardness ranges.',
            'recommendations': [
                'Perform direct measurements when possible',
                'Verify conversions with known standards',
                'Consider material type in conversions',
                'Conversions most accurate in mid-ranges'
            ]
        }

    # Helper methods for conversions and correlations

    def _classify_material_by_vickers(self, hv: float) -> str:
        """Classify material hardness level."""
        if hv < 50:
            return 'very_soft (e.g., lead, soft polymers)'
        elif hv < 150:
            return 'soft (e.g., aluminum, annealed steel)'
        elif hv < 300:
            return 'medium (e.g., mild steel, brass)'
        elif hv < 500:
            return 'hard (e.g., hardened steel, hard brass)'
        elif hv < 900:
            return 'very_hard (e.g., tool steel, hardened stainless)'
        else:
            return 'extremely_hard (e.g., carbides, ceramics, diamond)'

    def _vickers_to_tensile_strength(self, hv: float) -> Optional[float]:
        """
        Estimate tensile strength from Vickers hardness (empirical for steels).

        Approximate relation: σ_UTS ≈ HV × 3.28 MPa (for steels)
        """
        return hv * 3.28

    def _rockwell_to_vickers(self, hr: float, scale: str) -> Optional[float]:
        """
        Convert Rockwell to Vickers (approximate).

        Based on ASTM E140 conversion tables.
        """
        if scale == 'HRC':
            # Approximate polynomial fit
            if 20 <= hr <= 70:
                hv = 92.9 + 19.4 * hr + 0.134 * hr**2
                return hv
        elif scale == 'HRB':
            if 0 <= hr <= 100:
                # Approximate
                hv = 20 + 2.0 * hr
                return hv
        return None

    def _vickers_to_rockwell_c(self, hv: float) -> Optional[float]:
        """
        Convert Vickers to Rockwell C (approximate, inverse of conversion).
        """
        if hv < 200:
            return None  # Below HRC range
        # Approximate inverse
        hrc = (-19.4 + np.sqrt(19.4**2 + 4 * 0.134 * (hv - 92.9))) / (2 * 0.134)
        return hrc if 20 <= hrc <= 70 else None

    def _shore_a_to_modulus(self, shore_a: float) -> float:
        """
        Estimate tensile modulus from Shore A hardness.

        Empirical correlation: log10(E) ≈ 0.0235 × Shore_A - 0.6403
        where E is in MPa
        """
        log_e = 0.0235 * shore_a - 0.6403
        modulus_mpa = 10 ** log_e
        return modulus_mpa

    def _generate_vickers_recommendations(self, hv: float, cv: float) -> List[str]:
        """Generate testing recommendations based on results."""
        recs = []

        if cv > 5:
            recs.append("High variability detected - check surface preparation")
            recs.append("Ensure adequate spacing between indents (>2.5× diagonal)")

        if hv < 50:
            recs.append("Consider using lower load for soft materials")

        if hv > 1000:
            recs.append("Material is very hard - verify diamond indenter quality")

        recs.append("Cross-validate with Rockwell or Brinell if applicable")

        return recs

    def _generate_rockwell_recommendations(self, hr: float, scale: str,
                                         in_range: bool) -> List[str]:
        """Generate Rockwell testing recommendations."""
        recs = []

        if not in_range:
            recs.append(f"Reading outside typical range for {scale}")
            recs.append("Consider switching to more appropriate scale")

        recs.append("Verify sample thickness (≥10× indentation depth)")
        recs.append("Ensure flat, parallel surfaces")

        return recs

    def _generate_brinell_recommendations(self, ball_mm: float,
                                         load_kgf: float) -> List[str]:
        """Generate Brinell testing recommendations."""
        recs = []

        recs.append(f"Using {ball_mm}mm ball with {load_kgf}kgf load")
        recs.append("Verify impression diameter: 0.24D < d < 0.6D for valid test")
        recs.append("Maintain load for 10-15 seconds")
        recs.append("Measure two perpendicular diameters for each impression")

        return recs

    def _generate_shore_recommendations(self, shore: float, scale: str) -> List[str]:
        """Generate Shore durometer recommendations."""
        recs = []

        if shore < 10:
            if scale != 'Shore_00':
                recs.append("Material very soft - consider Shore 00 scale")

        if shore > 90:
            if scale != 'Shore_D':
                recs.append("Material very hard - consider Shore D scale")

        recs.append("Take reading after 15 seconds for standard measurement")
        recs.append("Use flat surface with minimum thickness of 6mm")

        return recs

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and metadata."""
        return {
            'agent_type': self.AGENT_TYPE,
            'version': self.VERSION,
            'supported_techniques': self.SUPPORTED_TECHNIQUES,
            'hardness_scales': {
                'vickers': 'HV - Universal microhardness',
                'rockwell': f'HR - {len(self.ROCKWELL_SCALES)} scales',
                'brinell': 'HB - Bulk hardness',
                'knoop': 'HK - Thin sections',
                'shore': 'Shore A/D/00 - Polymers',
                'mohs': 'Mohs 1-10 - Mineralogical'
            },
            'measurement_ranges': {
                'vickers': '1-3000+ HV',
                'rockwell_c': '20-70 HRC',
                'brinell': '5-650 HB',
                'shore_a': '0-100 Shore A'
            },
            'cross_validation_opportunities': [
                'Vickers ↔ Rockwell conversion',
                'Brinell ↔ Vickers correlation',
                'Hardness → Tensile strength estimation',
                'Shore A → Tensile modulus for polymers'
            ]
        }


if __name__ == '__main__':
    # Example usage
    agent = HardnessTestingAgent()

    # Example 1: Vickers hardness test on hardened steel
    result_vickers = agent.execute({
        'technique': 'vickers',
        'load_n': 9.807,  # 1 kgf
        'diagonal_measurements_um': [42.5, 43.1, 42.8, 43.0, 42.6],
        'material': 'hardened_steel'
    })
    print("Vickers Test Result:")
    print(f"  HV1 = {result_vickers['hardness_value']:.1f} ± {result_vickers['hardness_uncertainty']:.1f}")
    print(f"  Material: {result_vickers['material_classification']}")
    print(f"  Est. Tensile Strength: {result_vickers['estimated_tensile_strength_mpa']:.0f} MPa")
    print()

    # Example 2: Rockwell C test
    result_rockwell = agent.execute({
        'technique': 'rockwell',
        'scale': 'HRC',
        'measurements': [58.5, 59.0, 58.8, 59.2, 58.7]
    })
    print("Rockwell Test Result:")
    print(f"  HRC = {result_rockwell['hardness_value']:.1f} ± {result_rockwell['hardness_uncertainty']:.1f}")
    print(f"  Equivalent HV ≈ {result_rockwell['equivalent_vickers_hv']:.0f}")
    print()

    # Example 3: Shore A durometer on rubber
    result_shore = agent.execute({
        'technique': 'shore',
        'scale': 'Shore_A',
        'measurements': [62, 64, 63, 62, 63]
    })
    print("Shore A Test Result:")
    print(f"  Shore A = {result_shore['hardness_value']:.1f} ± {result_shore['hardness_uncertainty']:.1f}")
    print(f"  Est. Modulus ≈ {result_shore['estimated_tensile_modulus_mpa']:.2f} MPa")
