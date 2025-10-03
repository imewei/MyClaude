"""X-ray Scattering Specialist Agent.

VERSION 1.0.0 - Extracted from XRayAgent (scattering techniques only).

This agent specializes in X-ray scattering techniques for structural characterization
in reciprocal space (q-space). Covers length scales from nanometers to micrometers.

Capabilities:
- SAXS (Small-Angle X-ray Scattering): Nanostructure, particle size, Rg
- WAXS (Wide-Angle X-ray Scattering): Crystallinity, d-spacings, orientation
- GISAXS (Grazing Incidence SAXS): Thin film morphology, surface structure
- RSoXS (Resonant Soft X-ray Scattering): Chemical contrast, phase separation
- XPCS (X-ray Photon Correlation Spectroscopy): Slow dynamics, relaxation
- Time-resolved scattering: Kinetics, phase transitions, processing

Cross-Validation:
- SAXS ↔ DLS (particle size comparison)
- SAXS ↔ TEM (real vs reciprocal space)
- WAXS ↔ DSC (crystallinity vs thermal transitions)
- GISAXS ↔ AFM (thin film morphology)
"""

from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime

from base_agent import (
    ExperimentalAgent,
    AgentStatus,
    AgentResult,
    Capability,
    ValidationResult,
    AgentMetadata,
)


class XRayScatteringAgent(ExperimentalAgent):
    """X-ray scattering agent for structural characterization.

    Scattering techniques measure structure by analyzing diffraction patterns
    in reciprocal space (q-space). Provides information on:
    - Length scales: 1 nm - 1000 nm
    - Crystalline and amorphous materials
    - Bulk and thin film samples
    - Static structure and dynamics
    """

    NAME = "XRayScatteringAgent"
    VERSION = "1.0.0"

    SUPPORTED_TECHNIQUES = [
        'saxs',           # Small-angle X-ray scattering
        'waxs',           # Wide-angle X-ray scattering
        'gisaxs',         # Grazing incidence SAXS
        'rsoxs',          # Resonant soft X-ray scattering
        'xpcs',           # X-ray photon correlation spectroscopy
        'time_resolved',  # Time-resolved scattering
    ]

    # q-range coverage for different techniques (Å⁻¹)
    Q_RANGES = {
        'saxs': (0.001, 0.5),      # 1-600 nm features
        'waxs': (0.5, 3.0),        # 2-12 Å features (crystalline)
        'gisaxs': (0.001, 0.2),    # Thin film in-plane structure
        'rsoxs': (0.001, 0.1),     # Soft X-ray, nanoscale morphology
        'xpcs': (0.001, 0.1),      # Dynamics at specific q
    }

    def __init__(self):
        """Initialize XRayScatteringAgent."""
        metadata = AgentMetadata(
            name=self.NAME,
            version=self.VERSION,
            capabilities=self._define_capabilities(),
            dependencies=["numpy", "scipy"],
        )
        super().__init__(metadata)

    def _define_capabilities(self) -> List[Capability]:
        """Define agent capabilities."""
        return [
            Capability(
                name="SAXS",
                description="Small-angle X-ray scattering for nanostructure analysis",
                parameters={
                    "q_range": "Scattering vector range (Å⁻¹), default [0.001, 0.5]",
                    "n_points": "Number of data points, default 100",
                    "model": "Form factor model (sphere, cylinder, etc.)",
                },
                outputs=[
                    "Radius of gyration (Rg)",
                    "Particle size and polydispersity",
                    "Porod analysis (surface area)",
                    "Structure factor (inter-particle correlations)",
                    "Fractal dimension",
                ],
            ),
            Capability(
                name="WAXS",
                description="Wide-angle X-ray scattering for crystalline structure",
                parameters={
                    "q_range": "Scattering vector range (Å⁻¹), default [0.5, 3.0]",
                    "n_points": "Number of data points, default 200",
                },
                outputs=[
                    "Crystallinity percentage",
                    "d-spacings and Miller indices",
                    "Crystal orientation (Herman's parameter)",
                    "Peak positions and widths",
                    "Amorphous fraction",
                ],
            ),
            Capability(
                name="GISAXS",
                description="Grazing incidence SAXS for thin film morphology",
                parameters={
                    "qxy_range": "In-plane scattering vector (Å⁻¹)",
                    "qz_range": "Out-of-plane scattering vector (Å⁻¹)",
                    "incident_angle_deg": "Grazing angle, typically 0.1-0.5°",
                },
                outputs=[
                    "In-plane correlation length",
                    "Domain spacing and ordering",
                    "Film thickness and roughness",
                    "Morphology type (cylinders, lamellae, etc.)",
                    "Orientation relative to substrate",
                ],
            ),
            Capability(
                name="RSoXS",
                description="Resonant soft X-ray scattering for chemical contrast",
                parameters={
                    "energy_ev": "Photon energy (eV), near absorption edges",
                    "q_range": "Scattering vector range (Å⁻¹)",
                },
                outputs=[
                    "Domain spacing with chemical selectivity",
                    "Domain purity and composition",
                    "Phase separation length scales",
                    "Interface width and mixing",
                    "Electronic structure information",
                ],
            ),
            Capability(
                name="XPCS",
                description="X-ray photon correlation spectroscopy for slow dynamics",
                parameters={
                    "q_value": "Fixed scattering vector (Å⁻¹)",
                    "max_time_sec": "Maximum delay time for correlation",
                },
                outputs=[
                    "Intensity correlation function g2(t)",
                    "Relaxation times and stretching exponents",
                    "Diffusion coefficients",
                    "Dynamic heterogeneity",
                    "Aging and non-equilibrium behavior",
                ],
            ),
            Capability(
                name="Time-Resolved",
                description="Time-resolved scattering for kinetics and phase transitions",
                parameters={
                    "time_resolution_ms": "Temporal resolution (milliseconds)",
                    "duration_sec": "Total measurement time (seconds)",
                    "base_technique": "Underlying scattering technique (SAXS, WAXS)",
                },
                outputs=[
                    "Structural evolution over time",
                    "Kinetic rate constants",
                    "Phase transition mechanisms",
                    "Avrami exponents for crystallization",
                    "Intermediate phases",
                ],
            ),
        ]

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute X-ray scattering analysis."""
        try:
            technique = input_data.get('technique', 'saxs').lower()

            if technique not in self.SUPPORTED_TECHNIQUES:
                return AgentResult(
                    agent_name=self.metadata.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=[f"Unsupported technique: {technique}. "
                           f"Supported: {', '.join(self.SUPPORTED_TECHNIQUES)}"],
                    warnings=[]
                )

            # Route to appropriate analysis method
            if technique == 'saxs':
                result_data = self._execute_saxs(input_data)
            elif technique == 'waxs':
                result_data = self._execute_waxs(input_data)
            elif technique == 'gisaxs':
                result_data = self._execute_gisaxs(input_data)
            elif technique == 'rsoxs':
                result_data = self._execute_rsoxs(input_data)
            elif technique == 'xpcs':
                result_data = self._execute_xpcs(input_data)
            elif technique == 'time_resolved':
                result_data = self._execute_time_resolved(input_data)
            else:
                raise ValueError(f"Routing error for technique: {technique}")

            # Add metadata
            result_data['timestamp'] = datetime.now().isoformat()
            result_data['agent'] = self.NAME
            result_data['version'] = self.VERSION

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                errors=[],
                warnings=self._generate_warnings(technique, input_data)
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=[str(e)],
                warnings=[]
            )

    def _execute_saxs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SAXS analysis for bulk structure.

        SAXS provides information on:
        - Particle size and shape
        - Structure factor (inter-particle correlations)
        - Fractal dimension
        - Radius of gyration (Rg)
        """
        # Simulate SAXS data processing
        q_range = input_data.get('q_range', [0.001, 0.5])  # Å^-1
        n_points = input_data.get('n_points', 100)

        q = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), n_points)

        # Guinier analysis (low-q region)
        rg_nm = 5.0  # Radius of gyration
        i0 = 1000.0  # Forward scattering
        guinier_region = q < 1.0 / rg_nm

        # Porod analysis (high-q region)
        porod_constant = 1e6  # Surface area related
        porod_region = q > 0.1

        # Form factor (simplified sphere)
        radius_nm = 8.0
        volume = (4/3) * np.pi * (radius_nm * 10)**3  # Å^3

        intensity = np.zeros_like(q)
        for i, q_val in enumerate(q):
            if q_val * radius_nm * 10 < 0.1:
                # Guinier regime
                intensity[i] = i0 * np.exp(-(q_val * rg_nm * 10)**2 / 3)
            else:
                # Porod regime
                intensity[i] = porod_constant / q_val**4

        # Add realistic noise
        noise = np.random.normal(0, 0.05 * intensity)
        intensity += noise

        return {
            'technique': 'SAXS',
            'scattering_vector_inv_angstrom': q.tolist(),
            'intensity_arbitrary_units': intensity.tolist(),
            'guinier_analysis': {
                'radius_of_gyration_nm': rg_nm,
                'forward_scattering_i0': i0,
                'guinier_region_valid': True
            },
            'porod_analysis': {
                'porod_constant': porod_constant,
                'specific_surface_area_m2_g': 150.0,  # Estimated
                'interface_sharpness': 'sharp'  # vs 'diffuse'
            },
            'form_factor_fit': {
                'model': 'sphere',
                'radius_nm': radius_nm,
                'polydispersity': 0.15,
                'chi_squared': 1.2
            },
            'structure_factor': {
                'peak_position_inv_angstrom': 0.05,
                'correlation_length_nm': 12.0,
                'structure_factor_model': 'hard_sphere'
            },
            'physical_properties': {
                'particle_size_nm': radius_nm * 2,
                'size_distribution_width': 0.15,
                'aggregation_state': 'dispersed',
                'fractal_dimension': None
            }
        }

    def _execute_waxs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute WAXS analysis for crystalline structure.

        WAXS provides:
        - Crystallinity percentage
        - d-spacings
        - Crystal orientation
        - Polymorphs identification
        """
        q_range = input_data.get('q_range', [0.5, 3.0])  # Å^-1
        n_points = input_data.get('n_points', 200)

        q = np.linspace(q_range[0], q_range[1], n_points)

        # Simulate crystalline peaks + amorphous halo
        intensity = 100 * np.ones_like(q)  # Background

        # Crystalline peaks (d-spacings ~ 4-5 Å)
        peak_positions = [1.2, 1.5, 2.0]  # Å^-1
        peak_widths = [0.05, 0.05, 0.04]
        peak_heights = [800, 500, 300]

        for pos, width, height in zip(peak_positions, peak_widths, peak_heights):
            intensity += height * np.exp(-((q - pos) / width)**2)

        # Amorphous halo
        amorphous_center = 1.4
        amorphous_width = 0.3
        intensity += 400 * np.exp(-((q - amorphous_center) / amorphous_width)**2)

        # Calculate crystallinity
        crystalline_area = sum(peak_heights) * 0.05
        total_area = np.trapz(intensity, q)
        crystallinity = (crystalline_area / total_area) * 100

        return {
            'technique': 'WAXS',
            'scattering_vector_inv_angstrom': q.tolist(),
            'intensity_arbitrary_units': intensity.tolist(),
            'crystalline_peaks': [
                {
                    'q_position_inv_angstrom': pos,
                    'd_spacing_angstrom': 2 * np.pi / pos,
                    'fwhm_inv_angstrom': width,
                    'relative_intensity': height / max(peak_heights),
                    'miller_indices': '(hkl)'  # Would require indexing
                }
                for pos, width, height in zip(peak_positions, peak_widths, peak_heights)
            ],
            'crystallinity_analysis': {
                'crystallinity_percent': crystallinity,
                'crystalline_phase': 'alpha',  # Example
                'amorphous_fraction_percent': 100 - crystallinity
            },
            'orientation': {
                'orientation_parameter': 0.6,  # Herman's orientation
                'preferred_direction': 'in_plane',
                'texture_quality': 'moderate'
            }
        }

    def _execute_gisaxs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GISAXS analysis for thin film morphology.

        GISAXS provides:
        - In-plane structure
        - Out-of-plane structure
        - Domain orientation
        - Thin film morphology
        """
        # 2D scattering pattern analysis
        qxy_range = input_data.get('qxy_range', [0.001, 0.1])  # in-plane
        qz_range = input_data.get('qz_range', [0.001, 0.2])   # out-of-plane

        # Simulate typical GISAXS features
        correlation_length_nm = 30.0
        domain_spacing_nm = 25.0

        return {
            'technique': 'GISAXS',
            'in_plane_structure': {
                'qxy_peak_position_inv_angstrom': 2 * np.pi / (domain_spacing_nm * 10),
                'correlation_length_nm': correlation_length_nm,
                'domain_spacing_nm': domain_spacing_nm,
                'ordering_quality': 0.75  # 0-1 scale
            },
            'out_of_plane_structure': {
                'film_thickness_nm': 50.0,
                'interface_roughness_nm': 2.5,
                'vertical_correlation': 'weak',
                'layer_spacing_nm': None  # For multilayers
            },
            'morphology': {
                'structure_type': 'hexagonal_cylinders',  # lamellar, spheres, etc.
                'orientation': 'perpendicular',  # to substrate
                'grain_size_nm': 100.0,
                'defect_density': 'low'
            },
            'scattering_features': {
                'yoneda_peak_present': True,
                'correlation_peak_present': True,
                'form_factor_oscillations': False
            },
            'substrate_interaction': {
                'wetting_behavior': 'neutral',  # preferential, etc.
                'interface_quality': 'sharp',
                'critical_angle_deg': 0.15
            }
        }

    def _execute_rsoxs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RSoXS analysis for chemical contrast.

        RSoXS (Resonant Soft X-ray Scattering) provides:
        - Chemical composition maps
        - Domain purity
        - Phase separation length scales
        - Electronic structure information
        """
        energy_ev = input_data.get('energy_ev', 284.0)  # Carbon K-edge

        return {
            'technique': 'RSoXS',
            'resonant_energy_ev': energy_ev,
            'chemical_contrast': {
                'domain_spacing_nm': 20.0,
                'domain_purity': 0.85,  # 0-1 scale
                'composition_profile': 'sharp',  # vs 'diffuse'
                'mixing_parameter': 0.15  # Flory-Huggins chi
            },
            'energy_scan': {
                'energy_range_ev': [280, 290],
                'resonant_peaks_ev': [284.2, 285.5, 287.1],
                'peak_assignments': ['C=C', 'C-H', 'C=O'],
                'contrast_variation': 'significant'
            },
            'phase_separation': {
                'length_scale_nm': 18.0,
                'interface_width_nm': 3.0,
                'morphology': 'bicontinuous',
                'connectivity': 'high'
            },
            'electronic_structure': {
                'pi_star_transitions': True,
                'sigma_star_transitions': True,
                'orientation_dependence': 'anisotropic'
            }
        }

    def _execute_xpcs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute XPCS analysis for slow dynamics.

        XPCS (X-ray Photon Correlation Spectroscopy) provides:
        - Dynamics over seconds to hours
        - Relaxation times
        - Collective motion
        - Non-equilibrium dynamics
        """
        q_value = input_data.get('q_value', 0.01)  # Å^-1
        max_time_sec = input_data.get('max_time_sec', 100.0)

        # Simulate correlation function decay
        n_times = 50
        times = np.logspace(-1, np.log10(max_time_sec), n_times)

        # Exponential or stretched exponential decay
        beta = 0.8  # Stretching exponent
        tau = 10.0  # Relaxation time (seconds)

        g2 = 1 + 0.3 * np.exp(-(times / tau)**beta)

        return {
            'technique': 'XPCS',
            'q_value_inv_angstrom': q_value,
            'correlation_function': {
                'delay_times_sec': times.tolist(),
                'g2_minus_1': (g2 - 1).tolist(),
                'baseline': 1.0,
                'contrast_beta': 0.3
            },
            'dynamics_analysis': {
                'relaxation_time_sec': tau,
                'stretching_exponent': beta,
                'dynamics_type': 'compressed_exponential' if beta > 1 else 'stretched_exponential',
                'diffusion_coefficient_cm2_s': 1e-10
            },
            'q_dependence': {
                'scaling_exponent': 2.0,  # q^2 for diffusive
                'transport_mechanism': 'diffusive',
                'ballistic_regime_present': False
            },
            'non_equilibrium_features': {
                'aging_present': False,
                'two_step_relaxation': False,
                'dynamic_heterogeneity': 'moderate'
            }
        }

    def _execute_time_resolved(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute time-resolved X-ray scattering analysis.

        Time-resolved methods provide:
        - Kinetics of structural changes
        - Phase transitions
        - Chemical reactions
        - Processing dynamics
        """
        time_resolution = input_data.get('time_resolution_ms', 1.0)
        measurement_duration = input_data.get('duration_sec', 100.0)

        n_frames = int(measurement_duration * 1000 / time_resolution)
        times = np.linspace(0, measurement_duration, min(n_frames, 100))

        # Simulate structural evolution
        # Example: Crystallization kinetics (Avrami equation)
        k_rate = 0.1  # Rate constant
        n_avrami = 2.0  # Avrami exponent
        crystallinity = 1 - np.exp(-(k_rate * times)**n_avrami)

        return {
            'technique': 'Time-Resolved X-ray Scattering',
            'temporal_resolution_ms': time_resolution,
            'measurement_duration_sec': measurement_duration,
            'kinetics': {
                'times_sec': times.tolist(),
                'crystallinity_evolution': crystallinity.tolist(),
                'rate_constant': k_rate,
                'avrami_exponent': n_avrami,
                'half_time_sec': 1.0 / k_rate
            },
            'structural_changes': {
                'initial_phase': 'amorphous',
                'final_phase': 'crystalline',
                'transition_mechanism': 'nucleation_and_growth',
                'intermediate_phases': ['mesomorphic']
            },
            'process_conditions': {
                'temperature_controlled': True,
                'operando_measurement': True,
                'reversibility': 'irreversible'
            }
        }

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate X-ray scattering input."""
        errors = []
        warnings = []

        # Check technique
        if 'technique' not in data:
            errors.append("Missing required field: 'technique'")
        else:
            technique = data['technique'].lower()
            if technique not in self.SUPPORTED_TECHNIQUES:
                errors.append(f"Unsupported technique: {technique}")

        # Technique-specific validation
        technique = data.get('technique', '').lower()

        if technique == 'saxs':
            q_range = data.get('q_range', [0.001, 0.5])
            if q_range[0] >= q_range[1]:
                errors.append("q_range[0] must be less than q_range[1]")
            if q_range[0] < 0:
                errors.append("q_range values must be positive")

        elif technique == 'waxs':
            q_range = data.get('q_range', [0.5, 3.0])
            if q_range[0] < 0.3:
                warnings.append("WAXS typically starts at q > 0.3 Å⁻¹")

        elif technique == 'gisaxs':
            incident_angle = data.get('incident_angle_deg', 0.2)
            if incident_angle > 1.0:
                warnings.append("Incident angle > 1° may not be grazing incidence")

        elif technique == 'rsoxs':
            energy_ev = data.get('energy_ev', 284.0)
            if energy_ev < 100 or energy_ev > 2000:
                warnings.append("RSoXS typically uses 100-2000 eV (soft X-rays)")

        elif technique == 'xpcs':
            q_value = data.get('q_value', 0.01)
            if q_value < 0.001:
                warnings.append("Very low q may have insufficient coherence")
            max_time = data.get('max_time_sec', 100.0)
            if max_time > 10000:
                warnings.append("Very long measurements may have beam damage")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _generate_warnings(self, technique: str, input_data: Dict[str, Any]) -> List[str]:
        """Generate technique-specific warnings."""
        warnings = []

        if technique == 'saxs':
            if 'sample_concentration' in input_data:
                conc = input_data['sample_concentration']
                if conc > 50:  # mg/mL
                    warnings.append("High concentration may cause inter-particle effects")

        elif technique == 'waxs':
            if input_data.get('sample_thickness_um', 10) > 100:
                warnings.append("Thick samples may have absorption effects")

        elif technique == 'gisaxs':
            if input_data.get('film_thickness_nm', 50) < 10:
                warnings.append("Very thin films may have weak scattering signal")

        elif technique == 'xpcs':
            if input_data.get('max_time_sec', 100) > 1000:
                warnings.append("Long acquisition times may cause radiation damage")

        return warnings

    # ========================================================================
    # Cross-Validation Methods
    # ========================================================================

    @staticmethod
    def cross_validate_with_dls(saxs_result: Dict[str, Any],
                                 dls_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate SAXS particle size with DLS.

        Comparison:
        - SAXS: Number-averaged, all particles
        - DLS: Intensity-averaged, biased to large particles

        Expect DLS > SAXS for polydisperse samples.
        """
        saxs_size = saxs_result.get('physical_properties', {}).get('particle_size_nm', 0)
        dls_size = dls_result.get('hydrodynamic_diameter_nm', 0)

        ratio = dls_size / saxs_size if saxs_size > 0 else 0

        agreement = "good" if 0.9 < ratio < 1.5 else "poor"

        return {
            'comparison': 'SAXS vs DLS particle size',
            'saxs_size_nm': saxs_size,
            'dls_size_nm': dls_size,
            'ratio_dls_saxs': ratio,
            'agreement': agreement,
            'interpretation': (
                "DLS includes hydrodynamic layer; SAXS measures core structure. "
                f"Ratio {ratio:.2f} suggests {'minimal' if ratio < 1.2 else 'significant'} solvation."
            )
        }

    @staticmethod
    def cross_validate_with_tem(saxs_result: Dict[str, Any],
                                 tem_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate SAXS structure with TEM real-space imaging.

        Comparison:
        - SAXS: Reciprocal space, ensemble average
        - TEM: Real space, local imaging

        Both should agree on length scales and morphology.
        """
        saxs_size = saxs_result.get('physical_properties', {}).get('particle_size_nm', 0)
        tem_size = tem_result.get('average_particle_size_nm', 0)

        saxs_morphology = saxs_result.get('form_factor_fit', {}).get('model', 'unknown')
        tem_morphology = tem_result.get('morphology', 'unknown')

        size_agreement = abs(saxs_size - tem_size) / tem_size < 0.2 if tem_size > 0 else False

        return {
            'comparison': 'SAXS vs TEM structure',
            'saxs_size_nm': saxs_size,
            'tem_size_nm': tem_size,
            'saxs_morphology': saxs_morphology,
            'tem_morphology': tem_morphology,
            'size_agreement': 'good' if size_agreement else 'poor',
            'interpretation': (
                "SAXS provides ensemble average in solution/bulk; "
                "TEM shows dried individual particles. "
                f"Agreement {'confirms' if size_agreement else 'suggests aggregation or drying artifacts'}."
            )
        }

    @staticmethod
    def cross_validate_with_dsc(waxs_result: Dict[str, Any],
                                 dsc_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate WAXS crystallinity with DSC melting enthalpy.

        Comparison:
        - WAXS: Crystallinity from diffraction
        - DSC: Crystallinity from melting enthalpy

        Both methods should give similar crystallinity values.
        """
        waxs_crystallinity = waxs_result.get('crystallinity_analysis', {}).get('crystallinity_percent', 0)

        # Calculate DSC crystallinity from enthalpy
        delta_h_fusion = dsc_result.get('enthalpy_j_g', 0)
        delta_h_100 = 100.0  # J/g for 100% crystalline (material-specific)
        dsc_crystallinity = (delta_h_fusion / delta_h_100) * 100 if delta_h_100 > 0 else 0

        difference = abs(waxs_crystallinity - dsc_crystallinity)
        agreement = "good" if difference < 10 else "moderate" if difference < 20 else "poor"

        return {
            'comparison': 'WAXS vs DSC crystallinity',
            'waxs_crystallinity_percent': waxs_crystallinity,
            'dsc_crystallinity_percent': dsc_crystallinity,
            'absolute_difference': difference,
            'agreement': agreement,
            'interpretation': (
                "WAXS measures long-range order; DSC measures thermodynamic melting. "
                f"Difference {difference:.1f}% {'is acceptable' if difference < 10 else 'suggests different phases or orientations'}."
            )
        }

    @staticmethod
    def cross_validate_with_afm(gisaxs_result: Dict[str, Any],
                                 afm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate GISAXS thin film structure with AFM topography.

        Comparison:
        - GISAXS: In-plane structure (buried), reciprocal space
        - AFM: Surface topography, real space

        Domain spacing and roughness should correlate.
        """
        gisaxs_spacing = gisaxs_result.get('in_plane_structure', {}).get('domain_spacing_nm', 0)
        gisaxs_roughness = gisaxs_result.get('out_of_plane_structure', {}).get('interface_roughness_nm', 0)

        afm_roughness = afm_result.get('rms_roughness_nm', 0)
        afm_feature_size = afm_result.get('feature_size_nm', 0)

        roughness_ratio = gisaxs_roughness / afm_roughness if afm_roughness > 0 else 0
        spacing_ratio = gisaxs_spacing / afm_feature_size if afm_feature_size > 0 else 0

        return {
            'comparison': 'GISAXS vs AFM thin film structure',
            'gisaxs_domain_spacing_nm': gisaxs_spacing,
            'afm_feature_size_nm': afm_feature_size,
            'gisaxs_roughness_nm': gisaxs_roughness,
            'afm_roughness_nm': afm_roughness,
            'spacing_correlation': 'good' if 0.8 < spacing_ratio < 1.2 else 'poor',
            'roughness_correlation': 'good' if 0.5 < roughness_ratio < 2.0 else 'poor',
            'interpretation': (
                "GISAXS probes buried structure; AFM measures surface. "
                f"Spacing ratio {spacing_ratio:.2f} and roughness ratio {roughness_ratio:.2f} "
                f"{'confirm' if 0.8 < spacing_ratio < 1.2 else 'suggest surface/bulk differences'}."
            )
        }


# Example usage
if __name__ == "__main__":
    agent = XRayScatteringAgent()

    # Test SAXS
    saxs_input = {
        'technique': 'saxs',
        'q_range': [0.001, 0.5],
        'n_points': 100,
    }
    result = agent.execute(saxs_input)
    print(f"SAXS Result: {result.status}")
    if result.data:
        print(f"Particle size: {result.data.get('physical_properties', {}).get('particle_size_nm')} nm")
        print(f"Rg: {result.data.get('guinier_analysis', {}).get('radius_of_gyration_nm')} nm")

    # Test WAXS
    waxs_input = {
        'technique': 'waxs',
        'q_range': [0.5, 3.0],
    }
    result = agent.execute(waxs_input)
    print(f"\nWAXS Result: {result.status}")
    if result.data:
        print(f"Crystallinity: {result.data.get('crystallinity_analysis', {}).get('crystallinity_percent'):.1f}%")
