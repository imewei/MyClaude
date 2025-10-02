"""X-ray Scattering Agent for soft matter characterization.

This agent specializes in X-ray scattering and spectroscopy techniques:
- SAXS/WAXS: Small/wide angle scattering for structure
- GISAXS: Grazing incidence for thin films and interfaces
- RSoXS: Resonant soft X-ray for chemical contrast
- XPCS: X-ray photon correlation spectroscopy for dynamics
- XAS: X-ray absorption spectroscopy
- Time-resolved methods: Pump-probe and stroboscopic techniques

Expert in electron density contrast, high spatial resolution studies,
and operando characterization of soft matter systems.
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


class XRayAgent(ExperimentalAgent):
    """X-ray scattering and spectroscopy agent for soft matter systems.

    Capabilities:
    - SAXS/WAXS: Bulk structure analysis (Angstrom to micrometer)
    - GISAXS: Thin film and interface morphology
    - RSoXS: Resonant scattering for chemical contrast
    - XPCS: Slow dynamics characterization (seconds to hours)
    - XAS: Electronic structure and oxidation states
    - Time-resolved: Kinetics and dynamics studies

    Key advantages:
    - High spatial resolution (sub-nanometer)
    - Chemical contrast via resonant scattering
    - Fast acquisition for time-resolved studies
    - Operando capabilities
    """

    VERSION = "1.0.0"

    # Supported X-ray techniques
    SUPPORTED_TECHNIQUES = [
        'saxs',           # Small-angle X-ray scattering
        'waxs',           # Wide-angle X-ray scattering
        'gisaxs',         # Grazing incidence SAXS
        'rsoxs',          # Resonant soft X-ray scattering
        'xpcs',           # X-ray photon correlation spectroscopy
        'xas',            # X-ray absorption spectroscopy
        'time_resolved',  # Time-resolved techniques
    ]

    # Supported sample types
    SUPPORTED_SAMPLES = [
        'polymer',
        'colloid',
        'biomaterial',
        'thin_film',
        'liquid_crystal',
        'nanoparticle',
        'interface',
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize X-ray agent.

        Args:
            config: Configuration including:
                - beamline: Synchrotron beamline name
                - energy: X-ray energy in keV
                - detector: Detector type
                - sample_detector_distance: in mm
        """
        super().__init__(config)
        self.beamline = self.config.get('beamline', 'generic')
        self.energy_kev = self.config.get('energy', 10.0)
        self.detector = self.config.get('detector', 'pilatus')
        self.sample_detector_distance = self.config.get('sample_detector_distance', 1000.0)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute X-ray scattering/spectroscopy analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or data_array: X-ray data
                - q_range (optional): q-range for analysis
                - parameters: Technique-specific parameters

        Returns:
            AgentResult with X-ray analysis results
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

        # Route to technique-specific handler
        try:
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
            elif technique == 'xas':
                result_data = self._execute_xas(input_data)
            elif technique == 'time_resolved':
                result_data = self._execute_time_resolved(input_data)
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
                    'beamline': self.beamline,
                    'energy_kev': self.energy_kev,
                    **input_data.get('parameters', {})
                },
                execution_time_sec=execution_time,
                environment={
                    'detector': self.detector,
                    'sample_detector_distance_mm': self.sample_detector_distance
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'technique': technique,
                    'execution_time_sec': execution_time,
                    'beamline': self.beamline,
                    'energy_kev': self.energy_kev
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

    def _execute_xas(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute XAS analysis for electronic structure.

        XAS (X-ray Absorption Spectroscopy) provides:
        - Oxidation states
        - Local coordination
        - Electronic structure
        - Chemical speciation
        """
        element = input_data.get('element', 'Fe')
        edge = input_data.get('edge', 'K')

        # Simulate XANES and EXAFS regions
        energy_range_ev = input_data.get('energy_range_ev', [7000, 8000])

        return {
            'technique': 'XAS',
            'element': element,
            'edge': edge,
            'xanes_analysis': {
                'edge_position_ev': 7112.0,
                'oxidation_state': '+3',
                'coordination_geometry': 'octahedral',
                'pre_edge_features': True,
                'white_line_intensity': 1.5
            },
            'exafs_analysis': {
                'first_shell_distance_angstrom': 2.05,
                'coordination_number': 6.0,
                'debye_waller_factor': 0.005,
                'second_shell_present': True
            },
            'chemical_state': {
                'species': f'{element}2O3',
                'phase': 'crystalline',
                'defects_present': 'oxygen_vacancies',
                'mixed_valence': False
            },
            'electronic_structure': {
                'density_of_states_features': ['eg', 't2g'],
                'band_gap_ev': 2.1,
                'metal_ligand_hybridization': 'strong'
            }
        }

    def _execute_time_resolved(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute time-resolved X-ray analysis.

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
            'technique': 'Time-Resolved X-ray',
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
        """Validate X-ray analysis input."""
        errors = []
        warnings = []

        # Check technique
        if 'technique' not in data:
            errors.append("Missing required field: 'technique'")
        elif data['technique'].lower() not in self.SUPPORTED_TECHNIQUES:
            errors.append(f"Unsupported technique: {data['technique']}. "
                         f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Check data source
        if 'data_file' not in data and 'data_array' not in data:
            warnings.append("No data provided; will use simulated data")

        # Technique-specific validation
        technique = data.get('technique', '').lower()

        if technique == 'xpcs':
            if 'q_value' not in data.get('parameters', {}):
                warnings.append("XPCS: q_value not specified, using default")

        if technique == 'xas':
            if 'element' not in data.get('parameters', {}):
                warnings.append("XAS: element not specified, using default")
            if 'edge' not in data.get('parameters', {}):
                warnings.append("XAS: edge not specified, using K-edge")

        if technique == 'rsoxs':
            if 'energy_ev' not in data.get('parameters', {}):
                warnings.append("RSoXS: energy not specified, using C K-edge default")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources for X-ray analysis."""
        technique = data.get('technique', '').lower()

        # Base requirements
        cpu_cores = 1
        memory_gb = 2.0
        estimated_time_sec = 60.0

        # Adjust based on technique
        if technique == 'xpcs':
            # XPCS requires correlation analysis
            cpu_cores = 4
            memory_gb = 8.0
            estimated_time_sec = 300.0
        elif technique == 'gisaxs':
            # 2D pattern analysis
            cpu_cores = 2
            memory_gb = 4.0
            estimated_time_sec = 120.0
        elif technique == 'time_resolved':
            # Many frames
            n_frames = data.get('duration_sec', 100) / data.get('time_resolution_ms', 1.0) * 1000
            cpu_cores = 4
            memory_gb = max(4.0, n_frames * 0.01)
            estimated_time_sec = 180.0

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=0,
            estimated_time_sec=estimated_time_sec,
            execution_environment=ExecutionEnvironment.LOCAL
        )

    def get_capabilities(self) -> List[Capability]:
        """Return X-ray agent capabilities."""
        return [
            Capability(
                name="SAXS Analysis",
                description="Small-angle X-ray scattering for bulk structure",
                input_types=["scattering_data", "q_range"],
                output_types=["intensity_profile", "form_factor", "structure_factor"],
                typical_use_cases=[
                    "Particle size distribution",
                    "Polymer morphology",
                    "Protein structure in solution"
                ]
            ),
            Capability(
                name="GISAXS Analysis",
                description="Grazing incidence SAXS for thin film morphology",
                input_types=["2d_scattering_pattern", "incidence_angle"],
                output_types=["in_plane_structure", "out_of_plane_structure"],
                typical_use_cases=[
                    "Block copolymer thin films",
                    "Nanoparticle arrays",
                    "Surface morphology"
                ]
            ),
            Capability(
                name="RSoXS Analysis",
                description="Resonant soft X-ray scattering for chemical contrast",
                input_types=["energy_scan_data", "scattering_data"],
                output_types=["chemical_contrast", "domain_composition"],
                typical_use_cases=[
                    "Polymer blend phase separation",
                    "Organic solar cell morphology",
                    "Chemical mapping"
                ]
            ),
            Capability(
                name="XPCS Analysis",
                description="X-ray photon correlation spectroscopy for dynamics",
                input_types=["time_series_data", "q_value"],
                output_types=["correlation_function", "relaxation_times"],
                typical_use_cases=[
                    "Colloidal dynamics",
                    "Glass transition",
                    "Slow relaxation processes"
                ]
            ),
            Capability(
                name="XAS Analysis",
                description="X-ray absorption spectroscopy for electronic structure",
                input_types=["absorption_spectrum", "element", "edge"],
                output_types=["oxidation_state", "coordination", "electronic_structure"],
                typical_use_cases=[
                    "Oxidation state determination",
                    "Local structure",
                    "Chemical speciation"
                ]
            ),
            Capability(
                name="Time-Resolved X-ray",
                description="Fast time-resolved structural characterization",
                input_types=["time_series_scattering", "process_parameters"],
                output_types=["kinetics", "structural_evolution"],
                typical_use_cases=[
                    "Crystallization kinetics",
                    "Phase transitions",
                    "Operando characterization"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return X-ray agent metadata."""
        return AgentMetadata(
            name="XRayAgent",
            version=self.VERSION,
            description="X-ray scattering and spectroscopy expert for soft matter systems",
            author="Materials Science Multi-Agent System",
            capabilities=self.get_capabilities(),
            dependencies=[
                'numpy',
                'scipy',
                'matplotlib',
                'fabio',  # For reading detector images
                'pyFAI',  # For azimuthal integration
            ],
            supported_formats=[
                'tiff', 'edf', 'cbf',  # Detector formats
                'dat', 'chi',  # Reduced data
                'hdf5', 'nexus'  # Synchrotron formats
            ]
        )

    def connect_instrument(self) -> bool:
        """Connect to X-ray beamline (placeholder).

        Returns:
            True if connection successful
        """
        # In production: connect to beamline control system
        # e.g., EPICS, Tango, Sardana
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw X-ray detector data (placeholder).

        Args:
            raw_data: Raw detector images or spectra

        Returns:
            Processed data (calibrated, integrated, corrected)
        """
        # In production:
        # - Dark/flat field correction
        # - Spatial distortion correction
        # - Azimuthal integration
        # - Absolute intensity calibration
        return raw_data

    # Integration methods for cross-agent collaboration

    @staticmethod
    def validate_with_neutron_sans(xray_result: Dict[str, Any],
                                    sans_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate SAXS with neutron SANS data.

        Complements electron density contrast (X-ray) with
        scattering length density contrast (neutron).

        Args:
            xray_result: SAXS analysis result
            sans_result: SANS analysis result from neutron agent

        Returns:
            Validation report with consistency checks
        """
        saxs_rg = xray_result.get('guinier_analysis', {}).get('radius_of_gyration_nm', 0)
        sans_rg = sans_result.get('guinier_analysis', {}).get('radius_of_gyration_nm', 0)

        rg_agreement = abs(saxs_rg - sans_rg) / saxs_rg if saxs_rg > 0 else 1.0

        return {
            'validation_type': 'SAXS_SANS_cross_check',
            'rg_agreement_percent': (1 - rg_agreement) * 100,
            'consistent': rg_agreement < 0.1,  # Within 10%
            'contrast_mechanisms': {
                'xray': 'electron_density',
                'neutron': 'scattering_length_density'
            },
            'complementary_info': 'Neutron provides hydrogen sensitivity',
            'recommendation': 'Results consistent' if rg_agreement < 0.1
                            else 'Further investigation needed'
        }

    @staticmethod
    def correlate_with_microscopy(gisaxs_result: Dict[str, Any],
                                   tem_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate GISAXS structure with TEM imaging.

        Args:
            gisaxs_result: GISAXS morphology analysis
            tem_result: TEM imaging result from EM agent

        Returns:
            Correlation analysis
        """
        gisaxs_spacing = gisaxs_result.get('in_plane_structure', {}).get('domain_spacing_nm', 0)
        tem_spacing = tem_result.get('particle_analysis', {}).get('mean_spacing_nm', 0)

        spacing_match = abs(gisaxs_spacing - tem_spacing) / gisaxs_spacing if gisaxs_spacing > 0 else 1.0

        return {
            'correlation_type': 'GISAXS_TEM',
            'reciprocal_vs_real_space': {
                'gisaxs_q_space_nm': gisaxs_spacing,
                'tem_real_space_nm': tem_spacing,
                'agreement_percent': (1 - spacing_match) * 100
            },
            'complementarity': {
                'gisaxs_advantages': 'Bulk statistics, buried interfaces',
                'tem_advantages': 'Real space imaging, local defects'
            },
            'consistent': spacing_match < 0.15,
            'recommendation': 'Good agreement' if spacing_match < 0.15
                            else 'Check calibration or sample heterogeneity'
        }

    @staticmethod
    def extract_structure_for_simulation(saxs_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural parameters for MD simulation setup.

        Args:
            saxs_result: SAXS analysis with structure information

        Returns:
            Parameters formatted for simulation agent
        """
        return {
            'structure_type': 'from_saxs_analysis',
            'particle_size_nm': saxs_result.get('form_factor_fit', {}).get('radius_nm', 5.0) * 2,
            'polydispersity': saxs_result.get('form_factor_fit', {}).get('polydispersity', 0.1),
            'structure_factor_model': saxs_result.get('structure_factor', {}).get('structure_factor_model', 'hard_sphere'),
            'volume_fraction': 0.3,  # Would need additional info
            'correlation_length_nm': saxs_result.get('structure_factor', {}).get('correlation_length_nm', 10.0),
            'simulation_suggestions': {
                'box_size_nm': saxs_result.get('structure_factor', {}).get('correlation_length_nm', 10.0) * 5,
                'n_particles': 1000,
                'ensemble': 'NVT',
                'validation': 'Compare simulated S(q) with experimental SAXS'
            }
        }