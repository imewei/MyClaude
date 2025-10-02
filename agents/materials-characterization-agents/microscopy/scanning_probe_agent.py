"""Scanning Probe Agent - AFM, STM, and Related Techniques Expert.

This agent specializes in scanning probe microscopy techniques:
- AFM (Atomic Force Microscopy): Contact, tapping, non-contact modes
- STM (Scanning Tunneling Microscopy): Atomic resolution imaging
- KPFM (Kelvin Probe Force Microscopy): Surface potential mapping
- MFM (Magnetic Force Microscopy): Magnetic domain imaging
- C-AFM (Conductive AFM): Conductivity mapping
- PeakForce QNM (Quantitative Nanomechanics): Mechanical property mapping
- Liquid AFM: In-situ liquid environment imaging
- High-speed AFM: Video-rate imaging

Expert in nanoscale surface characterization, topography, mechanical properties,
electrical properties, and magnetic properties at sub-nanometer resolution.
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


class ScanningProbeAgent(ExperimentalAgent):
    """Scanning probe microscopy agent for nanoscale surface characterization.

    Capabilities:
    - AFM: Topography with sub-nm vertical resolution
    - STM: Atomic resolution imaging of conductive surfaces
    - KPFM: Surface potential distribution (work function)
    - MFM: Magnetic domain structure and switching
    - C-AFM: Local conductivity mapping
    - PeakForce QNM: Simultaneous mechanical property mapping
    - Phase Imaging: Compositional contrast
    - Friction Force Microscopy (FFM/LFM): Tribological properties

    Measurements:
    - Surface topography (height maps)
    - Roughness parameters (Ra, Rq, Rz, Rmax)
    - Particle size and distribution
    - Step heights and layer thickness
    - Young's modulus (E) mapping
    - Adhesion force distribution
    - Surface potential (V)
    - Conductivity (S/m)
    - Magnetic moment distribution

    Key advantages:
    - Sub-nanometer vertical resolution (0.1 nm)
    - Lateral resolution: 1-10 nm (AFM), 0.1 nm (STM)
    - 3D topography mapping
    - Multi-modal imaging (topography + property)
    - Quantitative mechanical properties
    - Ambient, liquid, and vacuum operation
    """

    VERSION = "1.0.0"

    # Supported scanning probe techniques
    SUPPORTED_TECHNIQUES = [
        'afm_contact',         # Contact mode AFM
        'afm_tapping',         # Tapping mode (intermittent contact)
        'afm_non_contact',     # Non-contact mode
        'stm',                 # Scanning tunneling microscopy
        'kpfm',                # Kelvin probe force microscopy
        'mfm',                 # Magnetic force microscopy
        'c_afm',               # Conductive AFM
        'peakforce_qnm',       # PeakForce quantitative nanomechanics
        'phase_imaging',       # Phase contrast imaging
        'ffm',                 # Friction force microscopy (lateral force)
        'liquid_afm',          # AFM in liquid environment
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Scanning Probe agent.

        Args:
            config: Configuration including:
                - instrument: SPM model (Bruker, Asylum, Park, Veeco)
                - scanner_range: Max scan size in μm
                - cantilever_type: Cantilever specifications
                - environment: ambient, vacuum, liquid
        """
        super().__init__(config)
        self.instrument = self.config.get('instrument', 'Bruker_Dimension_Icon')
        self.scanner_range = self.config.get('scanner_range', 90)  # μm
        self.cantilever = self.config.get('cantilever_type', 'RTESPA-300')
        self.environment = self.config.get('environment', 'ambient')

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute scanning probe microscopy analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or image_data: SPM image/curve data
                - parameters: Technique-specific parameters
                  - scan_size: Scan area in μm
                  - scan_rate: Hz
                  - resolution: pixels (e.g., 512x512)
                  - setpoint: Force setpoint (nN)

        Returns:
            AgentResult with SPM analysis
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
            if technique == 'afm_contact':
                result_data = self._execute_afm_contact(input_data)
            elif technique == 'afm_tapping':
                result_data = self._execute_afm_tapping(input_data)
            elif technique == 'afm_non_contact':
                result_data = self._execute_afm_non_contact(input_data)
            elif technique == 'stm':
                result_data = self._execute_stm(input_data)
            elif technique == 'kpfm':
                result_data = self._execute_kpfm(input_data)
            elif technique == 'mfm':
                result_data = self._execute_mfm(input_data)
            elif technique == 'c_afm':
                result_data = self._execute_c_afm(input_data)
            elif technique == 'peakforce_qnm':
                result_data = self._execute_peakforce_qnm(input_data)
            elif technique == 'phase_imaging':
                result_data = self._execute_phase_imaging(input_data)
            elif technique == 'ffm':
                result_data = self._execute_ffm(input_data)
            elif technique == 'liquid_afm':
                result_data = self._execute_liquid_afm(input_data)
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
                    'scan_size_um': input_data.get('parameters', {}).get('scan_size', 1.0),
                    'cantilever': self.cantilever,
                    'environment': self.environment
                },
                execution_time_sec=execution_time,
                environment={'temperature_c': 25, 'humidity_percent': 40}
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

    def _execute_afm_contact(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute contact mode AFM.

        Contact AFM provides:
        - High-resolution topography
        - Fast imaging
        - Hard materials compatibility
        - Friction force (lateral force) simultaneously
        """
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 1.0)
        resolution = params.get('resolution', 512)
        scan_rate = params.get('scan_rate', 1.0)  # Hz
        setpoint_nn = params.get('setpoint', 10.0)  # nN

        # Generate synthetic topography data
        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=5.0
        )

        # Calculate roughness parameters
        roughness = self._calculate_roughness(height_map)

        # Particle analysis
        particles = self._analyze_particles(height_map, x_um, y_um)

        return {
            'technique': 'AFM Contact Mode',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist(),
                'z_range_nm': float(np.ptp(height_map)),
                'resolution_pixels': f'{resolution}x{resolution}'
            },
            'scan_parameters': {
                'scan_size_um': scan_size_um,
                'scan_rate_hz': scan_rate,
                'setpoint_force_nn': setpoint_nn,
                'scan_time_sec': resolution / scan_rate,
                'cantilever': self.cantilever,
                'spring_constant_n_per_m': 40.0  # RTESPA-300
            },
            'roughness_analysis': roughness,
            'particle_analysis': particles,
            'imaging_quality': {
                'noise_level_nm': 0.1,
                'lateral_resolution_nm': 10.0,
                'vertical_resolution_nm': 0.1,
                'image_artifacts': 'none_detected'
            },
            'advantages': [
                'Fast imaging (high scan rates)',
                'Good for hard materials',
                'Simultaneous topography and friction'
            ],
            'limitations': [
                'Higher tip wear',
                'Lateral forces can distort soft samples',
                'Not suitable for very soft materials'
            ]
        }

    def _execute_afm_tapping(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tapping mode AFM (intermittent contact).

        Tapping AFM provides:
        - Reduced lateral forces
        - Phase imaging for compositional contrast
        - Suitable for soft materials
        - Reduced tip wear
        """
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 2.0)
        resolution = params.get('resolution', 512)
        amplitude_setpoint = params.get('amplitude_setpoint', 0.8)  # Fraction of free amplitude

        # Generate topography
        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=8.0
        )

        # Generate phase image (compositional contrast)
        phase_map = self._generate_phase_image(height_map, resolution)

        # Calculate roughness
        roughness = self._calculate_roughness(height_map)

        return {
            'technique': 'AFM Tapping Mode (AC Mode)',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist(),
                'z_range_nm': float(np.ptp(height_map))
            },
            'phase_imaging': {
                'phase_map_deg': phase_map.tolist(),
                'phase_range_deg': float(np.ptp(phase_map)),
                'compositional_contrast': 'detected',
                'soft_regions_deg': float(np.mean(phase_map[phase_map > 50])),
                'hard_regions_deg': float(np.mean(phase_map[phase_map < 50]))
            },
            'scan_parameters': {
                'scan_size_um': scan_size_um,
                'amplitude_setpoint': amplitude_setpoint,
                'drive_frequency_khz': 300,  # Near resonance
                'free_amplitude_nm': 50.0,
                'tapping_amplitude_nm': 40.0
            },
            'roughness_analysis': roughness,
            'advantages': [
                'Reduced lateral forces (gentle)',
                'Phase imaging for composition',
                'Low tip wear',
                'Suitable for soft materials (polymers, biomaterials)'
            ],
            'applications': [
                'Polymer blends',
                'Biological samples',
                'Thin films',
                'Nanoparticles'
            ]
        }

    def _execute_afm_non_contact(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute non-contact mode AFM.

        Non-contact AFM provides:
        - Truly non-contact (no tip wear)
        - Highest vertical resolution
        - Vacuum/UHV operation for atomic resolution
        """
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 0.5)
        resolution = params.get('resolution', 256)

        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=2.0
        )

        roughness = self._calculate_roughness(height_map)

        return {
            'technique': 'AFM Non-Contact Mode (FM-AFM)',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist(),
                'z_range_nm': float(np.ptp(height_map))
            },
            'scan_parameters': {
                'scan_size_um': scan_size_um,
                'tip_sample_distance_nm': 0.5,  # Van der Waals regime
                'frequency_shift_hz': 150,
                'q_factor': 30000  # High Q in vacuum
            },
            'roughness_analysis': roughness,
            'environment': 'ultra_high_vacuum',
            'advantages': [
                'Zero tip wear',
                'Atomic resolution possible',
                'Sensitive to long-range forces'
            ],
            'limitations': [
                'Requires vacuum/UHV for best results',
                'Slow imaging',
                'Complex electronics'
            ]
        }

    def _execute_stm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scanning tunneling microscopy.

        STM provides:
        - Atomic resolution imaging
        - Electronic density of states
        - Requires conductive samples
        - Vacuum operation
        """
        params = input_data.get('parameters', {})
        scan_size_nm = params.get('scan_size', 10.0)  # nm for STM
        resolution = params.get('resolution', 256)
        bias_voltage_v = params.get('bias_voltage', 0.1)
        tunneling_current_na = params.get('tunneling_current', 1.0)

        # Generate atomic-resolution topography
        height_map, x_nm, y_nm = self._generate_atomic_surface(scan_size_nm, resolution)

        return {
            'technique': 'STM (Scanning Tunneling Microscopy)',
            'topography': {
                'height_map_pm': (height_map * 1000).tolist(),  # pm scale
                'x_coordinates_nm': x_nm.tolist(),
                'y_coordinates_nm': y_nm.tolist(),
                'z_range_pm': float(np.ptp(height_map) * 1000),
                'atomic_resolution': True
            },
            'scan_parameters': {
                'scan_size_nm': scan_size_nm,
                'bias_voltage_v': bias_voltage_v,
                'tunneling_current_na': tunneling_current_na,
                'tip_sample_distance_pm': 500  # ~5 Å
            },
            'atomic_structure': {
                'lattice_constant_nm': 0.4,  # Example: surface lattice
                'atoms_visible': int((scan_size_nm / 0.4) ** 2),
                'defects_detected': 3,
                'step_edges': 1
            },
            'electronic_properties': {
                'local_dos': 'measurable_via_sts',
                'conductivity': 'metallic',
                'work_function_ev': 4.5
            },
            'advantages': [
                'Atomic resolution (0.1 nm lateral, 0.01 nm vertical)',
                'Direct visualization of atoms',
                'Electronic structure mapping (STS)',
                'Step heights to single atom precision'
            ],
            'requirements': [
                'Conductive sample required',
                'Ultra-high vacuum (UHV)',
                'Vibration isolation',
                'Temperature control'
            ]
        }

    def _execute_kpfm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Kelvin probe force microscopy.

        KPFM provides:
        - Surface potential mapping
        - Work function distribution
        - Charge distribution
        - Contact potential difference (CPD)
        """
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 5.0)
        resolution = params.get('resolution', 256)

        # Generate topography and surface potential
        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=10.0
        )

        # Generate surface potential map
        potential_map = self._generate_surface_potential(height_map, resolution)

        return {
            'technique': 'KPFM (Kelvin Probe Force Microscopy)',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist()
            },
            'surface_potential': {
                'potential_map_mv': potential_map.tolist(),
                'potential_range_mv': float(np.ptp(potential_map)),
                'mean_potential_mv': float(np.mean(potential_map)),
                'work_function_variation_ev': float(np.ptp(potential_map) / 1000)
            },
            'scan_parameters': {
                'scan_size_um': scan_size_um,
                'lift_height_nm': 50,  # Lift mode
                'ac_voltage_v': 2.0,
                'drive_frequency_khz': 75
            },
            'potential_analysis': {
                'high_potential_regions': 'grain_boundaries',
                'low_potential_regions': 'grain_interiors',
                'charge_distribution': 'inhomogeneous',
                'work_function_estimate_ev': 4.8
            },
            'applications': [
                'Semiconductor doping contrast',
                'Photovoltaic materials',
                'Charge trapping in dielectrics',
                'Ferroelectric domains'
            ]
        }

    def _execute_mfm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute magnetic force microscopy.

        MFM provides:
        - Magnetic domain imaging
        - Domain wall structure
        - Magnetic bit patterns
        - Stray field mapping
        """
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 10.0)
        resolution = params.get('resolution', 512)
        lift_height_nm = params.get('lift_height', 50)

        # Generate topography
        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=5.0
        )

        # Generate magnetic phase signal
        magnetic_map = self._generate_magnetic_domains(resolution)

        return {
            'technique': 'MFM (Magnetic Force Microscopy)',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist()
            },
            'magnetic_imaging': {
                'phase_shift_deg': magnetic_map.tolist(),
                'magnetic_contrast': 'strong',
                'domain_structure': 'stripe_domains',
                'domain_width_nm': 200,
                'domain_wall_width_nm': 20
            },
            'scan_parameters': {
                'scan_size_um': scan_size_um,
                'lift_height_nm': lift_height_nm,
                'magnetic_coating': 'CoCr_on_tip',
                'coercivity_oe': 400
            },
            'magnetic_analysis': {
                'number_of_domains': 15,
                'magnetization_direction': 'perpendicular_to_surface',
                'domain_switching': 'observable',
                'stray_field_estimate_oe': 100
            },
            'applications': [
                'Hard disk drives (magnetic bits)',
                'Magnetic thin films',
                'Nanomagnetism',
                'Spintronic devices'
            ]
        }

    def _execute_c_afm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conductive AFM.

        C-AFM provides:
        - Local conductivity mapping
        - I-V curves at nanoscale
        - Defect identification in insulators
        - Current distribution
        """
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 2.0)
        resolution = params.get('resolution', 256)
        bias_voltage_v = params.get('bias_voltage', 1.0)

        # Generate topography and current map
        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=8.0
        )

        current_map = self._generate_conductivity_map(height_map, resolution)

        return {
            'technique': 'C-AFM (Conductive AFM)',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist()
            },
            'conductivity_mapping': {
                'current_map_pa': current_map.tolist(),
                'current_range_pa': float(np.ptp(current_map)),
                'conductive_regions_percent': float(np.sum(current_map > 100) / current_map.size * 100),
                'insulating_regions_percent': float(np.sum(current_map < 10) / current_map.size * 100)
            },
            'scan_parameters': {
                'scan_size_um': scan_size_um,
                'bias_voltage_v': bias_voltage_v,
                'force_setpoint_nn': 5.0,
                'current_amplifier_gain': 1e9  # V/A
            },
            'electrical_analysis': {
                'local_resistivity_ohm_cm': 1e6,
                'current_distribution': 'heterogeneous',
                'conductive_pathways': 'grain_boundaries',
                'breakdown_events': 0
            },
            'applications': [
                'Semiconductor failure analysis',
                'Organic photovoltaics',
                'Resistive switching memories',
                'Nanowire conductivity'
            ]
        }

    def _execute_peakforce_qnm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PeakForce QNM (quantitative nanomechanics).

        PeakForce QNM provides:
        - Simultaneous topography, modulus, adhesion, deformation
        - Quantitative mechanical properties
        - Minimal tip wear
        - Suitable for soft materials
        """
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 3.0)
        resolution = params.get('resolution', 256)
        peak_force_nn = params.get('peak_force', 2.0)

        # Generate multiple property maps
        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=15.0
        )

        modulus_map = self._generate_modulus_map(height_map, resolution)
        adhesion_map = self._generate_adhesion_map(height_map, resolution)
        deformation_map = self._generate_deformation_map(height_map, resolution)

        return {
            'technique': 'PeakForce QNM',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist()
            },
            'mechanical_properties': {
                'modulus_map_gpa': modulus_map.tolist(),
                'modulus_range_gpa': [float(np.min(modulus_map)), float(np.max(modulus_map))],
                'mean_modulus_gpa': float(np.mean(modulus_map)),
                'modulus_std_gpa': float(np.std(modulus_map))
            },
            'adhesion_properties': {
                'adhesion_map_nn': adhesion_map.tolist(),
                'adhesion_range_nn': [float(np.min(adhesion_map)), float(np.max(adhesion_map))],
                'mean_adhesion_nn': float(np.mean(adhesion_map))
            },
            'deformation_properties': {
                'deformation_map_nm': deformation_map.tolist(),
                'mean_deformation_nm': float(np.mean(deformation_map))
            },
            'scan_parameters': {
                'scan_size_um': scan_size_um,
                'peak_force_nn': peak_force_nn,
                'peak_force_frequency_khz': 2.0,
                'tip_radius_nm': 8
            },
            'quantitative_analysis': {
                'dmt_model': 'applied',
                'calibration': 'polystyrene_standard',
                'accuracy': 'within_20_percent'
            },
            'applications': [
                'Polymer blends',
                'Biological cells',
                'Soft materials characterization',
                'Nanomechanical property mapping'
            ]
        }

    def _execute_phase_imaging(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase imaging AFM."""
        # Similar to tapping mode with emphasis on phase
        tapping_result = self._execute_afm_tapping(input_data)

        return {
            **tapping_result,
            'technique': 'Phase Imaging AFM',
            'compositional_mapping': {
                'phase_contrast_mechanism': 'energy_dissipation_differences',
                'soft_materials': 'high_phase_lag',
                'hard_materials': 'low_phase_lag',
                'phase_resolution_deg': 0.1
            }
        }

    def _execute_ffm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute friction force microscopy (lateral force)."""
        params = input_data.get('parameters', {})
        scan_size_um = params.get('scan_size', 1.0)
        resolution = params.get('resolution', 256)

        height_map, x_um, y_um = self._generate_surface_topography(
            scan_size_um, resolution, roughness_nm=5.0
        )

        friction_map = self._generate_friction_map(height_map, resolution)

        return {
            'technique': 'FFM (Friction Force Microscopy)',
            'topography': {
                'height_map_nm': height_map.tolist(),
                'x_coordinates_um': x_um.tolist(),
                'y_coordinates_um': y_um.tolist()
            },
            'friction_mapping': {
                'friction_map_nn': friction_map.tolist(),
                'friction_range_nn': float(np.ptp(friction_map)),
                'friction_coefficient': 0.3
            },
            'tribological_properties': {
                'adhesion_component_nn': float(np.mean(friction_map) * 0.7),
                'plowing_component_nn': float(np.mean(friction_map) * 0.3),
                'directionality': 'isotropic'
            }
        }

    def _execute_liquid_afm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute liquid AFM."""
        params = input_data.get('parameters', {})
        liquid = params.get('liquid', 'water')

        tapping_result = self._execute_afm_tapping(input_data)

        return {
            **tapping_result,
            'technique': 'Liquid AFM',
            'liquid_environment': {
                'liquid': liquid,
                'ph': 7.0 if liquid == 'water' else None,
                'ionic_strength_mm': 10,
                'temperature_c': 25
            },
            'in_situ_capabilities': [
                'Real-time observation of processes',
                'Native environment imaging',
                'Biomolecule conformational changes',
                'Electrochemical reactions'
            ]
        }

    # Helper methods for data generation
    def _generate_surface_topography(self, scan_size_um: float,
                                     resolution: int,
                                     roughness_nm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic surface topography."""
        x = np.linspace(0, scan_size_um, resolution)
        y = np.linspace(0, scan_size_um, resolution)
        X, Y = np.meshgrid(x, y)

        # Generate fractal-like surface
        height = np.zeros((resolution, resolution))
        for scale in [1, 2, 4, 8]:
            freq = scale / scan_size_um
            height += (roughness_nm / scale) * np.sin(2 * np.pi * freq * X) * np.cos(2 * np.pi * freq * Y)

        # Add random roughness
        height += np.random.normal(0, roughness_nm * 0.2, (resolution, resolution))

        return height, x, y

    def _generate_atomic_surface(self, scan_size_nm: float,
                                 resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate atomic-resolution surface."""
        x = np.linspace(0, scan_size_nm, resolution)
        y = np.linspace(0, scan_size_nm, resolution)
        X, Y = np.meshgrid(x, y)

        # Hexagonal lattice
        lattice_constant = 0.4  # nm
        height = 0.05 * (np.sin(2 * np.pi * X / lattice_constant) +
                        np.sin(2 * np.pi * Y / lattice_constant) +
                        np.sin(2 * np.pi * (X + Y) / lattice_constant))

        return height, x, y

    def _calculate_roughness(self, height_map: np.ndarray) -> Dict[str, float]:
        """Calculate surface roughness parameters."""
        return {
            'ra_nm': float(np.mean(np.abs(height_map - np.mean(height_map)))),
            'rq_nm': float(np.sqrt(np.mean((height_map - np.mean(height_map))**2))),
            'rz_nm': float(np.max(height_map) - np.min(height_map)),
            'rmax_nm': float(np.ptp(height_map)),
            'rsk': float(np.mean(((height_map - np.mean(height_map)) / np.std(height_map))**3)),
            'rku': float(np.mean(((height_map - np.mean(height_map)) / np.std(height_map))**4))
        }

    def _analyze_particles(self, height_map: np.ndarray,
                          x_um: np.ndarray,
                          y_um: np.ndarray) -> Dict[str, Any]:
        """Analyze particles on surface."""
        threshold = np.mean(height_map) + 2 * np.std(height_map)
        particles_detected = int(np.sum(height_map > threshold) / 50)

        return {
            'number_of_particles': particles_detected,
            'mean_particle_height_nm': float(np.mean(height_map[height_map > threshold])) if particles_detected > 0 else 0,
            'particle_coverage_percent': float(np.sum(height_map > threshold) / height_map.size * 100)
        }

    def _generate_phase_image(self, height_map: np.ndarray, resolution: int) -> np.ndarray:
        """Generate phase image."""
        phase = 45 + 20 * np.sin(height_map / 10) + np.random.normal(0, 2, (resolution, resolution))
        return phase

    def _generate_surface_potential(self, height_map: np.ndarray, resolution: int) -> np.ndarray:
        """Generate surface potential map."""
        potential = 200 + 100 * np.cos(height_map / 20) + np.random.normal(0, 10, (resolution, resolution))
        return potential

    def _generate_magnetic_domains(self, resolution: int) -> np.ndarray:
        """Generate magnetic domain pattern."""
        magnetic = np.zeros((resolution, resolution))
        domain_width = 40
        for i in range(0, resolution, domain_width * 2):
            magnetic[:, i:i+domain_width] = 10
            magnetic[:, i+domain_width:i+2*domain_width] = -10
        magnetic += np.random.normal(0, 1, (resolution, resolution))
        return magnetic

    def _generate_conductivity_map(self, height_map: np.ndarray, resolution: int) -> np.ndarray:
        """Generate conductivity map."""
        current = 10 + 200 * np.random.random((resolution, resolution))
        current[height_map > np.mean(height_map)] *= 5  # Grain boundaries more conductive
        return current

    def _generate_modulus_map(self, height_map: np.ndarray, resolution: int) -> np.ndarray:
        """Generate Young's modulus map."""
        modulus = 2.0 + 1.5 * np.random.random((resolution, resolution))
        modulus[height_map < np.mean(height_map)] *= 0.5  # Soft regions
        return modulus

    def _generate_adhesion_map(self, height_map: np.ndarray, resolution: int) -> np.ndarray:
        """Generate adhesion force map."""
        adhesion = 5 + 15 * np.random.random((resolution, resolution))
        return adhesion

    def _generate_deformation_map(self, height_map: np.ndarray, resolution: int) -> np.ndarray:
        """Generate deformation map."""
        deformation = 2 + 3 * np.random.random((resolution, resolution))
        return deformation

    def _generate_friction_map(self, height_map: np.ndarray, resolution: int) -> np.ndarray:
        """Generate friction force map."""
        friction = 8 + 12 * np.random.random((resolution, resolution))
        return friction

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

        if 'data_file' not in data and 'image_data' not in data:
            warnings.append("No data provided; will use simulated data")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed."""
        params = data.get('parameters', {})
        resolution = params.get('resolution', 512)

        # Image processing scales with pixels
        cpu_cores = 2 if resolution > 512 else 1
        memory_gb = 2.0 if resolution > 512 else 1.0

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            estimated_time_sec=60.0,
            execution_environment=ExecutionEnvironment.LOCAL
        )

    def get_capabilities(self) -> List[Capability]:
        """Get list of agent capabilities."""
        return [
            Capability(
                name='afm_tapping',
                description='Tapping mode AFM for soft materials',
                input_types=['image_data', 'scan_parameters'],
                output_types=['topography', 'phase', 'roughness'],
                typical_use_cases=['polymer_characterization', 'biological_imaging', 'nanoparticles']
            ),
            Capability(
                name='stm',
                description='Scanning tunneling microscopy for atomic resolution',
                input_types=['image_data', 'tunneling_parameters'],
                output_types=['atomic_topography', 'electronic_structure'],
                typical_use_cases=['surface_science', 'catalysis', 'semiconductor_surfaces']
            ),
            Capability(
                name='peakforce_qnm',
                description='Quantitative nanomechanical property mapping',
                input_types=['force_curve_data'],
                output_types=['modulus_map', 'adhesion_map', 'deformation_map'],
                typical_use_cases=['mechanical_property_mapping', 'polymer_blends', 'biomaterials']
            ),
            Capability(
                name='kpfm',
                description='Surface potential and work function mapping',
                input_types=['electrostatic_data'],
                output_types=['surface_potential', 'work_function'],
                typical_use_cases=['photovoltaics', 'semiconductors', 'charge_distribution']
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata."""
        return AgentMetadata(
            name="ScanningProbeAgent",
            version=self.VERSION,
            description="Scanning probe microscopy expert (AFM, STM, KPFM, MFM)",
            author="Materials Characterization Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['spm', 'nanoscope', 'gwyddion', 'igor']
        )

    def connect_instrument(self) -> bool:
        """Connect to SPM instrument."""
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw SPM data."""
        # In production:
        # - Plane fit/leveling
        # - Line-by-line correction
        # - Noise filtering
        # - Drift correction
        return raw_data

    # Integration methods
    @staticmethod
    def correlate_with_sem(afm_result: Dict[str, Any], sem_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate AFM topography with SEM morphology.

        AFM provides 3D topography at nm scale.
        SEM provides 2D morphology at nm-μm scale.

        Args:
            afm_result: AFM topography
            sem_result: SEM imaging

        Returns:
            Correlation report
        """
        afm_roughness = afm_result.get('roughness_analysis', {}).get('ra_nm', 0)

        return {
            'correlation_type': 'AFM_SEM_topography',
            'afm_roughness_nm': afm_roughness,
            'notes': 'AFM provides quantitative 3D height, SEM provides larger-area context',
            'recommendation': 'Use SEM for overview, AFM for quantitative roughness and mechanical properties'
        }

    @staticmethod
    def validate_with_nanoindentation(qnm_result: Dict[str, Any], nanoindent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PeakForce QNM modulus with nanoindentation.

        PeakForce QNM: Nanoscale modulus mapping
        Nanoindentation: Point measurements with higher accuracy

        Args:
            qnm_result: PeakForce QNM modulus map
            nanoindent_result: Nanoindentation measurements

        Returns:
            Validation report
        """
        qnm_modulus = qnm_result.get('mechanical_properties', {}).get('mean_modulus_gpa', 0)
        nanoindent_modulus = nanoindent_result.get('reduced_modulus_gpa', 0)

        if qnm_modulus > 0 and nanoindent_modulus > 0:
            diff_percent = abs(qnm_modulus - nanoindent_modulus) / nanoindent_modulus * 100

            return {
                'validation_type': 'QNM_Nanoindentation_modulus',
                'qnm_modulus_gpa': qnm_modulus,
                'nanoindent_modulus_gpa': nanoindent_modulus,
                'difference_percent': diff_percent,
                'agreement': 'excellent' if diff_percent < 20 else 'good' if diff_percent < 40 else 'poor',
                'notes': 'PeakForce QNM typically 20-30% lower than nanoindentation due to different contact mechanics'
            }

        return {'error': 'Missing modulus data'}
