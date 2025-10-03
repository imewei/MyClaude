"""
Optical Microscopy Agent for Materials Characterization

This agent provides comprehensive optical microscopy capabilities including:
- Brightfield microscopy (transmission and reflection)
- Darkfield microscopy
- Phase contrast microscopy
- Differential Interference Contrast (DIC/Nomarski)
- Confocal laser scanning microscopy
- Fluorescence microscopy
- Polarized light microscopy (PLM)
- Digital holographic microscopy

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


class MicroscopyMode(Enum):
    """Optical microscopy imaging modes"""
    BRIGHTFIELD_TRANSMISSION = "brightfield_transmission"
    BRIGHTFIELD_REFLECTION = "brightfield_reflection"
    DARKFIELD = "darkfield"
    PHASE_CONTRAST = "phase_contrast"
    DIC = "dic"  # Differential Interference Contrast
    CONFOCAL = "confocal"
    FLUORESCENCE = "fluorescence"
    POLARIZED_LIGHT = "polarized_light"
    DIGITAL_HOLOGRAPHIC = "digital_holographic"


class IlluminationMode(Enum):
    """Illumination configurations"""
    KOHLER = "kohler"
    CRITICAL = "critical"
    OBLIQUE = "oblique"
    EPIILLUMINATION = "epi"
    DIAILLUMINATION = "dia"


@dataclass
class OpticalMicroscopyResult:
    """Results from optical microscopy measurement"""
    mode: str
    image_data: np.ndarray  # 2D or 3D array
    resolution: Dict[str, float]  # lateral and axial resolution
    magnification: float
    numerical_aperture: float
    field_of_view: Dict[str, float]  # width and height in micrometers
    image_quality_metrics: Dict[str, float]
    features_detected: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str


class OpticalMicroscopyAgent(ExperimentalAgent):
    """
    Agent for optical microscopy characterization of materials.

    Capabilities:
    - Multiple contrast modes (brightfield, darkfield, phase, DIC)
    - Advanced techniques (confocal, fluorescence, polarized light)
    - Quantitative image analysis
    - 3D reconstruction and optical sectioning
    - Multi-channel fluorescence imaging
    - Birefringence and crystallinity analysis
    """

    NAME = "OpticalMicroscopyAgent"
    VERSION = "1.0.0"
    DESCRIPTION = "Comprehensive optical microscopy for materials characterization"

    SUPPORTED_TECHNIQUES = [
        'brightfield_transmission',
        'brightfield_reflection',
        'darkfield',
        'phase_contrast',
        'dic',
        'confocal',
        'fluorescence',
        'polarized_light',
        'digital_holographic'
    ]

    # Optical parameters
    WAVELENGTHS = {
        'uv': 365,      # nm
        'blue': 470,    # nm
        'green': 530,   # nm
        'red': 630,     # nm
        'white': 550    # nm (average)
    }

    OBJECTIVE_SPECS = {
        '4x': {'NA': 0.10, 'WD': 17.0, 'resolution': 2.75},
        '10x': {'NA': 0.25, 'WD': 10.0, 'resolution': 1.10},
        '20x': {'NA': 0.40, 'WD': 3.0, 'resolution': 0.69},
        '40x': {'NA': 0.65, 'WD': 0.6, 'resolution': 0.42},
        '60x': {'NA': 0.85, 'WD': 0.3, 'resolution': 0.32},
        '100x': {'NA': 1.30, 'WD': 0.2, 'resolution': 0.21}  # Oil immersion
    }

    def __init__(self):
        super().__init__(self.NAME, self.VERSION, self.DESCRIPTION)
        self.capabilities = {
            'max_magnification': 1000,
            'min_magnification': 4,
            'max_numerical_aperture': 1.40,
            'lateral_resolution_limit': 0.2,  # micrometers
            'axial_resolution_limit': 0.5,    # micrometers
            'field_of_view_range': (0.1, 5000),  # micrometers
            'wavelength_range': (365, 700),   # nm
            'supported_modes': self.SUPPORTED_TECHNIQUES,
            'z_stack_capable': True,
            'time_lapse_capable': True,
            'multi_channel_capable': True
        }

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute optical microscopy measurement.

        Args:
            input_data: Dictionary containing:
                - technique: Microscopy mode
                - sample_info: Sample description
                - objective: Objective lens (4x, 10x, 20x, etc.)
                - wavelength: Illumination wavelength or color
                - exposure_time: Camera exposure in ms
                - z_stack: Optional z-stack parameters
                - channels: Optional multi-channel config

        Returns:
            AgentResult with microscopy images and analysis
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
            'brightfield_transmission': self._execute_brightfield_transmission,
            'brightfield_reflection': self._execute_brightfield_reflection,
            'darkfield': self._execute_darkfield,
            'phase_contrast': self._execute_phase_contrast,
            'dic': self._execute_dic,
            'confocal': self._execute_confocal,
            'fluorescence': self._execute_fluorescence,
            'polarized_light': self._execute_polarized_light,
            'digital_holographic': self._execute_digital_holographic
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

    def _execute_brightfield_transmission(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute brightfield transmission microscopy"""
        objective = input_data.get('objective', '10x')
        wavelength = input_data.get('wavelength', 'white')
        image_size = input_data.get('image_size', (1024, 1024))
        exposure = input_data.get('exposure_time', 100)  # ms

        # Get optical parameters
        obj_specs = self.OBJECTIVE_SPECS[objective]
        na = obj_specs['NA']
        wl = self.WAVELENGTHS.get(wavelength, 550)

        # Calculate resolution (Abbe limit)
        lateral_resolution = 0.61 * wl / (na * 1000)  # micrometers

        # Calculate field of view
        magnification = int(objective.replace('x', ''))
        fov_width = 22000 / magnification  # micrometers (22mm sensor)
        fov_height = 22000 / magnification

        # Simulate transmission image
        image = self._simulate_transmission_image(
            size=image_size,
            resolution=lateral_resolution,
            sample_type=input_data.get('sample_info', {}).get('type', 'general')
        )

        # Analyze image
        features = self._detect_features_brightfield(image, lateral_resolution)
        quality_metrics = self._calculate_image_quality(image)

        return {
            'image': image,
            'mode': 'brightfield_transmission',
            'optical_parameters': {
                'magnification': magnification,
                'numerical_aperture': na,
                'wavelength_nm': wl,
                'lateral_resolution_um': lateral_resolution,
                'working_distance_mm': obj_specs['WD'],
                'depth_of_field_um': self._calculate_dof(na, wl, magnification)
            },
            'field_of_view': {
                'width_um': fov_width,
                'height_um': fov_height,
                'area_um2': fov_width * fov_height
            },
            'image_quality': quality_metrics,
            'features_detected': features,
            'intensity_statistics': {
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'min': float(np.min(image)),
                'max': float(np.max(image)),
                'dynamic_range': float(np.max(image) - np.min(image))
            },
            'acquisition_parameters': {
                'exposure_time_ms': exposure,
                'illumination': 'kohler',
                'condenser_na': min(0.9 * na, 0.95)
            }
        }

    def _execute_brightfield_reflection(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute brightfield reflection microscopy (metallography)"""
        objective = input_data.get('objective', '20x')
        image_size = input_data.get('image_size', (2048, 2048))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))
        na = obj_specs['NA']
        wl = self.WAVELENGTHS['white']

        lateral_resolution = 0.61 * wl / (na * 1000)
        fov = 22000 / magnification

        # Simulate reflection image (grain structure)
        image = self._simulate_reflection_image(
            size=image_size,
            grain_size=input_data.get('grain_size_um', 50),
            resolution=lateral_resolution
        )

        # Grain analysis
        grains = self._analyze_grains(image, lateral_resolution)
        quality_metrics = self._calculate_image_quality(image)

        return {
            'image': image,
            'mode': 'brightfield_reflection',
            'optical_parameters': {
                'magnification': magnification,
                'numerical_aperture': na,
                'lateral_resolution_um': lateral_resolution,
                'working_distance_mm': obj_specs['WD']
            },
            'field_of_view': {
                'width_um': fov,
                'height_um': fov
            },
            'grain_analysis': grains,
            'image_quality': quality_metrics,
            'surface_features': {
                'grain_boundaries_detected': len(grains.get('grain_boundaries', [])),
                'phase_regions': grains.get('phase_count', 1),
                'texture_uniformity': grains.get('uniformity_index', 0.0)
            }
        }

    def _execute_darkfield(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute darkfield microscopy for scatter/defect detection"""
        objective = input_data.get('objective', '10x')
        image_size = input_data.get('image_size', (1024, 1024))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))
        na = obj_specs['NA']

        # Darkfield requires higher condenser NA than objective
        condenser_na = input_data.get('condenser_na', 1.2)
        if condenser_na <= na:
            condenser_na = na * 1.3

        # Simulate darkfield image (only scattered light)
        image = self._simulate_darkfield_image(
            size=image_size,
            defect_density=input_data.get('defect_density', 0.001),
            particle_size_nm=input_data.get('particle_size_nm', 500)
        )

        # Detect scattering centers
        scatterers = self._detect_scattering_centers(image)
        quality_metrics = self._calculate_image_quality(image)

        return {
            'image': image,
            'mode': 'darkfield',
            'optical_parameters': {
                'magnification': magnification,
                'objective_na': na,
                'condenser_na': condenser_na,
                'hollow_cone_angle': np.degrees(np.arcsin(condenser_na))
            },
            'scattering_analysis': scatterers,
            'image_quality': quality_metrics,
            'defect_statistics': {
                'total_defects': scatterers['count'],
                'defect_density_per_um2': scatterers['density'],
                'mean_scatter_intensity': scatterers['mean_intensity'],
                'size_distribution': scatterers['size_distribution']
            }
        }

    def _execute_phase_contrast(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase contrast microscopy for transparent specimens"""
        objective = input_data.get('objective', '40x')
        phase_ring = input_data.get('phase_ring', 'Ph2')  # Ph1, Ph2, Ph3
        image_size = input_data.get('image_size', (1024, 1024))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))
        na = obj_specs['NA']

        # Phase contrast converts phase to amplitude
        phase_shift = {'Ph1': 90, 'Ph2': 90, 'Ph3': 90}[phase_ring]
        absorption_factor = {'Ph1': 0.25, 'Ph2': 0.5, 'Ph3': 0.75}[phase_ring]

        # Simulate phase contrast image
        image = self._simulate_phase_contrast_image(
            size=image_size,
            refractive_index_var=input_data.get('ri_variation', 0.02),
            thickness_var_um=input_data.get('thickness_var_um', 2.0),
            phase_shift=phase_shift
        )

        # Analyze phase features
        phase_features = self._analyze_phase_features(image)
        quality_metrics = self._calculate_image_quality(image)

        return {
            'image': image,
            'mode': 'phase_contrast',
            'optical_parameters': {
                'magnification': magnification,
                'numerical_aperture': na,
                'phase_ring': phase_ring,
                'phase_shift_degrees': phase_shift,
                'absorption_factor': absorption_factor
            },
            'phase_analysis': phase_features,
            'image_quality': quality_metrics,
            'optical_path_differences': {
                'mean_opd_nm': phase_features.get('mean_opd', 0),
                'std_opd_nm': phase_features.get('std_opd', 0),
                'max_opd_nm': phase_features.get('max_opd', 0)
            },
            'halo_artifacts': {
                'present': phase_features.get('halo_detected', False),
                'intensity': phase_features.get('halo_intensity', 0)
            }
        }

    def _execute_dic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Differential Interference Contrast (Nomarski) microscopy"""
        objective = input_data.get('objective', '60x')
        prism_setting = input_data.get('prism_setting', 0.5)  # 0-1 (shear amount)
        image_size = input_data.get('image_size', (2048, 2048))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))
        na = obj_specs['NA']
        wl = self.WAVELENGTHS['green']

        # DIC provides optical sectioning and 3D-like appearance
        shear_amount_nm = prism_setting * 200  # Typical shear: 0-200 nm

        # Simulate DIC image with gradient detection
        image = self._simulate_dic_image(
            size=image_size,
            shear_nm=shear_amount_nm,
            topography_relief_nm=input_data.get('relief_nm', 100),
            ri_gradients=input_data.get('ri_gradients', True)
        )

        # Analyze gradients and topography
        gradient_analysis = self._analyze_dic_gradients(image, shear_amount_nm)
        quality_metrics = self._calculate_image_quality(image)

        return {
            'image': image,
            'mode': 'dic',
            'optical_parameters': {
                'magnification': magnification,
                'numerical_aperture': na,
                'wavelength_nm': wl,
                'shear_distance_nm': shear_amount_nm,
                'prism_setting': prism_setting,
                'lateral_resolution_um': 0.61 * wl / (na * 1000)
            },
            'gradient_analysis': gradient_analysis,
            'image_quality': quality_metrics,
            'topography_estimation': {
                'relief_range_nm': gradient_analysis.get('relief_range', 0),
                'slope_angles_deg': gradient_analysis.get('slope_angles', []),
                'edge_enhancement': gradient_analysis.get('edge_sharpness', 0)
            },
            'optical_sectioning': {
                'effective_thickness_um': self._calculate_optical_section_thickness(na),
                'out_of_focus_rejection': 0.7  # DIC rejects ~70% out-of-focus light
            }
        }

    def _execute_confocal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute confocal laser scanning microscopy"""
        objective = input_data.get('objective', '60x')
        laser_wavelength = input_data.get('laser_wavelength', 488)  # nm
        pinhole_size_au = input_data.get('pinhole_au', 1.0)  # Airy units
        z_stack_params = input_data.get('z_stack', None)
        image_size = input_data.get('image_size', (512, 512))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))
        na = obj_specs['NA']

        # Confocal resolution
        lateral_res = 0.4 * laser_wavelength / (na * 1000)  # micrometers
        axial_res = 1.4 * (laser_wavelength / 1000) * (1.515 / (na ** 2))  # micrometers (n=1.515 oil)

        # Generate z-stack if requested
        if z_stack_params:
            z_start = z_stack_params.get('start_um', 0)
            z_end = z_stack_params.get('end_um', 10)
            z_step = z_stack_params.get('step_um', 0.5)
            z_positions = np.arange(z_start, z_end, z_step)

            image_stack = self._simulate_confocal_stack(
                size=image_size,
                z_positions=z_positions,
                lateral_res=lateral_res,
                axial_res=axial_res,
                pinhole_au=pinhole_size_au
            )

            # 3D reconstruction
            reconstruction = self._reconstruct_3d(image_stack, z_positions)
        else:
            image_stack = self._simulate_confocal_image(
                size=image_size,
                lateral_res=lateral_res
            )
            reconstruction = None

        # Analyze confocal image
        features = self._analyze_confocal_features(image_stack)
        quality_metrics = self._calculate_image_quality(
            image_stack if image_stack.ndim == 2 else image_stack[len(image_stack)//2]
        )

        return {
            'image': image_stack,
            'mode': 'confocal',
            'optical_parameters': {
                'magnification': magnification,
                'numerical_aperture': na,
                'laser_wavelength_nm': laser_wavelength,
                'pinhole_size_au': pinhole_size_au,
                'lateral_resolution_um': lateral_res,
                'axial_resolution_um': axial_res
            },
            'z_stack_info': {
                'enabled': z_stack_params is not None,
                'slices': len(z_positions) if z_stack_params else 1,
                'z_range_um': (z_start, z_end) if z_stack_params else None,
                'z_step_um': z_step if z_stack_params else None
            },
            '3d_reconstruction': reconstruction,
            'features_analyzed': features,
            'image_quality': quality_metrics,
            'optical_sectioning': {
                'thickness_um': axial_res,
                'rejection_efficiency': self._calculate_pinhole_rejection(pinhole_size_au)
            }
        }

    def _execute_fluorescence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fluorescence microscopy with multi-channel support"""
        objective = input_data.get('objective', '40x')
        channels = input_data.get('channels', [
            {'name': 'DAPI', 'ex': 358, 'em': 461},
            {'name': 'FITC', 'ex': 495, 'em': 519}
        ])
        image_size = input_data.get('image_size', (1024, 1024))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))
        na = obj_specs['NA']

        # Acquire multi-channel images
        channel_images = []
        channel_analysis = []

        for channel in channels:
            ex_wl = channel['ex']
            em_wl = channel['em']

            # Resolution depends on emission wavelength
            lateral_res = 0.61 * em_wl / (na * 1000)

            # Simulate fluorescence image
            fl_image = self._simulate_fluorescence_image(
                size=image_size,
                emission_wl=em_wl,
                quantum_yield=channel.get('qy', 0.8),
                concentration=channel.get('concentration', 1.0),
                photobleaching=input_data.get('photobleaching', 0.1)
            )

            channel_images.append(fl_image)

            # Analyze fluorescence
            analysis = self._analyze_fluorescence_channel(fl_image, channel['name'])
            channel_analysis.append(analysis)

        # Composite image (RGB overlay)
        composite = self._create_fluorescence_composite(channel_images, channels)

        # Colocalization analysis if multiple channels
        colocalization = None
        if len(channels) >= 2:
            colocalization = self._analyze_colocalization(
                channel_images[0],
                channel_images[1],
                channels[0]['name'],
                channels[1]['name']
            )

        quality_metrics = self._calculate_image_quality(composite)

        return {
            'channel_images': channel_images,
            'composite_image': composite,
            'mode': 'fluorescence',
            'optical_parameters': {
                'magnification': magnification,
                'numerical_aperture': na,
                'channels': channels,
                'light_source': input_data.get('light_source', 'LED')
            },
            'channel_analysis': channel_analysis,
            'colocalization': colocalization,
            'image_quality': quality_metrics,
            'fluorescence_statistics': {
                'total_channels': len(channels),
                'signal_to_noise': [ch.get('snr', 0) for ch in channel_analysis],
                'photobleaching_rate': input_data.get('photobleaching', 0.1),
                'autofluorescence_level': 'low'
            }
        }

    def _execute_polarized_light(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute polarized light microscopy for birefringence analysis"""
        objective = input_data.get('objective', '10x')
        analyzer_angle = input_data.get('analyzer_angle', 90)  # degrees
        compensator = input_data.get('compensator', None)  # 'lambda', 'quarter_wave', etc.
        image_size = input_data.get('image_size', (1024, 1024))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))

        # Simulate polarized light image
        image = self._simulate_polarized_light_image(
            size=image_size,
            birefringence=input_data.get('birefringence', 0.01),
            crystal_orientation=input_data.get('orientation_deg', 45),
            analyzer_angle=analyzer_angle,
            compensator=compensator
        )

        # Analyze birefringence
        birefringence_analysis = self._analyze_birefringence(
            image,
            analyzer_angle,
            compensator
        )

        quality_metrics = self._calculate_image_quality(image)

        return {
            'image': image,
            'mode': 'polarized_light',
            'optical_parameters': {
                'magnification': magnification,
                'polarizer_angle': 0,  # Fixed at 0
                'analyzer_angle': analyzer_angle,
                'compensator': compensator,
                'extinction_position': birefringence_analysis.get('extinction_angle', 0)
            },
            'birefringence_analysis': birefringence_analysis,
            'image_quality': quality_metrics,
            'crystallographic_info': {
                'birefringent': birefringence_analysis.get('is_birefringent', False),
                'retardation_nm': birefringence_analysis.get('retardation', 0),
                'interference_colors': birefringence_analysis.get('michel_levy_order', 0),
                'optic_sign': birefringence_analysis.get('optic_sign', 'unknown')
            }
        }

    def _execute_digital_holographic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute digital holographic microscopy for quantitative phase imaging"""
        wavelength = input_data.get('wavelength', 532)  # nm (green laser)
        objective = input_data.get('objective', '20x')
        image_size = input_data.get('image_size', (1024, 1024))

        obj_specs = self.OBJECTIVE_SPECS[objective]
        magnification = int(objective.replace('x', ''))
        na = obj_specs['NA']

        # DHM provides quantitative phase and amplitude
        hologram = self._simulate_hologram(
            size=image_size,
            wavelength=wavelength,
            phase_profile=input_data.get('phase_map', None)
        )

        # Reconstruct phase and amplitude
        reconstruction = self._reconstruct_hologram(hologram, wavelength)
        phase_map = reconstruction['phase']
        amplitude_map = reconstruction['amplitude']

        # Quantitative analysis
        phase_analysis = self._analyze_quantitative_phase(phase_map, wavelength)

        # Calculate thickness/refractive index
        ri_analysis = self._calculate_ri_from_phase(
            phase_map,
            wavelength,
            input_data.get('reference_ri', 1.33)
        )

        quality_metrics = self._calculate_image_quality(amplitude_map)

        return {
            'hologram': hologram,
            'phase_map': phase_map,
            'amplitude_map': amplitude_map,
            'mode': 'digital_holographic',
            'optical_parameters': {
                'wavelength_nm': wavelength,
                'magnification': magnification,
                'numerical_aperture': na,
                'axial_resolution_nm': wavelength / 2,  # λ/2 phase sensitivity
                'lateral_resolution_um': 0.61 * wavelength / (na * 1000)
            },
            'phase_analysis': phase_analysis,
            'refractive_index_analysis': ri_analysis,
            'image_quality': quality_metrics,
            'quantitative_measurements': {
                'phase_sensitivity_rad': 0.01,
                'height_sensitivity_nm': (wavelength / (4 * np.pi)),
                'optical_thickness_range_nm': phase_analysis.get('opd_range', 0),
                'dry_mass_pg': phase_analysis.get('dry_mass', 0)  # For cells/particles
            }
        }

    # Helper methods for image simulation

    def _simulate_transmission_image(self, size: Tuple[int, int], resolution: float, sample_type: str) -> np.ndarray:
        """Simulate brightfield transmission image"""
        image = np.ones(size, dtype=np.float32) * 200  # Bright background

        # Add sample features based on type
        if sample_type == 'fibers':
            # Add fiber structures
            for _ in range(20):
                x = np.random.randint(0, size[0])
                y = np.random.randint(0, size[1])
                angle = np.random.uniform(0, 180)
                length = int(np.random.uniform(50, 200))
                self._add_fiber(image, x, y, angle, length, absorption=0.4)

        elif sample_type == 'particles':
            # Add particles
            for _ in range(50):
                x = np.random.randint(20, size[0]-20)
                y = np.random.randint(20, size[1]-20)
                radius = int(np.random.uniform(5, 20))
                self._add_particle(image, x, y, radius, absorption=0.5)

        else:  # general
            # Add general structures
            for _ in range(30):
                x = np.random.randint(0, size[0])
                y = np.random.randint(0, size[1])
                self._add_random_feature(image, x, y)

        # Add noise
        noise = np.random.normal(0, 5, size)
        image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)

    def _simulate_reflection_image(self, size: Tuple[int, int], grain_size: float, resolution: float) -> np.ndarray:
        """Simulate brightfield reflection image with grain structure"""
        image = np.zeros(size, dtype=np.float32)

        # Create grain structure using Voronoi-like approach
        num_grains = int((size[0] * size[1]) / (grain_size ** 2))
        grain_centers = np.random.randint(0, min(size), (num_grains, 2))

        # Assign each pixel to nearest grain
        y, x = np.ogrid[:size[0], :size[1]]
        for i, (gx, gy) in enumerate(grain_centers):
            dist = np.sqrt((x - gx)**2 + (y - gy)**2)
            grain_intensity = np.random.uniform(80, 180)
            mask = dist < grain_size / 2
            image[mask] = grain_intensity

        # Add grain boundaries (darker)
        from scipy import ndimage
        gradient = ndimage.gaussian_gradient_magnitude(image, sigma=2)
        boundaries = gradient > np.percentile(gradient, 90)
        image[boundaries] = 40

        # Add noise
        noise = np.random.normal(0, 8, size)
        image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)

    def _simulate_darkfield_image(self, size: Tuple[int, int], defect_density: float, particle_size_nm: float) -> np.ndarray:
        """Simulate darkfield image (only scattered light visible)"""
        image = np.zeros(size, dtype=np.float32)  # Dark background

        # Add scattering centers (bright spots)
        num_scatterers = int(size[0] * size[1] * defect_density)

        for _ in range(num_scatterers):
            x = np.random.randint(0, size[0])
            y = np.random.randint(0, size[1])

            # Intensity depends on particle size (Rayleigh/Mie scattering)
            intensity = (particle_size_nm / 500) ** 4 * np.random.uniform(100, 255)
            intensity = min(intensity, 255)

            # Add scattered light spot
            self._add_scatter_spot(image, x, y, intensity, sigma=2)

        # Add weak background scatter
        image += np.random.uniform(0, 10, size)

        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def _simulate_phase_contrast_image(self, size: Tuple[int, int], refractive_index_var: float,
                                       thickness_var_um: float, phase_shift: float) -> np.ndarray:
        """Simulate phase contrast image"""
        # Create phase map
        phase_map = np.random.normal(0, thickness_var_um * refractive_index_var * 2 * np.pi / 0.55, size)

        # Add structures with different optical path differences
        for _ in range(10):
            x = np.random.randint(50, size[0]-50)
            y = np.random.randint(50, size[1]-50)
            radius = np.random.randint(20, 50)
            opd = np.random.uniform(-np.pi, np.pi)
            self._add_phase_object(phase_map, x, y, radius, opd)

        # Convert phase to intensity (phase contrast transfer function)
        # I = 1 + 2*sin(φ) for small phase shifts
        image = 128 + 127 * np.sin(phase_map + np.radians(phase_shift))

        # Add halo artifact (characteristic of phase contrast)
        from scipy import ndimage
        gradient = ndimage.gaussian_gradient_magnitude(image, sigma=3)
        halo = gradient * 0.3
        image = image + halo

        # Add noise
        noise = np.random.normal(0, 5, size)
        image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)

    def _simulate_dic_image(self, size: Tuple[int, int], shear_nm: float,
                           topography_relief_nm: float, ri_gradients: bool) -> np.ndarray:
        """Simulate DIC image with gradient detection"""
        # Create height/phase map
        height_map = np.zeros(size, dtype=np.float32)

        # Add topographic features
        for _ in range(15):
            x = np.random.randint(50, size[0]-50)
            y = np.random.randint(50, size[1]-50)
            radius = np.random.randint(20, 80)
            height = np.random.uniform(0, topography_relief_nm)
            self._add_topography_feature(height_map, x, y, radius, height)

        # Calculate gradient (DIC is sensitive to gradients)
        from scipy import ndimage
        # Gradient in shear direction (assume 45 degrees)
        grad_x = ndimage.sobel(height_map, axis=1)
        grad_y = ndimage.sobel(height_map, axis=0)
        gradient = (grad_x + grad_y) / np.sqrt(2)  # 45-degree gradient

        # Convert gradient to intensity (DIC contrast)
        # Normalize and shift to [0, 255]
        gradient_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
        image = 128 + 127 * (gradient_norm - 0.5) * 2

        # Add slight bias (shadow-cast effect)
        bias = np.random.uniform(10, 30)
        image = image + bias

        # Add noise
        noise = np.random.normal(0, 3, size)
        image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)

    def _simulate_confocal_image(self, size: Tuple[int, int], lateral_res: float) -> np.ndarray:
        """Simulate single confocal image"""
        image = np.zeros(size, dtype=np.float32)

        # Add in-focus structures
        for _ in range(20):
            x = np.random.randint(20, size[0]-20)
            y = np.random.randint(20, size[1]-20)
            radius = int(np.random.uniform(5, 15))
            intensity = np.random.uniform(100, 250)
            self._add_gaussian_spot(image, x, y, radius, intensity)

        # Confocal has less noise due to pinhole
        noise = np.random.normal(0, 2, size)
        image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)

    def _simulate_confocal_stack(self, size: Tuple[int, int], z_positions: np.ndarray,
                                lateral_res: float, axial_res: float, pinhole_au: float) -> np.ndarray:
        """Simulate confocal z-stack"""
        stack = []

        # Create 3D object positions
        num_objects = 30
        object_positions = []
        for _ in range(num_objects):
            obj_z = np.random.uniform(z_positions.min(), z_positions.max())
            obj_x = np.random.randint(20, size[0]-20)
            obj_y = np.random.randint(20, size[1]-20)
            obj_radius = np.random.uniform(3, 10)
            obj_intensity = np.random.uniform(100, 250)
            object_positions.append((obj_x, obj_y, obj_z, obj_radius, obj_intensity))

        # Generate images at each z position
        for z in z_positions:
            image = np.zeros(size, dtype=np.float32)

            # Add objects (intensity depends on z distance)
            for obj_x, obj_y, obj_z, obj_radius, obj_intensity in object_positions:
                z_dist = abs(z - obj_z)

                # Gaussian z-response (confocal PSF)
                z_response = np.exp(-(z_dist**2) / (2 * (axial_res/2.35)**2))

                # Add to image if in range
                if z_response > 0.1:  # Threshold for visibility
                    effective_intensity = obj_intensity * z_response
                    self._add_gaussian_spot(image, obj_x, obj_y, int(obj_radius), effective_intensity)

            # Add minimal noise
            noise = np.random.normal(0, 2, size)
            image = np.clip(image + noise, 0, 255)
            stack.append(image.astype(np.uint8))

        return np.array(stack)

    def _simulate_fluorescence_image(self, size: Tuple[int, int], emission_wl: float,
                                    quantum_yield: float, concentration: float,
                                    photobleaching: float) -> np.ndarray:
        """Simulate fluorescence image"""
        image = np.zeros(size, dtype=np.float32)

        # Add fluorescent regions
        num_regions = int(20 * concentration)
        for _ in range(num_regions):
            x = np.random.randint(20, size[0]-20)
            y = np.random.randint(20, size[1]-20)
            radius = np.random.randint(10, 40)

            # Intensity depends on quantum yield and photobleaching
            base_intensity = 200 * quantum_yield * (1 - photobleaching)
            intensity = np.random.uniform(0.7 * base_intensity, base_intensity)

            self._add_gaussian_spot(image, x, y, radius, intensity)

        # Add autofluorescence background
        background = np.random.uniform(5, 15, size)
        image += background

        # Add Poisson noise (shot noise from photons)
        image = np.random.poisson(image).astype(np.float32)

        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def _simulate_polarized_light_image(self, size: Tuple[int, int], birefringence: float,
                                       crystal_orientation: float, analyzer_angle: float,
                                       compensator: Optional[str]) -> np.ndarray:
        """Simulate polarized light microscopy image"""
        # Malus' law: I = I0 * cos²(θ)
        # For birefringent materials: interference colors appear

        image = np.zeros(size, dtype=np.float32)

        # Create regions with different orientations
        for _ in range(10):
            x = np.random.randint(50, size[0]-50)
            y = np.random.randint(50, size[1]-50)
            radius = np.random.randint(30, 80)

            # Local crystal orientation
            local_orientation = np.random.uniform(0, 180)

            # Calculate transmitted intensity
            angle_diff = local_orientation - analyzer_angle
            retardation = birefringence * 10000  # nm (assume 10 μm thickness)

            # Add compensator effect
            if compensator == 'lambda':
                retardation += 550  # Add full wave
            elif compensator == 'quarter_wave':
                retardation += 137.5  # Add quarter wave

            # Interference intensity
            phase = 2 * np.pi * retardation / 550  # 550 nm wavelength
            intensity = (np.sin(np.radians(angle_diff)) ** 2) * (np.sin(phase/2) ** 2) * 255

            self._add_crystal_region(image, x, y, radius, intensity)

        # Add noise
        noise = np.random.normal(0, 5, size)
        image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)

    def _simulate_hologram(self, size: Tuple[int, int], wavelength: float,
                          phase_profile: Optional[np.ndarray]) -> np.ndarray:
        """Simulate digital hologram"""
        if phase_profile is None:
            # Create random phase profile
            phase_profile = np.random.uniform(0, 2*np.pi, size)

            # Add phase objects
            for _ in range(10):
                x = np.random.randint(50, size[0]-50)
                y = np.random.randint(50, size[1]-50)
                radius = np.random.randint(20, 50)
                phase_val = np.random.uniform(0, 2*np.pi)
                self._add_phase_object(phase_profile, x, y, radius, phase_val)

        # Create reference wave
        k = 2 * np.pi / wavelength  # Wave vector
        y, x = np.ogrid[:size[0], :size[1]]
        reference_angle = np.radians(3)  # 3 degree off-axis
        reference = np.exp(1j * k * x * np.sin(reference_angle))

        # Object wave
        object_wave = np.exp(1j * phase_profile)

        # Interference (hologram)
        hologram = np.abs(object_wave + reference) ** 2

        # Normalize
        hologram = ((hologram - hologram.min()) / (hologram.max() - hologram.min()) * 255)

        return hologram.astype(np.uint8)

    # Analysis helper methods

    def _detect_features_brightfield(self, image: np.ndarray, resolution: float) -> List[Dict[str, Any]]:
        """Detect features in brightfield image"""
        from scipy import ndimage

        # Threshold image
        threshold = np.mean(image) - np.std(image)
        binary = image < threshold

        # Label features
        labeled, num_features = ndimage.label(binary)

        features = []
        for i in range(1, num_features + 1):
            feature_mask = labeled == i
            area_pixels = np.sum(feature_mask)
            area_um2 = area_pixels * (resolution ** 2)

            # Calculate centroid
            coords = np.argwhere(feature_mask)
            centroid = coords.mean(axis=0)

            # Calculate equivalent diameter
            diameter_um = 2 * np.sqrt(area_um2 / np.pi)

            features.append({
                'id': i,
                'area_um2': float(area_um2),
                'diameter_um': float(diameter_um),
                'centroid': centroid.tolist(),
                'mean_intensity': float(np.mean(image[feature_mask]))
            })

        return features

    def _analyze_grains(self, image: np.ndarray, resolution: float) -> Dict[str, Any]:
        """Analyze grain structure in reflection image"""
        from scipy import ndimage

        # Edge detection for grain boundaries
        edges = ndimage.sobel(image)
        boundaries = edges > np.percentile(edges, 85)

        # Watershed segmentation for grains
        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(~boundaries)
        local_max = distance > np.percentile(distance, 90)
        markers, num_grains = ndimage.label(local_max)

        # Calculate grain sizes
        grain_sizes = []
        for i in range(1, num_grains + 1):
            grain_mask = markers == i
            area_pixels = np.sum(grain_mask)
            area_um2 = area_pixels * (resolution ** 2)
            grain_sizes.append(area_um2)

        return {
            'grain_count': num_grains,
            'mean_grain_area_um2': float(np.mean(grain_sizes)) if grain_sizes else 0,
            'grain_size_distribution': {
                'min': float(np.min(grain_sizes)) if grain_sizes else 0,
                'max': float(np.max(grain_sizes)) if grain_sizes else 0,
                'std': float(np.std(grain_sizes)) if grain_sizes else 0
            },
            'grain_boundaries': int(np.sum(boundaries)),
            'phase_count': 1,  # Simplified
            'uniformity_index': float(1 - np.std(grain_sizes) / (np.mean(grain_sizes) + 1e-6)) if grain_sizes else 0
        }

    def _detect_scattering_centers(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect scattering centers in darkfield image"""
        from scipy import ndimage

        # Threshold bright spots
        threshold = np.mean(image) + 2 * np.std(image)
        bright_spots = image > threshold

        # Label spots
        labeled, num_spots = ndimage.label(bright_spots)

        intensities = []
        sizes = []

        for i in range(1, num_spots + 1):
            spot_mask = labeled == i
            intensities.append(np.mean(image[spot_mask]))
            sizes.append(np.sum(spot_mask))

        return {
            'count': num_spots,
            'density': num_spots / (image.shape[0] * image.shape[1]) if image.size > 0 else 0,
            'mean_intensity': float(np.mean(intensities)) if intensities else 0,
            'intensity_distribution': {
                'min': float(np.min(intensities)) if intensities else 0,
                'max': float(np.max(intensities)) if intensities else 0,
                'std': float(np.std(intensities)) if intensities else 0
            },
            'size_distribution': {
                'mean_pixels': float(np.mean(sizes)) if sizes else 0,
                'std_pixels': float(np.std(sizes)) if sizes else 0
            }
        }

    def _analyze_phase_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze phase contrast features"""
        # Estimate optical path differences from intensity
        # Phase contrast: I ≈ I0 * (1 + 2*φ) for small phase

        mean_intensity = np.mean(image)
        phase_estimate = (image - mean_intensity) / (2 * mean_intensity + 1e-6)

        # Convert to OPD (assuming 550 nm wavelength)
        opd_nm = phase_estimate * 550 / (2 * np.pi)

        # Detect halo artifacts
        from scipy import ndimage
        gradient = ndimage.gaussian_gradient_magnitude(image.astype(float), sigma=2)
        halo_detected = np.percentile(gradient, 95) > np.percentile(gradient, 50) * 2

        return {
            'mean_opd': float(np.mean(opd_nm)),
            'std_opd': float(np.std(opd_nm)),
            'max_opd': float(np.max(np.abs(opd_nm))),
            'halo_detected': bool(halo_detected),
            'halo_intensity': float(np.percentile(gradient, 95)) if halo_detected else 0,
            'contrast': float(np.std(image) / (np.mean(image) + 1e-6))
        }

    def _analyze_dic_gradients(self, image: np.ndarray, shear_nm: float) -> Dict[str, Any]:
        """Analyze DIC gradients and estimate topography"""
        from scipy import ndimage

        # Calculate gradients
        grad_x = ndimage.sobel(image.astype(float), axis=1)
        grad_y = ndimage.sobel(image.astype(float), axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Estimate slope angles
        # DIC intensity ∝ gradient of OPD
        # Assume typical contrast: 50 gray levels per π phase shift
        slopes_rad = gradient_magnitude / 50 * (np.pi / 100)
        slopes_deg = np.degrees(slopes_rad)

        # Estimate relief from shear and intensity
        relief_estimate_nm = (image.astype(float) - 128) / 127 * shear_nm * 10

        return {
            'mean_gradient': float(np.mean(gradient_magnitude)),
            'max_gradient': float(np.max(gradient_magnitude)),
            'slope_angles': [float(np.mean(slopes_deg)), float(np.std(slopes_deg))],
            'relief_range': float(np.ptp(relief_estimate_nm)),
            'edge_sharpness': float(np.percentile(gradient_magnitude, 95))
        }

    def _analyze_confocal_features(self, image_stack: np.ndarray) -> Dict[str, Any]:
        """Analyze confocal image or stack"""
        if image_stack.ndim == 3:
            # Z-stack analysis
            num_slices = image_stack.shape[0]

            # Find objects in each slice
            objects_per_slice = []
            for z_slice in image_stack:
                threshold = np.mean(z_slice) + np.std(z_slice)
                binary = z_slice > threshold
                from scipy import ndimage
                labeled, num = ndimage.label(binary)
                objects_per_slice.append(num)

            # Maximum intensity projection
            mip = np.max(image_stack, axis=0)

            return {
                'type': 'z_stack',
                'num_slices': num_slices,
                'objects_per_slice': objects_per_slice,
                'total_unique_objects': int(np.max(objects_per_slice)),
                'mip_intensity_range': [float(np.min(mip)), float(np.max(mip))]
            }
        else:
            # Single image
            threshold = np.mean(image_stack) + np.std(image_stack)
            binary = image_stack > threshold
            from scipy import ndimage
            labeled, num_objects = ndimage.label(binary)

            return {
                'type': 'single_image',
                'num_objects': num_objects,
                'mean_intensity': float(np.mean(image_stack)),
                'contrast': float(np.std(image_stack) / (np.mean(image_stack) + 1e-6))
            }

    def _analyze_fluorescence_channel(self, image: np.ndarray, channel_name: str) -> Dict[str, Any]:
        """Analyze single fluorescence channel"""
        # Calculate SNR
        signal = np.percentile(image, 95)
        noise = np.std(image[image < np.percentile(image, 10)])
        snr = signal / (noise + 1e-6)

        # Detect fluorescent objects
        threshold = np.mean(image) + 2 * np.std(image)
        binary = image > threshold

        from scipy import ndimage
        labeled, num_objects = ndimage.label(binary)

        # Calculate intensities
        intensities = []
        for i in range(1, num_objects + 1):
            obj_mask = labeled == i
            intensities.append(np.mean(image[obj_mask]))

        return {
            'channel_name': channel_name,
            'snr': float(snr),
            'num_objects': num_objects,
            'mean_signal_intensity': float(np.mean(intensities)) if intensities else 0,
            'background_level': float(np.percentile(image, 5)),
            'saturation_percentage': float(np.sum(image >= 255) / image.size * 100)
        }

    def _analyze_colocalization(self, image1: np.ndarray, image2: np.ndarray,
                               name1: str, name2: str) -> Dict[str, Any]:
        """Analyze colocalization between two channels"""
        # Pearson correlation coefficient
        pearson_r = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]

        # Manders' coefficients
        threshold1 = np.mean(image1) + np.std(image1)
        threshold2 = np.mean(image2) + np.std(image2)

        above1 = image1 > threshold1
        above2 = image2 > threshold2

        # M1: fraction of channel 1 overlapping with channel 2
        m1 = np.sum(image1[above2]) / (np.sum(image1) + 1e-6)

        # M2: fraction of channel 2 overlapping with channel 1
        m2 = np.sum(image2[above1]) / (np.sum(image2) + 1e-6)

        # Overlap coefficient
        overlap = np.sum(np.minimum(image1, image2)) / np.sqrt(np.sum(image1**2) * np.sum(image2**2) + 1e-6)

        return {
            'channel_1': name1,
            'channel_2': name2,
            'pearson_correlation': float(pearson_r),
            'manders_m1': float(m1),
            'manders_m2': float(m2),
            'overlap_coefficient': float(overlap),
            'colocalization_level': 'high' if pearson_r > 0.7 else 'moderate' if pearson_r > 0.4 else 'low'
        }

    def _analyze_birefringence(self, image: np.ndarray, analyzer_angle: float,
                              compensator: Optional[str]) -> Dict[str, Any]:
        """Analyze birefringence from polarized light image"""
        # Find extinction positions (minimum intensity)
        min_intensity = np.min(image)
        extinction_mask = image < (min_intensity + 20)

        # Estimate extinction angle
        from scipy import ndimage
        if np.sum(extinction_mask) > 0:
            y_coords, x_coords = np.where(extinction_mask)
            extinction_angle = float(np.mean(np.arctan2(y_coords - image.shape[0]/2,
                                                         x_coords - image.shape[1]/2)))
            extinction_angle = np.degrees(extinction_angle) % 180
        else:
            extinction_angle = 0

        # Estimate retardation from intensity variation
        intensity_range = np.max(image) - np.min(image)
        # Rough estimate: full sine wave = 1st order (550 nm retardation)
        retardation_nm = (intensity_range / 255) * 550

        # Michel-Levy order
        michel_levy_order = int(retardation_nm / 550)

        # Determine if birefringent
        is_birefringent = intensity_range > 50  # Threshold

        # Optic sign (simplified)
        if compensator == 'quarter_wave':
            # Check if adding or subtracting retardation
            optic_sign = 'positive' if np.mean(image) > 128 else 'negative'
        else:
            optic_sign = 'unknown'

        return {
            'is_birefringent': bool(is_birefringent),
            'extinction_angle': float(extinction_angle),
            'retardation': float(retardation_nm),
            'michel_levy_order': michel_levy_order,
            'optic_sign': optic_sign,
            'interference_color': self._retardation_to_color(retardation_nm)
        }

    def _reconstruct_hologram(self, hologram: np.ndarray, wavelength: float) -> Dict[str, np.ndarray]:
        """Reconstruct phase and amplitude from hologram"""
        # Simplified digital holography reconstruction
        # In practice, would use Fourier transform method

        # FFT
        hologram_fft = np.fft.fft2(hologram)
        hologram_fft_shifted = np.fft.fftshift(hologram_fft)

        # Filter to select +1 order (simplified)
        mask = np.zeros_like(hologram_fft_shifted)
        center = np.array(hologram.shape) // 2
        mask[center[0]:, center[1]:] = 1

        filtered = hologram_fft_shifted * mask

        # Inverse FFT
        reconstructed = np.fft.ifft2(np.fft.ifftshift(filtered))

        # Extract amplitude and phase
        amplitude = np.abs(reconstructed)
        phase = np.angle(reconstructed)

        # Unwrap phase
        from scipy.ndimage import uniform_filter
        phase_unwrapped = phase + 2 * np.pi * np.round((uniform_filter(phase, size=5) - phase) / (2 * np.pi))

        # Normalize
        amplitude = ((amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-6) * 255).astype(np.uint8)

        return {
            'amplitude': amplitude,
            'phase': phase_unwrapped
        }

    def _analyze_quantitative_phase(self, phase_map: np.ndarray, wavelength: float) -> Dict[str, Any]:
        """Analyze quantitative phase map"""
        # Convert phase to optical path difference
        opd_nm = phase_map * wavelength / (2 * np.pi)

        # Calculate dry mass (for biological samples)
        # Dry mass ∝ integrated OPD
        alpha = 0.002  # specific refraction increment (pg/µm²/rad)
        pixel_area_um2 = 0.1  # Example
        dry_mass_pg = np.sum(phase_map) * alpha * pixel_area_um2

        return {
            'mean_phase_rad': float(np.mean(phase_map)),
            'std_phase_rad': float(np.std(phase_map)),
            'phase_range_rad': float(np.ptp(phase_map)),
            'mean_opd_nm': float(np.mean(opd_nm)),
            'opd_range': float(np.ptp(opd_nm)),
            'dry_mass': float(dry_mass_pg)
        }

    def _calculate_ri_from_phase(self, phase_map: np.ndarray, wavelength: float,
                                 reference_ri: float) -> Dict[str, Any]:
        """Calculate refractive index or thickness from phase"""
        # φ = 2π/λ * (n - n_ref) * t
        # If thickness known, can get RI; if RI known, can get thickness

        opd_nm = phase_map * wavelength / (2 * np.pi)

        # Assume typical cell/particle thickness of 5 µm
        assumed_thickness_um = 5.0
        delta_n = opd_nm / (assumed_thickness_um * 1000)
        calculated_ri = reference_ri + delta_n

        return {
            'reference_ri': reference_ri,
            'mean_calculated_ri': float(np.mean(calculated_ri)),
            'ri_range': [float(np.min(calculated_ri)), float(np.max(calculated_ri))],
            'assumed_thickness_um': assumed_thickness_um,
            'opd_range_nm': [float(np.min(opd_nm)), float(np.max(opd_nm))]
        }

    def _calculate_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics"""
        # Contrast
        contrast = float(np.std(image) / (np.mean(image) + 1e-6))

        # Sharpness (gradient-based)
        from scipy import ndimage
        gradient = ndimage.gaussian_gradient_magnitude(image.astype(float), sigma=1)
        sharpness = float(np.mean(gradient))

        # SNR estimate
        signal = float(np.percentile(image, 95))
        noise = float(np.std(image[image < np.percentile(image, 10)]))
        snr = signal / (noise + 1e-6)

        return {
            'contrast': contrast,
            'sharpness': sharpness,
            'snr': float(snr),
            'mean_intensity': float(np.mean(image)),
            'dynamic_range': float(np.ptp(image))
        }

    def _create_fluorescence_composite(self, images: List[np.ndarray],
                                      channels: List[Dict[str, Any]]) -> np.ndarray:
        """Create RGB composite from fluorescence channels"""
        if len(images) == 0:
            return np.zeros((512, 512, 3), dtype=np.uint8)

        # Create RGB image
        composite = np.zeros((*images[0].shape, 3), dtype=np.uint8)

        # Map channels to RGB
        color_map = {
            'DAPI': (0, 0, 255),     # Blue
            'FITC': (0, 255, 0),     # Green
            'TRITC': (255, 0, 0),    # Red
            'Cy5': (255, 0, 255),    # Magenta
            'default': (255, 255, 255)
        }

        for i, (image, channel) in enumerate(zip(images, channels)):
            color = color_map.get(channel['name'], color_map['default'])
            for c in range(3):
                if color[c] > 0:
                    composite[:, :, c] = np.maximum(composite[:, :, c],
                                                    (image * color[c] / 255).astype(np.uint8))

        return composite

    def _reconstruct_3d(self, stack: np.ndarray, z_positions: np.ndarray) -> Dict[str, Any]:
        """3D reconstruction from z-stack"""
        # Maximum intensity projection
        mip = np.max(stack, axis=0)

        # Average intensity projection
        avg = np.mean(stack, axis=0)

        # Calculate volume
        threshold = np.mean(stack) + np.std(stack)
        volume_voxels = np.sum(stack > threshold)
        voxel_size_um3 = 0.1 * 0.1 * (z_positions[1] - z_positions[0])  # Example
        volume_um3 = volume_voxels * voxel_size_um3

        return {
            'max_intensity_projection': mip,
            'average_intensity_projection': avg,
            'z_range_um': [float(z_positions.min()), float(z_positions.max())],
            'num_slices': len(z_positions),
            'volume_um3': float(volume_um3),
            'reconstruction_method': 'simple_projection'
        }

    # Additional helper methods for image generation

    def _add_fiber(self, image: np.ndarray, x: int, y: int, angle: float, length: int, absorption: float):
        """Add fiber structure to image"""
        angle_rad = np.radians(angle)
        for i in range(-length//2, length//2):
            xi = int(x + i * np.cos(angle_rad))
            yi = int(y + i * np.sin(angle_rad))
            if 0 <= xi < image.shape[0] and 0 <= yi < image.shape[1]:
                image[xi, yi] *= (1 - absorption)

    def _add_particle(self, image: np.ndarray, x: int, y: int, radius: int, absorption: float):
        """Add particle to image"""
        y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        image[mask] *= (1 - absorption)

    def _add_random_feature(self, image: np.ndarray, x: int, y: int):
        """Add random feature to image"""
        size = np.random.randint(10, 30)
        absorption = np.random.uniform(0.2, 0.6)
        self._add_particle(image, x, y, size, absorption)

    def _add_scatter_spot(self, image: np.ndarray, x: int, y: int, intensity: float, sigma: float):
        """Add scattering spot (Gaussian)"""
        y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
        gaussian = intensity * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        image[:] = np.minimum(image + gaussian, 255)

    def _add_phase_object(self, phase_map: np.ndarray, x: int, y: int, radius: int, phase_value: float):
        """Add phase object to phase map"""
        y_grid, x_grid = np.ogrid[:phase_map.shape[0], :phase_map.shape[1]]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        phase_map[mask] += phase_value

    def _add_topography_feature(self, height_map: np.ndarray, x: int, y: int, radius: int, height: float):
        """Add topographic feature (Gaussian bump)"""
        y_grid, x_grid = np.ogrid[:height_map.shape[0], :height_map.shape[1]]
        gaussian = height * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * (radius/3)**2))
        height_map[:] += gaussian

    def _add_gaussian_spot(self, image: np.ndarray, x: int, y: int, radius: int, intensity: float):
        """Add Gaussian spot to image"""
        y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
        gaussian = intensity * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * (radius/2.35)**2))
        image[:] += gaussian

    def _add_crystal_region(self, image: np.ndarray, x: int, y: int, radius: int, intensity: float):
        """Add crystalline region with specific intensity"""
        y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        image[mask] = intensity

    def _calculate_dof(self, na: float, wavelength: float, magnification: int) -> float:
        """Calculate depth of field"""
        # DOF ≈ λ*n / (NA²) + (n * e) / (M * NA)
        # where e is smallest resolvable distance in image plane (~10 µm for human eye)
        n = 1.0  # Air
        e_um = 10
        dof = (wavelength / 1000) * n / (na ** 2) + (n * e_um) / (magnification * na)
        return float(dof)

    def _calculate_optical_section_thickness(self, na: float) -> float:
        """Calculate optical section thickness for DIC"""
        # Approximate: similar to confocal without pinhole
        wavelength = 550  # nm
        thickness = 1.8 * (wavelength / 1000) / (na ** 2)
        return float(thickness)

    def _calculate_pinhole_rejection(self, pinhole_au: float) -> float:
        """Calculate out-of-focus light rejection efficiency"""
        # Smaller pinhole = better rejection
        # 1 AU = optimal trade-off
        if pinhole_au <= 1.0:
            rejection = 0.85
        elif pinhole_au <= 2.0:
            rejection = 0.70
        else:
            rejection = 0.50
        return rejection

    def _retardation_to_color(self, retardation_nm: float) -> str:
        """Convert retardation to Michel-Levy interference color"""
        if retardation_nm < 100:
            return 'gray'
        elif retardation_nm < 550:
            return 'first_order_gray_to_white'
        elif retardation_nm < 1100:
            return 'first_order_yellow_to_red'
        elif retardation_nm < 1650:
            return 'second_order_blue_to_green'
        else:
            order = int(retardation_nm / 550)
            return f'{order}_order'

    # Cross-validation methods

    @staticmethod
    def validate_with_sem(optical_result: Dict[str, Any], sem_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate optical microscopy with SEM"""
        # Compare feature sizes
        optical_features = optical_result.get('features_detected', [])
        sem_features = sem_result.get('features', [])

        if not optical_features or not sem_features:
            return {'validation': 'insufficient_data'}

        # Compare mean feature size
        optical_mean_size = np.mean([f['diameter_um'] for f in optical_features])
        sem_mean_size = sem_result.get('mean_feature_size_um', 0)

        size_agreement = abs(optical_mean_size - sem_mean_size) / (sem_mean_size + 1e-6) < 0.2

        return {
            'technique_pair': 'OpticalMicroscopy-SEM',
            'parameter': 'feature_size',
            'optical_value_um': float(optical_mean_size),
            'sem_value_um': float(sem_mean_size),
            'agreement': 'good' if size_agreement else 'poor',
            'relative_difference': float(abs(optical_mean_size - sem_mean_size) / (sem_mean_size + 1e-6)),
            'note': 'Optical resolution limited to ~0.2 μm; SEM provides higher resolution'
        }

    @staticmethod
    def validate_with_afm(optical_result: Dict[str, Any], afm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate optical (DIC/holographic) with AFM topography"""
        # Compare topography measurements
        optical_relief = optical_result.get('topography_estimation', {}).get('relief_range_nm', 0)
        afm_roughness = afm_result.get('roughness', {}).get('Rz', 0) * 1000  # Convert µm to nm

        if optical_relief == 0 or afm_roughness == 0:
            return {'validation': 'insufficient_data'}

        # DIC/DHM provides semi-quantitative topography
        agreement = abs(optical_relief - afm_roughness) / (afm_roughness + 1e-6) < 0.3

        return {
            'technique_pair': 'OpticalMicroscopy-AFM',
            'parameter': 'surface_relief',
            'optical_value_nm': float(optical_relief),
            'afm_value_nm': float(afm_roughness),
            'agreement': 'reasonable' if agreement else 'poor',
            'relative_difference': float(abs(optical_relief - afm_roughness) / (afm_roughness + 1e-6)),
            'note': 'AFM provides quantitative topography; optical methods are semi-quantitative'
        }

    @staticmethod
    def validate_with_xrd(plm_result: Dict[str, Any], xrd_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate polarized light microscopy with XRD crystallinity"""
        # Compare crystallinity assessment
        plm_birefringent = plm_result.get('crystallographic_info', {}).get('birefringent', False)
        xrd_crystalline = xrd_result.get('phase_composition', {}).get('crystalline_fraction', 0) > 0.1

        agreement = plm_birefringent == xrd_crystalline

        return {
            'technique_pair': 'PLM-XRD',
            'parameter': 'crystallinity',
            'plm_birefringent': plm_birefringent,
            'xrd_crystalline': xrd_crystalline,
            'agreement': 'good' if agreement else 'discrepancy',
            'note': 'PLM detects optical anisotropy; XRD confirms crystalline structure'
        }

    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters"""
        errors = []

        # Check required fields
        if 'technique' not in input_data:
            errors.append("Missing required field: technique")
        else:
            technique = input_data['technique'].lower()
            if technique not in self.SUPPORTED_TECHNIQUES:
                errors.append(f"Unsupported technique: {technique}. Supported: {self.SUPPORTED_TECHNIQUES}")

        # Validate objective
        if 'objective' in input_data:
            if input_data['objective'] not in self.OBJECTIVE_SPECS:
                errors.append(f"Invalid objective: {input_data['objective']}. Available: {list(self.OBJECTIVE_SPECS.keys())}")

        # Validate wavelength
        if 'wavelength' in input_data:
            wl = input_data['wavelength']
            if isinstance(wl, (int, float)):
                if not (300 <= wl <= 800):
                    errors.append(f"Wavelength {wl} nm out of range (300-800 nm)")
            elif isinstance(wl, str):
                if wl not in self.WAVELENGTHS:
                    errors.append(f"Unknown wavelength color: {wl}")

        # Validate z-stack parameters
        if 'z_stack' in input_data:
            z_params = input_data['z_stack']
            if 'start_um' in z_params and 'end_um' in z_params:
                if z_params['start_um'] >= z_params['end_um']:
                    errors.append("Z-stack start must be less than end")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def estimate_resources(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate computational and experimental resources"""
        technique = input_data.get('technique', 'brightfield_transmission')
        image_size = input_data.get('image_size', (1024, 1024))
        z_stack = input_data.get('z_stack', None)

        # Estimate acquisition time
        base_time_ms = {
            'brightfield_transmission': 50,
            'brightfield_reflection': 50,
            'darkfield': 100,
            'phase_contrast': 80,
            'dic': 100,
            'confocal': 500,
            'fluorescence': 200,
            'polarized_light': 150,
            'digital_holographic': 100
        }.get(technique, 100)

        if z_stack:
            num_slices = int((z_stack['end_um'] - z_stack['start_um']) / z_stack.get('step_um', 0.5))
            total_time_s = base_time_ms * num_slices / 1000
        else:
            total_time_s = base_time_ms / 1000

        # Estimate memory
        bytes_per_pixel = 1 if technique != 'fluorescence' else 3
        if z_stack:
            memory_mb = image_size[0] * image_size[1] * num_slices * bytes_per_pixel / 1e6
        else:
            memory_mb = image_size[0] * image_size[1] * bytes_per_pixel / 1e6

        return {
            'estimated_time_seconds': float(total_time_s),
            'estimated_memory_mb': float(memory_mb),
            'cpu_intensive': technique in ['confocal', 'digital_holographic'],
            'gpu_acceleration': technique in ['confocal', 'digital_holographic'],
            'recommended_parallelization': z_stack is not None
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
            'available_objectives': list(self.OBJECTIVE_SPECS.keys()),
            'wavelength_options': list(self.WAVELENGTHS.keys()),
            'resolution_range_um': [0.2, 3.0],
            'magnification_range': [4, 1000],
            'capabilities': self.capabilities,
            'cross_validation_methods': [
                'validate_with_sem',
                'validate_with_afm',
                'validate_with_xrd'
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
    agent = OpticalMicroscopyAgent()

    # Example: Brightfield transmission
    result = agent.execute({
        'technique': 'brightfield_transmission',
        'objective': '40x',
        'wavelength': 'white',
        'sample_info': {'type': 'fibers'},
        'exposure_time': 100
    })
    print("Brightfield transmission result:", result.status)

    # Example: Confocal z-stack
    result = agent.execute({
        'technique': 'confocal',
        'objective': '60x',
        'laser_wavelength': 488,
        'pinhole_au': 1.0,
        'z_stack': {
            'start_um': 0,
            'end_um': 20,
            'step_um': 0.5
        }
    })
    print("Confocal z-stack result:", result.status)
    print(f"Number of slices: {result.data.get('z_stack_info', {}).get('slices', 0)}")
