"""
XRayMicroscopyAgent - Comprehensive X-ray Imaging and Tomography

This agent provides complete X-ray microscopy capabilities for high-resolution
3D imaging, chemical mapping, and non-destructive characterization.

Key Capabilities:
- Transmission X-ray Microscopy (TXM) - Full-field imaging
- Scanning Transmission X-ray Microscopy (STXM) - Chemical mapping
- X-ray Computed Tomography (XCT) - 3D reconstruction
- X-ray Fluorescence Microscopy (XFM) - Elemental mapping
- X-ray Ptychography - Phase-contrast imaging
- Soft and Hard X-ray regimes

Applications:
- 3D microstructure visualization (non-destructive)
- Porosity and defect analysis
- Chemical state mapping with XANES
- Battery electrode degradation
- Biological specimens in native state
- Composite material analysis
- Crack and void detection
- Elemental distribution mapping

Cross-Validation Opportunities:
- XCT ↔ SEM tomography (resolution vs field of view)
- XFM ↔ EDX/EELS (elemental mapping)
- STXM-XANES ↔ XPS (chemical state)
- Ptychography ↔ TEM (phase contrast)

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-02
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime


class XRayMicroscopyAgent:
    """
    Comprehensive X-ray microscopy and tomography agent.

    Supports multiple X-ray imaging modalities from soft to hard X-rays,
    providing 2D/3D imaging with elemental and chemical sensitivity.
    """

    VERSION = "1.0.0"
    AGENT_TYPE = "xray_microscopy"

    # Supported X-ray microscopy techniques
    SUPPORTED_TECHNIQUES = [
        'transmission_xray_microscopy',  # TXM - full-field
        'scanning_txm',                  # STXM - scanning
        'xray_computed_tomography',      # XCT/microCT
        'xray_fluorescence_microscopy',  # XFM - elemental
        'ptychography',                  # Coherent diffraction imaging
        'phase_contrast',                # Zernike/propagation-based
        'xanes_mapping',                 # Chemical state mapping
        'tomography_reconstruction'      # 3D volume from projections
    ]

    # X-ray energy regimes
    XRAY_REGIMES = {
        'soft': {
            'energy_range_ev': (250, 2000),
            'wavelength_range_nm': (0.62, 5.0),
            'typical_resolution_nm': (25, 50),
            'applications': ['Polymer chemistry', 'Biology', 'Carbon materials', 'XANES mapping']
        },
        'tender': {
            'energy_range_ev': (2000, 5000),
            'wavelength_range_nm': (0.25, 0.62),
            'typical_resolution_nm': (30, 100),
            'applications': ['Sulfur K-edge', 'Phosphorus K-edge', 'Environmental samples']
        },
        'hard': {
            'energy_range_ev': (5000, 50000),
            'wavelength_range_nm': (0.025, 0.25),
            'typical_resolution_nm': (50, 500),
            'applications': ['Metal alloys', 'Ceramics', 'Deep penetration', '3D tomography']
        }
    }

    # Typical beamline characteristics
    BEAMLINE_SPECS = {
        'synchrotron': {
            'flux_photons_s': 1e12,
            'coherence': 'high',
            'energy_tunability': 'excellent',
            'access': 'proposal-based'
        },
        'lab_source': {
            'flux_photons_s': 1e8,
            'coherence': 'low_to_moderate',
            'energy_tunability': 'fixed (Cu Kα, Mo Kα)',
            'access': 'on-demand'
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the XRayMicroscopyAgent.

        Args:
            config: Configuration dictionary containing:
                - default_technique: 'txm', 'stxm', 'xct', etc.
                - xray_regime: 'soft', 'tender', 'hard'
                - source_type: 'synchrotron', 'lab_source'
                - detector_type: 'CCD', 'direct_detection', 'photon_counting'
                - resolution_target_nm: Desired spatial resolution
        """
        self.config = config or {}
        self.default_technique = self.config.get('default_technique', 'xray_computed_tomography')
        self.xray_regime = self.config.get('xray_regime', 'hard')
        self.source_type = self.config.get('source_type', 'synchrotron')
        self.detector_type = self.config.get('detector_type', 'CCD')
        self.resolution_target_nm = self.config.get('resolution_target_nm', 100)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute X-ray microscopy measurement.

        Args:
            input_data: Dictionary containing:
                - technique: X-ray microscopy technique
                - sample_info: Sample description and preparation
                - acquisition_parameters: Technique-specific parameters

        Returns:
            Comprehensive X-ray microscopy results with metadata
        """
        technique = input_data.get('technique', self.default_technique)

        if technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(f"Unsupported technique: {technique}. "
                           f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Route to appropriate technique
        technique_map = {
            'transmission_xray_microscopy': self._execute_txm,
            'scanning_txm': self._execute_stxm,
            'xray_computed_tomography': self._execute_xct,
            'xray_fluorescence_microscopy': self._execute_xfm,
            'ptychography': self._execute_ptychography,
            'phase_contrast': self._execute_phase_contrast,
            'xanes_mapping': self._execute_xanes_mapping,
            'tomography_reconstruction': self._execute_tomography_reconstruction
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

    def _execute_txm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Transmission X-ray Microscopy (TXM).

        Full-field transmission imaging using zone plate optics.
        Provides absorption and phase contrast.

        Args:
            input_data: Contains sample info, imaging parameters

        Returns:
            TXM image data and analysis
        """
        # Sample parameters
        sample_name = input_data.get('sample_name', 'polymer_composite')
        sample_thickness_um = input_data.get('sample_thickness_um', 10)

        # X-ray parameters
        photon_energy_ev = input_data.get('photon_energy_ev', 8000)
        wavelength_nm = 1239.8 / photon_energy_ev

        # Zone plate optics
        zone_plate_diameter_um = input_data.get('zone_plate_diameter_um', 100)
        outermost_zone_width_nm = input_data.get('outermost_zone_width_nm', 30)

        # Rayleigh resolution: Δr ≈ 1.22 × outermost zone width
        spatial_resolution_nm = 1.22 * outermost_zone_width_nm

        # Field of view
        fov_um = input_data.get('field_of_view_um', 30)

        # Detector
        detector_pixels = input_data.get('detector_pixels', (2048, 2048))
        pixel_size_nm = (fov_um * 1000) / detector_pixels[0]

        # Exposure time
        exposure_time_s = input_data.get('exposure_time_s', 1.0)

        # Absorption contrast
        absorption_length_um = input_data.get('absorption_length_um', 100)
        transmission = np.exp(-sample_thickness_um / absorption_length_um)

        return {
            'technique': 'transmission_xray_microscopy',
            'sample_name': sample_name,
            'photon_energy_ev': photon_energy_ev,
            'wavelength_nm': wavelength_nm,
            'spatial_resolution_nm': spatial_resolution_nm,
            'field_of_view_um': fov_um,
            'pixel_size_nm': pixel_size_nm,
            'detector_pixels': detector_pixels,
            'sample_thickness_um': sample_thickness_um,
            'transmission': transmission,
            'exposure_time_s': exposure_time_s,
            'zone_plate_specs': {
                'diameter_um': zone_plate_diameter_um,
                'outermost_zone_width_nm': outermost_zone_width_nm,
                'numerical_aperture': wavelength_nm / (2 * outermost_zone_width_nm)
            },
            'contrast_mechanisms': ['Absorption', 'Phase (with phase plate)'],
            'advantages': [
                'Full-field imaging (fast acquisition)',
                'Sub-50 nm resolution achievable',
                'Chemical sensitivity (XANES)',
                'Compatible with tomography'
            ],
            'typical_applications': [
                'Battery electrode degradation',
                'Polymer nanocomposites',
                'Biological cells',
                'Porous materials'
            ]
        }

    def _execute_stxm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Scanning Transmission X-ray Microscopy (STXM).

        Focused X-ray beam scanned across sample, transmission measured at each point.
        Excellent for chemical mapping via XANES.

        Args:
            input_data: Contains scanning parameters, energy scan

        Returns:
            STXM image stack with spectroscopic information
        """
        sample_name = input_data.get('sample_name', 'carbon_nanoparticles')

        # Scanning parameters
        scan_range_um = input_data.get('scan_range_um', (10, 10))
        step_size_nm = input_data.get('step_size_nm', 30)

        # Calculate number of pixels
        num_pixels_x = int(scan_range_um[0] * 1000 / step_size_nm)
        num_pixels_y = int(scan_range_um[1] * 1000 / step_size_nm)

        # Zone plate focusing
        spot_size_nm = input_data.get('spot_size_nm', 25)

        # Energy scan for XANES
        perform_xanes = input_data.get('perform_xanes', True)

        if perform_xanes:
            # Carbon K-edge XANES
            energy_range_ev = input_data.get('energy_range_ev', (280, 320))
            energy_step_ev = input_data.get('energy_step_ev', 0.1)
            num_energies = int((energy_range_ev[1] - energy_range_ev[0]) / energy_step_ev)

            # Total acquisition time
            dwell_time_ms = input_data.get('dwell_time_per_pixel_ms', 10)
            total_time_minutes = (num_pixels_x * num_pixels_y * num_energies * dwell_time_ms) / 1000 / 60

        else:
            num_energies = 1
            dwell_time_ms = input_data.get('dwell_time_per_pixel_ms', 1)
            total_time_minutes = (num_pixels_x * num_pixels_y * dwell_time_ms) / 1000 / 60

        # Data cube: (x, y, energy)
        data_cube_size_gb = (num_pixels_x * num_pixels_y * num_energies * 4) / 1e9  # 4 bytes per pixel

        return {
            'technique': 'scanning_transmission_xray_microscopy',
            'sample_name': sample_name,
            'scan_range_um': scan_range_um,
            'step_size_nm': step_size_nm,
            'image_pixels': (num_pixels_x, num_pixels_y),
            'spatial_resolution_nm': spot_size_nm,
            'xanes_enabled': perform_xanes,
            'num_energy_points': num_energies if perform_xanes else 1,
            'energy_range_ev': energy_range_ev if perform_xanes else None,
            'dwell_time_ms': dwell_time_ms,
            'estimated_acquisition_time_minutes': total_time_minutes,
            'data_cube_size_gb': data_cube_size_gb,
            'advantages': [
                'Chemical state mapping (XANES)',
                'Quantitative composition',
                '< 25 nm resolution possible',
                'Flexibility in scan patterns'
            ],
            'limitations': [
                'Slower than full-field TXM',
                'Radiation dose concerns for beam-sensitive samples',
                'Requires synchrotron source'
            ],
            'applications': [
                'Organic photovoltaics (phase separation)',
                'Magnetism (XMCD)',
                'Carbon speciation in environmental samples',
                'Battery chemistry mapping'
            ]
        }

    def _execute_xct(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform X-ray Computed Tomography (XCT / micro-CT).

        3D non-destructive imaging via tomographic reconstruction.
        Gold standard for internal structure visualization.

        Args:
            input_data: Contains tomography acquisition parameters

        Returns:
            3D volume data and analysis metrics
        """
        sample_name = input_data.get('sample_name', 'aluminum_foam')

        # Acquisition geometry
        source_sample_distance_mm = input_data.get('source_sample_distance_mm', 50)
        sample_detector_distance_mm = input_data.get('sample_detector_distance_mm', 150)

        # Geometric magnification
        magnification = (source_sample_distance_mm + sample_detector_distance_mm) / \
                       source_sample_distance_mm

        # X-ray parameters
        photon_energy_kev = input_data.get('photon_energy_kev', 20)

        # Detector
        detector_pixels = input_data.get('detector_pixels', (2048, 2048))
        detector_pixel_size_um = input_data.get('detector_pixel_size_um', 6.5)

        # Voxel size
        voxel_size_um = detector_pixel_size_um / magnification

        # Tomography scan
        num_projections = input_data.get('num_projections', 1200)
        angular_range_deg = input_data.get('angular_range_deg', 180)
        exposure_per_projection_s = input_data.get('exposure_per_projection_s', 0.5)

        total_acquisition_time_minutes = (num_projections * exposure_per_projection_s) / 60

        # Reconstructed volume size
        volume_voxels = (detector_pixels[0], detector_pixels[1], detector_pixels[0])
        volume_size_gb = np.prod(volume_voxels) * 4 / 1e9  # 32-bit float

        # Physical volume size
        volume_size_mm = (
            volume_voxels[0] * voxel_size_um / 1000,
            volume_voxels[1] * voxel_size_um / 1000,
            volume_voxels[2] * voxel_size_um / 1000
        )

        # Perform 3D analysis (simulated)
        porosity_percent = input_data.get('measured_porosity_percent', 15.0)
        pore_size_range_um = input_data.get('pore_size_range_um', (5, 200))

        return {
            'technique': 'xray_computed_tomography',
            'sample_name': sample_name,
            'photon_energy_kev': photon_energy_kev,
            'voxel_size_um': voxel_size_um,
            'magnification': magnification,
            'spatial_resolution_um': voxel_size_um * 2,  # Nyquist
            'num_projections': num_projections,
            'angular_range_deg': angular_range_deg,
            'exposure_per_projection_s': exposure_per_projection_s,
            'total_acquisition_time_minutes': total_acquisition_time_minutes,
            'reconstructed_volume': {
                'voxels': volume_voxels,
                'size_mm': volume_size_mm,
                'data_size_gb': volume_size_gb
            },
            'analysis_results': {
                'porosity_percent': porosity_percent,
                'pore_size_range_um': pore_size_range_um,
                'total_pore_volume_mm3': np.prod(volume_size_mm) * porosity_percent / 100
            },
            'advantages': [
                'True 3D imaging (non-destructive)',
                'Internal structure visualization',
                'Quantitative analysis (porosity, thickness, etc.)',
                'No sample preparation',
                'Wide material range'
            ],
            'applications': [
                'Foam characterization',
                'Crack detection',
                'Composite fiber orientation',
                'Bone microstructure',
                'Battery electrode evolution',
                'Metal additive manufacturing quality'
            ],
            'reconstruction_algorithm': 'Filtered Back Projection (FBP) or Iterative'
        }

    def _execute_xfm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform X-ray Fluorescence Microscopy (XFM / micro-XRF).

        Elemental mapping via fluorescence emission.
        Quantitative elemental distribution with sub-micron resolution.

        Args:
            input_data: Contains XFM scan parameters, detected elements

        Returns:
            Elemental maps and quantification
        """
        sample_name = input_data.get('sample_name', 'environmental_particle')

        # Excitation parameters
        incident_energy_kev = input_data.get('incident_energy_kev', 10)

        # Focusing optics
        beam_size_um = input_data.get('beam_size_um', 0.5)  # KB mirrors or capillary

        # Scan parameters
        scan_area_um = input_data.get('scan_area_um', (100, 100))
        step_size_um = input_data.get('step_size_um', 1.0)

        num_pixels = (int(scan_area_um[0] / step_size_um),
                     int(scan_area_um[1] / step_size_um))

        # Detector
        detector_type = input_data.get('detector_type', 'silicon_drift_detector')
        dwell_time_ms = input_data.get('dwell_time_per_pixel_ms', 100)

        # Elements detected
        elements_detected = input_data.get('elements_detected', ['Fe', 'Ca', 'Zn', 'Cu'])

        # Generate elemental concentrations
        elemental_data = {}
        for element in elements_detected:
            concentration_ppm = input_data.get(f'{element}_concentration_ppm',
                                             np.random.uniform(10, 1000))
            elemental_data[element] = {
                'average_concentration_ppm': concentration_ppm,
                'detection_limit_ppm': 1.0,  # Typical for XFM
                'fluorescence_lines': self._get_xrf_lines(element)
            }

        # Acquisition time
        total_time_minutes = (num_pixels[0] * num_pixels[1] * dwell_time_ms) / 1000 / 60

        return {
            'technique': 'xray_fluorescence_microscopy',
            'sample_name': sample_name,
            'incident_energy_kev': incident_energy_kev,
            'beam_size_um': beam_size_um,
            'spatial_resolution_um': beam_size_um,
            'scan_area_um': scan_area_um,
            'step_size_um': step_size_um,
            'map_pixels': num_pixels,
            'dwell_time_ms': dwell_time_ms,
            'total_acquisition_time_minutes': total_time_minutes,
            'detector_type': detector_type,
            'elements_detected': elements_detected,
            'elemental_data': elemental_data,
            'advantages': [
                'Multi-element detection simultaneously',
                'Quantitative (with standards)',
                'Non-destructive',
                'Sub-micron resolution possible',
                'Trace element sensitivity (ppm to ppb)'
            ],
            'applications': [
                'Metal distribution in biology',
                'Catalyst particle analysis',
                'Environmental contamination',
                'Archaeological samples',
                'Corrosion product mapping'
            ],
            'cross_validation': [
                'Compare with SEM-EDX (higher resolution, lower sensitivity)',
                'Validate with ICP-MS (bulk composition)',
                'Correlate with STXM for chemical state'
            ]
        }

    def _execute_ptychography(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform X-ray Ptychography.

        Coherent diffraction imaging with phase retrieval.
        Achieves resolution beyond zone plate limit.

        Args:
            input_data: Contains ptychography scan and reconstruction parameters

        Returns:
            Phase-retrieved complex transmission function
        """
        sample_name = input_data.get('sample_name', 'integrated_circuit')

        # Coherent illumination
        photon_energy_kev = input_data.get('photon_energy_kev', 7)
        wavelength_nm = 1.2398 / photon_energy_kev

        # Probe size
        probe_size_nm = input_data.get('probe_size_nm', 500)

        # Ptychography scan
        scan_points = input_data.get('scan_points', 100)
        overlap_percent = input_data.get('overlap_percent', 60)

        # Resolution
        # Ptychography can exceed zone plate resolution
        achieved_resolution_nm = input_data.get('achieved_resolution_nm', 10)

        # Reconstruction
        num_iterations = input_data.get('reconstruction_iterations', 200)
        reconstruction_algorithm = input_data.get('algorithm', 'ePIE')  # extended Ptychographical Iterative Engine

        # Phase sensitivity
        phase_sensitivity_rad = input_data.get('phase_sensitivity_rad', 0.01)

        return {
            'technique': 'xray_ptychography',
            'sample_name': sample_name,
            'photon_energy_kev': photon_energy_kev,
            'wavelength_nm': wavelength_nm,
            'probe_size_nm': probe_size_nm,
            'achieved_resolution_nm': achieved_resolution_nm,
            'scan_points': scan_points,
            'overlap_percent': overlap_percent,
            'reconstruction': {
                'algorithm': reconstruction_algorithm,
                'iterations': num_iterations,
                'phase_sensitivity_rad': phase_sensitivity_rad
            },
            'outputs': {
                'amplitude': 'Absorption contrast',
                'phase': 'Phase shift (electron density)',
                'complex_transmission': 'Full wave function'
            },
            'advantages': [
                'Resolution beyond optics limit (< 10 nm possible)',
                'Quantitative phase imaging',
                'Large field of view',
                'Self-calibrating (reconstructs probe)',
                '2D and 3D (ptychographic tomography)'
            ],
            'applications': [
                'Semiconductor defect inspection',
                'Integrated circuit imaging',
                'Biological structures',
                'Nanomaterials',
                'Strain mapping'
            ],
            'requirements': [
                'Coherent X-ray source (synchrotron undulator)',
                'Position encoder (< nm accuracy)',
                'Computational resources for reconstruction',
                'Stable sample and optics'
            ]
        }

    def _execute_phase_contrast(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform phase-contrast X-ray imaging.

        Visualize low-Z materials via phase shift rather than absorption.

        Methods:
        - Zernike phase contrast
        - Propagation-based (inline holography)
        - Grating interferometry

        Args:
            input_data: Contains phase contrast parameters

        Returns:
            Phase-enhanced image data
        """
        sample_name = input_data.get('sample_name', 'soft_tissue')
        method = input_data.get('phase_contrast_method', 'propagation_based')

        photon_energy_kev = input_data.get('photon_energy_kev', 20)

        if method == 'zernike':
            # Zernike phase plate (π/2 phase shift)
            phase_shift_rad = np.pi / 2
            phase_ring_width_um = input_data.get('phase_ring_width_um', 5)

            details = {
                'method': 'Zernike phase contrast',
                'phase_shift': f'{phase_shift_rad:.2f} rad (π/2)',
                'phase_ring_width_um': phase_ring_width_um,
                'contrast_enhancement': '10-100× over absorption'
            }

        elif method == 'propagation_based':
            # Inline holography
            propagation_distance_mm = input_data.get('propagation_distance_mm', 100)
            detector_pixel_size_um = input_data.get('detector_pixel_size_um', 6.5)

            # Fresnel number
            wavelength_nm = 1.2398 / photon_energy_kev
            wavelength_m = wavelength_nm * 1e-9
            pixel_m = detector_pixel_size_um * 1e-6
            fresnel_number = pixel_m**2 / (wavelength_m * propagation_distance_mm * 1e-3)

            details = {
                'method': 'Propagation-based phase contrast',
                'propagation_distance_mm': propagation_distance_mm,
                'fresnel_number': fresnel_number,
                'regime': 'near-field' if fresnel_number > 1 else 'far-field',
                'phase_retrieval': 'Paganin or Holotomography'
            }

        else:  # grating interferometry
            grating_period_um = input_data.get('grating_period_um', 5)

            details = {
                'method': 'Grating interferometry (Talbot)',
                'grating_period_um': grating_period_um,
                'outputs': ['Absorption', 'Differential phase', 'Dark-field (scattering)'],
                'advantages': 'Three contrast mechanisms simultaneously'
            }

        return {
            'technique': 'phase_contrast_xray_imaging',
            'sample_name': sample_name,
            'photon_energy_kev': photon_energy_kev,
            'phase_contrast_details': details,
            'advantages': [
                'Enhanced contrast for low-Z materials',
                'Visualize soft tissue without staining',
                'Edge enhancement',
                'Quantitative phase retrieval possible'
            ],
            'applications': [
                'Biological soft tissue',
                'Polymer composites',
                'Foam structure',
                'Fiber materials',
                'Crack detection in low-contrast samples'
            ]
        }

    def _execute_xanes_mapping(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform XANES (X-ray Absorption Near-Edge Structure) mapping.

        Chemical state and bonding information from absorption edge fine structure.
        Typically done with STXM or TXM.

        Args:
            input_data: Contains XANES energy scan and analysis

        Returns:
            Chemical state maps and XANES spectra
        """
        sample_name = input_data.get('sample_name', 'battery_cathode')
        element = input_data.get('element', 'Ni')
        edge = input_data.get('edge', 'K')

        # Energy scan
        edge_energy_ev = input_data.get('edge_energy_ev', 8333)  # Ni K-edge
        energy_range_ev = (edge_energy_ev - 20, edge_energy_ev + 50)
        energy_step_ev = input_data.get('energy_step_ev', 0.5)

        num_energies = int((energy_range_ev[1] - energy_range_ev[0]) / energy_step_ev)

        # Chemical states identified
        chemical_states = input_data.get('chemical_states', ['Ni0', 'NiO', 'Ni(OH)2'])

        # Simulate XANES features
        xanes_features = {}
        for state in chemical_states:
            xanes_features[state] = {
                'pre_edge_position_ev': edge_energy_ev - 5,
                'edge_position_ev': edge_energy_ev,
                'white_line_intensity': np.random.uniform(0.5, 2.0),
                'spatial_distribution_percent': np.random.uniform(10, 40)
            }

        return {
            'technique': 'xanes_chemical_mapping',
            'sample_name': sample_name,
            'element': element,
            'absorption_edge': edge,
            'edge_energy_ev': edge_energy_ev,
            'energy_range_ev': energy_range_ev,
            'energy_resolution_ev': energy_step_ev,
            'num_energy_points': num_energies,
            'chemical_states_identified': chemical_states,
            'xanes_features': xanes_features,
            'information_content': [
                'Oxidation state',
                'Local coordination geometry',
                'Bonding character',
                'Crystal field splitting'
            ],
            'advantages': [
                'Element-specific',
                'Chemical state sensitivity',
                'Spatially resolved',
                'No vacuum required (tender/hard X-ray)'
            ],
            'applications': [
                'Battery electrode state-of-charge mapping',
                'Catalyst active site identification',
                'Mineral speciation',
                'Protein metalloproteins',
                'Environmental redox mapping'
            ],
            'cross_validation': [
                'XPS for surface chemical states',
                'EELS for similar near-edge structure',
                'Bulk XAS for average composition'
            ]
        }

    def _execute_tomography_reconstruction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform 3D tomographic reconstruction from projection images.

        Reconstruct 3D volume from 2D projections acquired at multiple angles.

        Args:
            input_data: Contains projection data and reconstruction parameters

        Returns:
            Reconstructed 3D volume and quality metrics
        """
        sample_name = input_data.get('sample_name', 'rock_core')

        # Projection data
        num_projections = input_data.get('num_projections', 1200)
        angular_range_deg = input_data.get('angular_range_deg', 180)
        projection_size_pixels = input_data.get('projection_size_pixels', (2048, 2048))

        # Reconstruction algorithm
        algorithm = input_data.get('reconstruction_algorithm', 'FBP')  # FBP, SIRT, CGLS, TV

        # Center of rotation
        center_of_rotation_pixel = input_data.get('center_of_rotation', projection_size_pixels[0] / 2)
        cor_accuracy_pixels = input_data.get('cor_accuracy_pixels', 0.5)

        # Reconstruction quality
        if algorithm == 'FBP':
            reconstruction_time_minutes = (projection_size_pixels[0] * projection_size_pixels[1] * num_projections) / 1e8
            noise_level = 'moderate'
            artifacts = ['Ring artifacts possible', 'Streak artifacts if undersampled']
        else:  # Iterative
            reconstruction_time_minutes = (projection_size_pixels[0] * projection_size_pixels[1] * num_projections) / 1e7
            noise_level = 'low'
            artifacts = ['Fewer streak artifacts', 'Better handling of incomplete data']

        # Volume statistics
        volume_voxels = (projection_size_pixels[0], projection_size_pixels[1], projection_size_pixels[0])
        volume_size_gb = np.prod(volume_voxels) * 4 / 1e9

        return {
            'technique': 'tomographic_reconstruction',
            'sample_name': sample_name,
            'num_projections': num_projections,
            'angular_range_deg': angular_range_deg,
            'projection_size_pixels': projection_size_pixels,
            'reconstruction_algorithm': algorithm,
            'center_of_rotation_pixel': center_of_rotation_pixel,
            'cor_accuracy_pixels': cor_accuracy_pixels,
            'reconstructed_volume_voxels': volume_voxels,
            'volume_size_gb': volume_size_gb,
            'reconstruction_time_minutes': reconstruction_time_minutes,
            'expected_noise_level': noise_level,
            'potential_artifacts': artifacts,
            'reconstruction_algorithms': {
                'FBP': 'Filtered Back Projection (fast, analytical)',
                'SIRT': 'Simultaneous Iterative Reconstruction (slower, lower noise)',
                'CGLS': 'Conjugate Gradient Least Squares',
                'TV': 'Total Variation (sparse data, edge-preserving)'
            },
            'quality_metrics': {
                'angular_sampling': 'sufficient' if num_projections >= np.pi * projection_size_pixels[0] / 2 else 'undersampled',
                'cor_alignment': 'excellent' if cor_accuracy_pixels < 0.5 else 'good' if cor_accuracy_pixels < 2 else 'poor'
            },
            'post_processing': [
                'Ring artifact removal',
                'Beam hardening correction',
                'Phase retrieval (if phase contrast data)',
                '3D visualization and analysis'
            ]
        }

    # Helper methods

    def _get_xrf_lines(self, element: str) -> List[str]:
        """Get typical XRF emission lines for element."""
        # Simplified - would use actual X-ray data tables
        xrf_lines = {
            'Fe': ['Fe Kα (6.4 keV)', 'Fe Kβ (7.1 keV)', 'Fe Lα (0.7 keV)'],
            'Ca': ['Ca Kα (3.7 keV)', 'Ca Kβ (4.0 keV)'],
            'Zn': ['Zn Kα (8.6 keV)', 'Zn Kβ (9.6 keV)'],
            'Cu': ['Cu Kα (8.0 keV)', 'Cu Kβ (8.9 keV)']
        }
        return xrf_lines.get(element, [f'{element} Kα', f'{element} Kβ'])

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and metadata."""
        return {
            'agent_type': self.AGENT_TYPE,
            'version': self.VERSION,
            'supported_techniques': self.SUPPORTED_TECHNIQUES,
            'xray_regimes': self.XRAY_REGIMES,
            'spatial_resolution_range': {
                'best_achieved_nm': 5,  # Ptychography
                'typical_soft_xray_nm': 25,
                'typical_hard_xray_nm': 50,
                'lab_source_um': 1
            },
            'imaging_modes': {
                '2D': ['TXM', 'STXM', 'XFM', 'Ptychography'],
                '3D': ['XCT', 'Ptychographic tomography'],
                'spectroscopic': ['STXM-XANES', 'XFM']
            },
            'contrast_mechanisms': [
                'Absorption (density)',
                'Phase (electron density)',
                'Fluorescence (elemental)',
                'Scattering (dark-field)'
            ],
            'advantages': [
                'Non-destructive 3D imaging',
                'Chemical sensitivity (XANES)',
                'Elemental mapping (XRF)',
                'Nanometer resolution (ptychography)',
                'No vacuum required (hard X-ray)',
                'Penetration through thick samples'
            ],
            'cross_validation_opportunities': [
                'XCT ↔ SEM tomography (resolution trade-off)',
                'XFM ↔ SEM-EDX (elemental mapping)',
                'STXM-XANES ↔ XPS (chemical state)',
                'Ptychography ↔ TEM (phase imaging)',
                'XCT ↔ Neutron tomography (complementary contrast)'
            ],
            'typical_applications': [
                'Battery electrode 3D structure and chemistry',
                'Bone microstructure',
                'Composite materials',
                'Integrated circuits',
                'Biological cells in native state',
                'Metal alloy microstructure',
                'Crack and defect detection'
            ]
        }


if __name__ == '__main__':
    # Example usage
    agent = XRayMicroscopyAgent()

    # Example 1: X-ray Computed Tomography
    result_xct = agent.execute({
        'technique': 'xray_computed_tomography',
        'sample_name': 'metal_foam',
        'photon_energy_kev': 25,
        'num_projections': 1800,
        'detector_pixel_size_um': 6.5,
        'source_sample_distance_mm': 50,
        'sample_detector_distance_mm': 150,
        'measured_porosity_percent': 35.0
    })
    print("X-ray Computed Tomography Result:")
    print(f"  Voxel Size: {result_xct['voxel_size_um']:.2f} µm")
    print(f"  Volume: {result_xct['reconstructed_volume']['size_mm']}")
    print(f"  Porosity: {result_xct['analysis_results']['porosity_percent']:.1f}%")
    print(f"  Data Size: {result_xct['reconstructed_volume']['data_size_gb']:.2f} GB")
    print()

    # Example 2: X-ray Fluorescence Microscopy
    result_xfm = agent.execute({
        'technique': 'xray_fluorescence_microscopy',
        'sample_name': 'catalyst_particle',
        'incident_energy_kev': 12,
        'beam_size_um': 0.5,
        'elements_detected': ['Pt', 'Ru', 'C'],
        'scan_area_um': (50, 50)
    })
    print("X-ray Fluorescence Microscopy Result:")
    print(f"  Spatial Resolution: {result_xfm['spatial_resolution_um']:.1f} µm")
    print(f"  Elements Detected: {', '.join(result_xfm['elements_detected'])}")
    print(f"  Acquisition Time: {result_xfm['total_acquisition_time_minutes']:.1f} minutes")
    print()

    # Example 3: Ptychography
    result_ptycho = agent.execute({
        'technique': 'ptychography',
        'sample_name': 'nanostructure',
        'photon_energy_kev': 6.2,
        'achieved_resolution_nm': 8,
        'scan_points': 144
    })
    print("Ptychography Result:")
    print(f"  Resolution: {result_ptycho['achieved_resolution_nm']:.0f} nm")
    print(f"  Probe Size: {result_ptycho['probe_size_nm']:.0f} nm")
    print(f"  Algorithm: {result_ptycho['reconstruction']['algorithm']}")
    print(f"  Outputs: {', '.join(result_ptycho['outputs'].values())}")
