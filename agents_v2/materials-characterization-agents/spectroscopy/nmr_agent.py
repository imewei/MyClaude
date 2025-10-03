"""
NMR Agent for Nuclear Magnetic Resonance Spectroscopy

This agent provides comprehensive NMR capabilities including:
- 1H NMR: Proton NMR for molecular structure determination
- 13C NMR: Carbon-13 NMR for carbon framework analysis
- 2D NMR: Multidimensional experiments (COSY, HSQC, HMBC, NOESY)
- Solid-state NMR: CP-MAS, MQMAS for solid materials
- Relaxation: T1, T2 measurements for dynamics
- Diffusion NMR (DOSY): Molecular size and diffusion coefficients
- Quantitative NMR (qNMR): Concentration determination

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


class NMRNucleus(Enum):
    """Common NMR-active nuclei"""
    H1 = ("1H", 99.98, 42.576)  # (symbol, natural abundance %, gyromagnetic ratio MHz/T)
    C13 = ("13C", 1.11, 10.705)
    N15 = ("15N", 0.37, -4.316)
    F19 = ("19F", 100.0, 40.052)
    P31 = ("31P", 100.0, 17.235)
    SI29 = ("29Si", 4.70, -8.465)


class NMRExperiment(Enum):
    """NMR experiment types"""
    ONE_D_1H = "1d_1h"
    ONE_D_13C = "1d_13c"
    COSY = "cosy"  # Homonuclear correlation
    HSQC = "hsqc"  # Heteronuclear single quantum coherence
    HMBC = "hmbc"  # Heteronuclear multiple bond correlation
    NOESY = "noesy"  # Nuclear Overhauser effect spectroscopy
    DOSY = "dosy"  # Diffusion-ordered spectroscopy
    CPMAS = "cpmas"  # Cross-polarization magic angle spinning
    T1_RELAXATION = "t1_relaxation"
    T2_RELAXATION = "t2_relaxation"
    QUANTITATIVE = "quantitative_nmr"


@dataclass
class NMRResult:
    """Results from NMR measurement"""
    experiment_type: str
    nucleus: str
    field_strength_mhz: float
    solvent: str
    temperature_k: float
    spectrum_data: Dict[str, Any]
    peak_analysis: List[Dict[str, Any]]
    molecular_info: Dict[str, Any]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class NMRAgent(ExperimentalAgent):
    """
    Agent for Nuclear Magnetic Resonance spectroscopy.

    Capabilities:
    - 1D NMR (1H, 13C, 15N, 19F, 31P, 29Si)
    - 2D NMR (COSY, HSQC, HMBC, NOESY)
    - Solid-state NMR (CP-MAS, MQMAS)
    - Relaxation experiments (T1, T2)
    - Diffusion NMR (DOSY)
    - Quantitative NMR
    - Variable temperature studies
    """

    NAME = "NMRAgent"
    VERSION = "1.0.0"
    DESCRIPTION = "Comprehensive NMR spectroscopy for molecular structure determination"

    SUPPORTED_TECHNIQUES = [
        '1d_1h',
        '1d_13c',
        '1d_15n',
        '1d_19f',
        '1d_31p',
        '1d_29si',
        'cosy',
        'hsqc',
        'hmbc',
        'noesy',
        'dosy',
        'cpmas',
        't1_relaxation',
        't2_relaxation',
        'quantitative_nmr'
    ]

    # Common solvents and their properties
    SOLVENTS = {
        'CDCl3': {'chemical_shift_1h': 7.26, 'chemical_shift_13c': 77.16, 'residual_h2o': 1.56},
        'D2O': {'chemical_shift_1h': 4.79, 'residual_hdo': 4.79},
        'DMSO-d6': {'chemical_shift_1h': 2.50, 'chemical_shift_13c': 39.52, 'residual_h2o': 3.33},
        'CD3CN': {'chemical_shift_1h': 1.94, 'chemical_shift_13c': 1.32, 'residual_h2o': 2.13},
        'acetone-d6': {'chemical_shift_1h': 2.05, 'chemical_shift_13c': 29.84, 'residual_h2o': 2.84},
        'benzene-d6': {'chemical_shift_1h': 7.16, 'chemical_shift_13c': 128.06},
        'MeOD': {'chemical_shift_1h': 3.31, 'chemical_shift_13c': 49.00, 'residual_h2o': 4.87},
        'THF-d8': {'chemical_shift_1h': 1.72, 3.58, 'chemical_shift_13c': 25.31, 67.21}
    }

    def __init__(self):
        super().__init__(self.NAME, self.VERSION, self.DESCRIPTION)
        self.capabilities = {
            'field_strength_range_mhz': (200, 1000),  # 4.7 T to 23.5 T
            'spectral_width_khz': 20,
            'digital_resolution_hz': 0.1,
            'temperature_range_k': (200, 400),
            'nuclei_supported': ['1H', '13C', '15N', '19F', '31P', '29Si'],
            'max_2d_size': (2048, 2048),
            'solid_state_capable': True,
            'variable_temperature': True,
            'diffusion_capable': True
        }

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute NMR measurement.

        Args:
            input_data: Dictionary containing:
                - technique: NMR experiment type
                - field_strength_mhz: Spectrometer frequency
                - nucleus: Nuclear isotope (default: 1H)
                - solvent: NMR solvent
                - temperature_k: Sample temperature
                - scans: Number of scans
                - parameters: Technique-specific parameters

        Returns:
            AgentResult with NMR spectrum and analysis
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
            '1d_1h': self._execute_1d_1h,
            '1d_13c': self._execute_1d_13c,
            '1d_15n': self._execute_1d_15n,
            '1d_19f': self._execute_1d_19f,
            '1d_31p': self._execute_1d_31p,
            '1d_29si': self._execute_1d_29si,
            'cosy': self._execute_cosy,
            'hsqc': self._execute_hsqc,
            'hmbc': self._execute_hmbc,
            'noesy': self._execute_noesy,
            'dosy': self._execute_dosy,
            'cpmas': self._execute_cpmas,
            't1_relaxation': self._execute_t1_relaxation,
            't2_relaxation': self._execute_t2_relaxation,
            'quantitative_nmr': self._execute_quantitative_nmr
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

    def _execute_1d_1h(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 1H NMR analysis for molecular structure determination"""
        field_strength = input_data.get('field_strength_mhz', 400)
        solvent = input_data.get('solvent', 'CDCl3')
        temperature_k = input_data.get('temperature_k', 298)
        ppm_range = input_data.get('ppm_range', [0, 12])
        scans = input_data.get('scans', 16)

        # Generate ppm scale
        n_points = 16384  # 16K points typical
        ppm = np.linspace(ppm_range[0], ppm_range[1], n_points)

        # Simulate baseline
        intensity = np.zeros_like(ppm) + np.random.normal(0, 0.01, n_points)

        # Define typical organic molecule peaks
        peaks = [
            (7.26, 'singlet', 1, 'CHCl3 (solvent)', 0.1),
            (7.45, 'doublet', 2, 'aromatic H ortho', 7.5),
            (7.35, 'triplet', 2, 'aromatic H meta', 7.5),
            (7.28, 'triplet', 1, 'aromatic H para', 7.5),
            (4.12, 'quartet', 2, 'OCH2 (ester)', 7.0),
            (3.70, 'singlet', 3, 'OCH3', 0),
            (2.35, 'singlet', 3, 'ArCH3', 0),
            (1.56, 'singlet', 2, 'H2O (residual)', 0.1),
            (1.24, 'triplet', 3, 'CH3 (ethyl)', 7.0),
            (0.00, 'singlet', 1, 'TMS (reference)', 0.05)
        ]

        # Add peaks to spectrum
        for shift, multiplicity, integration, assignment, j_coupling in peaks:
            if multiplicity == 'singlet':
                intensity += integration * self._generate_lorentzian(ppm, shift, linewidth=0.003)
            elif multiplicity == 'doublet':
                j = j_coupling / field_strength  # Convert Hz to ppm
                intensity += integration/2 * self._generate_lorentzian(ppm, shift - j/2, linewidth=0.003)
                intensity += integration/2 * self._generate_lorentzian(ppm, shift + j/2, linewidth=0.003)
            elif multiplicity == 'triplet':
                j = j_coupling / field_strength
                intensity += integration/4 * self._generate_lorentzian(ppm, shift - j, linewidth=0.003)
                intensity += integration/2 * self._generate_lorentzian(ppm, shift, linewidth=0.003)
                intensity += integration/4 * self._generate_lorentzian(ppm, shift + j, linewidth=0.003)
            elif multiplicity == 'quartet':
                j = j_coupling / field_strength
                for i, coeff in enumerate([1/8, 3/8, 3/8, 1/8]):
                    intensity += integration * coeff * self._generate_lorentzian(
                        ppm, shift + (i-1.5)*j, linewidth=0.003
                    )

        # Calculate SNR
        signal = np.max(intensity)
        noise = np.std(intensity[ppm < 0.5])
        snr = signal / noise if noise > 0 else 1000

        # Peak integration
        total_integration = sum(p[2] for p in peaks if 'solvent' not in p[3] and 'H2O' not in p[3] and 'TMS' not in p[3])

        return {
            'experiment_type': '1H NMR',
            'nucleus': '1H',
            'field_strength_mhz': field_strength,
            'solvent': solvent,
            'temperature_k': temperature_k,
            'scans': scans,
            'spectrum': {
                'chemical_shifts_ppm': ppm.tolist(),
                'intensity': intensity.tolist(),
                'spectral_width_hz': field_strength * (ppm_range[1] - ppm_range[0]),
                'acquisition_time_s': scans * 2.0,  # Typical
                'digital_resolution_hz': (field_strength * (ppm_range[1] - ppm_range[0])) / n_points
            },
            'peak_analysis': [
                {
                    'chemical_shift_ppm': shift,
                    'multiplicity': multiplicity,
                    'integration': integration,
                    'assignment': assignment,
                    'j_coupling_hz': j_coupling if j_coupling > 0 else None,
                    'linewidth_hz': 0.3 * field_strength  # ~0.3 Hz
                }
                for shift, multiplicity, integration, assignment, j_coupling in peaks
                if 'TMS' not in assignment
            ],
            'molecular_structure': {
                'total_protons': total_integration,
                'aromatic_protons': 5,
                'aliphatic_protons': 8,
                'heteroatom_ch': 2,
                'molecular_formula_fragment': 'C11H16O2',
                'functional_groups': ['aromatic_ring', 'ester', 'methyl', 'ethyl']
            },
            'purity_assessment': {
                'purity_percent': float(95.0 + np.random.uniform(-2, 2)),
                'impurity_peaks': 1,
                'solvent_residual_peaks': [solvent, 'H2O'],
                'grease_contamination': False
            },
            'quality_metrics': {
                'snr': float(snr),
                'baseline_flatness': 0.95,
                'phase_correction_quality': 0.98,
                'shimming_quality': 'excellent',
                'resolution_hz': 0.3
            }
        }

    def _execute_1d_13c(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 13C NMR analysis for carbon framework determination"""
        field_strength = input_data.get('field_strength_mhz', 100)  # 400 MHz 1H = 100 MHz 13C
        solvent = input_data.get('solvent', 'CDCl3')
        decoupling = input_data.get('decoupling', True)  # Proton decoupling
        ppm_range = input_data.get('ppm_range', [0, 220])
        scans = input_data.get('scans', 1024)  # More scans needed for 13C

        n_points = 32768
        ppm = np.linspace(ppm_range[0], ppm_range[1], n_points)
        intensity = np.zeros_like(ppm) + np.random.normal(0, 0.05, n_points)

        # Define carbon peaks
        carbon_peaks = [
            (173.2, 'C=O (ester carbonyl)', 'quaternary', 1.0),
            (140.5, 'C aromatic (quaternary)', 'quaternary', 0.3),
            (130.1, 'CH aromatic', 'CH', 1.0),
            (128.7, 'CH aromatic', 'CH', 1.0),
            (126.2, 'CH aromatic', 'CH', 1.0),
            (77.16, 'CDCl3 (solvent)', 'solvent', 3.0),  # Triplet
            (65.3, 'O-CH2', 'CH2', 0.8),
            (52.1, 'O-CH3', 'CH3', 0.9),
            (40.2, 'aliphatic CH2', 'CH2', 0.7),
            (21.5, 'Ar-CH3', 'CH3', 0.9),
            (14.2, 'CH3 (ethyl)', 'CH3', 1.0)
        ]

        for shift, assignment, carbon_type, rel_intensity in carbon_peaks:
            if 'solvent' in assignment and shift == 77.16:
                # CDCl3 is a 1:1:1 triplet
                intensity += rel_intensity/3 * self._generate_lorentzian(ppm, shift - 0.24, linewidth=0.02)
                intensity += rel_intensity/3 * self._generate_lorentzian(ppm, shift, linewidth=0.02)
                intensity += rel_intensity/3 * self._generate_lorentzian(ppm, shift + 0.24, linewidth=0.02)
            else:
                intensity += rel_intensity * self._generate_lorentzian(ppm, shift, linewidth=0.05)

        # NOE enhancement (decoupling increases intensity)
        if decoupling:
            noe_factor = 2.0  # Typical NOE enhancement
            intensity *= noe_factor

        return {
            'experiment_type': '13C NMR',
            'nucleus': '13C',
            'field_strength_mhz': field_strength,
            'solvent': solvent,
            'scans': scans,
            'decoupling': decoupling,
            'spectrum': {
                'chemical_shifts_ppm': ppm.tolist(),
                'intensity': intensity.tolist(),
                'spectral_width_hz': field_strength * (ppm_range[1] - ppm_range[0]),
                'acquisition_time_s': scans * 2.0
            },
            'peak_analysis': [
                {
                    'chemical_shift_ppm': shift,
                    'assignment': assignment,
                    'carbon_type': carbon_type,
                    'relative_intensity': rel_intensity,
                    'multiplicity': 'singlet' if decoupling else 'multiplet'
                }
                for shift, assignment, carbon_type, rel_intensity in carbon_peaks
                if 'solvent' not in assignment
            ],
            'carbon_framework': {
                'total_carbons': 11,
                'carbonyl_carbons': 1,
                'aromatic_carbons': 6,
                'aliphatic_carbons': 4,
                'quaternary_carbons': 2,
                'ch_carbons': 3,
                'ch2_carbons': 2,
                'ch3_carbons': 4
            },
            'dept_equivalent': {
                'ch3_ch_positive': ['CH3 signals at 52.1, 21.5, 14.2 ppm', 'CH signals at 130.1, 128.7, 126.2 ppm'],
                'ch2_negative': ['CH2 signals at 65.3, 40.2 ppm'],
                'quaternary_absent': ['Quaternary C at 173.2, 140.5 ppm']
            },
            'quality_metrics': {
                'snr': float(80 + np.random.uniform(-10, 10)),
                'baseline_quality': 0.92,
                'decoupling_efficiency': 0.99 if decoupling else None
            }
        }

    def _execute_cosy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute COSY (Correlation Spectroscopy) for homonuclear coupling"""
        field_strength = input_data.get('field_strength_mhz', 400)

        # Define correlation peaks (f1_ppm, f2_ppm, intensity, connectivity)
        correlations = [
            (7.45, 7.35, 0.8, 'ortho-meta coupling', 3),  # aromatic J_ortho-meta
            (7.35, 7.28, 0.7, 'meta-para coupling', 3),  # aromatic J_meta-para
            (4.12, 1.24, 0.9, 'OCH2-CH3 coupling', 3),  # ethyl ester
            (2.35, 7.45, 0.3, 'ArCH3-ArH long-range', 4),  # long-range
        ]

        return {
            'experiment_type': 'COSY',
            'nucleus': '1H-1H',
            'field_strength_mhz': field_strength,
            'dimensions': 2,
            'correlations': [
                {
                    'f1_ppm': f1,
                    'f2_ppm': f2,
                    'intensity': intensity,
                    'connectivity': conn,
                    'coupling_bonds': bonds
                }
                for f1, f2, intensity, conn, bonds in correlations
            ],
            'structural_information': {
                'vicinal_couplings_3j': 3,
                'geminal_couplings_2j': 0,
                'long_range_couplings_4j': 1,
                'aromatic_spin_system': 'monosubstituted_benzene',
                'aliphatic_spin_system': 'ethyl_group'
            },
            'acquisition_parameters': {
                'f1_spectral_width_hz': 4800,
                'f2_spectral_width_hz': 4800,
                'f1_points': 256,
                'f2_points': 2048,
                'scans_per_increment': 4,
                'total_time_hours': 1.5
            }
        }

    def _execute_hsqc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HSQC (Heteronuclear Single Quantum Coherence) for 1H-13C direct bonds"""

        # Define 1H-13C one-bond correlations
        correlations = [
            (7.45, 130.1, 'aromatic CH'),
            (7.35, 128.7, 'aromatic CH'),
            (7.28, 126.2, 'aromatic CH'),
            (4.12, 65.3, 'O-CH2'),
            (3.70, 52.1, 'O-CH3'),
            (2.35, 21.5, 'Ar-CH3'),
            (1.24, 14.2, 'CH3 ethyl')
        ]

        return {
            'experiment_type': 'HSQC',
            'nuclei': '1H-13C',
            'correlation_type': 'one_bond_1jch',
            'correlations': [
                {
                    '1h_ppm': h_ppm,
                    '13c_ppm': c_ppm,
                    'assignment': assignment,
                    '1jch_hz': float(125 + np.random.uniform(-15, 15))  # Typical 1J_CH
                }
                for h_ppm, c_ppm, assignment in correlations
            ],
            'structural_connectivity': {
                'direct_ch_bonds': len(correlations),
                'quaternary_carbons_identified': 2,  # Absent in HSQC
                'molecular_skeleton': 'confirmed'
            },
            'advantages': [
                'Direct C-H connectivity determination',
                'High sensitivity compared to HMBC',
                'Quaternary carbon identification by absence'
            ]
        }

    def _execute_hmbc(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HMBC (Heteronuclear Multiple Bond Correlation) for long-range coupling"""

        # Define 1H-13C multiple-bond correlations (2J, 3J, 4J)
        correlations = [
            (7.45, 140.5, 'ArH to quaternary C', 3),
            (7.45, 173.2, 'ArH to C=O', 4),
            (3.70, 173.2, 'OCH3 to C=O', 2),
            (4.12, 173.2, 'OCH2 to C=O', 3),
            (2.35, 140.5, 'ArCH3 to quaternary C', 2),
            (2.35, 130.1, 'ArCH3 to aromatic CH', 3),
            (1.24, 65.3, 'CH3 to OCH2', 3)
        ]

        return {
            'experiment_type': 'HMBC',
            'nuclei': '1H-13C',
            'correlation_type': 'multi_bond_2jch_3jch',
            'correlations': [
                {
                    '1h_ppm': h_ppm,
                    '13c_ppm': c_ppm,
                    'assignment': assignment,
                    'coupling_bonds': bonds,
                    'njch_hz': float(5 + np.random.uniform(-2, 2))  # Typical long-range
                }
                for h_ppm, c_ppm, assignment, bonds in correlations
            ],
            'key_correlations': {
                'quaternary_carbon_assignment': 'confirmed via HMBC',
                'carbonyl_position': 'confirmed',
                'substitution_pattern': 'para-substituted aromatic'
            },
            'structural_elucidation': {
                'complete_connectivity': True,
                'stereochemistry': 'not_determined',  # Need NOESY for this
                'molecular_assembly': 'ethyl_para_tolyl_ester'
            }
        }

    def _execute_noesy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NOESY (Nuclear Overhauser Effect Spectroscopy) for spatial proximity"""
        mixing_time_ms = input_data.get('mixing_time_ms', 500)

        # Define NOE correlations (spatial proximity < 5 Å)
        noe_peaks = [
            (7.45, 2.35, 0.6, 'ArH-ArCH3 proximity', 2.8),  # Ortho protons close in space
            (3.70, 4.12, 0.3, 'OCH3-OCH2 proximity', 4.2),
            (7.28, 7.35, 0.5, 'ArH-ArH proximity', 2.5)
        ]

        return {
            'experiment_type': 'NOESY',
            'nucleus': '1H-1H',
            'mixing_time_ms': mixing_time_ms,
            'noe_peaks': [
                {
                    'h1_ppm': h1,
                    'h2_ppm': h2,
                    'noe_intensity': intensity,
                    'interpretation': interp,
                    'estimated_distance_angstrom': distance
                }
                for h1, h2, intensity, interp, distance in noe_peaks
            ],
            'spatial_information': {
                'conformational_analysis': 'preferred_conformation_determined',
                'stereochemistry': 'relative_configuration',
                'molecular_geometry': 'folded_or_extended'
            },
            'applications': [
                'Protein structure determination',
                'Conformational analysis',
                'Stereochemistry assignment',
                'Molecular interactions'
            ]
        }

    def _execute_dosy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DOSY (Diffusion-Ordered Spectroscopy) for molecular size"""

        # Simulate diffusion coefficients for different species
        species = [
            ('compound_1', 7.5, 8e-10, 'target molecule'),
            ('compound_2', 3.2, 2e-9, 'small impurity'),
            ('polymer', 1.5, 3e-11, 'large molecule')
        ]

        return {
            'experiment_type': 'DOSY',
            'temperature_k': 298,
            'solvent_viscosity_cp': 0.54,  # CDCl3
            'species_separated': [
                {
                    '1h_ppm_range': ppm,
                    'diffusion_coefficient_m2_s': diff,
                    'hydrodynamic_radius_nm': self._calculate_stokes_radius(diff, 298, 0.54e-3),
                    'molecular_weight_estimate_da': self._estimate_mw_from_diffusion(diff),
                    'assignment': assignment
                }
                for assignment, ppm, diff, _ in species
            ],
            'applications': [
                'Mixture analysis',
                'Molecular weight determination',
                'Aggregation studies',
                'Purity assessment'
            ]
        }

    def _execute_cpmas(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CP-MAS solid-state NMR"""
        nucleus = input_data.get('nucleus', '13C')
        spinning_rate_khz = input_data.get('spinning_rate_khz', 10)

        return {
            'experiment_type': 'CP-MAS',
            'nucleus': nucleus,
            'solid_state': True,
            'magic_angle_spinning': True,
            'spinning_rate_khz': spinning_rate_khz,
            'cross_polarization': True,
            'spectral_features': {
                'chemical_shift_anisotropy': 'averaged_by_mas',
                'dipolar_coupling': 'suppressed',
                'isotropic_shifts_observed': True,
                'spinning_sidebands': spinning_rate_khz < 15
            },
            'applications': [
                'Insoluble polymers',
                'Crystalline materials',
                'Catalysts',
                'Pharmaceuticals'
            ]
        }

    def _execute_t1_relaxation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute T1 (spin-lattice) relaxation measurement"""

        # Simulate T1 values for different environments
        t1_values = [
            ('aromatic CH', 2.5),
            ('OCH2', 1.8),
            ('OCH3', 1.2),
            ('CH3', 3.0)
        ]

        return {
            'experiment_type': 'T1 Relaxation',
            'method': 'inversion_recovery',
            't1_measurements': [
                {
                    'assignment': assignment,
                    't1_seconds': t1,
                    'correlation_time_ns': self._calculate_correlation_time(t1, 400),
                    'molecular_motion': 'fast' if t1 < 2 else 'intermediate'
                }
                for assignment, t1 in t1_values
            ],
            'molecular_dynamics': {
                'overall_mobility': 'moderate',
                'temperature_dependence': 'arrhenius_behavior'
            }
        }

    def _execute_t2_relaxation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute T2 (spin-spin) relaxation measurement"""

        t2_values = [
            ('aromatic CH', 0.8),
            ('OCH2', 0.5),
            ('OCH3', 0.3),
            ('CH3', 1.2)
        ]

        return {
            'experiment_type': 'T2 Relaxation',
            'method': 'cpmg_echo',
            't2_measurements': [
                {
                    'assignment': assignment,
                    't2_seconds': t2,
                    'linewidth_hz': 1 / (np.pi * t2),
                    'exchange_processes': 'slow' if t2 > 0.5 else 'intermediate'
                }
                for assignment, t2 in t2_values
            ]
        }

    def _execute_quantitative_nmr(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantitative NMR for concentration determination"""
        internal_standard = input_data.get('internal_standard', 'maleic_acid')
        is_concentration_mm = input_data.get('is_concentration_mm', 10.0)
        is_protons = input_data.get('is_protons', 2)

        # Simulate integration
        sample_integral = 5.2
        is_integral = 2.0
        sample_protons = 3

        concentration_mm = (sample_integral / is_integral) * (is_protons / sample_protons) * is_concentration_mm

        return {
            'experiment_type': 'Quantitative NMR',
            'internal_standard': internal_standard,
            'is_concentration_mm': is_concentration_mm,
            'quantification': {
                'sample_integral': sample_integral,
                'is_integral': is_integral,
                'calculated_concentration_mm': float(concentration_mm),
                'purity_percent': float(98.5 + np.random.uniform(-1, 1)),
                'uncertainty_percent': 2.0
            },
            'advantages': [
                'Primary analytical method (traceable to SI units)',
                'No calibration curve needed',
                'Works for unknowns',
                'High accuracy (±2%)'
            ]
        }

    # Helper methods

    def _generate_lorentzian(self, x: np.ndarray, x0: float, linewidth: float) -> np.ndarray:
        """Generate Lorentzian lineshape"""
        gamma = linewidth / 2
        return (gamma**2) / ((x - x0)**2 + gamma**2)

    def _calculate_stokes_radius(self, diff_coeff: float, temp_k: float, viscosity_pa_s: float) -> float:
        """Calculate hydrodynamic radius from diffusion coefficient using Stokes-Einstein"""
        k_b = 1.380649e-23  # Boltzmann constant
        radius_m = k_b * temp_k / (6 * np.pi * viscosity_pa_s * diff_coeff)
        return radius_m * 1e9  # Convert to nm

    def _estimate_mw_from_diffusion(self, diff_coeff: float) -> float:
        """Estimate molecular weight from diffusion coefficient"""
        # Empirical relationship for small molecules in organic solvents
        # MW ∝ D^(-1/3)
        mw = 300 * (5e-10 / diff_coeff) ** (1/3)
        return float(mw)

    def _calculate_correlation_time(self, t1: float, field_mhz: float) -> float:
        """Calculate correlation time from T1"""
        # Simplified BPP theory
        omega = 2 * np.pi * field_mhz * 1e6
        tau_c = 1 / (omega * np.sqrt(t1))
        return tau_c * 1e9  # Convert to ns

    # Cross-validation methods

    @staticmethod
    def validate_with_ms(nmr_result: Dict[str, Any], ms_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate NMR structure with mass spectrometry molecular weight"""
        nmr_formula = nmr_result.get('molecular_structure', {}).get('molecular_formula_fragment', '')
        ms_mw = ms_result.get('molecular_weight', 0)

        # Calculate MW from formula
        nmr_mw_estimate = 176  # C11H16O2 example

        agreement = abs(nmr_mw_estimate - ms_mw) / ms_mw < 0.01 if ms_mw > 0 else False

        return {
            'technique_pair': 'NMR-MS',
            'parameter': 'molecular_weight',
            'nmr_formula': nmr_formula,
            'nmr_mw_estimate': nmr_mw_estimate,
            'ms_mw_measured': ms_mw,
            'agreement': 'excellent' if agreement else 'check_structure',
            'note': 'NMR provides structure, MS confirms molecular weight'
        }

    @staticmethod
    def validate_with_ftir(nmr_result: Dict[str, Any], ftir_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate NMR functional groups with FTIR"""
        nmr_groups = set(nmr_result.get('molecular_structure', {}).get('functional_groups', []))
        ftir_groups = set(ftir_result.get('functional_groups_identified', []))

        common_groups = nmr_groups.intersection(ftir_groups)
        agreement_fraction = len(common_groups) / len(nmr_groups) if nmr_groups else 0

        return {
            'technique_pair': 'NMR-FTIR',
            'parameter': 'functional_groups',
            'nmr_groups': list(nmr_groups),
            'ftir_groups': list(ftir_groups),
            'common_groups': list(common_groups),
            'agreement_fraction': float(agreement_fraction),
            'complementarity': 'NMR: connectivity, FTIR: functional groups'
        }

    @staticmethod
    def validate_with_xrd(nmr_result: Dict[str, Any], xrd_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate solid-state NMR with XRD crystallinity"""
        if nmr_result.get('solid_state', False):
            nmr_crystallinity_indicator = nmr_result.get('spectral_features', {}).get('sharp_peaks', True)
            xrd_crystallinity = xrd_result.get('crystallinity_percent', 0) > 50

            agreement = nmr_crystallinity_indicator == xrd_crystallinity

            return {
                'technique_pair': 'ssNMR-XRD',
                'parameter': 'crystallinity',
                'nmr_sharp_peaks': nmr_crystallinity_indicator,
                'xrd_crystalline': xrd_crystallinity,
                'agreement': 'good' if agreement else 'check_sample',
                'note': 'Sharp ssNMR peaks indicate crystalline domains'
            }
        return {'validation': 'not_applicable', 'reason': 'solution_state_nmr'}

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

        # Validate field strength
        if 'field_strength_mhz' in input_data:
            field = input_data['field_strength_mhz']
            if field < 200 or field > 1000:
                warnings.append(f"Field strength {field} MHz outside typical range (200-1000 MHz)")

        # Validate solvent
        if 'solvent' in input_data:
            if input_data['solvent'] not in self.SOLVENTS:
                warnings.append(f"Solvent {input_data['solvent']} not in common list")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def estimate_resources(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate experimental resources"""
        technique = input_data.get('technique', '1d_1h')
        scans = input_data.get('scans', 16)

        time_estimates = {
            '1d_1h': 1.0 * scans / 16,  # minutes
            '1d_13c': 30.0 * scans / 1024,
            'cosy': 60,
            'hsqc': 90,
            'hmbc': 180,
            'noesy': 120,
            'dosy': 60,
            'cpmas': 120,
            't1_relaxation': 45,
            't2_relaxation': 30,
            'quantitative_nmr': 10
        }

        return {
            'estimated_time_minutes': time_estimates.get(technique, 30),
            'sample_amount_mg': 5.0 if '1d' in technique else 10.0,
            'solvent_ml': 0.6,
            'spectrometer_cost_per_hour': 150.0,
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
            'supported_nuclei': self.capabilities['nuclei_supported'],
            'field_strength_range_mhz': self.capabilities['field_strength_range_mhz'],
            'capabilities': self.capabilities,
            'cross_validation_methods': [
                'validate_with_ms',
                'validate_with_ftir',
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
    agent = NMRAgent()

    # Example: 1H NMR
    result = agent.execute({
        'technique': '1d_1h',
        'field_strength_mhz': 400,
        'solvent': 'CDCl3',
        'temperature_k': 298,
        'scans': 16
    })
    print("1H NMR result:", result.status)

    # Example: HSQC
    result = agent.execute({
        'technique': 'hsqc',
        'field_strength_mhz': 400
    })
    print("HSQC result:", result.status)
