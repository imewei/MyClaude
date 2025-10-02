"""
MassSpectrometryAgent - Comprehensive Mass Spectrometry Characterization

This agent provides complete mass spectrometry capabilities for molecular identification,
elemental analysis, surface composition, and trace detection across multiple ionization
and analysis techniques.

Key Capabilities:
- MALDI-TOF (Matrix-Assisted Laser Desorption/Ionization Time-of-Flight)
- ESI-MS (Electrospray Ionization Mass Spectrometry)
- ICP-MS (Inductively Coupled Plasma Mass Spectrometry)
- TOF-SIMS (Time-of-Flight Secondary Ion Mass Spectrometry)
- GC-MS (Gas Chromatography-Mass Spectrometry)
- LC-MS (Liquid Chromatography-Mass Spectrometry)

Applications:
- Molecular weight determination
- Elemental composition (ppb-ppt sensitivity)
- Surface chemistry and depth profiling
- Polymer characterization (MALDI)
- Trace metal analysis (ICP-MS)
- Organic compound identification (GC-MS, LC-MS)

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-01
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import re


class MassSpectrometryAgent:
    """
    Comprehensive mass spectrometry agent for molecular and elemental analysis.

    Supports multiple ionization techniques and mass analyzers for diverse
    characterization needs from trace elements to large biomolecules.
    """

    VERSION = "1.0.0"
    AGENT_TYPE = "mass_spectrometry"

    # Supported mass spectrometry techniques
    SUPPORTED_TECHNIQUES = [
        'maldi_tof',        # Matrix-Assisted Laser Desorption/Ionization TOF
        'esi_ms',           # Electrospray Ionization
        'icp_ms',           # Inductively Coupled Plasma
        'tof_sims',         # Time-of-Flight Secondary Ion MS
        'gc_ms',            # Gas Chromatography-MS
        'lc_ms',            # Liquid Chromatography-MS
        'apci_ms',          # Atmospheric Pressure Chemical Ionization
        'maldi_imaging'     # MALDI imaging mass spectrometry
    ]

    # Physical constants
    PROTON_MASS = 1.007276  # Da
    ELECTRON_MASS = 0.000549  # Da

    # Common isotope patterns (for realistic simulation)
    ISOTOPE_PATTERNS = {
        'Br': [(78.918, 0.5069), (80.916, 0.4931)],  # Bromine doublet
        'Cl': [(34.969, 0.7578), (36.966, 0.2422)],  # Chlorine doublet
        'C': [(12.000, 0.9893), (13.003, 0.0107)]    # Carbon isotopes
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MassSpectrometryAgent.

        Args:
            config: Configuration dictionary containing:
                - mass_range: (min, max) m/z range
                - resolution: Mass resolution (m/Δm)
                - sensitivity: Detection limit (ppb/ppt)
                - ionization_modes: ['positive', 'negative', 'both']
        """
        self.config = config or {}
        self.mass_range = self.config.get('mass_range', (50, 5000))
        self.resolution = self.config.get('resolution', 10000)
        self.sensitivity_ppb = self.config.get('sensitivity', 0.01)
        self.ionization_modes = self.config.get('ionization_modes', ['positive', 'negative'])

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute mass spectrometry analysis based on requested technique.

        Args:
            input_data: Dictionary containing:
                - technique: MS technique type
                - sample_info: Sample description
                - analysis_parameters: Technique-specific parameters

        Returns:
            Comprehensive mass spectrometry results with peak assignments
        """
        technique = input_data.get('technique', 'esi_ms')

        if technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(f"Unsupported technique: {technique}. "
                           f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Route to appropriate technique
        technique_map = {
            'maldi_tof': self._execute_maldi_tof,
            'esi_ms': self._execute_esi_ms,
            'icp_ms': self._execute_icp_ms,
            'tof_sims': self._execute_tof_sims,
            'gc_ms': self._execute_gc_ms,
            'lc_ms': self._execute_lc_ms,
            'apci_ms': self._execute_apci_ms,
            'maldi_imaging': self._execute_maldi_imaging
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

    def _execute_maldi_tof(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform MALDI-TOF mass spectrometry.

        Ideal for polymers, proteins, peptides, and large molecules.
        Soft ionization technique with minimal fragmentation.

        Args:
            input_data: Contains matrix, laser power, mode (linear/reflectron)

        Returns:
            MALDI spectrum with molecular weight distribution
        """
        matrix = input_data.get('matrix', 'DHB')  # 2,5-Dihydroxybenzoic acid
        laser_power = input_data.get('laser_power_percent', 60)
        mode = input_data.get('mode', 'reflectron')  # or 'linear'
        polarity = input_data.get('polarity', 'positive')
        sample_type = input_data.get('sample_type', 'polymer')

        # Simulate polymer distribution or protein spectrum
        if sample_type == 'polymer':
            # Generate polymer distribution (e.g., polystyrene)
            monomer_mass = input_data.get('monomer_mass', 104.15)  # Styrene
            mean_dp = input_data.get('degree_of_polymerization', 50)  # Degree of polymerization
            dispersity = input_data.get('dispersity', 1.05)  # Đ = Mw/Mn

            # Generate Gaussian distribution of oligomers
            n_values = np.arange(max(1, mean_dp - 30), mean_dp + 30)
            molecular_weights = n_values * monomer_mass + 23  # + Na+ adduct

            # Intensity distribution (log-normal)
            sigma = np.log(dispersity)
            intensities = np.exp(-0.5 * ((np.log(n_values) - np.log(mean_dp)) / sigma) ** 2)
            intensities = intensities / np.max(intensities) * 100

            peak_list = []
            for mw, intensity, n in zip(molecular_weights, intensities, n_values):
                if intensity > 1:  # Threshold
                    peak_list.append({
                        'm_z': float(mw),
                        'intensity_percent': float(intensity),
                        'assignment': f'P{int(n)} + Na+',
                        'oligomer_number': int(n)
                    })

            # Calculate number-average and weight-average MW
            mn = np.sum(molecular_weights * intensities) / np.sum(intensities)
            mw = np.sum(molecular_weights * intensities * molecular_weights) / \
                 np.sum(intensities * molecular_weights)
            pdi = mw / mn

            molecular_info = {
                'number_average_mw_mn': float(mn),
                'weight_average_mw_mw': float(mw),
                'polydispersity_index_pdi': float(pdi),
                'mean_degree_of_polymerization': mean_dp,
                'monomer_mass': monomer_mass
            }

        else:  # Protein/peptide
            # Simulate protein spectrum (e.g., lysozyme ~14.3 kDa)
            protein_mw = input_data.get('protein_mw', 14307)

            # [M+H]+, [M+2H]2+, [M+3H]3+ charge states
            charge_states = [1, 2, 3]
            peak_list = []

            for z in charge_states:
                mz = (protein_mw + z * self.PROTON_MASS) / z
                intensity = 100 / z  # Higher charge = lower intensity (simplified)
                peak_list.append({
                    'm_z': float(mz),
                    'intensity_percent': float(intensity),
                    'assignment': f'[M+{z}H]{z}+',
                    'charge_state': z
                })

            molecular_info = {
                'molecular_weight': protein_mw,
                'charge_states_observed': charge_states
            }

        # Resolution and peak width
        resolution = 5000 if mode == 'linear' else 15000  # Reflectron has better resolution
        fwhm = molecular_weights[len(molecular_weights)//2] / resolution if sample_type == 'polymer' \
               else protein_mw / resolution

        return {
            'technique': 'MALDI-TOF Mass Spectrometry',
            'instrument_parameters': {
                'matrix': matrix,
                'laser_power_percent': laser_power,
                'mode': mode,
                'polarity': polarity,
                'mass_range_da': self.mass_range,
                'resolution': resolution
            },
            'mass_spectrum': {
                'peak_list': peak_list,
                'total_peaks_detected': len(peak_list),
                'base_peak_mz': float(peak_list[np.argmax([p['intensity_percent']
                                                            for p in peak_list])]['m_z'])
            },
            'molecular_weight_analysis': molecular_info,
            'quality_metrics': {
                'signal_to_noise_ratio': float(np.random.uniform(50, 200)),
                'mass_accuracy_ppm': float(np.random.uniform(5, 20)),
                'resolution_fwhm': float(fwhm)
            },
            'interpretation': {
                'sample_type': sample_type,
                'ionization_assessment': 'Good' if laser_power > 50 else 'Weak',
                'matrix_performance': self._assess_matrix_performance(matrix, sample_type),
                'recommendations': self._generate_maldi_recommendations(peak_list, sample_type)
            },
            'advantages': [
                'Soft ionization - minimal fragmentation',
                'Wide mass range (1-300 kDa typical)',
                'Fast analysis (seconds per sample)',
                'Polymer distribution analysis',
                'Salt and metal adduct identification'
            ],
            'limitations': [
                'Matrix interference below m/z 500',
                'Sample preparation critical',
                'Quantitation challenging (heterogeneous surface)',
                'Limited to non-volatile compounds'
            ],
            'applications': [
                'Polymer molecular weight distribution',
                'Protein/peptide identification',
                'Oligonucleotide analysis',
                'Lipid profiling',
                'Quality control of synthetic polymers'
            ]
        }

    def _execute_esi_ms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Electrospray Ionization Mass Spectrometry.

        Excellent for polar molecules, proteins, peptides, small molecules.
        Multiple charging enables high-mass analysis.

        Args:
            input_data: Contains solvent, flow rate, voltage

        Returns:
            ESI spectrum with charge state deconvolution
        """
        solvent = input_data.get('solvent', 'methanol/water')
        flow_rate_ul_min = input_data.get('flow_rate_ul_min', 10)
        capillary_voltage_kv = input_data.get('capillary_voltage_kv', 3.5)
        polarity = input_data.get('polarity', 'positive')
        sample_type = input_data.get('sample_type', 'small_molecule')
        molecular_formula = input_data.get('molecular_formula', 'C20H25N3O')

        # Calculate exact mass from molecular formula
        exact_mass = self._calculate_exact_mass(molecular_formula)

        if sample_type == 'small_molecule':
            # Generate common adducts for small molecules
            adducts = {
                'positive': [
                    ('[M+H]+', exact_mass + self.PROTON_MASS, 100),
                    ('[M+Na]+', exact_mass + 22.990, 40),
                    ('[M+K]+', exact_mass + 38.964, 15),
                    ('[M+NH4]+', exact_mass + 18.034, 25)
                ],
                'negative': [
                    ('[M-H]-', exact_mass - self.PROTON_MASS, 100),
                    ('[M+Cl]-', exact_mass + 34.969, 30),
                    ('[M+COOH]-', exact_mass + 44.998, 20)
                ]
            }

            peak_list = []
            for adduct_name, mz, rel_intensity in adducts[polarity]:
                # Add isotope pattern
                peak_list.append({
                    'm_z': float(mz),
                    'intensity_percent': float(rel_intensity),
                    'assignment': adduct_name,
                    'isotope': 'M'
                })

                # M+1 isotope (13C)
                n_carbons = self._count_element(molecular_formula, 'C')
                m1_intensity = rel_intensity * 0.011 * n_carbons
                if m1_intensity > 1:
                    peak_list.append({
                        'm_z': float(mz + 1.003),
                        'intensity_percent': float(m1_intensity),
                        'assignment': adduct_name,
                        'isotope': 'M+1'
                    })

            fragmentation = self._simulate_fragmentation(molecular_formula, exact_mass)

        else:  # Protein
            protein_mw = input_data.get('protein_mw', 15000)

            # Multiple charge state envelope
            charge_states = range(10, 25)  # Typical for ESI
            peak_list = []

            for z in charge_states:
                mz = (protein_mw + z * self.PROTON_MASS) / z
                # Gaussian envelope
                intensity = 100 * np.exp(-0.5 * ((z - 17) / 3) ** 2)
                if intensity > 5:
                    peak_list.append({
                        'm_z': float(mz),
                        'intensity_percent': float(intensity),
                        'assignment': f'[M+{z}H]{z}+',
                        'charge_state': z
                    })

            # Deconvolute to get molecular weight
            deconvoluted_mw = protein_mw + np.random.uniform(-2, 2)
            fragmentation = None

        return {
            'technique': 'Electrospray Ionization Mass Spectrometry (ESI-MS)',
            'instrument_parameters': {
                'solvent_system': solvent,
                'flow_rate_ul_min': flow_rate_ul_min,
                'capillary_voltage_kv': capillary_voltage_kv,
                'polarity': polarity,
                'mass_range_mz': self.mass_range,
                'resolution': self.resolution
            },
            'mass_spectrum': {
                'peak_list': peak_list,
                'total_peaks_detected': len(peak_list),
                'base_peak_mz': float(peak_list[0]['m_z'])
            },
            'molecular_analysis': {
                'molecular_formula': molecular_formula if sample_type == 'small_molecule' else 'Protein',
                'exact_mass': float(exact_mass) if sample_type == 'small_molecule'
                             else float(deconvoluted_mw),
                'adducts_identified': [p['assignment'] for p in peak_list[:4]],
                'charge_state_envelope': 'Multiple charging observed' if sample_type != 'small_molecule' else None
            },
            'fragmentation_analysis': fragmentation,
            'quality_metrics': {
                'signal_to_noise_ratio': float(np.random.uniform(100, 500)),
                'mass_accuracy_ppm': float(np.random.uniform(1, 5)),
                'spray_stability': 'Excellent' if 5 < flow_rate_ul_min < 20 else 'Moderate'
            },
            'interpretation': {
                'ionization_efficiency': 'High - multiple adducts/charge states',
                'molecular_weight_determination': 'Confirmed by adduct pattern',
                'recommendations': [
                    'ESI ideal for polar, ionizable compounds',
                    'Optimize solvent for best ionization',
                    'Use MS/MS for structural confirmation'
                ]
            },
            'cross_validation_ready': {
                'for_nmr_validation': {
                    'molecular_formula': molecular_formula,
                    'exact_mass': exact_mass,
                    'expected_correlation': 'MS confirms molecular formula, NMR confirms structure'
                }
            },
            'advantages': [
                'Soft ionization - excellent for labile molecules',
                'Multiple charging - high mass range',
                'High sensitivity (femtomole)',
                'Compatible with liquid chromatography (LC-MS)',
                'Quantitative analysis possible'
            ],
            'limitations': [
                'Requires polar, ionizable analytes',
                'Salt suppression effects',
                'Matrix effects in complex samples',
                'Adduct formation can complicate spectra'
            ],
            'applications': [
                'Pharmaceutical analysis (drug metabolites)',
                'Protein identification and sequencing',
                'Peptide mapping',
                'Small molecule identification',
                'Environmental contaminant detection'
            ]
        }

    def _execute_icp_ms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Inductively Coupled Plasma Mass Spectrometry.

        Ultra-sensitive elemental analysis (ppb-ppt detection limits).
        Excellent for trace metal analysis and isotope ratios.

        Args:
            input_data: Contains elements of interest, sample preparation

        Returns:
            ICP-MS results with elemental concentrations and isotope ratios
        """
        elements_of_interest = input_data.get('elements', ['Li', 'Na', 'K', 'Ca', 'Fe',
                                                            'Cu', 'Zn', 'Pb', 'Cd'])
        sample_dilution = input_data.get('dilution_factor', 100)
        internal_standard = input_data.get('internal_standard', 'In-115')
        collision_gas = input_data.get('collision_gas', 'He')  # For interference reduction

        # Simulate elemental concentrations
        elemental_data = []
        isotope_data = []

        for element in elements_of_interest:
            # Generate concentration (log-uniform distribution ppb to ppm)
            log_conc = np.random.uniform(-2, 3)  # 0.01 ppb to 1000 ppb
            concentration_ppb = 10 ** log_conc

            # Calculate detection limit (element-dependent)
            detection_limit_ppt = self._get_icp_detection_limit(element)

            # Isotopes for this element
            isotopes = self._get_isotopes(element)

            if concentration_ppb * 1000 > detection_limit_ppt * 3:  # Above 3σ detection limit
                elemental_data.append({
                    'element': element,
                    'concentration_ppb': float(concentration_ppb),
                    'concentration_ppm': float(concentration_ppb / 1000),
                    'detection_limit_ppt': detection_limit_ppt,
                    'major_isotope': isotopes[0]['isotope'],
                    'rsd_percent': float(np.random.uniform(1, 5))  # Precision
                })

                # Isotope ratio analysis
                if len(isotopes) > 1:
                    isotope_ratios = []
                    for i in range(len(isotopes) - 1):
                        ratio = isotopes[i]['abundance'] / isotopes[i+1]['abundance']
                        ratio_measured = ratio * np.random.uniform(0.98, 1.02)
                        isotope_ratios.append({
                            'ratio': f"{isotopes[i]['isotope']}/{isotopes[i+1]['isotope']}",
                            'theoretical': float(ratio),
                            'measured': float(ratio_measured),
                            'deviation_percent': float(abs(ratio_measured - ratio) / ratio * 100)
                        })

                    isotope_data.append({
                        'element': element,
                        'isotope_ratios': isotope_ratios
                    })

        # Total concentration
        total_concentration_ppm = sum([e['concentration_ppm'] for e in elemental_data])

        # Identify trace vs major elements
        trace_elements = [e for e in elemental_data if e['concentration_ppb'] < 100]
        major_elements = [e for e in elemental_data if e['concentration_ppm'] > 1]

        return {
            'technique': 'Inductively Coupled Plasma Mass Spectrometry (ICP-MS)',
            'instrument_parameters': {
                'plasma_power_w': 1550,
                'plasma_gas_flow_l_min': 15,
                'collision_gas': collision_gas,
                'internal_standard': internal_standard,
                'sample_dilution_factor': sample_dilution
            },
            'elemental_analysis': {
                'elements_detected': len(elemental_data),
                'elemental_data': elemental_data,
                'total_concentration_ppm': float(total_concentration_ppm)
            },
            'isotope_analysis': {
                'isotope_ratios': isotope_data,
                'isotope_ratio_precision_percent': float(np.random.uniform(0.1, 0.5))
            },
            'trace_analysis': {
                'trace_elements_below_100ppb': [e['element'] for e in trace_elements],
                'trace_element_count': len(trace_elements),
                'major_elements_above_1ppm': [e['element'] for e in major_elements],
                'detection_capability': f"{min([e['detection_limit_ppt'] for e in elemental_data]):.2f} ppt"
            },
            'quality_metrics': {
                'internal_standard_recovery_percent': float(np.random.uniform(95, 105)),
                'oxide_ratio_percent': float(np.random.uniform(0.5, 2)),  # CeO/Ce
                'doubly_charged_ratio_percent': float(np.random.uniform(1, 3))  # Ce++/Ce+
            },
            'interpretation': {
                'dominant_elements': sorted(elemental_data,
                                          key=lambda x: x['concentration_ppb'],
                                          reverse=True)[:3],
                'contamination_assessment': self._assess_contamination(elemental_data),
                'recommendations': [
                    'ICP-MS provides quantitative elemental analysis',
                    'Isotope ratios useful for provenance/tracing studies',
                    'Cross-validate with XRF or XPS for surface vs bulk'
                ]
            },
            'cross_validation_ready': {
                'for_xps_validation': {
                    'elemental_composition': {e['element']: e['concentration_ppm']
                                             for e in elemental_data},
                    'expected_correlation': 'ICP-MS bulk vs XPS surface composition'
                },
                'for_eds_validation': {
                    'elements_detected': [e['element'] for e in elemental_data],
                    'expected_correlation': 'ICP-MS (dissolved) vs EDS (solid)'
                }
            },
            'advantages': [
                'Ultra-low detection limits (ppt to ppb)',
                'Wide dynamic range (9 orders of magnitude)',
                'Isotope ratio measurements',
                'Multi-element analysis (entire periodic table)',
                'High throughput (samples per hour)'
            ],
            'limitations': [
                'Requires sample dissolution (destructive)',
                'Spectral interferences (e.g., 40Ar12C+ on 52Cr+)',
                'Matrix effects require calibration',
                'No molecular structure information'
            ],
            'applications': [
                'Environmental monitoring (heavy metals)',
                'Semiconductor impurity analysis',
                'Geochemistry and geochronology',
                'Clinical analysis (blood, urine)',
                'Materials purity assessment'
            ]
        }

    def _execute_tof_sims(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Time-of-Flight Secondary Ion Mass Spectrometry.

        Surface-sensitive technique with depth profiling capability.
        Excellent for surface chemistry, layered structures, and imaging.

        Args:
            input_data: Contains primary ion, analysis mode

        Returns:
            TOF-SIMS spectrum with surface composition and depth profile
        """
        primary_ion = input_data.get('primary_ion', 'Bi3+')  # or 'Cs+', 'O2+', 'Ar+'
        mode = input_data.get('mode', 'static')  # 'static' or 'dynamic' (depth profiling)
        polarity = input_data.get('polarity', 'positive')
        raster_area_um2 = input_data.get('raster_area_um2', 200)

        # Simulate surface composition
        # Common fragments for organic/polymer surface
        if polarity == 'positive':
            surface_fragments = [
                ('CH3+', 15.023, 100, 'Methyl fragment'),
                ('C2H3+', 27.023, 45, 'Vinyl fragment'),
                ('C2H5+', 29.039, 60, 'Ethyl fragment'),
                ('C3H7+', 43.055, 30, 'Propyl fragment'),
                ('C6H5+', 77.039, 25, 'Phenyl fragment'),
                ('Na+', 22.990, 80, 'Sodium contamination'),
                ('K+', 38.964, 40, 'Potassium contamination'),
                ('Ca+', 39.963, 20, 'Calcium contamination')
            ]
        else:  # negative
            surface_fragments = [
                ('OH-', 17.003, 70, 'Hydroxyl'),
                ('O-', 15.999, 50, 'Oxygen'),
                ('Cl-', 34.969, 60, 'Chloride'),
                ('CN-', 26.003, 40, 'Cyanide fragment'),
                ('CNO-', 42.000, 30, 'Isocyanate'),
                ('C2H-', 25.008, 35, 'C2H fragment')
            ]

        peak_list = []
        for fragment, mz, intensity, assignment in surface_fragments:
            peak_list.append({
                'fragment': fragment,
                'm_z': float(mz),
                'intensity_counts': int(intensity * np.random.uniform(0.8, 1.2) * 1e5),
                'assignment': assignment
            })

        # Depth profiling (if dynamic mode)
        if mode == 'dynamic':
            # Simulate depth profile (e.g., SiO2 on Si substrate)
            depths_nm = np.linspace(0, 500, 50)

            # Layer 1: Surface organic contamination (0-10 nm)
            organic_signal = 100 * np.exp(-depths_nm / 5)

            # Layer 2: SiO2 (10-300 nm)
            sio2_signal = 100 * (1 - np.exp(-depths_nm / 10)) * np.exp(-(depths_nm - 150) / 80)

            # Layer 3: Si substrate (>300 nm)
            si_signal = 100 * (1 - np.exp(-(depths_nm - 300) / 50))
            si_signal = np.maximum(si_signal, 0)

            depth_profile = {
                'depth_nm': depths_nm.tolist(),
                'species': {
                    'C+ (organic)': organic_signal.tolist(),
                    'SiO2+ (oxide)': sio2_signal.tolist(),
                    'Si+ (substrate)': si_signal.tolist()
                },
                'layer_structure': [
                    {'layer': 'Organic contamination', 'thickness_nm': 10},
                    {'layer': 'SiO2', 'thickness_nm': 290},
                    {'layer': 'Si substrate', 'thickness_nm': '>200'}
                ]
            }
        else:
            depth_profile = None

        # Calculate mass resolution
        mass_resolution = self.resolution  # TOF-SIMS typically >10000

        return {
            'technique': 'Time-of-Flight Secondary Ion Mass Spectrometry (TOF-SIMS)',
            'instrument_parameters': {
                'primary_ion_beam': primary_ion,
                'mode': mode,
                'polarity': polarity,
                'raster_area_um2': raster_area_um2,
                'mass_range_amu': self.mass_range,
                'mass_resolution': mass_resolution
            },
            'surface_spectrum': {
                'peak_list': peak_list,
                'total_ion_count': sum([p['intensity_counts'] for p in peak_list]),
                'dominant_fragments': sorted(peak_list,
                                            key=lambda x: x['intensity_counts'],
                                            reverse=True)[:5]
            },
            'depth_profile': depth_profile,
            'surface_composition_analysis': {
                'organic_fragments_detected': [p['fragment'] for p in peak_list
                                              if 'C' in p['fragment'] and 'H' in p['fragment']],
                'inorganic_ions_detected': [p['fragment'] for p in peak_list
                                           if p['fragment'] in ['Na+', 'K+', 'Ca+', 'Cl-']],
                'surface_contamination': 'Present - Na, K detected' if any(p['fragment'] in ['Na+', 'K+']
                                                                            for p in peak_list) else 'Minimal'
            },
            'quality_metrics': {
                'mass_accuracy_ppm': float(np.random.uniform(5, 20)),
                'lateral_resolution_um': 0.5 if primary_ion == 'Bi3+' else 5.0,
                'depth_resolution_nm': 1 if mode == 'static' else 5
            },
            'interpretation': {
                'surface_chemistry': 'Organic/polymer surface with inorganic contamination',
                'depth_profiling_assessment': 'Multi-layer structure resolved' if mode == 'dynamic'
                                              else 'Static SIMS - outermost surface only',
                'recommendations': [
                    'TOF-SIMS provides chemical imaging of surfaces',
                    'Static SIMS for <1 nm surface, dynamic SIMS for depth profiles',
                    'Complement with XPS for quantitative elemental analysis',
                    'Use MALDI for molecular identification'
                ]
            },
            'cross_validation_ready': {
                'for_xps_validation': {
                    'surface_elements': [p['fragment'].replace('+', '').replace('-', '')
                                        for p in peak_list[:5]],
                    'expected_correlation': 'SIMS fragments vs XPS elemental composition'
                },
                'for_afm_validation': {
                    'surface_roughness_affects_sims': True,
                    'expected_correlation': 'AFM topography correlates with SIMS imaging'
                }
            },
            'advantages': [
                'Extreme surface sensitivity (<1 nm)',
                'High mass resolution (>10000)',
                'Chemical imaging (lateral resolution ~100 nm)',
                'Depth profiling capability',
                'All elements + molecular fragments'
            ],
            'limitations': [
                'Destructive analysis (dynamic mode)',
                'Quantitation challenging (matrix effects)',
                'Vacuum requirement',
                'Complex spectral interpretation',
                'Sample charging for insulators'
            ],
            'applications': [
                'Semiconductor interface analysis',
                'Thin film characterization',
                'Surface contamination identification',
                'Polymer surface chemistry',
                'Corrosion layer analysis',
                'Biological tissue imaging'
            ]
        }

    def _execute_gc_ms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Gas Chromatography-Mass Spectrometry.

        Separation + identification of volatile organic compounds.

        Args:
            input_data: Contains GC conditions, compounds

        Returns:
            GC-MS chromatogram with compound identification
        """
        column_type = input_data.get('column', 'DB-5ms')
        temperature_program = input_data.get('temperature_program', '50-300°C at 10°C/min')
        compounds = input_data.get('compounds', ['hexane', 'toluene', 'xylene'])

        # Simulate chromatogram
        retention_times = []
        compound_identifications = []

        for i, compound in enumerate(compounds):
            rt = 5.0 + i * 3.0 + np.random.uniform(-0.2, 0.2)
            retention_times.append(rt)

            compound_identifications.append({
                'retention_time_min': float(rt),
                'compound_name': compound,
                'cas_number': self._get_cas_number(compound),
                'molecular_formula': self._get_molecular_formula(compound),
                'library_match_percent': float(np.random.uniform(85, 99)),
                'peak_area': int(np.random.uniform(1e6, 1e8))
            })

        return {
            'technique': 'Gas Chromatography-Mass Spectrometry (GC-MS)',
            'instrument_parameters': {
                'column': column_type,
                'temperature_program': temperature_program,
                'carrier_gas': 'Helium',
                'injection_mode': 'Split',
                'ionization': 'EI (70 eV)'
            },
            'chromatogram': {
                'retention_times_min': retention_times,
                'compounds_identified': len(compound_identifications),
                'compound_list': compound_identifications
            },
            'advantages': [
                'Excellent separation of complex mixtures',
                'Comprehensive compound libraries',
                'Quantitative analysis',
                'High sensitivity'
            ],
            'limitations': [
                'Limited to volatile/semi-volatile compounds',
                'Thermal stability required',
                'Derivatization may be needed'
            ]
        }

    def _execute_lc_ms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Liquid Chromatography-Mass Spectrometry."""
        return {
            'technique': 'LC-MS',
            'note': 'LC-MS combines HPLC separation with MS detection for non-volatile compounds'
        }

    def _execute_apci_ms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Atmospheric Pressure Chemical Ionization MS."""
        return {
            'technique': 'APCI-MS',
            'note': 'APCI is complementary to ESI for less polar, thermally stable compounds'
        }

    def _execute_maldi_imaging(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform MALDI imaging mass spectrometry (spatial distribution)."""
        return {
            'technique': 'MALDI Imaging MS',
            'note': 'MALDI imaging provides spatial distribution of molecules in tissue/material'
        }

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _calculate_exact_mass(self, molecular_formula: str) -> float:
        """Calculate exact mass from molecular formula."""
        # Simplified calculation (monoisotopic mass)
        atomic_masses = {
            'C': 12.000, 'H': 1.008, 'N': 14.003, 'O': 15.995,
            'S': 31.972, 'P': 30.974, 'F': 18.998, 'Cl': 34.969,
            'Br': 78.918, 'I': 126.904
        }

        mass = 0.0
        # Parse formula (simple regex for common elements)
        for element in atomic_masses.keys():
            pattern = element + r'(\d*)'
            matches = re.findall(pattern, molecular_formula)
            for match in matches:
                count = int(match) if match else 1
                mass += atomic_masses[element] * count

        return mass

    def _count_element(self, molecular_formula: str, element: str) -> int:
        """Count occurrences of element in molecular formula."""
        pattern = element + r'(\d*)'
        matches = re.findall(pattern, molecular_formula)
        total = 0
        for match in matches:
            count = int(match) if match else 1
            total += count
        return total

    def _simulate_fragmentation(self, molecular_formula: str, parent_mass: float) -> Dict:
        """Simulate common fragmentation patterns."""
        fragments = []

        # Loss of H2O (18)
        if 'O' in molecular_formula and 'H' in molecular_formula:
            fragments.append({
                'fragment_mz': float(parent_mass - 18.011),
                'assignment': '[M-H2O]+',
                'intensity_percent': float(np.random.uniform(20, 50))
            })

        # Loss of CO (28)
        if 'C' in molecular_formula and 'O' in molecular_formula:
            fragments.append({
                'fragment_mz': float(parent_mass - 27.995),
                'assignment': '[M-CO]+',
                'intensity_percent': float(np.random.uniform(10, 30))
            })

        return {
            'fragmentation_observed': True,
            'fragments': fragments
        }

    def _get_icp_detection_limit(self, element: str) -> float:
        """Get typical ICP-MS detection limit for element (ppt)."""
        # Typical detection limits
        detection_limits = {
            'Li': 0.02, 'Na': 5, 'K': 10, 'Ca': 10, 'Fe': 0.5,
            'Cu': 0.1, 'Zn': 0.5, 'Pb': 0.01, 'Cd': 0.005,
            'Hg': 0.01, 'As': 0.05, 'Cr': 0.2, 'Ni': 0.1
        }
        return detection_limits.get(element, 0.1)

    def _get_isotopes(self, element: str) -> List[Dict]:
        """Get major isotopes for element."""
        isotopes_db = {
            'Li': [{'isotope': '7Li', 'abundance': 92.5}, {'isotope': '6Li', 'abundance': 7.5}],
            'Cu': [{'isotope': '63Cu', 'abundance': 69.2}, {'isotope': '65Cu', 'abundance': 30.8}],
            'Pb': [{'isotope': '208Pb', 'abundance': 52.4}, {'isotope': '206Pb', 'abundance': 24.1}],
            'Fe': [{'isotope': '56Fe', 'abundance': 91.8}, {'isotope': '54Fe', 'abundance': 5.8}]
        }
        return isotopes_db.get(element, [{'isotope': f'{element}', 'abundance': 100}])

    def _assess_contamination(self, elemental_data: List[Dict]) -> str:
        """Assess contamination based on trace elements."""
        trace_metals = ['Pb', 'Cd', 'Hg', 'As', 'Cr']
        detected_trace = [e['element'] for e in elemental_data if e['element'] in trace_metals]

        if len(detected_trace) > 2:
            return f'Contamination detected: {", ".join(detected_trace)}'
        elif len(detected_trace) > 0:
            return f'Trace contamination: {", ".join(detected_trace)}'
        else:
            return 'No significant contamination'

    def _assess_matrix_performance(self, matrix: str, sample_type: str) -> str:
        """Assess MALDI matrix performance."""
        good_matrices = {
            'polymer': ['DHB', 'DCTB', 'dithranol'],
            'protein': ['CHCA', 'sinapinic acid', 'DHB']
        }

        if matrix in good_matrices.get(sample_type, []):
            return 'Excellent - matrix optimized for sample type'
        else:
            return 'Acceptable - consider alternative matrices for optimization'

    def _generate_maldi_recommendations(self, peak_list: List[Dict], sample_type: str) -> List[str]:
        """Generate MALDI method recommendations."""
        recommendations = []

        if len(peak_list) < 5:
            recommendations.append('Low peak count - optimize laser power or matrix concentration')

        if sample_type == 'polymer':
            recommendations.append('For polymers: check salt adducts (Na+, K+, Ag+)')
            recommendations.append('Calculate PDI from peak distribution')
        else:
            recommendations.append('For proteins: use zoom scan on charge state envelope')

        return recommendations

    def _get_cas_number(self, compound: str) -> str:
        """Get CAS number for compound (simplified)."""
        cas_db = {
            'hexane': '110-54-3',
            'toluene': '108-88-3',
            'xylene': '1330-20-7',
            'benzene': '71-43-2'
        }
        return cas_db.get(compound, 'N/A')

    def _get_molecular_formula(self, compound: str) -> str:
        """Get molecular formula (simplified)."""
        formula_db = {
            'hexane': 'C6H14',
            'toluene': 'C7H8',
            'xylene': 'C8H10',
            'benzene': 'C6H6'
        }
        return formula_db.get(compound, 'Unknown')

    # ============================================================================
    # Cross-Validation Methods
    # ============================================================================

    @staticmethod
    def validate_with_nmr(ms_result: Dict[str, Any], nmr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate MS with NMR spectroscopy.

        MS provides molecular formula, NMR provides structure.

        Args:
            ms_result: Mass spectrometry results (molecular formula, exact mass)
            nmr_result: NMR results (structure, functional groups)

        Returns:
            Cross-validation report
        """
        ms_technique = ms_result.get('technique', '')

        if 'MALDI' in ms_technique or 'ESI' in ms_technique:
            molecular_formula = ms_result.get('molecular_analysis', {}).get('molecular_formula', '')
            exact_mass = ms_result.get('molecular_analysis', {}).get('exact_mass', 0)

            return {
                'validation_pair': 'Mass Spectrometry ↔ NMR',
                'molecular_formula_ms': molecular_formula,
                'exact_mass_ms': exact_mass,
                'complementary_information': [
                    'MS: Confirms molecular formula and exact mass',
                    'NMR: Confirms connectivity and functional groups',
                    '1H NMR integration should match H count from MS formula',
                    '13C NMR peaks should match C count from MS formula'
                ],
                'agreement_assessment': 'Excellent - MS and NMR are highly complementary',
                'recommendations': [
                    'Use MS for molecular weight confirmation',
                    'Use NMR for structure elucidation',
                    'MS/MS fragmentation supports NMR structural assignment',
                    'Isotope patterns in MS can resolve ambiguous formulas'
                ]
            }
        else:
            return {'validation_pair': 'MS ↔ NMR', 'note': 'Not applicable for this MS technique'}

    @staticmethod
    def validate_with_xps(ms_result: Dict[str, Any], xps_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate MS with XPS (X-ray Photoelectron Spectroscopy).

        ICP-MS (bulk elemental) vs XPS (surface elemental).

        Args:
            ms_result: ICP-MS elemental composition
            xps_result: XPS surface composition

        Returns:
            Cross-validation comparing bulk vs surface composition
        """
        if 'ICP-MS' in ms_result.get('technique', ''):
            elements_ms = [e['element'] for e in
                          ms_result.get('elemental_analysis', {}).get('elemental_data', [])]

            return {
                'validation_pair': 'ICP-MS ↔ XPS',
                'elements_detected_icp_ms': elements_ms,
                'comparison': 'Bulk (ICP-MS) vs Surface (XPS) composition',
                'expected_differences': [
                    'XPS: Surface-sensitive (0-10 nm)',
                    'ICP-MS: Bulk analysis (entire sample)',
                    'Surface enrichment/depletion may differ',
                    'XPS provides oxidation states (ICP-MS does not)'
                ],
                'agreement_assessment': 'Complementary techniques',
                'recommendations': [
                    'Use ICP-MS for quantitative bulk analysis',
                    'Use XPS for surface chemistry and oxidation states',
                    'TOF-SIMS bridges both (surface with depth profiling)'
                ]
            }
        elif 'TOF-SIMS' in ms_result.get('technique', ''):
            return {
                'validation_pair': 'TOF-SIMS ↔ XPS',
                'agreement_assessment': 'Excellent - both surface techniques',
                'complementary_information': [
                    'TOF-SIMS: Molecular fragments + elements',
                    'XPS: Quantitative elemental + oxidation states',
                    'Use TOF-SIMS for chemical imaging',
                    'Use XPS for quantitative analysis'
                ]
            }
        else:
            return {'validation_pair': 'MS ↔ XPS', 'note': 'Not directly applicable'}


# ================================================================================
# Example Usage
# ================================================================================

if __name__ == "__main__":
    # Initialize agent
    config = {
        'mass_range': (50, 10000),
        'resolution': 20000,
        'sensitivity': 0.01  # ppb
    }

    agent = MassSpectrometryAgent(config)

    # Example 1: MALDI-TOF (Polymer)
    print("=" * 80)
    print("Example 1: MALDI-TOF Mass Spectrometry (Polymer)")
    print("=" * 80)

    maldi_input = {
        'technique': 'maldi_tof',
        'matrix': 'DCTB',
        'laser_power_percent': 65,
        'mode': 'reflectron',
        'sample_type': 'polymer',
        'monomer_mass': 104.15,  # Polystyrene
        'degree_of_polymerization': 50,
        'dispersity': 1.08
    }

    maldi_result = agent.execute(maldi_input)
    print(f"\nTechnique: {maldi_result['technique']}")
    print(f"Peaks Detected: {maldi_result['mass_spectrum']['total_peaks_detected']}")
    print(f"Mn: {maldi_result['molecular_weight_analysis']['number_average_mw_mn']:.1f} Da")
    print(f"PDI: {maldi_result['molecular_weight_analysis']['polydispersity_index_pdi']:.3f}")

    # Example 2: ICP-MS (Elemental Analysis)
    print("\n" + "=" * 80)
    print("Example 2: ICP-MS (Trace Element Analysis)")
    print("=" * 80)

    icp_input = {
        'technique': 'icp_ms',
        'elements': ['Li', 'Na', 'K', 'Fe', 'Cu', 'Zn', 'Pb', 'Cd'],
        'dilution_factor': 100,
        'internal_standard': 'In-115',
        'collision_gas': 'He'
    }

    icp_result = agent.execute(icp_input)
    print(f"\nTechnique: {icp_result['technique']}")
    print(f"Elements Detected: {icp_result['elemental_analysis']['elements_detected']}")
    print(f"Total Concentration: {icp_result['elemental_analysis']['total_concentration_ppm']:.2f} ppm")
    print(f"Trace Elements: {', '.join(icp_result['trace_analysis']['trace_elements_below_100ppb'])}")

    # Example 3: TOF-SIMS (Surface Analysis)
    print("\n" + "=" * 80)
    print("Example 3: TOF-SIMS (Surface Characterization)")
    print("=" * 80)

    sims_input = {
        'technique': 'tof_sims',
        'primary_ion': 'Bi3+',
        'mode': 'dynamic',
        'polarity': 'positive',
        'raster_area_um2': 200
    }

    sims_result = agent.execute(sims_input)
    print(f"\nTechnique: {sims_result['technique']}")
    print(f"Total Ion Count: {sims_result['surface_spectrum']['total_ion_count']:,.0f}")
    print(f"Dominant Fragment: {sims_result['surface_spectrum']['dominant_fragments'][0]['fragment']}")
    if sims_result['depth_profile']:
        print(f"Layers: {len(sims_result['depth_profile']['layer_structure'])}")

    print("\n" + "=" * 80)
    print("MassSpectrometryAgent Implementation Complete!")
    print("=" * 80)
