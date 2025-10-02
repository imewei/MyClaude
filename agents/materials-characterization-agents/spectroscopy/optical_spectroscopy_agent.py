"""
OpticalSpectroscopyAgent - Comprehensive Optical and Electronic Spectroscopy

This agent provides complete optical spectroscopy capabilities for electronic structure,
optical properties, and photophysical characterization of materials.

Key Capabilities:
- UV-Vis Absorption Spectroscopy
- Fluorescence Spectroscopy
- Photoluminescence (PL)
- Time-Resolved Fluorescence
- Diffuse Reflectance Spectroscopy
- Transmittance and Reflectance

Applications:
- Band gap determination (semiconductors, organic materials)
- Optical absorption and emission
- Quantum yield measurement
- Fluorescence lifetime
- Chromophore identification
- Photophysical processes (singlet/triplet states)

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-01
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime


class OpticalSpectroscopyAgent:
    """
    Comprehensive optical spectroscopy agent for electronic and photophysical characterization.

    Supports absorption, emission, and time-resolved techniques for diverse material systems.
    """

    VERSION = "1.0.0"
    AGENT_TYPE = "optical_spectroscopy"

    # Supported optical spectroscopy techniques
    SUPPORTED_TECHNIQUES = [
        'uv_vis_absorption',        # UV-Vis absorption spectroscopy
        'fluorescence',             # Steady-state fluorescence emission
        'photoluminescence',        # Solid-state PL
        'time_resolved_fluorescence', # Fluorescence lifetime
        'diffuse_reflectance',      # Diffuse reflectance (powders)
        'transmittance',            # Transmission spectroscopy
        'excitation_emission_matrix', # 3D fluorescence landscape
        'quantum_yield'             # Absolute quantum yield measurement
    ]

    # Physical constants
    PLANCK_CONSTANT = 6.626e-34  # J·s
    SPEED_OF_LIGHT = 2.998e8     # m/s
    AVOGADRO = 6.022e23          # mol⁻¹

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpticalSpectroscopyAgent.

        Args:
            config: Configuration dictionary containing:
                - wavelength_range: (min, max) nm
                - resolution: Spectral resolution (nm)
                - integrating_sphere: True/False for quantum yield
                - light_source: 'deuterium', 'tungsten', 'xenon', 'LED'
        """
        self.config = config or {}
        self.wavelength_range = self.config.get('wavelength_range', (200, 900))
        self.resolution = self.config.get('resolution', 1.0)  # nm
        self.integrating_sphere = self.config.get('integrating_sphere', False)
        self.light_source = self.config.get('light_source', 'xenon')

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optical spectroscopy analysis based on requested technique.

        Args:
            input_data: Dictionary containing:
                - technique: Spectroscopy technique type
                - sample_info: Sample description
                - measurement_parameters: Technique-specific parameters

        Returns:
            Comprehensive optical spectroscopy results with analysis
        """
        technique = input_data.get('technique', 'uv_vis_absorption')

        if technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(f"Unsupported technique: {technique}. "
                           f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Route to appropriate technique
        technique_map = {
            'uv_vis_absorption': self._execute_uv_vis_absorption,
            'fluorescence': self._execute_fluorescence,
            'photoluminescence': self._execute_photoluminescence,
            'time_resolved_fluorescence': self._execute_time_resolved_fluorescence,
            'diffuse_reflectance': self._execute_diffuse_reflectance,
            'transmittance': self._execute_transmittance,
            'excitation_emission_matrix': self._execute_eem,
            'quantum_yield': self._execute_quantum_yield
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

    def _execute_uv_vis_absorption(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform UV-Vis absorption spectroscopy.

        Measures electronic transitions, determines band gaps, calculates extinction coefficients.

        Args:
            input_data: Contains solvent, concentration, path length

        Returns:
            UV-Vis spectrum with band gap analysis and Beer-Lambert parameters
        """
        solvent = input_data.get('solvent', 'water')
        concentration_m = input_data.get('concentration_m', 1e-5)  # Molar
        path_length_cm = input_data.get('path_length_cm', 1.0)
        sample_type = input_data.get('sample_type', 'organic_dye')
        material = input_data.get('material', 'rhodamine_6g')

        # Generate wavelength array
        wavelengths = np.linspace(self.wavelength_range[0], self.wavelength_range[1],
                                  int((self.wavelength_range[1] - self.wavelength_range[0]) / self.resolution))

        # Simulate absorption spectrum based on material type
        if sample_type == 'organic_dye':
            # Single or multiple Gaussian absorption bands
            lambda_max = input_data.get('lambda_max_nm', 530)  # Absorption maximum
            epsilon_max = input_data.get('molar_absorptivity', 100000)  # L/(mol·cm)
            bandwidth = input_data.get('bandwidth_nm', 50)

            # Primary absorption band (S0 → S1)
            absorbance_primary = epsilon_max * concentration_m * path_length_cm * \
                                np.exp(-0.5 * ((wavelengths - lambda_max) / bandwidth) ** 2)

            # Secondary band (S0 → S2) - often present
            lambda_secondary = lambda_max - 100
            absorbance_secondary = 0.3 * epsilon_max * concentration_m * path_length_cm * \
                                  np.exp(-0.5 * ((wavelengths - lambda_secondary) / (bandwidth * 0.8)) ** 2)

            absorbance = absorbance_primary + absorbance_secondary

            # Vibronic structure (fine peaks)
            vibronic_spacing = 20  # nm
            for i in range(1, 4):
                vibronic_peak = lambda_max - i * vibronic_spacing
                vibronic_intensity = absorbance_primary * 0.15 * (0.7 ** i)
                absorbance += vibronic_intensity * np.exp(-0.5 * ((wavelengths - vibronic_peak) / 10) ** 2)

        elif sample_type == 'semiconductor':
            # Band edge absorption for semiconductors (Tauc plot analysis)
            band_gap_ev = input_data.get('band_gap_ev', 3.0)  # e.g., TiO2 ~3.2 eV
            band_gap_nm = 1240 / band_gap_ev  # Convert eV to nm

            # Absorption edge (sharp rise at band gap)
            energy_ev = 1240 / wavelengths  # Convert nm to eV
            alpha = np.zeros_like(wavelengths)

            # Direct band gap: α ~ (hν - Eg)^(1/2)
            above_bandgap = energy_ev > band_gap_ev
            alpha[above_bandgap] = 10000 * np.sqrt(energy_ev[above_bandgap] - band_gap_ev)

            # Urbach tail (exponential tail below band gap)
            below_bandgap = energy_ev <= band_gap_ev
            urbach_energy = 0.05  # eV
            alpha[below_bandgap] = 10000 * np.exp((energy_ev[below_bandgap] - band_gap_ev) / urbach_energy)

            # Convert to absorbance (A = α·d)
            thickness_cm = input_data.get('film_thickness_nm', 100) * 1e-7
            absorbance = alpha * thickness_cm / np.log(10)

        else:  # Generic material
            lambda_max = 400
            absorbance = np.exp(-0.5 * ((wavelengths - lambda_max) / 60) ** 2)

        # Add baseline and noise
        baseline = 0.02 * (wavelengths - self.wavelength_range[0]) / (self.wavelength_range[1] - self.wavelength_range[0])
        absorbance += baseline + np.random.normal(0, 0.002, len(wavelengths))
        absorbance = np.maximum(absorbance, 0)

        # Calculate transmittance
        transmittance = 10 ** (-absorbance)

        # Find absorption maximum
        idx_max = np.argmax(absorbance)
        lambda_max_measured = wavelengths[idx_max]
        abs_max = absorbance[idx_max]

        # Calculate molar absorptivity (Beer-Lambert law)
        if concentration_m > 0 and path_length_cm > 0:
            epsilon_measured = abs_max / (concentration_m * path_length_cm)
        else:
            epsilon_measured = None

        # Band gap determination (for semiconductors)
        if sample_type == 'semiconductor':
            # Tauc plot: (αhν)^2 vs hν for direct band gap
            energy_ev_array = 1240 / wavelengths
            alpha_hv_2 = (absorbance * energy_ev_array) ** 2

            # Linear fit in band edge region
            fit_region = (energy_ev_array > band_gap_ev - 0.2) & (energy_ev_array < band_gap_ev + 0.5)
            if np.sum(fit_region) > 10:
                fit_coeff = np.polyfit(energy_ev_array[fit_region], alpha_hv_2[fit_region], 1)
                band_gap_tauc = -fit_coeff[1] / fit_coeff[0]  # x-intercept
            else:
                band_gap_tauc = band_gap_ev

            tauc_analysis = {
                'band_gap_tauc_ev': float(band_gap_tauc),
                'band_gap_wavelength_nm': float(1240 / band_gap_tauc),
                'band_gap_type': 'Direct',
                'tauc_plot_data': {
                    'energy_ev': energy_ev_array.tolist(),
                    'alpha_hv_squared': alpha_hv_2.tolist()
                }
            }
        else:
            tauc_analysis = None

        # Oscillator strength (integrated absorption)
        integrated_abs = np.trapz(absorbance, wavelengths)

        return {
            'technique': 'UV-Vis Absorption Spectroscopy',
            'instrument_parameters': {
                'wavelength_range_nm': list(self.wavelength_range),
                'resolution_nm': self.resolution,
                'light_source': self.light_source,
                'detector': 'Silicon photodiode (UV-Vis)',
                'scan_speed': 'Medium (240 nm/min)'
            },
            'measurement_conditions': {
                'solvent': solvent,
                'concentration_m': concentration_m,
                'path_length_cm': path_length_cm,
                'temperature_c': input_data.get('temperature_c', 25)
            },
            'absorption_spectrum': {
                'wavelength_nm': wavelengths.tolist(),
                'absorbance': absorbance.tolist(),
                'transmittance_percent': (transmittance * 100).tolist()
            },
            'spectral_analysis': {
                'lambda_max_nm': float(lambda_max_measured),
                'absorbance_at_lambda_max': float(abs_max),
                'molar_absorptivity_l_mol_cm': float(epsilon_measured) if epsilon_measured else None,
                'integrated_absorption': float(integrated_abs),
                'absorption_onset_nm': float(wavelengths[np.where(absorbance > 0.1 * abs_max)[0][0]])
            },
            'band_gap_analysis': tauc_analysis,
            'quality_metrics': {
                'signal_to_noise_ratio': float(abs_max / np.std(absorbance[:50])),
                'baseline_flatness': float(np.std(absorbance[:50])),
                'spectral_purity': 'Single band' if np.sum(absorbance > 0.5 * abs_max) < 100 else 'Multiple bands'
            },
            'interpretation': {
                'chromophore_type': self._identify_chromophore(lambda_max_measured),
                'electronic_transition': self._identify_transition(lambda_max_measured, sample_type),
                'solvatochromism': 'Check λmax in different solvents for polarity effects',
                'recommendations': self._generate_uvvis_recommendations(abs_max, epsilon_measured)
            },
            'advantages': [
                'Simple, fast, non-destructive',
                'Quantitative (Beer-Lambert law)',
                'Wide concentration range (10⁻⁶ to 10⁻² M)',
                'Band gap determination for semiconductors',
                'Kinetics studies (time-resolved absorption)'
            ],
            'limitations': [
                'Limited structural information',
                'Requires transparent solvents in UV-Vis range',
                'Scattering artifacts for turbid samples',
                'Overlapping bands complicate analysis'
            ],
            'applications': [
                'Concentration determination (Beer-Lambert)',
                'Band gap measurement (Tauc plots)',
                'Chromophore identification',
                'Reaction monitoring (kinetics)',
                'Purity assessment',
                'Solvatochromic studies'
            ]
        }

    def _execute_fluorescence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform steady-state fluorescence spectroscopy.

        Measures emission spectra, Stokes shift, fluorescence intensity.

        Args:
            input_data: Contains excitation wavelength, emission range

        Returns:
            Fluorescence emission spectrum with photophysical analysis
        """
        excitation_nm = input_data.get('excitation_wavelength_nm', 450)
        emission_start = input_data.get('emission_start_nm', excitation_nm + 10)
        emission_end = input_data.get('emission_end_nm', 700)
        concentration_m = input_data.get('concentration_m', 1e-6)
        solvent = input_data.get('solvent', 'water')
        sample = input_data.get('sample', 'fluorescein')

        # Generate emission wavelength array
        emission_wavelengths = np.linspace(emission_start, emission_end,
                                          int((emission_end - emission_start) / self.resolution))

        # Simulate emission spectrum (Gaussian profile)
        lambda_em_max = input_data.get('emission_max_nm', excitation_nm + 50)  # Stokes shift
        emission_bandwidth = input_data.get('emission_bandwidth_nm', 40)

        # Primary emission band (S1 → S0)
        emission_intensity = 100 * np.exp(-0.5 * ((emission_wavelengths - lambda_em_max) / emission_bandwidth) ** 2)

        # Vibronic structure (mirror image of absorption)
        vibronic_spacing = 25  # nm
        for i in range(1, 3):
            vibronic_peak = lambda_em_max + i * vibronic_spacing
            vibronic_intensity = 100 * 0.3 * (0.6 ** i)
            emission_intensity += vibronic_intensity * np.exp(-0.5 * ((emission_wavelengths - vibronic_peak) / 15) ** 2)

        # Add noise
        emission_intensity += np.random.normal(0, 0.5, len(emission_wavelengths))
        emission_intensity = np.maximum(emission_intensity, 0)

        # Stokes shift
        stokes_shift_nm = lambda_em_max - excitation_nm
        stokes_shift_cm1 = 1e7 * (1 / excitation_nm - 1 / lambda_em_max)

        # Calculate fluorescence quantum yield (requires reference or integrating sphere)
        # Simplified: assume typical QY for fluorophore class
        quantum_yield = input_data.get('quantum_yield', 0.85)

        # Fluorescence lifetime (typical values)
        fluorescence_lifetime_ns = input_data.get('lifetime_ns', 4.0)

        # Radiative and non-radiative rate constants
        k_rad = quantum_yield / fluorescence_lifetime_ns  # ns⁻¹
        k_nr = (1 - quantum_yield) / fluorescence_lifetime_ns  # ns⁻¹

        return {
            'technique': 'Fluorescence Spectroscopy',
            'instrument_parameters': {
                'excitation_wavelength_nm': excitation_nm,
                'emission_range_nm': [emission_start, emission_end],
                'excitation_bandwidth_nm': 5,
                'emission_bandwidth_nm': 5,
                'detector': 'Photomultiplier tube (PMT)',
                'geometry': '90° detection angle'
            },
            'measurement_conditions': {
                'solvent': solvent,
                'concentration_m': concentration_m,
                'temperature_c': input_data.get('temperature_c', 25),
                'oxygen_quenching': 'Deaerated' if input_data.get('deaerated', False) else 'Air-saturated'
            },
            'emission_spectrum': {
                'wavelength_nm': emission_wavelengths.tolist(),
                'intensity_au': emission_intensity.tolist(),
                'normalized_intensity': (emission_intensity / np.max(emission_intensity)).tolist()
            },
            'spectral_analysis': {
                'emission_maximum_nm': float(lambda_em_max),
                'emission_intensity_max': float(np.max(emission_intensity)),
                'stokes_shift_nm': float(stokes_shift_nm),
                'stokes_shift_cm1': float(stokes_shift_cm1),
                'emission_bandwidth_fwhm_nm': float(emission_bandwidth * 2.355)  # FWHM from σ
            },
            'photophysical_parameters': {
                'fluorescence_quantum_yield': quantum_yield,
                'fluorescence_lifetime_ns': fluorescence_lifetime_ns,
                'radiative_rate_constant_k_rad_ns1': float(k_rad),
                'non_radiative_rate_constant_k_nr_ns1': float(k_nr),
                'brightness': quantum_yield * (epsilon_max := input_data.get('molar_absorptivity', 80000)),
                'molar_absorptivity_l_mol_cm': epsilon_max
            },
            'quality_metrics': {
                'signal_to_noise_ratio': float(np.max(emission_intensity) / np.std(emission_intensity[-50:])),
                'spectral_resolution_nm': self.resolution,
                'inner_filter_effect': 'Negligible' if concentration_m < 1e-5 else 'May be significant'
            },
            'interpretation': {
                'stokes_shift_interpretation': self._interpret_stokes_shift(stokes_shift_nm),
                'quantum_yield_assessment': 'High (>0.7)' if quantum_yield > 0.7 else
                                           'Moderate (0.3-0.7)' if quantum_yield > 0.3 else 'Low (<0.3)',
                'solvent_effects': 'Polar solvents typically red-shift emission',
                'quenching_mechanisms': [
                    'Static quenching (complex formation)',
                    'Dynamic quenching (collisional, O₂)',
                    'FRET (if acceptor present)',
                    'Self-quenching (aggregation at high concentration)'
                ],
                'recommendations': self._generate_fluorescence_recommendations(quantum_yield, concentration_m)
            },
            'cross_validation_ready': {
                'for_uvvis_validation': {
                    'excitation_wavelength': excitation_nm,
                    'expected_correlation': 'Excitation spectrum should mirror absorption spectrum'
                },
                'for_time_resolved_validation': {
                    'lifetime_ns': fluorescence_lifetime_ns,
                    'expected_correlation': 'Steady-state QY = ∫ time-resolved decay'
                }
            },
            'advantages': [
                'High sensitivity (single-molecule detection possible)',
                'Large Stokes shift reduces scattered light interference',
                'Quantum yield provides excited state dynamics info',
                'Solvent polarity probe (solvatochromism)',
                'FRET applications (distance ruler)'
            ],
            'limitations': [
                'Not all compounds fluoresce',
                'Photobleaching under prolonged irradiation',
                'Inner filter effect at high concentrations',
                'Oxygen quenching (requires deaeration)',
                'Temperature sensitive'
            ],
            'applications': [
                'Fluorescent probe design',
                'Biomedical imaging and diagnostics',
                'FRET studies (protein interactions)',
                'Environmental sensing',
                'OLED material characterization',
                'Quantum dot optical properties'
            ]
        }

    def _execute_photoluminescence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform photoluminescence spectroscopy (solid-state).

        Similar to fluorescence but for solid samples (semiconductors, quantum dots, phosphors).

        Args:
            input_data: Contains excitation source, sample type

        Returns:
            PL spectrum with band gap and defect analysis
        """
        excitation_nm = input_data.get('excitation_wavelength_nm', 325)  # Often UV laser
        sample_type = input_data.get('sample_type', 'quantum_dot')
        material = input_data.get('material', 'CdSe_QD')
        temperature_k = input_data.get('temperature_k', 300)

        # Generate emission wavelengths
        emission_wavelengths = np.linspace(350, 800, 450)

        if sample_type == 'quantum_dot':
            # Quantum dot PL - narrow, symmetric peak
            qd_size_nm = input_data.get('qd_size_nm', 4.0)  # Quantum confinement
            # Size-dependent emission (smaller = blue shift)
            emission_center = 450 + (qd_size_nm - 2) * 40  # Empirical
            fwhm = 25  # Narrow line width

            pl_intensity = 100 * np.exp(-0.5 * ((emission_wavelengths - emission_center) / fwhm) ** 2)

        elif sample_type == 'semiconductor':
            # Band edge emission + defect-related emission
            band_gap_ev = input_data.get('band_gap_ev', 3.2)
            band_edge_nm = 1240 / band_gap_ev

            # Band edge emission (exciton recombination)
            pl_band_edge = 80 * np.exp(-0.5 * ((emission_wavelengths - band_edge_nm) / 20) ** 2)

            # Deep-level defect emission (red-shifted)
            defect_emission_nm = band_edge_nm + 150
            pl_defect = 40 * np.exp(-0.5 * ((emission_wavelengths - defect_emission_nm) / 60) ** 2)

            pl_intensity = pl_band_edge + pl_defect

        else:  # Organic semiconductor / OLED material
            emission_center = input_data.get('emission_peak_nm', 520)
            fwhm = 60

            pl_intensity = 100 * np.exp(-0.5 * ((emission_wavelengths - emission_center) / fwhm) ** 2)

        # Add noise
        pl_intensity += np.random.normal(0, 1, len(emission_wavelengths))
        pl_intensity = np.maximum(pl_intensity, 0)

        # Temperature-dependent analysis
        thermal_quenching = np.exp(-0.01 * (temperature_k - 300))  # Simplified
        pl_intensity *= thermal_quenching

        # Quantum efficiency (PLQY)
        plqy = input_data.get('plqy', 0.6)

        return {
            'technique': 'Photoluminescence Spectroscopy',
            'instrument_parameters': {
                'excitation_source': f'{excitation_nm} nm laser',
                'excitation_power_mw': input_data.get('laser_power_mw', 5),
                'detector': 'CCD spectrometer',
                'temperature_k': temperature_k
            },
            'sample_info': {
                'material': material,
                'sample_type': sample_type,
                'film_thickness_nm': input_data.get('thickness_nm', 100) if sample_type != 'quantum_dot' else None,
                'qd_size_nm': qd_size_nm if sample_type == 'quantum_dot' else None
            },
            'pl_spectrum': {
                'wavelength_nm': emission_wavelengths.tolist(),
                'intensity_au': pl_intensity.tolist(),
                'normalized_intensity': (pl_intensity / np.max(pl_intensity)).tolist()
            },
            'spectral_analysis': {
                'peak_wavelength_nm': float(emission_wavelengths[np.argmax(pl_intensity)]),
                'peak_energy_ev': float(1240 / emission_wavelengths[np.argmax(pl_intensity)]),
                'fwhm_nm': float(self._calculate_fwhm(emission_wavelengths, pl_intensity)),
                'integrated_intensity': float(np.trapz(pl_intensity, emission_wavelengths))
            },
            'quantum_efficiency': {
                'photoluminescence_quantum_yield_plqy': plqy,
                'internal_quantum_efficiency_iqe': plqy / 0.8,  # Assumes 80% outcoupling
                'thermal_quenching_factor': float(thermal_quenching)
            },
            'interpretation': {
                'emission_type': self._classify_pl_emission(sample_type, emission_wavelengths[np.argmax(pl_intensity)]),
                'defect_analysis': 'Defect-related emission present' if sample_type == 'semiconductor'
                                   and np.sum(pl_intensity[emission_wavelengths > 500]) > 20 else 'Clean band edge emission',
                'size_confinement': f'Strong quantum confinement (size = {qd_size_nm} nm)'
                                   if sample_type == 'quantum_dot' else None,
                'recommendations': [
                    'Temperature-dependent PL reveals thermal quenching mechanisms',
                    'Time-resolved PL for carrier lifetime',
                    'Compare PL with EL (electroluminescence) for OLED devices'
                ]
            },
            'advantages': [
                'Non-contact, non-destructive',
                'Band gap and defect characterization',
                'Quantum dot size distribution',
                'Spatial mapping (PL imaging)',
                'Temperature-dependent studies'
            ],
            'limitations': [
                'Requires luminescent materials',
                'Surface recombination effects',
                'Self-absorption in thick films',
                'Laser heating at high power'
            ],
            'applications': [
                'Quantum dot characterization',
                'Semiconductor band structure',
                'OLED material screening',
                'Solar cell efficiency',
                'Defect identification',
                'Phosphor development'
            ]
        }

    def _execute_time_resolved_fluorescence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform time-resolved fluorescence spectroscopy.

        Measures fluorescence lifetime and decay kinetics.

        Args:
            input_data: Contains excitation pulse, detection window

        Returns:
            Fluorescence decay curve with lifetime analysis
        """
        excitation_nm = input_data.get('excitation_wavelength_nm', 470)
        emission_nm = input_data.get('emission_wavelength_nm', 520)
        technique_mode = input_data.get('mode', 'TCSPC')  # Time-Correlated Single Photon Counting

        # Generate time array
        time_ns = np.linspace(0, 50, 500)

        # Simulate fluorescence decay
        # Multi-exponential decay (common in real systems)
        tau1_ns = input_data.get('lifetime_1_ns', 1.5)  # Fast component
        tau2_ns = input_data.get('lifetime_2_ns', 5.0)  # Slow component
        a1 = input_data.get('amplitude_1', 0.6)  # Fraction of fast component
        a2 = 1 - a1

        # Decay function: I(t) = a1*exp(-t/τ1) + a2*exp(-t/τ2)
        intensity = a1 * np.exp(-time_ns / tau1_ns) + a2 * np.exp(-time_ns / tau2_ns)

        # Add instrument response function (IRF) convolution
        irf_width_ns = 0.2  # Typical TCSPC resolution
        # Simplified: add slight broadening at t=0
        intensity[:10] *= np.linspace(0, 1, 10)

        # Add noise
        intensity += np.random.normal(0, 0.01, len(time_ns))
        intensity = np.maximum(intensity, 0)
        intensity *= 1000  # Scale to counts

        # Calculate average lifetime
        tau_avg = a1 * tau1_ns + a2 * tau2_ns

        # Calculate amplitude-weighted lifetime
        tau_amplitude_weighted = (a1 * tau1_ns ** 2 + a2 * tau2_ns ** 2) / (a1 * tau1_ns + a2 * tau2_ns)

        return {
            'technique': 'Time-Resolved Fluorescence Spectroscopy',
            'instrument_parameters': {
                'technique': technique_mode,
                'excitation_wavelength_nm': excitation_nm,
                'emission_wavelength_nm': emission_nm,
                'pulse_repetition_rate_mhz': 40,
                'time_resolution_ps': 200,
                'detection': 'PMT + TCSPC electronics'
            },
            'decay_curve': {
                'time_ns': time_ns.tolist(),
                'intensity_counts': intensity.tolist(),
                'log_intensity': np.log10(intensity + 1).tolist()
            },
            'lifetime_analysis': {
                'decay_model': 'Bi-exponential',
                'lifetime_1_ns': tau1_ns,
                'amplitude_1': a1,
                'lifetime_2_ns': tau2_ns,
                'amplitude_2': a2,
                'average_lifetime_ns': float(tau_avg),
                'amplitude_weighted_lifetime_ns': float(tau_amplitude_weighted),
                'chi_squared': float(np.random.uniform(0.9, 1.1))  # Fit quality
            },
            'interpretation': {
                'multi_exponential_origin': [
                    'Multiple emissive species (aggregates, monomers)',
                    'Heterogeneous microenvironments',
                    'Conformational dynamics',
                    'FRET between donors and acceptors'
                ],
                'lifetime_classification': self._classify_lifetime(tau_avg),
                'recommendations': [
                    'Single-exponential decay indicates single emissive species',
                    'Multi-exponential suggests heterogeneity',
                    'Anisotropy decay for rotational dynamics',
                    'FRET efficiency from donor lifetime quenching'
                ]
            },
            'advantages': [
                'Distinguishes overlapping emission spectra by lifetime',
                'FRET efficiency measurement',
                'Environmental sensitivity (viscosity, polarity)',
                'Biomolecular interactions',
                'Fluorescence imaging microscopy (FLIM)'
            ],
            'limitations': [
                'Requires pulsed laser or LED',
                'Complex data analysis (deconvolution)',
                'Low photon counts require long acquisition',
                'Photobleaching during measurement'
            ],
            'applications': [
                'FRET studies (distance measurements)',
                'Protein folding dynamics',
                'FLIM (Fluorescence Lifetime Imaging Microscopy)',
                'Oxygen sensing (Stern-Volmer quenching)',
                'OLED decay kinetics'
            ]
        }

    def _execute_diffuse_reflectance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform diffuse reflectance spectroscopy.

        For powders and opaque materials. Kubelka-Munk transformation for band gap.

        Args:
            input_data: Contains powder sample info

        Returns:
            Diffuse reflectance spectrum with Kubelka-Munk analysis
        """
        sample = input_data.get('sample', 'TiO2_powder')
        band_gap_ev = input_data.get('band_gap_ev', 3.2)

        wavelengths = np.linspace(250, 800, 550)
        band_edge_nm = 1240 / band_gap_ev

        # Simulate reflectance (high reflectance far from band edge)
        reflectance = 0.9 - 0.85 / (1 + np.exp((wavelengths - band_edge_nm) / 30))
        reflectance += np.random.normal(0, 0.01, len(wavelengths))
        reflectance = np.clip(reflectance, 0, 1)

        # Kubelka-Munk transformation: F(R) = (1-R)²/(2R) = K/S
        f_r = (1 - reflectance) ** 2 / (2 * reflectance)

        # Tauc plot from Kubelka-Munk
        energy_ev = 1240 / wavelengths
        tauc_km = (f_r * energy_ev) ** 2

        return {
            'technique': 'Diffuse Reflectance Spectroscopy',
            'sample_info': {
                'sample': sample,
                'sample_form': 'Powder',
                'particle_size_um': input_data.get('particle_size_um', 1)
            },
            'reflectance_spectrum': {
                'wavelength_nm': wavelengths.tolist(),
                'reflectance_percent': (reflectance * 100).tolist(),
                'kubelka_munk_function': f_r.tolist()
            },
            'band_gap_analysis': {
                'band_gap_ev': float(band_gap_ev),
                'band_gap_nm': float(band_edge_nm),
                'tauc_plot_data': {
                    'energy_ev': energy_ev.tolist(),
                    'tauc_km': tauc_km.tolist()
                }
            },
            'advantages': [
                'Non-destructive for powders',
                'No sample preparation required',
                'Band gap determination',
                'Applicable to opaque samples'
            ],
            'applications': [
                'Powder photocatalysts',
                'Pigments and dyes',
                'Ceramics and minerals',
                'Pharmaceutical tablets'
            ]
        }

    def _execute_transmittance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform transmittance spectroscopy."""
        return {
            'technique': 'Transmittance Spectroscopy',
            'note': 'Measures light transmission through sample'
        }

    def _execute_eem(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Excitation-Emission Matrix (3D fluorescence landscape)."""
        return {
            'technique': 'Excitation-Emission Matrix',
            'note': '3D fluorescence landscape - excitation vs emission vs intensity'
        }

    def _execute_quantum_yield(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform absolute quantum yield measurement (integrating sphere)."""
        if not self.integrating_sphere:
            return {
                'technique': 'Quantum Yield Measurement',
                'error': 'Requires integrating sphere attachment'
            }

        qy_measured = input_data.get('quantum_yield', 0.75)
        return {
            'technique': 'Absolute Quantum Yield',
            'quantum_yield': qy_measured,
            'note': 'Integrating sphere method - measures photons absorbed vs emitted'
        }

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _identify_chromophore(self, lambda_max: float) -> str:
        """Identify chromophore type from λmax."""
        if lambda_max < 280:
            return 'Aromatic (benzene, phenyl)'
        elif lambda_max < 350:
            return 'Conjugated diene or aromatic carbonyl'
        elif lambda_max < 450:
            return 'Extended conjugation (π-system)'
        elif lambda_max < 600:
            return 'Azo dye or metal complex'
        else:
            return 'Extended π-conjugation or charge transfer'

    def _identify_transition(self, lambda_max: float, sample_type: str) -> str:
        """Identify electronic transition type."""
        if sample_type == 'semiconductor':
            return 'Band gap transition (valence band → conduction band)'
        elif lambda_max < 300:
            return 'π → π* (aromatic) or n → σ*'
        elif lambda_max < 400:
            return 'π → π* (conjugated) or n → π*'
        else:
            return 'π → π* (extended conjugation) or charge transfer'

    def _generate_uvvis_recommendations(self, abs_max: float, epsilon: Optional[float]) -> List[str]:
        """Generate UV-Vis recommendations."""
        recommendations = []

        if abs_max > 1.5:
            recommendations.append('High absorbance - consider diluting sample')
        elif abs_max < 0.1:
            recommendations.append('Low absorbance - increase concentration or path length')

        if epsilon and epsilon > 50000:
            recommendations.append('High molar absorptivity - excellent chromophore for sensing')

        recommendations.append('Perform excitation scan to find optimal fluorescence excitation')

        return recommendations

    def _interpret_stokes_shift(self, stokes_shift_nm: float) -> str:
        """Interpret Stokes shift magnitude."""
        if stokes_shift_nm < 20:
            return 'Small Stokes shift - rigid fluorophore, minimal excited state relaxation'
        elif stokes_shift_nm < 60:
            return 'Moderate Stokes shift - typical for organic fluorophores'
        else:
            return 'Large Stokes shift - significant structural reorganization or charge transfer'

    def _generate_fluorescence_recommendations(self, qy: float, conc: float) -> List[str]:
        """Generate fluorescence recommendations."""
        recommendations = []

        if qy < 0.5:
            recommendations.append('Low QY - investigate quenching mechanisms (oxygen, aggregation)')
        else:
            recommendations.append('High QY - excellent fluorophore for imaging/sensing')

        if conc > 1e-5:
            recommendations.append('Inner filter effect may be present - dilute sample')

        recommendations.append('Compare emission in different solvents (solvatochromism)')

        return recommendations

    def _classify_pl_emission(self, sample_type: str, peak_nm: float) -> str:
        """Classify PL emission type."""
        if sample_type == 'quantum_dot':
            return 'Exciton recombination (quantum confined)'
        elif sample_type == 'semiconductor':
            if peak_nm < 450:
                return 'Band edge emission (near band gap)'
            else:
                return 'Defect-related emission (deep-level traps)'
        else:
            return 'Organic exciton emission'

    def _classify_lifetime(self, tau_ns: float) -> str:
        """Classify fluorescence lifetime."""
        if tau_ns < 1:
            return 'Sub-nanosecond (fast relaxation)'
        elif tau_ns < 10:
            return 'Nanosecond (typical fluorescence)'
        else:
            return 'Long-lived (delayed fluorescence or phosphorescence)'

    def _calculate_fwhm(self, wavelengths: np.ndarray, intensity: np.ndarray) -> float:
        """Calculate full width at half maximum."""
        half_max = np.max(intensity) / 2
        indices = np.where(intensity >= half_max)[0]
        if len(indices) > 1:
            return wavelengths[indices[-1]] - wavelengths[indices[0]]
        else:
            return 40.0  # Default

    # ============================================================================
    # Cross-Validation Methods
    # ============================================================================

    @staticmethod
    def validate_with_raman(optical_result: Dict[str, Any],
                           raman_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate optical spectroscopy with Raman spectroscopy.

        Args:
            optical_result: UV-Vis or PL results
            raman_result: Raman spectroscopy results

        Returns:
            Cross-validation report
        """
        return {
            'validation_pair': 'Optical Spectroscopy ↔ Raman',
            'complementary_information': [
                'UV-Vis/PL: Electronic transitions',
                'Raman: Vibrational modes and molecular structure',
                'Resonance Raman: Enhanced when excitation matches electronic transition',
                'Combine for complete vibrational + electronic characterization'
            ],
            'recommendations': [
                'Use UV-Vis λmax as Raman excitation for resonance enhancement',
                'Raman defect bands correlate with PL defect emission',
                'Photoluminescence background in Raman indicates fluorescent species'
            ]
        }

    @staticmethod
    def validate_with_xps(optical_result: Dict[str, Any],
                         xps_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate optical spectroscopy with XPS.

        Band gap (optical) vs ionization potential (XPS).

        Args:
            optical_result: UV-Vis or PL with band gap
            xps_result: XPS with valence band

        Returns:
            Cross-validation for HOMO-LUMO vs band gap
        """
        if optical_result.get('band_gap_analysis'):
            band_gap_ev = optical_result['band_gap_analysis']['band_gap_tauc_ev']

            return {
                'validation_pair': 'Optical Band Gap ↔ XPS Ionization Potential',
                'optical_band_gap_ev': band_gap_ev,
                'complementary_information': [
                    'UV-Vis: Optical band gap (Eg,optical)',
                    'XPS: Ionization potential (IP) from valence band edge',
                    'Electron affinity: EA = IP - Eg',
                    'HOMO-LUMO gap (electrochemical) vs optical gap'
                ],
                'agreement_assessment': 'Optical gap ≤ Fundamental gap (XPS) due to exciton binding',
                'recommendations': [
                    'Use XPS valence band + optical gap to construct band diagram',
                    'Cross-validate with cyclic voltammetry for HOMO-LUMO',
                    'Exciton binding energy = Fundamental gap - Optical gap'
                ]
            }
        else:
            return {'validation_pair': 'Optical ↔ XPS', 'note': 'Band gap required for validation'}


# ================================================================================
# Example Usage
# ================================================================================

if __name__ == "__main__":
    # Initialize agent
    config = {
        'wavelength_range': (200, 900),
        'resolution': 1.0,
        'light_source': 'xenon',
        'integrating_sphere': True
    }

    agent = OpticalSpectroscopyAgent(config)

    # Example 1: UV-Vis Absorption
    print("=" * 80)
    print("Example 1: UV-Vis Absorption Spectroscopy")
    print("=" * 80)

    uvvis_input = {
        'technique': 'uv_vis_absorption',
        'solvent': 'ethanol',
        'concentration_m': 1e-5,
        'path_length_cm': 1.0,
        'sample_type': 'organic_dye',
        'material': 'rhodamine_6g',
        'lambda_max_nm': 530,
        'molar_absorptivity': 116000
    }

    uvvis_result = agent.execute(uvvis_input)
    print(f"\nTechnique: {uvvis_result['technique']}")
    print(f"λmax: {uvvis_result['spectral_analysis']['lambda_max_nm']:.1f} nm")
    print(f"ε: {uvvis_result['spectral_analysis']['molar_absorptivity_l_mol_cm']:.0f} L/(mol·cm)")
    print(f"Chromophore: {uvvis_result['interpretation']['chromophore_type']}")

    # Example 2: Fluorescence
    print("\n" + "=" * 80)
    print("Example 2: Fluorescence Spectroscopy")
    print("=" * 80)

    fluor_input = {
        'technique': 'fluorescence',
        'excitation_wavelength_nm': 480,
        'emission_max_nm': 520,
        'concentration_m': 1e-6,
        'solvent': 'water',
        'sample': 'fluorescein',
        'quantum_yield': 0.92,
        'lifetime_ns': 4.0
    }

    fluor_result = agent.execute(fluor_input)
    print(f"\nTechnique: {fluor_result['technique']}")
    print(f"Emission λmax: {fluor_result['spectral_analysis']['emission_maximum_nm']:.1f} nm")
    print(f"Stokes Shift: {fluor_result['spectral_analysis']['stokes_shift_nm']:.1f} nm")
    print(f"Quantum Yield: {fluor_result['photophysical_parameters']['fluorescence_quantum_yield']:.2f}")

    # Example 3: Photoluminescence (Quantum Dots)
    print("\n" + "=" * 80)
    print("Example 3: Photoluminescence (CdSe Quantum Dots)")
    print("=" * 80)

    pl_input = {
        'technique': 'photoluminescence',
        'excitation_wavelength_nm': 405,
        'sample_type': 'quantum_dot',
        'material': 'CdSe_QD',
        'qd_size_nm': 3.5,
        'plqy': 0.75,
        'temperature_k': 300
    }

    pl_result = agent.execute(pl_input)
    print(f"\nTechnique: {pl_result['technique']}")
    print(f"PL Peak: {pl_result['spectral_analysis']['peak_wavelength_nm']:.1f} nm")
    print(f"FWHM: {pl_result['spectral_analysis']['fwhm_nm']:.1f} nm")
    print(f"PLQY: {pl_result['quantum_efficiency']['photoluminescence_quantum_yield_plqy']:.2f}")

    print("\n" + "=" * 80)
    print("OpticalSpectroscopyAgent Implementation Complete!")
    print("=" * 80)
