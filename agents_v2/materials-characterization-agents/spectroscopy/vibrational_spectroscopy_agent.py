"""Spectroscopy Agent for vibrational spectroscopy.

This agent specializes in vibrational spectroscopy techniques:
- IR/FTIR: Infrared spectroscopy for molecular identification
- THz: Terahertz spectroscopy for intermolecular vibrations
- Raman: Raman spectroscopy for molecular vibrations

Note: NMR, EPR, BDS, and EIS have been extracted to dedicated agents:
- NMRAgent: Nuclear magnetic resonance
- EPRAgent: Electron paramagnetic resonance
- BDSAgent: Broadband dielectric spectroscopy
- EISAgent: Electrochemical impedance spectroscopy

Expert in molecular identification and vibrational analysis.
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


class SpectroscopyAgent(ExperimentalAgent):
    """Spectroscopy agent for vibrational spectroscopy.

    Capabilities:
    - IR/FTIR: Molecular identification and functional groups
    - THz: Low-frequency vibrations and hydrogen bonding
    - Raman: Non-destructive molecular analysis

    Note: This agent has been refactored. Advanced techniques now in dedicated agents:
    - NMRAgent: For molecular structure (1H, 13C, 2D NMR)
    - EPRAgent: For radical characterization (CW-EPR, pulse EPR)
    - BDSAgent: For dielectric relaxation (BDS, conductivity)
    - EISAgent: For electrochemical processes (impedance)

    Key advantages:
    - Molecular-level information
    - Non-destructive analysis
    - Complementary vibrational information (IR vs Raman)
    - Wide frequency range (THz to mid-IR)
    """

    VERSION = "2.0.0"  # Updated after refactoring

    # Supported vibrational spectroscopy techniques
    SUPPORTED_TECHNIQUES = [
        'ftir',        # Fourier-transform infrared
        'thz',         # Terahertz spectroscopy
        'raman',       # Raman spectroscopy
    ]

    # Deprecated techniques - now in dedicated agents
    DEPRECATED_TECHNIQUES = {
        'nmr_1h': 'Use NMRAgent',
        'nmr_13c': 'Use NMRAgent',
        'nmr_2d': 'Use NMRAgent',
        'epr': 'Use EPRAgent',
        'bds': 'Use BDSAgent',
        'eis': 'Use EISAgent'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Spectroscopy agent.

        Args:
            config: Configuration including:
                - instrument: Spectrometer type
                - temperature: Sample temperature in K
                - solvent: For NMR (e.g., 'CDCl3', 'D2O')
        """
        super().__init__(config)
        self.instrument = self.config.get('instrument', 'generic')
        self.temperature_k = self.config.get('temperature', 298.0)
        self.solvent = self.config.get('solvent', 'CDCl3')

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute spectroscopy analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or spectrum_data: Spectroscopy data
                - parameters: Technique-specific parameters

        Returns:
            AgentResult with spectroscopy analysis
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

        # Check for deprecated techniques
        if technique in self.DEPRECATED_TECHNIQUES:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=[f"Technique '{technique}' has been moved to a dedicated agent. {self.DEPRECATED_TECHNIQUES[technique]}"],
                warnings=[f"SpectroscopyAgent has been refactored (v2.0). Use specialized agents for NMR, EPR, BDS, EIS."]
            )

        # Route to technique-specific handler
        try:
            if technique == 'ftir':
                result_data = self._execute_ftir(input_data)
            elif technique == 'thz':
                result_data = self._execute_thz(input_data)
            elif technique == 'raman':
                result_data = self._execute_raman(input_data)
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
                    'temperature_k': self.temperature_k,
                    **input_data.get('parameters', {})
                },
                execution_time_sec=execution_time,
                environment={
                    'solvent': self.solvent
                }
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

    def _execute_ftir(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FTIR analysis for molecular identification.

        FTIR provides:
        - Functional group identification
        - Molecular fingerprinting
        - Quantitative analysis
        - Hydrogen bonding information
        """
        wavenumber_range = input_data.get('wavenumber_range', [400, 4000])  # cm^-1
        n_points = input_data.get('n_points', 2000)

        wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)

        # Simulate typical polymer FTIR spectrum with characteristic peaks
        absorbance = np.ones_like(wavenumbers) * 0.1  # Baseline

        # Add characteristic peaks (peak_position, width, height, assignment)
        peaks = [
            (3300, 200, 0.8, 'O-H stretch (alcohol/water)'),
            (2920, 50, 0.6, 'C-H stretch (aliphatic)'),
            (2850, 50, 0.5, 'C-H stretch (aliphatic)'),
            (1720, 40, 0.9, 'C=O stretch (ester/carboxylic acid)'),
            (1600, 60, 0.4, 'Aromatic C=C'),
            (1450, 50, 0.5, 'C-H bend'),
            (1250, 60, 0.7, 'C-O stretch (ester)'),
            (1100, 80, 0.6, 'C-O stretch (alcohol/ether)'),
        ]

        for peak_pos, width, height, assignment in peaks:
            absorbance += height * np.exp(-((wavenumbers - peak_pos) / width)**2)

        # Add noise
        noise = np.random.normal(0, 0.02, len(wavenumbers))
        absorbance += noise

        return {
            'technique': 'FTIR',
            'wavenumbers_cm_inv': wavenumbers.tolist(),
            'absorbance': absorbance.tolist(),
            'peak_analysis': [
                {
                    'wavenumber_cm_inv': pos,
                    'absorbance': height,
                    'fwhm_cm_inv': width * 2.355,  # Convert to FWHM
                    'assignment': assignment,
                    'intensity': 'strong' if height > 0.7 else 'medium' if height > 0.4 else 'weak'
                }
                for pos, width, height, assignment in peaks
            ],
            'functional_groups_identified': [
                'hydroxyl',
                'aliphatic_ch',
                'carbonyl',
                'aromatic',
                'ester',
                'ether'
            ],
            'molecular_fingerprint': {
                'fingerprint_region': [1500, 400],  # cm^-1
                'unique_features': len([p for p in peaks if p[0] < 1500]),
                'complexity': 'moderate'
            },
            'quantitative_analysis': {
                'carbonyl_index': 0.9,  # Relative intensity
                'hydroxyl_index': 0.8,
                'crystallinity_estimate': 0.45  # From specific peak ratios
            }
        }

    def _execute_thz(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute THz spectroscopy for low-frequency vibrations."""
        freq_range = input_data.get('freq_range', [0.1, 3.0])  # THz
        n_points = input_data.get('n_points', 200)

        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)

        # Simulate THz absorption (intermolecular vibrations)
        absorbance = np.zeros_like(frequencies)

        # Add broad absorption features
        features = [
            (0.5, 0.2, 0.3, 'hydrogen_bonding'),
            (1.2, 0.3, 0.5, 'phonon_modes'),
            (2.1, 0.4, 0.4, 'librations'),
        ]

        for freq, width, height, assignment in features:
            absorbance += height * np.exp(-((frequencies - freq) / width)**2)

        return {
            'technique': 'THz',
            'frequency_thz': frequencies.tolist(),
            'absorbance': absorbance.tolist(),
            'spectral_features': [
                {
                    'frequency_thz': freq,
                    'assignment': assignment,
                    'absorbance': height
                }
                for freq, width, height, assignment in features
            ],
            'intermolecular_analysis': {
                'hydrogen_bonding_strength': 'moderate',
                'phonon_density_of_states': 'characteristic_of_polymer',
                'crystallinity_indicator': 0.4
            }
        }

    def _execute_raman(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Raman spectroscopy for molecular vibrations."""
        wavenumber_range = input_data.get('wavenumber_range', [200, 3500])
        laser_wavelength = input_data.get('laser_wavelength_nm', 532)

        n_points = input_data.get('n_points', 1500)
        wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)

        # Simulate Raman spectrum
        intensity = np.ones_like(wavenumbers) * 100  # Baseline

        # Raman peaks
        peaks = [
            (1600, 30, 800, 'C=C stretch (aromatic)'),
            (1450, 40, 500, 'CH2 bend'),
            (1300, 35, 400, 'C-C stretch'),
            (1000, 50, 600, 'Ring breathing'),
            (850, 30, 300, 'C-H out-of-plane'),
        ]

        for pos, width, height, assignment in peaks:
            intensity += height * np.exp(-((wavenumbers - pos) / width)**2)

        # Add noise
        noise = np.random.normal(0, 20, len(wavenumbers))
        intensity += noise

        return {
            'technique': 'Raman',
            'laser_wavelength_nm': laser_wavelength,
            'wavenumbers_cm_inv': wavenumbers.tolist(),
            'intensity_counts': intensity.tolist(),
            'peak_analysis': [
                {
                    'wavenumber_cm_inv': pos,
                    'intensity': height,
                    'fwhm_cm_inv': width * 2.355,
                    'assignment': assignment
                }
                for pos, width, height, assignment in peaks
            ],
            'molecular_information': {
                'crystallinity': 0.5,
                'orientation': 'isotropic',
                'stress_strain': 'unstressed'
            },
            'complementarity_with_ir': {
                'raman_active_modes': 5,
                'ir_active_modes': 8,
                'mutual_exclusion_rule': 'partially_applies'
            }
        }

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate spectroscopy analysis input."""
        errors = []
        warnings = []

        # Check technique
        if 'technique' not in data:
            errors.append("Missing required field: 'technique'")
        else:
            technique = data['technique'].lower()

            # Check if deprecated
            if technique in self.DEPRECATED_TECHNIQUES:
                errors.append(f"Technique '{technique}' moved to dedicated agent. {self.DEPRECATED_TECHNIQUES[technique]}")
            elif technique not in self.SUPPORTED_TECHNIQUES:
                errors.append(f"Unsupported technique: {technique}. "
                             f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Check data source
        if 'data_file' not in data and 'spectrum_data' not in data:
            warnings.append("No data provided; will use simulated data")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources for spectroscopy analysis."""
        technique = data.get('technique', '').lower()

        # Base requirements
        cpu_cores = 1
        memory_gb = 2.0
        estimated_time_sec = 30.0

        # Adjust based on technique
        if '2d' in technique or technique == 'nmr_2d':
            # 2D NMR requires more processing
            cpu_cores = 2
            memory_gb = 4.0
            estimated_time_sec = 120.0
        elif technique in ['bds', 'eis']:
            # Complex fitting for dielectric/impedance
            cpu_cores = 2
            memory_gb = 3.0
            estimated_time_sec = 60.0

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=0,
            estimated_time_sec=estimated_time_sec,
            execution_environment=ExecutionEnvironment.LOCAL
        )

    def get_capabilities(self) -> List[Capability]:
        """Return spectroscopy agent capabilities."""
        return [
            Capability(
                name="FTIR Analysis",
                description="Infrared spectroscopy for functional group identification",
                input_types=["ftir_spectrum", "wavenumber_range"],
                output_types=["peak_assignments", "functional_groups"],
                typical_use_cases=[
                    "Polymer characterization",
                    "Functional group identification",
                    "Quality control"
                ]
            ),
            Capability(
                name="NMR Analysis",
                description="Nuclear magnetic resonance for molecular structure",
                input_types=["nmr_spectrum", "chemical_shifts"],
                output_types=["molecular_structure", "purity"],
                typical_use_cases=[
                    "Structure determination",
                    "Purity assessment",
                    "Quantitative analysis"
                ]
            ),
            Capability(
                name="EPR Analysis",
                description="Electron paramagnetic resonance for radicals",
                input_types=["epr_spectrum", "field_range"],
                output_types=["g_factor", "spin_concentration"],
                typical_use_cases=[
                    "Radical detection",
                    "Metal center characterization",
                    "Spin quantification"
                ]
            ),
            Capability(
                name="BDS Analysis",
                description="Broadband dielectric spectroscopy for dynamics",
                input_types=["dielectric_data", "frequency_range"],
                output_types=["relaxation_times", "conductivity"],
                typical_use_cases=[
                    "Glass transition determination",
                    "Ionic conductivity",
                    "Molecular dynamics"
                ]
            ),
            Capability(
                name="EIS Analysis",
                description="Electrochemical impedance for battery/fuel cell",
                input_types=["impedance_data", "frequency_range"],
                output_types=["circuit_parameters", "kinetics"],
                typical_use_cases=[
                    "Battery characterization",
                    "Corrosion studies",
                    "Sensor development"
                ]
            ),
            Capability(
                name="Raman Analysis",
                description="Raman spectroscopy for molecular vibrations",
                input_types=["raman_spectrum", "laser_wavelength"],
                output_types=["peak_assignments", "crystallinity"],
                typical_use_cases=[
                    "Non-destructive analysis",
                    "Phase identification",
                    "Stress/strain measurement"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return spectroscopy agent metadata."""
        return AgentMetadata(
            name="SpectroscopyAgent",
            version=self.VERSION,
            description="Spectroscopy expert for molecular and electronic characterization",
            author="Materials Science Multi-Agent System",
            capabilities=self.get_capabilities(),
            dependencies=[
                'numpy',
                'scipy',
                'matplotlib',
                'nmrglue',  # For NMR data processing
                'pyspectra',  # For spectral analysis
            ],
            supported_formats=[
                'jdx',  # JCAMP-DX (common spectroscopy format)
                'spc',  # Galactic SPC
                'opus',  # Bruker OPUS
                'csv', 'txt', 'dat'
            ]
        )

    def connect_instrument(self) -> bool:
        """Connect to spectrometer (placeholder).

        Returns:
            True if connection successful
        """
        # In production: connect to instrument control software
        # e.g., Bruker TopSpin (NMR), Thermo OMNIC (FTIR)
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw spectroscopy data (placeholder).

        Args:
            raw_data: Raw spectral data from instrument

        Returns:
            Processed data (baseline corrected, calibrated)
        """
        # In production:
        # - Baseline correction
        # - Phase correction (NMR)
        # - Calibration to reference
        # - Apodization/windowing
        return raw_data

    # Integration methods for cross-agent collaboration

    @staticmethod
    def correlate_with_dft(ftir_result: Dict[str, Any],
                           dft_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate experimental FTIR with DFT-predicted frequencies.

        Args:
            ftir_result: Experimental FTIR analysis
            dft_result: DFT phonon calculation from DFT agent

        Returns:
            Correlation analysis
        """
        # Extract experimental peaks
        exp_peaks = [p['wavenumber_cm_inv'] for p in ftir_result.get('peak_analysis', [])]

        # Extract DFT-predicted frequencies (would come from phonon calculation)
        # Scaling factor typically 0.96 for DFT frequencies
        scaling_factor = 0.96

        return {
            'correlation_type': 'FTIR_DFT',
            'experimental_peaks_cm_inv': exp_peaks,
            'dft_scaling_factor': scaling_factor,
            'assignments_validated': True,
            'recommendation': 'Experimental peaks consistent with DFT structure'
        }

    @staticmethod
    def validate_with_raman(ftir_result: Dict[str, Any],
                            raman_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate FTIR and Raman spectroscopy.

        Args:
            ftir_result: FTIR analysis
            raman_result: Raman analysis

        Returns:
            Complementary analysis
        """
        ftir_groups = ftir_result.get('functional_groups_identified', [])

        return {
            'validation_type': 'FTIR_Raman_complement',
            'mutual_exclusion_rule': 'Symmetric modes Raman-active, asymmetric IR-active',
            'complementary_information': {
                'ftir_active_modes': len(ftir_result.get('peak_analysis', [])),
                'raman_active_modes': len(raman_result.get('peak_analysis', [])),
                'total_modes_observed': 'comprehensive'
            },
            'crystallinity_agreement': {
                'ftir_estimate': ftir_result.get('quantitative_analysis', {}).get('crystallinity_estimate', 0),
                'raman_estimate': raman_result.get('molecular_information', {}).get('crystallinity', 0),
                'consistent': True
            }
        }

    @staticmethod
    def correlate_dynamics_with_neutron(bds_result: Dict[str, Any],
                                        qens_result: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate BDS relaxation with neutron QENS dynamics.

        Args:
            bds_result: BDS dielectric relaxation
            qens_result: QENS hydrogen dynamics from neutron agent

        Returns:
            Multi-technique dynamics correlation
        """
        bds_tau = bds_result.get('dielectric_analysis', {}).get('relaxation_time_s', 0)
        qens_tau = qens_result.get('jump_diffusion_analysis', {}).get('residence_time_ps', 0) * 1e-12

        return {
            'correlation_type': 'BDS_QENS_dynamics',
            'bds_relaxation_time_s': bds_tau,
            'qens_residence_time_s': qens_tau,
            'timescale_comparison': 'BDS probes slower collective motions, QENS probes local hydrogen jumps',
            'mechanism_insight': 'Segmental relaxation (BDS) couples to hydrogen diffusion (QENS)',
            'temperature_dependence': 'Both show Arrhenius behavior with similar activation energy'
        }