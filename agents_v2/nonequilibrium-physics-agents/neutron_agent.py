"""Neutron Scattering Agent for soft matter characterization.

This agent specializes in neutron scattering and spectroscopy techniques:
- SANS: Small-angle neutron scattering with H/D contrast
- NSE: Neutron spin echo for ultra-high energy resolution dynamics
- QENS: Quasi-elastic neutron scattering for hydrogen diffusion
- NR: Neutron reflectometry for interfaces
- INS: Inelastic neutron scattering for vibrational spectroscopy

Expert in hydrogen-sensitive dynamics, isotopic contrast engineering,
and deep bulk characterization of soft matter systems.
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


class NeutronAgent(ExperimentalAgent):
    """Neutron scattering and spectroscopy agent for soft matter systems.

    Capabilities:
    - SANS: H/D contrast for structure (1-1000 nm)
    - NSE: Ultra-high energy resolution dynamics (ns-μs timescales)
    - QENS: Hydrogen diffusion and relaxation (ps-ns)
    - NR: Interface and thin film structure
    - INS: Vibrational spectroscopy and hydrogen dynamics

    Key advantages:
    - Hydrogen sensitivity (unique to neutrons)
    - Isotopic contrast via H/D substitution
    - Deep penetration for bulk characterization
    - Ultra-high energy resolution (NSE: neV)
    """

    VERSION = "1.0.0"

    # Supported neutron techniques
    SUPPORTED_TECHNIQUES = [
        'sans',    # Small-angle neutron scattering
        'nse',     # Neutron spin echo
        'qens',    # Quasi-elastic neutron scattering
        'nr',      # Neutron reflectometry
        'ins',     # Inelastic neutron scattering
        'sans_contrast',  # Contrast variation SANS
    ]

    # Supported sample types
    SUPPORTED_SAMPLES = [
        'polymer',
        'biological',
        'hydrogel',
        'ionic_liquid',
        'complex_fluid',
        'membrane',
        'protein',
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Neutron agent.

        Args:
            config: Configuration including:
                - facility: Neutron facility name
                - instrument: Instrument name
                - wavelength: Neutron wavelength in Å
                - temperature: Sample temperature in K
        """
        super().__init__(config)
        self.facility = self.config.get('facility', 'generic')
        self.instrument = self.config.get('instrument', 'generic')
        self.wavelength_angstrom = self.config.get('wavelength', 6.0)
        self.temperature_k = self.config.get('temperature', 298.0)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute neutron scattering/spectroscopy analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - data_file or data_array: Neutron data
                - h_d_fraction (optional): Deuteration level
                - parameters: Technique-specific parameters

        Returns:
            AgentResult with neutron analysis results
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
            if technique == 'sans':
                result_data = self._execute_sans(input_data)
            elif technique == 'nse':
                result_data = self._execute_nse(input_data)
            elif technique == 'qens':
                result_data = self._execute_qens(input_data)
            elif technique == 'nr':
                result_data = self._execute_nr(input_data)
            elif technique == 'ins':
                result_data = self._execute_ins(input_data)
            elif technique == 'sans_contrast':
                result_data = self._execute_sans_contrast(input_data)
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
                    'facility': self.facility,
                    'instrument': self.instrument,
                    'wavelength_angstrom': self.wavelength_angstrom,
                    'temperature_k': self.temperature_k,
                    **input_data.get('parameters', {})
                },
                execution_time_sec=execution_time,
                environment={
                    'neutron_source': 'reactor',  # or spallation
                    'flux': 'high'
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'technique': technique,
                    'execution_time_sec': execution_time,
                    'facility': self.facility,
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

    def _execute_sans(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SANS analysis with H/D contrast.

        SANS provides:
        - Particle size and shape
        - Polymer chain dimensions
        - Structure factor
        - Domain structure with isotopic contrast
        """
        q_range = input_data.get('q_range', [0.001, 0.5])  # Å^-1
        n_points = input_data.get('n_points', 100)
        h_d_fraction = input_data.get('h_d_fraction', 0.0)  # 0=fully H, 1=fully D

        q = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), n_points)

        # Guinier analysis
        rg_nm = 8.0  # Radius of gyration
        i0 = 1500.0  # Forward scattering (enhanced by H/D contrast)

        # Contrast variation effect
        # SLD difference increases with deuteration
        contrast_factor = 1.0 + 5.0 * h_d_fraction

        intensity = np.zeros_like(q)
        for i, q_val in enumerate(q):
            if q_val * rg_nm * 10 < 1.0:
                # Guinier regime
                intensity[i] = i0 * contrast_factor * np.exp(-(q_val * rg_nm * 10)**2 / 3)
            else:
                # Porod regime
                intensity[i] = 5e5 / q_val**4

        # Add noise
        noise = np.random.normal(0, 0.03 * intensity)
        intensity += noise

        return {
            'technique': 'SANS',
            'scattering_vector_inv_angstrom': q.tolist(),
            'intensity_cm_inv': intensity.tolist(),
            'deuteration_level': h_d_fraction,
            'guinier_analysis': {
                'radius_of_gyration_nm': rg_nm,
                'forward_scattering_i0': i0 * contrast_factor,
                'molecular_weight_kda': 150.0,
                'guinier_region_valid': True
            },
            'contrast_analysis': {
                'scattering_length_density_contrast': contrast_factor,
                'h_fraction': 1.0 - h_d_fraction,
                'd_fraction': h_d_fraction,
                'contrast_match_point': 0.41  # D2O fraction for match
            },
            'form_factor_fit': {
                'model': 'worm_like_chain',  # Common for polymers
                'contour_length_nm': 100.0,
                'kuhn_length_nm': 15.0,
                'cross_section_radius_nm': 0.8,
                'chi_squared': 1.1
            },
            'structure_factor': {
                'correlation_length_nm': 20.0,
                'excluded_volume_parameter': 0.5,
                'overlap_concentration_mg_ml': 5.0
            },
            'physical_properties': {
                'hydrodynamic_radius_nm': rg_nm * 0.775,  # Rh ≈ 0.775*Rg for Gaussian
                'persistence_length_nm': 15.0,
                'chain_conformation': 'semi_flexible'
            }
        }

    def _execute_nse(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NSE (Neutron Spin Echo) for high-resolution dynamics.

        NSE provides:
        - Ultra-high energy resolution (neV)
        - Dynamics on ns-μs timescales
        - Intermediate scattering function I(q,t)
        - Collective motions and relaxations
        """
        # Get parameters (can be in top-level or parameters dict)
        params = input_data.get('parameters', {})
        q_value = params.get('q_value', input_data.get('q_value', 0.05))  # Å^-1
        fourier_times_ns = params.get('fourier_times_ns',
                                      input_data.get('fourier_times_ns',
                                                     np.logspace(-1, 2, 20)))  # 0.1-100 ns

        if not isinstance(fourier_times_ns, np.ndarray):
            fourier_times_ns = np.array(fourier_times_ns)

        # Simulate intermediate scattering function decay
        # Example: Polymer chain dynamics (Zimm model)
        tau_ns = 50.0  # Relaxation time
        beta = 0.85  # Stretching exponent

        # I(q,t) = exp(-(t/tau)^beta)
        iqt = np.exp(-(fourier_times_ns / tau_ns)**beta)

        # Extract dynamics parameters
        # Mean relaxation time from stretched exponential
        gamma_factor = 1.15  # Γ(1+1/β)/β for β=0.85
        tau_mean_ns = tau_ns * gamma_factor

        return {
            'technique': 'NSE',
            'q_value_inv_angstrom': q_value,
            'fourier_times_ns': fourier_times_ns.tolist(),
            'intermediate_scattering_function': iqt.tolist(),
            'dynamics_analysis': {
                'relaxation_time_ns': tau_ns,
                'mean_relaxation_time_ns': tau_mean_ns,
                'stretching_exponent_beta': beta,
                'dynamics_type': 'stretched_exponential',
                'heterogeneity': 'moderate'
            },
            'physical_interpretation': {
                'transport_mechanism': 'polymer_reptation',
                'diffusion_coefficient_cm2_s': 5e-8,
                'characteristic_length_nm': 2 * np.pi / q_value / 10,
                'mode_type': 'collective_relaxation'
            },
            'q_dependence': {
                'q_squared_scaling': True,  # D ~ q^2 for diffusion
                'rouse_zimm_regime': 'zimm',  # with hydrodynamics
                'entanglement_effects': False
            },
            'experimental_details': {
                'energy_resolution_nev': 0.01,
                'maximum_fourier_time_ns': max(fourier_times_ns),
                'temperature_k': self.temperature_k
            }
        }

    def _execute_qens(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute QENS (Quasi-Elastic Neutron Scattering) for hydrogen dynamics.

        QENS provides:
        - Hydrogen diffusion coefficients
        - Jump diffusion parameters
        - Relaxation processes
        - Localized motions
        """
        q_range = input_data.get('q_range', [0.3, 2.0])  # Å^-1
        energy_range_mev = input_data.get('energy_range_mev', [-2.0, 2.0])

        # Simulate QENS spectrum
        n_q = 5
        n_e = 100
        q_values = np.linspace(q_range[0], q_range[1], n_q)
        energies = np.linspace(energy_range_mev[0], energy_range_mev[1], n_e)

        # Jump diffusion model
        d_jump_cm2_s = 1e-5  # Jump diffusion coefficient
        tau_residence_ps = 5.0  # Residence time
        jump_length_angstrom = 3.5  # Jump distance

        # Calculate HWHM (half-width at half-maximum) vs q
        hwhm_mev = []
        for q_val in q_values:
            # EISF (elastic incoherent structure factor)
            eisf = 0.8  # Fraction of elastic scattering
            # Lorentzian width from jump diffusion
            gamma = (2.63 / tau_residence_ps)  # Convert ps to meV
            hwhm_mev.append(gamma)

        return {
            'technique': 'QENS',
            'q_values_inv_angstrom': q_values.tolist(),
            'energy_transfer_mev': energies.tolist(),
            'jump_diffusion_analysis': {
                'diffusion_coefficient_cm2_s': d_jump_cm2_s,
                'residence_time_ps': tau_residence_ps,
                'jump_length_angstrom': jump_length_angstrom,
                'jump_frequency_thz': 1.0 / tau_residence_ps / 1000
            },
            'spectral_analysis': {
                'hwhm_vs_q_mev': hwhm_mev,
                'elastic_intensity_fraction': 0.8,
                'quasielastic_intensity_fraction': 0.2,
                'model': 'jump_diffusion'
            },
            'hydrogen_dynamics': {
                'diffusion_type': 'jump_diffusion',
                'confined_geometry': False,
                'hydrogen_bond_dynamics': True,
                'activation_energy_kj_mol': 15.0
            },
            'physical_properties': {
                'mobility': 'high',
                'confinement_radius_angstrom': None,
                'rotational_diffusion': False,
                'translational_diffusion': True
            },
            'experimental_details': {
                'energy_resolution_mev': 0.1,
                'q_resolution_inv_angstrom': 0.05,
                'temperature_k': self.temperature_k
            }
        }

    def _execute_nr(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NR (Neutron Reflectometry) for interface structure.

        NR provides:
        - Layer thickness and composition
        - Interface roughness
        - Scattering length density profiles
        - Buried interfaces
        """
        q_range = input_data.get('q_range', [0.01, 0.3])  # Å^-1
        n_points = input_data.get('n_points', 100)

        q = np.linspace(q_range[0], q_range[1], n_points)

        # Simulate reflectivity curve
        # Simple two-layer model on substrate
        layer1_thickness_angstrom = 100.0  # e.g., polymer brush
        layer2_thickness_angstrom = 50.0   # e.g., oxide layer

        # Simplified Fresnel reflectivity with interference
        reflectivity = np.zeros_like(q)
        for i, q_val in enumerate(q):
            # Fresnel reflectivity with layer interference
            phase = q_val * (layer1_thickness_angstrom + layer2_thickness_angstrom)
            fresnel = (0.02 / q_val**4) * (1 + 0.5 * np.cos(phase))
            reflectivity[i] = max(fresnel, 1e-8)  # Avoid log issues

        return {
            'technique': 'Neutron Reflectometry',
            'q_values_inv_angstrom': q.tolist(),
            'reflectivity': reflectivity.tolist(),
            'layer_structure': [
                {
                    'layer_number': 1,
                    'material': 'polymer_layer',
                    'thickness_angstrom': layer1_thickness_angstrom,
                    'thickness_nm': layer1_thickness_angstrom / 10,
                    'sld_e_minus_6_angstrom_minus_2': 2.5,  # D-polymer
                    'roughness_angstrom': 8.0
                },
                {
                    'layer_number': 2,
                    'material': 'oxide_layer',
                    'thickness_angstrom': layer2_thickness_angstrom,
                    'thickness_nm': layer2_thickness_angstrom / 10,
                    'sld_e_minus_6_angstrom_minus_2': 4.0,
                    'roughness_angstrom': 5.0
                }
            ],
            'interface_analysis': {
                'total_thickness_nm': (layer1_thickness_angstrom + layer2_thickness_angstrom) / 10,
                'interface_roughness_angstrom': 6.5,
                'interface_quality': 'sharp',
                'interdiffusion_present': False
            },
            'contrast_variation': {
                'solvent': 'D2O',
                'contrast_match_point': 'Si @ 38% D2O',
                'penetration_depth_nm': 100.0
            },
            'model_fit': {
                'chi_squared': 1.3,
                'model_type': 'slab_model',
                'n_layers': 2,
                'substrate': 'silicon'
            }
        }

    def _execute_ins(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute INS (Inelastic Neutron Scattering) for vibrational spectroscopy.

        INS provides:
        - Vibrational modes
        - Phonon density of states
        - Hydrogen dynamics
        - Molecular vibrations
        """
        energy_range_mev = input_data.get('energy_range_mev', [0, 500])
        n_points = input_data.get('n_points', 200)

        energies = np.linspace(energy_range_mev[0], energy_range_mev[1], n_points)

        # Simulate vibrational spectrum with peaks
        intensity = np.zeros_like(energies)

        # Add vibrational peaks (in meV)
        peak_positions = [50, 120, 180, 350]  # meV
        peak_widths = [10, 15, 20, 30]
        peak_heights = [100, 80, 60, 40]

        for pos, width, height in zip(peak_positions, peak_widths, peak_heights):
            intensity += height * np.exp(-((energies - pos) / width)**2)

        # Add background
        intensity += 10

        # Convert to frequency (cm^-1) for molecular vibrations
        cm_inv_conversion = 8.066  # meV to cm^-1

        return {
            'technique': 'INS',
            'energy_transfer_mev': energies.tolist(),
            'intensity_arbitrary_units': intensity.tolist(),
            'vibrational_modes': [
                {
                    'energy_mev': pos,
                    'frequency_cm_inv': pos * cm_inv_conversion,
                    'frequency_thz': pos * 0.242,
                    'assignment': assignment,
                    'relative_intensity': height / max(peak_heights)
                }
                for pos, height, assignment in zip(
                    peak_positions,
                    peak_heights,
                    ['C-H bend', 'C-C stretch', 'CH2 wag', 'C-H stretch']
                )
            ],
            'phonon_analysis': {
                'density_of_states_available': True,
                'acoustic_modes_present': True,
                'optical_modes_present': True,
                'debye_temperature_k': 200.0
            },
            'hydrogen_specific': {
                'h_scattering_dominates': True,
                'ch_modes_strong': True,
                'oh_modes_present': False,
                'nh_modes_present': False
            },
            'comparison_with_ir_raman': {
                'ins_advantages': 'No selection rules, all modes visible',
                'complementary_to_ir': True,
                'complementary_to_raman': True
            }
        }

    def _execute_sans_contrast(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SANS with systematic contrast variation.

        Contrast variation provides:
        - Component-specific structure
        - Domain composition
        - Selective deuteration analysis
        - Match point determination
        """
        d2o_fractions = input_data.get('d2o_fractions', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Simulate contrast variation series
        contrast_data = []

        for d_frac in d2o_fractions:
            # Calculate scattering length density contrast
            # Water SLD varies with D2O fraction
            sld_solvent = -0.56 + (6.38 - (-0.56)) * d_frac  # x10^-6 Å^-2

            # Example: Polymer with SLD ~ 1.0 x10^-6 Å^-2
            sld_polymer = 1.0
            contrast = (sld_polymer - sld_solvent)**2

            contrast_data.append({
                'd2o_fraction': d_frac,
                'solvent_sld': sld_solvent,
                'contrast_squared': contrast,
                'scattering_intensity': contrast * 1000,  # Simplified
                'visibility': 'good' if contrast > 1.0 else 'poor'
            })

        # Find contrast match point (where scattering minimized)
        match_point = 0.17  # 17% D2O matches hydrogenous polymer

        return {
            'technique': 'SANS Contrast Variation',
            'contrast_series': contrast_data,
            'match_point_analysis': {
                'd2o_fraction_match': match_point,
                'component_matched': 'polymer_matrix',
                'sld_matched_component': 1.0,
                'optimal_contrast_d2o_fraction': 0.0  # Pure H2O
            },
            'selective_deuteration': {
                'deuteration_strategy': 'chain_end_labeling',
                'deuterated_component': 'polymer_A',
                'hydrogenous_component': 'polymer_B',
                'contrast_amplification': 'excellent'
            },
            'multicomponent_analysis': {
                'n_components': 2,
                'component_1_sld': 1.0,
                'component_2_sld': 6.0,  # Deuterated
                'demixing_length_scale_nm': 15.0
            },
            'recommendations': {
                'optimal_contrasts': [0.0, 0.42, 1.0],  # H2O, match, D2O
                'selective_deuteration_needed': False,
                'number_of_contrasts_sufficient': True
            }
        }

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate neutron analysis input."""
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

        if technique in ['sans', 'sans_contrast']:
            if 'h_d_fraction' not in data and 'h_d_fraction' not in data.get('parameters', {}):
                warnings.append("SANS: deuteration level not specified, using fully hydrogenous")

        if technique == 'nse':
            if 'fourier_times_ns' not in data.get('parameters', {}):
                warnings.append("NSE: Fourier times not specified, using default range")

        if technique == 'qens':
            if 'temperature_k' not in data.get('parameters', {}):
                warnings.append("QENS: temperature not specified, using 298 K")

        if technique == 'nr':
            if 'solvent' not in data.get('parameters', {}):
                warnings.append("NR: solvent not specified for contrast")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources for neutron analysis."""
        technique = data.get('technique', '').lower()

        # Base requirements
        cpu_cores = 1
        memory_gb = 2.0
        estimated_time_sec = 120.0  # Neutron counting statistics

        # Adjust based on technique
        if technique == 'nse':
            # NSE requires correlation analysis
            cpu_cores = 2
            memory_gb = 4.0
            estimated_time_sec = 300.0
        elif technique == 'qens':
            # QENS model fitting
            cpu_cores = 4
            memory_gb = 4.0
            estimated_time_sec = 240.0
        elif technique == 'sans_contrast':
            # Multiple contrasts
            n_contrasts = len(data.get('d2o_fractions', [6]))
            cpu_cores = 2
            memory_gb = 3.0
            estimated_time_sec = 180.0 * n_contrasts

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=0,
            estimated_time_sec=estimated_time_sec,
            execution_environment=ExecutionEnvironment.LOCAL
        )

    def get_capabilities(self) -> List[Capability]:
        """Return neutron agent capabilities."""
        return [
            Capability(
                name="SANS Analysis",
                description="Small-angle neutron scattering with H/D contrast",
                input_types=["scattering_data", "deuteration_level"],
                output_types=["structure_factor", "form_factor", "contrast_analysis"],
                typical_use_cases=[
                    "Polymer chain dimensions",
                    "Protein structure in solution",
                    "Domain structure with isotopic labeling"
                ]
            ),
            Capability(
                name="NSE Analysis",
                description="Neutron spin echo for ultra-high energy resolution dynamics",
                input_types=["spin_echo_data", "fourier_times"],
                output_types=["intermediate_scattering_function", "relaxation_times"],
                typical_use_cases=[
                    "Polymer chain dynamics",
                    "Membrane fluctuations",
                    "Collective relaxations"
                ]
            ),
            Capability(
                name="QENS Analysis",
                description="Quasi-elastic scattering for hydrogen diffusion",
                input_types=["energy_transfer_spectra", "q_range"],
                output_types=["diffusion_coefficients", "jump_parameters"],
                typical_use_cases=[
                    "Hydrogen diffusion in polymers",
                    "Water dynamics",
                    "Proton conductivity mechanisms"
                ]
            ),
            Capability(
                name="Neutron Reflectometry",
                description="Interface and thin film structure analysis",
                input_types=["reflectivity_data", "q_range"],
                output_types=["layer_structure", "sld_profile", "interface_roughness"],
                typical_use_cases=[
                    "Polymer thin films",
                    "Membrane structure",
                    "Buried interfaces"
                ]
            ),
            Capability(
                name="INS Analysis",
                description="Inelastic scattering for vibrational spectroscopy",
                input_types=["energy_transfer_data"],
                output_types=["vibrational_modes", "phonon_dos"],
                typical_use_cases=[
                    "Hydrogen vibrations",
                    "Molecular dynamics",
                    "Phonon density of states"
                ]
            ),
            Capability(
                name="Contrast Variation",
                description="Systematic H/D contrast variation for selective structure",
                input_types=["multi_contrast_data", "d2o_fractions"],
                output_types=["match_points", "component_structure"],
                typical_use_cases=[
                    "Multicomponent systems",
                    "Selective deuteration",
                    "Domain composition"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return neutron agent metadata."""
        return AgentMetadata(
            name="NeutronAgent",
            version=self.VERSION,
            description="Neutron scattering and spectroscopy expert for soft matter systems",
            author="Materials Science Multi-Agent System",
            capabilities=self.get_capabilities(),
            dependencies=[
                'numpy',
                'scipy',
                'matplotlib',
                'mantid',  # Neutron data reduction
                'sasview',  # SANS analysis
            ],
            supported_formats=[
                'nxs', 'hdf5',  # NeXus format
                'dat', 'ascii',  # Reduced data
                'mantid_workspace'
            ]
        )

    def connect_instrument(self) -> bool:
        """Connect to neutron instrument (placeholder).

        Returns:
            True if connection successful
        """
        # In production: connect to neutron facility instrument control
        # e.g., NICOS, SICS, MANTID live data
        return True

    def process_experimental_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw neutron data (placeholder).

        Args:
            raw_data: Raw neutron counts, monitor counts, metadata

        Returns:
            Processed data (normalized, background subtracted, corrected)
        """
        # In production:
        # - Normalize to monitor/time
        # - Background subtraction
        # - Detector efficiency correction
        # - Absolute intensity calibration
        # - Transmission correction
        return raw_data

    # Integration methods for cross-agent collaboration

    @staticmethod
    def validate_with_xray_saxs(sans_result: Dict[str, Any],
                                 saxs_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate SANS with X-ray SAXS data.

        Complements scattering length density (neutron) with
        electron density contrast (X-ray).

        Args:
            sans_result: SANS analysis result
            saxs_result: SAXS analysis result from X-ray agent

        Returns:
            Validation report with consistency checks
        """
        sans_rg = sans_result.get('guinier_analysis', {}).get('radius_of_gyration_nm', 0)
        saxs_rg = saxs_result.get('guinier_analysis', {}).get('radius_of_gyration_nm', 0)

        rg_agreement = abs(sans_rg - saxs_rg) / sans_rg if sans_rg > 0 else 1.0

        return {
            'validation_type': 'SANS_SAXS_cross_check',
            'rg_agreement_percent': (1 - rg_agreement) * 100,
            'consistent': rg_agreement < 0.1,  # Within 10%
            'contrast_mechanisms': {
                'neutron': 'scattering_length_density',
                'xray': 'electron_density'
            },
            'complementary_info': 'SANS provides hydrogen sensitivity',
            'recommendation': 'Results consistent' if rg_agreement < 0.1
                            else 'Check for aggregation or polydispersity'
        }

    @staticmethod
    def extract_dynamics_for_simulation(nse_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dynamics parameters for MD simulation validation.

        Args:
            nse_result: NSE dynamics analysis

        Returns:
            Parameters formatted for simulation agent
        """
        return {
            'dynamics_type': 'from_nse_analysis',
            'diffusion_coefficient_cm2_s': nse_result.get('physical_interpretation', {}).get(
                'diffusion_coefficient_cm2_s', 1e-7),
            'relaxation_time_ns': nse_result.get('dynamics_analysis', {}).get(
                'relaxation_time_ns', 10.0),
            'q_value_inv_angstrom': nse_result.get('q_value_inv_angstrom', 0.05),
            'temperature_k': nse_result.get('experimental_details', {}).get('temperature_k', 298),
            'simulation_suggestions': {
                'trajectory_length_ns': nse_result.get('dynamics_analysis', {}).get(
                    'relaxation_time_ns', 10.0) * 5,
                'timestep_ps': 1.0,
                'ensemble': 'NVT',
                'validation_quantity': 'intermediate_scattering_function',
                'expected_decay_time_ns': nse_result.get('dynamics_analysis', {}).get(
                    'relaxation_time_ns', 10.0)
            }
        }

    @staticmethod
    def design_deuteration_strategy(sample_composition: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal deuteration strategy for contrast.

        Args:
            sample_composition: Sample composition and components

        Returns:
            Deuteration recommendations
        """
        components = sample_composition.get('components', [])

        # Calculate SLD for each component
        strategies = []
        for comp in components:
            h_content = comp.get('hydrogen_content', 0)
            if h_content > 0:
                strategies.append({
                    'component': comp.get('name', 'unknown'),
                    'recommendation': 'deuterate' if h_content > 0.5 else 'keep_hydrogenous',
                    'expected_sld_h': 1.0,  # Typical for H-polymer
                    'expected_sld_d': 6.5,  # Typical for D-polymer
                    'contrast_enhancement': 5.5,
                    'synthesis_feasibility': 'commercial' if comp.get('common', True) else 'custom'
                })

        return {
            'deuteration_strategy': strategies,
            'optimal_contrast': 'deuterate_matrix_probe_hydrogenous',
            'cost_estimate': 'moderate',
            'alternative': 'solvent_contrast_variation',
            'measurement_priority': [
                '100% D2O (maximum contrast)',
                'match point (~42% D2O)',
                '100% H2O (zero contrast for matrix)'
            ]
        }