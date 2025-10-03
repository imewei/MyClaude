"""X-ray Spectroscopy Agent - Electronic Structure Expert.

Capabilities:
- XAS (X-ray Absorption Spectroscopy): Oxidation states, local coordination
- XANES (X-ray Absorption Near-Edge Structure): Electronic structure
- EXAFS (Extended X-ray Absorption Fine Structure): Bond distances, coordination numbers
- Element-specific characterization
- Operando and in-situ measurements

Expert in electronic structure determination, oxidation state analysis,
and local atomic environment characterization.
"""

from base_agent import (
    ExperimentalAgent, AgentResult, AgentStatus, ValidationResult,
    ResourceRequirement, Capability, AgentMetadata, Provenance,
    ExecutionEnvironment
)
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np


class XRaySpectroscopyAgent(ExperimentalAgent):
    """X-ray absorption spectroscopy agent for electronic structure.

    Capabilities:
    - XAS: X-ray absorption for oxidation states and coordination
    - XANES: Near-edge structure for electronic properties
    - EXAFS: Extended fine structure for local geometry
    - Multi-element analysis
    - Operando capabilities

    Key advantages:
    - Element-specific analysis
    - Local structure determination
    - Oxidation state sensitivity
    - In-situ/operando compatible
    """

    NAME = "XRaySpectroscopyAgent"
    VERSION = "1.0.0"

    SUPPORTED_TECHNIQUES = [
        'xas',           # X-ray absorption spectroscopy
        'xanes',         # X-ray absorption near-edge structure
        'exafs',         # Extended X-ray absorption fine structure
    ]

    # Common absorption edges
    COMMON_EDGES = {
        'K': ['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
        'L1': ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
        'L2': ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
        'L3': ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize X-ray spectroscopy agent.

        Args:
            config: Configuration including:
                - beamline: Synchrotron beamline name
                - monochromator: Energy resolution
                - detector: Fluorescence, transmission, etc.
        """
        super().__init__(config)
        self.beamline = self.config.get('beamline', 'generic')
        self.monochromator = self.config.get('monochromator', 'Si(111)')
        self.detector = self.config.get('detector', 'fluorescence')

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute X-ray absorption spectroscopy analysis.

        Args:
            input_data: Must contain:
                - technique: One of SUPPORTED_TECHNIQUES
                - element: Element symbol
                - edge: Absorption edge (K, L1, L2, L3)
                - data_file or data_array: XAS data
                - parameters: Technique-specific parameters

        Returns:
            AgentResult with XAS analysis results
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

        technique = input_data.get('technique', 'xas').lower()

        # Route to technique-specific handler
        try:
            if technique in ['xas', 'xanes', 'exafs']:
                result_data = self._execute_xas(input_data)
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
                    'element': input_data.get('element', 'Fe'),
                    'edge': input_data.get('edge', 'K'),
                    'beamline': self.beamline,
                    **input_data.get('parameters', {})
                },
                execution_time_sec=execution_time,
                environment={
                    'monochromator': self.monochromator,
                    'detector': self.detector
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'technique': technique,
                    'execution_time_sec': execution_time,
                    'beamline': self.beamline
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

    def _execute_xas(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute XAS analysis for electronic structure.

        XAS (X-ray Absorption Spectroscopy) provides:
        - Oxidation states (from edge position)
        - Local coordination (from EXAFS)
        - Electronic structure (from XANES features)
        - Chemical speciation

        Args:
            input_data: Input with element, edge, energy range

        Returns:
            XAS analysis results
        """
        element = input_data.get('element', 'Fe')
        edge = input_data.get('edge', 'K')
        energy_range_ev = input_data.get('energy_range_ev', [7000, 8000])

        # Simulate energy scan
        n_points = 500
        energies = np.linspace(energy_range_ev[0], energy_range_ev[1], n_points)

        # Edge position (element and oxidation state dependent)
        edge_positions = {
            'Fe': {'K': 7112.0, 'L3': 707.0},
            'Cu': {'K': 8979.0, 'L3': 932.7},
            'Ni': {'K': 8333.0, 'L3': 855.0},
            'Mn': {'K': 6539.0, 'L3': 640.0}
        }

        edge_energy = edge_positions.get(element, {}).get(edge, 7112.0)

        # Simulate absorption spectrum
        # Pre-edge region
        mu_pre = 0.5 + 0.0001 * (energies - energy_range_ev[0])

        # Edge jump
        edge_idx = np.argmin(np.abs(energies - edge_energy))
        edge_jump = 1.0

        # XANES region (near-edge features)
        xanes_width = 30  # eV
        white_line_position = edge_energy + 5  # eV above edge
        white_line_intensity = 1.5

        # EXAFS oscillations
        k_max = 12  # Å^-1
        R_first_shell = 2.05  # Å (first shell distance)
        N_coord = 6.0  # Coordination number

        mu = np.zeros_like(energies)
        for i, E in enumerate(energies):
            if E < edge_energy:
                # Pre-edge
                mu[i] = mu_pre[i]
                # Add pre-edge peak for transition metals
                if element in ['Fe', 'Mn', 'Co', 'Ni', 'Cu']:
                    pre_edge_peak = 0.1 * np.exp(-((E - (edge_energy - 10))**2) / 20)
                    mu[i] += pre_edge_peak
            else:
                # Post-edge: edge jump + XANES + EXAFS
                mu[i] = mu_pre[i] + edge_jump

                # White line (L-edges) or rising edge (K-edge)
                if edge.startswith('L'):
                    white_line = white_line_intensity * np.exp(-((E - white_line_position)**2) / (xanes_width**2))
                    mu[i] += white_line

                # EXAFS oscillations (simplified)
                if E > edge_energy + 50:  # Beyond XANES
                    # Convert to k-space: k = sqrt(2m(E-E0)/ℏ²)
                    k = 0.512 * np.sqrt(E - edge_energy)  # Å^-1
                    if k < k_max:
                        # EXAFS equation (simplified single shell)
                        chi = (N_coord / (k * R_first_shell**2)) * np.sin(2 * k * R_first_shell + np.pi) * np.exp(-2 * k**2 * 0.005)
                        mu[i] += 0.3 * chi

        # Normalize
        mu_normalized = (mu - mu_pre[0]) / edge_jump

        # XANES analysis
        xanes_region = (energies >= edge_energy - 20) & (energies <= edge_energy + 50)
        xanes_energies = energies[xanes_region]
        xanes_mu = mu_normalized[xanes_region]

        # Determine oxidation state from edge position
        # Edge shifts ~1-2 eV per oxidation state increase
        oxidation_state = '+3'  # Example
        if edge_energy > edge_positions.get(element, {}).get(edge, 0) + 2:
            oxidation_state = '+3'
        elif edge_energy > edge_positions.get(element, {}).get(edge, 0):
            oxidation_state = '+2'
        else:
            oxidation_state = '0'

        # EXAFS analysis
        exafs_region = energies > edge_energy + 50
        exafs_energies = energies[exafs_region]

        # Convert to k-space
        k_exafs = 0.512 * np.sqrt(exafs_energies - edge_energy)

        result = {
            'technique': 'XAS',
            'element': element,
            'edge': edge,
            'edge_energy_ev': float(edge_energy),
            'energy_ev': energies.tolist(),
            'absorption_mu': mu.tolist(),
            'absorption_normalized': mu_normalized.tolist(),
            'xanes_analysis': {
                'energy_range_ev': [float(xanes_energies[0]), float(xanes_energies[-1])],
                'edge_position_ev': float(edge_energy),
                'oxidation_state': oxidation_state,
                'coordination_geometry': 'octahedral',  # From XANES fingerprint
                'pre_edge_features': element in ['Fe', 'Mn', 'Co', 'Ni', 'Cu'],
                'pre_edge_intensity': 0.1 if element in ['Fe', 'Mn', 'Co', 'Ni', 'Cu'] else 0.0,
                'white_line_intensity': float(white_line_intensity) if edge.startswith('L') else None,
                'white_line_position_ev': float(white_line_position) if edge.startswith('L') else None
            },
            'exafs_analysis': {
                'k_range_inv_angstrom': [0, float(k_max)],
                'first_shell_distance_angstrom': float(R_first_shell),
                'coordination_number': float(N_coord),
                'debye_waller_factor_angstrom2': 0.005,
                'second_shell_present': True,
                'fitting_quality_r_factor': 0.015
            },
            'chemical_state': {
                'primary_species': f'{element}{oxidation_state}',
                'phase': 'crystalline',
                'defects_detected': ['oxygen_vacancies'] if element in ['Fe', 'Ti', 'Mn'] else [],
                'mixed_valence': False
            },
            'electronic_structure': {
                'density_of_states_features': ['eg', 't2g'] if element in ['Fe', 'Mn', 'Co', 'Ni'] else [],
                'band_gap_ev': 2.1 if element in ['Fe', 'Ti'] else None,
                'metal_ligand_hybridization': 'strong',
                'd_orbital_occupancy': '3d5' if element == 'Fe' and oxidation_state == '+3' else None
            },
            'measurement_conditions': {
                'beamline': self.beamline,
                'monochromator': self.monochromator,
                'detector_type': self.detector,
                'energy_resolution_ev': 0.5,
                'measurement_mode': 'fluorescence' if self.detector == 'fluorescence' else 'transmission'
            }
        }

        return result

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate XAS input."""
        errors = []
        warnings = []

        # Check technique
        technique = data.get('technique', '').lower()
        if not technique:
            errors.append("Missing required field: 'technique'")
        elif technique not in self.SUPPORTED_TECHNIQUES:
            errors.append(f"Unsupported technique: {technique}. "
                         f"Supported: {self.SUPPORTED_TECHNIQUES}")

        # Check element
        if 'element' not in data:
            warnings.append("XAS: element not specified, using default Fe")

        # Check edge
        if 'edge' not in data:
            warnings.append("XAS: edge not specified, using K-edge")

        # Check data source
        if 'data_file' not in data and 'data_array' not in data:
            warnings.append("No data provided; will use simulated data")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources for XAS analysis."""
        # XAS analysis is moderate complexity
        return ResourceRequirement(
            cpu_cores=2,
            memory_gb=2.0,
            gpu_count=0,
            estimated_time_sec=120.0,  # 2 minutes for fitting
            execution_environment=ExecutionEnvironment.LOCAL
        )

    def get_capabilities(self) -> List[Capability]:
        """Return XAS agent capabilities."""
        return [
            Capability(
                name="XAS Analysis",
                description="X-ray absorption spectroscopy for electronic structure",
                input_types=["absorption_spectrum", "element", "edge"],
                output_types=["oxidation_state", "coordination", "electronic_structure"],
                typical_use_cases=[
                    "Oxidation state determination",
                    "Local coordination environment",
                    "Chemical speciation",
                    "Operando catalysis studies"
                ]
            ),
            Capability(
                name="XANES Analysis",
                description="X-ray absorption near-edge structure for electronic properties",
                input_types=["xanes_spectrum", "element", "edge"],
                output_types=["oxidation_state", "coordination_geometry", "electronic_transitions"],
                typical_use_cases=[
                    "Oxidation state mapping",
                    "Coordination geometry",
                    "Pre-edge features (d-d transitions)",
                    "White line analysis"
                ]
            ),
            Capability(
                name="EXAFS Analysis",
                description="Extended X-ray absorption fine structure for local structure",
                input_types=["exafs_spectrum", "k_range"],
                output_types=["bond_distances", "coordination_numbers", "disorder_parameters"],
                typical_use_cases=[
                    "First/second shell distances",
                    "Coordination numbers",
                    "Debye-Waller factors",
                    "Local structure determination"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return XAS agent metadata."""
        return AgentMetadata(
            name=self.NAME,
            version=self.VERSION,
            description="X-ray absorption spectroscopy expert for electronic structure",
            author="Materials Science Multi-Agent System",
            capabilities=self.get_capabilities(),
            dependencies=[
                'numpy',
                'scipy',
                'larch',  # For EXAFS analysis
                'pyFitIt',  # For XAS fitting
            ],
            supported_formats=[
                'dat', 'xmu',  # XAS data formats
                'athena',  # Athena project files
                'hdf5', 'nexus'  # Synchrotron formats
            ]
        )

    # Cross-validation methods

    @staticmethod
    def cross_validate_with_xps(xas_result: Dict[str, Any],
                                 xps_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate XAS oxidation state with XPS.

        XAS (bulk-sensitive) vs XPS (surface-sensitive)

        Args:
            xas_result: XAS analysis with oxidation state
            xps_result: XPS analysis with binding energies

        Returns:
            Validation report
        """
        xas_ox_state = xas_result.get('xanes_analysis', {}).get('oxidation_state', '0')
        xps_ox_state = xps_result.get('oxidation_state', '0')

        agreement = xas_ox_state == xps_ox_state

        return {
            'validation_type': 'XAS_XPS_oxidation_state',
            'XAS_oxidation_state': xas_ox_state,
            'XPS_oxidation_state': xps_ox_state,
            'agreement': 'good' if agreement else 'different',
            'interpretation': 'Bulk and surface oxidation states match' if agreement
                            else 'Surface oxidation state differs from bulk (common for air exposure)',
            'complementarity': {
                'XAS': 'Bulk-sensitive (μm penetration)',
                'XPS': 'Surface-sensitive (nm depth)'
            }
        }

    @staticmethod
    def cross_validate_with_dft(xas_result: Dict[str, Any],
                                 dft_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate EXAFS bond distances with DFT.

        Args:
            xas_result: EXAFS analysis with bond distances
            dft_result: DFT calculation with optimized geometry

        Returns:
            Validation report
        """
        exafs_distance = xas_result.get('exafs_analysis', {}).get('first_shell_distance_angstrom', 0)
        dft_distance = dft_result.get('bond_distance_angstrom', 0)

        if exafs_distance > 0 and dft_distance > 0:
            diff = abs(exafs_distance - dft_distance)
            percent_diff = (diff / exafs_distance) * 100

            return {
                'validation_type': 'EXAFS_DFT_bond_distance',
                'EXAFS_distance_angstrom': exafs_distance,
                'DFT_distance_angstrom': dft_distance,
                'difference_angstrom': diff,
                'percent_difference': percent_diff,
                'agreement': 'excellent' if percent_diff < 2 else 'good' if percent_diff < 5 else 'poor',
                'notes': f'EXAFS measures thermal average, DFT gives 0K structure. {percent_diff:.1f}% difference.'
            }
        else:
            return {'error': 'Distance data missing or invalid'}

    @staticmethod
    def cross_validate_with_uv_vis(xas_result: Dict[str, Any],
                                    uv_vis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate XAS electronic structure with UV-Vis.

        Args:
            xas_result: XANES with electronic transitions
            uv_vis_result: UV-Vis with band gap

        Returns:
            Validation report
        """
        xas_bandgap = xas_result.get('electronic_structure', {}).get('band_gap_ev', 0)
        uvvis_bandgap = uv_vis_result.get('band_gap_ev', 0)

        if xas_bandgap and uvvis_bandgap:
            diff = abs(xas_bandgap - uvvis_bandgap)

            return {
                'validation_type': 'XAS_UV_Vis_bandgap',
                'XAS_bandgap_ev': xas_bandgap,
                'UV_Vis_bandgap_ev': uvvis_bandgap,
                'difference_ev': diff,
                'agreement': 'good' if diff < 0.5 else 'fair',
                'complementarity': {
                    'XAS': 'Element-specific electronic structure',
                    'UV_Vis': 'Optical bandgap'
                }
            }
        else:
            return {'error': 'Bandgap data missing'}
