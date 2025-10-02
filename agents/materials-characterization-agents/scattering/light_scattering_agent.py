"""Light Scattering Expert Agent (Refactored).

VERSION 2.0.0 - Raman spectroscopy removed (use SpectroscopyAgent instead).

Capabilities:
- Dynamic Light Scattering (DLS): Particle sizing (1 nm - 10 μm)
- Static Light Scattering (SLS): Molecular weight, radius of gyration
- 3D Cross-Correlation DLS: Turbid samples
- Multi-Speckle DLS: Fast kinetics (milliseconds)

DEPRECATED:
- Raman → Use SpectroscopyAgent (vibrational spectroscopy, not scattering)
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from base_agent import (
    ExperimentalAgent,
    AgentResult,
    AgentStatus,
    ValidationResult,
    ResourceRequirement,
    Capability,
    AgentMetadata,
    Provenance,
    ExecutionEnvironment,
    ValidationError,
    ExecutionError
)


class LightScatteringAgent(ExperimentalAgent):
    """Light scattering characterization agent (refactored).

    VERSION 2.0.0 - Focused on light scattering only (Raman removed).

    Supports light scattering techniques:
    - DLS (Dynamic Light Scattering): Size distribution
    - SLS (Static Light Scattering): Molecular weight, Rg
    - 3D-DLS: Suppress multiple scattering
    - Multi-speckle: Time-resolved kinetics

    DEPRECATED:
    - Raman → Use SpectroscopyAgent (vibrational spectroscopy)
    """

    VERSION = "2.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize light scattering agent.

        Args:
            config: Configuration with instrument settings, calibration, etc.
        """
        super().__init__(config)
        self.supported_techniques = ['DLS', 'SLS', '3D-DLS', 'multi-speckle']

        # Deprecated techniques
        self.deprecated_techniques = {
            'Raman': 'Use SpectroscopyAgent for Raman spectroscopy (vibrational technique, not scattering)',
            'raman': 'Use SpectroscopyAgent for Raman spectroscopy (vibrational technique, not scattering)'
        }

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute light scattering measurement/analysis.

        Args:
            input_data: Input with keys:
                - technique: str (DLS, SLS, Raman, etc.)
                - sample_file: str (path to data file) OR sample_description: dict
                - parameters: dict (technique-specific parameters)
                - mode: str ('measure' or 'analyze', default='analyze')

        Returns:
            AgentResult with scattering data and analysis

        Example:
            >>> agent = LightScatteringAgent()
            >>> result = agent.execute({
            ...     'technique': 'DLS',
            ...     'sample_file': 'polymer_solution.dat',
            ...     'parameters': {'temperature': 298, 'angle': 90}
            ... })
        """
        start_time = datetime.now()
        technique = input_data.get('technique', 'DLS')

        try:
            # Check for deprecated techniques
            if technique in self.deprecated_techniques:
                return AgentResult(
                    agent_name=self.metadata.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=[f"Technique '{technique}' is deprecated. {self.deprecated_techniques[technique]}"],
                    warnings=[]
                )

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

            # Route to appropriate technique
            if technique == 'DLS':
                result_data = self._execute_dls(input_data)
            elif technique == 'SLS':
                result_data = self._execute_sls(input_data)
            elif technique == '3D-DLS':
                result_data = self._execute_3d_dls(input_data)
            elif technique == 'multi-speckle':
                result_data = self._execute_multispeckle(input_data)
            else:
                raise ExecutionError(f"Unsupported technique: {technique}")

            # Create provenance record
            execution_time = (datetime.now() - start_time).total_seconds()
            provenance = Provenance(
                agent_name=self.metadata.name,
                agent_version=self.VERSION,
                timestamp=start_time,
                input_hash=self._compute_cache_key(input_data),
                parameters=input_data.get('parameters', {}),
                execution_time_sec=execution_time,
                environment={
                    'instrument': self.instrument_config.get('model', 'simulated'),
                    'temperature': input_data.get('parameters', {}).get('temperature', 298)
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'technique': technique,
                    'execution_time_sec': execution_time
                },
                warnings=validation.warnings,
                provenance=provenance
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                metadata={'execution_time_sec': execution_time},
                errors=[f"Execution failed: {str(e)}"]
            )

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data.

        Args:
            data: Input data to validate

        Returns:
            ValidationResult with status and errors/warnings
        """
        errors = []
        warnings = []

        # Check required fields
        technique = data.get('technique')
        if not technique:
            errors.append("Missing required field: 'technique'")
        elif technique not in self.supported_techniques:
            errors.append(
                f"Unsupported technique: {technique}. "
                f"Supported: {', '.join(self.supported_techniques)}"
            )

        # Check for data source
        if 'sample_file' not in data and 'sample_description' not in data:
            errors.append("Must provide either 'sample_file' or 'sample_description'")

        # Validate parameters
        params = data.get('parameters', {})
        if technique == 'DLS':
            if 'temperature' in params:
                temp = params['temperature']
                if not (273 <= temp <= 373):
                    warnings.append(f"Temperature {temp}K outside typical range (273-373K)")
            if 'angle' in params:
                angle = params['angle']
                if angle not in [90, 173]:  # Common DLS angles
                    warnings.append(f"Unusual scattering angle: {angle}°")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources.

        Args:
            data: Input data

        Returns:
            ResourceRequirement
        """
        technique = data.get('technique', 'DLS')

        # Light scattering analysis is typically fast and local
        if technique in ['DLS', 'SLS', '3D-DLS']:
            return ResourceRequirement(
                cpu_cores=1,
                memory_gb=0.5,
                gpu_count=0,
                estimated_time_sec=60,  # ~1 minute for typical DLS/SLS
                execution_environment=ExecutionEnvironment.LOCAL
            )
        elif technique == 'multi-speckle':
            return ResourceRequirement(
                cpu_cores=4,
                memory_gb=2.0,
                gpu_count=0,
                estimated_time_sec=30,  # Fast acquisition
                execution_environment=ExecutionEnvironment.LOCAL
            )
        else:
            return ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=120,
                execution_environment=ExecutionEnvironment.LOCAL
            )

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name="DLS",
                description="Dynamic Light Scattering for particle size distribution",
                input_types=["intensity_autocorrelation", "sample_description"],
                output_types=["size_distribution", "hydrodynamic_radius"],
                typical_use_cases=[
                    "Nanoparticle sizing (1-1000 nm)",
                    "Polymer characterization",
                    "Aggregation studies",
                    "Protein oligomerization"
                ]
            ),
            Capability(
                name="SLS",
                description="Static Light Scattering for molecular weight and size",
                input_types=["angular_scattering", "concentration_series"],
                output_types=["molecular_weight", "radius_of_gyration", "second_virial_coefficient"],
                typical_use_cases=[
                    "Molecular weight determination",
                    "Polymer characterization",
                    "Protein interactions"
                ]
            ),
            Capability(
                name="3D-DLS",
                description="3D Cross-Correlation DLS for turbid samples",
                input_types=["cross_correlation_data"],
                output_types=["size_distribution"],
                typical_use_cases=[
                    "Concentrated suspensions",
                    "Turbid samples",
                    "Multiple scattering suppression"
                ]
            ),
            Capability(
                name="multi-speckle",
                description="Multi-speckle DLS for fast kinetics",
                input_types=["speckle_pattern_sequence"],
                output_types=["time_resolved_size", "aggregation_kinetics"],
                typical_use_cases=[
                    "Fast aggregation kinetics",
                    "Time-resolved processes",
                    "Millisecond dynamics"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata
        """
        return AgentMetadata(
            name="LightScatteringAgent",
            version=self.VERSION,
            description="Light scattering and optical characterization expert",
            author="Materials Science Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dat', 'txt', 'csv', 'hdf5']
        )

    # Instrument connection

    def connect_instrument(self) -> bool:
        """Connect to light scattering instrument.

        Returns:
            True if connected, False otherwise
        """
        # In real implementation: connect to instrument via API/driver
        # For now: simulate or use pre-recorded data
        instrument_mode = self.instrument_config.get('mode', 'simulated')
        if instrument_mode == 'simulated':
            return True
        # TODO: Implement real instrument connection
        return False

    def process_experimental_data(self, raw_data: Any) -> Dict[str, Any]:
        """Process raw experimental data.

        Args:
            raw_data: Raw data from instrument

        Returns:
            Processed data dictionary
        """
        # In real implementation: instrument-specific data processing
        # For now: return as-is
        return {'raw': raw_data}

    # Technique implementations

    def _execute_dls(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DLS (Dynamic Light Scattering) analysis.

        Args:
            input_data: Input with sample data and parameters

        Returns:
            DLS results (size distribution, mean size, PDI, etc.)
        """
        params = input_data.get('parameters', {})
        temperature = params.get('temperature', 298)  # K
        viscosity = params.get('viscosity', 0.89e-3)  # Pa·s (water at 25°C)
        angle = params.get('angle', 90)  # degrees

        # For demo: simulate or load real data
        # In production: parse intensity autocorrelation function g2(τ)
        # and perform cumulant or CONTIN analysis

        # Simulated result
        result = {
            'technique': 'DLS',
            'size_distribution': {
                'mean_diameter_nm': 85.3,
                'pdi': 0.12,  # Polydispersity index
                'peak_1': {'diameter_nm': 83.5, 'intensity_percent': 95},
                'peak_2': {'diameter_nm': 250, 'intensity_percent': 5}
            },
            'hydrodynamic_radius_nm': 42.7,
            'count_rate_kcps': 450,
            'temperature_K': temperature,
            'viscosity_Pa_s': viscosity,
            'scattering_angle_deg': angle,
            'quality_metrics': {
                'correlation_function_fit': 0.995,  # R²
                'baseline': 1.002
            }
        }

        return result

    def _execute_sls(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SLS (Static Light Scattering) analysis.

        Args:
            input_data: Input with angular scattering data

        Returns:
            SLS results (Mw, Rg, A2, etc.)
        """
        # In production: Zimm plot or Debye plot analysis
        # I(q) = KC/R(q) = 1/Mw * (1 + q²Rg²/3) + 2A2*C + ...

        result = {
            'technique': 'SLS',
            'molecular_weight_dalton': 1.5e5,
            'radius_of_gyration_nm': 18.5,
            'second_virial_coefficient': 2.3e-4,  # mol·mL/g²
            'zimm_plot': {
                'fit_quality_R2': 0.998
            }
        }

        return result


    def _execute_3d_dls(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 3D Cross-Correlation DLS.

        Suppresses multiple scattering by cross-correlating signals from
        two detectors observing the same scattering volume.

        Args:
            input_data: Input with cross-correlation data

        Returns:
            3D-DLS results (size distribution without multiple scattering)
        """
        result = {
            'technique': '3D-DLS',
            'size_distribution': {
                'mean_diameter_nm': 210,
                'pdi': 0.18
            },
            'multiple_scattering_suppression': True,
            'quality_metrics': {
                'cross_correlation_amplitude': 0.85,
                'intercept': 0.95
            }
        }

        return result

    def _execute_multispeckle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-speckle DLS for fast kinetics.

        Uses camera to capture multiple speckles simultaneously,
        enabling millisecond time resolution.

        Args:
            input_data: Input with speckle pattern sequence

        Returns:
            Time-resolved size/aggregation data
        """
        result = {
            'technique': 'multi-speckle',
            'time_resolved_data': {
                'timestamps_ms': list(range(0, 1000, 10)),  # 0-1000 ms in 10 ms steps
                'mean_diameter_nm': [50 + i * 0.5 for i in range(100)],  # Growing particles
                'aggregation_rate_nm_per_s': 50
            },
            'temporal_resolution_ms': 10,
            'spatial_resolution_pixels': 512
        }

        return result

    # Integration methods for synergy with other agents

    def validate_with_sans_saxs(self, dls_result: Dict[str, Any],
                                 sans_saxs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate DLS results with SANS/SAXS data.

        DLS gives hydrodynamic radius (Rh), SANS/SAXS gives radius of gyration (Rg).
        For spheres: Rg/Rh ≈ 0.775
        For random coils: Rg/Rh ≈ 1.5

        Args:
            dls_result: DLS measurement result
            sans_saxs_data: SANS or SAXS result with Rg

        Returns:
            Cross-validation results with shape factor
        """
        rh = dls_result['hydrodynamic_radius_nm']
        rg = sans_saxs_data.get('radius_of_gyration_nm', 0)

        if rg > 0:
            rho = rg / rh  # Shape factor
            if 0.7 < rho < 0.85:
                shape = 'sphere'
            elif 1.3 < rho < 1.7:
                shape = 'random_coil'
            elif 1.7 < rho < 2.5:
                shape = 'rod'
            else:
                shape = 'complex'

            validation = {
                'consistent': True,
                'Rg_nm': rg,
                'Rh_nm': rh,
                'shape_factor_Rg_Rh': rho,
                'inferred_shape': shape,
                'notes': f'DLS and SANS/SAXS results consistent with {shape} morphology'
            }
        else:
            validation = {
                'consistent': False,
                'error': 'SANS/SAXS data missing Rg'
            }

        return validation

    def compare_with_md_simulation(self, dls_result: Dict[str, Any],
                                   md_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare DLS size with MD simulation prediction.

        Args:
            dls_result: DLS measurement
            md_result: MD simulation result

        Returns:
            Comparison results
        """
        exp_size = dls_result['size_distribution']['mean_diameter_nm']
        sim_size = md_result.get('predicted_size_nm', 0)

        if sim_size > 0:
            percent_diff = abs(exp_size - sim_size) / exp_size * 100
            agreement = percent_diff < 15  # Within 15% considered good

            comparison = {
                'experimental_size_nm': exp_size,
                'simulated_size_nm': sim_size,
                'percent_difference': percent_diff,
                'agreement': 'good' if agreement else 'poor',
                'notes': f'{"Good" if agreement else "Poor"} agreement between DLS and MD ({percent_diff:.1f}% difference)'
            }
        else:
            comparison = {
                'agreement': 'unknown',
                'error': 'MD simulation size not available'
            }

        return comparison