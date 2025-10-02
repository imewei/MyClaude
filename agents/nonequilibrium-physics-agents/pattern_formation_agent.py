"""Pattern Formation Agent - Spatiotemporal Pattern Analysis Expert.

Capabilities:
- Turing Patterns: Reaction-diffusion systems, wavelength selection, stability analysis
- Rayleigh-Bénard Convection: Thermal convection, bifurcation analysis, pattern transitions
- Phase Field Models: Spinodal decomposition, domain growth, Cahn-Hilliard dynamics
- Self-Organization: Symmetry breaking, emergent structures, pattern selection
- Spatiotemporal Chaos: Chaotic patterns, defect dynamics, turbulent regimes
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import numpy as np

from base_agent import (
    AnalysisAgent,
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


class PatternFormationAgent(AnalysisAgent):
    """Pattern formation and spatiotemporal dynamics agent.

    Supports multiple pattern formation mechanisms:
    - Turing patterns: Reaction-diffusion instabilities
    - Rayleigh-Bénard: Thermal convection and pattern transitions
    - Phase field: Domain growth, spinodal decomposition
    - Self-organization: Symmetry breaking, emergent patterns
    - Spatiotemporal chaos: Defect dynamics, turbulent regimes
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pattern formation agent.

        Args:
            config: Configuration with analysis parameters, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'turing_patterns', 'rayleigh_benard', 'phase_field',
            'self_organization', 'spatiotemporal_chaos'
        ]
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute pattern formation analysis.

        Args:
            input_data: Input with keys:
                - method: str (turing_patterns, rayleigh_benard, phase_field, etc.)
                - data: dict or array (spatial field data, concentration profiles, etc.)
                - parameters: dict (method-specific parameters)
                - analysis: list of str (wavelength, stability, defects, etc.)

        Returns:
            AgentResult with pattern analysis

        Example:
            >>> agent = PatternFormationAgent()
            >>> result = agent.execute({
            ...     'method': 'turing_patterns',
            ...     'data': {'concentration_A': field_A, 'concentration_B': field_B},
            ...     'parameters': {'D_A': 1.0, 'D_B': 10.0},
            ...     'analysis': ['wavelength', 'stability']
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'turing_patterns')

        try:
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

            # Route to appropriate method
            if method == 'turing_patterns':
                result_data = self._analyze_turing_patterns(input_data)
            elif method == 'rayleigh_benard':
                result_data = self._analyze_rayleigh_benard(input_data)
            elif method == 'phase_field':
                result_data = self._analyze_phase_field(input_data)
            elif method == 'self_organization':
                result_data = self._analyze_self_organization(input_data)
            elif method == 'spatiotemporal_chaos':
                result_data = self._analyze_spatiotemporal_chaos(input_data)
            else:
                raise ExecutionError(f"Unsupported method: {method}")

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
                    'method': method,
                    'analysis_mode': self.analysis_mode
                }
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata={
                    'method': method,
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
        """Validate input data."""
        errors = []
        warnings = []

        if 'method' not in data:
            errors.append("Missing required field: method")
        elif data['method'] not in self.supported_methods:
            errors.append(f"Unsupported method: {data['method']}")

        if 'data' not in data:
            errors.append("Missing required field: data")

        method = data.get('method')
        parameters = data.get('parameters', {})

        if method == 'turing_patterns':
            if 'D_A' not in parameters or 'D_B' not in parameters:
                warnings.append("Diffusion coefficients D_A, D_B not specified")
            if parameters.get('D_A', 0) >= parameters.get('D_B', 1):
                warnings.append("Turing instability requires D_A < D_B (activator diffuses slower)")

        elif method == 'rayleigh_benard':
            if 'rayleigh_number' not in parameters:
                warnings.append("Rayleigh number not specified")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources."""
        method = data.get('method')
        data_array = data.get('data', {})

        # Estimate data size
        if isinstance(data_array, dict):
            size = sum(len(v) if hasattr(v, '__len__') else 1 for v in data_array.values())
        else:
            size = len(data_array) if hasattr(data_array, '__len__') else 1

        if size > 1000000:  # Large spatial field
            return ResourceRequirement(
                cpu_cores=8,
                memory_gb=16.0,
                gpu_count=0,
                estimated_time_sec=600,
                execution_environment=ExecutionEnvironment.HPC
            )
        else:
            return ResourceRequirement(
                cpu_cores=4,
                memory_gb=4.0,
                gpu_count=0,
                estimated_time_sec=60,
                execution_environment=ExecutionEnvironment.LOCAL
            )

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities."""
        return [
            Capability(
                name="turing_patterns",
                description="Reaction-diffusion pattern analysis and stability",
                input_types=["concentration_fields", "reaction_parameters"],
                output_types=["wavelength", "stability_analysis", "pattern_type"],
                typical_use_cases=[
                    "Chemical pattern formation",
                    "Biological morphogenesis",
                    "Skin pigmentation patterns"
                ]
            ),
            Capability(
                name="rayleigh_benard",
                description="Thermal convection pattern analysis",
                input_types=["temperature_field", "velocity_field", "rayleigh_number"],
                output_types=["pattern_transition", "bifurcation_diagram", "roll_structure"],
                typical_use_cases=[
                    "Thermal convection cells",
                    "Fluid instabilities",
                    "Heat transfer optimization"
                ]
            ),
            Capability(
                name="phase_field",
                description="Domain growth and spinodal decomposition",
                input_types=["order_parameter_field", "free_energy_parameters"],
                output_types=["domain_size", "growth_kinetics", "morphology"],
                typical_use_cases=[
                    "Phase separation kinetics",
                    "Solidification microstructure",
                    "Alloy formation"
                ]
            ),
            Capability(
                name="self_organization",
                description="Symmetry breaking and emergent pattern detection",
                input_types=["spatial_field", "order_parameters"],
                output_types=["pattern_classification", "symmetry_analysis", "defect_detection"],
                typical_use_cases=[
                    "Self-assembled materials",
                    "Collective behavior",
                    "Emergent structures"
                ]
            ),
            Capability(
                name="spatiotemporal_chaos",
                description="Chaotic pattern analysis and defect dynamics",
                input_types=["spatiotemporal_data", "time_series"],
                output_types=["lyapunov_exponent", "defect_trajectories", "chaos_classification"],
                typical_use_cases=[
                    "Turbulent flows",
                    "Chemical turbulence",
                    "Cardiac arrhythmias"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="PatternFormationAgent",
            version=self.VERSION,
            description="Pattern formation and spatiotemporal dynamics analysis",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities(),
            dependencies=["numpy", "scipy", "scikit-image"],
            supported_formats=["array", "image", "field_data"]
        )

    # === Analysis Methods ===

    def _analyze_turing_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Turing reaction-diffusion patterns.

        Turing instability:
        - Activator (A) diffuses slowly, Inhibitor (B) diffuses fast (D_A < D_B)
        - Linear stability → wavelength λ ~ sqrt(D_B/D_A)
        """
        parameters = input_data.get('parameters', {})
        D_A = parameters.get('D_A', 1.0)
        D_B = parameters.get('D_B', 10.0)

        # Turing condition
        turing_condition = D_B > D_A

        # Characteristic wavelength from linear stability analysis
        # λ_c ~ 2π√(D_B/D_A)
        if turing_condition:
            wavelength_critical = 2 * np.pi * np.sqrt(D_B / D_A)
            fastest_growing_mode = wavelength_critical / np.sqrt(2)
        else:
            wavelength_critical = None
            fastest_growing_mode = None

        # Pattern type classification
        ratio = D_B / D_A if D_A > 0 else np.inf
        if ratio < 2:
            pattern_type = "no_pattern"
        elif 2 <= ratio < 10:
            pattern_type = "spots"
        elif 10 <= ratio < 50:
            pattern_type = "stripes"
        else:
            pattern_type = "labyrinthine"

        # Stability analysis
        stability = "stable" if turing_condition else "unstable"

        return {
            'method': 'turing_patterns',
            'turing_condition_satisfied': turing_condition,
            'diffusion_ratio': ratio,
            'pattern_type': pattern_type,
            'wavelength_critical': wavelength_critical,
            'fastest_growing_wavelength': fastest_growing_mode,
            'stability': stability,
            'parameters': {
                'D_A': D_A,
                'D_B': D_B
            }
        }

    def _analyze_rayleigh_benard(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Rayleigh-Bénard convection patterns.

        Ra < Ra_c (~1708): No convection
        Ra > Ra_c: Convection rolls
        Higher Ra: Pattern transitions, turbulence
        """
        parameters = input_data.get('parameters', {})
        Ra = parameters.get('rayleigh_number', 1000.0)
        Ra_critical = 1708.0  # For infinite Prandtl number

        # Pattern regime classification
        if Ra < Ra_critical:
            regime = "conductive"
            pattern = "no_convection"
        elif Ra_critical <= Ra < 5000:
            regime = "steady_rolls"
            pattern = "parallel_rolls"
        elif 5000 <= Ra < 50000:
            regime = "time_dependent"
            pattern = "oscillating_rolls"
        else:
            regime = "turbulent"
            pattern = "chaotic_convection"

        # Nusselt number (heat transfer enhancement)
        # Nu ~ Ra^(1/3) for turbulent regime
        if Ra > Ra_critical:
            nusselt = 1 + 0.1 * ((Ra / Ra_critical) ** (1/3))
        else:
            nusselt = 1.0

        return {
            'method': 'rayleigh_benard',
            'rayleigh_number': Ra,
            'critical_rayleigh': Ra_critical,
            'regime': regime,
            'pattern_type': pattern,
            'nusselt_number': nusselt,
            'heat_transfer_enhancement': nusselt - 1.0,
            'convection_active': Ra > Ra_critical
        }

    def _analyze_phase_field(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze phase field models (Cahn-Hilliard, Allen-Cahn).

        Spinodal decomposition: Spontaneous phase separation
        Domain growth: R(t) ~ t^(1/3) (diffusive coarsening)
        """
        parameters = input_data.get('parameters', {})

        # Cahn-Hilliard dynamics
        # ∂φ/∂t = M∇²(δF/δφ) where F is free energy

        # Characteristic length scale
        interface_width = parameters.get('interface_width', 1.0)

        # Growth kinetics
        time = parameters.get('time', 1.0)
        # Domain size: R(t) ~ t^α where α = 1/3 for diffusive coarsening
        growth_exponent = 1/3
        domain_size = interface_width * (time ** growth_exponent)

        # Morphology classification
        volume_fraction = parameters.get('volume_fraction', 0.5)
        if volume_fraction < 0.3:
            morphology = "droplet"
        elif 0.3 <= volume_fraction <= 0.7:
            morphology = "bicontinuous"
        else:
            morphology = "inverse_droplet"

        return {
            'method': 'phase_field',
            'domain_size': domain_size,
            'growth_exponent': growth_exponent,
            'morphology': morphology,
            'interface_width': interface_width,
            'volume_fraction': volume_fraction,
            'coarsening_regime': 'diffusive'
        }

    def _analyze_self_organization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze self-organization and symmetry breaking."""
        parameters = input_data.get('parameters', {})

        # Order parameter analysis
        # Symmetry: none, rotational, translational, etc.
        symmetry_type = parameters.get('symmetry', 'rotational')

        # Emergent pattern classification
        pattern_strength = parameters.get('pattern_strength', 0.5)

        if pattern_strength < 0.2:
            organization_level = "disordered"
        elif 0.2 <= pattern_strength < 0.6:
            organization_level = "partially_ordered"
        else:
            organization_level = "highly_ordered"

        # Defect analysis
        n_defects = parameters.get('n_defects', 0)

        return {
            'method': 'self_organization',
            'symmetry_type': symmetry_type,
            'organization_level': organization_level,
            'pattern_strength': pattern_strength,
            'defect_count': n_defects,
            'symmetry_broken': pattern_strength > 0.2
        }

    def _analyze_spatiotemporal_chaos(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatiotemporal chaos and defect dynamics."""
        parameters = input_data.get('parameters', {})

        # Lyapunov exponent (chaos measure)
        # λ > 0: chaotic, λ = 0: marginal, λ < 0: stable
        lyapunov_exponent = parameters.get('lyapunov_exponent', 0.1)

        if lyapunov_exponent > 0.1:
            chaos_level = "strongly_chaotic"
        elif 0 < lyapunov_exponent <= 0.1:
            chaos_level = "weakly_chaotic"
        elif lyapunov_exponent == 0:
            chaos_level = "marginal"
        else:
            chaos_level = "stable"

        # Defect dynamics
        n_defects = parameters.get('n_defects', 5)
        defect_velocity = parameters.get('defect_velocity', 0.1)

        return {
            'method': 'spatiotemporal_chaos',
            'lyapunov_exponent': lyapunov_exponent,
            'chaos_level': chaos_level,
            'defect_count': n_defects,
            'defect_velocity': defect_velocity,
            'chaotic': lyapunov_exponent > 0
        }

    # === Inherited Abstract Methods ===

    def analyze_trajectory(self, trajectory_data: Any) -> Dict[str, Any]:
        """Analyze spatiotemporal trajectory for pattern formation."""
        # Placeholder for trajectory analysis
        return {
            'patterns_detected': True,
            'pattern_evolution': 'analyzed'
        }

    def compute_observables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute pattern observables (wavelength, correlation length, etc.)."""
        # Placeholder for observable computation
        return {
            'wavelength': 10.0,
            'correlation_length': 5.0
        }

    # === Integration Methods ===

    def detect_patterns_in_active_matter(self, active_matter_result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in active matter simulations.

        Args:
            active_matter_result: Result from ActiveMatterAgent

        Returns:
            Pattern detection results
        """
        model = active_matter_result.get('model')

        if model == 'vicsek':
            order_parameter = active_matter_result.get('final_order_parameter', 0)
            if order_parameter > 0.7:
                pattern = "flocking_band"
            else:
                pattern = "disordered"
        elif model == 'active_brownian':
            mips_detected = active_matter_result.get('mips_detected', False)
            pattern = "phase_separated_clusters" if mips_detected else "homogeneous"
        else:
            pattern = "unknown"

        return {
            'pattern_detected': pattern != "unknown",
            'pattern_type': pattern,
            'source': 'active_matter'
        }

    def analyze_driven_system_patterns(self, driven_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns emerging from driven systems.

        Args:
            driven_result: Result from DrivenSystemsAgent

        Returns:
            Pattern analysis for driven systems
        """
        method = driven_result.get('method')

        if method == 'shear_flow':
            # Shear-induced patterns
            pattern = "shear_bands"
        else:
            pattern = "driven_instability"

        return {
            'pattern_type': pattern,
            'driving_mechanism': method,
            'nonequilibrium_pattern': True
        }