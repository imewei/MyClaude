"""Tensile Testing Agent - Mechanical Testing Expert.

Capabilities:
- Tensile Testing: Stress-strain curves, Young's modulus, yield/ultimate strength
- Compression Testing: Compression modulus, yield strength, densification
- Flexural (3-point/4-point): Flexural modulus, flexural strength
- Cyclic Loading: Fatigue, hysteresis, Mullins effect
- Strain Rate Effects: Rate-dependent properties
- Multi-axial: Biaxial and planar extension
- Cross-validation: With DMA (E comparison), nanoindentation (local E), DFT (elastic constants)
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


class TensileTestingAgent(ExperimentalAgent):
    """Tensile and mechanical testing characterization agent.

    Supports multiple mechanical test techniques:
    - tensile: Uniaxial tensile testing
    - compression: Uniaxial compression testing
    - flexural_3point: Three-point bending
    - flexural_4point: Four-point bending
    - cyclic: Cyclic loading and fatigue
    - strain_rate_sweep: Rate-dependent properties
    - biaxial: Biaxial extension
    - planar: Planar extension
    """

    NAME = "TensileTestingAgent"
    VERSION = "1.0.0"

    SUPPORTED_TECHNIQUES = [
        'tensile',
        'compression',
        'flexural_3point',
        'flexural_4point',
        'cyclic',
        'strain_rate_sweep',
        'biaxial',
        'planar'
    ]

    # Typical test parameters
    STRAIN_RATE_RANGE = (1e-5, 1.0)  # 1/s (quasi-static to dynamic)
    MAX_STRAIN = 2.0  # 200% (for elastomers)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tensile testing agent.

        Args:
            config: Configuration with instrument settings, calibration, etc.
        """
        super().__init__(config)
        self.instrument_config = config or {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute mechanical testing measurement/analysis.

        Args:
            input_data: Input with keys:
                - technique: str (tensile, compression, flexural_3point, etc.)
                - sample_file: str (path to data file) OR sample_description: dict
                - parameters: dict (technique-specific parameters)
                - mode: str ('measure' or 'analyze', default='analyze')

        Returns:
            AgentResult with mechanical test data and analysis

        Example:
            >>> agent = TensileTestingAgent()
            >>> result = agent.execute({
            ...     'technique': 'tensile',
            ...     'sample_description': {'material': 'polycarbonate', 'geometry': 'dog_bone'},
            ...     'parameters': {
            ...         'strain_rate': 0.01,
            ...         'temperature': 298
            ...     }
            ... })
        """
        start_time = datetime.now()
        technique = input_data.get('technique', 'tensile')

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

            # Route to appropriate technique
            if technique == 'tensile':
                result_data = self._execute_tensile(input_data)
            elif technique == 'compression':
                result_data = self._execute_compression(input_data)
            elif technique == 'flexural_3point':
                result_data = self._execute_flexural_3point(input_data)
            elif technique == 'flexural_4point':
                result_data = self._execute_flexural_4point(input_data)
            elif technique == 'cyclic':
                result_data = self._execute_cyclic(input_data)
            elif technique == 'strain_rate_sweep':
                result_data = self._execute_strain_rate_sweep(input_data)
            elif technique == 'biaxial':
                result_data = self._execute_biaxial(input_data)
            elif technique == 'planar':
                result_data = self._execute_planar(input_data)
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
                    'technique': technique
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
        elif technique not in self.SUPPORTED_TECHNIQUES:
            errors.append(
                f"Unsupported technique: {technique}. "
                f"Supported: {', '.join(self.SUPPORTED_TECHNIQUES)}"
            )

        # Check for data source
        if 'sample_file' not in data and 'sample_description' not in data:
            errors.append("Must provide either 'sample_file' or 'sample_description'")

        # Validate parameters
        params = data.get('parameters', {})

        if 'strain_rate' in params:
            strain_rate = params['strain_rate']
            if strain_rate <= 0:
                errors.append("strain_rate must be positive")
            elif strain_rate < 1e-5 or strain_rate > 10:
                warnings.append(f"Strain rate {strain_rate} 1/s is outside typical range (1e-5 to 1 1/s)")

        if 'temperature' in params:
            temp = params['temperature']
            if temp < 100 or temp > 600:
                warnings.append(f"Temperature {temp} K is outside typical range (100-600 K)")

        if technique == 'cyclic':
            if 'num_cycles' in params and params['num_cycles'] < 1:
                errors.append("num_cycles must be >= 1")

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
        technique = data.get('technique', 'tensile')
        params = data.get('parameters', {})

        # Base resource requirement
        base_time = 600  # 10 minutes

        # Adjust for cyclic tests
        if technique == 'cyclic':
            num_cycles = params.get('num_cycles', 100)
            base_time = 300 + num_cycles * 5  # 5 seconds per cycle

        resource_map = {
            'tensile': ResourceRequirement(
                cpu_cores=1,
                memory_gb=0.5,
                gpu_count=0,
                estimated_time_sec=600,
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'compression': ResourceRequirement(
                cpu_cores=1,
                memory_gb=0.5,
                gpu_count=0,
                estimated_time_sec=600,
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'flexural_3point': ResourceRequirement(
                cpu_cores=1,
                memory_gb=0.5,
                gpu_count=0,
                estimated_time_sec=900,
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'flexural_4point': ResourceRequirement(
                cpu_cores=1,
                memory_gb=0.5,
                gpu_count=0,
                estimated_time_sec=900,
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'cyclic': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=base_time,
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'strain_rate_sweep': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=1800,  # 30 minutes (multiple rates)
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'biaxial': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=1200,
                execution_environment=ExecutionEnvironment.LOCAL
            ),
            'planar': ResourceRequirement(
                cpu_cores=2,
                memory_gb=1.0,
                gpu_count=0,
                estimated_time_sec=1200,
                execution_environment=ExecutionEnvironment.LOCAL
            )
        }

        return resource_map.get(technique, ResourceRequirement(
            cpu_cores=1,
            memory_gb=0.5,
            gpu_count=0,
            estimated_time_sec=600,
            execution_environment=ExecutionEnvironment.LOCAL
        ))

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name="tensile",
                description="Uniaxial tensile testing for stress-strain behavior",
                input_types=["strain_rate", "temperature", "geometry"],
                output_types=["stress_strain_curve", "youngs_modulus", "yield_strength", "ultimate_strength", "toughness"],
                typical_use_cases=[
                    "Young's modulus (E) determination",
                    "Yield and ultimate strength",
                    "Toughness and ductility",
                    "Material selection and QC"
                ]
            ),
            Capability(
                name="compression",
                description="Uniaxial compression testing",
                input_types=["strain_rate", "temperature"],
                output_types=["stress_strain_curve", "compression_modulus", "yield_strength", "densification"],
                typical_use_cases=[
                    "Compression modulus",
                    "Foam characterization",
                    "Densification behavior",
                    "Structural integrity"
                ]
            ),
            Capability(
                name="flexural",
                description="Three-point and four-point bending tests",
                input_types=["span_length", "loading_rate", "geometry"],
                output_types=["flexural_modulus", "flexural_strength", "load_deflection_curve"],
                typical_use_cases=[
                    "Flexural modulus and strength",
                    "Beam stiffness",
                    "Brittle material characterization",
                    "Composite testing"
                ]
            ),
            Capability(
                name="cyclic",
                description="Cyclic loading for fatigue and hysteresis",
                input_types=["strain_amplitude", "num_cycles", "frequency"],
                output_types=["hysteresis_loops", "energy_dissipation", "mullins_effect", "fatigue_life"],
                typical_use_cases=[
                    "Fatigue characterization",
                    "Hysteresis and energy dissipation",
                    "Mullins effect (elastomers)",
                    "Cyclic softening/hardening"
                ]
            ),
            Capability(
                name="strain_rate_sweep",
                description="Rate-dependent mechanical properties",
                input_types=["strain_rate_range"],
                output_types=["modulus_vs_rate", "strength_vs_rate", "rate_sensitivity"],
                typical_use_cases=[
                    "Rate-dependent modulus",
                    "Strain rate sensitivity",
                    "Impact vs. quasi-static",
                    "Viscoelastic characterization"
                ]
            ),
            Capability(
                name="biaxial",
                description="Biaxial extension for multiaxial properties",
                input_types=["stretch_ratio_x", "stretch_ratio_y"],
                output_types=["biaxial_stress", "biaxial_modulus", "anisotropy"],
                typical_use_cases=[
                    "Film and membrane testing",
                    "Anisotropy characterization",
                    "Constitutive modeling",
                    "Equi-biaxial properties"
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata
        """
        return AgentMetadata(
            name=self.NAME,
            version=self.VERSION,
            description="Tensile and mechanical testing characterization expert",
            author="Materials Science Agent System",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dat', 'txt', 'csv', 'xlsx', 'instron']
        )

    # Technique implementations

    def _execute_tensile(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tensile testing.

        Measures stress-strain curve, modulus, strength

        Args:
            input_data: Input with strain_rate, temperature, etc.

        Returns:
            Tensile test results
        """
        params = input_data.get('parameters', {})
        strain_rate = params.get('strain_rate', 0.01)  # 1/s
        temperature = params.get('temperature', 298)  # K
        max_strain = params.get('max_strain', 0.15)  # 15%

        # Simulate elastic-plastic material (e.g., ductile polymer or metal)
        strain = np.linspace(0, max_strain, 200)

        E = params.get('youngs_modulus', 2e9)  # Pa (default: ~2 GPa for polymer)
        yield_strain = 0.02  # 2% yield strain
        yield_stress = E * yield_strain

        # Elastic region + plastic region with strain hardening
        stress = np.where(
            strain <= yield_strain,
            E * strain,  # Elastic (Hooke's law)
            yield_stress + 0.5 * E * (strain - yield_strain)  # Plastic with strain hardening
        )

        # Find ultimate strength
        ultimate_stress = np.max(stress)
        ultimate_strain = strain[np.argmax(stress)]

        # Calculate toughness (area under curve)
        toughness = np.trapezoid(stress, strain)

        result = {
            'technique': 'tensile',
            'strain': strain.tolist(),
            'stress_Pa': stress.tolist(),
            'youngs_modulus_E_Pa': float(E),
            'yield_stress_Pa': float(yield_stress),
            'yield_strain': float(yield_strain),
            'ultimate_stress_Pa': float(ultimate_stress),
            'ultimate_strain': float(ultimate_strain),
            'strain_at_break': float(strain[-1]),
            'toughness_J_per_m3': float(toughness),
            'strain_rate_1_per_s': strain_rate,
            'temperature_K': temperature,
            'analysis': {
                'material_behavior': 'ductile' if ultimate_strain > 0.05 else 'brittle',
                'strain_hardening': 'yes' if ultimate_stress > yield_stress * 1.1 else 'minimal'
            }
        }

        return result

    def _execute_compression(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compression testing.

        Args:
            input_data: Input with strain_rate, temperature

        Returns:
            Compression test results
        """
        params = input_data.get('parameters', {})
        strain_rate = params.get('strain_rate', 0.01)  # 1/s
        temperature = params.get('temperature', 298)  # K
        max_strain = params.get('max_strain', 0.70)  # 70% (for foams)

        # Simulate foam compression (three regions: linear, plateau, densification)
        strain = np.linspace(0, max_strain, 300)

        # Linear elastic region
        E = 1e7  # Pa (foam modulus ~10 MPa)
        linear_strain = 0.05
        plateau_strain = 0.60
        plateau_stress = E * linear_strain

        # Piecewise stress-strain
        stress = np.piecewise(
            strain,
            [strain <= linear_strain,
             (strain > linear_strain) & (strain <= plateau_strain),
             strain > plateau_strain],
            [lambda s: E * s,  # Linear elastic
             plateau_stress,  # Plateau (cell collapse)
             lambda s: plateau_stress + 10 * E * (s - plateau_strain)**2]  # Densification
        )

        result = {
            'technique': 'compression',
            'strain': strain.tolist(),
            'stress_Pa': stress.tolist(),
            'compression_modulus_E_Pa': float(E),
            'plateau_stress_Pa': float(plateau_stress),
            'plateau_strain_range': [float(linear_strain), float(plateau_strain)],
            'densification_strain': float(plateau_strain),
            'energy_absorption_J_per_m3': float(np.trapezoid(stress, strain)),
            'strain_rate_1_per_s': strain_rate,
            'temperature_K': temperature,
            'analysis': {
                'material_type': 'cellular foam',
                'regions': 'linear elastic → plateau → densification'
            }
        }

        return result

    def _execute_flexural_3point(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute three-point bending test.

        Args:
            input_data: Input with span_length, loading_rate, geometry

        Returns:
            Flexural test results
        """
        params = input_data.get('parameters', {})
        span_length = params.get('span_length', 0.05)  # m (50 mm)
        loading_rate = params.get('loading_rate', 0.01)  # m/s
        beam_width = params.get('beam_width', 0.01)  # m
        beam_height = params.get('beam_height', 0.003)  # m
        temperature = params.get('temperature', 298)  # K

        # Simulate flexural behavior
        deflection = np.linspace(0, 0.005, 100)  # m (5 mm max deflection)

        # For 3-point bending: σ_f = 3FL / (2bh²), ε_f = 6Dh / L²
        # F = load, L = span, b = width, h = height, D = deflection

        # Simulate linear elastic beam
        E_flexural = 2.5e9  # Pa (flexural modulus, often > tensile E)
        I = beam_width * beam_height**3 / 12  # Second moment of area

        # Load-deflection: F = 48EI D / L³ (for 3-point bending)
        load = 48 * E_flexural * I * deflection / (span_length ** 3)

        # Flexural stress and strain at outer fiber
        flexural_stress = 3 * load * span_length / (2 * beam_width * beam_height**2)
        flexural_strain = 6 * deflection * beam_height / (span_length ** 2)

        # Find flexural strength (max stress)
        flexural_strength = np.max(flexural_stress)

        result = {
            'technique': 'flexural_3point',
            'deflection_m': deflection.tolist(),
            'load_N': load.tolist(),
            'flexural_stress_Pa': flexural_stress.tolist(),
            'flexural_strain': flexural_strain.tolist(),
            'flexural_modulus_E_Pa': float(E_flexural),
            'flexural_strength_Pa': float(flexural_strength),
            'span_length_m': span_length,
            'beam_geometry': {
                'width_m': beam_width,
                'height_m': beam_height,
                'second_moment_area_m4': float(I)
            },
            'loading_rate_m_per_s': loading_rate,
            'temperature_K': temperature
        }

        return result

    def _execute_flexural_4point(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute four-point bending test.

        Args:
            input_data: Input with span_length, loading_span, geometry

        Returns:
            Flexural test results
        """
        params = input_data.get('parameters', {})
        support_span = params.get('support_span', 0.08)  # m (80 mm)
        loading_span = params.get('loading_span', 0.04)  # m (40 mm, inner span)
        loading_rate = params.get('loading_rate', 0.01)  # m/s
        beam_width = params.get('beam_width', 0.01)  # m
        beam_height = params.get('beam_height', 0.003)  # m
        temperature = params.get('temperature', 298)  # K

        # Simulate 4-point bending
        deflection = np.linspace(0, 0.004, 100)  # m

        E_flexural = 2.5e9  # Pa
        I = beam_width * beam_height**3 / 12

        # For 4-point: more complex formula, but similar approach
        # Advantage: pure bending moment in center region
        load = 32 * E_flexural * I * deflection / (support_span ** 3)

        # Flexural stress (outer fiber in pure bending region)
        M = load * (support_span - loading_span) / 4  # Bending moment
        flexural_stress = M * (beam_height / 2) / I

        flexural_strength = np.max(flexural_stress)

        result = {
            'technique': 'flexural_4point',
            'deflection_m': deflection.tolist(),
            'load_N': load.tolist(),
            'flexural_stress_Pa': flexural_stress.tolist(),
            'flexural_modulus_E_Pa': float(E_flexural),
            'flexural_strength_Pa': float(flexural_strength),
            'support_span_m': support_span,
            'loading_span_m': loading_span,
            'beam_geometry': {
                'width_m': beam_width,
                'height_m': beam_height
            },
            'loading_rate_m_per_s': loading_rate,
            'temperature_K': temperature,
            'notes': 'Pure bending in center region between loading points'
        }

        return result

    def _execute_cyclic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cyclic loading test.

        Args:
            input_data: Input with strain_amplitude, num_cycles, frequency

        Returns:
            Cyclic test results with hysteresis
        """
        params = input_data.get('parameters', {})
        strain_amplitude = params.get('strain_amplitude', 0.05)  # 5%
        num_cycles = params.get('num_cycles', 10)
        frequency = params.get('frequency', 1.0)  # Hz
        temperature = params.get('temperature', 298)  # K

        # Simulate cyclic behavior with Mullins effect (softening)
        points_per_cycle = 100
        total_points = num_cycles * points_per_cycle

        time = np.linspace(0, num_cycles / frequency, total_points)
        strain = strain_amplitude * np.sin(2 * np.pi * frequency * time)

        # Simulate stress with cyclic softening (Mullins effect)
        E_initial = 2e9  # Pa
        stress = np.zeros_like(strain)

        for i, (t, eps) in enumerate(zip(time, strain)):
            cycle_num = int(t * frequency)
            # Softening: modulus decreases with cycle number
            E_cycle = E_initial * (1 - 0.3 * (1 - np.exp(-cycle_num / 3)))
            stress[i] = E_cycle * eps

        # Calculate energy dissipation per cycle (hysteresis area)
        energy_per_cycle = []
        for cycle in range(num_cycles):
            start = cycle * points_per_cycle
            end = (cycle + 1) * points_per_cycle
            energy = np.abs(np.trapezoid(stress[start:end], strain[start:end]))
            energy_per_cycle.append(energy)

        result = {
            'technique': 'cyclic',
            'time_s': time.tolist(),
            'strain': strain.tolist(),
            'stress_Pa': stress.tolist(),
            'num_cycles': num_cycles,
            'strain_amplitude': strain_amplitude,
            'frequency_Hz': frequency,
            'energy_dissipation_per_cycle_J_per_m3': energy_per_cycle,
            'total_energy_dissipation_J_per_m3': float(np.sum(energy_per_cycle)),
            'cyclic_softening': {
                'initial_modulus_Pa': float(E_initial),
                'stabilized_modulus_Pa': float(E_initial * 0.7),
                'softening_percent': 30.0
            },
            'temperature_K': temperature,
            'analysis': {
                'mullins_effect': 'present (stress softening)',
                'hysteresis': 'significant energy dissipation'
            }
        }

        return result

    def _execute_strain_rate_sweep(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strain rate sweep.

        Args:
            input_data: Input with strain_rate_range

        Returns:
            Rate-dependent properties
        """
        params = input_data.get('parameters', {})
        strain_rate_range = params.get('strain_rate_range', [1e-4, 1.0])  # 1/s
        temperature = params.get('temperature', 298)  # K

        # Simulate rate-dependent behavior
        strain_rates = np.logspace(np.log10(strain_rate_range[0]), np.log10(strain_rate_range[1]), 10)

        # Rate-dependent modulus (stiffer at higher rates)
        E_ref = 1e9  # Pa (reference modulus at 0.01 1/s)
        rate_ref = 0.01
        rate_sensitivity = 0.1  # Rate sensitivity parameter

        modulus = E_ref * (strain_rates / rate_ref) ** rate_sensitivity

        # Rate-dependent strength
        strength_ref = 50e6  # Pa (50 MPa)
        strength = strength_ref * (strain_rates / rate_ref) ** (rate_sensitivity * 1.5)

        result = {
            'technique': 'strain_rate_sweep',
            'strain_rate_1_per_s': strain_rates.tolist(),
            'youngs_modulus_Pa': modulus.tolist(),
            'yield_strength_Pa': strength.tolist(),
            'rate_sensitivity_m': float(rate_sensitivity),
            'reference_modulus_Pa': float(E_ref),
            'reference_strain_rate_1_per_s': rate_ref,
            'temperature_K': temperature,
            'analysis': {
                'rate_dependence': 'positive (stiffer at higher rates)',
                'modulus_increase_per_decade': f'{(modulus[-1] / modulus[0]) ** (1 / np.log10(strain_rates[-1] / strain_rates[0])):.2f}x'
            }
        }

        return result

    def _execute_biaxial(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute biaxial extension test.

        Args:
            input_data: Input with stretch ratios

        Returns:
            Biaxial test results
        """
        params = input_data.get('parameters', {})
        stretch_ratio_x = params.get('stretch_ratio_x', [1.0, 2.0])
        stretch_ratio_y = params.get('stretch_ratio_y', [1.0, 2.0])
        temperature = params.get('temperature', 298)  # K

        # Simulate equi-biaxial extension
        lambda_x = np.linspace(stretch_ratio_x[0], stretch_ratio_x[1], 50)
        lambda_y = np.linspace(stretch_ratio_y[0], stretch_ratio_y[1], 50)

        # Neo-Hookean model for rubber elasticity: σ = G(λ² - 1/λ)
        G = 1e6  # Pa (shear modulus)

        stress_x = G * (lambda_x**2 - 1 / lambda_x)
        stress_y = G * (lambda_y**2 - 1 / lambda_y)

        result = {
            'technique': 'biaxial',
            'stretch_ratio_x': lambda_x.tolist(),
            'stretch_ratio_y': lambda_y.tolist(),
            'stress_x_Pa': stress_x.tolist(),
            'stress_y_Pa': stress_y.tolist(),
            'shear_modulus_G_Pa': float(G),
            'biaxial_modulus_Pa': float(2 * G),  # For equi-biaxial
            'temperature_K': temperature,
            'test_mode': 'equi-biaxial',
            'analysis': {
                'constitutive_model': 'Neo-Hookean',
                'anisotropy': 'isotropic (σ_x ≈ σ_y)'
            }
        }

        return result

    def _execute_planar(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planar extension test (pure shear).

        Args:
            input_data: Input with stretch_ratio

        Returns:
            Planar extension results
        """
        params = input_data.get('parameters', {})
        stretch_ratio = params.get('stretch_ratio', [1.0, 2.5])
        temperature = params.get('temperature', 298)  # K

        # Planar extension: λ₁ = λ, λ₂ = 1, λ₃ = 1/λ
        lambda_1 = np.linspace(stretch_ratio[0], stretch_ratio[1], 50)

        # Neo-Hookean: σ = G(λ² - 1/λ²) for planar
        G = 1e6  # Pa
        stress = G * (lambda_1**2 - 1 / lambda_1**2)

        result = {
            'technique': 'planar',
            'stretch_ratio': lambda_1.tolist(),
            'stress_Pa': stress.tolist(),
            'shear_modulus_G_Pa': float(G),
            'temperature_K': temperature,
            'test_mode': 'pure shear (planar extension)',
            'analysis': {
                'constitutive_model': 'Neo-Hookean',
                'deformation': 'λ₁ = λ, λ₂ = 1, λ₃ = 1/λ'
            }
        }

        return result

    # Cross-validation methods

    @staticmethod
    def cross_validate_with_dma(tensile_result: Dict[str, Any],
                                 dma_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate tensile modulus with DMA modulus.

        Args:
            tensile_result: Tensile test result with Young's modulus
            dma_result: DMA result with E'

        Returns:
            Cross-validation results
        """
        tensile_E = tensile_result.get('youngs_modulus_E_Pa', 0)

        # Extract DMA E' (use glassy or high-freq value)
        if dma_result.get('technique') == 'temperature_sweep':
            dma_E = dma_result.get('E_glassy_Pa', 0)
        elif dma_result.get('technique') == 'frequency_sweep':
            E_prime_list = dma_result.get('storage_modulus_E_prime_Pa', [])
            dma_E = E_prime_list[-1] if E_prime_list else 0
        else:
            return {'error': 'DMA result must be temperature_sweep or frequency_sweep'}

        if tensile_E > 0 and dma_E > 0:
            percent_diff = abs(tensile_E - dma_E) / tensile_E * 100
            agreement = percent_diff < 20

            return {
                'tensile_E_Pa': tensile_E,
                'DMA_E_prime_Pa': dma_E,
                'percent_difference': percent_diff,
                'agreement': 'excellent' if percent_diff < 10 else 'good' if percent_diff < 20 else 'poor',
                'notes': f"Tensile E and DMA E' should be similar. {percent_diff:.1f}% difference"
            }
        else:
            return {'error': 'Modulus data missing or invalid'}

    @staticmethod
    def cross_validate_with_nanoindentation(tensile_result: Dict[str, Any],
                                            nanoindent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate bulk modulus with nanoindentation.

        Args:
            tensile_result: Tensile test result
            nanoindent_result: Nanoindentation result

        Returns:
            Cross-validation results
        """
        bulk_E = tensile_result.get('youngs_modulus_E_Pa', 0)
        local_E = nanoindent_result.get('reduced_modulus_Er_Pa', 0)

        # Nanoindentation gives reduced modulus Er
        # Need to correct for indenter (diamond): Er = 1/((1-ν²)/E_sample + (1-ν_i²)/E_i)
        # Approximation: E_sample ≈ Er for stiff indenters

        if bulk_E > 0 and local_E > 0:
            percent_diff = abs(bulk_E - local_E) / bulk_E * 100

            return {
                'bulk_tensile_E_Pa': bulk_E,
                'local_nanoindent_Er_Pa': local_E,
                'percent_difference': percent_diff,
                'agreement': 'good' if percent_diff < 30 else 'poor',
                'notes': f"Nanoindentation measures local E, tensile measures bulk. {percent_diff:.1f}% difference"
            }
        else:
            return {'error': 'Modulus data missing or invalid'}

    @staticmethod
    def cross_validate_with_dft(tensile_result: Dict[str, Any],
                                 dft_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate experimental modulus with DFT elastic constants.

        Args:
            tensile_result: Tensile test result
            dft_result: DFT calculation with elastic constants

        Returns:
            Cross-validation results
        """
        exp_E = tensile_result.get('youngs_modulus_E_Pa', 0)

        # Extract DFT elastic constants
        C11 = dft_result.get('elastic_constant_C11_Pa', 0)
        C12 = dft_result.get('elastic_constant_C12_Pa', 0)

        if exp_E > 0 and C11 > 0:
            # For isotropic: E = (C11 - 2*C12)(C11 + C12) / (C11 + 2*C12)
            if C12 > 0:
                dft_E = (C11 - 2*C12) * (C11 + C12) / (C11 + 2*C12)
            else:
                dft_E = C11  # Simplified

            percent_diff = abs(exp_E - dft_E) / exp_E * 100

            return {
                'experimental_E_Pa': exp_E,
                'DFT_predicted_E_Pa': dft_E,
                'DFT_C11_Pa': C11,
                'DFT_C12_Pa': C12,
                'percent_difference': percent_diff,
                'agreement': 'good' if percent_diff < 30 else 'poor',
                'notes': f"DFT overestimates (0K, perfect crystal). {percent_diff:.1f}% difference"
            }
        else:
            return {'error': 'Modulus or elastic constant data missing'}
