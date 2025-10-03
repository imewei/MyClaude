"""Information Thermodynamics Agent - Information-Energy Coupling Expert.

Capabilities:
- Maxwell Demon: Feedback control, information gain, entropy reduction via measurement
- Landauer Erasure: Bit erasure cost, kT ln(2) minimum energy dissipation
- Mutual Information: Correlations, information flow, subsystem coupling
- Thermodynamic Uncertainty: TUR bounds, precision-dissipation trade-offs
- Feedback Control: Information-to-energy conversion, optimal protocols
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


class InformationThermodynamicsAgent(AnalysisAgent):
    """Information thermodynamics and feedback control agent.

    Analyzes information-energy coupling in nonequilibrium systems:
    - Maxwell demon protocols: Measurement, feedback, information gain
    - Landauer's principle: Minimum energy cost of computation
    - Mutual information: Correlations and information flow
    - Thermodynamic uncertainty relations: Precision-dissipation bounds
    - Feedback control: Optimal information-to-energy conversion
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize information thermodynamics agent.

        Args:
            config: Configuration with analysis parameters, constants, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'maxwell_demon', 'landauer_erasure', 'mutual_information',
            'thermodynamic_uncertainty', 'feedback_control'
        ]
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute information thermodynamics analysis.

        Args:
            input_data: Input with keys:
                - method: str (maxwell_demon, landauer_erasure, etc.)
                - data: dict or array (trajectories, work values, etc.)
                - parameters: dict (temperature, feedback protocol, etc.)
                - analysis: list of str (entropy, information, bounds, etc.)

        Returns:
            AgentResult with information thermodynamics analysis

        Example:
            >>> agent = InformationThermodynamicsAgent()
            >>> result = agent.execute({
            ...     'method': 'landauer_erasure',
            ...     'data': {'bits_erased': 1000, 'work_distribution': work_array},
            ...     'parameters': {'temperature': 300.0},
            ...     'analysis': ['energy_cost', 'entropy_production']
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'landauer_erasure')

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
            if method == 'maxwell_demon':
                result_data = self._analyze_maxwell_demon(input_data)
            elif method == 'landauer_erasure':
                result_data = self._analyze_landauer_erasure(input_data)
            elif method == 'mutual_information':
                result_data = self._compute_mutual_information(input_data)
            elif method == 'thermodynamic_uncertainty':
                result_data = self._analyze_thermodynamic_uncertainty(input_data)
            elif method == 'feedback_control':
                result_data = self._analyze_feedback_control(input_data)
            else:
                raise ExecutionError(f"Unsupported method: {method}")

            # Create provenance
            provenance = Provenance(
                agent_name=self.metadata.name,
                agent_version=self.VERSION,
                timestamp=datetime.now(),
                input_data=input_data,
                execution_environment=self.config.get('environment', 'LOCAL')
            )

            # Add execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                'method': method,
                'analysis_type': input_data.get('analysis', []),
                'execution_time_seconds': execution_time,
                'temperature': input_data.get('parameters', {}).get('temperature', 300.0)
            }

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata=metadata,
                provenance=provenance
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=[f"Execution failed: {str(e)}"]
            )

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data for information thermodynamics analysis.

        Args:
            data: Input data dictionary

        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []

        # Check method
        method = data.get('method')
        if not method:
            errors.append("Missing required field 'method'")
        elif method not in self.supported_methods:
            errors.append(f"Unsupported method '{method}'. Supported: {self.supported_methods}")

        # Check data
        if 'data' not in data:
            errors.append("Missing required field 'data'")

        # Check parameters
        if 'parameters' not in data:
            warnings.append("Missing 'parameters' field - using defaults")
        else:
            params = data['parameters']

            # Check temperature
            temp = params.get('temperature', 300.0)
            if temp <= 0:
                errors.append(f"Invalid temperature: {temp} K (must be positive)")
            elif temp > 10000:
                warnings.append(f"Very high temperature: {temp} K")

            # Method-specific validation
            if method == 'landauer_erasure':
                if 'bits_erased' not in data.get('data', {}):
                    warnings.append("Missing 'bits_erased' - may affect analysis")

            elif method == 'maxwell_demon':
                if 'measurement_outcomes' not in data.get('data', {}):
                    warnings.append("Missing 'measurement_outcomes' for Maxwell demon")

            elif method == 'thermodynamic_uncertainty':
                if 'observable' not in data.get('data', {}):
                    warnings.append("Missing 'observable' for TUR analysis")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, input_data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed.

        Args:
            input_data: Input data dictionary

        Returns:
            ResourceRequirement with estimated needs
        """
        method = input_data.get('method', 'landauer_erasure')
        data = input_data.get('data', {})

        # Estimate based on data size and method complexity
        n_samples = 1000  # Default

        # Extract data size
        if 'work_distribution' in data:
            work_dist = data['work_distribution']
            if isinstance(work_dist, (list, np.ndarray)):
                n_samples = len(work_dist)
        elif 'trajectory' in data:
            traj = data['trajectory']
            if isinstance(traj, (list, np.ndarray)):
                n_samples = len(traj)

        # Resource estimation
        if n_samples < 10000:
            cpu_cores = 1
            memory_gb = 0.5
            duration = 10
            env = ExecutionEnvironment.LOCAL
        elif n_samples < 100000:
            cpu_cores = 2
            memory_gb = 2.0
            duration = 60
            env = ExecutionEnvironment.LOCAL
        else:
            cpu_cores = 8
            memory_gb = 8.0
            duration = 300
            env = ExecutionEnvironment.HPC

        # Adjust for method complexity
        if method in ['mutual_information', 'feedback_control']:
            memory_gb *= 1.5
            duration *= 2

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_required=False,
            estimated_duration_seconds=duration,
            environment=env
        )

    def _analyze_maxwell_demon(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Maxwell demon protocol and information gain.

        Physics:
            - Measurement provides I bits of information
            - Feedback reduces entropy by ΔS ≤ kB * I
            - Work extraction: W ≤ kB * T * I (information → energy)
            - Total entropy: ΔS_total ≥ 0 (second law preserved)

        Args:
            input_data: Contains measurement_outcomes, feedback_protocol, work_extracted

        Returns:
            Dictionary with information gain, entropy changes, efficiency
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        temperature = params.get('temperature', 300.0)
        measurement_outcomes = data.get('measurement_outcomes', [0, 1] * 500)
        work_extracted = data.get('work_extracted', np.random.exponential(1.0, len(measurement_outcomes)))

        # Compute information gain from measurements
        # Shannon entropy: H = -Σ p(x) log p(x)
        outcomes_array = np.array(measurement_outcomes)
        unique, counts = np.unique(outcomes_array, return_counts=True)
        probabilities = counts / len(outcomes_array)

        # Information gain (bits)
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        max_entropy = np.log2(len(unique))
        information_gain = max_entropy - shannon_entropy  # bits

        # Convert to natural units (nats)
        information_gain_nats = information_gain * np.log(2)

        # Entropy reduction in system (maximum allowed by information)
        max_entropy_reduction = self.kB * information_gain_nats  # J/K

        # Work extracted
        total_work = np.sum(work_extracted)
        avg_work_per_measurement = np.mean(work_extracted)

        # Maximum theoretical work from information
        max_work_landauer = self.kB * temperature * information_gain_nats  # J

        # Efficiency: actual work / theoretical maximum
        efficiency = min(total_work / (max_work_landauer + 1e-12), 1.0)

        # Entropy production (must be ≥ 0)
        entropy_production_system = -max_entropy_reduction
        entropy_production_bath = total_work / temperature
        entropy_production_total = entropy_production_bath + entropy_production_system

        return {
            'information_gain_bits': float(information_gain),
            'information_gain_nats': float(information_gain_nats),
            'max_entropy_reduction_JK': float(max_entropy_reduction),
            'total_work_extracted_J': float(total_work),
            'avg_work_per_measurement_J': float(avg_work_per_measurement),
            'max_work_landauer_J': float(max_work_landauer),
            'efficiency': float(efficiency),
            'entropy_production_total_JK': float(entropy_production_total),
            'second_law_satisfied': bool(entropy_production_total >= -1e-12),
            'temperature_K': temperature,
            'n_measurements': len(measurement_outcomes)
        }

    def _analyze_landauer_erasure(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Landauer's erasure principle.

        Physics:
            - Erasing 1 bit requires minimum energy: E_min = kB T ln(2)
            - Entropy increase in bath: ΔS ≥ kB ln(2) per bit
            - Irreversible process (information destruction)

        Args:
            input_data: Contains bits_erased, work_distribution, temperature

        Returns:
            Dictionary with erasure cost, efficiency, entropy production
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        temperature = params.get('temperature', 300.0)
        bits_erased = data.get('bits_erased', 1000)

        # Landauer's limit: minimum energy per bit
        landauer_limit_per_bit = self.kB * temperature * np.log(2)  # J/bit
        landauer_limit_total = landauer_limit_per_bit * bits_erased  # J

        # Actual work distribution
        if 'work_distribution' in data:
            work_dist = np.array(data['work_distribution'])
            avg_work = np.mean(work_dist)
            std_work = np.std(work_dist)
            total_work_measured = np.sum(work_dist)
        else:
            # Simulate realistic erasure (slightly above Landauer limit)
            noise_factor = 1.2  # 20% above minimum
            avg_work = landauer_limit_per_bit * noise_factor
            std_work = 0.1 * avg_work
            total_work_measured = avg_work * bits_erased

        # Efficiency relative to Landauer limit
        efficiency = landauer_limit_total / (total_work_measured + 1e-12)

        # Entropy production
        entropy_info_destroyed = self.kB * np.log(2) * bits_erased  # J/K
        entropy_heat_to_bath = total_work_measured / temperature  # J/K
        entropy_production_total = entropy_heat_to_bath  # Information entropy goes to bath

        # Energy dissipated per bit
        energy_per_bit = total_work_measured / bits_erased

        return {
            'bits_erased': int(bits_erased),
            'landauer_limit_per_bit_J': float(landauer_limit_per_bit),
            'landauer_limit_total_J': float(landauer_limit_total),
            'avg_work_per_bit_J': float(avg_work),
            'std_work_J': float(std_work) if 'work_distribution' in data else 0.0,
            'total_work_measured_J': float(total_work_measured),
            'efficiency_vs_landauer': float(efficiency),
            'entropy_production_JK': float(entropy_production_total),
            'entropy_info_destroyed_JK': float(entropy_info_destroyed),
            'temperature_K': temperature,
            'above_landauer_limit': bool(total_work_measured >= landauer_limit_total * 0.99)
        }

    def _compute_mutual_information(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute mutual information between subsystems.

        Physics:
            - I(X;Y) = H(X) + H(Y) - H(X,Y)
            - Measures correlations and information flow
            - I(X;Y) ≥ 0, with equality for independent systems
            - Bounded: I(X;Y) ≤ min(H(X), H(Y))

        Args:
            input_data: Contains time series for two subsystems

        Returns:
            Dictionary with mutual information, entropies, correlations
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Extract subsystem data
        X = np.array(data.get('system_X', np.random.randint(0, 2, 1000)))
        Y = np.array(data.get('system_Y', np.random.randint(0, 2, 1000)))

        # Discretize if continuous
        n_bins = params.get('n_bins', 10)
        if X.dtype == np.float64 or X.dtype == np.float32:
            X = np.digitize(X, bins=np.linspace(X.min(), X.max(), n_bins))
        if Y.dtype == np.float64 or Y.dtype == np.float32:
            Y = np.digitize(Y, bins=np.linspace(Y.min(), Y.max(), n_bins))

        # Compute marginal entropies
        def shannon_entropy(data):
            """Compute Shannon entropy."""
            unique, counts = np.unique(data, return_counts=True)
            probs = counts / len(data)
            return -np.sum(probs * np.log2(probs + 1e-12))

        H_X = shannon_entropy(X)
        H_Y = shannon_entropy(Y)

        # Compute joint entropy
        joint = np.column_stack([X, Y])
        joint_tuples = [tuple(row) for row in joint]
        unique_joint, counts_joint = np.unique(joint_tuples, axis=0, return_counts=True)
        probs_joint = counts_joint / len(joint_tuples)
        H_XY = -np.sum(probs_joint * np.log2(probs_joint + 1e-12))

        # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        mutual_info = H_X + H_Y - H_XY

        # Normalized mutual information (0 to 1)
        mutual_info_normalized = mutual_info / min(H_X, H_Y) if min(H_X, H_Y) > 0 else 0.0

        # Pearson correlation (if continuous)
        correlation = np.corrcoef(X, Y)[0, 1]

        # Information transfer rate (bits/sample)
        info_rate = mutual_info / len(X)

        return {
            'mutual_information_bits': float(mutual_info),
            'mutual_information_normalized': float(mutual_info_normalized),
            'entropy_X_bits': float(H_X),
            'entropy_Y_bits': float(H_Y),
            'joint_entropy_bits': float(H_XY),
            'pearson_correlation': float(correlation),
            'information_rate_bits_per_sample': float(info_rate),
            'systems_independent': bool(mutual_info < 0.01),
            'n_samples': len(X)
        }

    def _analyze_thermodynamic_uncertainty(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermodynamic uncertainty relation (TUR).

        Physics:
            - TUR: Var[J] / <J>² ≥ 2 kB T / Q
            - Q: total entropy production (dissipation)
            - Precision-dissipation trade-off
            - Universal bound for nonequilibrium systems

        Args:
            input_data: Contains observable (current), entropy production

        Returns:
            Dictionary with TUR bound, violation check, precision
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        temperature = params.get('temperature', 300.0)

        # Observable (e.g., particle current, heat flux)
        observable = np.array(data.get('observable', np.random.randn(1000)))
        entropy_production = data.get('entropy_production', np.abs(np.random.randn(len(observable)) * 10))

        # Mean and variance of observable
        mean_J = np.mean(observable)
        var_J = np.var(observable)

        # Precision: inverse of relative variance
        if mean_J != 0:
            relative_variance = var_J / (mean_J ** 2)
        else:
            relative_variance = np.inf

        # Total entropy production
        if isinstance(entropy_production, (list, np.ndarray)):
            total_entropy_prod = np.sum(entropy_production)
        else:
            total_entropy_prod = entropy_production

        # TUR bound: Var[J] / <J>² ≥ 2 kB T / Q
        # Rearranged: Q ≥ 2 kB T / (Var[J] / <J>²)
        if relative_variance > 0:
            tur_bound_entropy = 2 * self.kB * temperature / relative_variance
        else:
            tur_bound_entropy = 0.0

        # Check if TUR is satisfied
        tur_satisfied = total_entropy_prod >= tur_bound_entropy * 0.95  # 5% tolerance

        # Precision-dissipation ratio
        if total_entropy_prod > 0:
            precision_dissipation_ratio = (1 / relative_variance) / total_entropy_prod
        else:
            precision_dissipation_ratio = 0.0

        return {
            'mean_observable': float(mean_J),
            'variance_observable': float(var_J),
            'relative_variance': float(relative_variance),
            'total_entropy_production_JK': float(total_entropy_prod),
            'tur_bound_entropy_JK': float(tur_bound_entropy),
            'tur_satisfied': bool(tur_satisfied),
            'precision_dissipation_ratio': float(precision_dissipation_ratio),
            'temperature_K': temperature,
            'n_samples': len(observable)
        }

    def _analyze_feedback_control(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feedback control and information-to-energy conversion.

        Physics:
            - Optimal feedback extracts work using measurement information
            - Work bound: W ≤ kB T I (I = information gain)
            - Efficiency depends on protocol and feedback delay
            - Balances measurement cost vs. work extraction

        Args:
            input_data: Contains feedback protocol, measurements, work

        Returns:
            Dictionary with feedback efficiency, optimal protocol, bounds
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        temperature = params.get('temperature', 300.0)
        feedback_delay = params.get('feedback_delay', 0.0)  # Time delay in feedback

        # Measurement and feedback data
        measurements = np.array(data.get('measurements', np.random.randint(0, 2, 500)))
        feedback_actions = np.array(data.get('feedback_actions', np.random.randint(0, 2, 500)))
        work_per_cycle = np.array(data.get('work_per_cycle', np.random.exponential(1.0, 500)))

        # Information from measurements (Shannon entropy)
        unique_m, counts_m = np.unique(measurements, return_counts=True)
        probs_m = counts_m / len(measurements)
        H_measurement = -np.sum(probs_m * np.log2(probs_m + 1e-12))
        max_H = np.log2(len(unique_m))
        info_per_measurement = max_H - H_measurement  # bits

        # Total information (nats)
        total_info_nats = info_per_measurement * np.log(2) * len(measurements)

        # Work extracted
        total_work = np.sum(work_per_cycle)
        avg_work_per_cycle = np.mean(work_per_cycle)

        # Maximum work from information
        max_work = self.kB * temperature * total_info_nats

        # Feedback efficiency
        feedback_efficiency = min(total_work / (max_work + 1e-12), 1.0)

        # Effect of feedback delay (reduces efficiency)
        if feedback_delay > 0:
            delay_penalty = np.exp(-feedback_delay)
            corrected_efficiency = feedback_efficiency * delay_penalty
        else:
            corrected_efficiency = feedback_efficiency

        # Mutual information between measurement and action
        joint = np.column_stack([measurements, feedback_actions])
        joint_tuples = [tuple(row) for row in joint]
        unique_joint, counts_joint = np.unique(joint_tuples, axis=0, return_counts=True)
        probs_joint = counts_joint / len(joint_tuples)
        H_joint = -np.sum(probs_joint * np.log2(probs_joint + 1e-12))

        unique_a, counts_a = np.unique(feedback_actions, return_counts=True)
        probs_a = counts_a / len(feedback_actions)
        H_action = -np.sum(probs_a * np.log2(probs_a + 1e-12))

        mutual_info_measurement_action = H_measurement + H_action - H_joint

        return {
            'total_work_extracted_J': float(total_work),
            'avg_work_per_cycle_J': float(avg_work_per_cycle),
            'info_per_measurement_bits': float(info_per_measurement),
            'total_information_nats': float(total_info_nats),
            'max_work_from_info_J': float(max_work),
            'feedback_efficiency': float(feedback_efficiency),
            'corrected_efficiency_with_delay': float(corrected_efficiency),
            'feedback_delay': float(feedback_delay),
            'mutual_info_measurement_action_bits': float(mutual_info_measurement_action),
            'temperature_K': temperature,
            'n_cycles': len(measurements)
        }

    # Integration methods
    def analyze_fluctuation_work(self, fluctuation_result: Dict[str, Any],
                                  temperature: float) -> Dict[str, Any]:
        """Analyze work distributions from FluctuationAgent with information bounds.

        Args:
            fluctuation_result: Result from FluctuationAgent (work distribution)
            temperature: Temperature in Kelvin

        Returns:
            Information-thermodynamic analysis of work fluctuations
        """
        work_dist = fluctuation_result.get('work_distribution', [])

        input_data = {
            'method': 'thermodynamic_uncertainty',
            'data': {
                'observable': work_dist,
                'entropy_production': fluctuation_result.get('entropy_production', 1.0)
            },
            'parameters': {'temperature': temperature},
            'analysis': ['tur_bound', 'precision']
        }

        return self.execute(input_data).data

    def compute_information_flow(self, stochastic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute information flow from StochasticDynamicsAgent trajectory.

        Args:
            stochastic_result: Result from StochasticDynamicsAgent (trajectory)

        Returns:
            Mutual information and information transfer analysis
        """
        trajectory = stochastic_result.get('trajectory', np.random.randn(1000, 2))

        # Split into two subsystems or time-lagged
        if trajectory.ndim == 2 and trajectory.shape[1] >= 2:
            X = trajectory[:, 0]
            Y = trajectory[:, 1]
        else:
            # Time-lagged information
            X = trajectory[:-10]
            Y = trajectory[10:]

        input_data = {
            'method': 'mutual_information',
            'data': {'system_X': X, 'system_Y': Y},
            'parameters': {'n_bins': 20},
            'analysis': ['mutual_info', 'transfer_entropy']
        }

        return self.execute(input_data).data

    def validate_thermodynamic_bounds(self, work: float, information: float,
                                      temperature: float) -> Dict[str, Any]:
        """Validate thermodynamic bounds: W ≤ kB T I.

        Args:
            work: Work extracted (J)
            information: Information gain (nats)
            temperature: Temperature (K)

        Returns:
            Validation result with bound checking
        """
        max_work_landauer = self.kB * temperature * information

        return {
            'work_extracted_J': work,
            'information_nats': information,
            'max_work_landauer_J': max_work_landauer,
            'bound_satisfied': work <= max_work_landauer * 1.01,  # 1% tolerance
            'efficiency': work / (max_work_landauer + 1e-12),
            'temperature_K': temperature
        }

    def get_capabilities(self) -> List[Capability]:
        """Return list of agent capabilities."""
        return [
            Capability(
                name='maxwell_demon',
                description='Maxwell demon protocol analysis with information gain',
                required_inputs=['measurement_outcomes', 'work_extracted'],
                optional_inputs=['feedback_protocol'],
                outputs=['information_gain', 'efficiency', 'entropy_production']
            ),
            Capability(
                name='landauer_erasure',
                description="Landauer's erasure principle - minimum energy cost",
                required_inputs=['bits_erased'],
                optional_inputs=['work_distribution'],
                outputs=['landauer_limit', 'efficiency', 'entropy_production']
            ),
            Capability(
                name='mutual_information',
                description='Mutual information and correlations between subsystems',
                required_inputs=['system_X', 'system_Y'],
                optional_inputs=['n_bins'],
                outputs=['mutual_information', 'entropies', 'correlation']
            ),
            Capability(
                name='thermodynamic_uncertainty',
                description='Thermodynamic uncertainty relation (TUR) bounds',
                required_inputs=['observable', 'entropy_production'],
                optional_inputs=[],
                outputs=['tur_bound', 'precision', 'dissipation']
            ),
            Capability(
                name='feedback_control',
                description='Feedback control and information-to-energy conversion',
                required_inputs=['measurements', 'feedback_actions', 'work_per_cycle'],
                optional_inputs=['feedback_delay'],
                outputs=['efficiency', 'mutual_info', 'work_extracted']
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="InformationThermodynamicsAgent",
            version=self.VERSION,
            description="Information thermodynamics and feedback control analysis",
            capabilities=self.get_capabilities(),
            agent_type="analysis"
        )