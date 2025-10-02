"""Large Deviation Theory Agent - Rare Events and Transition Path Analysis Expert.

Capabilities:
- Rare Event Sampling: Importance sampling, cloning algorithms for extreme fluctuations
- Transition Path Sampling: TPS, committor analysis, reactive flux calculations
- Dynamical Phase Transitions: s-ensemble simulations, critical points
- Rate Function Calculation: Level 2.5 large deviations, Legendre transforms
- s-Ensemble Simulation: Biased ensemble generation and reweighting
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

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


class LargeDeviationAgent(AnalysisAgent):
    """Large deviation theory and rare event analysis agent.

    Analyzes rare fluctuations and transition paths:
    - Rare event sampling via importance sampling and cloning
    - Transition path sampling (TPS) for reactive trajectories
    - Dynamical phase transitions in biased ensembles
    - Rate function calculations (Cramér, Gartner-Ellis)
    - s-ensemble simulations for atypical trajectories
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize large deviation agent.

        Args:
            config: Configuration with sampling parameters, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'rare_event_sampling', 'transition_path_sampling',
            'dynamical_phase_transition', 'rate_function_calculation',
            's_ensemble_simulation'
        ]
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute large deviation analysis.

        Args:
            input_data: Input with keys:
                - method: str (rare_event_sampling, etc.)
                - data: dict or array (trajectories, observables, etc.)
                - parameters: dict (bias parameter s, temperature, etc.)
                - analysis: list of str (rate_function, committor, etc.)

        Returns:
            AgentResult with large deviation analysis

        Example:
            >>> agent = LargeDeviationAgent()
            >>> result = agent.execute({
            ...     'method': 'rare_event_sampling',
            ...     'data': {'trajectories': trajectories, 'observable': A_values},
            ...     'parameters': {'bias_parameter': 2.0, 'target_activity': 10.0},
            ...     'analysis': ['rate_function', 'reweighting']
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'rare_event_sampling')

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
            if method == 'rare_event_sampling':
                result_data = self._rare_event_sampling(input_data)
            elif method == 'transition_path_sampling':
                result_data = self._transition_path_sampling(input_data)
            elif method == 'dynamical_phase_transition':
                result_data = self._dynamical_phase_transition(input_data)
            elif method == 'rate_function_calculation':
                result_data = self._rate_function_calculation(input_data)
            elif method == 's_ensemble_simulation':
                result_data = self._s_ensemble_simulation(input_data)
            else:
                raise ExecutionError(f"Unsupported method: {method}")

            # Create provenance
            provenance = Provenance(
                agent_name=self.metadata.name,
                agent_version=self.VERSION,
                timestamp=datetime.now(),
                input_hash=self._compute_input_hash(input_data),
                parameters=input_data.get('parameters', {}),
                execution_time_sec=(datetime.now() - start_time).total_seconds()
            )

            # Add execution metadata
            metadata = {
                'method': method,
                'analysis_type': input_data.get('analysis', []),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
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
        """Validate input data for large deviation analysis.

        Args:
            data: Input data dictionary

        Returns:
            ValidationResult with validation status
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
        else:
            data_dict = data['data']

            # Method-specific validation
            if method == 'rare_event_sampling':
                if 'observable' not in data_dict and 'trajectories' not in data_dict:
                    warnings.append("Missing 'observable' or 'trajectories' for rare event sampling")

            elif method == 'transition_path_sampling':
                if 'trajectory' not in data_dict:
                    warnings.append("Missing 'trajectory' for TPS")

            elif method == 'dynamical_phase_transition':
                if 's_values' not in data_dict and 'observable' not in data_dict:
                    warnings.append("Missing 's_values' or 'observable' for DPT analysis")

        # Check parameters
        if 'parameters' not in data:
            warnings.append("Missing 'parameters' field - using defaults")
        else:
            params = data['parameters']

            # Check bias parameter for s-ensemble
            if 'bias_parameter' in params:
                s = params['bias_parameter']
                if abs(s) > 10:
                    warnings.append(f"Large bias parameter |s| = {abs(s)} may cause numerical issues")

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
        method = input_data.get('method', 'rare_event_sampling')
        data = input_data.get('data', {})

        # Estimate based on data size and method complexity
        n_trajectories = 1000  # Default
        trajectory_length = 1000  # Default

        # Extract data size
        if 'trajectories' in data:
            trajs = data['trajectories']
            if isinstance(trajs, list):
                n_trajectories = len(trajs)
                if len(trajs) > 0 and isinstance(trajs[0], (list, np.ndarray)):
                    trajectory_length = len(trajs[0])
        elif 'observable' in data:
            obs = data['observable']
            if isinstance(obs, (list, np.ndarray)):
                n_trajectories = len(obs)

        # Resource estimation
        total_samples = n_trajectories * trajectory_length

        if total_samples < 100000:
            cpu_cores = 2
            memory_gb = 1.0
            duration = 60
            env = ExecutionEnvironment.LOCAL
        elif total_samples < 1000000:
            cpu_cores = 4
            memory_gb = 4.0
            duration = 300
            env = ExecutionEnvironment.LOCAL
        else:
            cpu_cores = 16
            memory_gb = 16.0
            duration = 1800
            env = ExecutionEnvironment.HPC

        # Adjust for method complexity
        if method in ['transition_path_sampling', 'dynamical_phase_transition']:
            memory_gb *= 1.5
            duration *= 2

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_required=False,
            estimated_duration_seconds=duration,
            environment=env
        )

    def _rare_event_sampling(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform rare event sampling via importance sampling or cloning.

        Physics:
            - Large deviation function: I(a) = lim_{T→∞} -1/T * ln P(A_T ≈ a)
            - Importance sampling: Sample from biased distribution P_bias ∝ exp(-s*A)
            - Reweight: ⟨f⟩ = ⟨f * w⟩_bias where w = exp(s*A) / Z_s
            - Cloning algorithm: Duplicate high-activity, kill low-activity trajectories

        Args:
            input_data: Contains observable, trajectories, bias_parameter

        Returns:
            Dictionary with rate function, reweighted observables, rare event probabilities
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Extract observable values
        if 'observable' in data:
            observable = np.array(data['observable'])
        elif 'trajectories' in data:
            # Compute observable from trajectories
            trajectories = data['trajectories']
            observable = np.array([np.mean(traj) for traj in trajectories])
        else:
            observable = np.random.randn(1000)  # Fallback

        # Bias parameter for importance sampling
        s = params.get('bias_parameter', 1.0)
        target_activity = params.get('target_activity', np.mean(observable) + 2*np.std(observable))

        # Importance sampling weights
        # w(x) = exp(s * A(x))
        weights = np.exp(s * observable)
        weights_normalized = weights / np.sum(weights)

        # Estimate scaled cumulant generating function (SCGF)
        # θ(s) = ln⟨exp(s*A)⟩
        theta_s = logsumexp(s * observable) - np.log(len(observable))

        # Rate function via Legendre transform
        # I(a) = sup_s [s*a - θ(s)]
        a_values = np.linspace(observable.min(), observable.max(), 50)
        rate_function = np.zeros(len(a_values))

        for i, a in enumerate(a_values):
            # Legendre transform at this a
            def objective(s_opt):
                theta_opt = logsumexp(s_opt * observable) - np.log(len(observable))
                return -(s_opt * a - theta_opt)  # Negative for minimization

            result = minimize(objective, x0=0.0, method='BFGS')
            rate_function[i] = -result.fun if result.success else 0.0

        # Probability of rare event (a > target)
        # P(A > a*) ≈ exp(-T * I(a*))
        idx_target = np.argmin(np.abs(a_values - target_activity))
        I_target = rate_function[idx_target]

        # Effective sample size (ESS) for importance sampling
        ess = (np.sum(weights)**2) / np.sum(weights**2)
        ess_fraction = ess / len(observable)

        # Rare event statistics
        rare_events = observable > target_activity
        n_rare = np.sum(rare_events)
        rare_fraction = n_rare / len(observable)

        return {
            'bias_parameter_s': float(s),
            'scaled_cumulant_generating_function': float(theta_s),
            'a_values': a_values.tolist(),
            'rate_function_I': rate_function.tolist(),
            'target_activity': float(target_activity),
            'rate_at_target': float(I_target),
            'probability_estimate': float(np.exp(-I_target)),  # Approximate for T=1
            'n_rare_events': int(n_rare),
            'rare_event_fraction': float(rare_fraction),
            'effective_sample_size': float(ess),
            'ess_fraction': float(ess_fraction),
            'importance_weights_min': float(weights.min()),
            'importance_weights_max': float(weights.max()),
            'n_samples': len(observable)
        }

    def _transition_path_sampling(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform transition path sampling (TPS) analysis.

        Physics:
            - Committor: p_B(x) = probability of reaching B before A from x
            - Reactive flux: k = (1/τ) * ∫ p(q) δ(q - q†) |q̇| dq
            - Transition state: q† where p_B(q†) = 0.5
            - TPS ensemble: Paths from A to B with specified length

        Args:
            input_data: Contains trajectory, region_A, region_B

        Returns:
            Dictionary with committor, transition states, reactive flux
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Extract trajectory
        trajectory = np.array(data.get('trajectory', np.random.randn(1000)))

        # Define regions A and B
        threshold_A = params.get('region_A_max', np.percentile(trajectory, 25))
        threshold_B = params.get('region_B_min', np.percentile(trajectory, 75))

        # Compute committor (simplified - fraction of future in B)
        committor = np.zeros(len(trajectory))
        for i in range(len(trajectory) - 1):
            # Simple estimate: does trajectory reach B before returning to A?
            future = trajectory[i+1:]
            reaches_B = np.any(future >= threshold_B)
            reaches_A = np.any(future <= threshold_A)

            if reaches_B and not reaches_A:
                committor[i] = 1.0
            elif reaches_A and not reaches_B:
                committor[i] = 0.0
            else:
                # Mixed or neither
                committor[i] = 0.5

        # Find transition state (committor ≈ 0.5)
        idx_transition = np.argmin(np.abs(committor - 0.5))
        transition_state = trajectory[idx_transition]

        # Reactive trajectories (cross from A to B)
        in_A = trajectory <= threshold_A
        in_B = trajectory >= threshold_B

        # Find A→B transitions
        transitions = []
        i = 0
        while i < len(trajectory) - 1:
            if in_A[i]:
                # Start in A, look for B
                j = i + 1
                while j < len(trajectory) and not in_A[j] and not in_B[j]:
                    j += 1
                if j < len(trajectory) and in_B[j]:
                    transitions.append((i, j))
                    i = j
                else:
                    i += 1
            else:
                i += 1

        n_transitions = len(transitions)

        # Estimate transition rate
        if n_transitions > 0:
            mean_transition_time = np.mean([t[1] - t[0] for t in transitions])
            transition_rate = 1.0 / mean_transition_time if mean_transition_time > 0 else 0.0
        else:
            mean_transition_time = 0.0
            transition_rate = 0.0

        return {
            'threshold_A': float(threshold_A),
            'threshold_B': float(threshold_B),
            'committor_values': committor.tolist(),
            'transition_state_position': float(transition_state),
            'transition_state_index': int(idx_transition),
            'transition_state_committor': float(committor[idx_transition]),
            'n_transitions': int(n_transitions),
            'transition_times': [t[1] - t[0] for t in transitions],
            'mean_transition_time': float(mean_transition_time),
            'transition_rate': float(transition_rate),
            'trajectory_length': len(trajectory)
        }

    def _dynamical_phase_transition(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dynamical phase transitions in biased ensembles.

        Physics:
            - Scaled cumulant generating function: θ(s) = lim_{T→∞} 1/T * ln⟨exp(-s*A_T)⟩
            - First-order DPT: Non-analytic θ(s) (kink or discontinuity)
            - Critical s*: Phase transition point
            - Active/inactive phases: High/low activity regions

        Args:
            input_data: Contains s_values, observable time series

        Returns:
            Dictionary with θ(s), phase diagram, critical point
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Extract observable
        observable = np.array(data.get('observable', np.random.exponential(1.0, 10000)))

        # Range of bias parameters
        if 's_values' in data:
            s_values = np.array(data['s_values'])
        else:
            s_min = params.get('s_min', -2.0)
            s_max = params.get('s_max', 2.0)
            n_s = params.get('n_s_points', 20)
            s_values = np.linspace(s_min, s_max, n_s)

        # Compute θ(s) for each s
        theta_values = np.zeros(len(s_values))

        for i, s in enumerate(s_values):
            # θ(s) = ln⟨exp(-s*A)⟩ (note sign convention)
            theta_values[i] = logsumexp(-s * observable) - np.log(len(observable))

        # Detect phase transition (non-analytic point)
        # Compute second derivative d²θ/ds²
        if len(s_values) > 2:
            dtheta = np.gradient(theta_values, s_values)
            d2theta = np.gradient(dtheta, s_values)

            # Find maximum curvature (potential transition)
            idx_critical = np.argmax(np.abs(d2theta))
            s_critical = s_values[idx_critical]
            theta_critical = theta_values[idx_critical]

            # Check if genuine transition (large enough curvature)
            curvature_max = np.abs(d2theta[idx_critical])
            has_transition = curvature_max > 0.1
        else:
            s_critical = 0.0
            theta_critical = theta_values[0] if len(theta_values) > 0 else 0.0
            dtheta = np.zeros_like(s_values)
            d2theta = np.zeros_like(s_values)
            has_transition = False
            curvature_max = 0.0

        # Activity at different s values
        # ⟨A⟩_s = -dθ/ds
        mean_activity = -dtheta

        return {
            's_values': s_values.tolist(),
            'theta_values': theta_values.tolist(),
            'dtheta_ds': dtheta.tolist(),
            'd2theta_ds2': d2theta.tolist(),
            's_critical': float(s_critical),
            'theta_critical': float(theta_critical),
            'has_phase_transition': bool(has_transition),
            'max_curvature': float(curvature_max),
            'mean_activity_vs_s': mean_activity.tolist(),
            'n_bias_points': len(s_values),
            'n_samples': len(observable)
        }

    def _rate_function_calculation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate large deviation rate function.

        Physics:
            - Cramér's theorem: P(A_T ≈ a) ~ exp(-T * I(a))
            - Gartner-Ellis: I(a) = sup_s [s*a - θ(s)] (Legendre transform)
            - Level 2.5: Rate function for empirical measure π_T

        Args:
            input_data: Contains observable time series

        Returns:
            Dictionary with rate function I(a), Legendre transform
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Extract observable
        observable = np.array(data.get('observable', np.random.randn(5000)))

        # Grid for rate function
        n_points = params.get('n_grid_points', 100)
        a_min = params.get('a_min', observable.min())
        a_max = params.get('a_max', observable.max())
        a_values = np.linspace(a_min, a_max, n_points)

        # Compute SCGF θ(s) via sampling
        s_grid = np.linspace(-5, 5, 50)
        theta_grid = np.array([logsumexp(s * observable) - np.log(len(observable)) for s in s_grid])

        # Rate function via Legendre transform
        rate_function = np.zeros(len(a_values))
        optimal_s = np.zeros(len(a_values))

        for i, a in enumerate(a_values):
            # I(a) = sup_s [s*a - θ(s)]
            legendre = s_grid * a - theta_grid
            idx_max = np.argmax(legendre)
            rate_function[i] = legendre[idx_max]
            optimal_s[i] = s_grid[idx_max]

        # Normalize rate function (I(⟨A⟩) should be ≈ 0)
        mean_a = np.mean(observable)
        idx_mean = np.argmin(np.abs(a_values - mean_a))
        rate_function -= rate_function[idx_mean]  # Shift so I(mean) = 0

        # Estimate probability decay
        # P(A ≈ a) ~ exp(-T * I(a)) for large T
        # Here T = 1 for empirical estimate
        log_prob_estimate = -rate_function

        return {
            'a_values': a_values.tolist(),
            'rate_function_I': rate_function.tolist(),
            'optimal_s_values': optimal_s.tolist(),
            'log_probability_estimate': log_prob_estimate.tolist(),
            'mean_observable': float(mean_a),
            'std_observable': float(np.std(observable)),
            'rate_at_mean': float(rate_function[idx_mean]),
            's_grid': s_grid.tolist(),
            'theta_grid': theta_grid.tolist(),
            'n_samples': len(observable),
            'n_grid_points': n_points
        }

    def _s_ensemble_simulation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate s-ensemble (biased ensemble) simulation.

        Physics:
            - Biased distribution: P_s(x) ∝ P(x) * exp(-s * A(x))
            - Modified dynamics: Enforce atypical behavior
            - Reweighting: ⟨f⟩ = ⟨f * w⟩_s where w = exp(s * A)

        Args:
            input_data: Contains trajectories, bias parameter s

        Returns:
            Dictionary with biased statistics, reweighting factors
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Extract data
        if 'trajectories' in data:
            trajectories = data['trajectories']
            observable_per_traj = np.array([np.sum(traj) for traj in trajectories])
        elif 'observable' in data:
            observable_per_traj = np.array(data['observable'])
        else:
            observable_per_traj = np.random.exponential(1.0, 1000)

        # Bias parameter
        s = params.get('bias_parameter', 1.0)

        # Compute s-ensemble weights
        # w(x) = exp(-s * A(x))
        weights_s = np.exp(-s * observable_per_traj)
        weights_s_normalized = weights_s / np.sum(weights_s)

        # Statistics in s-ensemble
        mean_s = np.average(observable_per_traj, weights=weights_s_normalized)
        var_s = np.average((observable_per_traj - mean_s)**2, weights=weights_s_normalized)
        std_s = np.sqrt(var_s)

        # Compare with unbiased ensemble
        mean_unbiased = np.mean(observable_per_traj)
        std_unbiased = np.std(observable_per_traj)

        # Effective sample size
        ess = (np.sum(weights_s)**2) / np.sum(weights_s**2)
        ess_fraction = ess / len(observable_per_traj)

        # Scaled cumulant generating function
        theta_s = np.log(np.mean(weights_s))

        return {
            'bias_parameter_s': float(s),
            'theta_s': float(theta_s),
            'mean_unbiased': float(mean_unbiased),
            'std_unbiased': float(std_unbiased),
            'mean_s_ensemble': float(mean_s),
            'std_s_ensemble': float(std_s),
            'bias_shift': float(mean_s - mean_unbiased),
            'effective_sample_size': float(ess),
            'ess_fraction': float(ess_fraction),
            'min_weight': float(weights_s.min()),
            'max_weight': float(weights_s.max()),
            'n_trajectories': len(observable_per_traj)
        }

    # Integration methods
    def analyze_driven_rare_events(self, driven_result: Dict[str, Any],
                                   observable_key: str = 'work') -> Dict[str, Any]:
        """Analyze rare events in driven system results from DrivenSystemsAgent.

        Args:
            driven_result: Result from DrivenSystemsAgent
            observable_key: Key for observable (e.g., 'work', 'dissipation')

        Returns:
            Rare event analysis of driven system
        """
        observable = driven_result.get(observable_key, [])

        input_data = {
            'method': 'rare_event_sampling',
            'data': {'observable': observable},
            'parameters': {'bias_parameter': 2.0},
            'analysis': ['rate_function', 'rare_events']
        }

        return self.execute(input_data).data

    def compute_transition_rates(self, stochastic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute transition rates from StochasticDynamicsAgent trajectory.

        Args:
            stochastic_result: Result from StochasticDynamicsAgent

        Returns:
            Transition rate analysis via TPS
        """
        trajectory = stochastic_result.get('trajectory', [])

        input_data = {
            'method': 'transition_path_sampling',
            'data': {'trajectory': trajectory},
            'parameters': {},
            'analysis': ['committor', 'reactive_flux']
        }

        return self.execute(input_data).data

    def validate_fluctuation_tail(self, fluctuation_result: Dict[str, Any],
                                  observable: str = 'work') -> Dict[str, Any]:
        """Validate large deviation tails of fluctuation theorem.

        Args:
            fluctuation_result: Result from FluctuationAgent
            observable: Observable to analyze

        Returns:
            Rate function validation of fluctuation tails
        """
        obs_data = fluctuation_result.get(observable, [])

        input_data = {
            'method': 'rate_function_calculation',
            'data': {'observable': obs_data},
            'parameters': {},
            'analysis': ['tail_behavior', 'legendre_transform']
        }

        return self.execute(input_data).data

    def get_capabilities(self) -> List[Capability]:
        """Return list of agent capabilities."""
        return [
            Capability(
                name='rare_event_sampling',
                description='Importance sampling and cloning for rare events',
                input_types=['observable', 'trajectories', 'bias_parameter'],
                output_types=['rate_function', 'rare_event_probability', 'weights'],
                typical_use_cases=[
                    'Rare fluctuation analysis',
                    'Extreme event statistics',
                    'Importance sampling validation'
                ]
            ),
            Capability(
                name='transition_path_sampling',
                description='TPS analysis, committor, reactive flux',
                input_types=['trajectory', 'region_A', 'region_B'],
                output_types=['committor', 'transition_state', 'transition_rate'],
                typical_use_cases=[
                    'Reaction pathway analysis',
                    'Barrier crossing studies',
                    'Transition state identification'
                ]
            ),
            Capability(
                name='dynamical_phase_transition',
                description='s-ensemble phase transitions',
                input_types=['observable', 's_values'],
                output_types=['theta_s', 'critical_point', 'phase_diagram'],
                typical_use_cases=[
                    'Activity phase transitions',
                    'Dynamical criticality',
                    'Biased ensemble analysis'
                ]
            ),
            Capability(
                name='rate_function_calculation',
                description='Large deviation rate function via Legendre transform',
                input_types=['observable', 'n_grid_points'],
                output_types=['rate_function', 'probability_decay'],
                typical_use_cases=[
                    'Probability tail analysis',
                    'Large deviation principle',
                    'Cramér theorem validation'
                ]
            ),
            Capability(
                name='s_ensemble_simulation',
                description='Biased ensemble generation and reweighting',
                input_types=['observable', 'bias_parameter'],
                output_types=['biased_statistics', 'weights', 'ess'],
                typical_use_cases=[
                    'Tilted ensemble generation',
                    'Trajectory biasing',
                    'Atypical event sampling'
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="LargeDeviationAgent",
            version=self.VERSION,
            description="Large deviation theory and rare event analysis for nonequilibrium systems",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities()
        )

    # Abstract method implementations (required by AnalysisAgent)
    def analyze_trajectory(self, trajectory_data: Any) -> Dict[str, Any]:
        """Analyze trajectory using transition path sampling or rare events.

        Intelligently routes to appropriate method based on trajectory characteristics.

        Args:
            trajectory_data: Trajectory array, time series, or dict with trajectory

        Returns:
            Dictionary with analysis results (committor, rate function, etc.)
        """
        # Determine if this is trajectory-based or observable-based analysis
        if isinstance(trajectory_data, dict) and 'trajectory' in trajectory_data:
            # Use transition path sampling
            input_data = {
                'method': 'transition_path_sampling',
                'data': trajectory_data,
                'parameters': trajectory_data.get('parameters', {}),
                'analysis': ['committor', 'reactive_flux']
            }
        elif isinstance(trajectory_data, dict) and 'time_series' in trajectory_data:
            # Use dynamical phase transition analysis
            input_data = {
                'method': 'dynamical_phase_transition',
                'data': trajectory_data,
                'parameters': trajectory_data.get('parameters', {}),
                'analysis': ['scgf', 'critical_point']
            }
        else:
            # Assume it's an observable array for rare event analysis
            if isinstance(trajectory_data, (list, np.ndarray)):
                observable = trajectory_data
            else:
                observable = trajectory_data

            input_data = {
                'method': 'rare_event_sampling',
                'data': {'observable': observable if isinstance(observable, list) else observable.tolist()},
                'parameters': {},
                'analysis': ['rate_function']
            }

        result = self.execute(input_data)
        return result.data if result.status == AgentStatus.SUCCESS else {}

    def compute_observables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute large deviation observables (SCGF, rate function).

        Args:
            data: Dictionary with raw data (observable, trajectory, etc.)

        Returns:
            Dictionary with computed observables (rate_function, theta_s, etc.)
        """
        # Use rate function calculation method
        input_data = {
            'method': 'rate_function_calculation',
            'data': data,
            'parameters': data.get('parameters', {}),
            'analysis': ['rate_function', 'scgf']
        }

        result = self.execute(input_data)
        return result.data if result.status == AgentStatus.SUCCESS else {}

    def _compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Compute hash of input data for caching."""
        import hashlib
        import json
        hashable_data = {k: v for k, v in input_data.items() if not callable(v)}
        data_str = json.dumps(hashable_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]