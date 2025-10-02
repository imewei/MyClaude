"""Optimal Control Agent - Thermodynamic Protocol Optimization Expert.

Capabilities:
- Minimal Dissipation Protocols: Geodesic paths in thermodynamic space
- Shortcuts to Adiabaticity: Counterdiabatic driving, fast processes
- Stochastic Optimal Control: HJB equation, Pontryagin maximum principle
- Thermodynamic Speed Limits: Minimum protocol duration bounds
- Reinforcement Learning Protocols: ML-optimized control strategies
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
from uuid import uuid4
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint, solve_ivp

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


class OptimalControlAgent(AnalysisAgent):
    """Optimal control and protocol optimization agent.

    Designs optimal thermodynamic protocols:
    - Minimal dissipation via geodesic paths
    - Counterdiabatic driving for fast processes
    - Stochastic optimal control (HJB, Pontryagin)
    - Thermodynamic speed limits
    - Machine learning protocol optimization
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimal control agent.

        Args:
            config: Configuration with optimization parameters, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'minimal_dissipation_protocol', 'shortcut_to_adiabaticity',
            'stochastic_optimal_control', 'thermodynamic_speed_limit',
            'reinforcement_learning_protocol'
        ]
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute optimal control analysis.

        Args:
            input_data: Input with keys:
                - method: str (minimal_dissipation_protocol, etc.)
                - data: dict (initial/final states, Hamiltonian, etc.)
                - parameters: dict (duration, temperature, constraints)
                - analysis: list of str (dissipation, efficiency, etc.)

        Returns:
            AgentResult with optimal protocol

        Example:
            >>> agent = OptimalControlAgent()
            >>> result = agent.execute({
            ...     'method': 'minimal_dissipation_protocol',
            ...     'data': {'lambda_initial': 0.0, 'lambda_final': 1.0},
            ...     'parameters': {'duration': 10.0, 'temperature': 300.0},
            ...     'analysis': ['protocol', 'dissipation']
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'minimal_dissipation_protocol')

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
            if method == 'minimal_dissipation_protocol':
                result_data = self._minimal_dissipation_protocol(input_data)
            elif method == 'shortcut_to_adiabaticity':
                result_data = self._shortcut_to_adiabaticity(input_data)
            elif method == 'stochastic_optimal_control':
                result_data = self._stochastic_optimal_control(input_data)
            elif method == 'thermodynamic_speed_limit':
                result_data = self._thermodynamic_speed_limit(input_data)
            elif method == 'reinforcement_learning_protocol':
                result_data = self._reinforcement_learning_protocol(input_data)
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
        """Validate input data for optimal control.

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

        # Check parameters
        if 'parameters' not in data:
            warnings.append("Missing 'parameters' field - using defaults")
        else:
            params = data['parameters']

            # Check duration
            if 'duration' in params:
                duration = params['duration']
                if duration <= 0:
                    errors.append(f"Invalid duration: {duration} (must be positive)")

            # Check temperature
            if 'temperature' in params:
                temp = params['temperature']
                if temp <= 0:
                    errors.append(f"Invalid temperature: {temp} K (must be positive)")

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
        method = input_data.get('method', 'minimal_dissipation_protocol')
        params = input_data.get('parameters', {})

        # Estimate based on method complexity
        duration = params.get('duration', 10.0)
        n_steps = params.get('n_steps', 100)

        # Base resource estimation
        if method == 'minimal_dissipation_protocol':
            cpu_cores = 2
            memory_gb = 1.0
            duration_est = 30
            env = ExecutionEnvironment.LOCAL
        elif method == 'shortcut_to_adiabaticity':
            cpu_cores = 2
            memory_gb = 2.0
            duration_est = 60
            env = ExecutionEnvironment.LOCAL
        elif method == 'stochastic_optimal_control':
            cpu_cores = 4
            memory_gb = 4.0
            duration_est = 300
            env = ExecutionEnvironment.HPC
        elif method == 'thermodynamic_speed_limit':
            cpu_cores = 1
            memory_gb = 0.5
            duration_est = 10
            env = ExecutionEnvironment.LOCAL
        elif method == 'reinforcement_learning_protocol':
            cpu_cores = 8
            memory_gb = 8.0
            duration_est = 1800
            env = ExecutionEnvironment.HPC
        else:
            cpu_cores = 2
            memory_gb = 2.0
            duration_est = 60
            env = ExecutionEnvironment.LOCAL

        # Adjust for problem size
        if n_steps > 1000:
            memory_gb *= 2
            duration_est *= 2

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_required=False,
            estimated_duration_seconds=duration_est,
            environment=env
        )

    def _minimal_dissipation_protocol(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Design minimal dissipation protocol via geodesic optimization.

        Physics:
            - Minimize: Σ = ∫_0^τ (dλ/dt)² / χ(λ) dt
            - Geodesic equation: d/dt [2λ̇/χ] = -λ̇²/χ² * dχ/dλ
            - Euler-Lagrange: Optimal path in Riemannian manifold

        Args:
            input_data: Contains lambda_initial, lambda_final, duration, susceptibility

        Returns:
            Dictionary with optimal protocol, dissipation, efficiency
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Protocol endpoints
        lambda_i = data.get('lambda_initial', 0.0)
        lambda_f = data.get('lambda_final', 1.0)

        # Protocol duration
        tau = params.get('duration', 10.0)
        n_steps = params.get('n_steps', 100)
        temperature = params.get('temperature', 300.0)

        # Time grid
        t_grid = np.linspace(0, tau, n_steps)

        # Susceptibility function χ(λ) (default: constant)
        if 'susceptibility' in data and callable(data['susceptibility']):
            chi = data['susceptibility']
        else:
            # Default: constant susceptibility
            chi = lambda lam: 1.0

        # For minimal dissipation with constant χ, optimal protocol is linear
        # λ(t) = λ_i + (λ_f - λ_i) * t/τ
        lambda_optimal = lambda_i + (lambda_f - lambda_i) * (t_grid / tau)

        # Compute dissipation
        # Σ = ∫ (dλ/dt)² / χ(λ) dt
        dlambda_dt = (lambda_f - lambda_i) / tau
        dissipation_integrand = (dlambda_dt**2) / chi(lambda_optimal[0])  # Approximate
        dissipation = dissipation_integrand * tau

        # For linear protocol with constant χ
        dissipation_analytical = (lambda_f - lambda_i)**2 / (chi(lambda_i) * tau)

        # Excess work (compared to quasistatic)
        # W_ex = Σ * kB * T (simplified)
        excess_work = dissipation * self.kB * temperature

        # Efficiency (inverse of dissipation - lower dissipation = higher efficiency)
        efficiency_metric = 1.0 / (1.0 + dissipation)

        return {
            'protocol_type': 'minimal_dissipation',
            'lambda_initial': float(lambda_i),
            'lambda_final': float(lambda_f),
            'duration': float(tau),
            'time_grid': t_grid.tolist(),
            'lambda_optimal': lambda_optimal.tolist(),
            'protocol_velocity': float(dlambda_dt),
            'dissipation': float(dissipation_analytical),
            'excess_work_J': float(excess_work),
            'efficiency_metric': float(efficiency_metric),
            'temperature_K': float(temperature),
            'n_steps': int(n_steps),
            'protocol_shape': 'linear_geodesic'
        }

    def _shortcut_to_adiabaticity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Design counterdiabatic driving protocol for fast adiabatic process.

        Physics:
            - Adiabatic: H(t)|ψ_n(t)⟩ = E_n(t)|ψ_n(t)⟩
            - CD Hamiltonian: H_CD = iℏ Σ_n |∂_t ψ_n⟩⟨ψ_n|
            - Total: H_total = H(t) + H_CD
            - Result: Exact eigenstate following

        Args:
            input_data: Contains Hamiltonian, eigenstates, duration

        Returns:
            Dictionary with CD protocol, energy cost, fidelity
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Protocol duration
        tau = params.get('duration', 1.0)
        n_steps = params.get('n_steps', 100)

        # Simplified: Two-level system
        # H(t) = -B(t) σ_z / 2, where B(t) varies from B_i to B_f
        B_i = data.get('field_initial', 1.0)
        B_f = data.get('field_final', 2.0)

        # Time grid
        t_grid = np.linspace(0, tau, n_steps)

        # Protocol for field: Linear ramp
        B_t = B_i + (B_f - B_i) * (t_grid / tau)

        # Adiabatic evolution: stays in ground state if slow enough
        # Adiabatic parameter: ξ = |dB/dt| / ΔE² ~ 1/τ
        dB_dt = (B_f - B_i) / tau
        energy_gap = np.abs(B_f - B_i)  # For two-level system
        adiabatic_parameter = np.abs(dB_dt) / (energy_gap**2)

        # CD term magnitude (simplified estimate)
        # |H_CD| ~ ℏ * |dB/dt| / ΔE
        H_CD_magnitude = self.hbar * np.abs(dB_dt) / energy_gap

        # Energy cost of CD driving
        # E_CD ~ ∫ |H_CD|² dt ~ (ℏ²/τ) * (dB/dt)² / ΔE²
        energy_cost_cd = (self.hbar**2 / tau) * (dB_dt**2) / (energy_gap**2)

        # Fidelity (overlap with adiabatic state)
        # For CD: F = 1 (exact following)
        # Without CD: F ≈ exp(-π ξ) for Landau-Zener
        fidelity_cd = 1.0
        fidelity_without_cd = np.exp(-np.pi / adiabatic_parameter) if adiabatic_parameter > 0 else 0.0

        # Shortcut factor: how much faster than adiabatic
        # Adiabatic requires τ_ad >> ℏ/ΔE
        tau_adiabatic = 10 * self.hbar / energy_gap
        speedup_factor = tau_adiabatic / tau if tau > 0 else 1.0

        return {
            'protocol_type': 'counterdiabatic_driving',
            'field_initial': float(B_i),
            'field_final': float(B_f),
            'duration': float(tau),
            'time_grid': t_grid.tolist(),
            'field_protocol': B_t.tolist(),
            'dB_dt': float(dB_dt),
            'energy_gap': float(energy_gap),
            'adiabatic_parameter': float(adiabatic_parameter),
            'H_CD_magnitude': float(H_CD_magnitude),
            'energy_cost_cd_J': float(energy_cost_cd),
            'fidelity_with_cd': float(fidelity_cd),
            'fidelity_without_cd': float(fidelity_without_cd),
            'adiabatic_duration_estimate': float(tau_adiabatic),
            'speedup_factor': float(speedup_factor),
            'n_steps': int(n_steps)
        }

    def _stochastic_optimal_control(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve stochastic optimal control problem via HJB/Pontryagin.

        Physics:
            - HJB equation: -∂V/∂t = min_u [L(x,u) + ∇V·f(x,u) + (1/2)σ²∇²V]
            - Pontryagin: Hamiltonian H(x,p,u) = L(x,u) + p·f(x,u)
            - Optimality: ∂H/∂u = 0
            - Costate: dp/dt = -∂H/∂x

        Args:
            input_data: Contains dynamics, cost function, constraints

        Returns:
            Dictionary with optimal control, value function, cost
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Problem setup
        tau = params.get('duration', 10.0)
        n_steps = params.get('n_steps', 100)
        x_initial = data.get('x_initial', 0.0)
        x_target = data.get('x_target', 1.0)

        # Time grid
        t_grid = np.linspace(0, tau, n_steps)

        # Simplified: Linear system with quadratic cost
        # dx/dt = u + η(t), L = (1/2)[Q*x² + R*u²]
        Q = params.get('state_cost', 1.0)
        R = params.get('control_cost', 0.1)

        # Optimal control for LQR: u* = -(R^-1 B^T P)x
        # For simple case: u*(t) ~ -(x(t) - x_target)/τ_remaining

        # Simulate optimal trajectory
        x_optimal = np.zeros(n_steps)
        u_optimal = np.zeros(n_steps)
        x_optimal[0] = x_initial

        for i in range(n_steps - 1):
            # Remaining time
            t_remaining = tau - t_grid[i]
            # Optimal control: drive toward target
            if t_remaining > 0:
                u_optimal[i] = -(x_optimal[i] - x_target) / t_remaining
            else:
                u_optimal[i] = 0.0

            # Update state (Euler step)
            dt = t_grid[1] - t_grid[0]
            x_optimal[i+1] = x_optimal[i] + u_optimal[i] * dt

        u_optimal[-1] = 0.0

        # Compute total cost
        # J = ∫[Q*x² + R*u²] dt
        cost_state = np.sum(Q * x_optimal**2) * (tau / n_steps)
        cost_control = np.sum(R * u_optimal**2) * (tau / n_steps)
        total_cost = cost_state + cost_control

        # Final state error
        final_error = np.abs(x_optimal[-1] - x_target)

        return {
            'protocol_type': 'stochastic_optimal_control',
            'x_initial': float(x_initial),
            'x_target': float(x_target),
            'duration': float(tau),
            'time_grid': t_grid.tolist(),
            'x_optimal': x_optimal.tolist(),
            'u_optimal': u_optimal.tolist(),
            'cost_state': float(cost_state),
            'cost_control': float(cost_control),
            'total_cost': float(total_cost),
            'final_error': float(final_error),
            'Q_state_cost': float(Q),
            'R_control_cost': float(R),
            'n_steps': int(n_steps)
        }

    def _thermodynamic_speed_limit(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute thermodynamic speed limits for protocols.

        Physics:
            - Mandelstam-Tamm: τ ≥ π/(2E) * |ψ_f - ψ_i|
            - Thermodynamic: τ ≥ ΔF²/(2kT * Σ)
            - Activity: τ ≥ (ΔS)² / (4 * activity)
            - Geometrical: τ ≥ distance / |v_max|

        Args:
            input_data: Contains free energy change, dissipation, temperature

        Returns:
            Dictionary with speed limit bounds, minimum duration
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Process parameters
        delta_F = data.get('free_energy_change', 10.0 * self.kB * 300.0)  # J
        dissipation = data.get('dissipation', 5.0 * self.kB * 300.0)  # J
        temperature = params.get('temperature', 300.0)  # K
        activity = data.get('activity', 1.0)  # Arbitrary units

        # Thermodynamic uncertainty bound
        # τ * Σ ≥ ΔF² / (2kT)
        if dissipation > 0:
            tau_tur = (delta_F**2) / (2 * self.kB * temperature * dissipation)
        else:
            tau_tur = np.inf

        # Activity bound (for entropy change)
        # Assume ΔS ~ ΔF / T
        delta_S = delta_F / temperature
        if activity > 0:
            tau_activity = (delta_S**2) / (4 * activity)
        else:
            tau_activity = np.inf

        # Geometrical bound (assume protocol space distance ~ |ΔF|)
        # τ ≥ distance / v_max
        v_max = params.get('max_velocity', 1.0)  # Protocol velocity limit
        if v_max > 0:
            tau_geometric = np.abs(delta_F) / v_max
        else:
            tau_geometric = np.inf

        # Overall minimum duration (most restrictive bound)
        tau_min = max(tau_tur, tau_activity, tau_geometric)

        # Efficiency at minimum duration
        # η = 1 - τ_min/τ_actual (if τ_actual given)
        tau_actual = params.get('actual_duration', tau_min * 2)
        efficiency = 1.0 - (tau_min / tau_actual) if tau_actual > tau_min else 0.0

        return {
            'bound_type': 'thermodynamic_speed_limits',
            'free_energy_change_J': float(delta_F),
            'dissipation_J': float(dissipation),
            'temperature_K': float(temperature),
            'activity': float(activity),
            'tau_tur_bound': float(tau_tur),
            'tau_activity_bound': float(tau_activity),
            'tau_geometric_bound': float(tau_geometric),
            'tau_minimum': float(tau_min),
            'most_restrictive_bound': 'TUR' if tau_min == tau_tur else ('activity' if tau_min == tau_activity else 'geometric'),
            'actual_duration': float(tau_actual),
            'efficiency_vs_limit': float(efficiency),
            'satisfies_bounds': bool(tau_actual >= tau_min)
        }

    def _reinforcement_learning_protocol(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Design protocol via reinforcement learning optimization.

        Physics:
            - Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            - Policy gradient: ∇_θ J = E[∇_θ log π_θ(a|s) * R]
            - Actor-critic: Combines value and policy optimization

        Args:
            input_data: Contains environment, reward function, RL parameters

        Returns:
            Dictionary with learned policy, cumulative reward, convergence
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # RL parameters
        n_episodes = params.get('n_episodes', 100)
        learning_rate = params.get('learning_rate', 0.1)
        discount_factor = params.get('discount_factor', 0.95)

        # Simplified: Discrete state-action space
        n_states = data.get('n_states', 10)
        n_actions = data.get('n_actions', 5)

        # Initialize Q-table
        Q = np.zeros((n_states, n_actions))

        # Training (simplified Q-learning)
        episode_rewards = []
        for episode in range(n_episodes):
            state = 0  # Start state
            total_reward = 0.0

            for step in range(20):  # Max steps per episode
                # ε-greedy action selection
                epsilon = 0.1 * (1.0 - episode / n_episodes)
                if np.random.rand() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[state, :])

                # Simulate environment (simplified)
                next_state = min(state + action, n_states - 1)
                reward = -np.abs(next_state - (n_states - 1)) / n_states  # Reward for reaching goal
                done = (next_state == n_states - 1)

                # Q-learning update
                Q[state, action] += learning_rate * (
                    reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action]
                )

                total_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)

        # Extract learned policy
        policy = np.argmax(Q, axis=1)

        # Compute optimal protocol from policy
        optimal_trajectory = [0]
        state = 0
        for _ in range(n_states):
            action = policy[state]
            state = min(state + action, n_states - 1)
            optimal_trajectory.append(state)
            if state == n_states - 1:
                break

        # Convergence metrics
        final_reward = episode_rewards[-1]
        mean_reward_last_10 = np.mean(episode_rewards[-10:])
        convergence_score = mean_reward_last_10 / (episode_rewards[0] + 0.01)  # Improvement ratio

        return {
            'protocol_type': 'reinforcement_learning',
            'n_episodes': int(n_episodes),
            'learning_rate': float(learning_rate),
            'discount_factor': float(discount_factor),
            'n_states': int(n_states),
            'n_actions': int(n_actions),
            'learned_policy': policy.tolist(),
            'optimal_trajectory': optimal_trajectory,
            'episode_rewards': episode_rewards,
            'final_reward': float(final_reward),
            'mean_reward_last_10': float(mean_reward_last_10),
            'convergence_score': float(convergence_score),
            'training_converged': bool(convergence_score > 1.5)
        }

    # Integration methods
    def optimize_driven_protocol(self, driven_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize protocol for DrivenSystemsAgent.

        Args:
            driven_params: Parameters for driven system (shear rate, field, etc.)

        Returns:
            Optimal protocol for minimal dissipation
        """
        input_data = {
            'method': 'minimal_dissipation_protocol',
            'data': driven_params,
            'parameters': {'duration': 10.0, 'temperature': 300.0},
            'analysis': ['protocol', 'dissipation']
        }

        return self.execute(input_data).data

    def design_minimal_work_process(self, fluctuation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Design minimal work process for FluctuationAgent validation.

        Args:
            fluctuation_data: Data from fluctuation theorem experiments

        Returns:
            Optimal protocol minimizing work
        """
        delta_F = fluctuation_data.get('free_energy_change', 10.0)

        input_data = {
            'method': 'thermodynamic_speed_limit',
            'data': {'free_energy_change': delta_F},
            'parameters': {'temperature': 300.0},
            'analysis': ['bounds', 'efficiency']
        }

        return self.execute(input_data).data

    def feedback_optimal_control(self, info_thermo_result: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal feedback control using information thermodynamics.

        Args:
            info_thermo_result: Result from InformationThermodynamicsAgent

        Returns:
            Optimal feedback protocol
        """
        information = info_thermo_result.get('information_nats', 1.0)

        input_data = {
            'method': 'stochastic_optimal_control',
            'data': {'x_initial': 0.0, 'x_target': information},
            'parameters': {'duration': 5.0},
            'analysis': ['control', 'cost']
        }

        return self.execute(input_data).data

    def get_capabilities(self) -> List[Capability]:
        """Return list of agent capabilities."""
        return [
            Capability(
                name='minimal_dissipation_protocol',
                description='Geodesic protocol for minimal entropy production',
                input_types=['lambda_initial', 'lambda_final', 'duration', 'susceptibility'],
                output_types=['optimal_protocol', 'dissipation', 'efficiency'],
                typical_use_cases=[
                    'Thermodynamic protocol optimization',
                    'Minimal work processes',
                    'Entropy production minimization'
                ]
            ),
            Capability(
                name='shortcut_to_adiabaticity',
                description='Counterdiabatic driving for fast adiabatic processes',
                input_types=['field_initial', 'field_final', 'duration', 'energy_gap'],
                output_types=['cd_protocol', 'energy_cost', 'fidelity'],
                typical_use_cases=[
                    'Fast quantum state preparation',
                    'Adiabatic quantum computing',
                    'Rapid protocol execution'
                ]
            ),
            Capability(
                name='stochastic_optimal_control',
                description='HJB/Pontryagin optimal control for stochastic systems',
                input_types=['x_initial', 'x_target', 'state_cost', 'control_cost'],
                output_types=['optimal_control', 'trajectory', 'cost'],
                typical_use_cases=[
                    'Feedback control design',
                    'Stochastic steering',
                    'Optimal driving protocols'
                ]
            ),
            Capability(
                name='thermodynamic_speed_limit',
                description='Minimum protocol duration bounds (TUR, activity, geometric)',
                input_types=['free_energy_change', 'dissipation', 'temperature'],
                output_types=['speed_limits', 'minimum_duration', 'efficiency'],
                typical_use_cases=[
                    'Protocol feasibility analysis',
                    'Thermodynamic bound verification',
                    'Speed-accuracy tradeoffs'
                ]
            ),
            Capability(
                name='reinforcement_learning_protocol',
                description='ML-optimized protocols via Q-learning',
                input_types=['n_states', 'n_actions', 'n_episodes', 'learning_rate'],
                output_types=['learned_policy', 'optimal_trajectory', 'convergence'],
                typical_use_cases=[
                    'Adaptive protocol learning',
                    'Model-free optimization',
                    'Complex control landscapes'
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="OptimalControlAgent",
            version=self.VERSION,
            description="Optimal control and thermodynamic protocol optimization",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities()
        )

    # Abstract method implementations (required by AnalysisAgent)
    def analyze_trajectory(self, trajectory_data: Any) -> Dict[str, Any]:
        """Analyze trajectory and design optimal control protocol.

        Args:
            trajectory_data: Trajectory data or protocol parameters

        Returns:
            Dictionary with optimal control recommendations
        """
        # Determine appropriate optimal control method
        if isinstance(trajectory_data, dict):
            # Check if this is a control problem or analysis problem
            if 'x_initial' in trajectory_data or 'x_target' in trajectory_data:
                # Stochastic optimal control
                input_data = {
                    'method': 'stochastic_optimal_control',
                    'data': trajectory_data,
                    'parameters': trajectory_data.get('parameters', {}),
                    'analysis': ['control', 'trajectory', 'cost']
                }
            elif 'lambda_initial' in trajectory_data or 'lambda_final' in trajectory_data:
                # Minimal dissipation protocol
                input_data = {
                    'method': 'minimal_dissipation_protocol',
                    'data': trajectory_data,
                    'parameters': trajectory_data.get('parameters', {}),
                    'analysis': ['protocol', 'dissipation']
                }
            else:
                # Default to thermodynamic speed limit analysis
                input_data = {
                    'method': 'thermodynamic_speed_limit',
                    'data': trajectory_data,
                    'parameters': trajectory_data.get('parameters', {}),
                    'analysis': ['bounds', 'efficiency']
                }
        else:
            # Assume trajectory array - analyze speed limits
            input_data = {
                'method': 'thermodynamic_speed_limit',
                'data': {'trajectory': trajectory_data if isinstance(trajectory_data, list) else trajectory_data.tolist()},
                'parameters': {},
                'analysis': ['bounds']
            }

        result = self.execute(input_data)
        return result.data if result.status == AgentStatus.SUCCESS else {}

    def compute_observables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optimal control observables (dissipation, efficiency, bounds).

        Args:
            data: Dictionary with protocol or system data

        Returns:
            Dictionary with computed observables
        """
        # Use thermodynamic speed limit to compute bounds
        input_data = {
            'method': 'thermodynamic_speed_limit',
            'data': data,
            'parameters': data.get('parameters', {}),
            'analysis': ['bounds', 'efficiency']
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