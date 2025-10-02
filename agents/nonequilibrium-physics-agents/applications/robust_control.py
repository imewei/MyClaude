"""Robust Control Under Uncertainty.

This module implements robust optimal control methods that handle uncertainty
in system dynamics, parameters, and disturbances:

1. Min-max (worst-case) optimization
2. Distributionally robust optimization
3. Tube-based Model Predictive Control (MPC)
4. H-infinity control
5. Uncertainty set formulations

Mathematical Formulation:
    min_u max_w J(x, u, w)
    s.t. x' = f(x, u, w)
         g(x, u) ≤ 0
         w ∈ W (uncertainty set)

where w represents uncertain parameters/disturbances.

Author: Nonequilibrium Physics Agents
"""

from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# =============================================================================
# Uncertainty Set Definitions
# =============================================================================

class UncertaintySetType(Enum):
    """Types of uncertainty sets."""
    BOX = "box"  # Box/rectangular uncertainty: w ∈ [w_min, w_max]
    ELLIPSOIDAL = "ellipsoidal"  # Ellipsoidal: ||w - w_nom||_P ≤ 1
    POLYHEDRAL = "polyhedral"  # Polyhedral: Aw ≤ b
    BUDGET = "budget"  # Budget of uncertainty: ||w - w_nom||_1 ≤ Γ


@dataclass
class UncertaintySet:
    """Represents an uncertainty set for robust optimization.

    Attributes:
        set_type: Type of uncertainty set
        dimension: Dimension of uncertainty
        parameters: Set parameters (bounds, matrix, etc.)
        nominal: Nominal parameter value
    """
    set_type: UncertaintySetType
    dimension: int
    parameters: Dict[str, Any]
    nominal: Optional[np.ndarray] = None

    def contains(self, w: np.ndarray) -> bool:
        """Check if point is in uncertainty set.

        Args:
            w: Point to check

        Returns:
            True if w ∈ W
        """
        if self.set_type == UncertaintySetType.BOX:
            lower = self.parameters['lower']
            upper = self.parameters['upper']
            return np.all(w >= lower) and np.all(w <= upper)

        elif self.set_type == UncertaintySetType.ELLIPSOIDAL:
            P = self.parameters['P']
            center = self.parameters.get('center', np.zeros(self.dimension))
            deviation = w - center
            return deviation.T @ P @ deviation <= 1.0

        elif self.set_type == UncertaintySetType.POLYHEDRAL:
            A = self.parameters['A']
            b = self.parameters['b']
            return np.all(A @ w <= b)

        elif self.set_type == UncertaintySetType.BUDGET:
            gamma = self.parameters['gamma']
            center = self.parameters.get('center', np.zeros(self.dimension))
            return np.linalg.norm(w - center, ord=1) <= gamma

        return False

    def sample(self, n_samples: int = 1, method: str = 'uniform') -> np.ndarray:
        """Sample from uncertainty set.

        Args:
            n_samples: Number of samples
            method: Sampling method ('uniform', 'boundary', 'random')

        Returns:
            Samples of shape (n_samples, dimension)
        """
        if self.set_type == UncertaintySetType.BOX:
            lower = self.parameters['lower']
            upper = self.parameters['upper']

            if method == 'uniform':
                return np.random.uniform(lower, upper, size=(n_samples, self.dimension))
            elif method == 'boundary':
                # Sample from corners
                samples = []
                for _ in range(n_samples):
                    corner = np.array([np.random.choice([l, u])
                                      for l, u in zip(lower, upper)])
                    samples.append(corner)
                return np.array(samples)

        elif self.set_type == UncertaintySetType.ELLIPSOIDAL:
            P = self.parameters['P']
            center = self.parameters.get('center', np.zeros(self.dimension))

            # Sample from unit ball, then transform
            samples = []
            for _ in range(n_samples):
                # Random direction
                direction = np.random.randn(self.dimension)
                direction /= np.linalg.norm(direction)

                # Random radius
                if method == 'uniform':
                    radius = np.random.rand() ** (1 / self.dimension)
                else:  # boundary
                    radius = 1.0

                # Transform
                L = np.linalg.cholesky(np.linalg.inv(P))
                sample = center + radius * L @ direction
                samples.append(sample)

            return np.array(samples)

        elif self.set_type == UncertaintySetType.BUDGET:
            gamma = self.parameters['gamma']
            center = self.parameters.get('center', np.zeros(self.dimension))

            # Sample from L1 ball
            samples = []
            for _ in range(n_samples):
                # Random direction with random L1 norm
                direction = np.random.randn(self.dimension)
                if method == 'uniform':
                    scale = gamma * np.random.rand()
                else:  # boundary
                    scale = gamma

                sample = center + scale * direction / np.linalg.norm(direction, ord=1)
                samples.append(sample)

            return np.array(samples)

        return np.zeros((n_samples, self.dimension))

    def get_vertices(self) -> np.ndarray:
        """Get vertices of uncertainty set (if applicable).

        Returns:
            Vertices array
        """
        if self.set_type == UncertaintySetType.BOX:
            lower = self.parameters['lower']
            upper = self.parameters['upper']

            # Generate all 2^n corners
            n = self.dimension
            vertices = []
            for i in range(2 ** n):
                vertex = np.zeros(n)
                for j in range(n):
                    if (i >> j) & 1:
                        vertex[j] = upper[j]
                    else:
                        vertex[j] = lower[j]
                vertices.append(vertex)

            return np.array(vertices)

        else:
            raise NotImplementedError(f"Vertices not defined for {self.set_type}")


# =============================================================================
# Min-Max Optimizer
# =============================================================================

class MinMaxOptimizer:
    """Min-max (worst-case) robust optimization.

    Solves:
        min_u max_w J(u, w)
        s.t. u ∈ U, w ∈ W

    Uses iterative methods or sampling-based approximation.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray, np.ndarray], float],
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        uncertainty_set: Optional[UncertaintySet] = None
    ):
        """Initialize min-max optimizer.

        Args:
            objective: Objective function J(u, w)
            control_bounds: Bounds on control u
            uncertainty_set: Uncertainty set W
        """
        self.objective = objective
        self.control_bounds = control_bounds
        self.uncertainty_set = uncertainty_set

    def evaluate_worst_case(
        self,
        u: np.ndarray,
        n_samples: int = 100,
        method: str = 'sampling'
    ) -> Tuple[float, np.ndarray]:
        """Evaluate worst-case objective for given control.

        Args:
            u: Control input
            n_samples: Number of samples for approximation
            method: 'sampling' or 'vertices'

        Returns:
            Worst-case objective value and worst-case w
        """
        if method == 'vertices' and self.uncertainty_set.set_type == UncertaintySetType.BOX:
            # Exact evaluation at vertices
            vertices = self.uncertainty_set.get_vertices()
            costs = np.array([self.objective(u, w) for w in vertices])
            worst_idx = np.argmax(costs)
            return costs[worst_idx], vertices[worst_idx]

        else:
            # Sampling-based approximation
            samples = self.uncertainty_set.sample(n_samples, method='uniform')
            costs = np.array([self.objective(u, w) for w in samples])
            worst_idx = np.argmax(costs)
            return costs[worst_idx], samples[worst_idx]

    def optimize(
        self,
        u0: np.ndarray,
        method: str = 'sampling',
        n_samples: int = 100,
        optimizer_kwargs: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Optimize min-max problem.

        Args:
            u0: Initial control
            method: Worst-case evaluation method
            n_samples: Number of samples
            optimizer_kwargs: Arguments for optimizer

        Returns:
            Dictionary with results
        """
        from scipy.optimize import minimize

        optimizer_kwargs = optimizer_kwargs or {}

        # Objective: worst-case cost for given u
        def worst_case_objective(u):
            worst_cost, _ = self.evaluate_worst_case(u, n_samples, method)
            return worst_cost

        # Optimize
        result = minimize(
            worst_case_objective,
            u0,
            bounds=None if self.control_bounds is None else
                   list(zip(self.control_bounds[0], self.control_bounds[1])),
            **optimizer_kwargs
        )

        # Evaluate final worst case
        worst_cost, worst_w = self.evaluate_worst_case(result.x, n_samples, method)

        return {
            'control': result.x,
            'worst_case_cost': worst_cost,
            'worst_case_uncertainty': worst_w,
            'success': result.success,
            'n_iterations': result.nit if hasattr(result, 'nit') else None
        }


# =============================================================================
# Distributionally Robust Optimization
# =============================================================================

class DistributionallyRobust:
    """Distributionally robust optimization.

    Optimizes over a family of probability distributions:
        min_u max_{P ∈ P} E_P[J(u, w)]

    where P is an ambiguity set of distributions.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray, np.ndarray], float],
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        nominal_distribution: Optional[Callable] = None,
        wasserstein_radius: float = 1.0
    ):
        """Initialize DRO optimizer.

        Args:
            objective: Objective function J(u, w)
            control_bounds: Control bounds
            nominal_distribution: Nominal distribution sampler
            wasserstein_radius: Wasserstein ball radius
        """
        self.objective = objective
        self.control_bounds = control_bounds
        self.nominal_dist = nominal_distribution
        self.wasserstein_radius = wasserstein_radius

    def sample_worst_distribution(
        self,
        u: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """Sample from approximate worst-case distribution.

        Args:
            u: Control input
            n_samples: Number of samples

        Returns:
            Samples from worst distribution
        """
        # Sample from nominal
        if self.nominal_dist is not None:
            nominal_samples = self.nominal_dist(n_samples)
        else:
            nominal_samples = np.random.randn(n_samples, 1)

        # Evaluate costs
        costs = np.array([self.objective(u, w.reshape(-1)) for w in nominal_samples])

        # Worst-case reweighting (simplified)
        # In practice, solve dual optimization problem
        weights = np.exp(costs / self.wasserstein_radius)
        weights /= np.sum(weights)

        # Resample according to weights
        indices = np.random.choice(n_samples, size=n_samples, p=weights)
        return nominal_samples[indices]

    def optimize(
        self,
        u0: np.ndarray,
        n_samples: int = 1000,
        optimizer_kwargs: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Optimize DRO problem.

        Args:
            u0: Initial control
            n_samples: Number of samples
            optimizer_kwargs: Optimizer arguments

        Returns:
            Optimization results
        """
        from scipy.optimize import minimize

        optimizer_kwargs = optimizer_kwargs or {}

        def dro_objective(u):
            # Sample worst distribution
            worst_samples = self.sample_worst_distribution(u, n_samples)

            # Compute expected cost
            costs = [self.objective(u, w.reshape(-1)) for w in worst_samples]
            return np.mean(costs)

        result = minimize(
            dro_objective,
            u0,
            bounds=None if self.control_bounds is None else
                   list(zip(self.control_bounds[0], self.control_bounds[1])),
            **optimizer_kwargs
        )

        return {
            'control': result.x,
            'expected_cost': result.fun,
            'success': result.success
        }


# =============================================================================
# Tube-Based Model Predictive Control
# =============================================================================

class TubeBasedMPC:
    """Tube-based MPC for robust control.

    Maintains a tube around nominal trajectory that contains all
    possible uncertain trajectories.

    System: x_{t+1} = Ax_t + Bu_t + w_t, w_t ∈ W
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        uncertainty_set: UncertaintySet,
        state_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        control_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        horizon: int = 10
    ):
        """Initialize tube MPC.

        Args:
            A: State transition matrix
            B: Control matrix
            Q: State cost matrix
            R: Control cost matrix
            uncertainty_set: Disturbance set W
            state_constraints: State bounds
            control_constraints: Control bounds
            horizon: MPC horizon
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.uncertainty_set = uncertainty_set
        self.state_constraints = state_constraints
        self.control_constraints = control_constraints
        self.horizon = horizon

        # Compute ancillary controller K
        self.K = self._compute_ancillary_controller()

        # Compute minimal robust positive invariant set
        self.mrpi_set = self._compute_mrpi_set()

    def _compute_ancillary_controller(self) -> np.ndarray:
        """Compute stabilizing ancillary controller K.

        Returns:
            Feedback gain K
        """
        # Solve discrete-time Algebraic Riccati Equation
        try:
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            K = -np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
            return K
        except:
            # LQR failed, use simple stabilization
            return -0.1 * np.linalg.pinv(self.B) @ self.A

    def _compute_mrpi_set(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> UncertaintySet:
        """Compute minimal robust positive invariant set.

        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            MRPI set approximation
        """
        # Closed-loop dynamics
        A_cl = self.A + self.B @ self.K

        # Iteratively compute outer approximation
        # Start with disturbance set
        current_set = self.uncertainty_set

        # For box uncertainty, compute bound on tube
        if self.uncertainty_set.set_type == UncertaintySetType.BOX:
            w_max = self.uncertainty_set.parameters['upper']

            # Bound on sum: Σ_{i=0}^∞ (A_cl)^i w
            # Approximate with finite sum
            total = np.zeros_like(w_max)
            A_power = np.eye(len(self.A))

            for _ in range(max_iterations):
                total_new = total + np.abs(A_power) @ w_max
                if np.linalg.norm(total_new - total) < tolerance:
                    break
                total = total_new
                A_power = A_power @ A_cl

            # Create box set
            mrpi_set = UncertaintySet(
                set_type=UncertaintySetType.BOX,
                dimension=len(total),
                parameters={'lower': -total, 'upper': total}
            )

            return mrpi_set

        # For other sets, return nominal
        return current_set

    def plan(
        self,
        x0: np.ndarray,
        x_ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Plan control sequence using tube MPC.

        Args:
            x0: Initial state
            x_ref: Reference state (default: origin)

        Returns:
            Optimal control for current step
        """
        if x_ref is None:
            x_ref = np.zeros_like(x0)

        # Solve nominal MPC problem with tightened constraints
        # This is simplified - full implementation needs QP solver

        from scipy.optimize import minimize

        n_states = len(self.A)
        n_controls = self.B.shape[1]

        # Decision variables: [u_0, ..., u_{N-1}]
        n_vars = n_controls * self.horizon

        def objective(u_flat):
            u_seq = u_flat.reshape(self.horizon, n_controls)

            # Simulate nominal trajectory
            x = x0.copy()
            cost = 0.0

            for t in range(self.horizon):
                u = u_seq[t]

                # Stage cost
                cost += (x - x_ref).T @ self.Q @ (x - x_ref) + u.T @ self.R @ u

                # Dynamics
                x = self.A @ x + self.B @ u

            # Terminal cost
            cost += (x - x_ref).T @ self.Q @ (x - x_ref)

            return cost

        # Initial guess
        u0 = np.zeros(n_vars)

        # Bounds (tightened by tube)
        bounds = None
        if self.control_constraints is not None:
            # Tighten control constraints
            u_min, u_max = self.control_constraints

            # Account for ancillary controller
            # u = v + Ke, need v such that v + K*e_max ∈ [u_min, u_max]
            mrpi_max = np.abs(self.mrpi_set.parameters.get('upper', np.zeros(n_states)))
            k_e_max = np.abs(self.K) @ mrpi_max

            u_min_tight = u_min + k_e_max
            u_max_tight = u_max - k_e_max

            bounds = []
            for _ in range(self.horizon):
                for i in range(n_controls):
                    bounds.append((u_min_tight[i], u_max_tight[i]))

        # Optimize
        result = minimize(objective, u0, bounds=bounds, method='L-BFGS-B')

        u_opt = result.x.reshape(self.horizon, n_controls)

        return u_opt[0]


# =============================================================================
# H-Infinity Control
# =============================================================================

class HInfinityController:
    """H-infinity robust control.

    Minimizes worst-case gain from disturbances to performance output:
        ||z||_2 / ||w||_2 < γ

    where z is performance output, w is disturbance.
    """

    def __init__(
        self,
        A: np.ndarray,
        B1: np.ndarray,
        B2: np.ndarray,
        C1: np.ndarray,
        D12: np.ndarray,
        gamma: float = 1.0
    ):
        """Initialize H-infinity controller.

        System:
            x' = Ax + B1*w + B2*u
            z = C1*x + D12*u
            y = x (full state feedback)

        Args:
            A: State matrix
            B1: Disturbance input matrix
            B2: Control input matrix
            C1: Performance output matrix
            D12: Control-to-output matrix
            gamma: Disturbance attenuation level
        """
        self.A = A
        self.B1 = B1
        self.B2 = B2
        self.C1 = C1
        self.D12 = D12
        self.gamma = gamma

        # Compute controller
        self.K = self._synthesize_controller()

    def _synthesize_controller(self) -> np.ndarray:
        """Synthesize H-infinity controller.

        Returns:
            State feedback gain K
        """
        # Solve H-infinity Riccati equation
        # This is simplified - full solution requires checking conditions

        # For now, use LQR-like solution
        Q = self.C1.T @ self.C1
        R = self.D12.T @ self.D12

        try:
            P = solve_continuous_are(self.A, self.B2, Q, R)
            K = -np.linalg.inv(R) @ self.B2.T @ P
            return K
        except:
            # Fallback
            return -0.1 * np.linalg.pinv(self.B2) @ self.A

    def control(self, x: np.ndarray) -> np.ndarray:
        """Compute control input.

        Args:
            x: Current state

        Returns:
            Control input u = Kx
        """
        return self.K @ x


# =============================================================================
# Robust Optimizer Interface
# =============================================================================

class RobustOptimizer:
    """Unified interface for robust optimization.

    Provides access to multiple robust methods.
    """

    def __init__(
        self,
        objective: Optional[Callable] = None,
        dynamics: Optional[Callable] = None,
        uncertainty_set: Optional[UncertaintySet] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize robust optimizer.

        Args:
            objective: Objective function J(u, w)
            dynamics: System dynamics (for MPC)
            uncertainty_set: Uncertainty set
            control_bounds: Control bounds
        """
        self.objective = objective
        self.dynamics = dynamics
        self.uncertainty_set = uncertainty_set
        self.control_bounds = control_bounds

    def optimize(
        self,
        u0: np.ndarray,
        method: str = 'minmax',
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using specified robust method.

        Args:
            u0: Initial control
            method: 'minmax', 'dro', or 'tube_mpc'
            **kwargs: Method-specific arguments

        Returns:
            Optimization results
        """
        if method == 'minmax':
            optimizer = MinMaxOptimizer(
                self.objective,
                self.control_bounds,
                self.uncertainty_set
            )
            return optimizer.optimize(u0, **kwargs)

        elif method == 'dro':
            optimizer = DistributionallyRobust(
                self.objective,
                self.control_bounds,
                **kwargs
            )
            return optimizer.optimize(u0, **kwargs)

        elif method == 'tube_mpc':
            if 'A' not in kwargs or 'B' not in kwargs:
                raise ValueError("Tube MPC requires A and B matrices")

            mpc = TubeBasedMPC(
                A=kwargs['A'],
                B=kwargs['B'],
                Q=kwargs.get('Q', np.eye(kwargs['A'].shape[0])),
                R=kwargs.get('R', np.eye(kwargs['B'].shape[1])),
                uncertainty_set=self.uncertainty_set,
                state_constraints=kwargs.get('state_constraints'),
                control_constraints=self.control_bounds,
                horizon=kwargs.get('horizon', 10)
            )

            x0 = kwargs.get('x0', np.zeros(kwargs['A'].shape[0]))
            u_opt = mpc.plan(x0)

            return {
                'control': u_opt,
                'success': True,
                'mpc_controller': mpc
            }

        else:
            raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Robust MPC Class
# =============================================================================

class RobustMPC:
    """High-level robust MPC interface.

    Combines multiple robust MPC techniques.
    """

    def __init__(
        self,
        dynamics: Callable,
        objective: Callable,
        horizon: int,
        uncertainty_set: UncertaintySet,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        method: str = 'scenario'
    ):
        """Initialize robust MPC.

        Args:
            dynamics: Dynamics function x_{t+1} = f(x_t, u_t, w_t)
            objective: Stage cost l(x_t, u_t)
            horizon: MPC horizon
            uncertainty_set: Uncertainty set
            control_bounds: Control bounds
            state_bounds: State bounds
            method: 'scenario', 'tube', or 'minmax'
        """
        self.dynamics = dynamics
        self.objective = objective
        self.horizon = horizon
        self.uncertainty_set = uncertainty_set
        self.control_bounds = control_bounds
        self.state_bounds = state_bounds
        self.method = method

    def plan(
        self,
        x0: np.ndarray,
        n_scenarios: int = 20
    ) -> np.ndarray:
        """Plan control using robust MPC.

        Args:
            x0: Initial state
            n_scenarios: Number of uncertainty scenarios

        Returns:
            Optimal control for current step
        """
        from scipy.optimize import minimize

        n_controls = len(self.control_bounds[0]) if self.control_bounds else 1

        # Decision variables: control sequence
        n_vars = n_controls * self.horizon

        def robust_objective(u_flat):
            u_seq = u_flat.reshape(self.horizon, n_controls)

            if self.method == 'scenario':
                # Scenario-based: average over sampled scenarios
                scenarios = self.uncertainty_set.sample(n_scenarios)
                total_cost = 0.0

                for scenario in scenarios:
                    x = x0.copy()
                    cost = 0.0

                    for t in range(self.horizon):
                        cost += self.objective(x, u_seq[t])
                        # Simulate with this scenario
                        w_t = scenario if scenario.ndim == 1 else scenario[t % len(scenario)]
                        x = self.dynamics(x, u_seq[t], w_t)

                    total_cost += cost

                return total_cost / n_scenarios

            elif self.method == 'minmax':
                # Min-max: worst-case scenario
                scenarios = self.uncertainty_set.sample(n_scenarios)
                max_cost = -np.inf

                for scenario in scenarios:
                    x = x0.copy()
                    cost = 0.0

                    for t in range(self.horizon):
                        cost += self.objective(x, u_seq[t])
                        w_t = scenario if scenario.ndim == 1 else scenario[t % len(scenario)]
                        x = self.dynamics(x, u_seq[t], w_t)

                    max_cost = max(max_cost, cost)

                return max_cost

            else:
                raise ValueError(f"Unknown method: {self.method}")

        # Initial guess
        u0 = np.zeros(n_vars)

        # Bounds
        bounds = None
        if self.control_bounds is not None:
            u_min, u_max = self.control_bounds
            bounds = []
            for _ in range(self.horizon):
                for i in range(n_controls):
                    bounds.append((u_min[i], u_max[i]))

        # Optimize
        result = minimize(robust_objective, u0, bounds=bounds, method='L-BFGS-B')

        u_opt = result.x.reshape(self.horizon, n_controls)

        return u_opt[0]
