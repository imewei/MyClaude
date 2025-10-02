"""Stochastic Optimal Control.

This module implements stochastic control methods for systems with
random disturbances and probabilistic constraints:

1. Chance-constrained optimization
2. Risk-aware control (CVaR, mean-variance)
3. Stochastic Model Predictive Control
4. Scenario tree optimization
5. Sample Average Approximation (SAA)

Mathematical Formulation:
    min E[J(x, u, ξ)]
    s.t. P(g(x, u, ξ) ≤ 0) ≥ 1 - ε (chance constraints)
         x_{t+1} = f(x_t, u_t, ξ_t)

where ξ represents random variables.

Author: Nonequilibrium Physics Agents
"""

from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# =============================================================================
# Risk Measures
# =============================================================================

class RiskMeasure(Enum):
    """Types of risk measures."""
    EXPECTATION = "expectation"  # E[X]
    VARIANCE = "variance"  # Var[X]
    CVAR = "cvar"  # CVaR_α[X] = E[X | X ≥ VaR_α]
    WORST_CASE = "worst_case"  # max X
    MEAN_VARIANCE = "mean_variance"  # E[X] + λ·Var[X]


def compute_risk(
    samples: np.ndarray,
    measure: RiskMeasure,
    alpha: float = 0.95,
    lambda_risk: float = 1.0
) -> float:
    """Compute risk measure for samples.

    Args:
        samples: Array of cost samples
        measure: Risk measure type
        alpha: Confidence level (for CVaR)
        lambda_risk: Risk aversion parameter (for mean-variance)

    Returns:
        Risk value
    """
    if measure == RiskMeasure.EXPECTATION:
        return np.mean(samples)

    elif measure == RiskMeasure.VARIANCE:
        return np.var(samples)

    elif measure == RiskMeasure.CVAR:
        # Conditional Value at Risk
        var_alpha = np.percentile(samples, 100 * alpha)
        return np.mean(samples[samples >= var_alpha])

    elif measure == RiskMeasure.WORST_CASE:
        return np.max(samples)

    elif measure == RiskMeasure.MEAN_VARIANCE:
        return np.mean(samples) + lambda_risk * np.var(samples)

    else:
        raise ValueError(f"Unknown risk measure: {measure}")


# =============================================================================
# Chance-Constrained Optimization
# =============================================================================

@dataclass
class ChanceConstraint:
    """Represents a chance constraint.

    P(g(x, u, ξ) ≤ 0) ≥ 1 - ε

    Attributes:
        constraint_func: Constraint function g(x, u, xi)
        confidence_level: 1 - ε (e.g., 0.95 for 95% satisfaction)
        approximation: 'sampling', 'scenario', or 'reformulation'
    """
    constraint_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float]
    confidence_level: float = 0.95
    approximation: str = 'sampling'


class ChanceConstrainedOptimizer:
    """Optimizer for chance-constrained problems.

    Handles probabilistic constraints using various approximations.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray, np.ndarray], float],
        chance_constraints: List[ChanceConstraint],
        disturbance_sampler: Callable[[int], np.ndarray],
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize chance-constrained optimizer.

        Args:
            objective: Objective function J(x, u)
            chance_constraints: List of chance constraints
            disturbance_sampler: Function to sample disturbances
            control_bounds: Control bounds
        """
        self.objective = objective
        self.chance_constraints = chance_constraints
        self.disturbance_sampler = disturbance_sampler
        self.control_bounds = control_bounds

    def evaluate_constraint_satisfaction(
        self,
        x: np.ndarray,
        u: np.ndarray,
        constraint: ChanceConstraint,
        n_samples: int = 1000
    ) -> float:
        """Evaluate constraint satisfaction probability.

        Args:
            x: State
            u: Control
            constraint: Chance constraint
            n_samples: Number of samples

        Returns:
            Estimated satisfaction probability
        """
        # Sample disturbances
        samples = self.disturbance_sampler(n_samples)

        # Evaluate constraint for each sample
        satisfied = 0
        for i in range(n_samples):
            xi = samples[i] if samples.ndim > 1 else samples
            if constraint.constraint_func(x, u, xi) <= 0:
                satisfied += 1

        return satisfied / n_samples

    def check_feasibility(
        self,
        x: np.ndarray,
        u: np.ndarray,
        n_samples: int = 1000
    ) -> bool:
        """Check if control satisfies all chance constraints.

        Args:
            x: State
            u: Control
            n_samples: Number of samples

        Returns:
            True if all constraints satisfied
        """
        for constraint in self.chance_constraints:
            prob = self.evaluate_constraint_satisfaction(x, u, constraint, n_samples)
            if prob < constraint.confidence_level:
                return False
        return True

    def optimize(
        self,
        x0: np.ndarray,
        u0: np.ndarray,
        n_samples: int = 1000,
        n_constraint_samples: int = 100,
        method: str = 'penalty',
        penalty_weight: float = 1000.0,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """Optimize chance-constrained problem.

        Args:
            x0: Initial state
            u0: Initial control
            n_samples: Samples for objective
            n_constraint_samples: Samples for constraints
            method: 'penalty' or 'barrier'
            penalty_weight: Penalty/barrier weight
            **optimizer_kwargs: Optimizer arguments

        Returns:
            Optimization results
        """
        from scipy.optimize import minimize

        def penalized_objective(u):
            # Expected objective
            samples = self.disturbance_sampler(n_samples)
            obj = np.mean([self.objective(x0, u) for _ in range(n_samples)])

            # Penalty for constraint violation
            penalty = 0.0

            for constraint in self.chance_constraints:
                prob = self.evaluate_constraint_satisfaction(
                    x0, u, constraint, n_constraint_samples
                )

                violation = max(0, constraint.confidence_level - prob)

                if method == 'penalty':
                    penalty += penalty_weight * violation ** 2
                elif method == 'barrier':
                    if violation > 0:
                        penalty += penalty_weight * (-np.log(1e-10))
                    else:
                        # Barrier
                        margin = prob - constraint.confidence_level
                        if margin > 0:
                            penalty -= penalty_weight * np.log(margin)

            return obj + penalty

        # Optimize
        result = minimize(
            penalized_objective,
            u0,
            bounds=None if self.control_bounds is None else
                   list(zip(self.control_bounds[0], self.control_bounds[1])),
            **optimizer_kwargs
        )

        # Check final feasibility
        is_feasible = self.check_feasibility(x0, result.x, n_constraint_samples)

        return {
            'control': result.x,
            'objective': result.fun,
            'feasible': is_feasible,
            'success': result.success
        }


# =============================================================================
# CVaR Optimization
# =============================================================================

class CVaROptimizer:
    """Conditional Value at Risk (CVaR) optimization.

    Minimizes CVaR_α[J(u, ξ)] = min_{t} {t + 1/(1-α)·E[[J(u,ξ) - t]^+]}

    CVaR is a coherent risk measure that focuses on tail risk.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray, np.ndarray], float],
        disturbance_sampler: Callable[[int], np.ndarray],
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        alpha: float = 0.95
    ):
        """Initialize CVaR optimizer.

        Args:
            objective: Objective function J(u, xi)
            disturbance_sampler: Samples disturbances
            control_bounds: Control bounds
            alpha: Confidence level (e.g., 0.95 for 95%-CVaR)
        """
        self.objective = objective
        self.disturbance_sampler = disturbance_sampler
        self.control_bounds = control_bounds
        self.alpha = alpha

    def compute_cvar(
        self,
        u: np.ndarray,
        samples: np.ndarray
    ) -> float:
        """Compute CVaR for given control.

        Args:
            u: Control input
            samples: Disturbance samples

        Returns:
            CVaR value
        """
        # Evaluate objective for all samples
        costs = np.array([self.objective(u, xi) for xi in samples])

        # VaR (Value at Risk)
        var = np.percentile(costs, 100 * self.alpha)

        # CVaR: expected value in tail
        tail_costs = costs[costs >= var]
        if len(tail_costs) == 0:
            return var

        return np.mean(tail_costs)

    def optimize(
        self,
        u0: np.ndarray,
        n_samples: int = 1000,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """Optimize CVaR objective.

        Args:
            u0: Initial control
            n_samples: Number of samples
            **optimizer_kwargs: Optimizer arguments

        Returns:
            Optimization results
        """
        from scipy.optimize import minimize

        # Sample disturbances (fixed for optimization)
        samples = self.disturbance_sampler(n_samples)

        def cvar_objective(u):
            return self.compute_cvar(u, samples)

        # Optimize
        result = minimize(
            cvar_objective,
            u0,
            bounds=None if self.control_bounds is None else
                   list(zip(self.control_bounds[0], self.control_bounds[1])),
            **optimizer_kwargs
        )

        # Compute statistics
        costs = np.array([self.objective(result.x, xi) for xi in samples])

        return {
            'control': result.x,
            'cvar': result.fun,
            'mean_cost': np.mean(costs),
            'var': np.percentile(costs, 100 * self.alpha),
            'max_cost': np.max(costs),
            'success': result.success
        }


# =============================================================================
# Risk-Aware Optimizer
# =============================================================================

class RiskAwareOptimizer:
    """General risk-aware optimization framework.

    Supports multiple risk measures and objective combinations.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray, np.ndarray], float],
        disturbance_sampler: Callable[[int], np.ndarray],
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        risk_measure: RiskMeasure = RiskMeasure.MEAN_VARIANCE,
        risk_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize risk-aware optimizer.

        Args:
            objective: Objective function
            disturbance_sampler: Disturbance sampler
            control_bounds: Control bounds
            risk_measure: Type of risk measure
            risk_params: Risk measure parameters
        """
        self.objective = objective
        self.disturbance_sampler = disturbance_sampler
        self.control_bounds = control_bounds
        self.risk_measure = risk_measure
        self.risk_params = risk_params or {}

    def optimize(
        self,
        u0: np.ndarray,
        n_samples: int = 1000,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """Optimize risk-aware objective.

        Args:
            u0: Initial control
            n_samples: Number of samples
            **optimizer_kwargs: Optimizer arguments

        Returns:
            Optimization results
        """
        from scipy.optimize import minimize

        # Sample disturbances
        samples = self.disturbance_sampler(n_samples)

        def risk_objective(u):
            # Evaluate for all samples
            costs = np.array([self.objective(u, xi) for xi in samples])

            # Compute risk measure
            return compute_risk(
                costs,
                self.risk_measure,
                alpha=self.risk_params.get('alpha', 0.95),
                lambda_risk=self.risk_params.get('lambda', 1.0)
            )

        # Optimize
        result = minimize(
            risk_objective,
            u0,
            bounds=None if self.control_bounds is None else
                   list(zip(self.control_bounds[0], self.control_bounds[1])),
            **optimizer_kwargs
        )

        # Compute statistics
        final_costs = np.array([self.objective(result.x, xi) for xi in samples])

        return {
            'control': result.x,
            'risk_value': result.fun,
            'mean_cost': np.mean(final_costs),
            'std_cost': np.std(final_costs),
            'min_cost': np.min(final_costs),
            'max_cost': np.max(final_costs),
            'success': result.success
        }


# =============================================================================
# Stochastic Model Predictive Control
# =============================================================================

class StochasticMPC:
    """Stochastic Model Predictive Control.

    Solves:
        min E[Σ_t l(x_t, u_t)]
        s.t. x_{t+1} = f(x_t, u_t, ξ_t)
             P(g(x_t, u_t) ≤ 0) ≥ 1 - ε
    """

    def __init__(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        stage_cost: Callable[[np.ndarray, np.ndarray], float],
        disturbance_sampler: Callable[[int], np.ndarray],
        horizon: int,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        chance_constraints: Optional[List[ChanceConstraint]] = None,
        risk_measure: RiskMeasure = RiskMeasure.EXPECTATION
    ):
        """Initialize stochastic MPC.

        Args:
            dynamics: Dynamics x+ = f(x, u, xi)
            stage_cost: Stage cost l(x, u)
            disturbance_sampler: Samples disturbances
            horizon: MPC horizon
            control_bounds: Control bounds
            state_bounds: State bounds
            chance_constraints: Chance constraints
            risk_measure: Risk measure for objective
        """
        self.dynamics = dynamics
        self.stage_cost = stage_cost
        self.disturbance_sampler = disturbance_sampler
        self.horizon = horizon
        self.control_bounds = control_bounds
        self.state_bounds = state_bounds
        self.chance_constraints = chance_constraints or []
        self.risk_measure = risk_measure

    def simulate_scenario(
        self,
        x0: np.ndarray,
        u_sequence: np.ndarray,
        disturbances: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Simulate single scenario.

        Args:
            x0: Initial state
            u_sequence: Control sequence
            disturbances: Disturbance sequence

        Returns:
            Total cost and state trajectory
        """
        x = x0.copy()
        trajectory = [x]
        total_cost = 0.0

        for t in range(self.horizon):
            u = u_sequence[t]
            xi = disturbances[t] if disturbances.ndim > 1 else disturbances

            # Cost
            total_cost += self.stage_cost(x, u)

            # Dynamics
            x = self.dynamics(x, u, xi)
            trajectory.append(x)

        # Terminal cost
        total_cost += self.stage_cost(x, np.zeros_like(u))

        return total_cost, np.array(trajectory)

    def plan(
        self,
        x0: np.ndarray,
        n_scenarios: int = 50,
        risk_params: Optional[Dict] = None
    ) -> np.ndarray:
        """Plan control using stochastic MPC.

        Args:
            x0: Initial state
            n_scenarios: Number of scenarios for approximation
            risk_params: Risk measure parameters

        Returns:
            Optimal control for current step
        """
        from scipy.optimize import minimize

        n_states = len(x0)
        n_controls = len(self.control_bounds[0]) if self.control_bounds else 1

        # Decision variables: control sequence
        n_vars = n_controls * self.horizon

        # Sample scenarios
        scenario_disturbances = []
        for _ in range(n_scenarios):
            scenario = self.disturbance_sampler(self.horizon)
            scenario_disturbances.append(scenario)

        def stochastic_objective(u_flat):
            u_seq = u_flat.reshape(self.horizon, n_controls)

            # Evaluate for all scenarios
            costs = []
            for disturbances in scenario_disturbances:
                cost, _ = self.simulate_scenario(x0, u_seq, disturbances)
                costs.append(cost)

            costs = np.array(costs)

            # Apply risk measure
            return compute_risk(
                costs,
                self.risk_measure,
                **(risk_params or {})
            )

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
        result = minimize(stochastic_objective, u0, bounds=bounds, method='L-BFGS-B')

        u_opt = result.x.reshape(self.horizon, n_controls)

        return u_opt[0]


# =============================================================================
# Scenario Tree Optimization
# =============================================================================

@dataclass
class ScenarioTreeNode:
    """Node in scenario tree.

    Attributes:
        time: Time step
        state: State at this node
        probability: Probability of reaching this node
        parent: Parent node
        children: Child nodes
        control: Control applied at this node
    """
    time: int
    state: np.ndarray
    probability: float
    parent: Optional['ScenarioTreeNode'] = None
    children: List['ScenarioTreeNode'] = None
    control: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class ScenarioTreeOptimizer:
    """Optimization over scenario trees.

    Builds branching tree of possible future scenarios and optimizes
    control policy over entire tree.
    """

    def __init__(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        stage_cost: Callable[[np.ndarray, np.ndarray], float],
        disturbance_scenarios: List[Tuple[np.ndarray, float]],
        horizon: int,
        branching_factor: int = 3,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize scenario tree optimizer.

        Args:
            dynamics: Dynamics function
            stage_cost: Stage cost
            disturbance_scenarios: List of (disturbance, probability) tuples
            horizon: Planning horizon
            branching_factor: Number of branches per node
            control_bounds: Control bounds
        """
        self.dynamics = dynamics
        self.stage_cost = stage_cost
        self.disturbance_scenarios = disturbance_scenarios
        self.horizon = horizon
        self.branching_factor = branching_factor
        self.control_bounds = control_bounds

    def build_tree(
        self,
        x0: np.ndarray
    ) -> ScenarioTreeNode:
        """Build scenario tree.

        Args:
            x0: Initial state

        Returns:
            Root node of tree
        """
        # Root
        root = ScenarioTreeNode(
            time=0,
            state=x0,
            probability=1.0
        )

        # Build tree level by level
        current_level = [root]

        for t in range(self.horizon):
            next_level = []

            for node in current_level:
                # Sample scenarios for this node
                for i in range(min(self.branching_factor, len(self.disturbance_scenarios))):
                    xi, prob = self.disturbance_scenarios[i % len(self.disturbance_scenarios)]

                    # Create child (state will be filled during optimization)
                    child = ScenarioTreeNode(
                        time=t + 1,
                        state=node.state.copy(),  # Placeholder
                        probability=node.probability * prob,
                        parent=node
                    )

                    node.children.append(child)
                    next_level.append(child)

            current_level = next_level

        return root

    def get_all_nodes(
        self,
        root: ScenarioTreeNode
    ) -> List[ScenarioTreeNode]:
        """Get all nodes in tree.

        Args:
            root: Root node

        Returns:
            List of all nodes
        """
        nodes = [root]
        queue = [root]

        while queue:
            node = queue.pop(0)
            for child in node.children:
                nodes.append(child)
                queue.append(child)

        return nodes

    def optimize(
        self,
        x0: np.ndarray,
        u0_sequence: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Optimize over scenario tree.

        Args:
            x0: Initial state
            u0_sequence: Initial control sequence guess

        Returns:
            Optimization results with control policy
        """
        # Build tree
        root = self.build_tree(x0)
        all_nodes = self.get_all_nodes(root)

        # Count decision variables (one control per non-leaf node)
        non_leaf_nodes = [n for n in all_nodes if n.children]
        n_controls = len(self.control_bounds[0]) if self.control_bounds else 1
        n_vars = len(non_leaf_nodes) * n_controls

        # Map nodes to variable indices
        node_to_idx = {id(node): i for i, node in enumerate(non_leaf_nodes)}

        def tree_objective(u_flat):
            # Reshape controls
            controls = u_flat.reshape(len(non_leaf_nodes), n_controls)

            # Assign controls to nodes
            for i, node in enumerate(non_leaf_nodes):
                node.control = controls[i]

            # Forward simulation through tree
            total_cost = 0.0

            for node in all_nodes:
                if node.parent is None:
                    # Root - already has state
                    continue

                # Get parent control
                parent = node.parent
                u = parent.control

                # Get disturbance (sample from scenarios)
                xi, _ = self.disturbance_scenarios[0]  # Simplified

                # Propagate dynamics
                node.state = self.dynamics(parent.state, u, xi)

                # Accumulate cost
                if node.parent:
                    cost = self.stage_cost(parent.state, u)
                    total_cost += node.probability * cost

            return total_cost

        # Initial guess
        if u0_sequence is None:
            u0 = np.zeros(n_vars)
        else:
            u0 = u0_sequence.flatten()[:n_vars]

        # Bounds
        bounds = None
        if self.control_bounds is not None:
            u_min, u_max = self.control_bounds
            bounds = []
            for _ in range(len(non_leaf_nodes)):
                for i in range(n_controls):
                    bounds.append((u_min[i], u_max[i]))

        # Optimize
        from scipy.optimize import minimize
        result = minimize(tree_objective, u0, bounds=bounds, method='L-BFGS-B')

        # Extract optimal policy
        optimal_controls = result.x.reshape(len(non_leaf_nodes), n_controls)

        # Get first-stage control
        first_control = optimal_controls[0]

        return {
            'control': first_control,
            'full_policy': optimal_controls,
            'expected_cost': result.fun,
            'tree': root,
            'success': result.success
        }


# =============================================================================
# Sample Average Approximation
# =============================================================================

class SampleAverageApproximation:
    """Sample Average Approximation (SAA) for stochastic programs.

    Approximates:
        min E_ξ[f(x, ξ)]
    with:
        min (1/N)·Σᵢ f(x, ξⁱ)

    Provides statistical guarantees on solution quality.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray, np.ndarray], float],
        disturbance_sampler: Callable[[int], np.ndarray],
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize SAA.

        Args:
            objective: Objective f(u, xi)
            disturbance_sampler: Samples disturbances
            control_bounds: Control bounds
        """
        self.objective = objective
        self.disturbance_sampler = disturbance_sampler
        self.control_bounds = control_bounds

    def solve_saa(
        self,
        u0: np.ndarray,
        n_samples: int,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """Solve single SAA problem.

        Args:
            u0: Initial guess
            n_samples: Number of samples
            **optimizer_kwargs: Optimizer arguments

        Returns:
            SAA solution
        """
        from scipy.optimize import minimize

        # Sample scenarios
        samples = self.disturbance_sampler(n_samples)

        def saa_objective(u):
            costs = [self.objective(u, xi) for xi in samples]
            return np.mean(costs)

        # Optimize
        result = minimize(
            saa_objective,
            u0,
            bounds=None if self.control_bounds is None else
                   list(zip(self.control_bounds[0], self.control_bounds[1])),
            **optimizer_kwargs
        )

        return {
            'control': result.x,
            'objective': result.fun,
            'samples': samples,
            'success': result.success
        }

    def optimize_with_validation(
        self,
        u0: np.ndarray,
        n_samples_train: int = 1000,
        n_samples_val: int = 10000,
        n_replications: int = 10,
        confidence_level: float = 0.95,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """Solve SAA with statistical validation.

        Args:
            u0: Initial guess
            n_samples_train: Samples for training
            n_samples_val: Samples for validation
            n_replications: Number of replications
            confidence_level: Confidence level for bounds
            **optimizer_kwargs: Optimizer arguments

        Returns:
            Results with confidence bounds
        """
        # Solve multiple SAA problems
        solutions = []
        train_objectives = []

        for i in range(n_replications):
            result = self.solve_saa(u0, n_samples_train, **optimizer_kwargs)
            solutions.append(result['control'])
            train_objectives.append(result['objective'])

        # Select best solution (lowest SAA objective)
        best_idx = np.argmin(train_objectives)
        best_solution = solutions[best_idx]

        # Validate on large independent sample
        val_samples = self.disturbance_sampler(n_samples_val)
        val_costs = [self.objective(best_solution, xi) for xi in val_samples]
        true_objective_estimate = np.mean(val_costs)
        true_objective_std = np.std(val_costs) / np.sqrt(n_samples_val)

        # Confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = true_objective_estimate - z_score * true_objective_std
        ci_upper = true_objective_estimate + z_score * true_objective_std

        # Optimality gap estimate
        gap_estimate = true_objective_estimate - min(train_objectives)

        return {
            'control': best_solution,
            'train_objective': train_objectives[best_idx],
            'true_objective_estimate': true_objective_estimate,
            'true_objective_std': true_objective_std,
            'confidence_interval': (ci_lower, ci_upper),
            'optimality_gap_estimate': gap_estimate,
            'n_replications': n_replications,
            'all_solutions': solutions
        }


# =============================================================================
# Stochastic Optimizer Interface
# =============================================================================

class StochasticOptimizer:
    """Unified interface for stochastic optimization.

    Provides access to multiple stochastic methods.
    """

    def __init__(
        self,
        objective: Optional[Callable] = None,
        dynamics: Optional[Callable] = None,
        disturbance_sampler: Optional[Callable] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize stochastic optimizer.

        Args:
            objective: Objective function
            dynamics: System dynamics
            disturbance_sampler: Disturbance sampler
            control_bounds: Control bounds
        """
        self.objective = objective
        self.dynamics = dynamics
        self.disturbance_sampler = disturbance_sampler
        self.control_bounds = control_bounds

    def optimize(
        self,
        u0: np.ndarray,
        method: str = 'saa',
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using specified stochastic method.

        Args:
            u0: Initial control
            method: 'saa', 'cvar', 'risk_aware', 'chance_constrained', or 'stochastic_mpc'
            **kwargs: Method-specific arguments

        Returns:
            Optimization results
        """
        if method == 'saa':
            optimizer = SampleAverageApproximation(
                self.objective,
                self.disturbance_sampler,
                self.control_bounds
            )
            return optimizer.optimize_with_validation(u0, **kwargs)

        elif method == 'cvar':
            optimizer = CVaROptimizer(
                self.objective,
                self.disturbance_sampler,
                self.control_bounds,
                alpha=kwargs.get('alpha', 0.95)
            )
            return optimizer.optimize(u0, **kwargs)

        elif method == 'risk_aware':
            optimizer = RiskAwareOptimizer(
                self.objective,
                self.disturbance_sampler,
                self.control_bounds,
                risk_measure=kwargs.get('risk_measure', RiskMeasure.MEAN_VARIANCE),
                risk_params=kwargs.get('risk_params', {})
            )
            return optimizer.optimize(u0, **kwargs)

        elif method == 'chance_constrained':
            if 'chance_constraints' not in kwargs:
                raise ValueError("Chance constraints required")

            optimizer = ChanceConstrainedOptimizer(
                self.objective,
                kwargs['chance_constraints'],
                self.disturbance_sampler,
                self.control_bounds
            )

            x0 = kwargs.get('x0', np.zeros(len(u0)))
            return optimizer.optimize(x0, u0, **kwargs)

        elif method == 'stochastic_mpc':
            if self.dynamics is None:
                raise ValueError("Dynamics required for stochastic MPC")

            mpc = StochasticMPC(
                self.dynamics,
                self.objective,
                self.disturbance_sampler,
                horizon=kwargs.get('horizon', 10),
                control_bounds=self.control_bounds,
                risk_measure=kwargs.get('risk_measure', RiskMeasure.EXPECTATION)
            )

            x0 = kwargs.get('x0', np.zeros(len(u0)))
            u_opt = mpc.plan(x0, **kwargs)

            return {
                'control': u_opt,
                'success': True,
                'mpc_controller': mpc
            }

        else:
            raise ValueError(f"Unknown method: {method}")
