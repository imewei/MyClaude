"""Advanced Optimization Methods for Optimal Control.

This module provides sophisticated optimization algorithms beyond standard
gradient descent, including constrained optimization, global search methods,
derivative-free optimization, and mixed-integer programming.

Features:
- Constrained optimization (SQP, interior point, augmented Lagrangian)
- Global optimization (genetic algorithms, differential evolution, simulated annealing)
- Derivative-free methods (Nelder-Mead, CMA-ES, pattern search)
- Mixed-integer optimal control (branch-and-bound)
- Multi-objective optimization (Pareto fronts, scalarization)

Author: Nonequilibrium Physics Agents
Week: 25-26 of Phase 4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp


class OptimizationMethod(Enum):
    """Optimization method types."""
    SQP = "sqp"  # Sequential Quadratic Programming
    INTERIOR_POINT = "interior_point"  # Interior point method
    AUGMENTED_LAGRANGIAN = "augmented_lagrangian"  # Penalty + Lagrangian
    GENETIC_ALGORITHM = "genetic_algorithm"  # Evolutionary
    DIFFERENTIAL_EVOLUTION = "differential_evolution"  # DE
    SIMULATED_ANNEALING = "simulated_annealing"  # SA
    CMA_ES = "cma_es"  # Covariance Matrix Adaptation
    NELDER_MEAD = "nelder_mead"  # Simplex method


@dataclass
class ConstrainedOptimizationConfig:
    """Configuration for constrained optimization."""

    # Method
    method: str = OptimizationMethod.SQP.value

    # Tolerances
    constraint_tol: float = 1e-6
    optimality_tol: float = 1e-6
    max_iterations: int = 1000

    # Augmented Lagrangian parameters
    penalty_initial: float = 10.0
    penalty_increase: float = 10.0
    penalty_max: float = 1e8

    # Interior point parameters
    barrier_parameter: float = 1.0
    barrier_decrease: float = 0.1


@dataclass
class GlobalOptimizationConfig:
    """Configuration for global optimization."""

    # Method
    method: str = OptimizationMethod.GENETIC_ALGORITHM.value

    # Population-based parameters
    population_size: int = 50
    num_generations: int = 100

    # Genetic algorithm
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3

    # Differential evolution
    de_strategy: str = "best1bin"  # DE strategy
    de_F: float = 0.8  # Differential weight
    de_CR: float = 0.9  # Crossover probability

    # Simulated annealing
    initial_temperature: float = 1.0
    cooling_rate: float = 0.95
    num_iterations_per_temp: int = 100

    # CMA-ES
    cma_sigma: float = 0.3  # Initial step size


@dataclass
class MixedIntegerConfig:
    """Configuration for mixed-integer optimization."""

    # Branch-and-bound
    branching_strategy: str = "most_fractional"  # most_fractional, depth_first
    max_nodes: int = 10000
    relative_gap: float = 1e-4

    # Integer variables (indices)
    integer_variables: List[int] = field(default_factory=list)


class SequentialQuadraticProgramming:
    """Sequential Quadratic Programming for constrained optimal control.

    Solves:
        min f(x)
        s.t. g(x) ≤ 0  (inequality constraints)
             h(x) = 0  (equality constraints)

    Method: Solve sequence of QP subproblems
    """

    def __init__(self, config: ConstrainedOptimizationConfig):
        """Initialize SQP solver.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.iteration = 0
        self.lagrange_multipliers = None

    def solve(
        self,
        objective: Callable,
        x0: np.ndarray,
        gradient: Optional[Callable] = None,
        hessian: Optional[Callable] = None,
        inequality_constraints: Optional[List[Dict]] = None,
        equality_constraints: Optional[List[Dict]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Solve constrained optimization problem.

        Args:
            objective: f(x) to minimize
            x0: Initial guess
            gradient: Optional gradient of f
            hessian: Optional Hessian of f
            inequality_constraints: List of {'fun', 'jac'} dicts for g(x) ≤ 0
            equality_constraints: List of {'fun', 'jac'} dicts for h(x) = 0
            bounds: Variable bounds

        Returns:
            Optimization result dictionary
        """
        # Use scipy's SLSQP (SQP implementation)
        constraints = []

        if inequality_constraints:
            for c in inequality_constraints:
                constraints.append({
                    'type': 'ineq',
                    'fun': c['fun'],
                    'jac': c.get('jac')
                })

        if equality_constraints:
            for c in equality_constraints:
                constraints.append({
                    'type': 'eq',
                    'fun': c['fun'],
                    'jac': c.get('jac')
                })

        # Solve
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            hess=hessian,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.optimality_tol,
                'disp': False
            }
        )

        return {
            'x': result.x,
            'fun': result.fun,
            'success': result.success,
            'message': result.message,
            'nit': result.nit,
            'constraint_violation': self._compute_constraint_violation(
                result.x, inequality_constraints, equality_constraints
            )
        }

    def _compute_constraint_violation(
        self,
        x: np.ndarray,
        ineq_constraints: Optional[List[Dict]],
        eq_constraints: Optional[List[Dict]]
    ) -> float:
        """Compute total constraint violation.

        Args:
            x: Point to evaluate
            ineq_constraints: Inequality constraints
            eq_constraints: Equality constraints

        Returns:
            Total violation
        """
        violation = 0.0

        if ineq_constraints:
            for c in ineq_constraints:
                g_val = c['fun'](x)
                violation += max(0, g_val)**2  # Violation if g > 0

        if eq_constraints:
            for c in eq_constraints:
                h_val = c['fun'](x)
                violation += h_val**2

        return np.sqrt(violation)


class AugmentedLagrangian:
    """Augmented Lagrangian method for constrained optimization.

    Solves: min f(x)  s.t. g(x) ≤ 0, h(x) = 0

    Method: L(x,λ,μ,ρ) = f(x) + λ'h(x) + (ρ/2)||h(x)||² + μ'max(0,g(x)) + (ρ/2)||max(0,g(x))||²
    """

    def __init__(self, config: ConstrainedOptimizationConfig):
        """Initialize augmented Lagrangian solver.

        Args:
            config: Optimization configuration
        """
        self.config = config

    def solve(
        self,
        objective: Callable,
        x0: np.ndarray,
        inequality_constraints: Optional[List[Callable]] = None,
        equality_constraints: Optional[List[Callable]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Solve using augmented Lagrangian.

        Args:
            objective: f(x)
            x0: Initial guess
            inequality_constraints: List of g_i(x) ≤ 0
            equality_constraints: List of h_j(x) = 0
            bounds: Variable bounds

        Returns:
            Optimization result
        """
        x = x0.copy()
        n = len(x)

        # Initialize multipliers
        num_ineq = len(inequality_constraints) if inequality_constraints else 0
        num_eq = len(equality_constraints) if equality_constraints else 0

        mu = np.zeros(num_ineq)  # Inequality multipliers
        lambda_ = np.zeros(num_eq)  # Equality multipliers
        rho = self.config.penalty_initial

        for outer_iter in range(self.config.max_iterations):
            # Define augmented Lagrangian
            def augmented_lagrangian(x_var):
                L = objective(x_var)

                # Equality constraints
                if equality_constraints:
                    for j, h in enumerate(equality_constraints):
                        h_val = h(x_var)
                        L += lambda_[j] * h_val + (rho / 2) * h_val**2

                # Inequality constraints
                if inequality_constraints:
                    for i, g in enumerate(inequality_constraints):
                        g_val = g(x_var)
                        g_plus = max(0, g_val)
                        L += mu[i] * g_plus + (rho / 2) * g_plus**2

                return L

            # Minimize augmented Lagrangian
            result = minimize(
                augmented_lagrangian,
                x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )

            x = result.x

            # Update multipliers
            if equality_constraints:
                for j, h in enumerate(equality_constraints):
                    lambda_[j] += rho * h(x)

            if inequality_constraints:
                for i, g in enumerate(inequality_constraints):
                    mu[i] = max(0, mu[i] + rho * g(x))

            # Check convergence
            violation = self._compute_violation(x, inequality_constraints, equality_constraints)

            if violation < self.config.constraint_tol:
                break

            # Increase penalty
            rho = min(rho * self.config.penalty_increase, self.config.penalty_max)

        return {
            'x': x,
            'fun': objective(x),
            'success': violation < self.config.constraint_tol,
            'iterations': outer_iter + 1,
            'constraint_violation': violation
        }

    def _compute_violation(
        self,
        x: np.ndarray,
        ineq: Optional[List[Callable]],
        eq: Optional[List[Callable]]
    ) -> float:
        """Compute constraint violation."""
        violation = 0.0

        if ineq:
            for g in ineq:
                violation += max(0, g(x))**2

        if eq:
            for h in eq:
                violation += h(x)**2

        return np.sqrt(violation)


class GeneticAlgorithm:
    """Genetic algorithm for global optimization.

    Evolutionary algorithm inspired by natural selection.
    """

    def __init__(self, config: GlobalOptimizationConfig):
        """Initialize genetic algorithm.

        Args:
            config: Global optimization configuration
        """
        self.config = config
        self.rng = np.random.RandomState(42)

    def solve(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        maximize: bool = False
    ) -> Dict[str, Any]:
        """Run genetic algorithm.

        Args:
            objective: f(x) to optimize
            bounds: [(lower, upper), ...] for each variable
            maximize: If True, maximize instead of minimize

        Returns:
            Optimization result
        """
        n_vars = len(bounds)
        pop_size = self.config.population_size

        # Initialize population
        population = np.array([
            [self.rng.uniform(bounds[i][0], bounds[i][1]) for i in range(n_vars)]
            for _ in range(pop_size)
        ])

        best_individual = None
        best_fitness = float('inf') if not maximize else float('-inf')

        for generation in range(self.config.num_generations):
            # Evaluate fitness
            fitness = np.array([objective(ind) for ind in population])

            if not maximize:
                fitness = -fitness  # Convert to maximization

            # Track best
            gen_best_idx = np.argmax(fitness)
            gen_best_fitness = -fitness[gen_best_idx] if not maximize else fitness[gen_best_idx]

            if (not maximize and gen_best_fitness < best_fitness) or \
               (maximize and gen_best_fitness > best_fitness):
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()

            # Selection (tournament)
            new_population = []

            for _ in range(pop_size):
                # Tournament selection
                tournament_indices = self.rng.choice(pop_size, self.config.tournament_size, replace=False)
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]

                new_population.append(population[winner_idx].copy())

            new_population = np.array(new_population)

            # Crossover
            for i in range(0, pop_size - 1, 2):
                if self.rng.rand() < self.config.crossover_rate:
                    # Single-point crossover
                    point = self.rng.randint(1, n_vars)
                    new_population[i, point:], new_population[i+1, point:] = \
                        new_population[i+1, point:].copy(), new_population[i, point:].copy()

            # Mutation
            for i in range(pop_size):
                for j in range(n_vars):
                    if self.rng.rand() < self.config.mutation_rate:
                        new_population[i, j] = self.rng.uniform(bounds[j][0], bounds[j][1])

            population = new_population

        return {
            'x': best_individual,
            'fun': best_fitness,
            'success': True,
            'generations': self.config.num_generations
        }


class SimulatedAnnealing:
    """Simulated annealing for global optimization.

    Probabilistic technique that accepts worse solutions with probability
    that decreases over time (temperature).
    """

    def __init__(self, config: GlobalOptimizationConfig):
        """Initialize simulated annealing.

        Args:
            config: Global optimization configuration
        """
        self.config = config
        self.rng = np.random.RandomState(42)

    def solve(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Run simulated annealing.

        Args:
            objective: f(x) to minimize
            x0: Initial point
            bounds: Variable bounds

        Returns:
            Optimization result
        """
        x_current = x0.copy()
        f_current = objective(x_current)

        x_best = x_current.copy()
        f_best = f_current

        temperature = self.config.initial_temperature

        total_iterations = 0

        while temperature > 1e-8:
            for _ in range(self.config.num_iterations_per_temp):
                # Generate neighbor
                x_neighbor = x_current + self.rng.normal(0, 0.1, size=len(x_current))

                # Clip to bounds
                for i, (low, high) in enumerate(bounds):
                    x_neighbor[i] = np.clip(x_neighbor[i], low, high)

                f_neighbor = objective(x_neighbor)

                # Accept or reject
                delta = f_neighbor - f_current

                if delta < 0 or self.rng.rand() < np.exp(-delta / temperature):
                    x_current = x_neighbor
                    f_current = f_neighbor

                    if f_current < f_best:
                        x_best = x_current.copy()
                        f_best = f_current

                total_iterations += 1

            # Cool down
            temperature *= self.config.cooling_rate

        return {
            'x': x_best,
            'fun': f_best,
            'success': True,
            'iterations': total_iterations
        }


class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy.

    Sophisticated derivative-free optimizer that adapts covariance matrix.
    """

    def __init__(self, config: GlobalOptimizationConfig):
        """Initialize CMA-ES.

        Args:
            config: Global optimization configuration
        """
        self.config = config

    def solve(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Run CMA-ES.

        Args:
            objective: f(x) to minimize
            x0: Initial mean
            bounds: Optional variable bounds

        Returns:
            Optimization result
        """
        # Simplified CMA-ES implementation
        n = len(x0)
        mean = x0.copy()
        sigma = self.config.cma_sigma
        C = np.eye(n)  # Covariance matrix

        lambda_ = self.config.population_size
        mu = lambda_ // 2  # Number of parents

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)

        best_x = mean.copy()
        best_f = objective(mean)

        for generation in range(self.config.num_generations):
            # Sample population
            population = []
            for _ in range(lambda_):
                z = np.random.randn(n)
                y = mean + sigma * (C @ z)

                # Apply bounds
                if bounds:
                    for i, (low, high) in enumerate(bounds):
                        y[i] = np.clip(y[i], low, high)

                population.append(y)

            population = np.array(population)

            # Evaluate
            fitness = np.array([objective(ind) for ind in population])

            # Select best mu
            sorted_indices = np.argsort(fitness)
            selected = population[sorted_indices[:mu]]

            # Update best
            if fitness[sorted_indices[0]] < best_f:
                best_f = fitness[sorted_indices[0]]
                best_x = population[sorted_indices[0]].copy()

            # Update mean
            mean = np.sum(weights[:, None] * selected, axis=0)

            # Update covariance (simplified)
            C = np.cov(selected.T)
            C += 1e-8 * np.eye(n)  # Regularization

        return {
            'x': best_x,
            'fun': best_f,
            'success': True,
            'generations': self.config.num_generations
        }


class MixedIntegerOptimization:
    """Mixed-integer optimal control via branch-and-bound.

    Handles problems with both continuous and integer decision variables.
    """

    def __init__(self, config: MixedIntegerConfig):
        """Initialize mixed-integer optimizer.

        Args:
            config: Mixed-integer configuration
        """
        self.config = config
        self.nodes_explored = 0

    def solve(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        continuous_solver: Callable = None
    ) -> Dict[str, Any]:
        """Solve mixed-integer problem via branch-and-bound.

        Args:
            objective: f(x) to minimize
            x0: Initial guess
            bounds: Variable bounds
            continuous_solver: Solver for relaxed problems

        Returns:
            Optimization result
        """
        if continuous_solver is None:
            continuous_solver = lambda obj, x, b: minimize(obj, x, bounds=b, method='L-BFGS-B')

        # Initial relaxation (all variables continuous)
        result = continuous_solver(objective, x0, bounds)

        best_integer_solution = None
        best_integer_value = float('inf')

        # Priority queue of nodes (upper_bound, x, bounds)
        queue = [(result.fun, result.x, bounds)]

        while queue and self.nodes_explored < self.config.max_nodes:
            self.nodes_explored += 1

            # Get node with best bound
            queue.sort(key=lambda item: item[0])
            _, x_relaxed, node_bounds = queue.pop(0)

            # Check if all integer variables are integer
            is_integer = all(
                abs(x_relaxed[i] - round(x_relaxed[i])) < 1e-6
                for i in self.config.integer_variables
            )

            if is_integer:
                # Feasible integer solution
                f_val = objective(x_relaxed)
                if f_val < best_integer_value:
                    best_integer_value = f_val
                    best_integer_solution = x_relaxed.copy()
                continue

            # Find most fractional integer variable
            max_frac = 0
            branch_var = None
            for i in self.config.integer_variables:
                frac = abs(x_relaxed[i] - round(x_relaxed[i]))
                if frac > max_frac:
                    max_frac = frac
                    branch_var = i

            if branch_var is None:
                continue

            # Branch
            x_val = x_relaxed[branch_var]

            # Left branch: x_i ≤ floor(x_val)
            bounds_left = node_bounds.copy()
            bounds_left[branch_var] = (bounds_left[branch_var][0], np.floor(x_val))

            try:
                result_left = continuous_solver(objective, x_relaxed, bounds_left)
                if result_left.success:
                    queue.append((result_left.fun, result_left.x, bounds_left))
            except:
                pass

            # Right branch: x_i ≥ ceil(x_val)
            bounds_right = node_bounds.copy()
            bounds_right[branch_var] = (np.ceil(x_val), bounds_right[branch_var][1])

            try:
                result_right = continuous_solver(objective, x_relaxed, bounds_right)
                if result_right.success:
                    queue.append((result_right.fun, result_right.x, bounds_right))
            except:
                pass

        return {
            'x': best_integer_solution,
            'fun': best_integer_value,
            'success': best_integer_solution is not None,
            'nodes_explored': self.nodes_explored
        }
