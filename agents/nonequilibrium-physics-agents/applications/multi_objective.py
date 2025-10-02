"""Multi-Objective Optimization for Optimal Control.

This module implements multi-objective optimization methods for problems
with competing objectives:

1. Scalarization methods (Weighted sum, ε-constraint)
2. Pareto front computation (NBI, NSGA-II)
3. Interactive methods (trade-off analysis)
4. Visualization tools for Pareto fronts

Mathematical Formulation:
    min [f₁(x), f₂(x), ..., fₖ(x)]
    s.t. g(x) ≤ 0
         h(x) = 0

A solution x* is Pareto optimal if there is no x such that:
- fᵢ(x) ≤ fᵢ(x*) for all i, and
- fⱼ(x) < fⱼ(x*) for at least one j

Author: Nonequilibrium Physics Agents
"""

from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# =============================================================================
# Pareto Front Data Structure
# =============================================================================

@dataclass
class ParetoSolution:
    """A single solution on the Pareto front.

    Attributes:
        decision_variables: Decision variables x
        objectives: Objective values [f₁(x), ..., fₖ(x)]
        constraints: Constraint values (if any)
        metadata: Additional information
    """
    decision_variables: np.ndarray
    objectives: np.ndarray
    constraints: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class ParetoFront:
    """Container for Pareto front solutions.

    Stores and manages multiple Pareto optimal solutions.
    """

    def __init__(self):
        """Initialize empty Pareto front."""
        self.solutions: List[ParetoSolution] = []

    def add_solution(self, solution: ParetoSolution):
        """Add solution to Pareto front.

        Args:
            solution: Solution to add
        """
        self.solutions.append(solution)

    def get_objectives_matrix(self) -> np.ndarray:
        """Get matrix of all objective values.

        Returns:
            Array of shape (n_solutions, n_objectives)
        """
        return np.array([sol.objectives for sol in self.solutions])

    def get_decision_matrix(self) -> np.ndarray:
        """Get matrix of all decision variables.

        Returns:
            Array of shape (n_solutions, n_vars)
        """
        return np.array([sol.decision_variables for sol in self.solutions])

    def filter_dominated(self):
        """Remove dominated solutions from front."""
        objectives = self.get_objectives_matrix()
        n = len(self.solutions)

        # Check dominance
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # j dominates i if:
                # - all objectives are better or equal
                # - at least one objective is strictly better
                all_better_equal = np.all(objectives[j] <= objectives[i])
                some_strictly_better = np.any(objectives[j] < objectives[i])

                if all_better_equal and some_strictly_better:
                    is_dominated[i] = True
                    break

        # Keep only non-dominated
        self.solutions = [sol for i, sol in enumerate(self.solutions)
                         if not is_dominated[i]]

    def compute_hypervolume(self, reference_point: np.ndarray) -> float:
        """Compute hypervolume indicator.

        Args:
            reference_point: Reference point (worse than all solutions)

        Returns:
            Hypervolume value
        """
        objectives = self.get_objectives_matrix()

        if len(objectives) == 0:
            return 0.0

        # Sort by first objective
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objs = objectives[sorted_indices]

        # Compute hypervolume (2D case for now)
        if objectives.shape[1] == 2:
            hv = 0.0
            for i, obj in enumerate(sorted_objs):
                width = (reference_point[0] - obj[0]) if i == 0 else \
                       (sorted_objs[i-1][0] - obj[0])
                height = reference_point[1] - obj[1]
                hv += width * height
            return hv
        else:
            # For higher dimensions, use approximation
            volumes = np.prod(reference_point - objectives, axis=1)
            return np.sum(volumes)

    def __len__(self) -> int:
        """Number of solutions on front."""
        return len(self.solutions)


# =============================================================================
# Weighted Sum Method
# =============================================================================

class WeightedSumMethod:
    """Weighted sum scalarization method.

    Combines multiple objectives into single objective:
        f_scalar(x) = Σᵢ wᵢ·fᵢ(x)

    By varying weights, can trace out Pareto front.
    """

    def __init__(
        self,
        objectives: List[Callable],
        constraints: Optional[List[Callable]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize weighted sum method.

        Args:
            objectives: List of objective functions
            constraints: List of constraint functions
            bounds: Variable bounds (lower, upper)
        """
        self.objectives = objectives
        self.constraints = constraints or []
        self.bounds = bounds
        self.n_objectives = len(objectives)

    def scalarize(
        self,
        x: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Scalarize objectives with given weights.

        Args:
            x: Decision variables
            weights: Objective weights (must sum to 1)

        Returns:
            Scalar objective value
        """
        return sum(w * f(x) for w, f in zip(weights, self.objectives))

    def optimize_single(
        self,
        weights: np.ndarray,
        x0: Optional[np.ndarray] = None,
        method: str = 'L-BFGS-B',
        **kwargs
    ) -> ParetoSolution:
        """Optimize for single weight vector.

        Args:
            weights: Objective weights
            x0: Initial guess
            method: Optimization method
            **kwargs: Additional optimizer arguments

        Returns:
            Pareto solution
        """
        from scipy.optimize import minimize

        # Normalize weights
        weights = weights / np.sum(weights)

        # Objective function
        def objective(x):
            return self.scalarize(x, weights)

        # Constraints
        constraints_list = []
        for constraint in self.constraints:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x, c=constraint: -c(x)  # g(x) ≤ 0
            })

        # Initial guess
        if x0 is None:
            if self.bounds is not None:
                x0 = (self.bounds[0] + self.bounds[1]) / 2
            else:
                x0 = np.zeros(10)  # Default

        # Optimize
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=None if self.bounds is None else
                   list(zip(self.bounds[0], self.bounds[1])),
            constraints=constraints_list if constraints_list else None,
            **kwargs
        )

        # Compute all objectives
        obj_values = np.array([f(result.x) for f in self.objectives])

        # Compute constraints
        constraint_values = None
        if self.constraints:
            constraint_values = np.array([c(result.x) for c in self.constraints])

        return ParetoSolution(
            decision_variables=result.x,
            objectives=obj_values,
            constraints=constraint_values,
            metadata={'weights': weights, 'success': result.success}
        )

    def compute_pareto_front(
        self,
        n_points: int = 20,
        weight_generation: str = 'uniform',
        x0: Optional[np.ndarray] = None,
        **kwargs
    ) -> ParetoFront:
        """Compute Pareto front by varying weights.

        Args:
            n_points: Number of points to generate
            weight_generation: 'uniform' or 'random'
            x0: Initial guess
            **kwargs: Additional optimizer arguments

        Returns:
            Pareto front
        """
        pareto_front = ParetoFront()

        # Generate weight vectors
        if weight_generation == 'uniform':
            if self.n_objectives == 2:
                # Uniform spacing for 2 objectives
                weights_list = []
                for i in range(n_points):
                    w1 = i / (n_points - 1)
                    w2 = 1 - w1
                    weights_list.append(np.array([w1, w2]))
            else:
                # Random for higher dimensions
                weights_list = np.random.dirichlet(
                    np.ones(self.n_objectives),
                    size=n_points
                )
        else:
            # Random weights
            weights_list = np.random.dirichlet(
                np.ones(self.n_objectives),
                size=n_points
            )

        # Optimize for each weight vector
        for weights in weights_list:
            try:
                solution = self.optimize_single(weights, x0, **kwargs)
                if solution.metadata['success']:
                    pareto_front.add_solution(solution)
            except Exception as e:
                print(f"Warning: Optimization failed for weights {weights}: {e}")

        # Filter dominated solutions
        pareto_front.filter_dominated()

        return pareto_front


# =============================================================================
# ε-Constraint Method
# =============================================================================

class EpsilonConstraintMethod:
    """ε-constraint method for multi-objective optimization.

    Optimizes one objective while constraining others:
        min f₁(x)
        s.t. fᵢ(x) ≤ εᵢ for i = 2, ..., k
             g(x) ≤ 0
             h(x) = 0

    By varying ε values, traces out Pareto front.
    """

    def __init__(
        self,
        objectives: List[Callable],
        constraints: Optional[List[Callable]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        primary_objective: int = 0
    ):
        """Initialize ε-constraint method.

        Args:
            objectives: List of objective functions
            constraints: List of constraint functions
            bounds: Variable bounds
            primary_objective: Index of objective to optimize
        """
        self.objectives = objectives
        self.constraints = constraints or []
        self.bounds = bounds
        self.n_objectives = len(objectives)
        self.primary_idx = primary_objective

    def optimize_single(
        self,
        epsilon_values: np.ndarray,
        x0: Optional[np.ndarray] = None,
        method: str = 'SLSQP',
        **kwargs
    ) -> ParetoSolution:
        """Optimize for single ε vector.

        Args:
            epsilon_values: Constraint values for other objectives
            x0: Initial guess
            method: Optimization method
            **kwargs: Additional arguments

        Returns:
            Pareto solution
        """
        from scipy.optimize import minimize

        # Primary objective
        primary_obj = self.objectives[self.primary_idx]

        # Build constraints
        constraints_list = []

        # Original constraints
        for constraint in self.constraints:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x, c=constraint: -c(x)
            })

        # ε-constraints on other objectives
        eps_idx = 0
        for i, obj in enumerate(self.objectives):
            if i == self.primary_idx:
                continue

            epsilon = epsilon_values[eps_idx]
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x, o=obj, e=epsilon: e - o(x)
            })
            eps_idx += 1

        # Initial guess
        if x0 is None:
            if self.bounds is not None:
                x0 = (self.bounds[0] + self.bounds[1]) / 2
            else:
                x0 = np.zeros(10)

        # Optimize
        result = minimize(
            primary_obj,
            x0,
            method=method,
            bounds=None if self.bounds is None else
                   list(zip(self.bounds[0], self.bounds[1])),
            constraints=constraints_list,
            **kwargs
        )

        # Compute all objectives
        obj_values = np.array([f(result.x) for f in self.objectives])

        return ParetoSolution(
            decision_variables=result.x,
            objectives=obj_values,
            metadata={'epsilon': epsilon_values, 'success': result.success}
        )

    def compute_pareto_front(
        self,
        epsilon_grid: Optional[np.ndarray] = None,
        n_points: int = 20,
        x0: Optional[np.ndarray] = None,
        **kwargs
    ) -> ParetoFront:
        """Compute Pareto front by varying ε values.

        Args:
            epsilon_grid: Grid of ε values to try
            n_points: Number of points (if epsilon_grid not provided)
            x0: Initial guess
            **kwargs: Additional arguments

        Returns:
            Pareto front
        """
        pareto_front = ParetoFront()

        # Generate epsilon grid if not provided
        if epsilon_grid is None:
            # Need to estimate objective ranges first
            # Use weighted sum to get rough bounds
            ws = WeightedSumMethod(self.objectives, self.constraints, self.bounds)
            temp_front = ws.compute_pareto_front(n_points=10, x0=x0)

            if len(temp_front) == 0:
                print("Warning: Could not estimate objective ranges")
                return pareto_front

            obj_matrix = temp_front.get_objectives_matrix()

            # Create grid
            n_secondary = self.n_objectives - 1

            if n_secondary == 1:
                # Single ε parameter
                min_val = np.min(obj_matrix[:, 1 if self.primary_idx == 0 else 0])
                max_val = np.max(obj_matrix[:, 1 if self.primary_idx == 0 else 0])
                epsilon_grid = np.linspace(min_val, max_val, n_points).reshape(-1, 1)
            else:
                # Multiple ε parameters - use random sampling
                mins = np.min(obj_matrix, axis=0)
                maxs = np.max(obj_matrix, axis=0)

                # Remove primary objective
                mins = np.delete(mins, self.primary_idx)
                maxs = np.delete(maxs, self.primary_idx)

                # Random samples in ranges
                epsilon_grid = np.random.uniform(
                    mins,
                    maxs,
                    size=(n_points, n_secondary)
                )

        # Optimize for each ε vector
        for epsilon in epsilon_grid:
            try:
                solution = self.optimize_single(epsilon, x0, **kwargs)
                if solution.metadata['success']:
                    pareto_front.add_solution(solution)
            except Exception as e:
                print(f"Warning: Optimization failed for ε={epsilon}: {e}")

        # Filter dominated
        pareto_front.filter_dominated()

        return pareto_front


# =============================================================================
# Normal Boundary Intersection (NBI)
# =============================================================================

class NormalBoundaryIntersection:
    """Normal Boundary Intersection method.

    Generates evenly-distributed points on Pareto front by:
    1. Computing anchor points (single-objective optima)
    2. Constructing convex hull of anchor points
    3. Finding Pareto points along normal directions
    """

    def __init__(
        self,
        objectives: List[Callable],
        constraints: Optional[List[Callable]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize NBI method.

        Args:
            objectives: List of objective functions
            constraints: List of constraint functions
            bounds: Variable bounds
        """
        self.objectives = objectives
        self.constraints = constraints or []
        self.bounds = bounds
        self.n_objectives = len(objectives)
        self.anchor_points = None
        self.utopia_point = None

    def compute_anchor_points(
        self,
        x0: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Compute anchor points (single-objective optima).

        Args:
            x0: Initial guess
            **kwargs: Optimizer arguments

        Returns:
            Array of anchor points in objective space
        """
        from scipy.optimize import minimize

        anchor_points = []

        for i, obj in enumerate(self.objectives):
            # Optimize single objective
            if x0 is None:
                if self.bounds is not None:
                    x_init = (self.bounds[0] + self.bounds[1]) / 2
                else:
                    x_init = np.zeros(10)
            else:
                x_init = x0

            # Build constraints
            constraints_list = []
            for constraint in self.constraints:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x, c=constraint: -c(x)
                })

            # Optimize
            result = minimize(
                obj,
                x_init,
                bounds=None if self.bounds is None else
                       list(zip(self.bounds[0], self.bounds[1])),
                constraints=constraints_list if constraints_list else None,
                **kwargs
            )

            # Evaluate all objectives at this point
            anchor_obj = np.array([f(result.x) for f in self.objectives])
            anchor_points.append(anchor_obj)

        self.anchor_points = np.array(anchor_points)

        # Utopia point (best of each objective)
        self.utopia_point = np.min(self.anchor_points, axis=0)

        return self.anchor_points

    def compute_pareto_front(
        self,
        n_points: int = 20,
        x0: Optional[np.ndarray] = None,
        **kwargs
    ) -> ParetoFront:
        """Compute Pareto front using NBI.

        Args:
            n_points: Number of points to generate
            x0: Initial guess
            **kwargs: Optimizer arguments

        Returns:
            Pareto front
        """
        # Compute anchor points if not done
        if self.anchor_points is None:
            self.compute_anchor_points(x0, **kwargs)

        # For simplicity, use weighted sum as approximation
        # Full NBI implementation requires solving constrained optimization
        # problems along normal directions
        ws = WeightedSumMethod(self.objectives, self.constraints, self.bounds)
        return ws.compute_pareto_front(n_points, x0=x0, **kwargs)


# =============================================================================
# NSGA-II (Evolutionary Multi-Objective)
# =============================================================================

class NSGA2Optimizer:
    """Non-dominated Sorting Genetic Algorithm II.

    Evolutionary algorithm for multi-objective optimization using:
    - Non-dominated sorting
    - Crowding distance
    - Tournament selection
    """

    def __init__(
        self,
        objectives: List[Callable],
        constraints: Optional[List[Callable]] = None,
        bounds: Tuple[np.ndarray, np.ndarray] = None,
        population_size: int = 100,
        n_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1
    ):
        """Initialize NSGA-II.

        Args:
            objectives: List of objective functions
            constraints: List of constraint functions
            bounds: Variable bounds (required)
            population_size: Population size
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
        """
        self.objectives = objectives
        self.constraints = constraints or []
        self.bounds = bounds
        self.pop_size = population_size
        self.n_gen = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        if bounds is None:
            raise ValueError("Bounds required for NSGA-II")

        self.n_vars = len(bounds[0])
        self.n_objectives = len(objectives)

    def initialize_population(self) -> np.ndarray:
        """Initialize random population.

        Returns:
            Population array of shape (pop_size, n_vars)
        """
        lower, upper = self.bounds
        return np.random.uniform(
            lower,
            upper,
            size=(self.pop_size, self.n_vars)
        )

    def evaluate_population(
        self,
        population: np.ndarray
    ) -> np.ndarray:
        """Evaluate objectives for population.

        Args:
            population: Population array

        Returns:
            Objective values of shape (pop_size, n_objectives)
        """
        objectives = np.zeros((len(population), self.n_objectives))

        for i, individual in enumerate(population):
            for j, obj in enumerate(self.objectives):
                objectives[i, j] = obj(individual)

        return objectives

    def non_dominated_sort(
        self,
        objectives: np.ndarray
    ) -> List[List[int]]:
        """Perform non-dominated sorting.

        Args:
            objectives: Objective values

        Returns:
            List of fronts (each front is list of indices)
        """
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        # Find domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                i_dom_j = (np.all(objectives[i] <= objectives[j]) and
                          np.any(objectives[i] < objectives[j]))

                # Check if j dominates i
                j_dom_i = (np.all(objectives[j] <= objectives[i]) and
                          np.any(objectives[j] < objectives[i]))

                if i_dom_j:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif j_dom_i:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # First front: non-dominated solutions
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Subsequent fronts
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def crowding_distance(
        self,
        objectives: np.ndarray,
        front: List[int]
    ) -> np.ndarray:
        """Compute crowding distance for front.

        Args:
            objectives: Objective values
            front: Indices of solutions in front

        Returns:
            Crowding distances
        """
        n = len(front)
        distances = np.zeros(n)

        if n <= 2:
            return np.full(n, np.inf)

        # For each objective
        for m in range(self.n_objectives):
            # Sort by this objective
            sorted_indices = np.argsort(objectives[front, m])

            # Boundary points have infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Range of this objective
            obj_range = (objectives[front[sorted_indices[-1]], m] -
                        objectives[front[sorted_indices[0]], m])

            if obj_range == 0:
                continue

            # Distance for interior points
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    (objectives[front[sorted_indices[i + 1]], m] -
                     objectives[front[sorted_indices[i - 1]], m]) / obj_range
                )

        return distances

    def tournament_selection(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        fronts: List[List[int]],
        n_select: int
    ) -> np.ndarray:
        """Tournament selection.

        Args:
            population: Population array
            objectives: Objective values
            fronts: Non-dominated fronts
            n_select: Number to select

        Returns:
            Selected individuals
        """
        # Assign rank and crowding distance
        rank = np.zeros(len(population))
        crowding = np.zeros(len(population))

        for i, front in enumerate(fronts):
            rank[front] = i
            crowding[front] = self.crowding_distance(objectives, front)

        selected = []
        for _ in range(n_select):
            # Random tournament
            i, j = np.random.choice(len(population), size=2, replace=False)

            # Compare
            if rank[i] < rank[j]:
                winner = i
            elif rank[i] > rank[j]:
                winner = j
            elif crowding[i] > crowding[j]:
                winner = i
            else:
                winner = j

            selected.append(population[winner].copy())

        return np.array(selected)

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring
        """
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        eta = 20  # Distribution index

        offspring1 = np.zeros_like(parent1)
        offspring2 = np.zeros_like(parent2)

        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                offspring1[i] = 0.5 * ((1 + beta) * parent1[i] +
                                       (1 - beta) * parent2[i])
                offspring2[i] = 0.5 * ((1 - beta) * parent1[i] +
                                       (1 + beta) * parent2[i])
            else:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]

        # Clip to bounds
        lower, upper = self.bounds
        offspring1 = np.clip(offspring1, lower, upper)
        offspring2 = np.clip(offspring2, lower, upper)

        return offspring1, offspring2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        eta = 20  # Distribution index
        lower, upper = self.bounds

        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_prob:
                u = np.random.rand()
                delta_l = (mutated[i] - lower[i]) / (upper[i] - lower[i])
                delta_r = (upper[i] - mutated[i]) / (upper[i] - lower[i])

                if u < 0.5:
                    val = 2 * u + (1 - 2 * u) * (1 - delta_l) ** (eta + 1)
                    delta = val ** (1 / (eta + 1)) - 1
                else:
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_r) ** (eta + 1)
                    delta = 1 - val ** (1 / (eta + 1))

                mutated[i] += delta * (upper[i] - lower[i])
                mutated[i] = np.clip(mutated[i], lower[i], upper[i])

        return mutated

    def optimize(self, verbose: bool = False) -> ParetoFront:
        """Run NSGA-II optimization.

        Args:
            verbose: Print progress

        Returns:
            Pareto front
        """
        # Initialize
        population = self.initialize_population()

        for gen in range(self.n_gen):
            # Evaluate
            objectives = self.evaluate_population(population)

            # Non-dominated sort
            fronts = self.non_dominated_sort(objectives)

            # Selection
            parents = self.tournament_selection(
                population, objectives, fronts, self.pop_size
            )

            # Generate offspring
            offspring = []
            for i in range(0, self.pop_size, 2):
                parent1 = parents[i]
                parent2 = parents[min(i + 1, self.pop_size - 1)]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                offspring.extend([child1, child2])

            offspring = np.array(offspring[:self.pop_size])

            # Combine and select best
            combined = np.vstack([population, offspring])
            combined_obj = self.evaluate_population(combined)
            combined_fronts = self.non_dominated_sort(combined_obj)

            # Select next population
            next_pop = []
            for front in combined_fronts:
                if len(next_pop) + len(front) <= self.pop_size:
                    next_pop.extend(front)
                else:
                    # Fill remaining with best crowding distance
                    remaining = self.pop_size - len(next_pop)
                    crowding = self.crowding_distance(combined_obj, front)
                    sorted_indices = np.argsort(-crowding)
                    next_pop.extend([front[i] for i in sorted_indices[:remaining]])
                    break

            population = combined[next_pop]

            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}/{self.n_gen}")

        # Final evaluation
        objectives = self.evaluate_population(population)
        fronts = self.non_dominated_sort(objectives)

        # Build Pareto front from first front
        pareto_front = ParetoFront()
        for idx in fronts[0]:
            solution = ParetoSolution(
                decision_variables=population[idx],
                objectives=objectives[idx]
            )
            pareto_front.add_solution(solution)

        return pareto_front


# =============================================================================
# Multi-Objective Optimizer Interface
# =============================================================================

class MultiObjectiveOptimizer:
    """Unified interface for multi-objective optimization.

    Provides access to multiple methods through single API.
    """

    def __init__(
        self,
        objectives: List[Callable],
        constraints: Optional[List[Callable]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize multi-objective optimizer.

        Args:
            objectives: List of objective functions
            constraints: List of constraint functions
            bounds: Variable bounds
        """
        self.objectives = objectives
        self.constraints = constraints
        self.bounds = bounds

    def optimize(
        self,
        method: str = 'weighted_sum',
        n_points: int = 20,
        **kwargs
    ) -> ParetoFront:
        """Optimize using specified method.

        Args:
            method: 'weighted_sum', 'epsilon_constraint', 'nbi', or 'nsga2'
            n_points: Number of Pareto points to generate
            **kwargs: Method-specific arguments

        Returns:
            Pareto front
        """
        if method == 'weighted_sum':
            optimizer = WeightedSumMethod(
                self.objectives,
                self.constraints,
                self.bounds
            )
            return optimizer.compute_pareto_front(n_points, **kwargs)

        elif method == 'epsilon_constraint':
            optimizer = EpsilonConstraintMethod(
                self.objectives,
                self.constraints,
                self.bounds
            )
            return optimizer.compute_pareto_front(n_points=n_points, **kwargs)

        elif method == 'nbi':
            optimizer = NormalBoundaryIntersection(
                self.objectives,
                self.constraints,
                self.bounds
            )
            return optimizer.compute_pareto_front(n_points, **kwargs)

        elif method == 'nsga2':
            if self.bounds is None:
                raise ValueError("NSGA-II requires bounds")

            optimizer = NSGA2Optimizer(
                self.objectives,
                self.constraints,
                self.bounds,
                **kwargs
            )
            return optimizer.optimize()

        else:
            raise ValueError(f"Unknown method: {method}")
