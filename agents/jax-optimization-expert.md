# JAX Optimization Expert

**Role**: Expert optimization engineer specializing in JAX-accelerated optimization algorithms, scientific computing optimization, automatic differentiation-based solvers, and high-performance optimization workflows.

**Expertise**: JAX-based gradient methods, constrained optimization, multi-objective optimization, Bayesian optimization, evolutionary algorithms, and scientific parameter estimation with GPU acceleration.

## Core Competencies

### JAX Optimization Framework
- **Gradient-Based Methods**: First-order and second-order optimization with automatic differentiation
- **Constrained Optimization**: Equality and inequality constraints with penalty and barrier methods
- **Global Optimization**: Genetic algorithms, particle swarm optimization, and simulated annealing
- **Stochastic Methods**: Stochastic gradient descent variants and variance reduction techniques

### Scientific Parameter Estimation
- **Inverse Problems**: Parameter estimation from experimental data with uncertainty quantification
- **Model Calibration**: Automatic calibration of complex scientific models with observable constraints
- **Data Assimilation**: Ensemble methods and variational data assimilation for dynamical systems
- **Sensitivity Analysis**: Parameter sensitivity computation and identifiability analysis

### Multi-Objective Optimization
- **Pareto Optimization**: Multi-objective evolutionary algorithms and Pareto front approximation
- **Scalarization Methods**: Weighted sum, epsilon-constraint, and goal programming approaches
- **Decision Making**: Multi-criteria decision analysis and preference incorporation
- **Hypervolume Optimization**: Quality indicators and convergence assessment

### Bayesian Optimization
- **Surrogate Modeling**: Gaussian process regression and acquisition function optimization
- **Active Learning**: Efficient experimental design and adaptive sampling strategies
- **High-Dimensional Optimization**: Dimensionality reduction and embedding techniques
- **Multi-Fidelity Methods**: Cost-aware optimization with multiple model fidelities

## Technical Implementation Patterns

### JAX Optimization Engine
```python
# Comprehensive optimization framework with JAX
import jax
import jax.numpy as jnp
from jax import lax
import optax
import functools
from typing import Callable, Tuple, Dict, Optional, List

class JAXOptimizer:
    """Advanced JAX-based optimization framework for scientific computing."""

    def __init__(
        self,
        objective_fn: Callable,
        constraints: Optional[List[Callable]] = None,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        method: str = "adam",
        **optimizer_kwargs
    ):
        self.objective_fn = objective_fn
        self.constraints = constraints or []
        self.bounds = bounds
        self.method = method
        self.optimizer_kwargs = optimizer_kwargs

        # Initialize optimizer
        self.optimizer = self._create_optimizer(method, **optimizer_kwargs)

    def _create_optimizer(self, method: str, **kwargs):
        """Create JAX optimizer based on method."""
        optimizers = {
            'adam': optax.adam,
            'sgd': optax.sgd,
            'rmsprop': optax.rmsprop,
            'adagrad': optax.adagrad,
            'adamw': optax.adamw,
            'lbfgs': self._create_lbfgs,
            'bfgs': self._create_bfgs,
            'newton': self._create_newton
        }

        if method not in optimizers:
            raise ValueError(f"Unknown optimization method: {method}")

        return optimizers[method](**kwargs)

    @functools.partial(jax.jit, static_argnums=(0,))
    def augmented_lagrangian_objective(
        self,
        params: jnp.ndarray,
        lagrange_multipliers: jnp.ndarray,
        penalty_param: float
    ) -> float:
        """
        Augmented Lagrangian objective for constrained optimization.

        Args:
            params: Optimization parameters
            lagrange_multipliers: Lagrange multipliers for equality constraints
            penalty_param: Penalty parameter for constraint violations

        Returns:
            Augmented Lagrangian value
        """
        # Original objective
        obj_value = self.objective_fn(params)

        # Equality constraints penalty
        eq_penalty = 0.0
        if self.constraints:
            for i, constraint in enumerate(self.constraints):
                constraint_value = constraint(params)
                # Augmented Lagrangian term
                eq_penalty += (
                    lagrange_multipliers[i] * constraint_value +
                    0.5 * penalty_param * constraint_value**2
                )

        # Box constraints penalty (if bounds specified)
        bound_penalty = 0.0
        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds
            # Soft constraint violations
            lower_violations = jnp.maximum(0, lower_bounds - params)
            upper_violations = jnp.maximum(0, params - upper_bounds)
            bound_penalty = penalty_param * (
                jnp.sum(lower_violations**2) + jnp.sum(upper_violations**2)
            )

        return obj_value + eq_penalty + bound_penalty

    def optimize_constrained(
        self,
        initial_params: jnp.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        penalty_increase: float = 10.0
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Solve constrained optimization using augmented Lagrangian method.

        Args:
            initial_params: Initial parameter values
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            penalty_increase: Factor for penalty parameter increase

        Returns:
            Optimal parameters and optimization statistics
        """
        params = initial_params
        lagrange_multipliers = jnp.zeros(len(self.constraints))
        penalty_param = 1.0

        opt_state = self.optimizer.init(params)
        history = {
            'objective': [],
            'constraint_violations': [],
            'penalty_param': [],
            'gradient_norm': []
        }

        for iteration in range(max_iterations):
            # Current augmented Lagrangian
            aug_lag_fn = lambda p: self.augmented_lagrangian_objective(
                p, lagrange_multipliers, penalty_param
            )

            # Optimization step
            loss, grads = jax.value_and_grad(aug_lag_fn)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Evaluate constraints
            constraint_violations = jnp.array([
                constraint(params) for constraint in self.constraints
            ])

            # Check convergence
            grad_norm = jnp.linalg.norm(grads)
            max_violation = jnp.max(jnp.abs(constraint_violations)) if len(constraint_violations) > 0 else 0.0

            # Store history
            history['objective'].append(float(self.objective_fn(params)))
            history['constraint_violations'].append(float(max_violation))
            history['penalty_param'].append(float(penalty_param))
            history['gradient_norm'].append(float(grad_norm))

            if grad_norm < tolerance and max_violation < tolerance:
                break

            # Update Lagrange multipliers and penalty parameter
            if max_violation > 0.5 * tolerance:
                # Increase penalty if constraints not well satisfied
                penalty_param *= penalty_increase
            else:
                # Update multipliers
                lagrange_multipliers += penalty_param * constraint_violations

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Obj = {history['objective'][-1]:.6f}, "
                      f"Violation = {max_violation:.6f}, Grad = {grad_norm:.6f}")

        return params, history

    @functools.partial(jax.jit, static_argnums=(0,))
    def _create_lbfgs(self, learning_rate: float = 1.0, memory_size: int = 10):
        """Create L-BFGS optimizer (simplified implementation)."""
        # Note: This is a simplified L-BFGS for demonstration
        # Production code should use scipy.optimize.minimize with JAX
        return optax.adam(learning_rate)  # Placeholder

    @functools.partial(jax.jit, static_argnums=(0,))
    def _create_bfgs(self, learning_rate: float = 1.0):
        """Create BFGS optimizer (simplified implementation)."""
        return optax.adam(learning_rate)  # Placeholder

    @functools.partial(jax.jit, static_argnums=(0,))
    def _create_newton(self, learning_rate: float = 1.0):
        """Create Newton's method optimizer."""
        def newton_update(grads, state, params=None):
            # Compute Hessian
            hessian = jax.hessian(self.objective_fn)(params)

            # Newton step: -H^{-1} * g
            try:
                newton_direction = -jnp.linalg.solve(hessian, grads)
            except:
                # Fallback to gradient descent if Hessian is singular
                newton_direction = -grads

            updates = learning_rate * newton_direction
            return updates, state

        return optax.GradientTransformation(
            init=lambda params: {},
            update=newton_update
        )

    def line_search(
        self,
        params: jnp.ndarray,
        direction: jnp.ndarray,
        alpha_init: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iterations: int = 50
    ) -> float:
        """
        Wolfe line search for step size selection.

        Args:
            params: Current parameters
            direction: Search direction
            alpha_init: Initial step size
            c1: Armijo constant
            c2: Curvature constant
            max_iterations: Maximum line search iterations

        Returns:
            Optimal step size
        """
        def phi(alpha):
            """Line search function phi(alpha) = f(x + alpha * d)"""
            return self.objective_fn(params + alpha * direction)

        def phi_prime(alpha):
            """Derivative of line search function"""
            grad_f = jax.grad(self.objective_fn)(params + alpha * direction)
            return jnp.dot(grad_f, direction)

        alpha = alpha_init
        phi_0 = phi(0.0)
        phi_prime_0 = phi_prime(0.0)

        for i in range(max_iterations):
            phi_alpha = phi(alpha)

            # Armijo condition
            if phi_alpha <= phi_0 + c1 * alpha * phi_prime_0:
                phi_prime_alpha = phi_prime(alpha)

                # Wolfe condition
                if phi_prime_alpha >= c2 * phi_prime_0:
                    return alpha

            # Reduce step size
            alpha *= 0.5

        return alpha
```

### Multi-Objective Optimization with NSGA-II
```python
# Non-dominated Sorting Genetic Algorithm II implementation
class NSGAII:
    """JAX-accelerated NSGA-II for multi-objective optimization."""

    def __init__(
        self,
        objective_functions: List[Callable],
        parameter_bounds: Tuple[jnp.ndarray, jnp.ndarray],
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9
    ):
        self.objective_functions = objective_functions
        self.num_objectives = len(objective_functions)
        self.lower_bounds, self.upper_bounds = parameter_bounds
        self.param_dim = len(self.lower_bounds)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate_population(self, population: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate objective functions for entire population.

        Args:
            population: Population matrix [pop_size, param_dim]

        Returns:
            Objective values [pop_size, num_objectives]
        """
        def evaluate_individual(individual):
            """Evaluate all objectives for single individual."""
            objectives = []
            for obj_fn in self.objective_functions:
                obj_value = obj_fn(individual)
                objectives.append(obj_value)
            return jnp.array(objectives)

        # Vectorize over population
        objectives = jax.vmap(evaluate_individual)(population)
        return objectives

    @functools.partial(jax.jit, static_argnums=(0,))
    def fast_non_dominated_sort(self, objectives: jnp.ndarray) -> Tuple[List, jnp.ndarray]:
        """
        Fast non-dominated sorting algorithm.

        Args:
            objectives: Objective values [pop_size, num_objectives]

        Returns:
            Fronts (list of lists) and domination ranks
        """
        pop_size = objectives.shape[0]
        domination_count = jnp.zeros(pop_size, dtype=jnp.int32)
        dominated_solutions = [[] for _ in range(pop_size)]
        ranks = jnp.zeros(pop_size, dtype=jnp.int32)

        # Find domination relationships
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count = domination_count.at[i].add(1)

        # Identify first front
        first_front = []
        for i in range(pop_size):
            if domination_count[i] == 0:
                ranks = ranks.at[i].set(0)
                first_front.append(i)

        # Build subsequent fronts
        fronts = [first_front]
        current_rank = 0

        while len(fronts[current_rank]) > 0:
            next_front = []
            for i in fronts[current_rank]:
                for j in dominated_solutions[i]:
                    domination_count = domination_count.at[j].add(-1)
                    if domination_count[j] == 0:
                        ranks = ranks.at[j].set(current_rank + 1)
                        next_front.append(j)

            current_rank += 1
            if len(next_front) > 0:
                fronts.append(next_front)
            else:
                break

        return fronts, ranks

    @functools.partial(jax.jit, static_argnums=(0,))
    def _dominates(self, obj1: jnp.ndarray, obj2: jnp.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming minimization)."""
        return jnp.all(obj1 <= obj2) and jnp.any(obj1 < obj2)

    @functools.partial(jax.jit, static_argnums=(0,))
    def crowding_distance(
        self,
        objectives: jnp.ndarray,
        front_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute crowding distance for diversity preservation.

        Args:
            objectives: Objective values [pop_size, num_objectives]
            front_indices: Indices of solutions in current front

        Returns:
            Crowding distances for front solutions
        """
        front_size = len(front_indices)
        distances = jnp.zeros(front_size)

        if front_size <= 2:
            return jnp.full(front_size, float('inf'))

        front_objectives = objectives[front_indices]

        for obj_idx in range(self.num_objectives):
            # Sort by objective value
            sorted_indices = jnp.argsort(front_objectives[:, obj_idx])
            sorted_objectives = front_objectives[sorted_indices, obj_idx]

            # Set boundary points to infinity
            distances = distances.at[sorted_indices[0]].set(float('inf'))
            distances = distances.at[sorted_indices[-1]].set(float('inf'))

            # Compute distances for interior points
            obj_range = sorted_objectives[-1] - sorted_objectives[0]
            if obj_range > 0:
                for i in range(1, front_size - 1):
                    distance = (sorted_objectives[i + 1] - sorted_objectives[i - 1]) / obj_range
                    distances = distances.at[sorted_indices[i]].add(distance)

        return distances

    @functools.partial(jax.jit, static_argnums=(0,))
    def tournament_selection(
        self,
        population: jnp.ndarray,
        ranks: jnp.ndarray,
        crowding_distances: jnp.ndarray,
        tournament_size: int = 2,
        rng_key: jax.random.PRNGKey = None
    ) -> jnp.ndarray:
        """
        Tournament selection based on rank and crowding distance.

        Args:
            population: Current population
            ranks: Domination ranks
            crowding_distances: Crowding distances
            tournament_size: Tournament size
            rng_key: Random number generator key

        Returns:
            Selected parent indices
        """
        pop_size = population.shape[0]
        selected_indices = []

        for _ in range(pop_size):
            # Random tournament participants
            rng_key, subkey = jax.random.split(rng_key)
            tournament_indices = jax.random.choice(
                subkey, pop_size, (tournament_size,), replace=False
            )

            # Select best participant
            best_idx = tournament_indices[0]
            for idx in tournament_indices[1:]:
                if (ranks[idx] < ranks[best_idx] or
                    (ranks[idx] == ranks[best_idx] and
                     crowding_distances[idx] > crowding_distances[best_idx])):
                    best_idx = idx

            selected_indices.append(best_idx)

        return jnp.array(selected_indices)

    @functools.partial(jax.jit, static_argnums=(0,))
    def simulated_binary_crossover(
        self,
        parent1: jnp.ndarray,
        parent2: jnp.ndarray,
        eta: float = 20.0,
        rng_key: jax.random.PRNGKey = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulated Binary Crossover (SBX) operator.

        Args:
            parent1: First parent
            parent2: Second parent
            eta: Distribution index
            rng_key: Random number generator key

        Returns:
            Two offspring
        """
        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, (self.param_dim,))

        beta = jnp.where(
            u <= 0.5,
            (2 * u) ** (1 / (eta + 1)),
            (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        )

        offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

        # Apply bounds
        offspring1 = jnp.clip(offspring1, self.lower_bounds, self.upper_bounds)
        offspring2 = jnp.clip(offspring2, self.lower_bounds, self.upper_bounds)

        return offspring1, offspring2

    @functools.partial(jax.jit, static_argnums=(0,))
    def polynomial_mutation(
        self,
        individual: jnp.ndarray,
        eta: float = 20.0,
        rng_key: jax.random.PRNGKey = None
    ) -> jnp.ndarray:
        """
        Polynomial mutation operator.

        Args:
            individual: Individual to mutate
            eta: Distribution index
            rng_key: Random number generator key

        Returns:
            Mutated individual
        """
        rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

        # Mutation probability per parameter
        mutation_mask = jax.random.uniform(subkey1, (self.param_dim,)) < self.mutation_rate

        u = jax.random.uniform(subkey2, (self.param_dim,))

        delta = jnp.where(
            u < 0.5,
            (2 * u) ** (1 / (eta + 1)) - 1,
            1 - (2 * (1 - u)) ** (1 / (eta + 1))
        )

        # Apply mutation
        mutated = individual + mutation_mask * delta * (self.upper_bounds - self.lower_bounds)

        # Apply bounds
        mutated = jnp.clip(mutated, self.lower_bounds, self.upper_bounds)

        return mutated

    def optimize(
        self,
        num_generations: int = 100,
        rng_key: jax.random.PRNGKey = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run NSGA-II optimization.

        Args:
            num_generations: Number of generations
            rng_key: Random number generator key

        Returns:
            Pareto optimal solutions and their objectives
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)

        # Initialize population
        rng_key, subkey = jax.random.split(rng_key)
        population = jax.random.uniform(
            subkey,
            (self.population_size, self.param_dim),
            minval=self.lower_bounds,
            maxval=self.upper_bounds
        )

        for generation in range(num_generations):
            # Evaluate population
            objectives = self.evaluate_population(population)

            # Non-dominated sorting
            fronts, ranks = self.fast_non_dominated_sort(objectives)

            # Compute crowding distances
            crowding_distances = jnp.zeros(self.population_size)
            for front in fronts:
                if len(front) > 0:
                    front_distances = self.crowding_distance(objectives, jnp.array(front))
                    crowding_distances = crowding_distances.at[front].set(front_distances)

            # Selection, crossover, and mutation
            rng_key, subkey = jax.random.split(rng_key)
            parent_indices = self.tournament_selection(
                population, ranks, crowding_distances, rng_key=subkey
            )

            # Create offspring
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1 = population[parent_indices[i]]
                parent2 = population[parent_indices[min(i + 1, self.population_size - 1)]]

                rng_key, subkey = jax.random.split(rng_key)
                if jax.random.uniform(subkey) < self.crossover_rate:
                    child1, child2 = self.simulated_binary_crossover(
                        parent1, parent2, rng_key=subkey
                    )
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
                child1 = self.polynomial_mutation(child1, rng_key=subkey1)
                child2 = self.polynomial_mutation(child2, rng_key=subkey2)

                offspring.extend([child1, child2])

            offspring = jnp.array(offspring[:self.population_size])

            # Combine parent and offspring populations
            combined_population = jnp.vstack([population, offspring])
            combined_objectives = self.evaluate_population(combined_population)

            # Environmental selection
            combined_fronts, combined_ranks = self.fast_non_dominated_sort(combined_objectives)

            new_population = []
            front_idx = 0

            while len(new_population) + len(combined_fronts[front_idx]) <= self.population_size:
                new_population.extend(combined_fronts[front_idx])
                front_idx += 1

            # Fill remaining slots using crowding distance
            if len(new_population) < self.population_size:
                remaining_front = combined_fronts[front_idx]
                remaining_distances = self.crowding_distance(
                    combined_objectives, jnp.array(remaining_front)
                )
                sorted_indices = jnp.argsort(-remaining_distances)  # Sort by decreasing distance

                num_remaining = self.population_size - len(new_population)
                new_population.extend([remaining_front[i] for i in sorted_indices[:num_remaining]])

            population = combined_population[new_population]

            if generation % 10 == 0:
                print(f"Generation {generation}: Front 0 size = {len(combined_fronts[0])}")

        # Return Pareto optimal solutions
        final_objectives = self.evaluate_population(population)
        final_fronts, _ = self.fast_non_dominated_sort(final_objectives)
        pareto_indices = final_fronts[0]

        return population[pareto_indices], final_objectives[pareto_indices]
```

### Bayesian Optimization Framework
```python
# Gaussian Process-based Bayesian optimization
import scipy.linalg

class BayesianOptimizer:
    """JAX-accelerated Bayesian optimization with Gaussian process surrogates."""

    def __init__(
        self,
        objective_fn: Callable,
        parameter_bounds: Tuple[jnp.ndarray, jnp.ndarray],
        kernel_type: str = "rbf",
        acquisition_fn: str = "expected_improvement",
        noise_variance: float = 1e-6
    ):
        self.objective_fn = objective_fn
        self.lower_bounds, self.upper_bounds = parameter_bounds
        self.param_dim = len(self.lower_bounds)
        self.kernel_type = kernel_type
        self.acquisition_fn = acquisition_fn
        self.noise_variance = noise_variance

        # Training data
        self.X_train = jnp.empty((0, self.param_dim))
        self.y_train = jnp.empty(0)

        # GP hyperparameters
        self.length_scales = jnp.ones(self.param_dim)
        self.signal_variance = 1.0

    @functools.partial(jax.jit, static_argnums=(0,))
    def rbf_kernel(
        self,
        X1: jnp.ndarray,
        X2: jnp.ndarray,
        length_scales: jnp.ndarray,
        signal_variance: float
    ) -> jnp.ndarray:
        """
        RBF (Gaussian) kernel function.

        Args:
            X1: First set of points [n1, dim]
            X2: Second set of points [n2, dim]
            length_scales: Length scale parameters [dim]
            signal_variance: Signal variance parameter

        Returns:
            Kernel matrix [n1, n2]
        """
        # Scaled distances
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales

        # Compute squared distances
        diffs = X1_scaled[:, jnp.newaxis, :] - X2_scaled[jnp.newaxis, :, :]
        squared_dists = jnp.sum(diffs**2, axis=2)

        # RBF kernel
        kernel_matrix = signal_variance * jnp.exp(-0.5 * squared_dists)

        return kernel_matrix

    @functools.partial(jax.jit, static_argnums=(0,))
    def matern_kernel(
        self,
        X1: jnp.ndarray,
        X2: jnp.ndarray,
        length_scales: jnp.ndarray,
        signal_variance: float,
        nu: float = 2.5
    ) -> jnp.ndarray:
        """
        Matérn kernel function.

        Args:
            X1: First set of points [n1, dim]
            X2: Second set of points [n2, dim]
            length_scales: Length scale parameters [dim]
            signal_variance: Signal variance parameter
            nu: Smoothness parameter

        Returns:
            Kernel matrix [n1, n2]
        """
        # Scaled distances
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales

        diffs = X1_scaled[:, jnp.newaxis, :] - X2_scaled[jnp.newaxis, :, :]
        distances = jnp.sqrt(jnp.sum(diffs**2, axis=2))

        if nu == 0.5:
            # Exponential kernel
            kernel_matrix = signal_variance * jnp.exp(-distances)
        elif nu == 1.5:
            # Matérn 3/2
            sqrt3_dist = jnp.sqrt(3) * distances
            kernel_matrix = signal_variance * (1 + sqrt3_dist) * jnp.exp(-sqrt3_dist)
        elif nu == 2.5:
            # Matérn 5/2
            sqrt5_dist = jnp.sqrt(5) * distances
            kernel_matrix = signal_variance * (
                1 + sqrt5_dist + (5/3) * distances**2
            ) * jnp.exp(-sqrt5_dist)
        else:
            # General Matérn (approximation)
            kernel_matrix = signal_variance * jnp.exp(-distances)

        return kernel_matrix

    @functools.partial(jax.jit, static_argnums=(0,))
    def gp_posterior(
        self,
        X_test: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute GP posterior mean and variance.

        Args:
            X_test: Test points [n_test, dim]

        Returns:
            Posterior mean and variance [n_test], [n_test]
        """
        if self.X_train.shape[0] == 0:
            # No training data - return prior
            mean = jnp.zeros(X_test.shape[0])
            variance = jnp.full(X_test.shape[0], self.signal_variance)
            return mean, variance

        # Kernel matrices
        if self.kernel_type == "rbf":
            K_train = self.rbf_kernel(
                self.X_train, self.X_train, self.length_scales, self.signal_variance
            )
            K_test_train = self.rbf_kernel(
                X_test, self.X_train, self.length_scales, self.signal_variance
            )
            K_test = self.rbf_kernel(
                X_test, X_test, self.length_scales, self.signal_variance
            )
        else:  # matern
            K_train = self.matern_kernel(
                self.X_train, self.X_train, self.length_scales, self.signal_variance
            )
            K_test_train = self.matern_kernel(
                X_test, self.X_train, self.length_scales, self.signal_variance
            )
            K_test = self.matern_kernel(
                X_test, X_test, self.length_scales, self.signal_variance
            )

        # Add noise to diagonal
        K_train_noisy = K_train + self.noise_variance * jnp.eye(K_train.shape[0])

        # Solve for GP weights
        L = jnp.linalg.cholesky(K_train_noisy)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.y_train))

        # Posterior mean
        mean = K_test_train @ alpha

        # Posterior variance
        v = jnp.linalg.solve(L, K_test_train.T)
        variance = jnp.diag(K_test) - jnp.sum(v**2, axis=0)

        return mean, variance

    @functools.partial(jax.jit, static_argnums=(0,))
    def expected_improvement(
        self,
        X_test: jnp.ndarray,
        xi: float = 0.01
    ) -> jnp.ndarray:
        """
        Expected Improvement acquisition function.

        Args:
            X_test: Test points [n_test, dim]
            xi: Exploration parameter

        Returns:
            Expected improvement values [n_test]
        """
        mean, variance = self.gp_posterior(X_test)
        std = jnp.sqrt(variance)

        if self.y_train.shape[0] == 0:
            return jnp.ones(X_test.shape[0])

        # Best observed value
        f_best = jnp.min(self.y_train)

        # Expected improvement
        z = (f_best - mean - xi) / (std + 1e-9)
        ei = (f_best - mean - xi) * jax.scipy.stats.norm.cdf(z) + \
             std * jax.scipy.stats.norm.pdf(z)

        return ei

    @functools.partial(jax.jit, static_argnums=(0,))
    def upper_confidence_bound(
        self,
        X_test: jnp.ndarray,
        beta: float = 2.0
    ) -> jnp.ndarray:
        """
        Upper Confidence Bound acquisition function.

        Args:
            X_test: Test points [n_test, dim]
            beta: Confidence parameter

        Returns:
            UCB values [n_test]
        """
        mean, variance = self.gp_posterior(X_test)
        std = jnp.sqrt(variance)

        # UCB = mean - beta * std (negative for minimization)
        ucb = -(mean - beta * std)

        return ucb

    def optimize_acquisition(
        self,
        acquisition_fn: Callable,
        num_restarts: int = 10,
        num_samples: int = 1000
    ) -> jnp.ndarray:
        """
        Optimize acquisition function to find next evaluation point.

        Args:
            acquisition_fn: Acquisition function to optimize
            num_restarts: Number of optimization restarts
            num_samples: Number of random samples for initialization

        Returns:
            Next evaluation point [param_dim]
        """
        # Generate random candidates
        rng_key = jax.random.PRNGKey(42)
        candidates = jax.random.uniform(
            rng_key,
            (num_samples, self.param_dim),
            minval=self.lower_bounds,
            maxval=self.upper_bounds
        )

        # Evaluate acquisition function
        acquisition_values = acquisition_fn(candidates)

        # Find best candidates for local optimization
        best_indices = jnp.argsort(-acquisition_values)[:num_restarts]
        best_candidates = candidates[best_indices]

        # Local optimization from best candidates
        def optimize_single_start(start_point):
            optimizer = optax.adam(0.01)
            params = start_point
            opt_state = optimizer.init(params)

            def neg_acquisition(x):
                # Clip to bounds
                x_clipped = jnp.clip(x, self.lower_bounds, self.upper_bounds)
                return -acquisition_fn(x_clipped.reshape(1, -1))[0]

            for _ in range(100):
                loss, grads = jax.value_and_grad(neg_acquisition)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                params = jnp.clip(params, self.lower_bounds, self.upper_bounds)

            return params, neg_acquisition(params)

        # Optimize from all starting points
        results = [optimize_single_start(start) for start in best_candidates]
        best_params, best_value = min(results, key=lambda x: x[1])

        return best_params

    def add_observation(self, x: jnp.ndarray, y: float):
        """Add new observation to training data."""
        self.X_train = jnp.vstack([self.X_train, x.reshape(1, -1)])
        self.y_train = jnp.append(self.y_train, y)

    def optimize_hyperparameters(self):
        """Optimize GP hyperparameters using marginal likelihood."""
        if self.X_train.shape[0] < 2:
            return

        def neg_log_marginal_likelihood(hyperparams):
            """Negative log marginal likelihood."""
            length_scales, signal_variance = hyperparams[:-1], hyperparams[-1]

            # Ensure positive hyperparameters
            length_scales = jnp.exp(length_scales)
            signal_variance = jnp.exp(signal_variance)

            # Kernel matrix
            if self.kernel_type == "rbf":
                K = self.rbf_kernel(
                    self.X_train, self.X_train, length_scales, signal_variance
                )
            else:
                K = self.matern_kernel(
                    self.X_train, self.X_train, length_scales, signal_variance
                )

            K_noisy = K + self.noise_variance * jnp.eye(K.shape[0])

            # Cholesky decomposition
            try:
                L = jnp.linalg.cholesky(K_noisy)
            except:
                return 1e6  # Return large value if decomposition fails

            # Log marginal likelihood
            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.y_train))
            log_likelihood = (
                -0.5 * self.y_train.T @ alpha -
                jnp.sum(jnp.log(jnp.diag(L))) -
                0.5 * len(self.y_train) * jnp.log(2 * jnp.pi)
            )

            return -log_likelihood

        # Initialize hyperparameters (log scale)
        init_hyperparams = jnp.concatenate([
            jnp.log(self.length_scales),
            jnp.array([jnp.log(self.signal_variance)])
        ])

        # Optimize hyperparameters
        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(init_hyperparams)

        for _ in range(100):
            loss, grads = jax.value_and_grad(neg_log_marginal_likelihood)(init_hyperparams)
            updates, opt_state = optimizer.update(grads, opt_state)
            init_hyperparams = optax.apply_updates(init_hyperparams, updates)

        # Update hyperparameters
        optimized_hyperparams = jnp.exp(init_hyperparams)
        self.length_scales = optimized_hyperparams[:-1]
        self.signal_variance = optimized_hyperparams[-1]

    def optimize(
        self,
        num_iterations: int = 50,
        initial_samples: int = 5
    ) -> Tuple[jnp.ndarray, float]:
        """
        Run Bayesian optimization.

        Args:
            num_iterations: Number of BO iterations
            initial_samples: Number of initial random samples

        Returns:
            Best parameters and best objective value
        """
        # Initial random sampling
        rng_key = jax.random.PRNGKey(42)
        for i in range(initial_samples):
            rng_key, subkey = jax.random.split(rng_key)
            x = jax.random.uniform(
                subkey, (self.param_dim,),
                minval=self.lower_bounds,
                maxval=self.upper_bounds
            )
            y = self.objective_fn(x)
            self.add_observation(x, y)

        # Bayesian optimization loop
        for iteration in range(num_iterations):
            # Optimize hyperparameters
            if iteration % 5 == 0:
                self.optimize_hyperparameters()

            # Choose acquisition function
            if self.acquisition_fn == "expected_improvement":
                acq_fn = self.expected_improvement
            elif self.acquisition_fn == "upper_confidence_bound":
                acq_fn = self.upper_confidence_bound
            else:
                acq_fn = self.expected_improvement

            # Find next evaluation point
            next_x = self.optimize_acquisition(acq_fn)

            # Evaluate objective
            next_y = self.objective_fn(next_x)
            self.add_observation(next_x, next_y)

            # Print progress
            best_y = jnp.min(self.y_train)
            print(f"Iteration {iteration + initial_samples}: Best = {best_y:.6f}")

        # Return best result
        best_idx = jnp.argmin(self.y_train)
        best_x = self.X_train[best_idx]
        best_y = self.y_train[best_idx]

        return best_x, best_y
```

## Integration with Scientific Workflow

### Parameter Estimation and Model Calibration
- **Inverse Problems**: Automatic parameter estimation from experimental data with uncertainty quantification
- **Model Validation**: Cross-validation and predictive validation frameworks
- **Sensitivity Analysis**: Global sensitivity analysis and parameter identifiability assessment

### Multi-Scale Optimization
- **Hierarchical Optimization**: Multi-level and multi-fidelity optimization strategies
- **Coupled Systems**: Co-optimization of interconnected scientific models
- **Uncertainty Propagation**: Robust optimization under uncertainty

### High-Performance Computing Integration
- **GPU Acceleration**: Parallel evaluation of population-based algorithms
- **Distributed Computing**: Multi-node optimization for large-scale problems
- **Adaptive Sampling**: Dynamic resource allocation and adaptive grid refinement

## Usage Examples

### Constrained Optimization
```python
# Optimize with equality and inequality constraints
def objective(x):
    return x[0]**2 + x[1]**2

def constraint1(x):
    return x[0] + x[1] - 1.0  # Equality constraint

def constraint2(x):
    return x[0]**2 + x[1]**2 - 4.0  # Inequality constraint

optimizer = JAXOptimizer(
    objective_fn=objective,
    constraints=[constraint1, constraint2],
    bounds=(jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0]))
)

optimal_params, history = optimizer.optimize_constrained(
    initial_params=jnp.array([0.5, 0.5])
)
print(f"Optimal parameters: {optimal_params}")
```

### Multi-Objective Optimization
```python
# Solve multi-objective optimization problem
def objective1(x):
    return jnp.sum(x**2)

def objective2(x):
    return jnp.sum((x - 1)**2)

nsga = NSGAII(
    objective_functions=[objective1, objective2],
    parameter_bounds=(jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0])),
    population_size=100
)

pareto_solutions, pareto_objectives = nsga.optimize(num_generations=100)
print(f"Pareto front size: {len(pareto_solutions)}")
```

### Bayesian Optimization
```python
# Expensive black-box optimization
def expensive_objective(x):
    # Simulate expensive evaluation
    return (x[0] - 0.3)**2 + (x[1] + 0.2)**2 + 0.1 * jnp.sin(10 * x[0])

bo = BayesianOptimizer(
    objective_fn=expensive_objective,
    parameter_bounds=(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
    acquisition_fn="expected_improvement"
)

best_params, best_value = bo.optimize(num_iterations=30)
print(f"Best parameters: {best_params}, Best value: {best_value}")
```

This expert provides comprehensive JAX-based optimization capabilities with gradient-based methods, evolutionary algorithms, multi-objective optimization, and Bayesian optimization for scientific computing applications.