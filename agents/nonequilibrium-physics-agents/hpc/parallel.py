"""Parallel Optimization for Optimal Control.

This module provides parallel optimization capabilities for hyperparameter
tuning and parameter sweeps.

Features:
1. Grid search with parallel evaluation
2. Random search with adaptive sampling
3. Bayesian optimization (if scikit-optimize available)
4. Multi-objective optimization
5. Result aggregation and analysis

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
from itertools import product

# Try to import optimization libraries
try:
    from scipy.optimize import differential_evolution, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


# =============================================================================
# Parameter Specifications
# =============================================================================

@dataclass
class ParameterSpec:
    """Specification for a hyperparameter.

    Attributes:
        name: Parameter name
        param_type: "continuous", "integer", or "categorical"
        lower: Lower bound (for continuous/integer)
        upper: Upper bound (for continuous/integer)
        choices: List of choices (for categorical)
        log_scale: Whether to sample on log scale
    """
    name: str
    param_type: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False

    def sample(self, rng: Optional[np.random.Generator] = None) -> Any:
        """Sample a value from parameter space.

        Args:
            rng: Random number generator

        Returns:
            Sampled value
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.param_type == "continuous":
            if self.log_scale:
                log_lower = np.log10(self.lower)
                log_upper = np.log10(self.upper)
                log_value = rng.uniform(log_lower, log_upper)
                return 10 ** log_value
            else:
                return rng.uniform(self.lower, self.upper)

        elif self.param_type == "integer":
            if self.log_scale:
                log_lower = np.log10(self.lower)
                log_upper = np.log10(self.upper)
                log_value = rng.uniform(log_lower, log_upper)
                return int(10 ** log_value)
            else:
                return rng.integers(self.lower, self.upper + 1)

        elif self.param_type == "categorical":
            return rng.choice(self.choices)

        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")

    def grid_values(self, n_points: int = 10) -> List[Any]:
        """Generate grid of values.

        Args:
            n_points: Number of grid points

        Returns:
            List of values
        """
        if self.param_type == "categorical":
            return self.choices

        elif self.param_type == "continuous":
            if self.log_scale:
                return list(np.logspace(
                    np.log10(self.lower),
                    np.log10(self.upper),
                    n_points
                ))
            else:
                return list(np.linspace(self.lower, self.upper, n_points))

        elif self.param_type == "integer":
            if self.log_scale:
                values = np.logspace(
                    np.log10(self.lower),
                    np.log10(self.upper),
                    n_points
                )
                return sorted(list(set(int(v) for v in values)))
            else:
                return list(range(self.lower, self.upper + 1))

        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


# =============================================================================
# Parameter Sweep
# =============================================================================

@dataclass
class ParameterSweep:
    """Parameter sweep configuration.

    Attributes:
        parameters: List of parameter specifications
        objective_func: Function to minimize (takes params dict, returns float)
        maximize: Whether to maximize instead of minimize
        n_jobs: Number of parallel jobs
        results_dir: Directory to save results
    """
    parameters: List[ParameterSpec]
    objective_func: Callable[[Dict[str, Any]], float]
    maximize: bool = False
    n_jobs: int = 4
    results_dir: Optional[Path] = None

    def __post_init__(self):
        """Initialize results directory."""
        if self.results_dir:
            self.results_dir = Path(self.results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = []

    def evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate objective function.

        Args:
            params: Parameter dictionary

        Returns:
            Objective value (negated if maximizing)
        """
        value = self.objective_func(params)

        # Store result
        self.results.append({
            "params": params.copy(),
            "value": value
        })

        # Save if results_dir specified
        if self.results_dir:
            result_file = self.results_dir / f"result_{len(self.results)}.json"
            with open(result_file, 'w') as f:
                json.dump(self.results[-1], f, indent=2)

        # Return negated if maximizing
        return -value if self.maximize else value

    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """Get best parameters and value.

        Returns:
            Tuple of (best_params, best_value)
        """
        if not self.results:
            raise ValueError("No results available")

        if self.maximize:
            best = max(self.results, key=lambda r: r["value"])
        else:
            best = min(self.results, key=lambda r: r["value"])

        return best["params"], best["value"]


# =============================================================================
# Grid Search
# =============================================================================

class GridSearch(ParameterSweep):
    """Grid search over parameter space.

    Evaluates all combinations of parameter values on a grid.
    """

    def __init__(
        self,
        parameters: List[ParameterSpec],
        objective_func: Callable[[Dict[str, Any]], float],
        n_grid_points: int = 10,
        **kwargs
    ):
        """Initialize grid search.

        Args:
            parameters: Parameter specifications
            objective_func: Objective function
            n_grid_points: Number of grid points per dimension
            **kwargs: Additional arguments for ParameterSweep
        """
        super().__init__(parameters, objective_func, **kwargs)
        self.n_grid_points = n_grid_points

    def run(self, use_dask: bool = True) -> Tuple[Dict[str, Any], float]:
        """Run grid search.

        Args:
            use_dask: Whether to use Dask for parallelization

        Returns:
            Tuple of (best_params, best_value)
        """
        # Generate grid
        grid_values = [
            param.grid_values(self.n_grid_points)
            for param in self.parameters
        ]

        # Create parameter combinations
        param_names = [p.name for p in self.parameters]
        param_configs = []

        for values in product(*grid_values):
            config = dict(zip(param_names, values))
            param_configs.append(config)

        print(f"Grid search: {len(param_configs)} configurations")

        # Evaluate in parallel
        if use_dask:
            from .distributed import distribute_computation
            scores = distribute_computation(
                self.evaluate,
                param_configs,
                n_workers=self.n_jobs
            )
        else:
            # Sequential fallback
            scores = [self.evaluate(config) for config in param_configs]

        return self.get_best()


# =============================================================================
# Random Search
# =============================================================================

class RandomSearch(ParameterSweep):
    """Random search over parameter space.

    Samples parameters randomly and evaluates objective.
    """

    def __init__(
        self,
        parameters: List[ParameterSpec],
        objective_func: Callable[[Dict[str, Any]], float],
        n_samples: int = 100,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize random search.

        Args:
            parameters: Parameter specifications
            objective_func: Objective function
            n_samples: Number of random samples
            seed: Random seed
            **kwargs: Additional arguments for ParameterSweep
        """
        super().__init__(parameters, objective_func, **kwargs)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def run(self, use_dask: bool = True) -> Tuple[Dict[str, Any], float]:
        """Run random search.

        Args:
            use_dask: Whether to use Dask for parallelization

        Returns:
            Tuple of (best_params, best_value)
        """
        # Sample parameters
        param_configs = []

        for _ in range(self.n_samples):
            config = {}
            for param in self.parameters:
                config[param.name] = param.sample(self.rng)
            param_configs.append(config)

        print(f"Random search: {self.n_samples} samples")

        # Evaluate in parallel
        if use_dask:
            from .distributed import distribute_computation
            scores = distribute_computation(
                self.evaluate,
                param_configs,
                n_workers=self.n_jobs
            )
        else:
            scores = [self.evaluate(config) for config in param_configs]

        return self.get_best()


# =============================================================================
# Bayesian Optimization
# =============================================================================

class BayesianOptimization(ParameterSweep):
    """Bayesian optimization using Gaussian processes.

    Uses scikit-optimize for efficient hyperparameter search.
    """

    def __init__(
        self,
        parameters: List[ParameterSpec],
        objective_func: Callable[[Dict[str, Any]], float],
        n_calls: int = 50,
        n_initial_points: int = 10,
        acq_func: str = "EI",
        **kwargs
    ):
        """Initialize Bayesian optimization.

        Args:
            parameters: Parameter specifications
            objective_func: Objective function
            n_calls: Number of evaluations
            n_initial_points: Number of random initial points
            acq_func: Acquisition function ("EI", "PI", "LCB")
            **kwargs: Additional arguments for ParameterSweep
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize not available. Install with: pip install scikit-optimize")

        super().__init__(parameters, objective_func, **kwargs)
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func

    def _to_skopt_space(self):
        """Convert parameter specs to skopt space."""
        dimensions = []

        for param in self.parameters:
            if param.param_type == "continuous":
                if param.log_scale:
                    dim = Real(param.lower, param.upper, prior="log-uniform", name=param.name)
                else:
                    dim = Real(param.lower, param.upper, name=param.name)
            elif param.param_type == "integer":
                if param.log_scale:
                    dim = Integer(param.lower, param.upper, prior="log-uniform", name=param.name)
                else:
                    dim = Integer(param.lower, param.upper, name=param.name)
            elif param.param_type == "categorical":
                dim = Categorical(param.choices, name=param.name)
            else:
                raise ValueError(f"Unknown parameter type: {param.param_type}")

            dimensions.append(dim)

        return dimensions

    def run(self) -> Tuple[Dict[str, Any], float]:
        """Run Bayesian optimization.

        Returns:
            Tuple of (best_params, best_value)
        """
        # Convert to skopt space
        dimensions = self._to_skopt_space()

        # Wrapper for objective
        def objective_wrapper(param_values):
            params = dict(zip([p.name for p in self.parameters], param_values))
            return self.evaluate(params)

        # Run optimization
        print(f"Bayesian optimization: {self.n_calls} evaluations")

        result = gp_minimize(
            objective_wrapper,
            dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            verbose=True
        )

        # Extract best parameters
        best_params = dict(zip([p.name for p in self.parameters], result.x))
        best_value = -result.fun if self.maximize else result.fun

        return best_params, best_value


# =============================================================================
# Parallel Optimizer
# =============================================================================

class ParallelOptimizer:
    """High-level parallel optimization interface.

    Supports multiple optimization algorithms with unified interface.
    """

    def __init__(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        parameters: List[ParameterSpec],
        maximize: bool = False,
        n_jobs: int = 4,
        results_dir: Optional[Path] = None
    ):
        """Initialize optimizer.

        Args:
            objective_func: Objective function
            parameters: Parameter specifications
            maximize: Whether to maximize
            n_jobs: Number of parallel jobs
            results_dir: Directory for results
        """
        self.objective_func = objective_func
        self.parameters = parameters
        self.maximize = maximize
        self.n_jobs = n_jobs
        self.results_dir = results_dir

    def grid_search(
        self,
        n_grid_points: int = 10,
        use_dask: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """Run grid search.

        Args:
            n_grid_points: Grid points per dimension
            use_dask: Use Dask parallelization

        Returns:
            (best_params, best_value)
        """
        search = GridSearch(
            self.parameters,
            self.objective_func,
            n_grid_points=n_grid_points,
            maximize=self.maximize,
            n_jobs=self.n_jobs,
            results_dir=self.results_dir
        )

        return search.run(use_dask=use_dask)

    def random_search(
        self,
        n_samples: int = 100,
        seed: Optional[int] = None,
        use_dask: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """Run random search.

        Args:
            n_samples: Number of samples
            seed: Random seed
            use_dask: Use Dask parallelization

        Returns:
            (best_params, best_value)
        """
        search = RandomSearch(
            self.parameters,
            self.objective_func,
            n_samples=n_samples,
            seed=seed,
            maximize=self.maximize,
            n_jobs=self.n_jobs,
            results_dir=self.results_dir
        )

        return search.run(use_dask=use_dask)

    def bayesian_optimization(
        self,
        n_calls: int = 50,
        n_initial_points: int = 10,
        acq_func: str = "EI"
    ) -> Tuple[Dict[str, Any], float]:
        """Run Bayesian optimization.

        Args:
            n_calls: Number of evaluations
            n_initial_points: Initial random points
            acq_func: Acquisition function

        Returns:
            (best_params, best_value)
        """
        search = BayesianOptimization(
            self.parameters,
            self.objective_func,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            maximize=self.maximize,
            n_jobs=self.n_jobs,
            results_dir=self.results_dir
        )

        return search.run()


# =============================================================================
# Convenience Functions
# =============================================================================

def run_parallel_sweep(
    objective_func: Callable[[Dict[str, Any]], float],
    parameters: List[ParameterSpec],
    method: str = "grid",
    n_jobs: int = 4,
    **method_kwargs
) -> Tuple[Dict[str, Any], float]:
    """Run parallel parameter sweep.

    Args:
        objective_func: Objective function
        parameters: Parameter specifications
        method: "grid", "random", or "bayesian"
        n_jobs: Number of parallel jobs
        **method_kwargs: Method-specific arguments

    Returns:
        (best_params, best_value)
    """
    optimizer = ParallelOptimizer(
        objective_func,
        parameters,
        n_jobs=n_jobs
    )

    if method == "grid":
        return optimizer.grid_search(**method_kwargs)
    elif method == "random":
        return optimizer.random_search(**method_kwargs)
    elif method == "bayesian":
        return optimizer.bayesian_optimization(**method_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def create_parameter_grid(
    **param_ranges: Union[Tuple[float, float], List[Any]]
) -> List[ParameterSpec]:
    """Create parameter specifications from ranges.

    Args:
        **param_ranges: Keyword arguments with parameter ranges
            - Tuple (lower, upper) for continuous parameters
            - List for categorical parameters

    Returns:
        List of ParameterSpec objects

    Example:
        >>> params = create_parameter_grid(
        ...     learning_rate=(1e-4, 1e-2),
        ...     batch_size=[32, 64, 128],
        ...     n_layers=(2, 5)
        ... )
    """
    parameters = []

    for name, spec in param_ranges.items():
        if isinstance(spec, tuple) and len(spec) == 2:
            # Continuous or integer parameter
            lower, upper = spec
            if isinstance(lower, int) and isinstance(upper, int):
                param_type = "integer"
            else:
                param_type = "continuous"

            parameters.append(ParameterSpec(
                name=name,
                param_type=param_type,
                lower=lower,
                upper=upper
            ))

        elif isinstance(spec, list):
            # Categorical parameter
            parameters.append(ParameterSpec(
                name=name,
                param_type="categorical",
                choices=spec
            ))

        else:
            raise ValueError(f"Invalid specification for {name}: {spec}")

    return parameters


def analyze_sweep_results(
    results: List[Dict[str, Any]],
    top_k: int = 10
) -> Dict[str, Any]:
    """Analyze parameter sweep results.

    Args:
        results: List of result dictionaries
        top_k: Number of top results to include

    Returns:
        Analysis dictionary
    """
    if not results:
        return {}

    # Sort by value
    sorted_results = sorted(results, key=lambda r: r["value"])

    # Extract statistics
    values = [r["value"] for r in results]

    analysis = {
        "n_evaluations": len(results),
        "best_value": sorted_results[0]["value"],
        "best_params": sorted_results[0]["params"],
        "worst_value": sorted_results[-1]["value"],
        "mean_value": np.mean(values),
        "std_value": np.std(values),
        "median_value": np.median(values),
        "top_k_results": sorted_results[:top_k]
    }

    # Parameter importance (if enough samples)
    if len(results) >= 20:
        analysis["parameter_importance"] = compute_parameter_importance(results)

    return analysis


def compute_parameter_importance(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute parameter importance via variance analysis.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping parameter names to importance scores
    """
    # Extract parameter names
    param_names = list(results[0]["params"].keys())

    importance = {}

    for param_name in param_names:
        # Group by parameter value
        value_groups = {}

        for result in results:
            param_value = result["params"][param_name]

            # Convert to hashable type
            if isinstance(param_value, (list, dict)):
                param_value = str(param_value)

            if param_value not in value_groups:
                value_groups[param_value] = []

            value_groups[param_value].append(result["value"])

        # Compute variance explained
        group_means = [np.mean(values) for values in value_groups.values()]
        total_variance = np.var([r["value"] for r in results])
        between_variance = np.var(group_means)

        importance[param_name] = between_variance / (total_variance + 1e-10)

    return importance


# =============================================================================
# Week 33-34 Enhancements: Advanced Parameter Sweep Infrastructure
# =============================================================================

class AdaptiveSweep(ParameterSweep):
    """Adaptive parameter sweep based on previous results.

    Uses performance feedback to focus search on promising regions.
    """

    def __init__(
        self,
        param_specs: List[ParameterSpec],
        n_initial: int = 20,
        n_iterations: int = 5,
        n_per_iteration: int = 10,
        exploration_weight: float = 0.1
    ):
        """Initialize adaptive sweep.

        Args:
            param_specs: Parameter specifications
            n_initial: Number of initial random samples
            n_iterations: Number of adaptive iterations
            n_per_iteration: Samples per iteration
            exploration_weight: Balance exploration vs exploitation
        """
        super().__init__(param_specs)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_per_iteration = n_per_iteration
        self.exploration_weight = exploration_weight
        self.results_history = []

    def generate_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate adaptive samples based on history."""
        if len(self.results_history) < self.n_initial:
            # Initial random phase
            return self._random_samples(n_samples)
        else:
            # Adaptive phase
            return self._adaptive_samples(n_samples)

    def _random_samples(self, n: int) -> List[Dict[str, Any]]:
        """Generate random samples."""
        samples = []
        for _ in range(n):
            sample = {}
            for spec in self.param_specs:
                sample[spec.name] = self._sample_parameter(spec)
            samples.append(sample)
        return samples

    def _sample_parameter(self, spec: ParameterSpec) -> Any:
        """Sample single parameter."""
        if spec.param_type == "continuous":
            if spec.log_scale:
                return 10 ** np.random.uniform(
                    np.log10(spec.lower),
                    np.log10(spec.upper)
                )
            else:
                return np.random.uniform(spec.lower, spec.upper)
        elif spec.param_type == "integer":
            return np.random.randint(spec.lower, spec.upper + 1)
        else:  # categorical
            return np.random.choice(spec.choices)

    def _adaptive_samples(self, n: int) -> List[Dict[str, Any]]:
        """Generate samples focused on promising regions."""
        # Find best results
        sorted_results = sorted(
            self.results_history,
            key=lambda x: x["value"]
        )
        top_k = min(5, len(sorted_results) // 4)
        best_results = sorted_results[:top_k]

        samples = []
        for _ in range(n):
            # Sample with probability proportional to performance
            if np.random.rand() < self.exploration_weight:
                # Exploration: random sample
                sample = self._random_samples(1)[0]
            else:
                # Exploitation: perturb best result
                base = np.random.choice(best_results)
                sample = self._perturb_sample(base["params"])

            samples.append(sample)

        return samples

    def _perturb_sample(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb parameters around base."""
        perturbed = {}
        for spec in self.param_specs:
            base_value = base_params[spec.name]

            if spec.param_type == "continuous":
                # Gaussian perturbation
                std = (spec.upper - spec.lower) * 0.1
                perturbed_value = base_value + np.random.randn() * std
                perturbed_value = np.clip(perturbed_value, spec.lower, spec.upper)
                perturbed[spec.name] = perturbed_value

            elif spec.param_type == "integer":
                # Integer perturbation
                delta = max(1, int((spec.upper - spec.lower) * 0.1))
                perturbed_value = base_value + np.random.randint(-delta, delta + 1)
                perturbed_value = np.clip(perturbed_value, spec.lower, spec.upper)
                perturbed[spec.name] = int(perturbed_value)

            else:  # categorical
                # Occasional random choice
                if np.random.rand() < 0.2:
                    perturbed[spec.name] = np.random.choice(spec.choices)
                else:
                    perturbed[spec.name] = base_value

        return perturbed

    def add_result(self, params: Dict[str, Any], value: float):
        """Add result to history for adaptive sampling."""
        self.results_history.append({"params": params, "value": value})


class MultiObjectiveSweep:
    """Multi-objective parameter sweep.

    Optimizes multiple objectives simultaneously using Pareto frontier.
    """

    def __init__(self, param_specs: List[ParameterSpec]):
        """Initialize multi-objective sweep.

        Args:
            param_specs: Parameter specifications
        """
        self.param_specs = param_specs
        self.results = []

    def add_result(
        self,
        params: Dict[str, Any],
        objectives: Dict[str, float]
    ):
        """Add multi-objective result.

        Args:
            params: Parameter values
            objectives: Dictionary of objective values
        """
        self.results.append({
            "params": params,
            "objectives": objectives
        })

    def compute_pareto_frontier(self) -> List[Dict[str, Any]]:
        """Compute Pareto-optimal solutions.

        Returns:
            List of Pareto-optimal results
        """
        if not self.results:
            return []

        # Extract objectives as array
        objective_names = list(self.results[0]["objectives"].keys())
        n_objectives = len(objective_names)

        objectives_array = np.array([
            [r["objectives"][name] for name in objective_names]
            for r in self.results
        ])

        # Find Pareto frontier
        is_pareto = np.ones(len(self.results), dtype=bool)

        for i in range(len(self.results)):
            if not is_pareto[i]:
                continue

            # Check if any other point dominates this one
            for j in range(len(self.results)):
                if i == j or not is_pareto[j]:
                    continue

                # j dominates i if all objectives are ≤ and at least one is <
                dominates = np.all(objectives_array[j] <= objectives_array[i])
                strictly_better = np.any(objectives_array[j] < objectives_array[i])

                if dominates and strictly_better:
                    is_pareto[i] = False
                    break

        # Return Pareto-optimal results
        pareto_results = [
            self.results[i] for i in range(len(self.results))
            if is_pareto[i]
        ]

        return pareto_results

    def compute_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Compute hypervolume indicator.

        Args:
            reference_point: Reference point for hypervolume

        Returns:
            Hypervolume value
        """
        pareto = self.compute_pareto_frontier()

        if not pareto:
            return 0.0

        # Simple 2D hypervolume (can be extended)
        objective_names = list(reference_point.keys())

        if len(objective_names) == 2:
            # Sort by first objective
            sorted_pareto = sorted(
                pareto,
                key=lambda x: x["objectives"][objective_names[0]]
            )

            hypervolume = 0.0
            prev_x = reference_point[objective_names[0]]

            for result in sorted_pareto:
                x = result["objectives"][objective_names[0]]
                y = result["objectives"][objective_names[1]]
                ref_y = reference_point[objective_names[1]]

                width = prev_x - x
                height = ref_y - y

                if width > 0 and height > 0:
                    hypervolume += width * height

                prev_x = x

            return hypervolume

        else:
            # Fallback for higher dimensions (approximate)
            return len(pareto)


def sensitivity_analysis(
    objective_fn: Callable,
    params: Dict[str, Any],
    param_specs: List[ParameterSpec],
    n_samples: int = 10
) -> Dict[str, Dict[str, float]]:
    """Perform sensitivity analysis on parameters.

    Args:
        objective_fn: Objective function
        params: Nominal parameter values
        param_specs: Parameter specifications
        n_samples: Number of samples per parameter

    Returns:
        Dictionary mapping parameter names to sensitivity metrics
    """
    baseline_value = objective_fn(params)

    sensitivity = {}

    for spec in param_specs:
        param_name = spec.name

        # Sample parameter values
        if spec.param_type == "continuous":
            if spec.log_scale:
                values = np.logspace(
                    np.log10(spec.lower),
                    np.log10(spec.upper),
                    n_samples
                )
            else:
                values = np.linspace(spec.lower, spec.upper, n_samples)

        elif spec.param_type == "integer":
            values = np.linspace(spec.lower, spec.upper, n_samples, dtype=int)

        else:  # categorical
            values = spec.choices

        # Evaluate objective
        objective_values = []
        for value in values:
            test_params = params.copy()
            test_params[param_name] = value
            obj_value = objective_fn(test_params)
            objective_values.append(obj_value)

        # Compute sensitivity metrics
        obj_range = np.max(objective_values) - np.min(objective_values)
        obj_std = np.std(objective_values)
        obj_change = abs(np.mean(objective_values) - baseline_value)

        sensitivity[param_name] = {
            "range": obj_range,
            "std": obj_std,
            "mean_change": obj_change,
            "relative_importance": obj_std / (baseline_value + 1e-10)
        }

    return sensitivity


def visualize_sweep_results(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> str:
    """Generate visualization summary of sweep results.

    Args:
        results: List of sweep results
        output_path: Path to save visualization (None = return string)

    Returns:
        Text summary of results
    """
    if not results:
        return "No results to visualize"

    # Extract best result
    best_result = min(results, key=lambda x: x["value"])

    # Compute statistics
    values = [r["value"] for r in results]
    mean_value = np.mean(values)
    std_value = np.std(values)
    min_value = np.min(values)
    max_value = np.max(values)

    # Parameter importance
    importance = compute_parameter_importance(results)
    sorted_importance = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Generate summary
    summary = []
    summary.append("=" * 70)
    summary.append("PARAMETER SWEEP RESULTS")
    summary.append("=" * 70)
    summary.append(f"\nTotal evaluations: {len(results)}")
    summary.append(f"\nObjective Statistics:")
    summary.append(f"  Best:  {min_value:.6f}")
    summary.append(f"  Mean:  {mean_value:.6f}")
    summary.append(f"  Std:   {std_value:.6f}")
    summary.append(f"  Worst: {max_value:.6f}")

    summary.append(f"\nBest Parameters:")
    for param_name, param_value in best_result["params"].items():
        if isinstance(param_value, float):
            summary.append(f"  {param_name}: {param_value:.6f}")
        else:
            summary.append(f"  {param_name}: {param_value}")

    summary.append(f"\nParameter Importance:")
    for param_name, imp_value in sorted_importance[:5]:
        summary.append(f"  {param_name}: {imp_value:.4f}")

    summary.append("\n" + "=" * 70)

    text = "\n".join(summary)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(text)

    return text


def export_sweep_results(
    results: List[Dict[str, Any]],
    output_path: str,
    format: str = "json"
):
    """Export sweep results to file.

    Args:
        results: List of sweep results
        output_path: Output file path
        format: "json" or "csv"
    """
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    elif format == "csv":
        import csv

        if not results:
            return

        # Extract all parameter names
        param_names = list(results[0]["params"].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = param_names + ["objective_value"]
            writer.writerow(header)

            # Data rows
            for result in results:
                row = [
                    result["params"][name] for name in param_names
                ] + [result["value"]]
                writer.writerow(row)

    else:
        raise ValueError(f"Unsupported format: {format}")
