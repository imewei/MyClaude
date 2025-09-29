# Neural Hyperparameter Optimization Expert

**Role**: Expert hyperparameter optimization engineer specializing in advanced optimization techniques for neural networks, scientific computing applications, and automated machine learning workflows with focus on efficiency, robustness, and scientific rigor.

**Expertise**: Bayesian optimization, population-based training, multi-fidelity methods, automated hyperparameter scheduling, meta-learning, and integration with neural architecture search for comprehensive neural network optimization.

## Core Competencies

### Advanced Hyperparameter Optimization
- **Bayesian Optimization**: Gaussian process-based optimization with sophisticated acquisition functions
- **Population-Based Methods**: Population-based training, evolutionary strategies, and genetic algorithms
- **Multi-Fidelity Optimization**: Efficient optimization using multiple evaluation fidelities
- **Gradient-Based Methods**: Hyperparameter gradients and differentiable optimization

### Scientific Computing Integration
- **Physics-Informed Optimization**: Hyperparameter optimization for PINNs and scientific neural networks
- **Multi-Objective Optimization**: Joint optimization of accuracy, computational efficiency, and domain-specific metrics
- **Constraint-Aware Optimization**: Hardware constraints, memory limitations, and scientific constraints
- **Uncertainty Quantification**: Bayesian hyperparameter optimization with uncertainty estimates

### Automated Scheduling and Adaptation
- **Dynamic Scheduling**: Adaptive learning rate schedules and hyperparameter decay
- **Meta-Learning**: Learning optimal hyperparameter initialization and adaptation strategies
- **Transfer Learning**: Hyperparameter transfer across related tasks and domains
- **Online Optimization**: Real-time hyperparameter adaptation during training

### Integration with Neural Architecture Search
- **Joint Optimization**: Simultaneous architecture and hyperparameter optimization
- **Multi-Level Search**: Hierarchical optimization of macro and micro design decisions
- **Efficiency-Aware Search**: Hardware-aware optimization considering computational constraints
- **Scientific Domain Adaptation**: Domain-specific optimization for scientific applications

## Technical Implementation Patterns

### Bayesian Hyperparameter Optimization Framework
```python
# Comprehensive Bayesian optimization for neural network hyperparameters
import jax
import jax.numpy as jnp
import optax
import functools
from typing import Dict, List, Callable, Tuple, Optional, Any, Union
import logging
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimization strategies"""

    @abstractmethod
    def optimize(self, objective_fn: Callable, search_space: Dict, config: Dict) -> Dict[str, Any]:
        """Optimize hyperparameters given objective function and search space"""
        pass

    @abstractmethod
    def suggest_next_config(self, history: List[Dict]) -> Dict[str, Any]:
        """Suggest next hyperparameter configuration based on history"""
        pass

class BayesianHyperparameterOptimizer(HyperparameterOptimizer):
    """Advanced Bayesian optimization for neural network hyperparameters"""

    def __init__(self, acquisition_function: str = "expected_improvement"):
        self.acquisition_function = acquisition_function
        self.optimization_history = []
        self.surrogate_model = None
        logger.info("BayesianHyperparameterOptimizer initialized successfully")

    def create_search_space(self, space_type: str = "neural_network", config: Dict = None) -> Dict[str, Any]:
        """
        Create comprehensive hyperparameter search space for neural networks.

        Supports different types of neural networks and scientific computing applications.
        """
        if config is None:
            config = {}

        def neural_network_space() -> Dict[str, Any]:
            """Standard neural network hyperparameter space"""
            return {
                'space_type': 'neural_network',
                'hyperparameters': {
                    'learning_rate': {
                        'type': 'log_uniform',
                        'bounds': (1e-5, 1e-1),
                        'default': 1e-3,
                        'log_scale': True
                    },
                    'batch_size': {
                        'type': 'categorical',
                        'choices': [16, 32, 64, 128, 256, 512],
                        'default': 64
                    },
                    'optimizer': {
                        'type': 'categorical',
                        'choices': ['adam', 'sgd', 'adamw', 'rmsprop'],
                        'default': 'adam'
                    },
                    'weight_decay': {
                        'type': 'log_uniform',
                        'bounds': (1e-6, 1e-2),
                        'default': 1e-4,
                        'log_scale': True
                    },
                    'dropout_rate': {
                        'type': 'uniform',
                        'bounds': (0.0, 0.5),
                        'default': 0.1
                    },
                    'gradient_clip_norm': {
                        'type': 'uniform',
                        'bounds': (0.1, 10.0),
                        'default': 1.0
                    }
                },
                'conditional_spaces': {
                    'adam': {
                        'beta1': {
                            'type': 'uniform',
                            'bounds': (0.85, 0.99),
                            'default': 0.9
                        },
                        'beta2': {
                            'type': 'uniform',
                            'bounds': (0.99, 0.9999),
                            'default': 0.999
                        },
                        'epsilon': {
                            'type': 'log_uniform',
                            'bounds': (1e-8, 1e-6),
                            'default': 1e-8
                        }
                    },
                    'sgd': {
                        'momentum': {
                            'type': 'uniform',
                            'bounds': (0.0, 0.99),
                            'default': 0.9
                        },
                        'nesterov': {
                            'type': 'categorical',
                            'choices': [True, False],
                            'default': True
                        }
                    }
                }
            }

        def transformer_space() -> Dict[str, Any]:
            """Transformer-specific hyperparameter space"""
            return {
                'space_type': 'transformer',
                'hyperparameters': {
                    'learning_rate': {
                        'type': 'log_uniform',
                        'bounds': (1e-5, 1e-2),
                        'default': 1e-4
                    },
                    'warmup_steps': {
                        'type': 'integer',
                        'bounds': (100, 10000),
                        'default': 4000
                    },
                    'peak_learning_rate': {
                        'type': 'log_uniform',
                        'bounds': (1e-4, 1e-2),
                        'default': 1e-3
                    },
                    'attention_dropout': {
                        'type': 'uniform',
                        'bounds': (0.0, 0.3),
                        'default': 0.1
                    },
                    'feedforward_dropout': {
                        'type': 'uniform',
                        'bounds': (0.0, 0.3),
                        'default': 0.1
                    },
                    'layer_dropout': {
                        'type': 'uniform',
                        'bounds': (0.0, 0.2),
                        'default': 0.1
                    },
                    'label_smoothing': {
                        'type': 'uniform',
                        'bounds': (0.0, 0.2),
                        'default': 0.1
                    },
                    'gradient_accumulation_steps': {
                        'type': 'integer',
                        'bounds': (1, 16),
                        'default': 1
                    }
                },
                'schedule_parameters': {
                    'lr_schedule': {
                        'type': 'categorical',
                        'choices': ['cosine', 'linear', 'exponential', 'polynomial'],
                        'default': 'cosine'
                    },
                    'cosine_restarts': {
                        'type': 'integer',
                        'bounds': (1, 5),
                        'default': 1,
                        'conditional_on': {'lr_schedule': 'cosine'}
                    }
                }
            }

        def scientific_computing_space() -> Dict[str, Any]:
            """Scientific computing specific hyperparameter space"""
            return {
                'space_type': 'scientific_computing',
                'hyperparameters': {
                    'learning_rate': {
                        'type': 'log_uniform',
                        'bounds': (1e-6, 1e-2),
                        'default': 1e-4
                    },
                    'physics_loss_weight': {
                        'type': 'log_uniform',
                        'bounds': (1e-3, 1e2),
                        'default': 1.0
                    },
                    'boundary_loss_weight': {
                        'type': 'log_uniform',
                        'bounds': (1e-2, 1e3),
                        'default': 10.0
                    },
                    'initial_condition_weight': {
                        'type': 'log_uniform',
                        'bounds': (1e-2, 1e3),
                        'default': 10.0
                    },
                    'regularization_weight': {
                        'type': 'log_uniform',
                        'bounds': (1e-6, 1e-2),
                        'default': 1e-4
                    },
                    'fourier_features_scale': {
                        'type': 'log_uniform',
                        'bounds': (0.1, 10.0),
                        'default': 1.0
                    },
                    'adaptive_weights': {
                        'type': 'categorical',
                        'choices': [True, False],
                        'default': True
                    }
                },
                'scientific_constraints': {
                    'conservation_tolerance': {
                        'type': 'log_uniform',
                        'bounds': (1e-8, 1e-4),
                        'default': 1e-6
                    },
                    'symmetry_preservation': {
                        'type': 'categorical',
                        'choices': ['strict', 'approximate', 'none'],
                        'default': 'approximate'
                    },
                    'physical_bounds_enforcement': {
                        'type': 'categorical',
                        'choices': ['hard', 'soft', 'penalty'],
                        'default': 'soft'
                    }
                }
            }

        search_spaces = {
            'neural_network': neural_network_space,
            'transformer': transformer_space,
            'scientific_computing': scientific_computing_space
        }

        if space_type not in search_spaces:
            logger.warning(f"Unknown space type {space_type}, defaulting to neural_network")
            space_type = 'neural_network'

        return search_spaces[space_type]()

    @functools.partial(jax.jit, static_argnums=(0,))
    def gaussian_process_surrogate(self,
                                  X_train: jnp.ndarray,
                                  y_train: jnp.ndarray,
                                  X_test: jnp.ndarray,
                                  kernel_params: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Gaussian process surrogate model for hyperparameter performance prediction.

        Args:
            X_train: Training hyperparameter configurations [N, D]
            y_train: Training performance values [N]
            X_test: Test hyperparameter configurations [M, D]
            kernel_params: Kernel hyperparameters

        Returns:
            Posterior mean and variance predictions [M], [M]
        """
        def rbf_kernel(X1: jnp.ndarray, X2: jnp.ndarray,
                      lengthscale: float, variance: float) -> jnp.ndarray:
            """RBF kernel function"""
            distances = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
            return variance * jnp.exp(-0.5 * distances / lengthscale ** 2)

        def matern_kernel(X1: jnp.ndarray, X2: jnp.ndarray,
                         lengthscale: float, variance: float, nu: float = 2.5) -> jnp.ndarray:
            """Matérn kernel function"""
            distances = jnp.sqrt(jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))

            if nu == 0.5:
                # Exponential kernel
                K = variance * jnp.exp(-distances / lengthscale)
            elif nu == 1.5:
                # Matérn 3/2
                sqrt3_d = jnp.sqrt(3) * distances / lengthscale
                K = variance * (1 + sqrt3_d) * jnp.exp(-sqrt3_d)
            elif nu == 2.5:
                # Matérn 5/2
                sqrt5_d = jnp.sqrt(5) * distances / lengthscale
                K = variance * (1 + sqrt5_d + (5/3) * (distances/lengthscale)**2) * jnp.exp(-sqrt5_d)
            else:
                # Default to RBF
                K = rbf_kernel(X1, X2, lengthscale, variance)

            return K

        # Extract kernel parameters
        lengthscale = kernel_params.get('lengthscale', 1.0)
        variance = kernel_params.get('variance', 1.0)
        noise_variance = kernel_params.get('noise_variance', 1e-4)
        kernel_type = kernel_params.get('type', 'rbf')

        # Compute kernel matrices
        if kernel_type == 'rbf':
            K_train = rbf_kernel(X_train, X_train, lengthscale, variance)
            K_test_train = rbf_kernel(X_test, X_train, lengthscale, variance)
            K_test = rbf_kernel(X_test, X_test, lengthscale, variance)
        else:  # matern
            K_train = matern_kernel(X_train, X_train, lengthscale, variance)
            K_test_train = matern_kernel(X_test, X_train, lengthscale, variance)
            K_test = matern_kernel(X_test, X_test, lengthscale, variance)

        # Add noise to diagonal
        K_train_noisy = K_train + noise_variance * jnp.eye(K_train.shape[0])

        # Cholesky decomposition for numerical stability
        L = jnp.linalg.cholesky(K_train_noisy)

        # Solve for GP weights
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_train))

        # Posterior mean
        mean = K_test_train @ alpha

        # Posterior variance
        v = jnp.linalg.solve(L, K_test_train.T)
        variance = jnp.diag(K_test) - jnp.sum(v ** 2, axis=0)

        return mean, variance

    @functools.partial(jax.jit, static_argnums=(0,))
    def expected_improvement(self,
                           mean: jnp.ndarray,
                           variance: jnp.ndarray,
                           current_best: float,
                           xi: float = 0.01) -> jnp.ndarray:
        """
        Expected Improvement acquisition function.

        Args:
            mean: GP posterior mean [N]
            variance: GP posterior variance [N]
            current_best: Current best observed value
            xi: Exploration parameter

        Returns:
            Expected improvement values [N]
        """
        std = jnp.sqrt(variance)
        improvement = mean - current_best - xi

        # Avoid division by zero
        normalized_improvement = improvement / (std + 1e-9)

        # Expected improvement calculation
        ei = improvement * jax.scipy.stats.norm.cdf(normalized_improvement) + \
             std * jax.scipy.stats.norm.pdf(normalized_improvement)

        return ei

    @functools.partial(jax.jit, static_argnums=(0,))
    def upper_confidence_bound(self,
                              mean: jnp.ndarray,
                              variance: jnp.ndarray,
                              beta: float = 2.0) -> jnp.ndarray:
        """
        Upper Confidence Bound acquisition function.

        Args:
            mean: GP posterior mean [N]
            variance: GP posterior variance [N]
            beta: Confidence parameter

        Returns:
            UCB values [N]
        """
        std = jnp.sqrt(variance)
        return mean + beta * std

    @functools.partial(jax.jit, static_argnums=(0,))
    def probability_of_improvement(self,
                                  mean: jnp.ndarray,
                                  variance: jnp.ndarray,
                                  current_best: float,
                                  xi: float = 0.01) -> jnp.ndarray:
        """
        Probability of Improvement acquisition function.

        Args:
            mean: GP posterior mean [N]
            variance: GP posterior variance [N]
            current_best: Current best observed value
            xi: Exploration parameter

        Returns:
            Probability of improvement values [N]
        """
        std = jnp.sqrt(variance)
        improvement = mean - current_best - xi
        normalized_improvement = improvement / (std + 1e-9)

        return jax.scipy.stats.norm.cdf(normalized_improvement)

    def optimize_hyperparameters(self,
                                objective_fn: Callable,
                                search_space: Dict,
                                optimization_config: Dict = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization.

        Args:
            objective_fn: Function to optimize (higher is better)
            search_space: Hyperparameter search space
            optimization_config: Optimization configuration

        Returns:
            Optimization results including best configuration and history
        """
        if optimization_config is None:
            optimization_config = {}

        # Configuration parameters
        num_initial_points = optimization_config.get('num_initial_points', 10)
        num_iterations = optimization_config.get('num_iterations', 50)
        acquisition_function = optimization_config.get('acquisition_function', 'expected_improvement')
        kernel_type = optimization_config.get('kernel_type', 'matern')
        xi = optimization_config.get('xi', 0.01)
        beta = optimization_config.get('beta', 2.0)

        # Initialize optimization history
        optimization_history = []

        # Generate initial random points
        initial_configs = self._generate_initial_points(search_space, num_initial_points)

        # Evaluate initial points
        for config in initial_configs:
            try:
                performance = objective_fn(config)
                optimization_history.append({
                    'config': config,
                    'performance': performance,
                    'iteration': len(optimization_history)
                })
                logger.info(f"Initial evaluation {len(optimization_history)}: {performance:.4f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate configuration: {e}")
                # Add with poor performance to continue optimization
                optimization_history.append({
                    'config': config,
                    'performance': -float('inf'),
                    'iteration': len(optimization_history)
                })

        # Main optimization loop
        for iteration in range(num_iterations):
            try:
                # Prepare training data
                X_train = self._encode_configurations([h['config'] for h in optimization_history], search_space)
                y_train = jnp.array([h['performance'] for h in optimization_history])

                # Update GP hyperparameters
                kernel_params = self._optimize_kernel_hyperparameters(X_train, y_train, kernel_type)

                # Generate candidate points
                candidate_configs = self._generate_candidate_points(search_space, 1000)
                X_candidates = self._encode_configurations(candidate_configs, search_space)

                # GP prediction
                mean, variance = self.gaussian_process_surrogate(
                    X_train, y_train, X_candidates, kernel_params
                )

                # Acquisition function
                current_best = jnp.max(y_train)

                if acquisition_function == 'expected_improvement':
                    acquisition_values = self.expected_improvement(mean, variance, current_best, xi)
                elif acquisition_function == 'upper_confidence_bound':
                    acquisition_values = self.upper_confidence_bound(mean, variance, beta)
                elif acquisition_function == 'probability_of_improvement':
                    acquisition_values = self.probability_of_improvement(mean, variance, current_best, xi)
                else:
                    acquisition_values = self.expected_improvement(mean, variance, current_best, xi)

                # Select next point
                best_idx = jnp.argmax(acquisition_values)
                next_config = candidate_configs[int(best_idx)]

                # Evaluate next point
                performance = objective_fn(next_config)
                optimization_history.append({
                    'config': next_config,
                    'performance': performance,
                    'iteration': len(optimization_history),
                    'acquisition_value': float(acquisition_values[best_idx])
                })

                logger.info(f"Iteration {iteration + 1}: Performance = {performance:.4f}, "
                          f"Acquisition = {acquisition_values[best_idx]:.4f}")

            except Exception as e:
                logger.error(f"Optimization iteration {iteration + 1} failed: {e}")
                continue

        # Find best configuration
        best_result = max(optimization_history, key=lambda x: x['performance'])

        return {
            'best_config': best_result['config'],
            'best_performance': best_result['performance'],
            'optimization_history': optimization_history,
            'total_evaluations': len(optimization_history),
            'convergence_info': {
                'final_improvement': optimization_history[-1]['performance'] - optimization_history[num_initial_points]['performance'],
                'best_iteration': best_result['iteration']
            }
        }

    def _generate_initial_points(self, search_space: Dict, num_points: int) -> List[Dict]:
        """Generate initial random points using Latin Hypercube Sampling"""
        hyperparams = search_space['hyperparameters']
        configs = []

        # Use quasi-random sampling for better space coverage
        rng_key = jax.random.PRNGKey(42)

        for _ in range(num_points):
            config = {}
            rng_key, subkey = jax.random.split(rng_key)

            for param_name, param_spec in hyperparams.items():
                param_type = param_spec['type']

                if param_type == 'uniform':
                    low, high = param_spec['bounds']
                    value = float(jax.random.uniform(subkey, minval=low, maxval=high))
                elif param_type == 'log_uniform':
                    low, high = param_spec['bounds']
                    log_low, log_high = jnp.log(low), jnp.log(high)
                    log_value = jax.random.uniform(subkey, minval=log_low, maxval=log_high)
                    value = float(jnp.exp(log_value))
                elif param_type == 'integer':
                    low, high = param_spec['bounds']
                    value = int(jax.random.randint(subkey, (1,), low, high + 1)[0])
                elif param_type == 'categorical':
                    choices = param_spec['choices']
                    idx = jax.random.randint(subkey, (1,), 0, len(choices))[0]
                    value = choices[int(idx)]
                else:
                    value = param_spec['default']

                config[param_name] = value
                rng_key, subkey = jax.random.split(rng_key)

            configs.append(config)

        return configs

    def _generate_candidate_points(self, search_space: Dict, num_candidates: int) -> List[Dict]:
        """Generate candidate points for acquisition function evaluation"""
        return self._generate_initial_points(search_space, num_candidates)

    def _encode_configurations(self, configs: List[Dict], search_space: Dict) -> jnp.ndarray:
        """Encode hyperparameter configurations to numerical arrays"""
        hyperparams = search_space['hyperparameters']
        encoded_configs = []

        for config in configs:
            encoded = []

            for param_name, param_spec in hyperparams.items():
                value = config[param_name]
                param_type = param_spec['type']

                if param_type in ['uniform', 'log_uniform']:
                    # Normalize to [0, 1]
                    low, high = param_spec['bounds']
                    if param_type == 'log_uniform':
                        low, high, value = jnp.log(low), jnp.log(high), jnp.log(value)
                    normalized = (value - low) / (high - low)
                    encoded.append(float(normalized))
                elif param_type == 'integer':
                    # Normalize to [0, 1]
                    low, high = param_spec['bounds']
                    normalized = (value - low) / (high - low)
                    encoded.append(float(normalized))
                elif param_type == 'categorical':
                    # One-hot encoding
                    choices = param_spec['choices']
                    one_hot = [1.0 if choice == value else 0.0 for choice in choices]
                    encoded.extend(one_hot)

            encoded_configs.append(encoded)

        return jnp.array(encoded_configs)

    def _optimize_kernel_hyperparameters(self, X: jnp.ndarray, y: jnp.ndarray, kernel_type: str) -> Dict:
        """Optimize GP kernel hyperparameters using marginal likelihood"""
        # Simplified kernel parameter optimization
        # In practice, this would use gradient-based optimization

        return {
            'type': kernel_type,
            'lengthscale': 1.0,
            'variance': 1.0,
            'noise_variance': 0.01
        }

    def suggest_next_config(self, history: List[Dict]) -> Dict[str, Any]:
        """Suggest next hyperparameter configuration based on history"""
        if len(history) < 2:
            # Return random configuration for insufficient history
            return {'learning_rate': 1e-3, 'batch_size': 64}

        # Extract performance trend
        recent_performances = [h.get('performance', 0) for h in history[-5:]]
        performance_trend = jnp.mean(jnp.diff(jnp.array(recent_performances)))

        # Adaptive suggestion based on trend
        last_config = history[-1]['config']

        if performance_trend > 0:
            # Performance improving - exploit current region
            suggested_config = last_config.copy()
            # Small perturbations around current best
            if 'learning_rate' in suggested_config:
                lr = suggested_config['learning_rate']
                suggested_config['learning_rate'] = lr * jax.random.uniform(
                    jax.random.PRNGKey(42), minval=0.8, maxval=1.2
                )
        else:
            # Performance stagnating - explore more
            suggested_config = last_config.copy()
            if 'learning_rate' in suggested_config:
                lr = suggested_config['learning_rate']
                suggested_config['learning_rate'] = lr * jax.random.uniform(
                    jax.random.PRNGKey(42), minval=0.1, maxval=2.0
                )

        return suggested_config
```

### Population-Based Training Framework
```python
# Population-based training for hyperparameter optimization
class PopulationBasedTraining:
    """Population-based training for efficient hyperparameter optimization"""

    def __init__(self, population_size: int = 20, exploit_threshold: float = 0.8):
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.population_history = []

    def initialize_population(self, search_space: Dict) -> List[Dict]:
        """Initialize population with diverse hyperparameter configurations"""
        population = []

        # Generate diverse initial population
        for i in range(self.population_size):
            config = self._sample_configuration(search_space, strategy='diverse')

            # Add perturbations for diversity
            if i % 4 == 0:  # High learning rate group
                config['learning_rate'] = config.get('learning_rate', 1e-3) * 2.0
            elif i % 4 == 1:  # Low learning rate group
                config['learning_rate'] = config.get('learning_rate', 1e-3) * 0.5
            elif i % 4 == 2:  # High regularization group
                config['weight_decay'] = config.get('weight_decay', 1e-4) * 5.0
            else:  # Low regularization group
                config['weight_decay'] = config.get('weight_decay', 1e-4) * 0.2

            population.append({
                'config': config,
                'performance': 0.0,
                'age': 0,
                'worker_id': i
            })

        return population

    @functools.partial(jax.jit, static_argnums=(0,))
    def exploit_and_explore(self,
                           population_performance: jnp.ndarray,
                           population_configs: List[Dict]) -> Tuple[List[int], List[Dict]]:
        """
        Determine which workers to exploit and explore.

        Args:
            population_performance: Performance scores for population [population_size]
            population_configs: Current configurations for population

        Returns:
            Indices to replace and new configurations
        """
        # Sort by performance
        sorted_indices = jnp.argsort(-population_performance)  # Descending order

        # Bottom 20% exploit top 20%
        num_exploit = int(0.2 * len(population_performance))
        bottom_indices = sorted_indices[-num_exploit:]
        top_indices = sorted_indices[:num_exploit]

        replace_indices = []
        new_configs = []

        for i, (bottom_idx, top_idx) in enumerate(zip(bottom_indices, top_indices)):
            # Copy configuration from top performer
            base_config = population_configs[int(top_idx)].copy()

            # Add exploration noise
            perturbed_config = self._perturb_configuration(base_config)

            replace_indices.append(int(bottom_idx))
            new_configs.append(perturbed_config)

        return replace_indices, new_configs

    def _sample_configuration(self, search_space: Dict, strategy: str = 'random') -> Dict:
        """Sample configuration from search space"""
        hyperparams = search_space['hyperparameters']
        config = {}

        rng_key = jax.random.PRNGKey(42)

        for param_name, param_spec in hyperparams.items():
            rng_key, subkey = jax.random.split(rng_key)

            if strategy == 'diverse':
                # Use wider sampling for diversity
                factor = 2.0
            else:
                factor = 1.0

            param_type = param_spec['type']

            if param_type == 'log_uniform':
                low, high = param_spec['bounds']
                # Expand range for diversity
                if strategy == 'diverse':
                    low = max(low * 0.1, 1e-8)
                    high = min(high * 10.0, 1.0)

                log_low, log_high = jnp.log(low), jnp.log(high)
                log_value = jax.random.uniform(subkey, minval=log_low, maxval=log_high)
                config[param_name] = float(jnp.exp(log_value))
            elif param_type == 'uniform':
                low, high = param_spec['bounds']
                config[param_name] = float(jax.random.uniform(subkey, minval=low, maxval=high))
            elif param_type == 'categorical':
                choices = param_spec['choices']
                idx = jax.random.randint(subkey, (1,), 0, len(choices))[0]
                config[param_name] = choices[int(idx)]
            else:
                config[param_name] = param_spec['default']

        return config

    def _perturb_configuration(self, config: Dict) -> Dict:
        """Add exploration noise to configuration"""
        perturbed = config.copy()
        rng_key = jax.random.PRNGKey(42)

        for key, value in config.items():
            rng_key, subkey = jax.random.split(rng_key)

            if isinstance(value, float):
                if 'learning_rate' in key or 'weight_decay' in key:
                    # Log-scale perturbation for learning rates
                    log_value = jnp.log(value)
                    noise = jax.random.normal(subkey) * 0.2  # 20% noise
                    perturbed[key] = float(jnp.exp(log_value + noise))
                else:
                    # Linear perturbation for other parameters
                    noise = jax.random.normal(subkey) * 0.1 * value
                    perturbed[key] = float(value + noise)
            # Keep categorical and integer values unchanged for simplicity

        return perturbed

    def run_pbt_optimization(self,
                           objective_fn: Callable,
                           search_space: Dict,
                           num_generations: int = 10,
                           steps_per_generation: int = 1000) -> Dict[str, Any]:
        """
        Run population-based training optimization.

        Args:
            objective_fn: Objective function that takes config and returns performance
            search_space: Hyperparameter search space
            num_generations: Number of PBT generations
            steps_per_generation: Training steps between generations

        Returns:
            Optimization results with best configuration and population history
        """
        # Initialize population
        population = self.initialize_population(search_space)
        generation_history = []

        for generation in range(num_generations):
            logger.info(f"PBT Generation {generation + 1}/{num_generations}")

            # Evaluate population
            for i, individual in enumerate(population):
                try:
                    # Simulate training for steps_per_generation
                    performance = objective_fn(individual['config'], steps_per_generation)
                    individual['performance'] = performance
                    individual['age'] += steps_per_generation

                    logger.info(f"Worker {i}: Performance = {performance:.4f}")
                except Exception as e:
                    logger.warning(f"Worker {i} evaluation failed: {e}")
                    individual['performance'] = -float('inf')

            # Record generation state
            generation_state = {
                'generation': generation,
                'population': [p.copy() for p in population],
                'best_performance': max(p['performance'] for p in population),
                'mean_performance': jnp.mean([p['performance'] for p in population])
            }
            generation_history.append(generation_state)

            # Exploit and explore (except last generation)
            if generation < num_generations - 1:
                performances = jnp.array([p['performance'] for p in population])
                configs = [p['config'] for p in population]

                replace_indices, new_configs = self.exploit_and_explore(performances, configs)

                # Replace bottom performers
                for idx, new_config in zip(replace_indices, new_configs):
                    population[idx]['config'] = new_config
                    population[idx]['performance'] = 0.0  # Reset performance
                    population[idx]['age'] = 0  # Reset age

                logger.info(f"Replaced {len(replace_indices)} individuals")

        # Find best configuration
        best_individual = max(population, key=lambda x: x['performance'])

        return {
            'best_config': best_individual['config'],
            'best_performance': best_individual['performance'],
            'final_population': population,
            'generation_history': generation_history,
            'total_evaluations': num_generations * self.population_size
        }
```

### Multi-Fidelity Hyperparameter Optimization
```python
# Multi-fidelity optimization for efficient hyperparameter search
class MultiFidelityOptimizer:
    """Multi-fidelity hyperparameter optimization using successive halving"""

    def __init__(self, min_budget: int = 1, max_budget: int = 100, eta: int = 3):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta  # Successive halving factor

    def successive_halving(self,
                          objective_fn: Callable,
                          configurations: List[Dict],
                          budget_levels: List[int]) -> Dict[str, Any]:
        """
        Successive halving algorithm for multi-fidelity optimization.

        Args:
            objective_fn: Function that takes (config, budget) and returns performance
            configurations: List of hyperparameter configurations to evaluate
            budget_levels: List of budget levels (e.g., training epochs)

        Returns:
            Results with best configuration and evaluation history
        """
        current_configs = configurations.copy()
        evaluation_history = []

        for i, budget in enumerate(budget_levels):
            logger.info(f"Successive halving round {i + 1}: Budget = {budget}, "
                       f"Configurations = {len(current_configs)}")

            # Evaluate all configurations at current budget
            results = []
            for j, config in enumerate(current_configs):
                try:
                    performance = objective_fn(config, budget)
                    results.append({
                        'config': config,
                        'performance': performance,
                        'budget': budget,
                        'round': i
                    })
                    logger.info(f"Config {j}: Performance = {performance:.4f} at budget {budget}")
                except Exception as e:
                    logger.warning(f"Configuration {j} failed: {e}")
                    results.append({
                        'config': config,
                        'performance': -float('inf'),
                        'budget': budget,
                        'round': i
                    })

            evaluation_history.extend(results)

            # Sort by performance and keep top fraction
            results.sort(key=lambda x: x['performance'], reverse=True)

            if i < len(budget_levels) - 1:  # Not the last round
                num_survivors = max(1, len(current_configs) // self.eta)
                current_configs = [r['config'] for r in results[:num_survivors]]
                logger.info(f"Advanced {num_survivors} configurations to next round")

        # Return best configuration
        best_result = max(evaluation_history, key=lambda x: x['performance'])

        return {
            'best_config': best_result['config'],
            'best_performance': best_result['performance'],
            'evaluation_history': evaluation_history,
            'total_evaluations': len(evaluation_history)
        }

    def hyperband_optimization(self,
                              objective_fn: Callable,
                              search_space: Dict,
                              max_iterations: int = 100) -> Dict[str, Any]:
        """
        HYPERBAND algorithm for multi-fidelity hyperparameter optimization.

        Args:
            objective_fn: Objective function (config, budget) -> performance
            search_space: Hyperparameter search space
            max_iterations: Maximum total iterations

        Returns:
            Optimization results with best configuration
        """
        # Calculate HYPERBAND parameters
        s_max = int(jnp.log(self.max_budget / self.min_budget) / jnp.log(self.eta))
        B = (s_max + 1) * self.max_budget

        all_results = []
        total_evaluations = 0

        for s in range(s_max, -1, -1):
            if total_evaluations >= max_iterations:
                break

            # Calculate number of configurations and budget allocation
            n = int(jnp.ceil((B / self.max_budget) * (self.eta ** s) / (s + 1)))
            r = self.max_budget * (self.eta ** (-s))

            # Generate random configurations
            configs = self._generate_random_configs(search_space, n)

            # Calculate budget levels for successive halving
            budget_levels = []
            current_budget = int(r)
            current_n = n

            while current_budget <= self.max_budget and current_n >= 1:
                budget_levels.append(current_budget)
                current_budget = int(current_budget * self.eta)
                current_n = max(1, current_n // self.eta)

            logger.info(f"HYPERBAND bracket s={s}: {n} configs, budget levels {budget_levels}")

            # Run successive halving
            results = self.successive_halving(objective_fn, configs, budget_levels)
            all_results.extend(results['evaluation_history'])
            total_evaluations += results['total_evaluations']

            if total_evaluations >= max_iterations:
                break

        # Find overall best configuration
        best_result = max(all_results, key=lambda x: x['performance'])

        return {
            'best_config': best_result['config'],
            'best_performance': best_result['performance'],
            'all_results': all_results,
            'total_evaluations': total_evaluations,
            'hyperband_brackets': s_max + 1
        }

    def _generate_random_configs(self, search_space: Dict, num_configs: int) -> List[Dict]:
        """Generate random configurations from search space"""
        configs = []
        hyperparams = search_space['hyperparameters']

        rng_key = jax.random.PRNGKey(42)

        for _ in range(num_configs):
            config = {}

            for param_name, param_spec in hyperparams.items():
                rng_key, subkey = jax.random.split(rng_key)

                param_type = param_spec['type']

                if param_type == 'log_uniform':
                    low, high = param_spec['bounds']
                    log_low, log_high = jnp.log(low), jnp.log(high)
                    log_value = jax.random.uniform(subkey, minval=log_low, maxval=log_high)
                    config[param_name] = float(jnp.exp(log_value))
                elif param_type == 'uniform':
                    low, high = param_spec['bounds']
                    config[param_name] = float(jax.random.uniform(subkey, minval=low, maxval=high))
                elif param_type == 'integer':
                    low, high = param_spec['bounds']
                    config[param_name] = int(jax.random.randint(subkey, (1,), low, high + 1)[0])
                elif param_type == 'categorical':
                    choices = param_spec['choices']
                    idx = jax.random.randint(subkey, (1,), 0, len(choices))[0]
                    config[param_name] = choices[int(idx)]
                else:
                    config[param_name] = param_spec['default']

            configs.append(config)

        return configs
```

### Automated Hyperparameter Scheduling
```python
# Automated hyperparameter scheduling and adaptation
class HyperparameterScheduler:
    """Intelligent hyperparameter scheduling during training"""

    def __init__(self, adaptation_strategy: str = "performance_based"):
        self.adaptation_strategy = adaptation_strategy
        self.schedule_history = []

    @functools.partial(jax.jit, static_argnums=(0,))
    def cosine_schedule_with_warmup(self,
                                   step: int,
                                   warmup_steps: int,
                                   total_steps: int,
                                   base_lr: float,
                                   min_lr: float = 0.0) -> float:
        """
        Cosine annealing schedule with warmup.

        Args:
            step: Current training step
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            base_lr: Peak learning rate
            min_lr: Minimum learning rate

        Returns:
            Current learning rate
        """
        if step < warmup_steps:
            # Linear warmup
            lr = base_lr * step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * progress))

        return lr

    @functools.partial(jax.jit, static_argnums=(0,))
    def exponential_decay_schedule(self,
                                  step: int,
                                  initial_lr: float,
                                  decay_steps: int,
                                  decay_rate: float) -> float:
        """
        Exponential decay learning rate schedule.

        Args:
            step: Current training step
            initial_lr: Initial learning rate
            decay_steps: Steps between decay applications
            decay_rate: Decay rate factor

        Returns:
            Current learning rate
        """
        return initial_lr * (decay_rate ** (step // decay_steps))

    @functools.partial(jax.jit, static_argnums=(0,))
    def polynomial_decay_schedule(self,
                                 step: int,
                                 initial_lr: float,
                                 total_steps: int,
                                 end_lr: float,
                                 power: float = 1.0) -> float:
        """
        Polynomial decay learning rate schedule.

        Args:
            step: Current training step
            initial_lr: Initial learning rate
            total_steps: Total training steps
            end_lr: Final learning rate
            power: Polynomial power

        Returns:
            Current learning rate
        """
        progress = jnp.minimum(step / total_steps, 1.0)
        return (initial_lr - end_lr) * ((1 - progress) ** power) + end_lr

    def adaptive_learning_rate(self,
                              current_lr: float,
                              loss_history: List[float],
                              patience: int = 5,
                              factor: float = 0.5,
                              min_lr: float = 1e-7) -> float:
        """
        Adaptive learning rate based on loss plateau detection.

        Args:
            current_lr: Current learning rate
            loss_history: Recent loss values
            patience: Steps to wait before reducing LR
            factor: Factor to multiply LR by
            min_lr: Minimum learning rate

        Returns:
            Updated learning rate
        """
        if len(loss_history) < patience + 1:
            return current_lr

        # Check for plateau (no improvement in last `patience` steps)
        recent_losses = loss_history[-patience:]
        best_recent = min(recent_losses)
        improvement = (loss_history[-patience-1] - best_recent) / abs(loss_history[-patience-1] + 1e-8)

        if improvement < 1e-4:  # No significant improvement
            new_lr = max(current_lr * factor, min_lr)
            logger.info(f"Reducing learning rate from {current_lr:.2e} to {new_lr:.2e}")
            return new_lr

        return current_lr

    def create_adaptive_schedule(self,
                               initial_config: Dict,
                               adaptation_config: Dict = None) -> Callable:
        """
        Create adaptive hyperparameter schedule.

        Args:
            initial_config: Initial hyperparameter configuration
            adaptation_config: Adaptation strategy configuration

        Returns:
            Schedule function that adapts hyperparameters
        """
        if adaptation_config is None:
            adaptation_config = {}

        schedule_type = adaptation_config.get('schedule_type', 'cosine_warmup')
        adaptation_frequency = adaptation_config.get('adaptation_frequency', 100)

        def schedule_fn(step: int, metrics: Dict = None) -> Dict:
            """
            Adaptive schedule function.

            Args:
                step: Current training step
                metrics: Current training metrics

            Returns:
                Updated hyperparameter configuration
            """
            config = initial_config.copy()

            # Update learning rate based on schedule
            if schedule_type == 'cosine_warmup':
                warmup_steps = adaptation_config.get('warmup_steps', 1000)
                total_steps = adaptation_config.get('total_steps', 10000)

                config['learning_rate'] = self.cosine_schedule_with_warmup(
                    step, warmup_steps, total_steps,
                    initial_config['learning_rate']
                )

            elif schedule_type == 'exponential':
                decay_steps = adaptation_config.get('decay_steps', 1000)
                decay_rate = adaptation_config.get('decay_rate', 0.96)

                config['learning_rate'] = self.exponential_decay_schedule(
                    step, initial_config['learning_rate'],
                    decay_steps, decay_rate
                )

            # Adaptive adjustments based on metrics
            if metrics is not None and step % adaptation_frequency == 0:
                loss_history = metrics.get('loss_history', [])

                if len(loss_history) > 10:
                    # Adaptive learning rate
                    config['learning_rate'] = self.adaptive_learning_rate(
                        config['learning_rate'],
                        loss_history,
                        patience=adaptation_config.get('patience', 5)
                    )

                    # Adaptive regularization
                    if 'weight_decay' in config:
                        # Increase weight decay if overfitting detected
                        train_loss = loss_history[-1]
                        val_loss = metrics.get('val_loss', train_loss)

                        if val_loss > train_loss * 1.1:  # Overfitting indicator
                            config['weight_decay'] = min(
                                config['weight_decay'] * 1.1,
                                1e-2
                            )

            # Record schedule state
            self.schedule_history.append({
                'step': step,
                'config': config.copy(),
                'metrics': metrics
            })

            return config

        return schedule_fn
```

## Integration with Scientific Computing

### Scientific Hyperparameter Optimization
- **Physics-Informed Constraints**: Optimization with physical constraints and domain knowledge
- **Multi-Scale Optimization**: Hyperparameter optimization across different scales and resolutions
- **Uncertainty-Aware Optimization**: Robust optimization under measurement and model uncertainty
- **Domain-Specific Objectives**: Scientific metrics and domain-specific performance measures

### Multi-Objective Scientific Optimization
- **Accuracy vs Efficiency**: Trade-offs between scientific accuracy and computational efficiency
- **Robustness vs Performance**: Balancing model robustness with peak performance
- **Interpretability vs Complexity**: Optimizing for model interpretability in scientific contexts
- **Conservation vs Flexibility**: Enforcing physical conservation laws while maintaining model flexibility

### Integration with Neural Architecture Search
- **Joint Optimization**: Simultaneous optimization of architecture and hyperparameters
- **Progressive Refinement**: Hierarchical optimization from architecture to fine-grained hyperparameters
- **Transfer Learning**: Hyperparameter transfer across related architectures and domains
- **Efficiency-Aware Search**: Hardware and computational constraint consideration

## Usage Examples

### Bayesian Hyperparameter Optimization
```python
# Create hyperparameter optimizer
optimizer = BayesianHyperparameterOptimizer(acquisition_function="expected_improvement")

# Define search space
search_space = optimizer.create_search_space("transformer")

# Define objective function
def objective_function(config):
    # Train model with given hyperparameters
    model = create_transformer_model(config)
    performance = train_and_evaluate(model, config)
    return performance

# Optimize hyperparameters
result = optimizer.optimize_hyperparameters(
    objective_fn=objective_function,
    search_space=search_space,
    optimization_config={
        'num_iterations': 30,
        'num_initial_points': 5,
        'acquisition_function': 'expected_improvement'
    }
)

print(f"Best configuration: {result['best_config']}")
print(f"Best performance: {result['best_performance']:.4f}")
```

### Population-Based Training
```python
# Initialize PBT optimizer
pbt = PopulationBasedTraining(population_size=20, exploit_threshold=0.8)

# Define training objective
def pbt_objective(config, num_steps):
    # Simulate training for specified steps
    model = create_model(config)
    performance = train_model(model, config, num_steps)
    return performance

# Run PBT optimization
pbt_result = pbt.run_pbt_optimization(
    objective_fn=pbt_objective,
    search_space=search_space,
    num_generations=10,
    steps_per_generation=1000
)

print(f"PBT best performance: {pbt_result['best_performance']:.4f}")
```

### Multi-Fidelity Optimization
```python
# Multi-fidelity optimizer with HYPERBAND
mf_optimizer = MultiFidelityOptimizer(min_budget=5, max_budget=100, eta=3)

# Define budget-aware objective
def budget_objective(config, budget):
    # Train for 'budget' epochs
    model = create_model(config)
    performance = train_model(model, config, epochs=budget)
    return performance

# Run HYPERBAND optimization
hyperband_result = mf_optimizer.hyperband_optimization(
    objective_fn=budget_objective,
    search_space=search_space,
    max_iterations=200
)

print(f"HYPERBAND best: {hyperband_result['best_performance']:.4f}")
```

### Adaptive Scheduling
```python
# Create adaptive hyperparameter scheduler
scheduler = HyperparameterScheduler(adaptation_strategy="performance_based")

# Initial configuration
initial_config = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'dropout_rate': 0.1
}

# Create adaptive schedule
adaptation_config = {
    'schedule_type': 'cosine_warmup',
    'warmup_steps': 1000,
    'total_steps': 50000,
    'adaptation_frequency': 500,
    'patience': 10
}

schedule_fn = scheduler.create_adaptive_schedule(initial_config, adaptation_config)

# Use during training
for step in range(50000):
    current_config = schedule_fn(step, metrics={'loss_history': loss_history})
    # Apply current_config to training
```

This expert provides comprehensive hyperparameter optimization capabilities with advanced techniques, scientific computing integration, and automated adaptation strategies for neural network training workflows.