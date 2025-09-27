---
description: Compute sparse Jacobians in JAX for efficient optimization using sparsity patterns and NLSQ integration
category: jax-optimization
argument-hint: "[--sparsity-pattern] [--estimation-method] [--nlsq-integration] [--memory-efficient]"
allowed-tools: "*"
---

# /jax-sparse-jac

Compute sparse Jacobians in JAX for efficient optimization. Integrate with NLSQ for sparse problems, using jax.jacfwd with sparsity patterns.

## Description

Advanced sparse Jacobian computation in JAX for efficient optimization of problems with structured sparsity. Includes automatic sparsity pattern detection, memory-efficient computation strategies, and seamless integration with NLSQ solvers for large-scale optimization problems.

## Usage

```
/jax-sparse-jac [--sparsity-pattern] [--estimation-method] [--nlsq-integration] [--memory-efficient]
```

## What it does

1. Detect and exploit sparsity patterns in Jacobian matrices
2. Implement memory-efficient sparse Jacobian computation
3. Integrate sparse Jacobians with NLSQ optimization algorithms
4. Apply graph coloring for efficient derivative computation
5. Handle dynamic sparsity patterns and adaptive algorithms

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, jacfwd, jacrev, vmap
import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional, List, Set
import functools
from scipy.sparse import csr_matrix, csc_matrix
import networkx as nx

# Configure JAX for sparse computation
jax.config.update('jax_enable_x64', True)

try:
    from nlsq import CurveFit, LevenbergMarquardt, TrustRegionReflective
    NLSQ_AVAILABLE = True
except ImportError:
    NLSQ_AVAILABLE = False
    print("NLSQ not available. Some functionality will use JAX-only implementations.")

class SparseJacobianComputer:
    """Advanced sparse Jacobian computation with sparsity pattern exploitation"""

    def __init__(self,
                 sparsity_threshold: float = 1e-12,
                 memory_efficient: bool = True,
                 use_graph_coloring: bool = True,
                 adaptive_sparsity: bool = True):

        self.sparsity_threshold = sparsity_threshold
        self.memory_efficient = memory_efficient
        self.use_graph_coloring = use_graph_coloring
        self.adaptive_sparsity = adaptive_sparsity

        self.cached_sparsity_patterns = {}
        self.coloring_schemes = {}

    def detect_sparsity_pattern(self,
                              func: Callable,
                              x: jax.Array,
                              *args,
                              sample_points: int = 5) -> Tuple[jax.Array, Dict[str, Any]]:
        """Detect sparsity pattern by sampling function at multiple points"""

        print(f"Detecting sparsity pattern with {sample_points} sample points...")

        def compute_jacobian_sample(sample_x):
            return jacfwd(func)(sample_x, *args)

        # Sample at multiple points to detect consistent sparsity
        key = random.PRNGKey(42)
        sample_xs = []

        # Include the original point
        sample_xs.append(x)

        # Add perturbed versions
        for i in range(sample_points - 1):
            perturbation = 0.1 * random.normal(random.split(key, sample_points)[i], x.shape)
            sample_xs.append(x + perturbation)

        # Compute Jacobians at all sample points
        jacobians = []
        for sample_x in sample_xs:
            jac = compute_jacobian_sample(sample_x)
            jacobians.append(jac)

        # Determine sparsity pattern (elements that are zero across all samples)
        stacked_jacobians = jnp.stack(jacobians)
        max_abs_values = jnp.max(jnp.abs(stacked_jacobians), axis=0)
        sparsity_pattern = max_abs_values > self.sparsity_threshold

        # Analyze sparsity statistics
        total_elements = sparsity_pattern.size
        nonzero_elements = jnp.sum(sparsity_pattern)
        sparsity_ratio = 1.0 - (nonzero_elements / total_elements)

        pattern_info = {
            'pattern': sparsity_pattern,
            'total_elements': int(total_elements),
            'nonzero_elements': int(nonzero_elements),
            'sparsity_ratio': float(sparsity_ratio),
            'shape': sparsity_pattern.shape,
            'memory_savings': float(sparsity_ratio),
            'is_sparse': sparsity_ratio > 0.1  # Consider sparse if >10% zeros
        }

        print(f"Sparsity detected: {sparsity_ratio:.2%} sparse "
              f"({int(nonzero_elements)}/{int(total_elements)} nonzero elements)")

        return sparsity_pattern, pattern_info

    def graph_coloring_for_jacobian(self, sparsity_pattern: jax.Array) -> Dict[str, Any]:
        """Apply graph coloring to optimize sparse Jacobian computation"""

        if not self.use_graph_coloring:
            return {'colors': None, 'n_colors': sparsity_pattern.shape[1]}

        print("Applying graph coloring for efficient computation...")

        # Convert sparsity pattern to adjacency matrix
        # Two columns are adjacent if they share a nonzero row
        n_cols = sparsity_pattern.shape[1]
        adjacency = jnp.zeros((n_cols, n_cols))

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                # Check if columns i and j have overlapping nonzeros
                col_i_nonzeros = sparsity_pattern[:, i]
                col_j_nonzeros = sparsity_pattern[:, j]
                overlap = jnp.any(col_i_nonzeros & col_j_nonzeros)

                if overlap:
                    adjacency = adjacency.at[i, j].set(1)
                    adjacency = adjacency.at[j, i].set(1)

        # Use NetworkX for graph coloring (if available)
        try:
            import networkx as nx

            # Convert to NetworkX graph
            G = nx.from_numpy_array(np.array(adjacency))

            # Apply greedy coloring
            coloring = nx.greedy_color(G, strategy='largest_first')

            # Convert back to JAX arrays
            n_colors = max(coloring.values()) + 1 if coloring else 1
            color_assignment = jnp.zeros(n_cols, dtype=jnp.int32)

            for node, color in coloring.items():
                color_assignment = color_assignment.at[node].set(color)

            coloring_info = {
                'colors': color_assignment,
                'n_colors': n_colors,
                'compression_ratio': n_cols / n_colors,
                'method': 'graph_coloring'
            }

            print(f"Graph coloring: {n_cols} columns → {n_colors} colors "
                  f"(compression: {n_cols/n_colors:.1f}x)")

        except ImportError:
            # Fallback: simple greedy coloring
            print("NetworkX not available, using simple greedy coloring")
            color_assignment = jnp.arange(n_cols) % min(n_cols, 32)  # Max 32 colors
            n_colors = min(n_cols, 32)

            coloring_info = {
                'colors': color_assignment,
                'n_colors': n_colors,
                'compression_ratio': n_cols / n_colors,
                'method': 'simple_greedy'
            }

        return coloring_info

    def compute_sparse_jacobian_colored(self,
                                      func: Callable,
                                      x: jax.Array,
                                      sparsity_pattern: jax.Array,
                                      coloring_info: Dict[str, Any],
                                      *args) -> jax.Array:
        """Compute sparse Jacobian using graph coloring"""

        if coloring_info['colors'] is None:
            # Fallback to standard computation
            return self.compute_sparse_jacobian_standard(func, x, sparsity_pattern, *args)

        colors = coloring_info['colors']
        n_colors = coloring_info['n_colors']
        n_params = len(x)

        @jit
        def compute_colored_derivatives(color_idx):
            """Compute derivatives for all parameters of the same color"""
            # Create perturbation vector for this color
            perturbation_mask = (colors == color_idx).astype(jnp.float64)
            eps = 1e-8

            # Forward difference
            x_plus = x + eps * perturbation_mask
            f_plus = func(x_plus, *args)
            f_center = func(x, *args)

            # Compute derivatives
            derivatives = (f_plus - f_center) / eps
            return derivatives

        # Compute derivatives for each color
        all_derivatives = []
        for color_idx in range(n_colors):
            colored_derivs = compute_colored_derivatives(color_idx)
            all_derivatives.append(colored_derivs)

        # Reconstruct full Jacobian from colored derivatives
        output_dim = all_derivatives[0].shape[0]
        jacobian = jnp.zeros((output_dim, n_params))

        for color_idx in range(n_colors):
            # Find parameters with this color
            param_mask = (colors == color_idx)
            param_indices = jnp.where(param_mask)[0]

            # Assign derivatives to correct columns
            for i, param_idx in enumerate(param_indices):
                jacobian = jacobian.at[:, param_idx].set(all_derivatives[color_idx])

        # Apply sparsity pattern
        sparse_jacobian = jacobian * sparsity_pattern

        return sparse_jacobian

    def compute_sparse_jacobian_standard(self,
                                       func: Callable,
                                       x: jax.Array,
                                       sparsity_pattern: jax.Array,
                                       *args) -> jax.Array:
        """Compute sparse Jacobian using standard JAX AD"""

        @jit
        def sparse_jacobian_fn(params):
            full_jacobian = jacfwd(func)(params, *args)
            return full_jacobian * sparsity_pattern

        return sparse_jacobian_fn(x)

    def adaptive_sparsity_jacobian(self,
                                 func: Callable,
                                 x: jax.Array,
                                 *args,
                                 update_frequency: int = 10,
                                 iteration: int = 0) -> Tuple[jax.Array, Dict[str, Any]]:
        """Compute Jacobian with adaptive sparsity pattern detection"""

        func_id = id(func)

        # Check if we need to update sparsity pattern
        update_pattern = (
            func_id not in self.cached_sparsity_patterns or
            (self.adaptive_sparsity and iteration % update_frequency == 0)
        )

        if update_pattern:
            print(f"Updating sparsity pattern at iteration {iteration}")

            # Detect current sparsity pattern
            sparsity_pattern, pattern_info = self.detect_sparsity_pattern(func, x, *args)

            # Update graph coloring if needed
            if pattern_info['is_sparse']:
                coloring_info = self.graph_coloring_for_jacobian(sparsity_pattern)
            else:
                coloring_info = {'colors': None, 'n_colors': len(x)}

            # Cache the pattern and coloring
            self.cached_sparsity_patterns[func_id] = {
                'sparsity_pattern': sparsity_pattern,
                'pattern_info': pattern_info,
                'coloring_info': coloring_info,
                'last_updated': iteration
            }

        # Use cached pattern
        cached_data = self.cached_sparsity_patterns[func_id]
        sparsity_pattern = cached_data['sparsity_pattern']
        pattern_info = cached_data['pattern_info']
        coloring_info = cached_data['coloring_info']

        # Compute sparse Jacobian
        if pattern_info['is_sparse'] and self.use_graph_coloring:
            jacobian = self.compute_sparse_jacobian_colored(
                func, x, sparsity_pattern, coloring_info, *args
            )
        else:
            jacobian = self.compute_sparse_jacobian_standard(
                func, x, sparsity_pattern, *args
            )

        return jacobian, pattern_info

# Integration with NLSQ optimization
class SparseNLSQOptimizer:
    """NLSQ optimization with sparse Jacobian support"""

    def __init__(self, sparse_jacobian_computer: SparseJacobianComputer):
        self.sparse_jac = sparse_jacobian_computer

    def sparse_residual_function(self,
                                func: Callable,
                                sparsity_info: Dict[str, Any]) -> Callable:
        """Create residual function optimized for sparse Jacobians"""

        @jit
        def residual_fn(params, x_data, y_data):
            predictions = func(params, x_data)
            return predictions - y_data

        return residual_fn

    def sparse_nlsq_optimization(self,
                                residual_fn: Callable,
                                initial_params: jax.Array,
                                x_data: jax.Array,
                                y_data: jax.Array,
                                max_iterations: int = 100) -> Dict[str, Any]:
        """NLSQ optimization using sparse Jacobians"""

        if not NLSQ_AVAILABLE:
            return self._jax_sparse_nlsq_fallback(
                residual_fn, initial_params, x_data, y_data, max_iterations
            )

        params = initial_params
        iteration = 0
        convergence_history = []

        print("Starting sparse NLSQ optimization...")

        for iteration in range(max_iterations):
            # Compute residuals
            residuals = residual_fn(params, x_data, y_data)

            # Compute sparse Jacobian
            sparse_jacobian, sparsity_info = self.sparse_jac.adaptive_sparsity_jacobian(
                lambda p: residual_fn(p, x_data, y_data),
                params,
                iteration=iteration
            )

            # Check convergence
            residual_norm = jnp.linalg.norm(residuals)
            convergence_history.append(residual_norm)

            if residual_norm < 1e-8:
                print(f"Converged after {iteration} iterations")
                break

            # Sparse Gauss-Newton step
            try:
                # Use only nonzero parts for efficiency
                nonzero_mask = sparsity_info['pattern']
                sparse_J = sparse_jacobian * nonzero_mask

                # Solve normal equations: J^T J δ = -J^T r
                JtJ = jnp.dot(sparse_J.T, sparse_J)
                Jtr = jnp.dot(sparse_J.T, residuals)

                # Add regularization for stability
                regularization = 1e-6 * jnp.eye(JtJ.shape[0])
                delta = jnp.linalg.solve(JtJ + regularization, -Jtr)

                # Update parameters
                params = params + delta

            except jnp.linalg.LinAlgError:
                print(f"Linear algebra error at iteration {iteration}, using damped update")
                # Damped update
                gradient = jnp.dot(sparse_jacobian.T, residuals)
                params = params - 0.01 * gradient

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: residual norm = {residual_norm:.6e}, "
                      f"sparsity = {sparsity_info.get('sparsity_ratio', 0):.2%}")

        return {
            'params': params,
            'residual_norm': convergence_history[-1] if convergence_history else float('inf'),
            'iterations': iteration,
            'convergence_history': jnp.array(convergence_history),
            'final_sparsity_info': sparsity_info,
            'success': residual_norm < 1e-6 if 'residual_norm' in locals() else False
        }

    def _jax_sparse_nlsq_fallback(self,
                                residual_fn: Callable,
                                initial_params: jax.Array,
                                x_data: jax.Array,
                                y_data: jax.Array,
                                max_iterations: int) -> Dict[str, Any]:
        """JAX-only sparse NLSQ implementation when NLSQ library is not available"""

        print("Using JAX-only sparse NLSQ implementation")

        params = initial_params
        convergence_history = []

        for iteration in range(max_iterations):
            # Compute sparse Jacobian and residuals
            sparse_jacobian, sparsity_info = self.sparse_jac.adaptive_sparsity_jacobian(
                lambda p: residual_fn(p, x_data, y_data),
                params,
                iteration=iteration
            )

            residuals = residual_fn(params, x_data, y_data)
            residual_norm = jnp.linalg.norm(residuals)
            convergence_history.append(residual_norm)

            if residual_norm < 1e-8:
                break

            # Levenberg-Marquardt step with sparsity
            lambda_reg = 1e-3
            JtJ = jnp.dot(sparse_jacobian.T, sparse_jacobian)
            Jtr = jnp.dot(sparse_jacobian.T, residuals)

            # Add damping
            JtJ_damped = JtJ + lambda_reg * jnp.diag(jnp.diag(JtJ))

            try:
                delta = jnp.linalg.solve(JtJ_damped, -Jtr)
                params = params + delta
            except jnp.linalg.LinAlgError:
                # Gradient descent fallback
                gradient = jnp.dot(sparse_jacobian.T, residuals)
                params = params - 0.01 * gradient

        return {
            'params': params,
            'residual_norm': convergence_history[-1] if convergence_history else float('inf'),
            'iterations': len(convergence_history),
            'convergence_history': jnp.array(convergence_history),
            'success': convergence_history[-1] < 1e-6 if convergence_history else False
        }

# Specialized sparse problems
class SpecializedSparseProblems:
    """Examples of specialized sparse Jacobian problems"""

    @staticmethod
    def large_scale_parameter_estimation(n_params: int = 1000,
                                       n_observations: int = 5000,
                                       sparsity_ratio: float = 0.95) -> Tuple[Callable, jax.Array, jax.Array, jax.Array]:
        """Generate a large-scale sparse parameter estimation problem"""

        key = random.PRNGKey(42)

        # Generate sparse connectivity pattern
        n_connections = int(n_params * n_observations * (1 - sparsity_ratio))
        connection_indices = random.choice(
            key, n_params * n_observations, (n_connections,), replace=False
        )

        connectivity_matrix = jnp.zeros(n_params * n_observations)
        connectivity_matrix = connectivity_matrix.at[connection_indices].set(1.0)
        connectivity_matrix = connectivity_matrix.reshape(n_observations, n_params)

        # True parameters (sparse)
        true_params = random.normal(random.split(key)[0], (n_params,)) * connectivity_matrix.T
        true_params = jnp.sum(true_params, axis=0)  # Sum over connections

        # Generate observations
        x_data = jnp.linspace(0, 10, n_observations)

        def sparse_model(params, x):
            # Each observation depends only on a few parameters
            predictions = jnp.zeros(len(x))
            for i in range(len(x)):
                connected_params = params * connectivity_matrix[i]
                predictions = predictions.at[i].set(jnp.sum(connected_params * jnp.sin(x[i])))
            return predictions

        # Generate noisy observations
        y_true = sparse_model(true_params, x_data)
        noise = 0.1 * random.normal(random.split(key)[1], y_true.shape)
        y_data = y_true + noise

        return sparse_model, true_params, x_data, y_data

    @staticmethod
    def pde_discretization_problem() -> Tuple[Callable, jax.Array]:
        """Generate a PDE discretization problem with sparse Jacobian"""

        # 2D Poisson equation discretization
        nx, ny = 50, 50
        n_unknowns = nx * ny

        def poisson_residual(u_flat, source_term):
            u = u_flat.reshape(nx, ny)
            residual = jnp.zeros_like(u)

            # Interior points (5-point stencil)
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    laplacian = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j])
                    residual = residual.at[i, j].set(-laplacian - source_term[i, j])

            # Boundary conditions (Dirichlet: u = 0)
            residual = residual.at[0, :].set(u[0, :])
            residual = residual.at[-1, :].set(u[-1, :])
            residual = residual.at[:, 0].set(u[:, 0])
            residual = residual.at[:, -1].set(u[:, -1])

            return residual.flatten()

        # Source term
        x = jnp.linspace(0, 1, nx)
        y = jnp.linspace(0, 1, ny)
        X, Y = jnp.meshgrid(x, y)
        source = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

        # Initial guess
        initial_u = jnp.zeros(n_unknowns)

        return lambda u: poisson_residual(u, source), initial_u

# Comprehensive demonstration
def demonstrate_sparse_jacobian_computation():
    """Demonstrate sparse Jacobian computation capabilities"""

    print("=== Sparse Jacobian Computation Demonstration ===")

    # Initialize sparse Jacobian computer
    sparse_jac = SparseJacobianComputer(
        sparsity_threshold=1e-10,
        memory_efficient=True,
        use_graph_coloring=True,
        adaptive_sparsity=True
    )

    # Test 1: Simple sparse function
    print("\\n1. Simple Sparse Function Test:")

    def sparse_test_function(x):
        """Function with known sparse Jacobian pattern"""
        n = len(x)
        result = jnp.zeros(n)

        # Each output depends only on 2-3 inputs (sparse pattern)
        for i in range(n):
            if i == 0:
                result = result.at[i].set(x[0]**2 + x[1])
            elif i == n-1:
                result = result.at[i].set(x[i-1] + x[i]**2)
            else:
                result = result.at[i].set(x[i-1] + 2*x[i] + x[i+1])

        return result

    x_test = jnp.ones(20)
    sparse_jacobian, sparsity_info = sparse_jac.adaptive_sparsity_jacobian(
        sparse_test_function, x_test
    )

    print(f"Detected sparsity: {sparsity_info['sparsity_ratio']:.2%}")
    print(f"Jacobian shape: {sparse_jacobian.shape}")
    print(f"Nonzero elements: {sparsity_info['nonzero_elements']}")

    # Test 2: Large-scale parameter estimation
    print("\\n2. Large-Scale Parameter Estimation:")

    sparse_model, true_params, x_data, y_data = SpecializedSparseProblems.large_scale_parameter_estimation(
        n_params=100, n_observations=500, sparsity_ratio=0.9
    )

    def residual_function(params):
        return sparse_model(params, x_data) - y_data

    # Initialize sparse NLSQ optimizer
    sparse_nlsq = SparseNLSQOptimizer(sparse_jac)

    # Run optimization
    initial_guess = jnp.zeros_like(true_params)
    result = sparse_nlsq.sparse_nlsq_optimization(
        lambda p, x, y: sparse_model(p, x) - y,
        initial_guess, x_data, y_data,
        max_iterations=50
    )

    print(f"Optimization completed in {result['iterations']} iterations")
    print(f"Final residual norm: {result['residual_norm']:.6e}")
    print(f"Parameter error: {jnp.linalg.norm(result['params'] - true_params):.6e}")

    # Test 3: PDE discretization
    print("\\n3. PDE Discretization Problem:")

    pde_residual, initial_solution = SpecializedSparseProblems.pde_discretization_problem()

    # Analyze sparsity of PDE Jacobian
    pde_jacobian, pde_sparsity = sparse_jac.adaptive_sparsity_jacobian(
        pde_residual, initial_solution
    )

    print(f"PDE Jacobian sparsity: {pde_sparsity['sparsity_ratio']:.2%}")
    print(f"PDE problem size: {len(initial_solution)} unknowns")
    print(f"Memory savings: {pde_sparsity['memory_savings']:.1%}")

    # Performance comparison
    print("\\n4. Performance Comparison:")

    def compare_sparse_vs_dense():
        # Time sparse computation
        start_time = time.time()
        for _ in range(10):
            sparse_jac.adaptive_sparsity_jacobian(sparse_test_function, x_test)
        sparse_time = time.time() - start_time

        # Time dense computation
        start_time = time.time()
        for _ in range(10):
            jacfwd(sparse_test_function)(x_test)
        dense_time = time.time() - start_time

        speedup = dense_time / sparse_time
        print(f"Sparse computation time: {sparse_time:.4f}s")
        print(f"Dense computation time: {dense_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        return speedup

    speedup = compare_sparse_vs_dense()

    return sparse_jac, result, sparsity_info

if __name__ == "__main__":
    import time
    sparse_computer, optimization_result, final_sparsity = demonstrate_sparse_jacobian_computation()
    print("\\n=== Sparse Jacobian Computation Complete ===")
```

## Related Commands

- `/jax-nlsq-fit` - Apply sparse Jacobians to NLSQ curve fitting
- `/jax-nlsq-large` - Combine with massive dataset optimization
- `/jax-grad` - Understand automatic differentiation for sparse problems
- `/jax-cache-opt` - Cache sparse Jacobian computations efficiently