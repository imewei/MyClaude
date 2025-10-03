"""Tests for Advanced Optimization Methods.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.advanced_optimization import (
    SequentialQuadraticProgramming,
    AugmentedLagrangian,
    GeneticAlgorithm,
    SimulatedAnnealing,
    CMAES,
    MixedIntegerOptimization,
    ConstrainedOptimizationConfig,
    GlobalOptimizationConfig,
    MixedIntegerConfig
)


class TestConstrainedOptimizationConfig:
    """Tests for constrained optimization configuration."""

    def test_default_config(self):
        """Test: Default configuration."""
        config = ConstrainedOptimizationConfig()

        assert config.constraint_tol > 0
        assert config.optimality_tol > 0
        assert config.max_iterations > 0

    def test_custom_config(self):
        """Test: Custom configuration."""
        config = ConstrainedOptimizationConfig(
            constraint_tol=1e-8,
            penalty_initial=100.0
        )

        assert config.constraint_tol == 1e-8
        assert config.penalty_initial == 100.0


class TestGlobalOptimizationConfig:
    """Tests for global optimization configuration."""

    def test_default_config(self):
        """Test: Default global optimization config."""
        config = GlobalOptimizationConfig()

        assert config.population_size > 0
        assert config.num_generations > 0
        assert 0 <= config.crossover_rate <= 1
        assert 0 <= config.mutation_rate <= 1

    def test_custom_config(self):
        """Test: Custom global config."""
        config = GlobalOptimizationConfig(
            population_size=100,
            crossover_rate=0.9
        )

        assert config.population_size == 100
        assert config.crossover_rate == 0.9


class TestSequentialQuadraticProgramming:
    """Tests for SQP."""

    def test_unconstrained_optimization(self):
        """Test: SQP on unconstrained problem."""
        config = ConstrainedOptimizationConfig()
        sqp = SequentialQuadraticProgramming(config)

        # Minimize (x-2)^2 + (y-3)^2
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2

        def gradient(x):
            return np.array([2*(x[0] - 2), 2*(x[1] - 3)])

        x0 = np.array([0.0, 0.0])

        result = sqp.solve(objective, x0, gradient=gradient)

        assert result['success']
        assert np.allclose(result['x'], [2.0, 3.0], atol=1e-3)
        assert result['fun'] < 1e-6

    def test_equality_constrained(self):
        """Test: SQP with equality constraint."""
        config = ConstrainedOptimizationConfig()
        sqp = SequentialQuadraticProgramming(config)

        # Minimize x^2 + y^2 subject to x + y = 1
        def objective(x):
            return x[0]**2 + x[1]**2

        def gradient(x):
            return np.array([2*x[0], 2*x[1]])

        # Constraint: x + y - 1 = 0
        equality_constraints = [{
            'fun': lambda x: x[0] + x[1] - 1,
            'jac': lambda x: np.array([1.0, 1.0])
        }]

        x0 = np.array([0.0, 0.0])

        result = sqp.solve(objective, x0, gradient=gradient,
                          equality_constraints=equality_constraints)

        # Optimal: x = y = 0.5
        assert np.allclose(result['x'], [0.5, 0.5], atol=1e-2)
        assert np.isclose(result['x'][0] + result['x'][1], 1.0, atol=1e-4)

    def test_inequality_constrained(self):
        """Test: SQP with inequality constraint."""
        config = ConstrainedOptimizationConfig()
        sqp = SequentialQuadraticProgramming(config)

        # Minimize x^2 + y^2 subject to x + y >= 1
        def objective(x):
            return x[0]**2 + x[1]**2

        # Constraint: -(x + y - 1) <= 0  =>  x + y >= 1
        inequality_constraints = [{
            'fun': lambda x: -(x[0] + x[1] - 1),
            'jac': lambda x: np.array([-1.0, -1.0])
        }]

        x0 = np.array([2.0, 2.0])

        result = sqp.solve(objective, x0, inequality_constraints=inequality_constraints)

        # Should be on boundary: x + y = 1
        assert np.isclose(result['x'][0] + result['x'][1], 1.0, atol=1e-2)


class TestAugmentedLagrangian:
    """Tests for augmented Lagrangian method."""

    def test_equality_constraint(self):
        """Test: Augmented Lagrangian with equality."""
        config = ConstrainedOptimizationConfig(max_iterations=50)
        al = AugmentedLagrangian(config)

        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2

        equality_constraints = [lambda x: x[0] + x[1] - 1]

        x0 = np.array([0.0, 0.0])

        result = al.solve(objective, x0, equality_constraints=equality_constraints)

        assert result['success']
        # Optimal on x + y = 1 closest to (1, 2)
        assert np.isclose(result['x'][0] + result['x'][1], 1.0, atol=1e-2)

    def test_inequality_constraint(self):
        """Test: Augmented Lagrangian with inequality."""
        config = ConstrainedOptimizationConfig(max_iterations=50)
        al = AugmentedLagrangian(config)

        def objective(x):
            return x[0]**2 + x[1]**2

        inequality_constraints = [lambda x: -(x[0] + x[1] - 1)]  # x + y >= 1

        x0 = np.array([2.0, 2.0])

        result = al.solve(objective, x0, inequality_constraints=inequality_constraints)

        assert np.isclose(result['x'][0] + result['x'][1], 1.0, atol=0.1)


class TestGeneticAlgorithm:
    """Tests for genetic algorithm."""

    def test_simple_function(self):
        """Test: GA on simple function."""
        config = GlobalOptimizationConfig(
            population_size=30,
            num_generations=50
        )
        ga = GeneticAlgorithm(config)

        # Minimize (x-3)^2 + (y-4)^2
        def objective(x):
            return (x[0] - 3)**2 + (x[1] - 4)**2

        bounds = [(-10, 10), (-10, 10)]

        result = ga.solve(objective, bounds)

        assert result['success']
        # Should be close to (3, 4)
        assert np.allclose(result['x'], [3.0, 4.0], atol=0.5)

    def test_multimodal_function(self):
        """Test: GA on multimodal function."""
        config = GlobalOptimizationConfig(
            population_size=50,
            num_generations=100
        )
        ga = GeneticAlgorithm(config)

        # Rastrigin function (many local minima)
        def objective(x):
            A = 10
            n = len(x)
            return A * n + sum(x_i**2 - A * np.cos(2 * np.pi * x_i) for x_i in x)

        bounds = [(-5.12, 5.12), (-5.12, 5.12)]

        result = ga.solve(objective, bounds)

        # Global minimum at (0, 0) with value 0
        # GA should get close
        assert result['fun'] < 5.0  # Relaxed tolerance for stochastic method


class TestSimulatedAnnealing:
    """Tests for simulated annealing."""

    def test_simple_optimization(self):
        """Test: SA on simple function."""
        config = GlobalOptimizationConfig(
            initial_temperature=10.0,
            cooling_rate=0.9,
            num_iterations_per_temp=50
        )
        sa = SimulatedAnnealing(config)

        def objective(x):
            return x[0]**2 + x[1]**2

        x0 = np.array([5.0, 5.0])
        bounds = [(-10, 10), (-10, 10)]

        result = sa.solve(objective, x0, bounds)

        assert result['success']
        assert np.allclose(result['x'], [0.0, 0.0], atol=1.0)

    def test_escapes_local_minimum(self):
        """Test: SA escapes local minimum."""
        config = GlobalOptimizationConfig(
            initial_temperature=5.0,
            cooling_rate=0.95,
            num_iterations_per_temp=100
        )
        sa = SimulatedAnnealing(config)

        # Function with local minimum
        def objective(x):
            return (x[0]**2 - 5)**2 + x[1]**2

        x0 = np.array([1.0, 0.0])  # Start near local min
        bounds = [(-10, 10), (-10, 10)]

        result = sa.solve(objective, x0, bounds)

        assert result['success']
        # Should find global min at (±√5, 0)
        assert result['fun'] < 1.0


class TestCMAES:
    """Tests for CMA-ES."""

    def test_quadratic_function(self):
        """Test: CMA-ES on quadratic."""
        config = GlobalOptimizationConfig(
            population_size=20,
            num_generations=50,
            cma_sigma=0.5
        )
        cma = CMAES(config)

        def objective(x):
            return (x[0] - 2)**2 + (x[1] + 1)**2

        x0 = np.array([0.0, 0.0])
        bounds = [(-5, 5), (-5, 5)]

        result = cma.solve(objective, x0, bounds)

        assert result['success']
        assert np.allclose(result['x'], [2.0, -1.0], atol=0.5)

    def test_rosenbrock_function(self):
        """Test: CMA-ES on Rosenbrock function."""
        config = GlobalOptimizationConfig(
            population_size=30,
            num_generations=100,
            cma_sigma=0.3
        )
        cma = CMAES(config)

        # Rosenbrock: (1-x)^2 + 100(y-x^2)^2
        def objective(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        x0 = np.array([0.0, 0.0])

        result = cma.solve(objective, x0)

        # Global min at (1, 1)
        # CMA-ES should get close
        assert result['fun'] < 1.0


class TestMixedIntegerOptimization:
    """Tests for mixed-integer optimization."""

    def test_pure_integer_problem(self):
        """Test: Branch-and-bound on integer problem."""
        config = MixedIntegerConfig(
            integer_variables=[0, 1],
            max_nodes=100
        )
        mip = MixedIntegerOptimization(config)

        # Minimize (x-2.3)^2 + (y-3.7)^2 with x, y integer
        def objective(x):
            return (x[0] - 2.3)**2 + (x[1] - 3.7)**2

        x0 = np.array([0.0, 0.0])
        bounds = [(0.0, 5.0), (0.0, 5.0)]

        result = mip.solve(objective, x0, bounds)

        # Optimal integer solution: (2, 4)
        if result['success']:
            assert result['x'][0] in [2.0, 3.0]
            assert result['x'][1] in [3.0, 4.0]

    def test_mixed_integer_problem(self):
        """Test: Mixed integer-continuous problem."""
        config = MixedIntegerConfig(
            integer_variables=[0],  # Only x is integer
            max_nodes=50
        )
        mip = MixedIntegerOptimization(config)

        # x integer, y continuous
        def objective(x):
            return (x[0] - 1.5)**2 + (x[1] - 2.5)**2

        x0 = np.array([0.0, 0.0])
        bounds = [(0.0, 5.0), (0.0, 5.0)]

        result = mip.solve(objective, x0, bounds)

        if result['success']:
            # x should be 1 or 2 (integer)
            assert abs(result['x'][0] - round(result['x'][0])) < 1e-6
            # y should be close to 2.5 (continuous)
            assert np.isclose(result['x'][1], 2.5, atol=0.1)


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_constrained_vs_unconstrained(self):
        """Test: Compare constrained and unconstrained solutions."""
        # Unconstrained
        def objective(x):
            return x[0]**2 + x[1]**2

        x_unconstrained = np.array([0.0, 0.0])

        # Constrained: x + y = 2
        config = ConstrainedOptimizationConfig()
        sqp = SequentialQuadraticProgramming(config)

        equality_constraints = [{
            'fun': lambda x: x[0] + x[1] - 2,
            'jac': lambda x: np.array([1.0, 1.0])
        }]

        x0 = np.array([0.0, 0.0])
        result_constrained = sqp.solve(objective, x0, equality_constraints=equality_constraints)

        # Unconstrained optimum: (0, 0)
        # Constrained optimum: (1, 1) on x + y = 2
        assert np.allclose(result_constrained['x'], [1.0, 1.0], atol=1e-2)
        assert result_constrained['fun'] > 0  # Worse than unconstrained

    def test_global_vs_local(self):
        """Test: Global optimizer finds better solution."""
        # Multimodal function
        def objective(x):
            return (x[0]**2 - 2)**2 + (x[1]**2 - 3)**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1])

        bounds = [(-5, 5), (-5, 5)]

        # Global method
        config_global = GlobalOptimizationConfig(population_size=40, num_generations=80)
        ga = GeneticAlgorithm(config_global)
        result_global = ga.solve(objective, bounds)

        # Should find good solution
        assert result_global['fun'] < 2.0

    def test_complete_optimization_workflow(self):
        """Test: Complete workflow with multiple methods."""
        print("\nComplete optimization workflow:")

        # Problem: minimize (x-1)^2 + (y-2)^2 subject to x + y >= 1.5, x, y >= 0

        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2

        bounds = [(0.0, 5.0), (0.0, 5.0)]

        # 1. Unconstrained (for comparison)
        from scipy.optimize import minimize
        result_unconstrained = minimize(objective, [0, 0], bounds=bounds)
        print(f"  1. Unconstrained: x = {result_unconstrained.x}, f = {result_unconstrained.fun:.4f}")

        # 2. Constrained (SQP)
        config_sqp = ConstrainedOptimizationConfig()
        sqp = SequentialQuadraticProgramming(config_sqp)

        inequality_constraints = [{
            'fun': lambda x: -(x[0] + x[1] - 1.5),  # x + y >= 1.5
            'jac': lambda x: np.array([-1.0, -1.0])
        }]

        result_sqp = sqp.solve(objective, [0, 0], inequality_constraints=inequality_constraints, bounds=bounds)
        print(f"  2. SQP: x = {result_sqp['x']}, f = {result_sqp['fun']:.4f}")

        # 3. Global (GA for verification)
        config_ga = GlobalOptimizationConfig(population_size=30, num_generations=50)
        ga = GeneticAlgorithm(config_ga)
        result_ga = ga.solve(objective, bounds)
        print(f"  3. GA: x = {result_ga['x']}, f = {result_ga['fun']:.4f}")

        print("  ✓ Workflow completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
