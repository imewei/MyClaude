"""Tests for HPC Integration.

This test suite validates HPC capabilities including SLURM, Dask, and parallel optimization.

Test Categories:
1. SLURM configuration and job management
2. Dask cluster creation and execution
3. Parallel optimization (grid search, random search, Bayesian)
4. Integration tests

Author: Nonequilibrium Physics Agents
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import time

# Import HPC modules
from hpc.slurm import SLURMConfig, SLURMJob, get_slurm_info
from hpc.parallel import (
    ParameterSpec,
    GridSearch,
    RandomSearch,
    ParallelOptimizer,
    create_parameter_grid,
    analyze_sweep_results
)

# Check for optional dependencies
try:
    from hpc.distributed import (
        create_local_cluster,
        ParallelExecutor,
        distribute_computation
    )
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


# =============================================================================
# SLURM Tests
# =============================================================================

class TestSLURMConfig:
    """Test SLURM configuration."""

    def test_1_creation(self):
        """Test 1: Create SLURM config."""
        print("\n  Test 1: SLURM config creation")

        config = SLURMConfig(
            job_name="test_job",
            partition="general",
            nodes=1,
            cpus_per_task=4,
            mem="8GB",
            time="01:00:00"
        )

        assert config.job_name == "test_job"
        assert config.partition == "general"
        assert config.nodes == 1
        assert config.cpus_per_task == 4
        print(f"    Created config: {config.job_name}, {config.cpus_per_task} CPUs")

    def test_2_sbatch_header(self):
        """Test 2: Generate sbatch header."""
        print("\n  Test 2: SBATCH header generation")

        config = SLURMConfig(
            job_name="test",
            gres="gpu:1",
            setup_commands=["module load python", "source venv/bin/activate"]
        )

        header = config.to_sbatch_header()

        assert "#!/bin/bash" in header
        assert "#SBATCH --job-name=test" in header
        assert "#SBATCH --gres=gpu:1" in header
        assert "module load python" in header
        print(f"    Generated header: {len(header.split(chr(10)))} lines")

    def test_3_gpu_config(self):
        """Test 3: GPU configuration."""
        print("\n  Test 3: GPU configuration")

        config = SLURMConfig(
            job_name="gpu_job",
            gres="gpu:2",
            partition="gpu"
        )

        header = config.to_sbatch_header()

        assert "--gres=gpu:2" in header
        assert "--partition=gpu" in header
        print(f"    GPU config: 2 GPUs on gpu partition")

    def test_4_slurm_availability(self):
        """Test 4: Check SLURM availability."""
        print("\n  Test 4: SLURM availability check")

        info = get_slurm_info()

        print(f"    SLURM available: {info['available']}")
        if info['available']:
            print(f"    Version: {info['version']}")
            print(f"    Partitions: {info['partitions']}")
            print(f"    Nodes: {info['nodes']}")


# =============================================================================
# Dask Tests
# =============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestDask:
    """Test Dask distributed computing."""

    def test_5_local_cluster(self):
        """Test 5: Create local Dask cluster."""
        print("\n  Test 5: Local Dask cluster")

        cluster = create_local_cluster(n_workers=2, threads_per_worker=1)

        try:
            assert cluster is not None
            assert cluster.cluster_type == "local"

            # Get info
            from hpc.distributed import get_cluster_info
            info = get_cluster_info(cluster)

            print(f"    Workers: {info['n_workers']}")
            print(f"    Total cores: {info['total_cores']}")
            print(f"    Dashboard: {info['dashboard']}")

        finally:
            cluster.close()

    def test_6_parallel_map(self):
        """Test 6: Parallel map operation."""
        print("\n  Test 6: Parallel map")

        def square(x):
            return x ** 2

        inputs = list(range(10))

        with create_local_cluster(n_workers=2) as cluster:
            executor = ParallelExecutor(cluster)
            results = executor.map(square, inputs, show_progress=False)

        expected = [x ** 2 for x in inputs]

        assert results == expected
        print(f"    Mapped {len(inputs)} items successfully")

    def test_7_map_reduce(self):
        """Test 7: Map-reduce pattern."""
        print("\n  Test 7: Map-reduce")

        def map_func(x):
            return x ** 2

        def reduce_func(a, b):
            return a + b

        inputs = list(range(10))

        with create_local_cluster(n_workers=2) as cluster:
            executor = ParallelExecutor(cluster)
            result = executor.map_reduce(
                map_func,
                reduce_func,
                inputs,
                show_progress=False
            )

        expected = sum(x ** 2 for x in inputs)

        assert result == expected
        print(f"    Sum of squares: {result}")

    def test_8_distribute_computation(self):
        """Test 8: Distribute computation."""
        print("\n  Test 8: Distribute computation")

        def factorial(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

        inputs = [5, 6, 7, 8, 9, 10]

        results = distribute_computation(
            factorial,
            inputs,
            n_workers=2,
            cluster_type="local",
            show_progress=False
        )

        expected = [factorial(n) for n in inputs]

        assert results == expected
        print(f"    Computed {len(inputs)} factorials")


# =============================================================================
# Parameter Specification Tests
# =============================================================================

class TestParameterSpec:
    """Test parameter specifications."""

    def test_9_continuous_param(self):
        """Test 9: Continuous parameter."""
        print("\n  Test 9: Continuous parameter")

        param = ParameterSpec(
            name="learning_rate",
            param_type="continuous",
            lower=1e-4,
            upper=1e-2,
            log_scale=True
        )

        # Sample
        value = param.sample()

        assert 1e-4 <= value <= 1e-2
        print(f"    Sampled: {value:.6f}")

        # Grid
        grid = param.grid_values(n_points=5)

        assert len(grid) == 5
        assert grid[0] == pytest.approx(1e-4)
        assert grid[-1] == pytest.approx(1e-2)
        print(f"    Grid: {grid}")

    def test_10_integer_param(self):
        """Test 10: Integer parameter."""
        print("\n  Test 10: Integer parameter")

        param = ParameterSpec(
            name="batch_size",
            param_type="integer",
            lower=16,
            upper=128,
            log_scale=True
        )

        value = param.sample()

        assert 16 <= value <= 128
        assert isinstance(value, (int, np.integer))
        print(f"    Sampled: {value}")

    def test_11_categorical_param(self):
        """Test 11: Categorical parameter."""
        print("\n  Test 11: Categorical parameter")

        param = ParameterSpec(
            name="optimizer",
            param_type="categorical",
            choices=["adam", "sgd", "rmsprop"]
        )

        value = param.sample()

        assert value in ["adam", "sgd", "rmsprop"]
        print(f"    Sampled: {value}")

        grid = param.grid_values()
        assert grid == ["adam", "sgd", "rmsprop"]


# =============================================================================
# Grid Search Tests
# =============================================================================

class TestGridSearch:
    """Test grid search."""

    def test_12_simple_grid_search(self):
        """Test 12: Simple grid search."""
        print("\n  Test 12: Grid search")

        # Objective: minimize (x - 2)^2 + (y - 3)^2
        def objective(params):
            x = params["x"]
            y = params["y"]
            return (x - 2)**2 + (y - 3)**2

        parameters = [
            ParameterSpec("x", "continuous", lower=0, upper=5),
            ParameterSpec("y", "continuous", lower=0, upper=5)
        ]

        search = GridSearch(
            parameters,
            objective,
            n_grid_points=5,
            n_jobs=2
        )

        best_params, best_value = search.run(use_dask=False)

        print(f"    Best params: {best_params}")
        print(f"    Best value: {best_value:.4f}")

        # Should find minimum near (2, 3)
        assert abs(best_params["x"] - 2) < 1.5
        assert abs(best_params["y"] - 3) < 1.5
        assert best_value < 5.0


# =============================================================================
# Random Search Tests
# =============================================================================

class TestRandomSearch:
    """Test random search."""

    def test_13_simple_random_search(self):
        """Test 13: Random search."""
        print("\n  Test 13: Random search")

        # Objective: minimize sum of squares
        def objective(params):
            return sum(v**2 for v in params.values())

        parameters = [
            ParameterSpec("x", "continuous", lower=-10, upper=10),
            ParameterSpec("y", "continuous", lower=-10, upper=10),
            ParameterSpec("z", "continuous", lower=-10, upper=10)
        ]

        search = RandomSearch(
            parameters,
            objective,
            n_samples=50,
            seed=42,
            n_jobs=2
        )

        best_params, best_value = search.run(use_dask=False)

        print(f"    Best params: {best_params}")
        print(f"    Best value: {best_value:.4f}")

        # Should find values close to 0
        assert best_value < 10.0


# =============================================================================
# Parallel Optimizer Tests
# =============================================================================

class TestParallelOptimizer:
    """Test parallel optimizer."""

    def test_14_optimizer_interface(self):
        """Test 14: Parallel optimizer interface."""
        print("\n  Test 14: Parallel optimizer")

        # Simple quadratic objective
        def objective(params):
            return (params["a"] - 1)**2 + (params["b"] + 2)**2

        parameters = [
            ParameterSpec("a", "continuous", lower=-5, upper=5),
            ParameterSpec("b", "continuous", lower=-5, upper=5)
        ]

        optimizer = ParallelOptimizer(
            objective,
            parameters,
            n_jobs=2
        )

        # Try grid search
        best_params, best_value = optimizer.grid_search(
            n_grid_points=5,
            use_dask=False
        )

        print(f"    Grid search best: {best_params}, value={best_value:.4f}")

        assert abs(best_params["a"] - 1) < 1.5
        assert abs(best_params["b"] + 2) < 1.5


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_15_create_parameter_grid(self):
        """Test 15: Create parameter grid."""
        print("\n  Test 15: Create parameter grid")

        params = create_parameter_grid(
            learning_rate=(1e-4, 1e-2),
            batch_size=[32, 64, 128],
            n_layers=(2, 5)
        )

        assert len(params) == 3

        # Check types
        assert params[0].name == "learning_rate"
        assert params[0].param_type == "continuous"

        assert params[1].name == "batch_size"
        assert params[1].param_type == "categorical"

        assert params[2].name == "n_layers"
        assert params[2].param_type == "integer"

        print(f"    Created {len(params)} parameter specs")

    def test_16_analyze_results(self):
        """Test 16: Analyze sweep results."""
        print("\n  Test 16: Analyze results")

        # Create fake results
        results = [
            {"params": {"x": i}, "value": i**2}
            for i in range(-5, 6)
        ]

        analysis = analyze_sweep_results(results, top_k=3)

        print(f"    Best value: {analysis['best_value']}")
        print(f"    Mean value: {analysis['mean_value']:.2f}")
        print(f"    Std value: {analysis['std_value']:.2f}")

        assert analysis["best_value"] == 0
        assert analysis["n_evaluations"] == 11
        assert len(analysis["top_k_results"]) == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_17_optimal_control_sweep(self):
        """Test 17: Optimal control parameter sweep."""
        print("\n  Test 17: Optimal control parameter sweep")

        # Simulate optimal control objective
        # Minimize final state error with cost on control
        def control_objective(params):
            # Simulate simple dynamics: x_dot = -k*x + u
            k = params["damping"]
            T = params["duration"]
            dt = 0.01

            # Simple control law: u = -gain * x
            gain = params["gain"]

            # Simulate
            x = 1.0  # Initial state
            total_cost = 0.0
            t = 0.0

            while t < T:
                u = -gain * x
                x_next = x + (-k * x + u) * dt

                # Cost: state error + control effort
                total_cost += (x**2 + 0.1*u**2) * dt

                x = x_next
                t += dt

            # Add terminal cost
            total_cost += 10 * x**2

            return total_cost

        # Parameter specs
        parameters = [
            ParameterSpec("damping", "continuous", lower=0.1, upper=2.0),
            ParameterSpec("gain", "continuous", lower=0.0, upper=5.0),
            ParameterSpec("duration", "continuous", lower=1.0, upper=5.0)
        ]

        # Run random search
        search = RandomSearch(
            parameters,
            control_objective,
            n_samples=20,
            seed=42,
            n_jobs=2
        )

        best_params, best_value = search.run(use_dask=False)

        print(f"    Best parameters:")
        for k, v in best_params.items():
            print(f"      {k}: {v:.3f}")
        print(f"    Best cost: {best_value:.4f}")

        # Should find reasonably good solution
        assert best_value < 5.0


def run_all_tests():
    """Run all HPC tests."""
    print("\n" + "="*70)
    print("HPC Integration Tests")
    print("="*70)

    test_classes = [
        TestSLURMConfig,
        TestParameterSpec,
        TestGridSearch,
        TestRandomSearch,
        TestParallelOptimizer,
        TestUtilities,
        TestIntegration
    ]

    # Add Dask tests if available
    if DASK_AVAILABLE:
        test_classes.insert(1, TestDask)

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)

        test_obj = test_class()
        methods = [m for m in dir(test_obj) if m.startswith('test_')]

        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_obj, method_name)
                method()
                passed_tests += 1
                print(f"  ✓ {method_name}")
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
            except Exception as e:
                print(f"  ✗ {method_name}: Unexpected error: {e}")

    print("\n" + "="*70)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\nNote: Dask tests skipped (install with: pip install dask distributed)")

    return passed_tests == total_tests


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
