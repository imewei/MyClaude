"""Tests for Dask Distributed Computing.

Tests distributed execution, parallel computing, and cluster management.

Author: Nonequilibrium Physics Agents
Week: 31-32 of Phase 4
"""

import pytest
import numpy as np
import time
import tempfile
from pathlib import Path

# Try to import Dask
try:
    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Try to import dask-jobqueue
try:
    from dask_jobqueue import SLURMCluster
    DASK_JOBQUEUE_AVAILABLE = True
except ImportError:
    DASK_JOBQUEUE_AVAILABLE = False

# Import module (will have fallbacks if Dask unavailable)
from hpc.distributed import (
    DaskCluster,
    ParallelExecutor,
    distribute_computation,
    create_local_cluster,
    fault_tolerant_map,
    distributed_optimization,
    pipeline,
    distributed_cross_validation,
    scatter_gather_reduction,
    checkpoint_computation,
    DASK_AVAILABLE as MODULE_DASK_AVAILABLE
)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Tests for configuration and availability."""

    def test_dask_availability(self):
        """Test: Dask availability detection."""
        print("\nDask availability:")
        print(f"  DASK_AVAILABLE: {DASK_AVAILABLE}")
        print(f"  MODULE_DASK_AVAILABLE: {MODULE_DASK_AVAILABLE}")

        assert DASK_AVAILABLE == MODULE_DASK_AVAILABLE

    def test_dask_jobqueue_availability(self):
        """Test: Dask-jobqueue availability."""
        print("\nDask-jobqueue availability:")
        print(f"  DASK_JOBQUEUE_AVAILABLE: {DASK_JOBQUEUE_AVAILABLE}")

        # Just informational
        assert isinstance(DASK_JOBQUEUE_AVAILABLE, bool)


# ============================================================================
# Cluster Tests (require Dask)
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestDaskCluster:
    """Tests for DaskCluster."""

    def test_local_cluster_creation(self):
        """Test: Create local Dask cluster."""
        print("\nLocal cluster creation:")

        cluster = create_local_cluster(n_workers=2, threads_per_worker=1)

        assert cluster is not None
        assert cluster.client is not None

        # Check workers
        info = cluster.client.scheduler_info()
        print(f"  Workers: {len(info['workers'])}")

        cluster.close()
        print("  ✓ Cluster created and closed")

    def test_cluster_submit_gather(self):
        """Test: Submit and gather tasks."""
        print("\nSubmit and gather:")

        cluster = create_local_cluster(n_workers=2)

        # Submit simple task
        def add(x, y):
            return x + y

        future = cluster.submit(add, 2, 3)
        result = future.result()

        assert result == 5
        print(f"  ✓ Result: {result}")

        cluster.close()

    def test_cluster_map(self):
        """Test: Map function over data."""
        print("\nMap operation:")

        cluster = create_local_cluster(n_workers=2)

        # Map function
        def square(x):
            return x ** 2

        data = list(range(10))
        futures = [cluster.submit(square, x) for x in data]
        results = cluster.gather(futures)

        expected = [x**2 for x in data]
        assert results == expected

        print(f"  ✓ Mapped {len(data)} elements")

        cluster.close()

    def test_cluster_scatter(self):
        """Test: Scatter data to workers."""
        print("\nScatter data:")

        cluster = create_local_cluster(n_workers=2)

        # Scatter large array
        data = list(range(100))
        scattered = cluster.scatter(data)

        assert len(scattered) == len(data)
        print(f"  ✓ Scattered {len(data)} items")

        cluster.close()


# ============================================================================
# Parallel Execution Tests
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestParallelExecution:
    """Tests for parallel execution functions."""

    def test_distribute_computation(self):
        """Test: Distribute computation across workers."""
        print("\nDistribute computation:")

        def expensive_function(x):
            """Simulate expensive computation."""
            time.sleep(0.1)
            return x ** 2

        inputs = list(range(10))

        # Serial execution (baseline)
        start_serial = time.time()
        results_serial = [expensive_function(x) for x in inputs]
        time_serial = time.time() - start_serial

        # Parallel execution
        cluster = create_local_cluster(n_workers=4)
        start_parallel = time.time()
        results_parallel = distribute_computation(
            expensive_function,
            inputs,
            cluster=cluster
        )
        time_parallel = time.time() - start_parallel
        cluster.close()

        # Verify correctness
        assert results_parallel == results_serial

        # Check speedup (should be faster, but not strict for CI)
        print(f"  Serial time: {time_serial:.2f}s")
        print(f"  Parallel time: {time_parallel:.2f}s")
        print(f"  Speedup: {time_serial / time_parallel:.2f}x")

    def test_fault_tolerant_map(self):
        """Test: Fault-tolerant mapping with retries."""
        print("\nFault-tolerant map:")

        # Function that sometimes fails
        call_count = {"count": 0}

        def flaky_function(x):
            call_count["count"] += 1
            if call_count["count"] <= 3 and x == 5:
                raise ValueError("Simulated failure")
            return x * 2

        inputs = list(range(10))

        cluster = create_local_cluster(n_workers=2)
        results, failures = fault_tolerant_map(
            flaky_function,
            inputs,
            cluster=cluster,
            max_retries=5
        )
        cluster.close()

        # Should eventually succeed with retries
        assert len(failures) == 0
        assert results == [x * 2 for x in inputs]

        print(f"  ✓ Handled {call_count['count'] - len(inputs)} failures")
        print(f"  ✓ All {len(inputs)} tasks succeeded")

    def test_distributed_optimization(self):
        """Test: Distributed hyperparameter optimization."""
        print("\nDistributed optimization:")

        # Simple quadratic objective
        def objective(params):
            x = params['x']
            y = params['y']
            return (x - 2)**2 + (y + 1)**2

        parameter_ranges = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }

        cluster = create_local_cluster(n_workers=2)
        best_params, best_value = distributed_optimization(
            objective,
            parameter_ranges,
            n_samples=50,
            cluster=cluster,
            method="random"
        )
        cluster.close()

        # Should find approximately (2, -1)
        assert abs(best_params['x'] - 2.0) < 1.0
        assert abs(best_params['y'] + 1.0) < 1.0
        assert best_value < 1.0

        print(f"  Best params: x={best_params['x']:.2f}, y={best_params['y']:.2f}")
        print(f"  Best value: {best_value:.4f}")

    def test_scatter_gather_reduction(self):
        """Test: MapReduce with scatter/gather."""
        print("\nScatter-gather reduction:")

        data = list(range(100))

        def map_fn(x):
            return x ** 2

        def reduce_fn(results):
            return sum(results)

        cluster = create_local_cluster(n_workers=2)
        result = scatter_gather_reduction(
            data,
            map_fn,
            reduce_fn,
            cluster=cluster
        )
        cluster.close()

        expected = sum(x**2 for x in data)
        assert result == expected

        print(f"  ✓ MapReduce result: {result}")


# ============================================================================
# Pipeline Tests
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestPipeline:
    """Tests for data processing pipelines."""

    def test_simple_pipeline(self):
        """Test: Simple processing pipeline."""
        print("\nSimple pipeline:")

        # Define pipeline stages
        def stage1(data):
            return [x * 2 for x in data]

        def stage2(data):
            return [x + 10 for x in data]

        def stage3(data):
            return sum(data)

        stages = [stage1, stage2, stage3]
        initial_data = list(range(10))

        cluster = create_local_cluster(n_workers=2)
        result = pipeline(stages, initial_data, cluster=cluster)
        cluster.close()

        # Verify: sum((x*2 + 10) for x in range(10))
        expected = sum(x*2 + 10 for x in range(10))
        assert result == expected

        print(f"  ✓ Pipeline result: {result}")

    def test_pipeline_with_persistence(self):
        """Test: Pipeline with intermediate persistence."""
        print("\nPipeline with persistence:")

        def stage1(data):
            return [x ** 2 for x in data]

        def stage2(data):
            return [np.sqrt(x) for x in data]

        stages = [stage1, stage2]
        initial_data = list(range(10))

        cluster = create_local_cluster(n_workers=2)
        result = pipeline(
            stages,
            initial_data,
            cluster=cluster,
            persist_intermediate=True
        )
        cluster.close()

        # Should recover original (within tolerance)
        expected = [float(x) for x in initial_data]
        assert np.allclose(result, expected)

        print("  ✓ Pipeline with persistence completed")


# ============================================================================
# Cross-Validation Tests
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestCrossValidation:
    """Tests for distributed cross-validation."""

    def test_distributed_cross_validation(self):
        """Test: K-fold cross-validation."""
        print("\nDistributed cross-validation:")

        # Simple linear regression model
        def model_fn():
            return {'w': 0.0, 'b': 0.0}

        def train_fn(model, train_data):
            # Simple training (just average)
            X = np.array([x for x, y in train_data])
            Y = np.array([y for x, y in train_data])
            model['w'] = np.sum(X * Y) / np.sum(X ** 2) if np.sum(X ** 2) > 0 else 0
            model['b'] = np.mean(Y - model['w'] * X)
            return model

        def evaluate_fn(model, test_data):
            # MSE
            errors = []
            for x, y in test_data:
                pred = model['w'] * x + model['b']
                errors.append((y - pred) ** 2)
            return np.mean(errors)

        # Generate synthetic data: y = 2x + 1 + noise
        np.random.seed(42)
        data = [(x, 2*x + 1 + np.random.randn()*0.1) for x in np.linspace(0, 10, 100)]

        cluster = create_local_cluster(n_workers=2)
        results = distributed_cross_validation(
            model_fn,
            train_fn,
            evaluate_fn,
            data,
            n_folds=5,
            cluster=cluster
        )
        cluster.close()

        # Should have low mean error for linear data
        assert results['mean'] < 1.0
        assert results['n_folds'] == 5
        assert len(results['scores']) == 5

        print(f"  Mean CV score: {results['mean']:.4f}")
        print(f"  Std CV score: {results['std']:.4f}")


# ============================================================================
# Checkpoint Tests
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestCheckpointing:
    """Tests for checkpointed computations."""

    def test_checkpoint_computation(self):
        """Test: Computation with checkpointing."""
        print("\nCheckpoint computation:")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pkl"

            # Define expensive computation
            def computation():
                time.sleep(0.5)
                return {"result": 42, "data": list(range(100))}

            # First run (no checkpoint)
            cluster = create_local_cluster(n_workers=1)
            start1 = time.time()
            result1 = checkpoint_computation(
                computation,
                checkpoint_path,
                cluster=cluster,
                force_recompute=False
            )
            time1 = time.time() - start1

            # Second run (with checkpoint)
            start2 = time.time()
            result2 = checkpoint_computation(
                computation,
                checkpoint_path,
                cluster=cluster,
                force_recompute=False
            )
            time2 = time.time() - start2

            cluster.close()

            # Results should match
            assert result1 == result2
            assert result1["result"] == 42

            # Second run should be much faster (loaded from checkpoint)
            assert time2 < time1 * 0.5

            print(f"  First run: {time1:.2f}s")
            print(f"  Second run (checkpoint): {time2:.2f}s")
            print(f"  Speedup: {time1 / time2:.1f}x")


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_distributed_workflow(self):
        """Test: Complete distributed computing workflow."""
        print("\nComplete distributed workflow:")

        # 1. Create cluster
        print("  1. Creating cluster...")
        cluster = create_local_cluster(n_workers=4, threads_per_worker=1)

        # 2. Scatter data
        print("  2. Scattering data...")
        data = list(range(100))
        scattered = cluster.scatter(data)

        # 3. Parallel computation
        print("  3. Running parallel computation...")
        def process(x):
            return x ** 2 + 2 * x + 1

        futures = [cluster.submit(process, x) for x in scattered]
        results = cluster.gather(futures)

        # 4. Reduction
        print("  4. Reducing results...")
        total = sum(results)

        # 5. Verification
        expected = sum(process(x) for x in data)
        assert total == expected

        print(f"  ✓ Processed {len(data)} items")
        print(f"  ✓ Result: {total}")

        # 6. Cleanup
        cluster.close()
        print("  ✓ Workflow complete")


# ============================================================================
# Mock Tests (without Dask)
# ============================================================================

class TestWithoutDask:
    """Tests that work without Dask."""

    def test_fallback_behavior(self):
        """Test: Graceful fallback when Dask unavailable."""
        print("\nFallback behavior:")

        if not DASK_AVAILABLE:
            print("  Dask not available - testing fallbacks")

            # Module should still import
            assert MODULE_DASK_AVAILABLE == False

            # Functions should exist but may raise NotImplementedError
            # or work with simple fallback logic

            print("  ✓ Module imports successfully without Dask")
        else:
            print("  Dask is available - skipping fallback test")
            pytest.skip("Dask is available")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestPerformance:
    """Performance and scaling tests."""

    def test_scaling_with_workers(self):
        """Test: Performance scaling with number of workers."""
        print("\nScaling with workers:")

        def expensive_task(x):
            """CPU-bound task."""
            result = 0
            for i in range(10000):
                result += np.sum(np.random.rand(100))
            return result

        inputs = list(range(20))

        # Test with different worker counts
        for n_workers in [1, 2, 4]:
            cluster = create_local_cluster(n_workers=n_workers)

            start = time.time()
            futures = [cluster.submit(expensive_task, x) for x in inputs]
            results = cluster.gather(futures)
            elapsed = time.time() - start

            cluster.close()

            print(f"  {n_workers} workers: {elapsed:.2f}s")

        print("  ✓ Scaling test complete")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
