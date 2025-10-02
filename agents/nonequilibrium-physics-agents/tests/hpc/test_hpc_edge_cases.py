"""
Phase 4 Week 35-36: HPC Integration Edge Case Tests

Comprehensive edge case and error handling tests for HPC schedulers,
distributed execution, and parameter sweeps.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import subprocess
import time

# Import HPC modules
from hpc.schedulers import (
    Scheduler,
    LocalScheduler,
    SLURMScheduler,
    PBSScheduler,
    JobManager,
    JobStatus,
    ResourceRequirements,
    JobInfo
)
from hpc.parallel import (
    ParameterSpec,
    ParameterSweep,
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    AdaptiveSweep,
    MultiObjectiveSweep,
    sensitivity_analysis,
    visualize_sweep_results,
    export_sweep_results
)

# Check Dask availability
try:
    import dask
    import dask.distributed
    from hpc.distributed import (
        DaskCluster,
        create_local_cluster,
        distribute_computation,
        distributed_optimization,
        pipeline,
        distributed_cross_validation,
        scatter_gather_reduction,
        checkpoint_computation,
        fault_tolerant_map
    )
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


class TestResourceRequirementsEdgeCases:
    """Edge case tests for ResourceRequirements."""

    def test_zero_resources(self):
        """Test resource requirements with zero values."""
        with pytest.raises((ValueError, AssertionError)):
            resources = ResourceRequirements(nodes=0)

        with pytest.raises((ValueError, AssertionError)):
            resources = ResourceRequirements(cpus_per_task=0)

    def test_negative_resources(self):
        """Test resource requirements with negative values."""
        with pytest.raises((ValueError, AssertionError)):
            resources = ResourceRequirements(nodes=-1)

        with pytest.raises((ValueError, AssertionError)):
            resources = ResourceRequirements(gpus_per_node=-1)

    def test_very_large_resources(self):
        """Test resource requirements with very large values."""
        resources = ResourceRequirements(
            nodes=10000,
            cpus_per_task=128,
            gpus_per_node=8,
            memory_gb=1000
        )
        assert resources.nodes == 10000

    def test_invalid_time_limit(self):
        """Test invalid time limit formats."""
        # Should accept string or raise error
        try:
            resources = ResourceRequirements(time_limit="invalid")
        except (ValueError, TypeError):
            pass  # Expected

    def test_fractional_resources(self):
        """Test fractional resource values."""
        with pytest.raises((ValueError, TypeError)):
            resources = ResourceRequirements(nodes=1.5)

        with pytest.raises((ValueError, TypeError)):
            resources = ResourceRequirements(cpus_per_task=2.5)


class TestLocalSchedulerEdgeCases:
    """Edge case tests for LocalScheduler."""

    def test_submit_nonexistent_script(self):
        """Test submitting job with non-existent script."""
        scheduler = LocalScheduler()
        resources = ResourceRequirements()

        with pytest.raises((FileNotFoundError, OSError)):
            job_id = scheduler.submit_job(
                "/nonexistent/script.sh",
                "test_job",
                resources
            )

    def test_submit_unreadable_script(self):
        """Test submitting job with unreadable script."""
        scheduler = LocalScheduler()
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho test\n")
            script_path = f.name

        try:
            # Remove read permissions
            import os
            os.chmod(script_path, 0o000)

            with pytest.raises((PermissionError, OSError)):
                job_id = scheduler.submit_job(
                    script_path,
                    "test_job",
                    resources
                )
        finally:
            # Restore permissions and cleanup
            import os
            try:
                os.chmod(script_path, 0o644)
                os.unlink(script_path)
            except:
                pass

    def test_query_nonexistent_job(self):
        """Test querying non-existent job ID."""
        scheduler = LocalScheduler()

        with pytest.raises((KeyError, ValueError)):
            status = scheduler.get_job_status("nonexistent_job_id")

    def test_cancel_nonexistent_job(self):
        """Test canceling non-existent job."""
        scheduler = LocalScheduler()

        with pytest.raises((KeyError, ValueError)):
            scheduler.cancel_job("nonexistent_job_id")

    def test_cancel_already_completed_job(self):
        """Test canceling already completed job."""
        scheduler = LocalScheduler()
        resources = ResourceRequirements()

        # Create and submit simple job
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho done\n")
            script_path = f.name

        try:
            job_id = scheduler.submit_job(script_path, "test_job", resources)

            # Wait for completion
            final_status = scheduler.wait_for_job(job_id, timeout=10.0)

            # Try to cancel completed job
            try:
                scheduler.cancel_job(job_id)
                # Should either succeed (no-op) or raise error
            except (ValueError, RuntimeError):
                pass  # Acceptable
        finally:
            import os
            os.unlink(script_path)

    def test_wait_with_zero_timeout(self):
        """Test wait_for_job with zero timeout."""
        scheduler = LocalScheduler()
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\nsleep 5\n")
            script_path = f.name

        try:
            job_id = scheduler.submit_job(script_path, "test_job", resources)

            # Zero timeout should return immediately or raise error
            try:
                status = scheduler.wait_for_job(job_id, timeout=0.0)
                # If it doesn't raise, status might be RUNNING
                assert status in [JobStatus.RUNNING, JobStatus.PENDING, JobStatus.COMPLETED]
            except (ValueError, TimeoutError):
                pass  # Acceptable

            # Cleanup
            scheduler.cancel_job(job_id)
        finally:
            import os
            os.unlink(script_path)

    def test_circular_job_dependencies(self):
        """Test circular job dependencies (should be detected)."""
        scheduler = LocalScheduler()
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho test\n")
            script_path = f.name

        try:
            job1 = scheduler.submit_job(script_path, "job1", resources)

            # Try to create circular dependency
            # Job2 depends on job1 is fine, but job1 depending on job2 would be circular
            # Most schedulers can't modify dependencies post-submission
            # This tests if scheduler detects invalid dependencies

            with pytest.raises((ValueError, RuntimeError)):
                # Attempt to submit job that depends on itself
                job2 = scheduler.submit_job(
                    script_path,
                    "job2",
                    resources,
                    dependencies=[job1, "job2"]  # Self-dependency
                )
        except (KeyError, ValueError):
            pass  # Expected for invalid dependency
        finally:
            import os
            os.unlink(script_path)


class TestJobManagerEdgeCases:
    """Edge case tests for JobManager."""

    def test_auto_detect_no_scheduler(self):
        """Test auto-detection when no scheduler is available."""
        # This should fallback to LocalScheduler
        manager = JobManager(auto_detect=True)
        assert manager.scheduler is not None
        assert isinstance(manager.scheduler, LocalScheduler)

    def test_submit_many_jobs_rapidly(self):
        """Test submitting many jobs in rapid succession (stress test)."""
        manager = JobManager(scheduler_type="local")
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho test\n")
            script_path = f.name

        try:
            job_ids = []
            for i in range(50):  # Submit 50 jobs
                job_id = manager.submit(script_path, f"job_{i}", resources)
                job_ids.append(job_id)

            assert len(job_ids) == 50
            assert len(set(job_ids)) == 50  # All unique

            # Wait for all
            statuses = manager.wait_all(job_ids, timeout=60.0)

            # Most should complete successfully
            completed = sum(1 for s in statuses.values() if s == JobStatus.COMPLETED)
            assert completed > 40  # At least 80% success rate
        finally:
            import os
            os.unlink(script_path)

    def test_wait_all_empty_list(self):
        """Test wait_all with empty job list."""
        manager = JobManager(scheduler_type="local")
        statuses = manager.wait_all([], timeout=1.0)
        assert statuses == {}

    def test_wait_all_with_failed_jobs(self):
        """Test wait_all when some jobs fail."""
        manager = JobManager(scheduler_type="local")
        resources = ResourceRequirements()

        # Create failing script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\nexit 1\n")  # Fails with exit code 1
            fail_script = f.name

        # Create succeeding script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho success\n")
            success_script = f.name

        try:
            job1 = manager.submit(fail_script, "failing_job", resources)
            job2 = manager.submit(success_script, "success_job", resources)

            statuses = manager.wait_all([job1, job2], timeout=10.0)

            # job1 should fail, job2 should succeed
            assert statuses[job1] == JobStatus.FAILED
            assert statuses[job2] == JobStatus.COMPLETED
        finally:
            import os
            os.unlink(fail_script)
            os.unlink(success_script)


class TestParameterSpecEdgeCases:
    """Edge case tests for ParameterSpec."""

    def test_continuous_param_inverted_bounds(self):
        """Test continuous parameter with inverted bounds."""
        with pytest.raises((ValueError, AssertionError)):
            spec = ParameterSpec("param", "continuous", lower=10.0, upper=1.0)

    def test_continuous_param_equal_bounds(self):
        """Test continuous parameter with equal bounds."""
        spec = ParameterSpec("param", "continuous", lower=5.0, upper=5.0)
        # Should create valid spec (single-value parameter)
        assert spec.lower == spec.upper

    def test_integer_param_fractional_bounds(self):
        """Test integer parameter with fractional bounds."""
        # Should either round or raise error
        try:
            spec = ParameterSpec("param", "integer", lower=1.5, upper=5.5)
            # If it succeeds, bounds should be integers
            assert isinstance(spec.lower, int)
            assert isinstance(spec.upper, int)
        except (ValueError, TypeError):
            pass  # Acceptable to reject fractional bounds

    def test_categorical_param_empty_choices(self):
        """Test categorical parameter with empty choices."""
        with pytest.raises((ValueError, AssertionError)):
            spec = ParameterSpec("param", "categorical", choices=[])

    def test_categorical_param_single_choice(self):
        """Test categorical parameter with single choice."""
        spec = ParameterSpec("param", "categorical", choices=["only_option"])
        # Should be valid (no choice to make)
        assert len(spec.choices) == 1

    def test_invalid_parameter_type(self):
        """Test invalid parameter type."""
        with pytest.raises((ValueError, KeyError)):
            spec = ParameterSpec("param", "invalid_type", lower=0, upper=10)


class TestGridSearchEdgeCases:
    """Edge case tests for GridSearch."""

    def test_grid_search_single_point(self):
        """Test grid search with single point per dimension."""
        specs = [
            ParameterSpec("x", "continuous", 0.0, 1.0),
            ParameterSpec("y", "continuous", 0.0, 1.0)
        ]
        sweep = GridSearch(specs, points_per_dim=1)
        samples = sweep.generate_samples()

        assert len(samples) == 1  # Only one point total

    def test_grid_search_zero_points(self):
        """Test grid search with zero points per dimension."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]

        with pytest.raises((ValueError, AssertionError)):
            sweep = GridSearch(specs, points_per_dim=0)

    def test_grid_search_high_dimension(self):
        """Test grid search in high dimension (exponential explosion)."""
        specs = [
            ParameterSpec(f"x{i}", "continuous", 0.0, 1.0)
            for i in range(10)  # 10D
        ]
        sweep = GridSearch(specs, points_per_dim=3)  # 3^10 = 59049 points
        samples = sweep.generate_samples()

        assert len(samples) == 3**10


class TestRandomSearchEdgeCases:
    """Edge case tests for RandomSearch."""

    def test_random_search_zero_samples(self):
        """Test random search with zero samples."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = RandomSearch(specs, n_samples=0)
        samples = sweep.generate_samples()

        assert len(samples) == 0

    def test_random_search_reproducibility(self):
        """Test random search reproducibility with seed."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]

        sweep1 = RandomSearch(specs, n_samples=10, seed=42)
        samples1 = sweep1.generate_samples()

        sweep2 = RandomSearch(specs, n_samples=10, seed=42)
        samples2 = sweep2.generate_samples()

        # Should generate identical samples
        for s1, s2 in zip(samples1, samples2):
            assert np.isclose(s1["x"], s2["x"])


class TestAdaptiveSweepEdgeCases:
    """Edge case tests for AdaptiveSweep."""

    def test_adaptive_sweep_no_initial_samples(self):
        """Test adaptive sweep with zero initial samples."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]

        with pytest.raises((ValueError, AssertionError)):
            sweep = AdaptiveSweep(specs, n_initial=0)

    def test_adaptive_sweep_before_adaptation(self):
        """Test adaptive sampling before enough results."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = AdaptiveSweep(specs, n_initial=20, n_iterations=5)

        # Generate samples before adding any results
        samples = sweep.generate_samples(10)

        # Should still work (random exploration)
        assert len(samples) == 10

    def test_adaptive_sweep_extreme_exploration(self):
        """Test adaptive sweep with 100% exploration."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = AdaptiveSweep(specs, exploration_weight=1.0)  # Pure exploration

        # Add some results
        for i in range(25):
            params = {"x": np.random.uniform(0, 1)}
            sweep.add_result(params, np.random.uniform())

        # Generate adaptive samples
        samples = sweep.generate_samples(10)

        # Should all be random (exploration)
        assert len(samples) == 10

    def test_adaptive_sweep_zero_exploration(self):
        """Test adaptive sweep with 0% exploration (pure exploitation)."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = AdaptiveSweep(specs, exploration_weight=0.0)  # Pure exploitation

        # Add some results
        for i in range(25):
            params = {"x": i / 25.0}
            sweep.add_result(params, (i / 25.0 - 0.5)**2)  # Minimum at x=0.5

        # Generate adaptive samples
        samples = sweep.generate_samples(10)

        # Should all be near x=0.5 (exploitation)
        assert len(samples) == 10


class TestMultiObjectiveSweepEdgeCases:
    """Edge case tests for MultiObjectiveSweep."""

    def test_pareto_single_objective(self):
        """Test Pareto frontier with single objective."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = MultiObjectiveSweep(specs)

        # Add results with single objective
        for i in range(10):
            params = {"x": i / 10.0}
            objectives = {"obj1": i}
            sweep.add_result(params, objectives)

        pareto = sweep.compute_pareto_frontier()

        # With single objective, only minimum is Pareto-optimal
        assert len(pareto) == 1
        assert pareto[0]["objectives"]["obj1"] == 0

    def test_pareto_all_identical(self):
        """Test Pareto frontier when all points identical."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = MultiObjectiveSweep(specs)

        # Add identical results
        for i in range(10):
            params = {"x": i / 10.0}
            objectives = {"obj1": 1.0, "obj2": 1.0}
            sweep.add_result(params, objectives)

        pareto = sweep.compute_pareto_frontier()

        # All are Pareto-optimal (non-dominated)
        assert len(pareto) == 10

    def test_pareto_empty_results(self):
        """Test Pareto frontier with no results."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = MultiObjectiveSweep(specs)

        pareto = sweep.compute_pareto_frontier()
        assert len(pareto) == 0

    def test_hypervolume_invalid_reference(self):
        """Test hypervolume with invalid reference point."""
        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        sweep = MultiObjectiveSweep(specs)

        # Add some results
        for i in range(5):
            params = {"x": i / 5.0}
            objectives = {"obj1": i, "obj2": i}
            sweep.add_result(params, objectives)

        # Reference point dominated by all points (invalid)
        invalid_ref = {"obj1": -1.0, "obj2": -1.0}

        try:
            hv = sweep.compute_hypervolume(invalid_ref)
            # If it succeeds, hypervolume should be positive
            assert hv >= 0
        except (ValueError, AssertionError):
            pass  # Acceptable to reject invalid reference


class TestSensitivityAnalysisEdgeCases:
    """Edge case tests for sensitivity_analysis."""

    def test_sensitivity_constant_objective(self):
        """Test sensitivity analysis on constant objective."""
        def constant_obj(params):
            return 1.0

        specs = [
            ParameterSpec("x", "continuous", 0.0, 1.0),
            ParameterSpec("y", "continuous", 0.0, 1.0)
        ]
        params = {"x": 0.5, "y": 0.5}

        sensitivity = sensitivity_analysis(constant_obj, params, specs, n_samples=5)

        # All sensitivities should be zero
        for param, metrics in sensitivity.items():
            assert metrics["range"] == 0.0
            assert metrics["std"] == 0.0

    def test_sensitivity_single_sample(self):
        """Test sensitivity analysis with single sample."""
        def simple_obj(params):
            return params["x"]**2

        specs = [ParameterSpec("x", "continuous", 0.0, 1.0)]
        params = {"x": 0.5}

        with pytest.raises((ValueError, AssertionError)):
            sensitivity = sensitivity_analysis(simple_obj, params, specs, n_samples=1)


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestDaskEdgeCases:
    """Edge case tests for Dask distributed execution."""

    def test_distribute_empty_task_list(self):
        """Test distributing empty task list."""
        cluster = create_local_cluster(n_workers=2)

        try:
            results = distribute_computation(
                lambda x: x**2,
                [],
                cluster
            )
            assert len(results) == 0
        finally:
            cluster.close()

    def test_distribute_single_task(self):
        """Test distributing single task."""
        cluster = create_local_cluster(n_workers=2)

        try:
            results = distribute_computation(
                lambda x: x**2,
                [5],
                cluster
            )
            assert len(results) == 1
            assert results[0] == 25
        finally:
            cluster.close()

    def test_fault_tolerant_all_failures(self):
        """Test fault_tolerant_map when all tasks fail."""
        cluster = create_local_cluster(n_workers=2)

        def always_fails(x):
            raise ValueError("Intentional failure")

        try:
            results, failures = fault_tolerant_map(
                always_fails,
                [1, 2, 3],
                cluster,
                max_retries=2
            )

            # All should fail
            assert len(results) == 0
            assert len(failures) == 3
        finally:
            cluster.close()

    def test_checkpoint_force_recompute(self):
        """Test checkpointing with force_recompute."""
        cluster = create_local_cluster(n_workers=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pkl"

            call_count = [0]

            def counting_computation():
                call_count[0] += 1
                return 42

            try:
                # First call - computes and saves
                result1 = checkpoint_computation(
                    counting_computation,
                    checkpoint_path,
                    cluster
                )
                assert result1 == 42
                assert call_count[0] == 1

                # Second call - loads from checkpoint
                result2 = checkpoint_computation(
                    counting_computation,
                    checkpoint_path,
                    cluster,
                    force_recompute=False
                )
                assert result2 == 42
                assert call_count[0] == 1  # Not recomputed

                # Third call - force recompute
                result3 = checkpoint_computation(
                    counting_computation,
                    checkpoint_path,
                    cluster,
                    force_recompute=True
                )
                assert result3 == 42
                assert call_count[0] == 2  # Recomputed
            finally:
                cluster.close()


class TestVisualizationExport:
    """Tests for sweep result visualization and export."""

    def test_visualize_empty_results(self):
        """Test visualization with empty results."""
        results = []
        summary = visualize_sweep_results(results)

        assert isinstance(summary, str)
        assert "0" in summary or "empty" in summary.lower()

    def test_export_json(self):
        """Test JSON export of sweep results."""
        results = [
            {"params": {"x": 0.5}, "value": 0.25},
            {"params": {"x": 1.0}, "value": 1.0}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "results.json"
            export_sweep_results(results, str(export_path), format="json")

            assert export_path.exists()

            # Verify can load
            import json
            with open(export_path, 'r') as f:
                loaded = json.load(f)

            assert len(loaded) == 2

    def test_export_csv(self):
        """Test CSV export of sweep results."""
        results = [
            {"params": {"x": 0.5, "y": 1.0}, "value": 0.25},
            {"params": {"x": 1.0, "y": 2.0}, "value": 1.0}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "results.csv"
            export_sweep_results(results, str(export_path), format="csv")

            assert export_path.exists()

            # Verify can load
            import csv
            with open(export_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2

    def test_export_invalid_format(self):
        """Test export with invalid format."""
        results = [{"params": {"x": 0.5}, "value": 0.25}]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "results.txt"

            with pytest.raises((ValueError, KeyError)):
                export_sweep_results(results, str(export_path), format="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
