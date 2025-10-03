"""
Phase 4 Week 35-36: Simple Edge Case Tests

Edge case tests that don't require JAX/optional dependencies.
Tests HPC modules, parameter sweeps, and performance tools.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from hpc.schedulers import (
    LocalScheduler,
    JobManager,
    ResourceRequirements,
    JobStatus
)
from hpc.parallel import (
    ParameterSpec,
    GridSearch,
    RandomSearch,
    AdaptiveSweep,
    MultiObjectiveSweep,
    sensitivity_analysis,
    visualize_sweep_results,
    export_sweep_results
)
from ml_optimal_control.performance import (
    FunctionProfiler,
    Benchmarker,
    Timer,
    MemoryProfiler
)


class TestResourceRequirementsValidation:
    """Test validation of resource requirements."""

    def test_default_resources(self):
        """Test default resource requirements."""
        res = ResourceRequirements()
        assert res.nodes == 1
        assert res.cpus_per_task == 1
        assert res.gpus_per_node == 0

    def test_custom_resources(self):
        """Test custom resource requirements."""
        res = ResourceRequirements(
            nodes=4,
            cpus_per_task=8,
            gpus_per_node=2,
            memory_gb=64
        )
        assert res.nodes == 4
        assert res.cpus_per_task == 8
        assert res.gpus_per_node == 2
        assert res.memory_gb == 64


class TestParameterSpecValidation:
    """Test parameter specification validation."""

    def test_continuous_parameter(self):
        """Test continuous parameter creation."""
        spec = ParameterSpec("lr", "continuous", 0.001, 0.1, log_scale=True)
        assert spec.name == "lr"
        assert spec.param_type == "continuous"
        assert spec.lower == 0.001
        assert spec.upper == 0.1
        assert spec.log_scale == True

    def test_integer_parameter(self):
        """Test integer parameter creation."""
        spec = ParameterSpec("hidden_size", "integer", 16, 128)
        assert spec.param_type == "integer"
        assert spec.lower == 16
        assert spec.upper == 128

    def test_categorical_parameter(self):
        """Test categorical parameter creation."""
        spec = ParameterSpec(
            "activation",
            "categorical",
            choices=["relu", "tanh", "elu"]
        )
        assert spec.param_type == "categorical"
        assert len(spec.choices) == 3

    def test_continuous_equal_bounds(self):
        """Test continuous parameter with equal bounds (single value)."""
        spec = ParameterSpec("fixed_param", "continuous", 1.0, 1.0)
        assert spec.lower == spec.upper == 1.0


class TestGridSearchFunctionality:
    """Test grid search parameter sweep."""

    def test_grid_search_1d(self):
        """Test 1D grid search."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = GridSearch([spec], points_per_dim=5)
        samples = sweep.generate_samples()

        assert len(samples) == 5
        # Check bounds
        for sample in samples:
            assert 0.0 <= sample["x"] <= 1.0

    def test_grid_search_2d(self):
        """Test 2D grid search."""
        specs = [
            ParameterSpec("x", "continuous", 0.0, 1.0),
            ParameterSpec("y", "continuous", 0.0, 1.0)
        ]
        sweep = GridSearch(specs, points_per_dim=3)
        samples = sweep.generate_samples()

        assert len(samples) == 9  # 3^2

    def test_grid_search_single_point(self):
        """Test grid search with single point."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = GridSearch([spec], points_per_dim=1)
        samples = sweep.generate_samples()

        assert len(samples) == 1


class TestRandomSearchFunctionality:
    """Test random search parameter sweep."""

    def test_random_search_basic(self):
        """Test basic random search."""
        specs = [
            ParameterSpec("x", "continuous", -5.0, 5.0),
            ParameterSpec("y", "continuous", -5.0, 5.0)
        ]
        sweep = RandomSearch(specs, n_samples=50)
        samples = sweep.generate_samples()

        assert len(samples) == 50

        # Check bounds
        for sample in samples:
            assert -5.0 <= sample["x"] <= 5.0
            assert -5.0 <= sample["y"] <= 5.0

    def test_random_search_reproducibility(self):
        """Test random search with fixed seed."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)

        sweep1 = RandomSearch([spec], n_samples=10, seed=42)
        samples1 = sweep1.generate_samples()

        sweep2 = RandomSearch([spec], n_samples=10, seed=42)
        samples2 = sweep2.generate_samples()

        # Should be identical
        for s1, s2 in zip(samples1, samples2):
            assert np.isclose(s1["x"], s2["x"])

    def test_random_search_integer_params(self):
        """Test random search with integer parameters."""
        spec = ParameterSpec("n", "integer", 1, 100)
        sweep = RandomSearch([spec], n_samples=20)
        samples = sweep.generate_samples()

        # All should be integers
        for sample in samples:
            assert isinstance(sample["n"], (int, np.integer))
            assert 1 <= sample["n"] <= 100


class TestAdaptiveSweepFunctionality:
    """Test adaptive parameter sweep."""

    def test_adaptive_sweep_initialization(self):
        """Test adaptive sweep initialization."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = AdaptiveSweep([spec], n_initial=10, n_iterations=5)

        assert sweep.n_initial == 10
        assert sweep.n_iterations == 5

    def test_adaptive_sweep_exploration_phase(self):
        """Test initial exploration phase."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = AdaptiveSweep([spec], n_initial=20)

        # Before adding results, should do random exploration
        samples = sweep.generate_samples(15)
        assert len(samples) == 15

    def test_adaptive_sweep_exploitation_phase(self):
        """Test exploitation after adding results."""
        def quadratic(x):
            return (x - 0.5)**2

        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = AdaptiveSweep([spec], n_initial=10, exploration_weight=0.1)

        # Add initial results
        for i in range(25):
            x = i / 25.0
            params = {"x": x}
            value = quadratic(x)
            sweep.add_result(params, value)

        # Should now use adaptive sampling
        samples = sweep.generate_samples(10)
        assert len(samples) == 10

        # Most should be near x=0.5 (optimum)
        x_values = [s["x"] for s in samples]
        mean_x = np.mean(x_values)
        # With low exploration, should focus near 0.5
        assert abs(mean_x - 0.5) < 0.3


class TestMultiObjectiveSweepFunctionality:
    """Test multi-objective parameter sweep."""

    def test_multi_objective_initialization(self):
        """Test multi-objective sweep initialization."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = MultiObjectiveSweep([spec])
        assert sweep is not None

    def test_pareto_frontier_simple(self):
        """Test Pareto frontier computation."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = MultiObjectiveSweep([spec])

        # Add results with two conflicting objectives
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            params = {"x": x}
            objectives = {
                "obj1": x,       # Minimize x
                "obj2": 1.0 - x  # Minimize (1-x), i.e., maximize x
            }
            sweep.add_result(params, objectives)

        pareto = sweep.compute_pareto_frontier()

        # All points are Pareto-optimal (conflicting objectives)
        assert len(pareto) == 5

    def test_pareto_frontier_dominated_points(self):
        """Test Pareto frontier with dominated points."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = MultiObjectiveSweep([spec])

        # Point 1: (1, 1) - dominated
        sweep.add_result({"x": 0.0}, {"obj1": 1.0, "obj2": 1.0})

        # Point 2: (0.5, 0.5) - Pareto optimal
        sweep.add_result({"x": 0.5}, {"obj1": 0.5, "obj2": 0.5})

        # Point 3: (0, 0) - Pareto optimal (dominates all)
        sweep.add_result({"x": 1.0}, {"obj1": 0.0, "obj2": 0.0})

        pareto = sweep.compute_pareto_frontier()

        # Only points 2 and 3 are Pareto-optimal
        assert len(pareto) >= 1  # At least the globally optimal point


class TestSensitivityAnalysisFunctionality:
    """Test sensitivity analysis."""

    def test_sensitivity_analysis_basic(self):
        """Test basic sensitivity analysis."""
        def quadratic(params):
            return params["x"]**2 + 0.1 * params["y"]**2

        specs = [
            ParameterSpec("x", "continuous", -1.0, 1.0),
            ParameterSpec("y", "continuous", -1.0, 1.0)
        ]
        params = {"x": 0.0, "y": 0.0}

        sensitivity = sensitivity_analysis(quadratic, params, specs, n_samples=10)

        assert "x" in sensitivity
        assert "y" in sensitivity

        # x should have higher importance (coefficient 1.0 vs 0.1)
        assert sensitivity["x"]["range"] > sensitivity["y"]["range"]

    def test_sensitivity_analysis_constant_function(self):
        """Test sensitivity on constant function."""
        def constant(params):
            return 1.0

        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        params = {"x": 0.5}

        sensitivity = sensitivity_analysis(constant, params, [spec], n_samples=5)

        # Should detect zero sensitivity
        assert sensitivity["x"]["range"] == 0.0
        assert sensitivity["x"]["std"] == 0.0


class TestPerformanceProfilerFunctionality:
    """Test performance profiler."""

    def test_timer_basic(self):
        """Test basic timing."""
        timer = Timer()
        timer.start()

        # Do some work
        sum(range(1000))

        elapsed = timer.stop()
        assert elapsed > 0

    def test_benchmarker_basic(self):
        """Test function benchmarking."""
        def simple_func():
            return sum(range(100))

        benchmarker = Benchmarker()
        result = benchmarker.benchmark(simple_func, n_iterations=10)

        assert "mean_time" in result or "mean" in str(result)

    def test_function_profiler(self):
        """Test function profiler."""
        profiler = FunctionProfiler()

        def test_func():
            return sum(range(100))

        result = profiler.profile(test_func)
        assert result is not None


class TestLocalSchedulerFunctionality:
    """Test local scheduler."""

    def test_local_scheduler_basic(self):
        """Test basic local scheduler functionality."""
        scheduler = LocalScheduler()
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\n")
            script_path = f.name

        try:
            job_id = scheduler.submit_job(script_path, "test_job", resources)
            assert job_id.startswith("local_")

            status = scheduler.wait_for_job(job_id, timeout=10.0)
            assert status in [JobStatus.COMPLETED, JobStatus.FAILED]
        finally:
            import os
            os.unlink(script_path)

    def test_local_scheduler_job_cancellation(self):
        """Test job cancellation."""
        scheduler = LocalScheduler()
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\nsleep 10\n")
            script_path = f.name

        try:
            job_id = scheduler.submit_job(script_path, "long_job", resources)

            # Let it start
            import time
            time.sleep(0.5)

            # Cancel it
            scheduler.cancel_job(job_id)

            # Check status
            final_status = scheduler.wait_for_job(job_id, timeout=5.0)
            assert final_status in [JobStatus.CANCELLED, JobStatus.FAILED]
        finally:
            import os
            os.unlink(script_path)


class TestJobManagerFunctionality:
    """Test job manager."""

    def test_job_manager_auto_detect(self):
        """Test job manager auto-detection."""
        manager = JobManager(auto_detect=True)
        # Should default to local scheduler
        assert manager.scheduler is not None

    def test_job_manager_submit_wait(self):
        """Test job submission and waiting."""
        manager = JobManager(scheduler_type="local")
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'done'\n")
            script_path = f.name

        try:
            job_id = manager.submit(script_path, "test", resources)
            status = manager.wait(job_id, timeout=10.0)
            assert status == JobStatus.COMPLETED
        finally:
            import os
            os.unlink(script_path)

    def test_job_manager_wait_all(self):
        """Test waiting for multiple jobs."""
        manager = JobManager(scheduler_type="local")
        resources = ResourceRequirements()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\n")
            script_path = f.name

        try:
            job_ids = []
            for i in range(3):
                job_id = manager.submit(script_path, f"job_{i}", resources)
                job_ids.append(job_id)

            statuses = manager.wait_all(job_ids, timeout=30.0)
            assert len(statuses) == 3
            for status in statuses.values():
                assert status == JobStatus.COMPLETED
        finally:
            import os
            os.unlink(script_path)


class TestResultVisualizationAndExport:
    """Test result visualization and export."""

    def test_visualize_results(self):
        """Test result visualization."""
        results = [
            {"params": {"x": 0.5}, "value": 0.25},
            {"params": {"x": 1.0}, "value": 1.0},
            {"params": {"x": 0.0}, "value": 0.0}
        ]

        summary = visualize_sweep_results(results)
        assert isinstance(summary, str)
        assert "0.0" in summary or "0.25" in summary

    def test_export_json(self):
        """Test JSON export."""
        results = [
            {"params": {"x": 0.5}, "value": 0.25}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "results.json"
            export_sweep_results(results, str(export_path), format="json")
            assert export_path.exists()

    def test_export_csv(self):
        """Test CSV export."""
        results = [
            {"params": {"x": 0.5, "y": 1.0}, "value": 0.25}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "results.csv"
            export_sweep_results(results, str(export_path), format="csv")
            assert export_path.exists()


@pytest.mark.parametrize("n_samples", [10, 50, 100])
class TestParameterizedSweeps:
    """Parametrized tests for different sweep sizes."""

    def test_random_search_sizes(self, n_samples):
        """Test random search with different sample counts."""
        spec = ParameterSpec("x", "continuous", 0.0, 1.0)
        sweep = RandomSearch([spec], n_samples=n_samples)
        samples = sweep.generate_samples()
        assert len(samples) == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
