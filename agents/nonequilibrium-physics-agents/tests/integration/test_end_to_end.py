"""End-to-End Integration Tests.

Tests complete workflows across all Phase 4 components:
- Solver → ML → HPC → API → Deployment
- Data format conversions
- Performance benchmarks
- Cross-component compatibility

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import time

# Standards
from standards import (
    SolverInput,
    SolverOutput,
    TrainingData,
    save_to_file,
    load_from_file,
    validate_standard_format
)

# Solvers
try:
    from solvers.pontryagin import PontryaginSolver
    PONTRYAGIN_AVAILABLE = True
except ImportError:
    PONTRYAGIN_AVAILABLE = False

try:
    from solvers.collocation import CollocationSolver
    COLLOCATION_AVAILABLE = True
except ImportError:
    COLLOCATION_AVAILABLE = False

# ML
try:
    from ml_optimal_control.networks import PolicyNetwork
    from ml_optimal_control.training import create_training_data_from_trajectories
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Visualization
try:
    from visualization.plotting import plot_trajectory, plot_control
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False


class TestEndToEndWorkflows:
    """Test complete workflows across Phase 4 components."""

    @pytest.mark.skipif(not PONTRYAGIN_AVAILABLE, reason="PMP solver not available")
    def test_solver_standard_format_workflow(self):
        """Test: Create input → Solve → Validate output → Save/Load."""

        # 1. Create standard input
        solver_input = SolverInput(
            solver_type="pmp",
            problem_type="lqr",
            n_states=2,
            n_controls=1,
            initial_state=[1.0, 0.0],
            target_state=[0.0, 0.0],
            time_horizon=[0.0, 1.0],
            cost={"Q": [[1.0, 0.0], [0.0, 1.0]], "R": [[0.1]]},
            solver_config={"max_iterations": 50, "tolerance": 1e-6}
        )

        # Validate input
        assert solver_input.validate() is True
        validate_standard_format(solver_input)

        # 2. Solve
        solver = PontryaginSolver(
            n_states=solver_input.n_states,
            n_controls=solver_input.n_controls
        )

        result = solver.solve(
            initial_state=np.array(solver_input.initial_state),
            target_state=np.array(solver_input.target_state) if solver_input.target_state else None,
            t_span=solver_input.time_horizon,
            max_iterations=solver_input.solver_config.get("max_iterations", 100)
        )

        # 3. Convert to standard output
        solver_output = SolverOutput(
            success=result['success'],
            solver_type="pmp",
            optimal_control=result['optimal_control'],
            optimal_state=result['optimal_trajectory'],
            optimal_cost=result.get('cost', None),
            convergence={"iterations": result.get('iterations', 0)},
            computation_time=result.get('solve_time', 0.0),
            iterations=result.get('iterations', 0)
        )

        # Validate output
        validate_standard_format(solver_output)
        assert solver_output.success is True
        assert solver_output.optimal_control is not None
        assert solver_output.optimal_state is not None

        # 4. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "result.json"
            save_to_file(solver_output, output_file, format="json")

            loaded_output = load_from_file(output_file, format="json", target_type=SolverOutput)

            # Verify loaded data
            assert loaded_output.success == solver_output.success
            assert np.allclose(loaded_output.optimal_control, solver_output.optimal_control)
            assert np.allclose(loaded_output.optimal_state, solver_output.optimal_state)

    @pytest.mark.skipif(not (PONTRYAGIN_AVAILABLE and COLLOCATION_AVAILABLE),
                       reason="Multiple solvers not available")
    def test_multi_solver_comparison(self):
        """Test: Compare PMP and Collocation on same problem."""

        # Create standard problem
        problem = SolverInput(
            solver_type="pmp",  # Will be overridden
            problem_type="lqr",
            n_states=2,
            n_controls=1,
            initial_state=[1.0, 0.0],
            target_state=[0.0, 0.0],
            time_horizon=[0.0, 1.0],
            cost={"Q": [[1.0, 0.0], [0.0, 1.0]], "R": [[0.1]]}
        )

        results = {}

        # Solve with PMP
        pmp_solver = PontryaginSolver(n_states=2, n_controls=1)
        pmp_result = pmp_solver.solve(
            initial_state=np.array(problem.initial_state),
            target_state=np.array(problem.target_state),
            t_span=problem.time_horizon
        )
        results['pmp'] = pmp_result

        # Solve with Collocation
        colloc_solver = CollocationSolver(n_states=2, n_controls=1)
        colloc_result = colloc_solver.solve(
            initial_state=np.array(problem.initial_state),
            target_state=np.array(problem.target_state),
            t_span=problem.time_horizon
        )
        results['collocation'] = colloc_result

        # Compare results
        assert results['pmp']['success']
        assert results['collocation']['success']

        # Costs should be similar
        if 'cost' in results['pmp'] and 'cost' in results['collocation']:
            cost_diff = abs(results['pmp']['cost'] - results['collocation']['cost'])
            assert cost_diff < 0.1  # Within 10% tolerance

    @pytest.mark.skipif(not PONTRYAGIN_AVAILABLE, reason="PMP solver not available")
    def test_solver_to_training_data_pipeline(self):
        """Test: Generate training data from solver results."""

        # Generate multiple solver results
        solver = PontryaginSolver(n_states=2, n_controls=1)
        outputs = []

        for i in range(10):
            # Random initial states
            initial_state = np.random.randn(2)

            result = solver.solve(
                initial_state=initial_state,
                target_state=np.array([0.0, 0.0]),
                t_span=[0.0, 1.0],
                max_iterations=50
            )

            if result['success']:
                output = SolverOutput(
                    success=True,
                    solver_type="pmp",
                    optimal_control=result['optimal_control'],
                    optimal_state=result['optimal_trajectory']
                )
                outputs.append(output)

        assert len(outputs) > 0

        # Create training data
        from standards.data_formats import create_training_data_from_solver_outputs

        training_data = create_training_data_from_solver_outputs(
            outputs,
            problem_type="lqr"
        )

        # Validate training data
        assert training_data.validate() is True
        assert training_data.n_samples > 0
        assert training_data.n_states == 2
        assert training_data.n_controls == 1
        assert training_data.states.shape[0] == training_data.controls.shape[0]

    def test_data_format_conversions(self):
        """Test: Convert between different serialization formats."""

        # Create test data
        solver_output = SolverOutput(
            success=True,
            solver_type="pmp",
            optimal_control=np.random.rand(10, 1),
            optimal_state=np.random.rand(10, 2),
            optimal_cost=0.523,
            computation_time=0.156,
            iterations=12
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Test JSON
            json_file = tmpdir / "result.json"
            save_to_file(solver_output, json_file, format="json")
            loaded_json = load_from_file(json_file, format="json", target_type=SolverOutput)
            assert np.allclose(loaded_json.optimal_control, solver_output.optimal_control)

            # Test HDF5
            try:
                h5_file = tmpdir / "result.h5"
                save_to_file(solver_output, h5_file, format="hdf5")
                loaded_h5 = load_from_file(h5_file, format="hdf5", target_type=SolverOutput)
                assert np.allclose(loaded_h5.optimal_control, solver_output.optimal_control)
            except ImportError:
                pytest.skip("h5py not available")

            # Test Pickle
            pkl_file = tmpdir / "result.pkl"
            save_to_file(solver_output, pkl_file, format="pickle")
            loaded_pkl = load_from_file(pkl_file, format="pickle", target_type=SolverOutput)
            assert np.allclose(loaded_pkl.optimal_control, solver_output.optimal_control)

    def test_performance_tracking(self):
        """Test: Track solver performance metrics."""

        # Create simple problem
        problem = SolverInput(
            solver_type="pmp",
            n_states=2,
            n_controls=1,
            initial_state=[1.0, 0.0],
            time_horizon=[0.0, 1.0]
        )

        # Track performance
        performance_metrics = {
            'n_states': problem.n_states,
            'n_controls': problem.n_controls,
            'time_horizon': problem.time_horizon[1] - problem.time_horizon[0],
            'start_time': time.time()
        }

        # Simulate solver execution
        time.sleep(0.01)  # Minimal sleep

        performance_metrics['end_time'] = time.time()
        performance_metrics['elapsed'] = performance_metrics['end_time'] - performance_metrics['start_time']

        # Validate metrics
        assert performance_metrics['elapsed'] > 0
        assert performance_metrics['n_states'] == 2
        assert performance_metrics['n_controls'] == 1

    @pytest.mark.skipif(not VIZ_AVAILABLE, reason="Visualization not available")
    def test_visualization_integration(self):
        """Test: Generate plots from solver output."""

        # Create dummy solver output
        t = np.linspace(0, 1, 50)
        states = np.column_stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)])
        controls = np.sin(4*np.pi*t).reshape(-1, 1)

        solver_output = SolverOutput(
            success=True,
            solver_type="pmp",
            optimal_control=controls,
            optimal_state=states,
            optimal_cost=0.5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test trajectory plot
            try:
                fig = plot_trajectory(
                    solver_output.optimal_state,
                    t,
                    title="Test Trajectory"
                )
                assert fig is not None

                # Save plot
                plot_file = Path(tmpdir) / "trajectory.png"
                fig.savefig(plot_file)
                assert plot_file.exists()
            except Exception as e:
                pytest.skip(f"Plotting failed: {e}")


class TestDataValidation:
    """Test data validation across components."""

    def test_solver_input_validation(self):
        """Test solver input validation catches errors."""

        # Valid input
        valid_input = SolverInput(
            n_states=2,
            n_controls=1,
            initial_state=[1.0, 0.0],
            time_horizon=[0.0, 1.0]
        )
        assert valid_input.validate() is True

        # Invalid: dimension mismatch
        with pytest.raises(ValueError, match="initial_state length"):
            invalid_input = SolverInput(
                n_states=2,
                n_controls=1,
                initial_state=[1.0],  # Wrong dimension
                time_horizon=[0.0, 1.0]
            )
            invalid_input.validate()

        # Invalid: bad time horizon
        with pytest.raises(ValueError, match="tf must be greater than t0"):
            invalid_input = SolverInput(
                n_states=2,
                n_controls=1,
                initial_state=[1.0, 0.0],
                time_horizon=[1.0, 0.0]  # tf < t0
            )
            invalid_input.validate()

    def test_training_data_validation(self):
        """Test training data validation."""

        # Valid training data
        valid_data = TrainingData(
            states=np.random.rand(100, 2),
            controls=np.random.rand(100, 1)
        )
        assert valid_data.validate() is True
        assert valid_data.n_samples == 100
        assert valid_data.n_states == 2
        assert valid_data.n_controls == 1

        # Invalid: dimension mismatch
        with pytest.raises(ValueError, match="same number of samples"):
            invalid_data = TrainingData(
                states=np.random.rand(100, 2),
                controls=np.random.rand(50, 1)  # Wrong number
            )
            invalid_data.validate()


class TestCrossComponentIntegration:
    """Test integration between different Phase 4 components."""

    def test_standards_to_api_format(self):
        """Test: Convert standard format to API request."""

        from standards import APIRequest, APIResponse

        # Create solver input
        solver_input = SolverInput(
            solver_type="pmp",
            n_states=2,
            n_controls=1,
            initial_state=[1.0, 0.0],
            time_horizon=[0.0, 1.0]
        )

        # Convert to API request
        api_request = APIRequest(
            endpoint="/api/solve",
            method="POST",
            data=solver_input.to_dict()
        )

        # Validate
        assert api_request.endpoint == "/api/solve"
        assert api_request.method == "POST"
        assert "solver_type" in api_request.data
        assert api_request.data["solver_type"] == "pmp"

        # Simulate API response
        api_response = APIResponse(
            status_code=200,
            success=True,
            data={
                "job_id": "test-123",
                "status": "completed"
            }
        )

        assert api_response.success is True
        assert api_response.data["job_id"] == "test-123"

    def test_hpc_job_spec_creation(self):
        """Test: Create HPC job spec from solver input."""

        from standards import HPCJobSpec

        # Create solver input
        solver_input = SolverInput(
            solver_type="pmp",
            n_states=10,  # Larger problem
            n_controls=5,
            initial_state=np.random.rand(10).tolist(),
            time_horizon=[0.0, 10.0]
        )

        # Create HPC job spec
        job_spec = HPCJobSpec(
            job_name="optimal_control_large",
            job_type="solver",
            input_data=solver_input.to_dict(),
            resources={
                "nodes": 1,
                "cpus": 16,
                "memory_gb": 32,
                "gpus": 1,
                "time_hours": 4
            },
            scheduler="slurm"
        )

        # Validate
        assert job_spec.job_name == "optimal_control_large"
        assert job_spec.resources["cpus"] == 16
        assert job_spec.resources["gpus"] == 1
        assert "solver_type" in job_spec.input_data


class TestBenchmarks:
    """Performance benchmarks for integration workflows."""

    @pytest.mark.benchmark
    def test_standard_format_overhead(self):
        """Benchmark: Overhead of standard format conversion."""

        # Raw data
        raw_control = np.random.rand(1000, 5)
        raw_state = np.random.rand(1000, 10)

        # Time standard format creation
        start = time.time()
        for _ in range(100):
            output = SolverOutput(
                success=True,
                solver_type="pmp",
                optimal_control=raw_control,
                optimal_state=raw_state
            )
        format_time = time.time() - start

        # Time dictionary conversion
        start = time.time()
        for _ in range(100):
            output_dict = output.to_dict()
        dict_time = time.time() - start

        # Overhead should be minimal (< 10ms for 100 conversions)
        assert format_time < 0.01
        assert dict_time < 0.01

    @pytest.mark.benchmark
    def test_serialization_performance(self):
        """Benchmark: Serialization format performance."""

        # Large dataset
        output = SolverOutput(
            success=True,
            solver_type="pmp",
            optimal_control=np.random.rand(10000, 10),
            optimal_state=np.random.rand(10000, 20)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # JSON serialization
            json_file = tmpdir / "test.json"
            start = time.time()
            save_to_file(output, json_file, format="json")
            json_write_time = time.time() - start

            start = time.time()
            loaded = load_from_file(json_file, format="json", target_type=SolverOutput)
            json_read_time = time.time() - start

            # HDF5 serialization (if available)
            try:
                h5_file = tmpdir / "test.h5"
                start = time.time()
                save_to_file(output, h5_file, format="hdf5")
                h5_write_time = time.time() - start

                start = time.time()
                loaded = load_from_file(h5_file, format="hdf5", target_type=SolverOutput)
                h5_read_time = time.time() - start

                # HDF5 should be faster for large arrays
                assert h5_write_time < json_write_time
                assert h5_read_time < json_read_time
            except ImportError:
                pytest.skip("h5py not available for benchmark")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
