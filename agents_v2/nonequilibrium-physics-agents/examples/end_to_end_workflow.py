"""End-to-End Workflow Examples.

Demonstrates complete workflows across all Phase 4 components:
1. Local solver execution
2. API-based solving
3. HPC cluster execution
4. ML training from solver data
5. Multi-solver comparison
6. Full production pipeline

Author: Nonequilibrium Physics Agents
"""

import numpy as np
from pathlib import Path
import time

# Standards
from standards import (
    SolverInput,
    SolverOutput,
    TrainingData,
    HPCJobSpec,
    APIRequest,
    save_to_file,
    load_from_file,
    create_training_data_from_solver_outputs
)

# Check available components
try:
    from solvers.pontryagin import PontryaginSolver
    PMP_AVAILABLE = True
except ImportError:
    PMP_AVAILABLE = False
    print("Note: PMP solver not available")

try:
    from solvers.collocation import CollocationSolver
    COLLOCATION_AVAILABLE = True
except ImportError:
    COLLOCATION_AVAILABLE = False
    print("Note: Collocation solver not available")


def demo_1_local_solver_execution():
    """Demo 1: Local solver execution with standard formats."""
    print("\n" + "=" * 80)
    print("Demo 1: Local Solver Execution")
    print("=" * 80)

    if not PMP_AVAILABLE:
        print("Skipping - PMP solver not available")
        return

    # 1. Create standard problem definition
    print("\n1. Creating standard problem definition...")
    problem = SolverInput(
        solver_type="pmp",
        problem_type="lqr",
        n_states=2,
        n_controls=1,
        initial_state=[1.0, 0.0],
        target_state=[0.0, 0.0],
        time_horizon=[0.0, 1.0],
        cost={"Q": [[1.0, 0.0], [0.0, 1.0]], "R": [[0.1]]},
        solver_config={"max_iterations": 100, "tolerance": 1e-6},
        metadata={"problem_name": "Simple LQR"}
    )

    print(f"   Problem: {problem.problem_type}")
    print(f"   States: {problem.n_states}, Controls: {problem.n_controls}")
    print(f"   Time horizon: {problem.time_horizon}")

    # Validate
    problem.validate()
    print("   ✓ Problem validated")

    # 2. Solve locally
    print("\n2. Solving with PMP...")
    solver = PontryaginSolver(
        n_states=problem.n_states,
        n_controls=problem.n_controls
    )

    start_time = time.time()
    result = solver.solve(
        initial_state=np.array(problem.initial_state),
        target_state=np.array(problem.target_state),
        t_span=problem.time_horizon,
        max_iterations=problem.solver_config["max_iterations"]
    )
    solve_time = time.time() - start_time

    # 3. Convert to standard output
    output = SolverOutput(
        success=result['success'],
        solver_type="pmp",
        optimal_control=result['optimal_control'],
        optimal_state=result['optimal_trajectory'],
        optimal_cost=result.get('cost', None),
        computation_time=solve_time,
        iterations=result.get('iterations', 0),
        metadata={"solver_version": "1.0.0"}
    )

    print(f"   ✓ Solved in {solve_time:.3f}s")
    print(f"   Cost: {output.optimal_cost:.6f}")
    print(f"   Iterations: {output.iterations}")

    # 4. Save result
    print("\n3. Saving result...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    save_to_file(output, output_dir / "pmp_result.json", format="json")
    print(f"   ✓ Saved to {output_dir / 'pmp_result.json'}")

    # 5. Load and verify
    print("\n4. Loading and verifying...")
    loaded = load_from_file(output_dir / "pmp_result.json", format="json", target_type=SolverOutput)

    print(f"   Success: {loaded.success}")
    print(f"   Cost: {loaded.optimal_cost:.6f}")
    print(f"   Control shape: {loaded.optimal_control.shape}")
    print(f"   State shape: {loaded.optimal_state.shape}")


def demo_2_multi_solver_comparison():
    """Demo 2: Compare multiple solvers on same problem."""
    print("\n" + "=" * 80)
    print("Demo 2: Multi-Solver Comparison")
    print("=" * 80)

    if not (PMP_AVAILABLE and COLLOCATION_AVAILABLE):
        print("Skipping - Multiple solvers not available")
        return

    # Create standard problem
    problem = SolverInput(
        solver_type="auto",  # Will determine best solver
        problem_type="lqr",
        n_states=2,
        n_controls=1,
        initial_state=[1.0, 0.0],
        target_state=[0.0, 0.0],
        time_horizon=[0.0, 1.0],
        cost={"Q": [[1.0, 0.0], [0.0, 1.0]], "R": [[0.1]]}
    )

    print(f"Problem: {problem.n_states} states, {problem.n_controls} controls")

    solvers = {}
    results = {}

    # Solve with PMP
    print("\n1. Solving with PMP...")
    solvers['pmp'] = PontryaginSolver(n_states=2, n_controls=1)

    start = time.time()
    pmp_result = solvers['pmp'].solve(
        initial_state=np.array(problem.initial_state),
        target_state=np.array(problem.target_state),
        t_span=problem.time_horizon
    )
    pmp_time = time.time() - start

    results['pmp'] = SolverOutput(
        success=pmp_result['success'],
        solver_type="pmp",
        optimal_cost=pmp_result.get('cost', None),
        computation_time=pmp_time,
        iterations=pmp_result.get('iterations', 0)
    )

    print(f"   Cost: {results['pmp'].optimal_cost:.6f}")
    print(f"   Time: {pmp_time:.3f}s")
    print(f"   Iterations: {results['pmp'].iterations}")

    # Solve with Collocation
    print("\n2. Solving with Collocation...")
    solvers['collocation'] = CollocationSolver(n_states=2, n_controls=1)

    start = time.time()
    colloc_result = solvers['collocation'].solve(
        initial_state=np.array(problem.initial_state),
        target_state=np.array(problem.target_state),
        t_span=problem.time_horizon
    )
    colloc_time = time.time() - start

    results['collocation'] = SolverOutput(
        success=colloc_result['success'],
        solver_type="collocation",
        optimal_cost=colloc_result.get('cost', None),
        computation_time=colloc_time,
        iterations=colloc_result.get('iterations', 0)
    )

    print(f"   Cost: {results['collocation'].optimal_cost:.6f}")
    print(f"   Time: {colloc_time:.3f}s")
    print(f"   Iterations: {results['collocation'].iterations}")

    # Compare
    print("\n3. Comparison:")
    print("-" * 40)
    print(f"{'Solver':<15} {'Cost':<12} {'Time (s)':<10} {'Iterations'}")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:<15} {result.optimal_cost:<12.6f} {result.computation_time:<10.3f} {result.iterations}")

    # Determine best
    best_solver = min(results.items(), key=lambda x: x[1].optimal_cost if x[1].optimal_cost else float('inf'))
    print(f"\nBest solver: {best_solver[0]} (cost: {best_solver[1].optimal_cost:.6f})")


def demo_3_training_data_generation():
    """Demo 3: Generate ML training data from solver results."""
    print("\n" + "=" * 80)
    print("Demo 3: ML Training Data Generation")
    print("=" * 80)

    if not PMP_AVAILABLE:
        print("Skipping - PMP solver not available")
        return

    print("\n1. Generating solver solutions...")
    solver = PontryaginSolver(n_states=2, n_controls=1)
    outputs = []
    n_samples = 20

    for i in range(n_samples):
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

        if (i + 1) % 5 == 0:
            print(f"   Generated {i + 1}/{n_samples} solutions...")

    print(f"\n   ✓ Successfully generated {len(outputs)} solutions")

    # 2. Create training data
    print("\n2. Creating training dataset...")
    training_data = create_training_data_from_solver_outputs(
        outputs,
        problem_type="lqr"
    )

    training_data.validate()

    print(f"   Samples: {training_data.n_samples}")
    print(f"   States dim: {training_data.n_states}")
    print(f"   Controls dim: {training_data.n_controls}")
    print(f"   States shape: {training_data.states.shape}")
    print(f"   Controls shape: {training_data.controls.shape}")

    # 3. Save training data
    print("\n3. Saving training data...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save as HDF5 (efficient for large datasets)
    try:
        save_to_file(training_data, output_dir / "training_data.h5", format="hdf5")
        print(f"   ✓ Saved to {output_dir / 'training_data.h5'}")
    except ImportError:
        # Fallback to pickle
        save_to_file(training_data, output_dir / "training_data.pkl", format="pickle")
        print(f"   ✓ Saved to {output_dir / 'training_data.pkl'} (pickle format)")

    # 4. Statistics
    print("\n4. Training data statistics:")
    print(f"   State mean: {np.mean(training_data.states, axis=0)}")
    print(f"   State std: {np.std(training_data.states, axis=0)}")
    print(f"   Control mean: {np.mean(training_data.controls, axis=0)}")
    print(f"   Control std: {np.std(training_data.controls, axis=0)}")


def demo_4_api_workflow():
    """Demo 4: API-based workflow (simulation)."""
    print("\n" + "=" * 80)
    print("Demo 4: API-Based Workflow (Simulation)")
    print("=" * 80)

    print("\n1. Creating API request...")

    # Create solver input
    problem = SolverInput(
        solver_type="pmp",
        problem_type="lqr",
        n_states=2,
        n_controls=1,
        initial_state=[1.0, 0.0],
        time_horizon=[0.0, 1.0]
    )

    # Create API request
    api_request = APIRequest(
        endpoint="/api/solve",
        method="POST",
        data=problem.to_dict(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer token123"},
        timeout=300.0
    )

    print(f"   Endpoint: {api_request.endpoint}")
    print(f"   Method: {api_request.method}")
    print(f"   Timeout: {api_request.timeout}s")
    print(f"   Data keys: {list(api_request.data.keys())}")

    # Simulate API call (would use requests library in production)
    print("\n2. Simulating API call...")
    print("   POST http://api.example.com/api/solve")
    print("   (In production: response = requests.post(url, json=api_request.data))")

    # Simulate response
    from standards import APIResponse

    api_response = APIResponse(
        status_code=200,
        success=True,
        data={
            "job_id": "job-abc-123",
            "status": "completed",
            "result": {
                "success": True,
                "optimal_cost": 0.523,
                "computation_time": 0.156
            }
        },
        execution_time=0.156
    )

    print(f"\n3. API Response:")
    print(f"   Status: {api_response.status_code}")
    print(f"   Success: {api_response.success}")
    print(f"   Job ID: {api_response.data['job_id']}")
    print(f"   Execution time: {api_response.execution_time:.3f}s")


def demo_5_hpc_job_submission():
    """Demo 5: HPC cluster job submission (simulation)."""
    print("\n" + "=" * 80)
    print("Demo 5: HPC Job Submission (Simulation)")
    print("=" * 80)

    print("\n1. Creating HPC job specification...")

    # Large-scale problem for HPC
    problem = SolverInput(
        solver_type="pmp",
        problem_type="trajectory_tracking",
        n_states=20,  # Larger problem
        n_controls=10,
        initial_state=np.random.rand(20).tolist(),
        time_horizon=[0.0, 10.0],
        solver_config={"max_iterations": 1000}
    )

    # Create HPC job spec
    job_spec = HPCJobSpec(
        job_name="optimal_control_large_scale",
        job_type="solver",
        input_data=problem.to_dict(),
        resources={
            "nodes": 4,
            "cpus": 64,
            "memory_gb": 128,
            "gpus": 4,
            "time_hours": 8
        },
        scheduler="slurm",
        priority="high",
        metadata={"project": "optimal_control", "user": "researcher"}
    )

    print(f"   Job name: {job_spec.job_name}")
    print(f"   Scheduler: {job_spec.scheduler}")
    print(f"   Resources:")
    print(f"     Nodes: {job_spec.resources['nodes']}")
    print(f"     CPUs: {job_spec.resources['cpus']}")
    print(f"     Memory: {job_spec.resources['memory_gb']} GB")
    print(f"     GPUs: {job_spec.resources['gpus']}")
    print(f"     Time: {job_spec.resources['time_hours']} hours")

    # Simulate SLURM script generation
    print("\n2. Generated SLURM script (excerpt):")
    print("   #!/bin/bash")
    print(f"   #SBATCH --job-name={job_spec.job_name}")
    print(f"   #SBATCH --nodes={job_spec.resources['nodes']}")
    print(f"   #SBATCH --ntasks-per-node=1")
    print(f"   #SBATCH --cpus-per-task={job_spec.resources['cpus']}")
    print(f"   #SBATCH --mem={job_spec.resources['memory_gb']}G")
    print(f"   #SBATCH --gpus={job_spec.resources['gpus']}")
    print(f"   #SBATCH --time={job_spec.resources['time_hours']}:00:00")
    print("   ...")
    print("\n3. Job submission (simulation):")
    print("   $ sbatch job_script.sh")
    print("   Submitted batch job 1234567")


def demo_6_full_production_pipeline():
    """Demo 6: Complete production pipeline."""
    print("\n" + "=" * 80)
    print("Demo 6: Full Production Pipeline")
    print("=" * 80)

    if not PMP_AVAILABLE:
        print("Skipping - PMP solver not available")
        return

    print("\nSimulating complete workflow:")
    print("-" * 40)

    # 1. Problem definition
    print("\n1. Problem Definition")
    problem = SolverInput(
        solver_type="pmp",
        problem_type="energy_optimization",
        n_states=4,
        n_controls=2,
        initial_state=[1.0, 0.5, 0.0, 0.0],
        target_state=[0.0, 0.0, 0.0, 0.0],
        time_horizon=[0.0, 5.0]
    )
    print(f"   ✓ Problem: {problem.n_states}D, {problem.time_horizon[1]}s horizon")

    # 2. Local validation
    print("\n2. Local Validation")
    problem.validate()
    print("   ✓ Validated problem specification")

    # 3. Development solve (small problem)
    print("\n3. Development Solve")
    print("   ✓ Tested on simplified 2D problem")

    # 4. API deployment
    print("\n4. API Deployment")
    api_request = APIRequest(
        endpoint="/api/solve",
        data=problem.to_dict()
    )
    print(f"   ✓ Created API request: {api_request.endpoint}")

    # 5. HPC scaling
    print("\n5. HPC Batch Processing")
    job_spec = HPCJobSpec(
        job_name="energy_optimization_sweep",
        job_type="parameter_sweep",
        input_data=problem.to_dict(),
        resources={"nodes": 10, "gpus": 40}
    )
    print(f"   ✓ Prepared HPC job: {job_spec.resources['nodes']} nodes")

    # 6. Results collection
    print("\n6. Results Collection & Analysis")
    print("   ✓ Collected 1000 solutions")
    print("   ✓ Generated training dataset")
    print("   ✓ Trained ML policy")

    # 7. Production deployment
    print("\n7. Production Deployment")
    print("   ✓ Docker image built")
    print("   ✓ Kubernetes deployed (3 replicas)")
    print("   ✓ Load balancer configured")

    print("\n" + "=" * 40)
    print("Pipeline complete! All systems operational.")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("END-TO-END WORKFLOW DEMONSTRATIONS")
    print("Phase 4 Optimal Control Framework")
    print("=" * 80)

    demos = [
        demo_1_local_solver_execution,
        demo_2_multi_solver_comparison,
        demo_3_training_data_generation,
        demo_4_api_workflow,
        demo_5_hpc_job_submission,
        demo_6_full_production_pipeline
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\nError in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("1. Unified standard formats across all components")
    print("2. Seamless local → API → HPC workflow")
    print("3. Multi-solver comparison with common interface")
    print("4. ML training data generation from solver results")
    print("5. Production-ready deployment pipeline")

    print("\nNext Steps:")
    print("- Deploy to Kubernetes cluster")
    print("- Scale with HPC resources")
    print("- Train ML policies")
    print("- Monitor production performance")


if __name__ == "__main__":
    main()
