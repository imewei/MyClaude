"""Demonstrations of Dask Distributed Computing.

Shows practical applications of Dask for parallel and distributed optimal
control computations.

Author: Nonequilibrium Physics Agents
Week: 31-32 of Phase 4
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import Dask
try:
    import dask
    import dask.array as da
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("⚠ Dask not installed. Install with: pip install dask[complete] distributed")
    print("Demonstrations will show code structure only.\n")

from hpc.distributed import (
    create_local_cluster,
    distribute_computation,
    distributed_optimization,
    pipeline,
    scatter_gather_reduction,
    checkpoint_computation,
    fault_tolerant_map,
    DASK_AVAILABLE as MODULE_DASK_AVAILABLE
)


def demo_1_local_cluster():
    """Demo 1: Create and use local Dask cluster."""
    print("\n" + "="*70)
    print("DEMO 1: Local Dask Cluster")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⊘ Dask not available - showing code structure")
        print("""
# Create local cluster with 4 workers
cluster = create_local_cluster(n_workers=4, threads_per_worker=2)

# Submit task
def compute_control(state, params):
    # Optimal control computation
    return -params @ state  # LQR-style

future = cluster.submit(compute_control, state, K)
result = future.result()

# Cleanup
cluster.close()
        """)
        return

    print("\nScenario: Multi-worker local cluster for parallel computation")

    # Create cluster
    print("\n1. Creating local cluster...")
    cluster = create_local_cluster(n_workers=4, threads_per_worker=1)
    print(f"   Cluster: {cluster}")

    # Get cluster info
    info = cluster.client.scheduler_info()
    print(f"   Workers: {len(info['workers'])}")
    print(f"   Threads: {info['workers'][list(info['workers'].keys())[0]]['nthreads']}")

    # Submit simple computation
    print("\n2. Submitting computation...")
    def optimal_control_gain(A, B, Q, R):
        """Simplified LQR gain computation."""
        time.sleep(0.5)  # Simulate computation
        n = A.shape[0]
        # Simplified: just use identity
        return np.eye(n)

    A = np.random.randn(5, 5)
    B = np.random.randn(5, 2)
    Q = np.eye(5)
    R = np.eye(2)

    future = cluster.submit(optimal_control_gain, A, B, Q, R)
    K = future.result()

    print(f"   Gain shape: {K.shape}")

    # Cleanup
    cluster.close()
    print("\n→ Local cluster ideal for multi-core parallelism")
    print("→ No HPC cluster required")


def demo_2_parallel_computation():
    """Demo 2: Distribute computation across workers."""
    print("\n" + "="*70)
    print("DEMO 2: Distributed Computation")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⊘ Dask not available - showing code structure")
        print("""
# Distribute function evaluation
results = distribute_computation(
    func=solve_control_problem,
    inputs=initial_conditions,
    cluster=cluster
)
        """)
        return

    print("\nScenario: Parallel evaluation of control policies")

    def evaluate_control(x0):
        """Evaluate control starting from initial condition."""
        # Simulate trajectory
        x = x0.copy()
        cost = 0.0

        for _ in range(50):
            u = -0.5 * x  # Simple control law
            cost += np.sum(x**2) + np.sum(u**2)
            x = 0.9 * x + 0.1 * u + 0.01 * np.random.randn(*x.shape)

        return cost

    # Generate initial conditions
    np.random.seed(42)
    initial_conditions = [np.random.randn(3) for _ in range(20)]

    # Serial execution
    print("\n1. Serial execution:")
    start_serial = time.time()
    results_serial = [evaluate_control(x0) for x0 in initial_conditions]
    time_serial = time.time() - start_serial
    print(f"   Time: {time_serial:.2f}s")

    # Parallel execution
    print("\n2. Parallel execution:")
    cluster = create_local_cluster(n_workers=4)
    start_parallel = time.time()
    results_parallel = distribute_computation(
        evaluate_control,
        initial_conditions,
        cluster=cluster
    )
    time_parallel = time.time() - start_parallel
    cluster.close()

    print(f"   Time: {time_parallel:.2f}s")
    print(f"   Speedup: {time_serial / time_parallel:.2f}x")

    # Verify
    assert len(results_parallel) == len(results_serial)

    print("\n→ Distribute_computation handles task distribution")
    print("→ Automatic load balancing across workers")


def demo_3_hyperparameter_optimization():
    """Demo 3: Distributed hyperparameter optimization."""
    print("\n" + "="*70)
    print("DEMO 3: Distributed Hyperparameter Optimization")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⊘ Dask not available - showing code structure")
        print("""
# Optimize hyperparameters in parallel
best_params, best_value = distributed_optimization(
    objective=train_and_evaluate,
    parameter_ranges={
        'learning_rate': (1e-4, 1e-1),
        'hidden_size': (32, 256),
        'batch_size': (16, 128)
    },
    n_samples=100,
    cluster=cluster,
    method='latin'  # Latin hypercube sampling
)
        """)
        return

    print("\nScenario: Tune control network hyperparameters")

    # Objective function
    def train_controller(params):
        """Train and evaluate controller with given hyperparameters."""
        lr = params['learning_rate']
        hidden = int(params['hidden_size'])

        # Simulate training
        time.sleep(0.05)

        # Synthetic loss (minimize)
        # Optimal: lr=0.01, hidden=64
        loss = (np.log10(lr) + 2)**2 + (hidden - 64)**2 / 1000

        return loss

    # Define parameter ranges
    parameter_ranges = {
        'learning_rate': (1e-4, 1e-1),
        'hidden_size': (32, 128)
    }

    print("\nParameter ranges:")
    for name, (low, high) in parameter_ranges.items():
        print(f"  {name}: [{low}, {high}]")

    # Optimize
    print("\nOptimizing (50 samples)...")
    cluster = create_local_cluster(n_workers=4)
    best_params, best_value = distributed_optimization(
        train_controller,
        parameter_ranges,
        n_samples=50,
        cluster=cluster,
        method="random"
    )
    cluster.close()

    print(f"\nBest parameters:")
    print(f"  learning_rate: {best_params['learning_rate']:.6f}")
    print(f"  hidden_size: {best_params['hidden_size']:.1f}")
    print(f"  loss: {best_value:.4f}")

    print("\n→ Parallel hyperparameter search")
    print("→ Supports random, grid, and Latin hypercube sampling")


def demo_4_pipeline():
    """Demo 4: Data processing pipeline."""
    print("\n" + "="*70)
    print("DEMO 4: Data Processing Pipeline")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⊘ Dask not available - showing code structure")
        print("""
# Define pipeline stages
stages = [
    preprocess_data,
    extract_features,
    train_model,
    evaluate_model
]

result = pipeline(
    stages=stages,
    initial_data=raw_data,
    cluster=cluster,
    persist_intermediate=True
)
        """)
        return

    print("\nScenario: Multi-stage control system analysis")

    # Define pipeline stages
    def stage1_load_data(data):
        """Load and parse trajectory data."""
        print("  Stage 1: Loading data...")
        time.sleep(0.2)
        return {'trajectories': data, 'count': len(data)}

    def stage2_compute_features(data):
        """Compute features from trajectories."""
        print("  Stage 2: Computing features...")
        time.sleep(0.2)
        trajectories = data['trajectories']
        features = [np.mean(traj) for traj in trajectories]
        return {'features': features, 'count': len(features)}

    def stage3_aggregate(data):
        """Aggregate results."""
        print("  Stage 3: Aggregating...")
        time.sleep(0.2)
        return {
            'mean_feature': np.mean(data['features']),
            'std_feature': np.std(data['features']),
            'count': data['count']
        }

    stages = [stage1_load_data, stage2_compute_features, stage3_aggregate]

    # Initial data
    np.random.seed(42)
    trajectories = [np.random.randn(100) for _ in range(10)]

    print("\nExecuting pipeline:")
    cluster = create_local_cluster(n_workers=2)
    result = pipeline(
        stages,
        trajectories,
        cluster=cluster,
        persist_intermediate=False
    )
    cluster.close()

    print(f"\nPipeline result:")
    print(f"  Mean feature: {result['mean_feature']:.4f}")
    print(f"  Std feature: {result['std_feature']:.4f}")
    print(f"  Count: {result['count']}")

    print("\n→ Pipeline chains operations with Dask delayed")
    print("→ Optional intermediate persistence for fault tolerance")


def demo_5_map_reduce():
    """Demo 5: MapReduce pattern."""
    print("\n" + "="*70)
    print("DEMO 5: MapReduce Pattern")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⊘ Dask not available - showing code structure")
        print("""
# MapReduce pattern
result = scatter_gather_reduction(
    data=large_dataset,
    map_fn=process_batch,
    reduce_fn=aggregate_results,
    cluster=cluster
)
        """)
        return

    print("\nScenario: Compute aggregate statistics over distributed data")

    # Map function: compute statistics for each batch
    def map_statistics(batch):
        """Compute batch statistics."""
        return {
            'sum': np.sum(batch),
            'sum_sq': np.sum(batch ** 2),
            'count': len(batch)
        }

    # Reduce function: aggregate batch statistics
    def reduce_statistics(results):
        """Aggregate batch statistics into global statistics."""
        total_sum = sum(r['sum'] for r in results)
        total_sum_sq = sum(r['sum_sq'] for r in results)
        total_count = sum(r['count'] for r in results)

        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - mean ** 2

        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'count': total_count
        }

    # Generate data batches
    np.random.seed(42)
    data_batches = [np.random.randn(1000) + i*0.1 for i in range(10)]

    print(f"\nData: {len(data_batches)} batches, {sum(len(b) for b in data_batches)} total points")

    # MapReduce
    print("\nExecuting MapReduce...")
    cluster = create_local_cluster(n_workers=4)
    result = scatter_gather_reduction(
        data_batches,
        map_statistics,
        reduce_statistics,
        cluster=cluster
    )
    cluster.close()

    print(f"\nGlobal statistics:")
    print(f"  Mean: {result['mean']:.4f}")
    print(f"  Std: {result['std']:.4f}")
    print(f"  Count: {result['count']}")

    print("\n→ Scatter distributes data to workers")
    print("→ Map operates in parallel")
    print("→ Reduce aggregates results")


def demo_6_fault_tolerance():
    """Demo 6: Fault-tolerant computation."""
    print("\n" + "="*70)
    print("DEMO 6: Fault-Tolerant Computation")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⊘ Dask not available - showing code structure")
        print("""
# Fault-tolerant map with automatic retries
results, failures = fault_tolerant_map(
    func=solve_control_problem,
    inputs=problem_instances,
    cluster=cluster,
    max_retries=3
)

# Handle failures
if failures:
    for idx, error in failures:
        print(f"Failed on instance {idx}: {error}")
        """)
        return

    print("\nScenario: Robust computation with automatic retry")

    # Flaky function that sometimes fails
    call_count = [0]

    def flaky_solve(x):
        """Control solver that occasionally fails."""
        call_count[0] += 1

        # Fail 30% of the time on first attempt
        if call_count[0] % 10 < 3:
            raise RuntimeError("Simulated solver failure")

        # Otherwise succeed
        time.sleep(0.05)
        return x ** 2

    # Problem instances
    inputs = list(range(20))

    print(f"\nSolving {len(inputs)} problems (with simulated failures)...")

    cluster = create_local_cluster(n_workers=4)
    results, failures = fault_tolerant_map(
        flaky_solve,
        inputs,
        cluster=cluster,
        max_retries=3
    )
    cluster.close()

    print(f"\nResults:")
    print(f"  Successful: {sum(r is not None for r in results)}/{len(inputs)}")
    print(f"  Failed: {len(failures)}")
    print(f"  Total function calls: {call_count[0]}")

    if failures:
        print(f"\nFailures:")
        for idx, error in failures[:3]:
            print(f"  [{idx}]: {error}")

    print("\n→ Automatic retry on failure")
    print("→ Graceful degradation - returns partial results")
    print("→ Critical for long-running computations")


def demo_7_complete_workflow():
    """Demo 7: Complete distributed optimal control workflow."""
    print("\n" + "="*70)
    print("DEMO 7: Complete Distributed Workflow")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⊘ Dask not available - showing code structure")
        print("""
# Complete workflow example
cluster = create_local_cluster(n_workers=8)

# 1. Distribute parameter sweep
results = distribute_computation(evaluate_params, param_sets, cluster)

# 2. Find best parameters
best_params = param_sets[np.argmin(results)]

# 3. Train final model with best params
final_model = train_model(best_params, cluster)

# 4. Evaluate on test set
test_results = distribute_computation(
    lambda x: evaluate(final_model, x),
    test_data,
    cluster
)

cluster.close()
        """)
        return

    print("\nScenario: End-to-end neural control system design")

    # Simulate complete workflow
    print("\n" + "-"*70)
    print("WORKFLOW EXECUTION")
    print("-"*70)

    # Step 1: Create cluster
    print("\n1. Creating Dask cluster (8 workers)...")
    cluster = create_local_cluster(n_workers=8, threads_per_worker=1)
    print("   ✓ Cluster ready")

    # Step 2: Parameter sweep
    print("\n2. Running parameter sweep...")

    def evaluate_controller(params):
        """Evaluate controller with given parameters."""
        time.sleep(0.1)  # Simulate evaluation
        # Optimal at learning_rate=0.01
        lr = params['learning_rate']
        cost = (np.log10(lr) + 2)**2 + np.random.rand() * 0.1
        return cost

    # Generate parameter sets
    np.random.seed(42)
    param_sets = [
        {'learning_rate': 10 ** np.random.uniform(-4, -1)}
        for _ in range(32)
    ]

    results = distribute_computation(
        evaluate_controller,
        param_sets,
        cluster=cluster
    )

    best_idx = np.argmin(results)
    best_params = param_sets[best_idx]
    best_cost = results[best_idx]

    print(f"   Evaluated {len(param_sets)} parameter sets")
    print(f"   Best learning rate: {best_params['learning_rate']:.6f}")
    print(f"   Best cost: {best_cost:.4f}")

    # Step 3: Train final controller
    print("\n3. Training final controller...")

    def train_final_controller(params):
        """Train controller with best parameters."""
        time.sleep(0.5)
        return {
            'params': params,
            'weights': np.random.randn(10, 5),
            'performance': 0.95
        }

    future = cluster.submit(train_final_controller, best_params)
    final_controller = future.result()

    print(f"   ✓ Training complete")
    print(f"   Performance: {final_controller['performance']:.2%}")

    # Step 4: Distributed evaluation
    print("\n4. Evaluating on test set...")

    def evaluate_on_instance(test_case):
        """Evaluate controller on test instance."""
        time.sleep(0.05)
        # Simulate evaluation
        return np.random.rand() < final_controller['performance']

    test_cases = list(range(100))
    test_results = distribute_computation(
        evaluate_on_instance,
        test_cases,
        cluster=cluster
    )

    success_rate = sum(test_results) / len(test_results)
    print(f"   Test cases: {len(test_cases)}")
    print(f"   Success rate: {success_rate:.2%}")

    # Cleanup
    print("\n5. Cleanup...")
    cluster.close()
    print("   ✓ Cluster closed")

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)

    print("\n→ Complete end-to-end workflow")
    print("→ Parameter sweep → training → evaluation")
    print("→ All steps distributed across workers")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*15 + "DASK DISTRIBUTED COMPUTING DEMOS")
    print("="*70)

    if not DASK_AVAILABLE:
        print("\n⚠ NOTE: Dask is not installed")
        print("Install with: pip install dask[complete] distributed")
        print("\nDemos will show code structure only.")
        print("="*70)

    # Run demos
    demo_1_local_cluster()
    demo_2_parallel_computation()
    demo_3_hyperparameter_optimization()
    demo_4_pipeline()
    demo_5_map_reduce()
    demo_6_fault_tolerance()
    demo_7_complete_workflow()

    print("\n" + "="*70)
    if DASK_AVAILABLE:
        print("All demonstrations complete!")
    else:
        print("All code structures shown!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
