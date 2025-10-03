"""Demonstrations of HPC Scheduler Integration.

Shows practical applications of SLURM, PBS, and local schedulers for
submitting and managing optimal control computations on HPC clusters.

Author: Nonequilibrium Physics Agents
Week: 29-30 of Phase 4
"""

import sys
import time
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hpc.schedulers import (
    JobStatus,
    ResourceRequirements,
    JobInfo,
    SLURMScheduler,
    PBSScheduler,
    LocalScheduler,
    JobManager
)


def demo_1_local_scheduler():
    """Demo 1: Local scheduler for testing."""
    print("\n" + "="*70)
    print("DEMO 1: Local Scheduler (No HPC Required)")
    print("="*70)

    print("\nScenario: Test optimal control workflow locally")

    scheduler = LocalScheduler()
    print(f"\nScheduler: {scheduler.name}")

    # Create simple job script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Starting optimal control computation...'\n")
        f.write("echo 'Solving LQR problem...'\n")
        f.write("sleep 2\n")
        f.write("echo 'Solution found: K = [[-0.5, -1.2]]'\n")
        f.write("echo 'Computation complete'\n")
        script_path = f.name

    try:
        # Define resources (ignored for local, but good practice)
        resources = ResourceRequirements(
            cpus_per_task=4,
            memory_gb=8,
            time_hours=1.0
        )

        print("\n1. Submitting job...")
        job_id = scheduler.submit_job(script_path, "lqr_solve", resources)
        print(f"   Job ID: {job_id}")

        print("\n2. Monitoring status...")
        for i in range(10):
            status = scheduler.get_job_status(job_id)
            print(f"   [{i+1}] Status: {status.value}")

            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break

            time.sleep(0.5)

        print(f"\n3. Final status: {status.value}")

        # Get job info
        info = scheduler.jobs.get(job_id)
        if info:
            print(f"\nJob Information:")
            print(f"  Name: {info.name}")
            print(f"  Submit time: {info.submit_time:.2f}")
            print(f"  End time: {info.end_time:.2f}" if info.end_time else "  Still running")
            print(f"  Exit code: {info.exit_code}")

    finally:
        os.unlink(script_path)

    print("\n→ LocalScheduler executes jobs in background")
    print("→ Perfect for testing before HPC deployment")


def demo_2_resource_requirements():
    """Demo 2: Specifying resource requirements."""
    print("\n" + "="*70)
    print("DEMO 2: Resource Requirements Specification")
    print("="*70)

    print("\nScenario: Configure resources for different problem sizes")

    # Small problem
    print("\n1. Small problem (single node):")
    resources_small = ResourceRequirements(
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=4,
        gpus_per_node=0,
        memory_gb=16,
        time_hours=2.0,
        partition="short"
    )

    print(f"   Nodes: {resources_small.nodes}")
    print(f"   CPUs per task: {resources_small.cpus_per_task}")
    print(f"   Memory: {resources_small.memory_gb} GB")
    print(f"   Time limit: {resources_small.time_hours} hours")
    print(f"   Partition: {resources_small.partition}")

    # Medium problem with GPU
    print("\n2. Medium problem (GPU acceleration):")
    resources_medium = ResourceRequirements(
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,
        gpus_per_node=1,
        memory_gb=32,
        time_hours=12.0,
        partition="gpu",
        qos="high"
    )

    print(f"   CPUs: {resources_medium.cpus_per_task}")
    print(f"   GPUs: {resources_medium.gpus_per_node}")
    print(f"   Memory: {resources_medium.memory_gb} GB")
    print(f"   Time: {resources_medium.time_hours} hours")
    print(f"   Partition: {resources_medium.partition}")
    print(f"   QOS: {resources_medium.qos}")

    # Large problem (multi-node)
    print("\n3. Large problem (multi-node MPI):")
    resources_large = ResourceRequirements(
        nodes=16,
        tasks_per_node=32,
        cpus_per_task=2,
        gpus_per_node=0,
        memory_gb=128,
        time_hours=48.0,
        partition="long",
        account="project_ABC123"
    )

    print(f"   Nodes: {resources_large.nodes}")
    print(f"   Tasks per node: {resources_large.tasks_per_node}")
    print(f"   Total cores: {resources_large.nodes * resources_large.tasks_per_node * resources_large.cpus_per_task}")
    print(f"   Memory per node: {resources_large.memory_gb} GB")
    print(f"   Time: {resources_large.time_hours} hours")
    print(f"   Account: {resources_large.account}")

    print("\n→ Tailor resources to problem size")
    print("→ Consider CPU, GPU, memory, time requirements")
    print("→ Use appropriate partition and QOS")


def demo_3_job_manager():
    """Demo 3: High-level job manager interface."""
    print("\n" + "="*70)
    print("DEMO 3: JobManager Interface")
    print("="*70)

    print("\nScenario: Unified interface for any scheduler")

    # Auto-detect available scheduler
    manager = JobManager(auto_detect=True)
    print(f"\nAuto-detected scheduler: {manager.scheduler.name}")

    # Create job script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Computing optimal control...'\n")
        f.write("sleep 1\n")
        f.write("echo 'Done'\n")
        script_path = f.name

    try:
        print("\n1. Submit job:")
        resources = ResourceRequirements(cpus_per_task=2, memory_gb=4)
        job_id = manager.submit(script_path, "demo_job", resources=resources)
        print(f"   Job ID: {job_id}")

        print("\n2. Check status:")
        status = manager.status(job_id)
        print(f"   Status: {status.value}")

        print("\n3. Wait for completion:")
        final_status = manager.wait(job_id, poll_interval=0.5, timeout=10.0)
        print(f"   Final status: {final_status.value}")

        print("\n4. Get job info:")
        info = manager.get_info(job_id)
        if info:
            print(f"   Name: {info.name}")
            print(f"   Status: {info.status.value}")

        print("\n5. View queue:")
        queue = manager.queue()
        print(f"   Jobs in queue: {len(queue)}")

    finally:
        os.unlink(script_path)

    print("\n→ JobManager provides unified API")
    print("→ Works with SLURM, PBS, or local")
    print("→ Auto-detection simplifies deployment")


def demo_4_job_dependencies():
    """Demo 4: Job dependencies and workflows."""
    print("\n" + "="*70)
    print("DEMO 4: Job Dependencies and Workflows")
    print("="*70)

    print("\nScenario: Multi-stage optimal control pipeline")

    manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

    # Stage 1: Data preparation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Preparing training data...'\n")
        f.write("sleep 1\n")
        f.write("echo 'Data prepared'\n")
        script_prep = f.name

    # Stage 2: Model training (depends on stage 1)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Training neural network controller...'\n")
        f.write("sleep 1\n")
        f.write("echo 'Training complete'\n")
        script_train = f.name

    # Stage 3: Evaluation (depends on stage 2)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Evaluating controller performance...'\n")
        f.write("sleep 1\n")
        f.write("echo 'Evaluation complete'\n")
        script_eval = f.name

    try:
        resources = ResourceRequirements()

        print("\nWorkflow:")
        print("  prep → train → eval")

        print("\n1. Submit data preparation:")
        job_prep = manager.submit(script_prep, "prep", resources)
        print(f"   Job ID: {job_prep}")

        print("\n2. Submit training (depends on prep):")
        start_time = time.time()
        job_train = manager.scheduler.submit_job(
            script_train, "train", resources, dependencies=[job_prep]
        )
        train_wait = time.time() - start_time
        print(f"   Job ID: {job_train}")
        print(f"   Waited {train_wait:.2f}s for prep to complete")

        print("\n3. Submit evaluation (depends on train):")
        start_time = time.time()
        job_eval = manager.scheduler.submit_job(
            script_eval, "eval", resources, dependencies=[job_train]
        )
        eval_wait = time.time() - start_time
        print(f"   Job ID: {job_eval}")
        print(f"   Waited {eval_wait:.2f}s for train to complete")

        print("\n4. Final status:")
        time.sleep(1)
        print(f"   prep:  {manager.status(job_prep).value}")
        print(f"   train: {manager.status(job_train).value}")
        print(f"   eval:  {manager.status(job_eval).value}")

    finally:
        os.unlink(script_prep)
        os.unlink(script_train)
        os.unlink(script_eval)

    print("\n→ Dependencies ensure correct execution order")
    print("→ Build complex workflows from simple jobs")
    print("→ Automatic waiting for dependencies")


def demo_5_job_arrays():
    """Demo 5: Job arrays for parameter sweeps."""
    print("\n" + "="*70)
    print("DEMO 5: Job Arrays for Parameter Sweeps")
    print("="*70)

    print("\nScenario: Parameter sweep over different initial conditions")

    manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

    # Create array job script (uses $ARRAY_TASK_ID in practice)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("# In real HPC: TASK_ID=$SLURM_ARRAY_TASK_ID or $PBS_ARRAY_INDEX\n")
        f.write("echo 'Running task for parameter set'\n")
        f.write("sleep 0.5\n")
        f.write("echo 'Task complete'\n")
        script_path = f.name

    try:
        resources = ResourceRequirements(cpus_per_task=2, memory_gb=4)
        array_size = 5

        print(f"\n1. Submitting job array (size={array_size}):")
        job_ids = manager.submit_array(
            script_path,
            "param_sweep",
            array_size,
            resources=resources
        )

        print(f"   Submitted {len(job_ids)} jobs")
        for i, job_id in enumerate(job_ids):
            print(f"     [{i}] {job_id}")

        print("\n2. Waiting for all jobs to complete:")
        statuses = manager.wait_all(job_ids, poll_interval=0.5, timeout=15.0)

        completed = sum(1 for s in statuses.values() if s == JobStatus.COMPLETED)
        failed = sum(1 for s in statuses.values() if s == JobStatus.FAILED)

        print(f"   Completed: {completed}/{array_size}")
        print(f"   Failed: {failed}/{array_size}")

        print("\n3. Job statuses:")
        for job_id, status in statuses.items():
            print(f"   {job_id}: {status.value}")

    finally:
        os.unlink(script_path)

    print("\n→ Job arrays parallelize parameter sweeps")
    print("→ Each task gets unique ARRAY_TASK_ID")
    print("→ Efficient for embarrassingly parallel problems")


def demo_6_job_cancellation():
    """Demo 6: Job cancellation and error handling."""
    print("\n" + "="*70)
    print("DEMO 6: Job Cancellation and Error Handling")
    print("="*70)

    print("\nScenario: Cancel long-running job")

    manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

    # Create long job
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Starting long computation...'\n")
        f.write("sleep 60\n")  # Long job
        f.write("echo 'Done'\n")
        script_path = f.name

    try:
        resources = ResourceRequirements()

        print("\n1. Submit long job:")
        job_id = manager.submit(script_path, "long_job", resources)
        print(f"   Job ID: {job_id}")

        # Let it start
        time.sleep(0.5)

        print("\n2. Check status:")
        status = manager.status(job_id)
        print(f"   Status: {status.value}")

        print("\n3. Cancel job:")
        success = manager.cancel(job_id)
        print(f"   Cancelled: {success}")

        # Check final status
        time.sleep(0.5)
        status = manager.status(job_id)
        print(f"   Status after cancellation: {status.value}")

    finally:
        os.unlink(script_path)

    print("\n→ Cancel jobs that are no longer needed")
    print("→ Frees up cluster resources")
    print("→ Important for interactive workflows")


def demo_7_complete_workflow():
    """Demo 7: Complete HPC optimal control workflow."""
    print("\n" + "="*70)
    print("DEMO 7: Complete HPC Workflow")
    print("="*70)

    print("\nScenario: End-to-end optimal control computation")

    manager = JobManager(auto_detect=True)
    print(f"\nScheduler: {manager.scheduler.name}")

    # Create main computation script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write("#\n")
        f.write("# Optimal Control Computation\n")
        f.write("#\n")
        f.write("\n")
        f.write("echo '======================================'\n")
        f.write("echo 'Optimal Control HPC Computation'\n")
        f.write("echo '======================================'\n")
        f.write("\n")
        f.write("echo 'Step 1: Load problem specification'\n")
        f.write("sleep 0.5\n")
        f.write("echo '  - System: 10D nonlinear'\n")
        f.write("echo '  - Horizon: 100 steps'\n")
        f.write("echo '  - Method: Neural OC'\n")
        f.write("\n")
        f.write("echo 'Step 2: Initialize neural network'\n")
        f.write("sleep 0.5\n")
        f.write("echo '  - Architecture: [10, 64, 64, 10]'\n")
        f.write("echo '  - Parameters: 5,130'\n")
        f.write("\n")
        f.write("echo 'Step 3: Training'\n")
        f.write("for i in {1..5}; do\n")
        f.write("  echo \"  Epoch $i/5: loss = $(echo \"scale=4; 1.0 / $i\" | bc)\"\n")
        f.write("  sleep 0.2\n")
        f.write("done\n")
        f.write("\n")
        f.write("echo 'Step 4: Evaluation'\n")
        f.write("sleep 0.5\n")
        f.write("echo '  - Final cost: 2.45'\n")
        f.write("echo '  - Constraint violation: 0.001'\n")
        f.write("\n")
        f.write("echo '======================================'\n")
        f.write("echo 'Computation Complete!'\n")
        f.write("echo '======================================'\n")
        script_path = f.name

    try:
        # Configure resources
        resources = ResourceRequirements(
            nodes=1,
            cpus_per_task=8,
            gpus_per_node=1,
            memory_gb=32,
            time_hours=4.0,
            partition="gpu" if manager.scheduler.name != "Local" else None
        )

        print("\nResource Configuration:")
        print(f"  Nodes: {resources.nodes}")
        print(f"  CPUs per task: {resources.cpus_per_task}")
        print(f"  GPUs: {resources.gpus_per_node}")
        print(f"  Memory: {resources.memory_gb} GB")
        print(f"  Time limit: {resources.time_hours} hours")

        print("\n" + "-"*70)
        print("EXECUTION")
        print("-"*70)

        # Submit
        print("\n1. Submitting job...")
        job_id = manager.submit(
            script_path,
            "optimal_control_computation",
            resources=resources
        )
        print(f"   Job ID: {job_id}")

        # Monitor
        print("\n2. Monitoring progress...")
        start_time = time.time()

        while True:
            status = manager.status(job_id)
            elapsed = time.time() - start_time

            print(f"   [{elapsed:.1f}s] Status: {status.value}")

            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                break

            time.sleep(1.0)

        # Results
        print("\n3. Job completed!")
        info = manager.get_info(job_id)
        if info:
            total_time = info.end_time - info.submit_time if info.end_time else 0
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Exit code: {info.exit_code}")

            # Show output
            if info.stdout_path and os.path.exists(info.stdout_path):
                print(f"\n4. Output ({info.stdout_path}):")
                with open(info.stdout_path, 'r') as f:
                    output = f.read()
                    for line in output.split('\n'):
                        if line.strip():
                            print(f"   {line}")

        print("\n" + "="*70)
        print("WORKFLOW COMPLETE")
        print("="*70)

    finally:
        os.unlink(script_path)

    print("\n→ Complete workflow: submit → monitor → results")
    print("→ Resource specification ensures proper allocation")
    print("→ Works identically on local or HPC cluster")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*15 + "HPC SCHEDULER INTEGRATION DEMOS")
    print("="*70)

    # Run demos
    demo_1_local_scheduler()
    demo_2_resource_requirements()
    demo_3_job_manager()
    demo_4_job_dependencies()
    demo_5_job_arrays()
    demo_6_job_cancellation()
    demo_7_complete_workflow()

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
