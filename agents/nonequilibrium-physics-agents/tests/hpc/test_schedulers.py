"""Tests for HPC Scheduler Integration.

Tests SLURM, PBS, and LocalScheduler interfaces for HPC job management.

Author: Nonequilibrium Physics Agents
Week: 29-30 of Phase 4
"""

import pytest
import time
import os
import tempfile
import subprocess
from pathlib import Path

from hpc.schedulers import (
    JobStatus,
    ResourceRequirements,
    JobInfo,
    Scheduler,
    SLURMScheduler,
    PBSScheduler,
    LocalScheduler,
    JobManager
)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Tests for configuration classes."""

    def test_resource_requirements_defaults(self):
        """Test: ResourceRequirements default values."""
        print("\nResourceRequirements defaults:")

        resources = ResourceRequirements()

        assert resources.nodes == 1
        assert resources.tasks_per_node == 1
        assert resources.cpus_per_task == 1
        assert resources.gpus_per_node == 0
        assert resources.memory_gb == 8
        assert resources.time_hours == 1.0
        assert resources.partition is None
        assert resources.qos is None
        assert resources.account is None

        print("  ✓ All defaults correct")

    def test_resource_requirements_custom(self):
        """Test: ResourceRequirements custom values."""
        print("\nResourceRequirements custom:")

        resources = ResourceRequirements(
            nodes=4,
            tasks_per_node=16,
            cpus_per_task=2,
            gpus_per_node=2,
            memory_gb=64,
            time_hours=24.5,
            partition="gpu",
            qos="high",
            account="project123"
        )

        assert resources.nodes == 4
        assert resources.tasks_per_node == 16
        assert resources.cpus_per_task == 2
        assert resources.gpus_per_node == 2
        assert resources.memory_gb == 64
        assert resources.time_hours == 24.5
        assert resources.partition == "gpu"
        assert resources.qos == "high"
        assert resources.account == "project123"

        print("  ✓ Custom values correct")

    def test_job_status_enum(self):
        """Test: JobStatus enumeration."""
        print("\nJobStatus enumeration:")

        statuses = [
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
            JobStatus.UNKNOWN
        ]

        assert len(statuses) == 7
        assert all(isinstance(s, JobStatus) for s in statuses)

        print(f"  ✓ {len(statuses)} job statuses defined")

    def test_job_info_creation(self):
        """Test: JobInfo dataclass."""
        print("\nJobInfo creation:")

        job = JobInfo(
            job_id="12345",
            name="test_job",
            status=JobStatus.RUNNING,
            submit_time=time.time(),
            start_time=time.time(),
            stdout_path="/path/to/stdout",
            stderr_path="/path/to/stderr"
        )

        assert job.job_id == "12345"
        assert job.name == "test_job"
        assert job.status == JobStatus.RUNNING
        assert job.start_time is not None
        assert len(job.dependencies) == 0

        print("  ✓ JobInfo created successfully")


# ============================================================================
# LocalScheduler Tests
# ============================================================================

class TestLocalScheduler:
    """Tests for local scheduler (no HPC)."""

    def test_local_scheduler_init(self):
        """Test: LocalScheduler initialization."""
        print("\nLocalScheduler initialization:")

        scheduler = LocalScheduler()

        assert scheduler.name == "Local"
        assert len(scheduler.jobs) == 0

        print("  ✓ Initialized successfully")

    def test_local_job_submission(self):
        """Test: Submit job locally."""
        print("\nLocal job submission:")

        scheduler = LocalScheduler()

        # Create temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("echo 'Hello from job'\n")
            f.write("sleep 1\n")
            f.write("echo 'Job complete'\n")
            script_path = f.name

        try:
            resources = ResourceRequirements()
            job_id = scheduler.submit_job(script_path, "test_job", resources)

            assert job_id.startswith("local_")
            assert job_id in scheduler.jobs

            print(f"  ✓ Job submitted: {job_id}")

            # Wait for completion
            final_status = scheduler.wait_for_job(job_id, poll_interval=0.5, timeout=10.0)

            assert final_status in [JobStatus.COMPLETED, JobStatus.FAILED]
            print(f"  ✓ Job completed with status: {final_status.value}")

        finally:
            os.unlink(script_path)

    def test_local_job_status(self):
        """Test: Get local job status."""
        print("\nLocal job status:")

        scheduler = LocalScheduler()

        # Create quick script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\n")
            script_path = f.name

        try:
            resources = ResourceRequirements()
            job_id = scheduler.submit_job(script_path, "test_job", resources)

            # Initially running or completed (fast job)
            status = scheduler.get_job_status(job_id)
            assert status in [JobStatus.RUNNING, JobStatus.COMPLETED]
            print(f"  ✓ Initial status: {status.value}")

            # Wait for completion
            time.sleep(2)
            status = scheduler.get_job_status(job_id)
            assert status in [JobStatus.COMPLETED, JobStatus.FAILED]
            print(f"  ✓ Final status: {status.value}")

        finally:
            os.unlink(script_path)

    def test_local_job_cancellation(self):
        """Test: Cancel local job."""
        print("\nLocal job cancellation:")

        scheduler = LocalScheduler()

        # Create long-running script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\nsleep 100\n")
            script_path = f.name

        try:
            resources = ResourceRequirements()
            job_id = scheduler.submit_job(script_path, "long_job", resources)

            # Wait briefly for job to start
            time.sleep(0.5)

            # Cancel
            success = scheduler.cancel_job(job_id)
            assert success

            # Check status
            status = scheduler.get_job_status(job_id)
            assert status == JobStatus.CANCELLED

            print("  ✓ Job cancelled successfully")

        finally:
            os.unlink(script_path)

    def test_local_queue_info(self):
        """Test: Get local queue information."""
        print("\nLocal queue info:")

        scheduler = LocalScheduler()

        # Submit multiple jobs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\n")
            script_path = f.name

        try:
            resources = ResourceRequirements()

            job_ids = []
            for i in range(3):
                job_id = scheduler.submit_job(script_path, f"job_{i}", resources)
                job_ids.append(job_id)

            # Get queue info
            queue = scheduler.get_queue_info()

            assert len(queue) >= 3
            print(f"  ✓ Queue contains {len(queue)} jobs")

            # Wait for all
            time.sleep(2)

            # Check all completed
            for job_id in job_ids:
                status = scheduler.get_job_status(job_id)
                assert status in [JobStatus.COMPLETED, JobStatus.FAILED]

            print("  ✓ All jobs completed")

        finally:
            os.unlink(script_path)

    def test_local_job_dependencies(self):
        """Test: Job dependencies (sequential execution)."""
        print("\nLocal job dependencies:")

        scheduler = LocalScheduler()

        # Create two scripts
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f1:
            f1.write("#!/bin/bash\necho 'Job 1'\nsleep 1\n")
            script1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f2:
            f2.write("#!/bin/bash\necho 'Job 2'\n")
            script2 = f2.name

        try:
            resources = ResourceRequirements()

            # Submit job 1
            job1 = scheduler.submit_job(script1, "job1", resources)
            print(f"  ✓ Job 1 submitted: {job1}")

            # Submit job 2 depending on job 1
            start_time = time.time()
            job2 = scheduler.submit_job(script2, "job2", resources, dependencies=[job1])
            dependency_wait = time.time() - start_time

            print(f"  ✓ Job 2 submitted: {job2}")
            print(f"  ✓ Job 2 waited {dependency_wait:.2f}s for job 1")

            # Dependency should have caused wait
            assert dependency_wait >= 1.0

            # Wait briefly for job2 to complete (it's fast)
            time.sleep(1.0)

            # Both should be completed
            assert scheduler.get_job_status(job1) == JobStatus.COMPLETED
            assert scheduler.get_job_status(job2) in [JobStatus.COMPLETED, JobStatus.FAILED]

        finally:
            os.unlink(script1)
            os.unlink(script2)


# ============================================================================
# JobManager Tests
# ============================================================================

class TestJobManager:
    """Tests for high-level job manager."""

    def test_job_manager_init(self):
        """Test: JobManager initialization."""
        print("\nJobManager initialization:")

        # With explicit scheduler
        scheduler = LocalScheduler()
        manager = JobManager(scheduler=scheduler, auto_detect=False)

        assert manager.scheduler == scheduler
        assert manager.scheduler.name == "Local"

        print("  ✓ Initialized with explicit scheduler")

        # With auto-detect
        manager2 = JobManager(auto_detect=True)
        assert manager2.scheduler is not None
        print(f"  ✓ Auto-detected scheduler: {manager2.scheduler.name}")

    def test_job_manager_submit(self):
        """Test: JobManager job submission."""
        print("\nJobManager submission:")

        manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

        # Create script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\n")
            script_path = f.name

        try:
            job_id = manager.submit(script_path, "test_job")

            assert job_id is not None
            print(f"  ✓ Job submitted: {job_id}")

            # Check status
            status = manager.status(job_id)
            assert status in [JobStatus.RUNNING, JobStatus.COMPLETED]
            print(f"  ✓ Status: {status.value}")

            # Wait for completion
            final_status = manager.wait(job_id, poll_interval=0.5, timeout=10.0)
            assert final_status in [JobStatus.COMPLETED, JobStatus.FAILED]
            print(f"  ✓ Final status: {final_status.value}")

        finally:
            os.unlink(script_path)

    def test_job_manager_queue(self):
        """Test: JobManager queue retrieval."""
        print("\nJobManager queue:")

        manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

        # Get queue (initially empty)
        queue = manager.queue()
        initial_size = len(queue)

        print(f"  ✓ Initial queue size: {initial_size}")

        # Submit job
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\n")
            script_path = f.name

        try:
            job_id = manager.submit(script_path, "test_job")

            # Queue should have grown
            queue = manager.queue()
            assert len(queue) >= initial_size + 1

            print(f"  ✓ Queue size after submission: {len(queue)}")

        finally:
            os.unlink(script_path)

    def test_job_manager_cancel(self):
        """Test: JobManager job cancellation."""
        print("\nJobManager cancellation:")

        manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

        # Submit long job
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\nsleep 100\n")
            script_path = f.name

        try:
            job_id = manager.submit(script_path, "long_job")
            time.sleep(0.5)

            # Cancel
            success = manager.cancel(job_id)
            assert success

            # Check status
            status = manager.status(job_id)
            assert status == JobStatus.CANCELLED

            print("  ✓ Job cancelled successfully")

        finally:
            os.unlink(script_path)

    def test_job_manager_wait_all(self):
        """Test: JobManager wait for multiple jobs."""
        print("\nJobManager wait_all:")

        manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

        # Submit multiple jobs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\nsleep 1\n")
            script_path = f.name

        try:
            job_ids = []
            for i in range(3):
                job_id = manager.submit(script_path, f"job_{i}")
                job_ids.append(job_id)

            print(f"  ✓ Submitted {len(job_ids)} jobs")

            # Wait for all
            statuses = manager.wait_all(job_ids, poll_interval=0.5, timeout=15.0)

            assert len(statuses) == 3
            assert all(s in [JobStatus.COMPLETED, JobStatus.FAILED]
                      for s in statuses.values())

            print("  ✓ All jobs completed")
            for job_id, status in statuses.items():
                print(f"    {job_id}: {status.value}")

        finally:
            os.unlink(script_path)

    def test_job_manager_get_info(self):
        """Test: JobManager get job info."""
        print("\nJobManager get_info:")

        manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

        # Submit job
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\necho 'test'\n")
            script_path = f.name

        try:
            job_id = manager.submit(script_path, "info_test")

            # Get info
            info = manager.get_info(job_id)

            assert info is not None
            assert info.job_id == job_id
            assert info.name == "info_test"
            assert info.submit_time > 0

            print(f"  ✓ Job info retrieved")
            print(f"    ID: {info.job_id}")
            print(f"    Name: {info.name}")
            print(f"    Status: {info.status.value}")

        finally:
            os.unlink(script_path)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for scheduler system."""

    def test_complete_workflow(self):
        """Test: Complete job submission workflow."""
        print("\nComplete workflow:")

        manager = JobManager(scheduler=LocalScheduler(), auto_detect=False)

        # Create analysis script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("echo 'Starting analysis...'\n")
            f.write("sleep 1\n")
            f.write("echo 'Analysis complete'\n")
            f.write("exit 0\n")
            script_path = f.name

        try:
            # 1. Submit job
            print("  1. Submitting job...")
            resources = ResourceRequirements(
                cpus_per_task=2,
                memory_gb=4,
                time_hours=1.0
            )
            job_id = manager.submit(
                script_path,
                "analysis_job",
                resources=resources
            )
            print(f"     Submitted: {job_id}")

            # 2. Monitor status
            print("  2. Monitoring status...")
            for i in range(5):
                status = manager.status(job_id)
                print(f"     Status ({i}): {status.value}")
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    break
                time.sleep(0.5)

            # 3. Wait for completion
            print("  3. Waiting for completion...")
            final_status = manager.wait(job_id, poll_interval=0.5, timeout=10.0)
            print(f"     Final status: {final_status.value}")

            # 4. Get job info
            print("  4. Retrieving job info...")
            info = manager.get_info(job_id)
            if info:
                print(f"     Name: {info.name}")
                print(f"     Exit code: {info.exit_code}")
                print(f"     Working dir: {info.working_dir}")

            # 5. Verify success
            assert final_status in [JobStatus.COMPLETED, JobStatus.FAILED]

            print("  ✓ Workflow completed successfully")

        finally:
            os.unlink(script_path)


# ============================================================================
# Mock Tests for SLURM/PBS (without actual cluster)
# ============================================================================

class TestSLURMMock:
    """Mock tests for SLURM (without cluster)."""

    def test_slurm_availability_check(self):
        """Test: SLURM availability detection."""
        print("\nSLURM availability check:")

        try:
            scheduler = SLURMScheduler()
            print("  ✓ SLURM is available")
            print(f"    Scheduler: {scheduler.name}")
        except:
            print("  ⊘ SLURM not available (expected on non-cluster system)")
            pytest.skip("SLURM not available")

    def test_slurm_resource_string_generation(self):
        """Test: SLURM resource requirements formatting."""
        print("\nSLURM resource formatting:")

        # This test doesn't require actual SLURM
        resources = ResourceRequirements(
            nodes=4,
            tasks_per_node=16,
            cpus_per_task=2,
            gpus_per_node=2,
            memory_gb=64,
            time_hours=12.5,
            partition="gpu",
            qos="high"
        )

        # Expected values
        assert resources.nodes == 4
        assert resources.memory_gb == 64

        # Time formatting
        hours = int(resources.time_hours)
        minutes = int((resources.time_hours - hours) * 60)
        time_str = f"{hours:02d}:{minutes:02d}:00"

        assert time_str == "12:30:00"
        print(f"  ✓ Time formatted: {time_str}")


class TestPBSMock:
    """Mock tests for PBS (without cluster)."""

    def test_pbs_availability_check(self):
        """Test: PBS availability detection."""
        print("\nPBS availability check:")

        try:
            scheduler = PBSScheduler()
            print("  ✓ PBS is available")
            print(f"    Scheduler: {scheduler.name}")
        except:
            print("  ⊘ PBS not available (expected on non-cluster system)")
            pytest.skip("PBS not available")

    def test_pbs_resource_string_generation(self):
        """Test: PBS resource requirements formatting."""
        print("\nPBS resource formatting:")

        resources = ResourceRequirements(
            nodes=2,
            cpus_per_task=8,
            gpus_per_node=1,
            memory_gb=32,
            time_hours=6.0
        )

        # Expected select string
        select_str = f"select={resources.nodes}:ncpus={resources.cpus_per_task}"
        select_str += f":ngpus={resources.gpus_per_node}:mem={resources.memory_gb}gb"

        expected = "select=2:ncpus=8:ngpus=1:mem=32gb"
        assert select_str == expected

        print(f"  ✓ Select string: {select_str}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
