"""HPC Scheduler Integration for Optimal Control.

Provides interfaces for SLURM and PBS job schedulers to enable
large-scale optimal control computations on HPC clusters.

Features:
- Unified scheduler interface (SLURM, PBS, local)
- Job submission and monitoring
- Resource management (CPU, GPU, memory, time)
- Queue status and job arrays
- Automatic retry and error handling
- Job dependency management

Author: Nonequilibrium Physics Agents
Week: 29-30 of Phase 4
"""

import subprocess
import time
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ResourceRequirements:
    """Resource requirements for HPC job."""
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 1
    gpus_per_node: int = 0
    memory_gb: int = 8
    time_hours: float = 1.0
    partition: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None
    constraint: Optional[str] = None


@dataclass
class JobInfo:
    """Information about submitted job."""
    job_id: str
    name: str
    status: JobStatus
    submit_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    working_dir: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class Scheduler(ABC):
    """Abstract base class for HPC schedulers."""

    def __init__(self, name: str):
        """Initialize scheduler.

        Args:
            name: Scheduler name
        """
        self.name = name
        self.jobs: Dict[str, JobInfo] = {}

    @abstractmethod
    def submit_job(
        self,
        script_path: str,
        job_name: str,
        resources: ResourceRequirements,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Submit job to scheduler.

        Args:
            script_path: Path to job script
            job_name: Job name
            resources: Resource requirements
            dependencies: List of job IDs to depend on
            **kwargs: Additional scheduler-specific options

        Returns:
            job_id: Submitted job ID
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of job.

        Args:
            job_id: Job ID

        Returns:
            status: Job status
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel job.

        Args:
            job_id: Job ID

        Returns:
            success: True if cancelled successfully
        """
        pass

    @abstractmethod
    def get_queue_info(self) -> List[JobInfo]:
        """Get information about all jobs in queue.

        Returns:
            jobs: List of job info
        """
        pass

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None
    ) -> JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Job ID
            poll_interval: Polling interval in seconds
            timeout: Timeout in seconds (None = no timeout)

        Returns:
            status: Final job status
        """
        start_time = time.time()

        while True:
            status = self.get_job_status(job_id)

            if status in [JobStatus.COMPLETED, JobStatus.FAILED,
                         JobStatus.CANCELLED, JobStatus.TIMEOUT]:
                return status

            if timeout and (time.time() - start_time) > timeout:
                return JobStatus.TIMEOUT

            time.sleep(poll_interval)

    def submit_job_array(
        self,
        script_path: str,
        job_name: str,
        resources: ResourceRequirements,
        array_size: int,
        **kwargs
    ) -> List[str]:
        """Submit job array.

        Args:
            script_path: Path to job script
            job_name: Job name
            resources: Resource requirements
            array_size: Number of jobs in array
            **kwargs: Additional options

        Returns:
            job_ids: List of submitted job IDs
        """
        # Default implementation: submit individual jobs
        job_ids = []
        for i in range(array_size):
            name = f"{job_name}_{i}"
            job_id = self.submit_job(script_path, name, resources, **kwargs)
            job_ids.append(job_id)
        return job_ids


class SLURMScheduler(Scheduler):
    """SLURM workload manager interface."""

    def __init__(self):
        """Initialize SLURM scheduler."""
        super().__init__("SLURM")
        self._check_availability()

    def _check_availability(self) -> bool:
        """Check if SLURM is available."""
        try:
            subprocess.run(["sinfo", "--version"],
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def submit_job(
        self,
        script_path: str,
        job_name: str,
        resources: ResourceRequirements,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Submit job to SLURM.

        Args:
            script_path: Path to job script
            job_name: Job name
            resources: Resource requirements
            dependencies: List of job IDs to depend on
            **kwargs: Additional SLURM options

        Returns:
            job_id: SLURM job ID
        """
        # Build sbatch command
        cmd = ["sbatch"]

        # Job name
        cmd.extend(["--job-name", job_name])

        # Resources
        cmd.extend(["--nodes", str(resources.nodes)])
        cmd.extend(["--ntasks-per-node", str(resources.tasks_per_node)])
        cmd.extend(["--cpus-per-task", str(resources.cpus_per_task)])

        if resources.gpus_per_node > 0:
            cmd.extend(["--gres", f"gpu:{resources.gpus_per_node}"])

        cmd.extend(["--mem", f"{resources.memory_gb}G"])

        # Time limit
        hours = int(resources.time_hours)
        minutes = int((resources.time_hours - hours) * 60)
        cmd.extend(["--time", f"{hours:02d}:{minutes:02d}:00"])

        # Partition/QOS/Account
        if resources.partition:
            cmd.extend(["--partition", resources.partition])
        if resources.qos:
            cmd.extend(["--qos", resources.qos])
        if resources.account:
            cmd.extend(["--account", resources.account])
        if resources.constraint:
            cmd.extend(["--constraint", resources.constraint])

        # Dependencies
        if dependencies:
            dep_str = ":".join(dependencies)
            cmd.extend(["--dependency", f"afterok:{dep_str}"])

        # Output files
        stdout_path = kwargs.get("stdout", f"{job_name}_%j.out")
        stderr_path = kwargs.get("stderr", f"{job_name}_%j.err")
        cmd.extend(["--output", stdout_path])
        cmd.extend(["--error", stderr_path])

        # Additional options
        for key, value in kwargs.items():
            if key not in ["stdout", "stderr"]:
                cmd.extend([f"--{key}", str(value)])

        # Script path
        cmd.append(script_path)

        # Submit
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse job ID from output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Failed to parse job ID from: {result.stdout}")

        job_id = match.group(1)

        # Store job info
        self.jobs[job_id] = JobInfo(
            job_id=job_id,
            name=job_name,
            status=JobStatus.PENDING,
            submit_time=time.time(),
            stdout_path=stdout_path.replace("%j", job_id),
            stderr_path=stderr_path.replace("%j", job_id),
            working_dir=os.getcwd(),
            dependencies=dependencies or []
        )

        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of SLURM job.

        Args:
            job_id: SLURM job ID

        Returns:
            status: Job status
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True, text=True, check=True
            )

            state = result.stdout.strip().upper()

            # Map SLURM states to JobStatus
            status_map = {
                "PENDING": JobStatus.PENDING,
                "RUNNING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "CANCELLED": JobStatus.CANCELLED,
                "TIMEOUT": JobStatus.TIMEOUT,
                "COMPLETING": JobStatus.RUNNING,
                "CONFIGURING": JobStatus.PENDING,
                "RESIZING": JobStatus.RUNNING,
            }

            status = status_map.get(state, JobStatus.UNKNOWN)

            # Update stored job info
            if job_id in self.jobs:
                self.jobs[job_id].status = status

            return status

        except subprocess.CalledProcessError:
            # Job not found in queue, check sacct
            try:
                result = subprocess.run(
                    ["sacct", "-j", job_id, "-n", "-o", "State"],
                    capture_output=True, text=True, check=True
                )

                state = result.stdout.strip().split()[0].upper()

                status_map = {
                    "COMPLETED": JobStatus.COMPLETED,
                    "FAILED": JobStatus.FAILED,
                    "CANCELLED": JobStatus.CANCELLED,
                    "TIMEOUT": JobStatus.TIMEOUT,
                }

                return status_map.get(state, JobStatus.UNKNOWN)

            except subprocess.CalledProcessError:
                return JobStatus.UNKNOWN

    def cancel_job(self, job_id: str) -> bool:
        """Cancel SLURM job.

        Args:
            job_id: SLURM job ID

        Returns:
            success: True if cancelled successfully
        """
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True, check=True
            )

            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.CANCELLED

            return True
        except subprocess.CalledProcessError:
            return False

    def get_queue_info(self) -> List[JobInfo]:
        """Get information about all jobs in SLURM queue.

        Returns:
            jobs: List of job info
        """
        try:
            result = subprocess.run(
                ["squeue", "-u", os.getenv("USER", ""), "-o",
                 "%i|%j|%T|%V|%S|%e"],
                capture_output=True, text=True, check=True
            )

            jobs = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) < 6:
                    continue

                job_id, name, state, submit_time, start_time, end_time = parts

                status_map = {
                    "PENDING": JobStatus.PENDING,
                    "RUNNING": JobStatus.RUNNING,
                    "COMPLETED": JobStatus.COMPLETED,
                    "FAILED": JobStatus.FAILED,
                    "CANCELLED": JobStatus.CANCELLED,
                }

                status = status_map.get(state, JobStatus.UNKNOWN)

                job = JobInfo(
                    job_id=job_id,
                    name=name,
                    status=status,
                    submit_time=float(submit_time) if submit_time != "N/A" else time.time(),
                    start_time=float(start_time) if start_time != "N/A" else None,
                    end_time=float(end_time) if end_time != "N/A" else None
                )
                jobs.append(job)

            return jobs

        except subprocess.CalledProcessError:
            return []

    def submit_job_array(
        self,
        script_path: str,
        job_name: str,
        resources: ResourceRequirements,
        array_size: int,
        **kwargs
    ) -> List[str]:
        """Submit SLURM job array.

        Args:
            script_path: Path to job script
            job_name: Job name
            resources: Resource requirements
            array_size: Number of jobs in array
            **kwargs: Additional options

        Returns:
            job_ids: List of job IDs (format: "12345_0", "12345_1", ...)
        """
        # Add array option
        kwargs["array"] = f"0-{array_size-1}"

        # Submit
        base_job_id = self.submit_job(script_path, job_name, resources, **kwargs)

        # Generate array job IDs
        job_ids = [f"{base_job_id}_{i}" for i in range(array_size)]

        return job_ids


class PBSScheduler(Scheduler):
    """PBS (Portable Batch System) interface."""

    def __init__(self):
        """Initialize PBS scheduler."""
        super().__init__("PBS")
        self._check_availability()

    def _check_availability(self) -> bool:
        """Check if PBS is available."""
        try:
            subprocess.run(["qstat", "--version"],
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def submit_job(
        self,
        script_path: str,
        job_name: str,
        resources: ResourceRequirements,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Submit job to PBS.

        Args:
            script_path: Path to job script
            job_name: Job name
            resources: Resource requirements
            dependencies: List of job IDs to depend on
            **kwargs: Additional PBS options

        Returns:
            job_id: PBS job ID
        """
        # Build qsub command
        cmd = ["qsub"]

        # Job name
        cmd.extend(["-N", job_name])

        # Resources
        select_str = f"select={resources.nodes}:ncpus={resources.cpus_per_task}"
        if resources.gpus_per_node > 0:
            select_str += f":ngpus={resources.gpus_per_node}"
        select_str += f":mem={resources.memory_gb}gb"

        cmd.extend(["-l", select_str])

        # Time limit
        hours = int(resources.time_hours)
        minutes = int((resources.time_hours - hours) * 60)
        cmd.extend(["-l", f"walltime={hours:02d}:{minutes:02d}:00"])

        # Queue
        if resources.partition:
            cmd.extend(["-q", resources.partition])

        # Account
        if resources.account:
            cmd.extend(["-A", resources.account])

        # Dependencies
        if dependencies:
            dep_str = ":".join(dependencies)
            cmd.extend(["-W", f"depend=afterok:{dep_str}"])

        # Output files
        stdout_path = kwargs.get("stdout", f"{job_name}.o")
        stderr_path = kwargs.get("stderr", f"{job_name}.e")
        cmd.extend(["-o", stdout_path])
        cmd.extend(["-e", stderr_path])

        # Script path
        cmd.append(script_path)

        # Submit
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse job ID from output
        job_id = result.stdout.strip()

        # Store job info
        self.jobs[job_id] = JobInfo(
            job_id=job_id,
            name=job_name,
            status=JobStatus.PENDING,
            submit_time=time.time(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            working_dir=os.getcwd(),
            dependencies=dependencies or []
        )

        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of PBS job.

        Args:
            job_id: PBS job ID

        Returns:
            status: Job status
        """
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id],
                capture_output=True, text=True, check=True
            )

            # Parse job state
            match = re.search(r"job_state\s*=\s*(\w+)", result.stdout)
            if not match:
                return JobStatus.UNKNOWN

            state = match.group(1).upper()

            # Map PBS states to JobStatus
            status_map = {
                "Q": JobStatus.PENDING,    # Queued
                "R": JobStatus.RUNNING,    # Running
                "E": JobStatus.RUNNING,    # Exiting
                "H": JobStatus.PENDING,    # Held
                "T": JobStatus.PENDING,    # Moved
                "W": JobStatus.PENDING,    # Waiting
                "S": JobStatus.PENDING,    # Suspended
                "C": JobStatus.COMPLETED,  # Completed
                "F": JobStatus.FAILED,     # Failed
            }

            status = status_map.get(state, JobStatus.UNKNOWN)

            # Update stored job info
            if job_id in self.jobs:
                self.jobs[job_id].status = status

            return status

        except subprocess.CalledProcessError:
            return JobStatus.UNKNOWN

    def cancel_job(self, job_id: str) -> bool:
        """Cancel PBS job.

        Args:
            job_id: PBS job ID

        Returns:
            success: True if cancelled successfully
        """
        try:
            subprocess.run(
                ["qdel", job_id],
                capture_output=True, check=True
            )

            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.CANCELLED

            return True
        except subprocess.CalledProcessError:
            return False

    def get_queue_info(self) -> List[JobInfo]:
        """Get information about all jobs in PBS queue.

        Returns:
            jobs: List of job info
        """
        try:
            result = subprocess.run(
                ["qstat", "-u", os.getenv("USER", "")],
                capture_output=True, text=True, check=True
            )

            jobs = []
            for line in result.stdout.strip().split("\n")[2:]:  # Skip headers
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                job_id = parts[0]
                name = parts[1]
                state = parts[4]

                status_map = {
                    "Q": JobStatus.PENDING,
                    "R": JobStatus.RUNNING,
                    "E": JobStatus.RUNNING,
                    "H": JobStatus.PENDING,
                    "C": JobStatus.COMPLETED,
                    "F": JobStatus.FAILED,
                }

                status = status_map.get(state, JobStatus.UNKNOWN)

                job = JobInfo(
                    job_id=job_id,
                    name=name,
                    status=status,
                    submit_time=time.time()  # PBS doesn't provide in qstat
                )
                jobs.append(job)

            return jobs

        except subprocess.CalledProcessError:
            return []


class LocalScheduler(Scheduler):
    """Local execution (no scheduler) for testing."""

    def __init__(self):
        """Initialize local scheduler."""
        super().__init__("Local")
        self._job_counter = 0
        self._processes: Dict[str, subprocess.Popen] = {}

    def submit_job(
        self,
        script_path: str,
        job_name: str,
        resources: ResourceRequirements,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Submit job locally (execute in background).

        Args:
            script_path: Path to job script
            job_name: Job name
            resources: Resource requirements (ignored for local)
            dependencies: List of job IDs to depend on
            **kwargs: Additional options

        Returns:
            job_id: Local job ID
        """
        # Wait for dependencies
        if dependencies:
            for dep_id in dependencies:
                self.wait_for_job(dep_id)

        # Generate job ID
        self._job_counter += 1
        job_id = f"local_{self._job_counter}"

        # Prepare output files
        stdout_path = kwargs.get("stdout", f"{job_name}_{job_id}.out")
        stderr_path = kwargs.get("stderr", f"{job_name}_{job_id}.err")

        # Start process
        with open(stdout_path, "w") as stdout_file, \
             open(stderr_path, "w") as stderr_file:

            process = subprocess.Popen(
                ["bash", script_path],
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=os.getcwd()
            )

            self._processes[job_id] = process

        # Store job info
        self.jobs[job_id] = JobInfo(
            job_id=job_id,
            name=job_name,
            status=JobStatus.RUNNING,
            submit_time=time.time(),
            start_time=time.time(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            working_dir=os.getcwd(),
            dependencies=dependencies or []
        )

        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of local job.

        Args:
            job_id: Local job ID

        Returns:
            status: Job status
        """
        if job_id not in self._processes:
            # Check if we have job info (may have been cancelled)
            if job_id in self.jobs:
                return self.jobs[job_id].status
            return JobStatus.UNKNOWN

        process = self._processes[job_id]
        returncode = process.poll()

        if returncode is None:
            status = JobStatus.RUNNING
        elif returncode == 0:
            status = JobStatus.COMPLETED
        elif returncode < 0:
            # Negative return code means terminated by signal
            # Check if we marked it as cancelled
            if job_id in self.jobs and self.jobs[job_id].status == JobStatus.CANCELLED:
                status = JobStatus.CANCELLED
            else:
                status = JobStatus.FAILED
        else:
            status = JobStatus.FAILED

        # Update stored job info
        if job_id in self.jobs:
            # Don't overwrite CANCELLED status
            if self.jobs[job_id].status != JobStatus.CANCELLED:
                self.jobs[job_id].status = status
            if returncode is not None:
                self.jobs[job_id].end_time = time.time()
                self.jobs[job_id].exit_code = returncode

        return status if job_id not in self.jobs or self.jobs[job_id].status != JobStatus.CANCELLED else JobStatus.CANCELLED

    def cancel_job(self, job_id: str) -> bool:
        """Cancel local job.

        Args:
            job_id: Local job ID

        Returns:
            success: True if cancelled successfully
        """
        if job_id not in self._processes:
            return False

        try:
            process = self._processes[job_id]
            process.terminate()
            process.wait(timeout=5.0)

            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.CANCELLED

            return True
        except:
            return False

    def get_queue_info(self) -> List[JobInfo]:
        """Get information about all local jobs.

        Returns:
            jobs: List of job info
        """
        # Update all job statuses
        for job_id in list(self.jobs.keys()):
            self.get_job_status(job_id)

        return list(self.jobs.values())


class JobManager:
    """High-level job management interface."""

    def __init__(self, scheduler: Optional[Scheduler] = None, auto_detect: bool = True):
        """Initialize job manager.

        Args:
            scheduler: Scheduler instance (None = auto-detect)
            auto_detect: Auto-detect scheduler if None
        """
        if scheduler is None and auto_detect:
            scheduler = self._auto_detect_scheduler()

        self.scheduler = scheduler or LocalScheduler()
        self.job_history: List[JobInfo] = []

    def _auto_detect_scheduler(self) -> Scheduler:
        """Auto-detect available scheduler."""
        # Try SLURM first
        try:
            return SLURMScheduler()
        except:
            pass

        # Try PBS
        try:
            return PBSScheduler()
        except:
            pass

        # Fall back to local
        return LocalScheduler()

    def submit(
        self,
        script_path: str,
        job_name: str,
        resources: Optional[ResourceRequirements] = None,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Submit job.

        Args:
            script_path: Path to job script
            job_name: Job name
            resources: Resource requirements
            dependencies: List of job IDs to depend on
            **kwargs: Additional options

        Returns:
            job_id: Submitted job ID
        """
        if resources is None:
            resources = ResourceRequirements()

        job_id = self.scheduler.submit_job(
            script_path, job_name, resources, dependencies, **kwargs
        )

        return job_id

    def status(self, job_id: str) -> JobStatus:
        """Get job status.

        Args:
            job_id: Job ID

        Returns:
            status: Job status
        """
        return self.scheduler.get_job_status(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel job.

        Args:
            job_id: Job ID

        Returns:
            success: True if cancelled successfully
        """
        return self.scheduler.cancel_job(job_id)

    def wait(
        self,
        job_id: str,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None
    ) -> JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Job ID
            poll_interval: Polling interval in seconds
            timeout: Timeout in seconds

        Returns:
            status: Final job status
        """
        return self.scheduler.wait_for_job(job_id, poll_interval, timeout)

    def queue(self) -> List[JobInfo]:
        """Get queue information.

        Returns:
            jobs: List of job info
        """
        return self.scheduler.get_queue_info()

    def submit_array(
        self,
        script_path: str,
        job_name: str,
        array_size: int,
        resources: Optional[ResourceRequirements] = None,
        **kwargs
    ) -> List[str]:
        """Submit job array.

        Args:
            script_path: Path to job script
            job_name: Job name
            array_size: Number of jobs in array
            resources: Resource requirements
            **kwargs: Additional options

        Returns:
            job_ids: List of job IDs
        """
        if resources is None:
            resources = ResourceRequirements()

        job_ids = self.scheduler.submit_job_array(
            script_path, job_name, resources, array_size, **kwargs
        )

        return job_ids

    def wait_all(
        self,
        job_ids: List[str],
        poll_interval: float = 30.0,
        timeout: Optional[float] = None
    ) -> Dict[str, JobStatus]:
        """Wait for all jobs to complete.

        Args:
            job_ids: List of job IDs
            poll_interval: Polling interval in seconds
            timeout: Timeout in seconds

        Returns:
            statuses: Dict mapping job_id -> status
        """
        statuses = {}

        for job_id in job_ids:
            status = self.wait(job_id, poll_interval, timeout)
            statuses[job_id] = status

        return statuses

    def get_info(self, job_id: str) -> Optional[JobInfo]:
        """Get detailed job information.

        Args:
            job_id: Job ID

        Returns:
            info: Job info (None if not found)
        """
        return self.scheduler.jobs.get(job_id)

    def save_history(self, path: str):
        """Save job history to file.

        Args:
            path: Output file path
        """
        history = [
            {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status.value,
                "submit_time": job.submit_time,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "exit_code": job.exit_code,
                "working_dir": job.working_dir,
            }
            for job in self.job_history
        ]

        with open(path, "w") as f:
            json.dump(history, f, indent=2)
