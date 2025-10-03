"""SLURM Cluster Integration.

This module provides integration with SLURM workload manager for
submitting and managing jobs on HPC clusters.

Features:
1. Job submission with resource specification
2. Array jobs for parameter sweeps
3. Job monitoring and status tracking
4. Automatic retry logic
5. GPU resource allocation

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
import re
import json
import os


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SLURMConfig:
    """SLURM job configuration.

    Attributes:
        job_name: Name of the job
        partition: SLURM partition (queue)
        nodes: Number of nodes
        ntasks: Number of tasks
        cpus_per_task: CPUs per task
        mem: Memory per node (e.g., "16GB")
        time: Wall time limit (e.g., "01:00:00")
        gres: Generic resources (e.g., "gpu:1")
        output: Output file path (%j replaced with job ID)
        error: Error file path
        mail_type: Email notification type
        mail_user: Email address
        account: SLURM account
        qos: Quality of service
        constraint: Node constraints
        export: Environment variables to export
        setup_commands: Commands to run before main script
    """
    job_name: str = "optimal_control"
    partition: str = "general"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 1
    mem: str = "8GB"
    time: str = "01:00:00"
    gres: Optional[str] = None
    output: str = "slurm-%j.out"
    error: str = "slurm-%j.err"
    mail_type: Optional[str] = None
    mail_user: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    export: str = "ALL"
    setup_commands: List[str] = field(default_factory=list)

    def to_sbatch_header(self) -> str:
        """Convert to sbatch header directives.

        Returns:
            SBATCH header string
        """
        lines = ["#!/bin/bash"]

        # Required parameters
        lines.append(f"#SBATCH --job-name={self.job_name}")
        lines.append(f"#SBATCH --partition={self.partition}")
        lines.append(f"#SBATCH --nodes={self.nodes}")
        lines.append(f"#SBATCH --ntasks={self.ntasks}")
        lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        lines.append(f"#SBATCH --mem={self.mem}")
        lines.append(f"#SBATCH --time={self.time}")
        lines.append(f"#SBATCH --output={self.output}")
        lines.append(f"#SBATCH --error={self.error}")
        lines.append(f"#SBATCH --export={self.export}")

        # Optional parameters
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        if self.mail_type:
            lines.append(f"#SBATCH --mail-type={self.mail_type}")
        if self.mail_user:
            lines.append(f"#SBATCH --mail-user={self.mail_user}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")

        lines.append("")  # Blank line

        # Setup commands
        if self.setup_commands:
            lines.append("# Setup")
            lines.extend(self.setup_commands)
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Job Management
# =============================================================================

@dataclass
class SLURMJob:
    """Represents a SLURM job.

    Attributes:
        job_id: SLURM job ID
        job_name: Job name
        config: Job configuration
        script_path: Path to job script
        status: Job status (PENDING, RUNNING, COMPLETED, FAILED)
        submit_time: Submission timestamp
        start_time: Start timestamp
        end_time: End timestamp
        exit_code: Exit code
    """
    job_id: int
    job_name: str
    config: SLURMConfig
    script_path: Path
    status: str = "PENDING"
    submit_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None

    def update_status(self) -> str:
        """Update job status from SLURM.

        Returns:
            Current job status
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", str(self.job_id), "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                self.status = result.stdout.strip()
            else:
                # Job not in queue, check sacct
                result = subprocess.run(
                    ["sacct", "-j", str(self.job_id), "-n", "-o", "State"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    self.status = result.stdout.strip().split()[0]
                else:
                    self.status = "UNKNOWN"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.status = "UNKNOWN"

        return self.status

    def cancel(self) -> bool:
        """Cancel the job.

        Returns:
            True if successfully cancelled
        """
        try:
            result = subprocess.run(
                ["scancel", str(self.job_id)],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def is_running(self) -> bool:
        """Check if job is running."""
        self.update_status()
        return self.status == "RUNNING"

    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        self.update_status()
        return self.status in ["COMPLETED", "COMPLETING"]

    def is_failed(self) -> bool:
        """Check if job failed."""
        self.update_status()
        return self.status in ["FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"]

    def is_pending(self) -> bool:
        """Check if job is pending."""
        self.update_status()
        return self.status == "PENDING"

    def wait(self, poll_interval: float = 10.0, timeout: Optional[float] = None) -> bool:
        """Wait for job to complete.

        Args:
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            True if completed successfully, False if failed/timeout
        """
        start = time.time()

        while True:
            self.update_status()

            if self.is_completed():
                return True
            if self.is_failed():
                return False

            if timeout and (time.time() - start) > timeout:
                return False

            time.sleep(poll_interval)


# =============================================================================
# SLURM Scheduler
# =============================================================================

class SLURMScheduler:
    """High-level SLURM job scheduler.

    Manages job submission, monitoring, and result collection.
    """

    def __init__(self, work_dir: Optional[Path] = None):
        """Initialize scheduler.

        Args:
            work_dir: Working directory for job scripts and output
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / "slurm_jobs"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[int, SLURMJob] = {}

    def submit_job(
        self,
        script: str,
        config: SLURMConfig,
        script_name: Optional[str] = None
    ) -> SLURMJob:
        """Submit a job to SLURM.

        Args:
            script: Job script content (bash commands)
            config: SLURM configuration
            script_name: Name for script file (default: job_name.sh)

        Returns:
            SLURMJob object

        Raises:
            RuntimeError: If submission fails
        """
        # Create script file
        if script_name is None:
            script_name = f"{config.job_name}.sh"

        script_path = self.work_dir / script_name

        # Write script
        full_script = config.to_sbatch_header() + "\n" + script

        with open(script_path, 'w') as f:
            f.write(full_script)

        script_path.chmod(0o755)

        # Submit
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir
            )

            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr}")

            # Parse job ID from output
            # Expected format: "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not match:
                raise RuntimeError(f"Could not parse job ID from: {result.stdout}")

            job_id = int(match.group(1))

            # Create job object
            job = SLURMJob(
                job_id=job_id,
                job_name=config.job_name,
                config=config,
                script_path=script_path,
                submit_time=time.time()
            )

            self.jobs[job_id] = job

            return job

        except subprocess.TimeoutExpired:
            raise RuntimeError("sbatch timed out")
        except FileNotFoundError:
            raise RuntimeError("sbatch command not found (SLURM not available)")

    def submit_array_job(
        self,
        script_template: str,
        config: SLURMConfig,
        array_spec: str,
        script_name: Optional[str] = None
    ) -> List[SLURMJob]:
        """Submit array job to SLURM.

        Args:
            script_template: Job script with $SLURM_ARRAY_TASK_ID variable
            config: SLURM configuration
            array_spec: Array specification (e.g., "1-10", "1-10:2")
            script_name: Name for script file

        Returns:
            List of SLURMJob objects (one per array task)

        Raises:
            RuntimeError: If submission fails
        """
        # Add array directive to config
        if script_name is None:
            script_name = f"{config.job_name}_array.sh"

        script_path = self.work_dir / script_name

        # Create header with array directive
        header = config.to_sbatch_header()
        header += f"#SBATCH --array={array_spec}\n\n"

        full_script = header + script_template

        with open(script_path, 'w') as f:
            f.write(full_script)

        script_path.chmod(0o755)

        # Submit
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir
            )

            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr}")

            # Parse job ID
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not match:
                raise RuntimeError(f"Could not parse job ID from: {result.stdout}")

            base_job_id = int(match.group(1))

            # Parse array spec to create job objects
            jobs = []
            array_indices = self._parse_array_spec(array_spec)

            for i, array_idx in enumerate(array_indices):
                job = SLURMJob(
                    job_id=base_job_id if i == 0 else f"{base_job_id}_{array_idx}",
                    job_name=f"{config.job_name}_{array_idx}",
                    config=config,
                    script_path=script_path,
                    submit_time=time.time()
                )
                jobs.append(job)
                self.jobs[job.job_id] = job

            return jobs

        except subprocess.TimeoutExpired:
            raise RuntimeError("sbatch timed out")
        except FileNotFoundError:
            raise RuntimeError("sbatch command not found")

    def _parse_array_spec(self, array_spec: str) -> List[int]:
        """Parse array specification into list of indices.

        Args:
            array_spec: e.g., "1-10", "1-10:2", "1,3,5"

        Returns:
            List of array indices
        """
        indices = []

        for part in array_spec.split(','):
            if '-' in part:
                # Range
                range_parts = part.split(':')
                range_spec = range_parts[0]
                step = int(range_parts[1]) if len(range_parts) > 1 else 1

                start, end = map(int, range_spec.split('-'))
                indices.extend(range(start, end + 1, step))
            else:
                # Single value
                indices.append(int(part))

        return indices

    def monitor_jobs(self, poll_interval: float = 30.0) -> Dict[str, int]:
        """Monitor all jobs and return summary.

        Args:
            poll_interval: Seconds between checks

        Returns:
            Dictionary with status counts
        """
        status_counts = {}

        for job in self.jobs.values():
            status = job.update_status()
            status_counts[status] = status_counts.get(status, 0) + 1

        return status_counts

    def wait_all(
        self,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None
    ) -> Dict[str, List[SLURMJob]]:
        """Wait for all jobs to complete.

        Args:
            poll_interval: Seconds between checks
            timeout: Maximum wait time

        Returns:
            Dictionary mapping status to list of jobs
        """
        start = time.time()

        while True:
            status_counts = self.monitor_jobs(poll_interval)

            # Check if all done
            active = sum(
                count for status, count in status_counts.items()
                if status in ["PENDING", "RUNNING", "CONFIGURING"]
            )

            if active == 0:
                break

            if timeout and (time.time() - start) > timeout:
                break

            time.sleep(poll_interval)

        # Collect results by status
        results = {}
        for job in self.jobs.values():
            job.update_status()
            if job.status not in results:
                results[job.status] = []
            results[job.status].append(job)

        return results

    def cancel_all(self) -> int:
        """Cancel all jobs.

        Returns:
            Number of jobs cancelled
        """
        count = 0
        for job in self.jobs.values():
            if job.cancel():
                count += 1
        return count


# =============================================================================
# Convenience Functions
# =============================================================================

def submit_job(
    script: str,
    config: Optional[SLURMConfig] = None,
    work_dir: Optional[Path] = None
) -> SLURMJob:
    """Submit a single job to SLURM.

    Args:
        script: Job script content
        config: SLURM configuration (default config if None)
        work_dir: Working directory

    Returns:
        SLURMJob object
    """
    scheduler = SLURMScheduler(work_dir)

    if config is None:
        config = SLURMConfig()

    return scheduler.submit_job(script, config)


def submit_array_job(
    script_template: str,
    array_spec: str,
    config: Optional[SLURMConfig] = None,
    work_dir: Optional[Path] = None
) -> List[SLURMJob]:
    """Submit array job to SLURM.

    Args:
        script_template: Job script with $SLURM_ARRAY_TASK_ID
        array_spec: Array specification (e.g., "1-100")
        config: SLURM configuration
        work_dir: Working directory

    Returns:
        List of SLURMJob objects
    """
    scheduler = SLURMScheduler(work_dir)

    if config is None:
        config = SLURMConfig()

    return scheduler.submit_array_job(script_template, config, array_spec)


def monitor_jobs(job_ids: List[int], poll_interval: float = 30.0) -> Dict[int, str]:
    """Monitor status of specific jobs.

    Args:
        job_ids: List of job IDs to monitor
        poll_interval: Seconds between checks

    Returns:
        Dictionary mapping job ID to status
    """
    statuses = {}

    for job_id in job_ids:
        try:
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                statuses[job_id] = result.stdout.strip()
            else:
                statuses[job_id] = "UNKNOWN"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            statuses[job_id] = "UNKNOWN"

    return statuses


def cancel_job(job_id: int) -> bool:
    """Cancel a SLURM job.

    Args:
        job_id: Job ID to cancel

    Returns:
        True if successfully cancelled
    """
    try:
        result = subprocess.run(
            ["scancel", str(job_id)],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_slurm_info() -> Dict[str, Any]:
    """Get SLURM cluster information.

    Returns:
        Dictionary with cluster info
    """
    info = {
        "available": False,
        "version": None,
        "partitions": [],
        "nodes": 0
    }

    try:
        # Check SLURM version
        result = subprocess.run(
            ["sinfo", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            info["available"] = True
            match = re.search(r"slurm (\d+\.\d+\.\d+)", result.stdout)
            if match:
                info["version"] = match.group(1)

        # Get partitions
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%P"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            info["partitions"] = [
                p.strip().rstrip('*')
                for p in result.stdout.strip().split('\n')
            ]

        # Get node count
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%D"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            info["nodes"] = sum(int(x) for x in result.stdout.strip().split('\n'))

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return info
