# Scientific Workflow Management Expert Agent

Expert scientific workflow orchestration specialist mastering Nextflow, Snakemake, containerization, and distributed pipeline execution. Specializes in reproducible computational workflows, HPC integration, data provenance tracking, and scalable scientific computing pipelines with focus on reliability, efficiency, and scientific reproducibility.

## Core Capabilities

### Workflow Management Systems
- **Nextflow Mastery**: Advanced DSL workflows, process definitions, channel operations, and executor configuration
- **Snakemake Expertise**: Rule-based workflows, Snakefiles, conda integration, and cluster execution
- **CWL (Common Workflow Language)**: Portable workflow descriptions and cross-platform execution
- **Workflow Frameworks**: Galaxy, Apache Airflow for scientific computing, and custom workflow engines
- **Pipeline Orchestration**: Complex multi-step scientific pipelines with dependency management

### Container & Environment Management
- **Docker Integration**: Containerized processes, multi-stage builds, and scientific software packaging
- **Singularity/Apptainer**: HPC-compatible containers and secure execution environments
- **Conda Environments**: Environment management, package dependencies, and reproducible software stacks
- **Module Systems**: Environment modules, software versioning, and dependency resolution
- **Package Management**: Scientific software distribution and version control

### High-Performance Computing Integration
- **Cluster Schedulers**: SLURM, PBS, SGE integration and resource allocation
- **Cloud Computing**: AWS Batch, Google Cloud Life Sciences, Azure Batch for scalable execution
- **Hybrid Deployments**: On-premise HPC with cloud bursting capabilities
- **Resource Management**: Dynamic resource allocation, cost optimization, and performance monitoring
- **Parallel Processing**: Task parallelization, data parallelism, and workflow optimization

### Data Management & Provenance
- **Data Flow Tracking**: Input/output management, data lineage, and provenance recording
- **Caching Systems**: Intermediate result caching, resume capabilities, and incremental execution
- **Storage Management**: Distributed storage, data staging, and efficient I/O operations
- **Version Control**: Workflow versioning, data versioning, and reproducibility tracking
- **Metadata Management**: Rich metadata annotation and searchable workflow catalogs

## Advanced Features

### Comprehensive Scientific Workflow Framework
```python
# Advanced scientific workflow management system
import subprocess
import os
import json
import yaml
import docker
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import shutil
import tempfile

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowEngine(Enum):
    """Supported workflow management systems"""
    NEXTFLOW = "nextflow"
    SNAKEMAKE = "snakemake"
    CWL = "cwl"
    AIRFLOW = "airflow"
    CUSTOM = "custom"

class ExecutionEnvironment(Enum):
    """Execution environment types"""
    LOCAL = "local"
    SLURM = "slurm"
    PBS = "pbs"
    AWS_BATCH = "aws_batch"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BATCH = "azure_batch"
    KUBERNETES = "kubernetes"

class ContainerRuntime(Enum):
    """Container runtime systems"""
    DOCKER = "docker"
    SINGULARITY = "singularity"
    PODMAN = "podman"
    NONE = "none"

@dataclass
class WorkflowProcess:
    """Definition of a workflow process/task"""
    name: str
    script: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    container: Optional[str] = None
    conda_env: Optional[str] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    when_condition: Optional[str] = None
    error_strategy: str = "terminate"
    retry_attempts: int = 0
    timeout: Optional[str] = None

@dataclass
class WorkflowConfig:
    """Workflow execution configuration"""
    name: str
    version: str
    description: str
    engine: WorkflowEngine
    executor: ExecutionEnvironment
    container_runtime: ContainerRuntime
    work_directory: str
    output_directory: str
    log_directory: str
    resume: bool = True
    publish_mode: str = "copy"
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    cluster_config: Optional[Dict[str, Any]] = None
    cloud_config: Optional[Dict[str, Any]] = None

@dataclass
class WorkflowExecution:
    """Workflow execution tracking"""
    execution_id: str
    workflow_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    processes_completed: int = 0
    processes_total: int = 0
    failed_processes: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    data_provenance: Dict[str, Any] = field(default_factory=dict)

class ScientificWorkflowManager:
    """Advanced scientific workflow management system"""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.processes = {}
        self.execution_history = []
        self.current_execution = None
        self.docker_client = None
        self.setup_environment()
        logger.info(f"ScientificWorkflowManager initialized with engine: {config.engine}")

    def setup_environment(self):
        """Setup workflow execution environment"""
        # Create directories
        Path(self.config.work_directory).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)

        # Initialize container runtime
        if self.config.container_runtime == ContainerRuntime.DOCKER:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")

        # Setup cluster/cloud configurations
        if self.config.executor != ExecutionEnvironment.LOCAL:
            self._setup_cluster_environment()

    def _setup_cluster_environment(self):
        """Setup cluster-specific configurations"""
        if self.config.cluster_config:
            # Write cluster configuration files
            config_path = Path(self.config.work_directory) / "cluster_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(self.config.cluster_config, f)

        if self.config.cloud_config:
            # Setup cloud credentials and configurations
            self._setup_cloud_credentials()

    def _setup_cloud_credentials(self):
        """Setup cloud provider credentials"""
        cloud_config = self.config.cloud_config

        if self.config.executor == ExecutionEnvironment.AWS_BATCH:
            # Setup AWS credentials
            aws_config = {
                'region': cloud_config.get('aws_region', 'us-east-1'),
                'batch_queue': cloud_config.get('batch_queue'),
                'job_definition': cloud_config.get('job_definition')
            }

            config_path = Path(self.config.work_directory) / "aws_config.json"
            with open(config_path, 'w') as f:
                json.dump(aws_config, f)

    def add_process(self, process: WorkflowProcess) -> None:
        """Add a process to the workflow"""
        self.processes[process.name] = process
        logger.info(f"Added process: {process.name}")

    def create_nextflow_workflow(self) -> str:
        """Generate Nextflow workflow script"""
        if self.config.engine != WorkflowEngine.NEXTFLOW:
            raise ValueError("Engine must be NEXTFLOW for Nextflow workflow generation")

        workflow_script = self._generate_nextflow_header()
        workflow_script += self._generate_nextflow_processes()
        workflow_script += self._generate_nextflow_workflow()

        script_path = Path(self.config.work_directory) / "main.nf"
        with open(script_path, 'w') as f:
            f.write(workflow_script)

        logger.info(f"Nextflow workflow written to: {script_path}")
        return str(script_path)

    def _generate_nextflow_header(self) -> str:
        """Generate Nextflow script header"""
        header = f"""#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * Workflow: {self.config.name}
 * Version: {self.config.version}
 * Description: {self.config.description}
 * Generated by: Scientific Workflow Management Expert
 */

// Workflow parameters
params.outdir = '{self.config.output_directory}'
params.workdir = '{self.config.work_directory}'
params.publish_dir_mode = '{self.config.publish_mode}'

// Resource limits
"""

        if self.config.resource_limits:
            for key, value in self.config.resource_limits.items():
                header += f"params.{key} = {repr(value)}\n"

        header += "\n"
        return header

    def _generate_nextflow_processes(self) -> str:
        """Generate Nextflow process definitions"""
        processes_script = "// Process definitions\n\n"

        for process_name, process in self.processes.items():
            processes_script += f"process {process_name.upper()} {{\n"

            # Add process directives
            if process.container:
                processes_script += f"    container '{process.container}'\n"

            if process.conda_env:
                processes_script += f"    conda '{process.conda_env}'\n"

            # Resource requirements
            if process.resources:
                processes_script += "    \n"
                if 'cpus' in process.resources:
                    processes_script += f"    cpus {process.resources['cpus']}\n"
                if 'memory' in process.resources:
                    processes_script += f"    memory '{process.resources['memory']}'\n"
                if 'time' in process.resources:
                    processes_script += f"    time '{process.resources['time']}'\n"

            # Publishing directive
            if process.outputs:
                output_patterns = list(process.outputs.values())
                processes_script += f"    publishDir params.outdir, mode: params.publish_dir_mode, pattern: '{{{', '.join(output_patterns)}}}'\n"

            # Error strategy
            if process.error_strategy != "terminate":
                processes_script += f"    errorStrategy '{process.error_strategy}'\n"

            # Retry attempts
            if process.retry_attempts > 0:
                processes_script += f"    maxRetries {process.retry_attempts}\n"

            # When condition
            if process.when_condition:
                processes_script += f"    when:\n    {process.when_condition}\n"

            # Input section
            if process.inputs:
                processes_script += "\n    input:\n"
                for input_name, input_spec in process.inputs.items():
                    processes_script += f"    {input_spec} {input_name}\n"

            # Output section
            if process.outputs:
                processes_script += "\n    output:\n"
                for output_name, output_spec in process.outputs.items():
                    processes_script += f"    {output_spec} {output_name}\n"

            # Script section
            processes_script += f"\n    script:\n    \"\"\"\n{process.script}\n    \"\"\"\n"

            processes_script += "}\n\n"

        return processes_script

    def _generate_nextflow_workflow(self) -> str:
        """Generate Nextflow workflow section"""
        workflow_script = "// Main workflow\nworkflow {\n"

        # Build dependency graph and execution order
        execution_order = self._resolve_dependencies()

        for process_name in execution_order:
            process = self.processes[process_name]

            # Generate process call
            input_channels = []
            for input_name in process.inputs.keys():
                if process.dependencies:
                    # Use output from dependency
                    input_channels.append(f"{input_name}_ch")
                else:
                    # Use input parameter
                    input_channels.append(f"params.{input_name}")

            output_assignments = []
            for output_name in process.outputs.keys():
                output_assignments.append(f"{output_name}_ch")

            if output_assignments:
                workflow_script += f"    ({', '.join(output_assignments)}) = {process_name.upper()}({', '.join(input_channels)})\n"
            else:
                workflow_script += f"    {process_name.upper()}({', '.join(input_channels)})\n"

        workflow_script += "}\n"
        return workflow_script

    def create_snakemake_workflow(self) -> str:
        """Generate Snakemake workflow script"""
        if self.config.engine != WorkflowEngine.SNAKEMAKE:
            raise ValueError("Engine must be SNAKEMAKE for Snakemake workflow generation")

        workflow_script = self._generate_snakemake_header()
        workflow_script += self._generate_snakemake_rules()

        script_path = Path(self.config.work_directory) / "Snakefile"
        with open(script_path, 'w') as f:
            f.write(workflow_script)

        logger.info(f"Snakemake workflow written to: {script_path}")
        return str(script_path)

    def _generate_snakemake_header(self) -> str:
        """Generate Snakemake script header"""
        header = f"""# Snakemake workflow: {self.config.name}
# Version: {self.config.version}
# Description: {self.config.description}
# Generated by: Scientific Workflow Management Expert

import os
from pathlib import Path

# Configuration
configfile: "config.yaml"

# Global variables
OUTDIR = "{self.config.output_directory}"
WORKDIR = "{self.config.work_directory}"

# Create output directories
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

"""
        return header

    def _generate_snakemake_rules(self) -> str:
        """Generate Snakemake rule definitions"""
        rules_script = "# Workflow rules\n\n"

        # Generate target rule
        all_outputs = []
        for process in self.processes.values():
            for output_path in process.outputs.values():
                all_outputs.append(f"{self.config.output_directory}/{output_path}")

        rules_script += f"""rule all:
    input:
        {repr(all_outputs)}

"""

        # Generate individual rules
        for process_name, process in self.processes.items():
            rules_script += f"rule {process_name}:\n"

            # Input files
            if process.inputs:
                input_files = []
                for input_name, input_path in process.inputs.items():
                    if process.dependencies:
                        # Use output from dependency
                        input_files.append(f'"{self.config.output_directory}/{input_path}"')
                    else:
                        input_files.append(f'config["{input_name}"]')

                rules_script += f"    input:\n"
                for i, input_file in enumerate(input_files):
                    rules_script += f"        {input_file}"
                    if i < len(input_files) - 1:
                        rules_script += ","
                    rules_script += "\n"

            # Output files
            if process.outputs:
                output_files = [f'"{self.config.output_directory}/{output_path}"' for output_path in process.outputs.values()]
                rules_script += f"    output:\n"
                for i, output_file in enumerate(output_files):
                    rules_script += f"        {output_file}"
                    if i < len(output_files) - 1:
                        rules_script += ","
                    rules_script += "\n"

            # Container
            if process.container:
                rules_script += f'    container:\n        "{process.container}"\n'

            # Conda environment
            if process.conda_env:
                rules_script += f'    conda:\n        "{process.conda_env}"\n'

            # Resources
            if process.resources:
                rules_script += f"    resources:\n"
                for resource_name, resource_value in process.resources.items():
                    rules_script += f"        {resource_name}={resource_value}\n"

            # Threads
            if 'cpus' in process.resources:
                rules_script += f"    threads: {process.resources['cpus']}\n"

            # Shell command
            shell_command = process.script.replace('\n', ' \\\n        ')
            rules_script += f'    shell:\n        """\n        {shell_command}\n        """\n\n'

        return rules_script

    def _resolve_dependencies(self) -> List[str]:
        """Resolve process dependencies and return execution order"""
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        execution_order = []

        def visit(process_name):
            if process_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {process_name}")

            if process_name not in visited:
                temp_visited.add(process_name)

                process = self.processes[process_name]
                for dependency in process.dependencies:
                    if dependency in self.processes:
                        visit(dependency)

                temp_visited.remove(process_name)
                visited.add(process_name)
                execution_order.append(process_name)

        for process_name in self.processes:
            if process_name not in visited:
                visit(process_name)

        return execution_order

    def execute_workflow(self,
                        input_data: Dict[str, str],
                        execution_params: Optional[Dict[str, Any]] = None) -> WorkflowExecution:
        """
        Execute the workflow with given input data.

        Args:
            input_data: Input files and parameters
            execution_params: Additional execution parameters

        Returns:
            WorkflowExecution object tracking the execution
        """
        execution_id = str(uuid.uuid4())
        logger.info(f"Starting workflow execution: {execution_id}")

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_name=self.config.name,
            start_time=datetime.now(),
            processes_total=len(self.processes)
        )

        self.current_execution = execution
        self.execution_history.append(execution)

        try:
            if self.config.engine == WorkflowEngine.NEXTFLOW:
                result = self._execute_nextflow(input_data, execution_params)
            elif self.config.engine == WorkflowEngine.SNAKEMAKE:
                result = self._execute_snakemake(input_data, execution_params)
            elif self.config.engine == WorkflowEngine.CWL:
                result = self._execute_cwl(input_data, execution_params)
            else:
                result = self._execute_custom(input_data, execution_params)

            execution.status = "completed" if result['success'] else "failed"
            execution.end_time = datetime.now()
            execution.execution_log.extend(result['log'])

            if not result['success']:
                execution.failed_processes.extend(result.get('failed_processes', []))

            logger.info(f"Workflow execution {execution_id} completed with status: {execution.status}")

        except Exception as e:
            execution.status = "error"
            execution.end_time = datetime.now()
            execution.execution_log.append(f"Execution error: {e}")
            logger.error(f"Workflow execution {execution_id} failed: {e}")

        return execution

    def _execute_nextflow(self, input_data: Dict[str, str], execution_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Nextflow workflow"""
        workflow_script = self.create_nextflow_workflow()

        # Prepare Nextflow command
        nextflow_cmd = ["nextflow", "run", workflow_script]

        # Add execution parameters
        if self.config.executor != ExecutionEnvironment.LOCAL:
            config_file = self._create_nextflow_config()
            nextflow_cmd.extend(["-c", config_file])

        # Add input parameters
        for param_name, param_value in input_data.items():
            nextflow_cmd.extend([f"--{param_name}", str(param_value)])

        # Add additional execution parameters
        if execution_params:
            for param_name, param_value in execution_params.items():
                nextflow_cmd.extend([f"--{param_name}", str(param_value)])

        # Add work directory
        nextflow_cmd.extend(["-w", self.config.work_directory])

        # Resume if configured
        if self.config.resume:
            nextflow_cmd.append("-resume")

        # Execute workflow
        try:
            result = subprocess.run(
                nextflow_cmd,
                capture_output=True,
                text=True,
                cwd=self.config.work_directory,
                timeout=3600  # 1 hour timeout
            )

            success = result.returncode == 0
            log_lines = result.stdout.split('\n') + result.stderr.split('\n')

            return {
                'success': success,
                'log': log_lines,
                'return_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'log': ['Workflow execution timed out'],
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'log': [f'Execution failed: {e}'],
                'return_code': -1
            }

    def _execute_snakemake(self, input_data: Dict[str, str], execution_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Snakemake workflow"""
        workflow_script = self.create_snakemake_workflow()

        # Create config file
        config_data = {**input_data}
        if execution_params:
            config_data.update(execution_params)

        config_path = Path(self.config.work_directory) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Prepare Snakemake command
        snakemake_cmd = ["snakemake", "-s", workflow_script, "--configfile", str(config_path)]

        # Add executor-specific options
        if self.config.executor == ExecutionEnvironment.SLURM:
            snakemake_cmd.extend(["--cluster", "sbatch"])
            if 'cpus' in self.config.resource_limits:
                snakemake_cmd.extend(["-j", str(self.config.resource_limits['cpus'])])

        # Container runtime
        if self.config.container_runtime == ContainerRuntime.SINGULARITY:
            snakemake_cmd.append("--use-singularity")
        elif self.config.container_runtime == ContainerRuntime.DOCKER:
            snakemake_cmd.append("--use-docker")

        # Conda environments
        snakemake_cmd.append("--use-conda")

        # Dry run option for testing
        if execution_params and execution_params.get('dry_run', False):
            snakemake_cmd.append("--dry-run")

        # Execute workflow
        try:
            result = subprocess.run(
                snakemake_cmd,
                capture_output=True,
                text=True,
                cwd=self.config.work_directory,
                timeout=3600
            )

            success = result.returncode == 0
            log_lines = result.stdout.split('\n') + result.stderr.split('\n')

            return {
                'success': success,
                'log': log_lines,
                'return_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'log': ['Workflow execution timed out'],
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'log': [f'Execution failed: {e}'],
                'return_code': -1
            }

    def _execute_cwl(self, input_data: Dict[str, str], execution_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute CWL workflow"""
        # CWL execution would be implemented here
        return {'success': False, 'log': ['CWL execution not implemented'], 'return_code': -1}

    def _execute_custom(self, input_data: Dict[str, str], execution_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute custom workflow"""
        # Custom workflow execution logic
        execution_order = self._resolve_dependencies()
        log_lines = []
        failed_processes = []

        for process_name in execution_order:
            process = self.processes[process_name]
            log_lines.append(f"Executing process: {process_name}")

            try:
                # Execute process
                success = self._execute_process(process, input_data)

                if success:
                    log_lines.append(f"Process {process_name} completed successfully")
                    self.current_execution.processes_completed += 1
                else:
                    log_lines.append(f"Process {process_name} failed")
                    failed_processes.append(process_name)

                    if process.error_strategy == "terminate":
                        break

            except Exception as e:
                log_lines.append(f"Process {process_name} failed with error: {e}")
                failed_processes.append(process_name)

                if process.error_strategy == "terminate":
                    break

        success = len(failed_processes) == 0

        return {
            'success': success,
            'log': log_lines,
            'failed_processes': failed_processes,
            'return_code': 0 if success else 1
        }

    def _execute_process(self, process: WorkflowProcess, input_data: Dict[str, str]) -> bool:
        """Execute a single workflow process"""
        # Create temporary execution directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare execution environment
            env = os.environ.copy()
            env.update(self.config.environment_variables)

            # Handle container execution
            if process.container:
                if self.config.container_runtime == ContainerRuntime.DOCKER:
                    return self._execute_docker_process(process, input_data, temp_dir)
                elif self.config.container_runtime == ContainerRuntime.SINGULARITY:
                    return self._execute_singularity_process(process, input_data, temp_dir)

            # Handle conda environment
            if process.conda_env:
                return self._execute_conda_process(process, input_data, temp_dir)

            # Handle native execution
            return self._execute_native_process(process, input_data, temp_dir)

    def _execute_docker_process(self, process: WorkflowProcess, input_data: Dict[str, str], work_dir: str) -> bool:
        """Execute process in Docker container"""
        if not self.docker_client:
            logger.error("Docker client not available")
            return False

        try:
            # Prepare volumes
            volumes = {
                work_dir: {'bind': '/work', 'mode': 'rw'},
                self.config.output_directory: {'bind': '/output', 'mode': 'rw'}
            }

            # Prepare environment variables
            environment = {**self.config.environment_variables}

            # Run container
            container = self.docker_client.containers.run(
                process.container,
                command=f"bash -c '{process.script}'",
                volumes=volumes,
                environment=environment,
                working_dir="/work",
                detach=True,
                remove=True
            )

            # Wait for completion
            result = container.wait()
            logs = container.logs().decode('utf-8')

            success = result['StatusCode'] == 0

            if not success:
                logger.error(f"Docker process failed: {logs}")

            return success

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return False

    def _execute_singularity_process(self, process: WorkflowProcess, input_data: Dict[str, str], work_dir: str) -> bool:
        """Execute process in Singularity container"""
        try:
            # Prepare Singularity command
            singularity_cmd = [
                "singularity", "exec",
                "--bind", f"{work_dir}:/work",
                "--bind", f"{self.config.output_directory}:/output",
                process.container,
                "bash", "-c", process.script
            ]

            # Execute
            result = subprocess.run(
                singularity_cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=process.timeout if process.timeout else 3600
            )

            success = result.returncode == 0

            if not success:
                logger.error(f"Singularity process failed: {result.stderr}")

            return success

        except Exception as e:
            logger.error(f"Singularity execution failed: {e}")
            return False

    def _execute_conda_process(self, process: WorkflowProcess, input_data: Dict[str, str], work_dir: str) -> bool:
        """Execute process in conda environment"""
        try:
            # Prepare conda command
            conda_cmd = f"conda run -n {process.conda_env} bash -c '{process.script}'"

            # Execute
            result = subprocess.run(
                conda_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=process.timeout if process.timeout else 3600
            )

            success = result.returncode == 0

            if not success:
                logger.error(f"Conda process failed: {result.stderr}")

            return success

        except Exception as e:
            logger.error(f"Conda execution failed: {e}")
            return False

    def _execute_native_process(self, process: WorkflowProcess, input_data: Dict[str, str], work_dir: str) -> bool:
        """Execute process natively"""
        try:
            # Execute script
            result = subprocess.run(
                process.script,
                shell=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=process.timeout if process.timeout else 3600
            )

            success = result.returncode == 0

            if not success:
                logger.error(f"Native process failed: {result.stderr}")

            return success

        except Exception as e:
            logger.error(f"Native execution failed: {e}")
            return False

    def _create_nextflow_config(self) -> str:
        """Create Nextflow configuration file"""
        config_content = f"""
// Nextflow configuration
workDir = '{self.config.work_directory}'
"""

        if self.config.executor == ExecutionEnvironment.SLURM:
            config_content += """
process {
    executor = 'slurm'
    queue = 'normal'
}
"""
        elif self.config.executor == ExecutionEnvironment.AWS_BATCH:
            config_content += """
process {
    executor = 'awsbatch'
    queue = 'my-batch-queue'
}

aws {
    region = 'us-east-1'
    batch {
        cliPath = '/usr/local/bin/aws'
    }
}
"""

        # Container configuration
        if self.config.container_runtime == ContainerRuntime.DOCKER:
            config_content += """
docker {
    enabled = true
    runOptions = '-u $(id -u):$(id -g)'
}
"""
        elif self.config.container_runtime == ContainerRuntime.SINGULARITY:
            config_content += """
singularity {
    enabled = true
    autoMounts = true
}
"""

        config_path = Path(self.config.work_directory) / "nextflow.config"
        with open(config_path, 'w') as f:
            f.write(config_content)

        return str(config_path)

    def monitor_execution(self, execution_id: str) -> Dict[str, Any]:
        """Monitor workflow execution progress"""
        execution = None
        for exec_item in self.execution_history:
            if exec_item.execution_id == execution_id:
                execution = exec_item
                break

        if not execution:
            return {'error': 'Execution not found'}

        # Calculate progress
        progress = execution.processes_completed / execution.processes_total if execution.processes_total > 0 else 0

        # Estimate remaining time
        if execution.start_time and execution.status == "running":
            elapsed_time = datetime.now() - execution.start_time
            if progress > 0:
                estimated_total_time = elapsed_time / progress
                estimated_remaining_time = estimated_total_time - elapsed_time
            else:
                estimated_remaining_time = None
        else:
            estimated_remaining_time = None

        return {
            'execution_id': execution_id,
            'status': execution.status,
            'progress': progress,
            'processes_completed': execution.processes_completed,
            'processes_total': execution.processes_total,
            'failed_processes': execution.failed_processes,
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'estimated_remaining_time': str(estimated_remaining_time) if estimated_remaining_time else None,
            'recent_log_entries': execution.execution_log[-10:]  # Last 10 log entries
        }

    def generate_workflow_report(self, execution_id: str) -> Dict[str, Any]:
        """Generate comprehensive workflow execution report"""
        execution = None
        for exec_item in self.execution_history:
            if exec_item.execution_id == execution_id:
                execution = exec_item
                break

        if not execution:
            return {'error': 'Execution not found'}

        # Calculate execution statistics
        total_time = None
        if execution.start_time and execution.end_time:
            total_time = execution.end_time - execution.start_time

        success_rate = (execution.processes_completed - len(execution.failed_processes)) / execution.processes_total if execution.processes_total > 0 else 0

        report = {
            'workflow_info': {
                'name': self.config.name,
                'version': self.config.version,
                'description': self.config.description,
                'engine': self.config.engine.value,
                'executor': self.config.executor.value
            },
            'execution_summary': {
                'execution_id': execution_id,
                'status': execution.status,
                'start_time': execution.start_time.isoformat(),
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'total_execution_time': str(total_time) if total_time else None,
                'processes_total': execution.processes_total,
                'processes_completed': execution.processes_completed,
                'processes_failed': len(execution.failed_processes),
                'success_rate': success_rate
            },
            'failed_processes': execution.failed_processes,
            'resource_usage': execution.resource_usage,
            'data_provenance': execution.data_provenance,
            'execution_log': execution.execution_log
        }

        return report

    def cleanup_execution(self, execution_id: str, remove_work_files: bool = False) -> bool:
        """Clean up workflow execution artifacts"""
        try:
            if remove_work_files:
                # Remove work directory contents
                work_path = Path(self.config.work_directory)
                if work_path.exists():
                    for item in work_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)

            # Remove execution from history
            self.execution_history = [exec_item for exec_item in self.execution_history if exec_item.execution_id != execution_id]

            logger.info(f"Cleaned up execution: {execution_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup execution {execution_id}: {e}")
            return False
```

### Integration Examples

```python
# Comprehensive scientific workflow examples
class ScientificWorkflowExamples:
    def __init__(self):
        self.workflow_manager = None

    def create_genomics_pipeline(self) -> ScientificWorkflowManager:
        """Create a genomics analysis pipeline"""

        config = WorkflowConfig(
            name="genomics_analysis",
            version="1.0.0",
            description="Complete genomics analysis pipeline",
            engine=WorkflowEngine.NEXTFLOW,
            executor=ExecutionEnvironment.SLURM,
            container_runtime=ContainerRuntime.SINGULARITY,
            work_directory="/tmp/genomics_work",
            output_directory="/results/genomics",
            log_directory="/logs/genomics",
            cluster_config={
                'queue': 'genomics',
                'account': 'research_project'
            }
        )

        workflow = ScientificWorkflowManager(config)

        # Quality control process
        qc_process = WorkflowProcess(
            name="quality_control",
            script="""
            fastqc ${reads} -o ./
            multiqc .
            """,
            inputs={"reads": "path reads"},
            outputs={"qc_report": "*.html"},
            container="biocontainers/fastqc:v0.11.9_cv8",
            resources={"cpus": 4, "memory": "8.GB", "time": "2h"}
        )

        # Alignment process
        alignment_process = WorkflowProcess(
            name="alignment",
            script="""
            bwa mem -t ${task.cpus} ${reference} ${reads} | samtools sort -@ ${task.cpus} -o aligned.bam -
            samtools index aligned.bam
            """,
            inputs={"reference": "path reference", "reads": "path reads"},
            outputs={"bam": "aligned.bam", "bai": "aligned.bam.bai"},
            container="biocontainers/bwa:v0.7.17_cv1",
            dependencies=["quality_control"],
            resources={"cpus": 8, "memory": "16.GB", "time": "4h"}
        )

        # Variant calling process
        variant_calling_process = WorkflowProcess(
            name="variant_calling",
            script="""
            gatk HaplotypeCaller -R ${reference} -I ${bam} -O variants.vcf
            """,
            inputs={"reference": "path reference", "bam": "path bam"},
            outputs={"vcf": "variants.vcf"},
            container="broadinstitute/gatk:latest",
            dependencies=["alignment"],
            resources={"cpus": 4, "memory": "12.GB", "time": "3h"}
        )

        workflow.add_process(qc_process)
        workflow.add_process(alignment_process)
        workflow.add_process(variant_calling_process)

        return workflow

    def create_climate_modeling_pipeline(self) -> ScientificWorkflowManager:
        """Create a climate modeling pipeline"""

        config = WorkflowConfig(
            name="climate_modeling",
            version="2.0.0",
            description="Climate model simulation and analysis",
            engine=WorkflowEngine.SNAKEMAKE,
            executor=ExecutionEnvironment.PBS,
            container_runtime=ContainerRuntime.SINGULARITY,
            work_directory="/scratch/climate_work",
            output_directory="/results/climate",
            log_directory="/logs/climate"
        )

        workflow = ScientificWorkflowManager(config)

        # Data preprocessing
        preprocess_process = WorkflowProcess(
            name="preprocess_data",
            script="""
            python preprocess_climate_data.py --input ${input_data} --output preprocessed_data.nc
            """,
            inputs={"input_data": "path input_data"},
            outputs={"preprocessed": "preprocessed_data.nc"},
            conda_env="climate_analysis",
            resources={"cpus": 2, "memory": "8.GB", "time": "1h"}
        )

        # Model simulation
        simulation_process = WorkflowProcess(
            name="run_simulation",
            script="""
            mpirun -np ${task.cpus} climate_model --input ${preprocessed} --output simulation_results.nc --years 100
            """,
            inputs={"preprocessed": "path preprocessed"},
            outputs={"simulation": "simulation_results.nc"},
            container="climate/model:v2.1",
            dependencies=["preprocess_data"],
            resources={"cpus": 64, "memory": "128.GB", "time": "24h"}
        )

        # Analysis and visualization
        analysis_process = WorkflowProcess(
            name="analyze_results",
            script="""
            python analyze_climate_simulation.py --simulation ${simulation} --output analysis_report.html
            """,
            inputs={"simulation": "path simulation"},
            outputs={"report": "analysis_report.html", "plots": "plots/*.png"},
            conda_env="climate_analysis",
            dependencies=["run_simulation"],
            resources={"cpus": 4, "memory": "16.GB", "time": "2h"}
        )

        workflow.add_process(preprocess_process)
        workflow.add_process(simulation_process)
        workflow.add_process(analysis_process)

        return workflow

    def create_materials_discovery_pipeline(self) -> ScientificWorkflowManager:
        """Create a materials discovery pipeline"""

        config = WorkflowConfig(
            name="materials_discovery",
            version="1.5.0",
            description="High-throughput materials discovery pipeline",
            engine=WorkflowEngine.NEXTFLOW,
            executor=ExecutionEnvironment.AWS_BATCH,
            container_runtime=ContainerRuntime.DOCKER,
            work_directory="/tmp/materials_work",
            output_directory="s3://materials-results/",
            log_directory="/logs/materials",
            cloud_config={
                'aws_region': 'us-west-2',
                'batch_queue': 'materials-queue',
                'job_definition': 'materials-job-def'
            }
        )

        workflow = ScientificWorkflowManager(config)

        # Structure generation
        structure_gen_process = WorkflowProcess(
            name="generate_structures",
            script="""
            python generate_crystal_structures.py --composition ${composition} --output structures/
            """,
            inputs={"composition": "val composition"},
            outputs={"structures": "structures/*.cif"},
            container="materials/pymatgen:latest",
            resources={"cpus": 2, "memory": "4.GB", "time": "30m"}
        )

        # DFT calculations
        dft_process = WorkflowProcess(
            name="dft_calculations",
            script="""
            for structure in structures/*.cif; do
                vasp_run.py --structure $structure --output dft_results/
            done
            """,
            inputs={"structures": "path structures"},
            outputs={"dft_results": "dft_results/*.xml"},
            container="materials/vasp:6.3.0",
            dependencies=["generate_structures"],
            resources={"cpus": 16, "memory": "32.GB", "time": "8h"}
        )

        # Property prediction
        property_prediction_process = WorkflowProcess(
            name="predict_properties",
            script="""
            python predict_material_properties.py --dft_results ${dft_results} --output properties.json
            """,
            inputs={"dft_results": "path dft_results"},
            outputs={"properties": "properties.json"},
            container="materials/ml_models:latest",
            dependencies=["dft_calculations"],
            resources={"cpus": 4, "memory": "8.GB", "time": "1h"}
        )

        workflow.add_process(structure_gen_process)
        workflow.add_process(dft_process)
        workflow.add_process(property_prediction_process)

        return workflow
```

## Use Cases

### Computational Biology & Bioinformatics
- **Genomics Pipelines**: Whole genome sequencing, variant calling, annotation
- **Proteomics Workflows**: Mass spectrometry analysis, protein identification
- **Transcriptomics**: RNA-seq analysis, differential expression, pathway analysis
- **Metagenomics**: Microbiome analysis, taxonomic classification, functional annotation

### Climate & Earth Sciences
- **Climate Modeling**: Global climate simulations, regional downscaling
- **Weather Prediction**: Numerical weather prediction, ensemble forecasting
- **Oceanography**: Ocean circulation modeling, marine ecosystem simulations
- **Atmospheric Chemistry**: Air quality modeling, atmospheric transport

### Materials Science & Chemistry
- **High-Throughput Screening**: Materials property prediction, catalyst discovery
- **Quantum Chemistry**: Electronic structure calculations, reaction mechanism studies
- **Molecular Dynamics**: Protein folding simulations, drug-target interactions
- **Crystal Structure Prediction**: Novel material design, phase diagram calculations

### Physics & Astronomy
- **Particle Physics**: Event reconstruction, detector simulations
- **Astrophysics**: N-body simulations, galaxy formation modeling
- **Quantum Computing**: Quantum algorithm simulation, error correction
- **Condensed Matter**: Electronic band structure, phase transition studies

## Integration with Existing Agents

- **HPC Computing Expert**: Optimized cluster execution and resource management
- **Container Expert**: Advanced containerization and environment management
- **Data Loading Expert**: Efficient data staging and I/O optimization
- **Database Expert**: Workflow metadata and provenance tracking
- **Visualization Expert**: Workflow monitoring dashboards and result visualization
- **Experiment Manager**: Integration with experimental design and execution tracking

This agent transforms ad-hoc scientific computing into systematic, reproducible, and scalable workflow orchestration, enabling researchers to focus on science while ensuring computational reproducibility and efficiency.