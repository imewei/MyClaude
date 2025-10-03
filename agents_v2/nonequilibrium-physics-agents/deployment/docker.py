"""Docker Containerization for Optimal Control.

This module provides Docker container management:
- Multi-stage builds for optimized images
- GPU support (CUDA, JAX)
- Production and development configurations
- Registry integration (Docker Hub, ECR, GCR)
- Container health checks
- Resource limits and optimization

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import json
import re

# =============================================================================
# Docker Image Configuration
# =============================================================================

@dataclass
class DockerImageConfig:
    """Configuration for Docker image build.

    Attributes:
        name: Image name
        tag: Image tag
        base_image: Base Docker image
        python_version: Python version
        cuda_version: CUDA version (for GPU support)
        install_jax: Install JAX with GPU support
        install_gpu_deps: Install GPU dependencies
        requirements_file: Path to requirements.txt
        copy_files: Files/directories to copy
        expose_ports: Ports to expose
        env_vars: Environment variables
        build_args: Docker build arguments
    """
    name: str
    tag: str = "latest"
    base_image: str = "python:3.10-slim"
    python_version: str = "3.10"
    cuda_version: Optional[str] = None  # e.g., "12.1.0"
    install_jax: bool = True
    install_gpu_deps: bool = False
    requirements_file: str = "requirements.txt"
    copy_files: List[str] = field(default_factory=lambda: [
        "solvers", "ml_optimal_control", "applications",
        "hpc", "visualization", "gpu_kernels", "api"
    ])
    expose_ports: List[int] = field(default_factory=lambda: [8000, 8080])
    env_vars: Dict[str, str] = field(default_factory=dict)
    build_args: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/app"
    user: str = "appuser"

    def get_full_name(self) -> str:
        """Get full image name with tag."""
        return f"{self.name}:{self.tag}"


# =============================================================================
# Dockerfile Generation
# =============================================================================

class DockerBuilder:
    """Build and manage Docker images."""

    def __init__(self, config: DockerImageConfig):
        """Initialize Docker builder.

        Args:
            config: Docker image configuration
        """
        self.config = config

    def generate_dockerfile(self) -> str:
        """Generate Dockerfile content.

        Returns:
            Dockerfile content as string
        """
        lines = []

        # Multi-stage build for optimization
        lines.append("# Multi-stage Docker build for optimal control")
        lines.append("")

        # Stage 1: Builder
        lines.append("# ============================================================")
        lines.append("# Stage 1: Builder - Install dependencies")
        lines.append("# ============================================================")

        if self.config.cuda_version and self.config.install_gpu_deps:
            # Use CUDA base image for GPU support
            base = f"nvidia/cuda:{self.config.cuda_version}-cudnn8-runtime-ubuntu22.04"
            lines.append(f"FROM {base} as builder")
            lines.append("")
            lines.append("# Install Python")
            lines.append("RUN apt-get update && apt-get install -y \\")
            lines.append(f"    python{self.config.python_version} \\")
            lines.append(f"    python{self.config.python_version}-dev \\")
            lines.append("    python3-pip \\")
            lines.append("    git \\")
            lines.append("    && rm -rf /var/lib/apt/lists/*")
        else:
            # Use Python base image
            lines.append(f"FROM {self.config.base_image} as builder")

        lines.append("")
        lines.append("# Set working directory")
        lines.append(f"WORKDIR {self.config.working_dir}")
        lines.append("")

        # Copy requirements
        lines.append("# Copy requirements")
        lines.append(f"COPY {self.config.requirements_file} .")

        if self.config.install_gpu_deps:
            lines.append("COPY requirements-gpu.txt .")

        lines.append("")

        # Install Python packages
        lines.append("# Install Python dependencies")
        lines.append("RUN pip install --no-cache-dir --upgrade pip && \\")
        lines.append(f"    pip install --no-cache-dir -r {self.config.requirements_file}")

        if self.config.install_gpu_deps and self.config.install_jax:
            lines.append("")
            lines.append("# Install JAX with GPU support")
            lines.append("RUN pip install --no-cache-dir -r requirements-gpu.txt")

        lines.append("")

        # Stage 2: Runtime
        lines.append("# ============================================================")
        lines.append("# Stage 2: Runtime - Optimized production image")
        lines.append("# ============================================================")

        if self.config.cuda_version and self.config.install_gpu_deps:
            base = f"nvidia/cuda:{self.config.cuda_version}-cudnn8-runtime-ubuntu22.04"
            lines.append(f"FROM {base}")
            lines.append("")
            lines.append("# Install Python runtime")
            lines.append("RUN apt-get update && apt-get install -y \\")
            lines.append(f"    python{self.config.python_version} \\")
            lines.append("    && rm -rf /var/lib/apt/lists/*")
        else:
            lines.append(f"FROM {self.config.base_image}")

        lines.append("")

        # Create non-root user
        lines.append("# Create non-root user for security")
        lines.append(f"RUN useradd -m -u 1000 {self.config.user} && \\")
        lines.append(f"    mkdir -p {self.config.working_dir} && \\")
        lines.append(f"    chown -R {self.config.user}:{self.config.user} {self.config.working_dir}")
        lines.append("")

        # Set working directory
        lines.append(f"WORKDIR {self.config.working_dir}")
        lines.append("")

        # Copy installed packages from builder
        lines.append("# Copy Python packages from builder")
        python_ver = self.config.python_version
        lines.append(f"COPY --from=builder /usr/local/lib/python{python_ver}/site-packages /usr/local/lib/python{python_ver}/site-packages")
        lines.append("COPY --from=builder /usr/local/bin /usr/local/bin")
        lines.append("")

        # Copy application code
        lines.append("# Copy application code")
        for item in self.config.copy_files:
            lines.append(f"COPY --chown={self.config.user}:{self.config.user} {item} {self.config.working_dir}/{item}")
        lines.append("")

        # Set environment variables
        if self.config.env_vars:
            lines.append("# Set environment variables")
            for key, value in self.config.env_vars.items():
                lines.append(f"ENV {key}={value}")
            lines.append("")

        # Expose ports
        if self.config.expose_ports:
            lines.append("# Expose ports")
            for port in self.config.expose_ports:
                lines.append(f"EXPOSE {port}")
            lines.append("")

        # Health check
        lines.append("# Health check")
        lines.append("HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\")
        lines.append("    CMD python -c \"import sys; sys.exit(0)\" || exit 1")
        lines.append("")

        # Switch to non-root user
        lines.append(f"USER {self.config.user}")
        lines.append("")

        # Default command
        lines.append("# Default command")
        lines.append('CMD ["python", "-m", "api.rest_api"]')

        return "\n".join(lines)

    def write_dockerfile(self, output_path: Path = Path("Dockerfile")) -> Path:
        """Write Dockerfile to disk.

        Args:
            output_path: Path to write Dockerfile

        Returns:
            Path to written Dockerfile
        """
        dockerfile_content = self.generate_dockerfile()

        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)

        print(f"Generated Dockerfile at {output_path}")
        return output_path

    def build(
        self,
        dockerfile_path: Path = Path("Dockerfile"),
        context_path: Path = Path("."),
        no_cache: bool = False,
        pull: bool = True
    ) -> bool:
        """Build Docker image.

        Args:
            dockerfile_path: Path to Dockerfile
            context_path: Build context path
            no_cache: Don't use cache
            pull: Pull base image

        Returns:
            True if build succeeded
        """
        cmd = [
            "docker", "build",
            "-t", self.config.get_full_name(),
            "-f", str(dockerfile_path),
        ]

        # Add build args
        for key, value in self.config.build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

        if no_cache:
            cmd.append("--no-cache")

        if pull:
            cmd.append("--pull")

        cmd.append(str(context_path))

        print(f"Building Docker image: {self.config.get_full_name()}")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print("Build successful!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False

    def run(
        self,
        command: Optional[str] = None,
        ports: Optional[Dict[int, int]] = None,
        volumes: Optional[Dict[str, str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        detach: bool = True,
        remove: bool = True,
        name: Optional[str] = None,
        gpu: bool = False
    ) -> Optional[str]:
        """Run Docker container.

        Args:
            command: Command to run (overrides default)
            ports: Port mappings {container_port: host_port}
            volumes: Volume mounts {host_path: container_path}
            env_vars: Environment variables
            detach: Run in background
            remove: Remove container on exit
            name: Container name
            gpu: Enable GPU support

        Returns:
            Container ID if successful
        """
        cmd = ["docker", "run"]

        if detach:
            cmd.append("-d")

        if remove:
            cmd.append("--rm")

        if name:
            cmd.extend(["--name", name])

        if gpu:
            cmd.append("--gpus=all")

        # Port mappings
        if ports:
            for container_port, host_port in ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])

        # Volume mounts
        if volumes:
            for host_path, container_path in volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.append(self.config.get_full_name())

        if command:
            cmd.extend(command.split())

        print(f"Running container: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            container_id = result.stdout.strip()
            print(f"Container started: {container_id[:12]}")
            return container_id
        except subprocess.CalledProcessError as e:
            print(f"Failed to run container: {e}")
            print(f"stderr: {e.stderr}")
            return None

    def push(self, registry: Optional[str] = None) -> bool:
        """Push image to registry.

        Args:
            registry: Registry URL (e.g., 'docker.io', 'gcr.io')

        Returns:
            True if push succeeded
        """
        if registry:
            # Tag for registry
            registry_name = f"{registry}/{self.config.get_full_name()}"
            tag_cmd = ["docker", "tag", self.config.get_full_name(), registry_name]

            try:
                subprocess.run(tag_cmd, check=True)
                push_name = registry_name
            except subprocess.CalledProcessError as e:
                print(f"Failed to tag image: {e}")
                return False
        else:
            push_name = self.config.get_full_name()

        push_cmd = ["docker", "push", push_name]

        print(f"Pushing image: {push_name}")

        try:
            subprocess.run(push_cmd, check=True)
            print("Push successful!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Push failed: {e}")
            return False

    def inspect(self) -> Optional[Dict]:
        """Inspect Docker image.

        Returns:
            Image metadata
        """
        cmd = ["docker", "inspect", self.config.get_full_name()]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            return json.loads(result.stdout)[0]
        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as e:
            print(f"Failed to inspect image: {e}")
            return None


# =============================================================================
# High-Level Functions
# =============================================================================

def build_docker_image(
    name: str,
    tag: str = "latest",
    gpu: bool = False,
    **kwargs
) -> bool:
    """Build Docker image with sensible defaults.

    Args:
        name: Image name
        tag: Image tag
        gpu: Enable GPU support
        **kwargs: Additional DockerImageConfig parameters

    Returns:
        True if build succeeded
    """
    # Set GPU defaults
    if gpu:
        kwargs.setdefault('cuda_version', '12.1.0')
        kwargs.setdefault('install_gpu_deps', True)
        kwargs.setdefault('base_image', 'nvidia/cuda:12.1.0-base-ubuntu22.04')

    config = DockerImageConfig(name=name, tag=tag, **kwargs)
    builder = DockerBuilder(config)

    # Generate Dockerfile
    builder.write_dockerfile()

    # Build image
    return builder.build()


def run_docker_container(
    image_name: str,
    tag: str = "latest",
    **kwargs
) -> Optional[str]:
    """Run Docker container.

    Args:
        image_name: Image name
        tag: Image tag
        **kwargs: Arguments for DockerBuilder.run()

    Returns:
        Container ID if successful
    """
    config = DockerImageConfig(name=image_name, tag=tag)
    builder = DockerBuilder(config)
    return builder.run(**kwargs)


def push_to_registry(
    image_name: str,
    tag: str = "latest",
    registry: Optional[str] = None
) -> bool:
    """Push image to registry.

    Args:
        image_name: Image name
        tag: Image tag
        registry: Registry URL

    Returns:
        True if push succeeded
    """
    config = DockerImageConfig(name=image_name, tag=tag)
    builder = DockerBuilder(config)
    return builder.push(registry)


# =============================================================================
# Docker Compose Integration
# =============================================================================

def generate_docker_compose(
    services: List[str],
    output_path: Path = Path("docker-compose.yml")
) -> Path:
    """Generate docker-compose.yml for multi-service deployment.

    Args:
        services: List of services to include
        output_path: Path to write docker-compose.yml

    Returns:
        Path to written file
    """
    compose = {
        'version': '3.8',
        'services': {}
    }

    # API service
    if 'api' in services:
        compose['services']['api'] = {
            'build': '.',
            'ports': ['8000:8000'],
            'environment': {
                'ENVIRONMENT': 'production'
            },
            'healthcheck': {
                'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3
            }
        }

    # Worker service
    if 'worker' in services:
        compose['services']['worker'] = {
            'build': '.',
            'command': 'python -m hpc.worker',
            'environment': {
                'ENVIRONMENT': 'production'
            }
        }

    # GPU service
    if 'gpu' in services:
        compose['services']['gpu'] = {
            'build': '.',
            'command': 'python -m gpu_kernels.server',
            'deploy': {
                'resources': {
                    'reservations': {
                        'devices': [{
                            'driver': 'nvidia',
                            'count': 1,
                            'capabilities': ['gpu']
                        }]
                    }
                }
            }
        }

    # Write to file
    import yaml
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        yaml.dump(compose, f, default_flow_style=False)

    print(f"Generated docker-compose.yml at {output_path}")
    return output_path
