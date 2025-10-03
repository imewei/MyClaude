"""Deployment Infrastructure for Production.

This module provides comprehensive deployment capabilities:
1. Docker containerization
2. Kubernetes orchestration
3. Cloud platform integration (AWS, GCP, Azure)
4. CI/CD pipeline automation
5. Configuration management
6. Health monitoring and logging

Author: Nonequilibrium Physics Agents
"""

from .docker import (
    DockerBuilder,
    DockerImageConfig,
    build_docker_image,
    run_docker_container,
    push_to_registry,
)

from .kubernetes import (
    KubernetesDeployment,
    KubernetesService,
    KubernetesConfig,
    deploy_to_kubernetes,
    scale_deployment,
    get_deployment_status,
)

from .config_manager import (
    DeploymentConfig,
    EnvironmentConfig,
    load_config,
    validate_config,
    merge_configs,
)

from .monitoring import (
    HealthCheck,
    MetricsCollector,
    LogAggregator,
    AlertManager,
)

from .ci_cd import (
    CIConfig,
    CDConfig,
    GitHubActionsBuilder,
    GitLabCIBuilder,
    run_tests,
    build_and_deploy,
)

__all__ = [
    # Docker
    'DockerBuilder',
    'DockerImageConfig',
    'build_docker_image',
    'run_docker_container',
    'push_to_registry',
    # Kubernetes
    'KubernetesDeployment',
    'KubernetesService',
    'KubernetesConfig',
    'deploy_to_kubernetes',
    'scale_deployment',
    'get_deployment_status',
    # Configuration
    'DeploymentConfig',
    'EnvironmentConfig',
    'load_config',
    'validate_config',
    'merge_configs',
    # Monitoring
    'HealthCheck',
    'MetricsCollector',
    'LogAggregator',
    'AlertManager',
    # CI/CD
    'CIConfig',
    'CDConfig',
    'GitHubActionsBuilder',
    'GitLabCIBuilder',
    'run_tests',
    'build_and_deploy',
]

__version__ = '1.0.0'
