"""Configuration Management for Deployment.

Handles environment-specific configuration:
- Development, staging, production configs
- Secret management
- Configuration validation
- Environment variable handling

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import os


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str  # 'development', 'staging', 'production'
    image_name: str
    image_tag: str
    replicas: int = 3
    namespace: str = "default"
    resource_limits: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "1000m",
        "memory": "2Gi"
    })
    resource_requests: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "500m",
        "memory": "1Gi"
    })
    enable_autoscaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    name: str
    log_level: str = "INFO"
    debug: bool = False
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    api_keys: Dict[str, str] = field(default_factory=dict)


def load_config(config_path: Path, environment: str = "development") -> DeploymentConfig:
    """Load configuration from file.

    Args:
        config_path: Path to config file
        environment: Environment name

    Returns:
        Deployment configuration
    """
    with open(config_path) as f:
        config_data = json.load(f)

    env_config = config_data.get(environment, {})
    return DeploymentConfig(**env_config)


def validate_config(config: DeploymentConfig) -> bool:
    """Validate configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid
    """
    if config.replicas < 1:
        raise ValueError("replicas must be >= 1")

    if config.enable_autoscaling:
        if config.min_replicas < 1:
            raise ValueError("min_replicas must be >= 1")
        if config.max_replicas < config.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")

    return True


def merge_configs(base: DeploymentConfig, override: Dict[str, Any]) -> DeploymentConfig:
    """Merge configuration with overrides.

    Args:
        base: Base configuration
        override: Override dictionary

    Returns:
        Merged configuration
    """
    config_dict = asdict(base)
    config_dict.update(override)
    return DeploymentConfig(**config_dict)
