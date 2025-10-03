"""Kubernetes Orchestration for Optimal Control.

This module provides Kubernetes deployment management:
- Deployment manifest generation
- Service configuration
- ConfigMap and Secret management
- Horizontal Pod Autoscaling (HPA)
- Resource quotas and limits
- Health checks and readiness probes

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import yaml
import json


# =============================================================================
# Kubernetes Configuration
# =============================================================================

@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes deployment.

    Attributes:
        name: Deployment name
        namespace: Kubernetes namespace
        replicas: Number of replicas
        image: Container image
        image_pull_policy: Image pull policy
        ports: Container ports to expose
        env_vars: Environment variables
        resource_limits: Resource limits (CPU, memory)
        resource_requests: Resource requests (CPU, memory)
        enable_hpa: Enable Horizontal Pod Autoscaler
        min_replicas: Minimum replicas for HPA
        max_replicas: Maximum replicas for HPA
        target_cpu_percent: Target CPU utilization for HPA
    """
    name: str
    namespace: str = "default"
    replicas: int = 3
    image: str = "optimal-control:latest"
    image_pull_policy: str = "IfNotPresent"
    ports: List[int] = field(default_factory=lambda: [8000])
    env_vars: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "1000m",
        "memory": "2Gi"
    })
    resource_requests: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "500m",
        "memory": "1Gi"
    })
    enable_hpa: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_percent: int = 70
    enable_gpu: bool = False
    gpu_count: int = 1


# =============================================================================
# Kubernetes Deployment
# =============================================================================

class KubernetesDeployment:
    """Manage Kubernetes deployments."""

    def __init__(self, config: KubernetesConfig):
        """Initialize Kubernetes deployment.

        Args:
            config: Kubernetes configuration
        """
        self.config = config

    def generate_deployment_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes Deployment manifest.

        Returns:
            Deployment manifest as dictionary
        """
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.name,
                    "version": "v1"
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.name,
                            "version": "v1"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.config.name,
                            "image": self.config.image,
                            "imagePullPolicy": self.config.image_pull_policy,
                            "ports": [{"containerPort": port} for port in self.config.ports],
                            "env": [
                                {"name": key, "value": value}
                                for key, value in self.config.env_vars.items()
                            ],
                            "resources": {
                                "limits": self.config.resource_limits,
                                "requests": self.config.resource_requests
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self.config.ports[0]
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": self.config.ports[0]
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000
                        }
                    }
                }
            }
        }

        # Add GPU resources if enabled
        if self.config.enable_gpu:
            manifest["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = str(self.config.gpu_count)

        return manifest

    def generate_service_manifest(self, service_type: str = "ClusterIP") -> Dict[str, Any]:
        """Generate Kubernetes Service manifest.

        Args:
            service_type: Service type (ClusterIP, NodePort, LoadBalancer)

        Returns:
            Service manifest as dictionary
        """
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.name}-service",
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.name
                }
            },
            "spec": {
                "type": service_type,
                "selector": {
                    "app": self.config.name
                },
                "ports": [
                    {
                        "name": f"port-{port}",
                        "port": port,
                        "targetPort": port,
                        "protocol": "TCP"
                    }
                    for port in self.config.ports
                ]
            }
        }

        return manifest

    def generate_hpa_manifest(self) -> Optional[Dict[str, Any]]:
        """Generate Horizontal Pod Autoscaler manifest.

        Returns:
            HPA manifest or None if HPA disabled
        """
        if not self.config.enable_hpa:
            return None

        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.name}-hpa",
                "namespace": self.config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.config.name
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_cpu_percent
                            }
                        }
                    }
                ]
            }
        }

        return manifest

    def generate_configmap_manifest(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Generate ConfigMap manifest.

        Args:
            data: ConfigMap data

        Returns:
            ConfigMap manifest
        """
        manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.config.name}-config",
                "namespace": self.config.namespace
            },
            "data": data
        }

        return manifest

    def write_manifests(self, output_dir: Path = Path("k8s")) -> List[Path]:
        """Write all manifests to files.

        Args:
            output_dir: Output directory for manifests

        Returns:
            List of written file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written_files = []

        # Deployment
        deployment_file = output_dir / f"{self.config.name}-deployment.yaml"
        with open(deployment_file, 'w') as f:
            yaml.dump(self.generate_deployment_manifest(), f, default_flow_style=False)
        written_files.append(deployment_file)

        # Service
        service_file = output_dir / f"{self.config.name}-service.yaml"
        with open(service_file, 'w') as f:
            yaml.dump(self.generate_service_manifest(), f, default_flow_style=False)
        written_files.append(service_file)

        # HPA
        if self.config.enable_hpa:
            hpa_file = output_dir / f"{self.config.name}-hpa.yaml"
            with open(hpa_file, 'w') as f:
                yaml.dump(self.generate_hpa_manifest(), f, default_flow_style=False)
            written_files.append(hpa_file)

        print(f"Generated {len(written_files)} Kubernetes manifests in {output_dir}")
        return written_files

    def apply(self, manifest_path: Optional[Path] = None) -> bool:
        """Apply Kubernetes manifests.

        Args:
            manifest_path: Path to manifest file or directory

        Returns:
            True if successful
        """
        if manifest_path is None:
            # Generate and apply
            manifest_files = self.write_manifests()
            for manifest_file in manifest_files:
                if not self._apply_file(manifest_file):
                    return False
            return True
        else:
            return self._apply_file(Path(manifest_path))

    def _apply_file(self, file_path: Path) -> bool:
        """Apply single manifest file.

        Args:
            file_path: Path to manifest file

        Returns:
            True if successful
        """
        cmd = ["kubectl", "apply", "-f", str(file_path)]

        print(f"Applying manifest: {file_path}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully applied {file_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to apply {file_path}: {e}")
            print(f"stderr: {e.stderr}")
            return False

    def delete(self) -> bool:
        """Delete deployment and associated resources.

        Returns:
            True if successful
        """
        cmd = [
            "kubectl", "delete",
            "deployment", self.config.name,
            "service", f"{self.config.name}-service",
            "-n", self.config.namespace
        ]

        if self.config.enable_hpa:
            cmd.extend(["hpa", f"{self.config.name}-hpa"])

        print(f"Deleting deployment: {self.config.name}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Successfully deleted deployment")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to delete deployment: {e}")
            return False

    def scale(self, replicas: int) -> bool:
        """Scale deployment to specified number of replicas.

        Args:
            replicas: Desired number of replicas

        Returns:
            True if successful
        """
        cmd = [
            "kubectl", "scale",
            f"deployment/{self.config.name}",
            f"--replicas={replicas}",
            "-n", self.config.namespace
        ]

        print(f"Scaling deployment to {replicas} replicas")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully scaled to {replicas} replicas")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to scale: {e}")
            return False

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get deployment status.

        Returns:
            Deployment status information
        """
        cmd = [
            "kubectl", "get",
            f"deployment/{self.config.name}",
            "-n", self.config.namespace,
            "-o", "json"
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Failed to get status: {e}")
            return None

    def get_pods(self) -> List[str]:
        """Get list of pods for this deployment.

        Returns:
            List of pod names
        """
        cmd = [
            "kubectl", "get", "pods",
            "-l", f"app={self.config.name}",
            "-n", self.config.namespace,
            "-o", "jsonpath={.items[*].metadata.name}"
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout.strip().split()
        except subprocess.CalledProcessError as e:
            print(f"Failed to get pods: {e}")
            return []

    def get_logs(self, pod_name: Optional[str] = None, tail: int = 100) -> Optional[str]:
        """Get logs from deployment pods.

        Args:
            pod_name: Specific pod name (if None, uses first pod)
            tail: Number of log lines to retrieve

        Returns:
            Log output
        """
        if pod_name is None:
            pods = self.get_pods()
            if not pods:
                print("No pods found")
                return None
            pod_name = pods[0]

        cmd = [
            "kubectl", "logs",
            pod_name,
            "-n", self.config.namespace,
            f"--tail={tail}"
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed to get logs: {e}")
            return None


# =============================================================================
# Kubernetes Service
# =============================================================================

class KubernetesService:
    """Manage Kubernetes services."""

    def __init__(self, name: str, namespace: str = "default"):
        """Initialize Kubernetes service.

        Args:
            name: Service name
            namespace: Kubernetes namespace
        """
        self.name = name
        self.namespace = namespace

    def get_endpoint(self) -> Optional[str]:
        """Get service endpoint.

        Returns:
            Service endpoint URL
        """
        cmd = [
            "kubectl", "get",
            f"service/{self.name}",
            "-n", self.namespace,
            "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            ip = result.stdout.strip()
            if ip:
                return f"http://{ip}"
            return None
        except subprocess.CalledProcessError as e:
            print(f"Failed to get endpoint: {e}")
            return None


# =============================================================================
# High-Level Functions
# =============================================================================

def deploy_to_kubernetes(
    name: str,
    image: str,
    namespace: str = "default",
    replicas: int = 3,
    **kwargs
) -> bool:
    """Deploy application to Kubernetes with sensible defaults.

    Args:
        name: Deployment name
        image: Container image
        namespace: Kubernetes namespace
        replicas: Number of replicas
        **kwargs: Additional KubernetesConfig parameters

    Returns:
        True if deployment successful
    """
    config = KubernetesConfig(
        name=name,
        image=image,
        namespace=namespace,
        replicas=replicas,
        **kwargs
    )

    deployment = KubernetesDeployment(config)
    return deployment.apply()


def scale_deployment(
    name: str,
    replicas: int,
    namespace: str = "default"
) -> bool:
    """Scale Kubernetes deployment.

    Args:
        name: Deployment name
        replicas: Desired number of replicas
        namespace: Kubernetes namespace

    Returns:
        True if successful
    """
    config = KubernetesConfig(name=name, namespace=namespace)
    deployment = KubernetesDeployment(config)
    return deployment.scale(replicas)


def get_deployment_status(
    name: str,
    namespace: str = "default"
) -> Optional[Dict[str, Any]]:
    """Get deployment status.

    Args:
        name: Deployment name
        namespace: Kubernetes namespace

    Returns:
        Deployment status information
    """
    config = KubernetesConfig(name=name, namespace=namespace)
    deployment = KubernetesDeployment(config)
    return deployment.get_status()
