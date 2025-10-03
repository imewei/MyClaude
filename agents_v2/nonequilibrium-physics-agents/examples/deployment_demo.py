"""Deployment Infrastructure Demonstration.

This example demonstrates the deployment infrastructure for production systems:
1. Docker containerization with multi-stage builds
2. Kubernetes orchestration and deployment
3. REST API for solver access
4. Health monitoring and metrics collection
5. CI/CD automation

Author: Nonequilibrium Physics Agents
"""

import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Deployment imports
from deployment.docker import DockerBuilder, DockerImageConfig, build_docker_image
from deployment.kubernetes import (
    KubernetesDeployment, KubernetesConfig, deploy_to_kubernetes
)
from deployment.config_manager import (
    DeploymentConfig, EnvironmentConfig, validate_config, merge_configs
)
from deployment.monitoring import (
    MonitoringService, create_monitoring_service, Alert
)
from deployment.ci_cd import CICDPipeline, VersionManager


def demo_docker_containerization():
    """Demonstrate Docker containerization."""
    print("=" * 80)
    print("Docker Containerization Demo")
    print("=" * 80)

    # 1. Basic Docker configuration
    print("\n1. Basic Docker Configuration")
    print("-" * 40)

    config = DockerImageConfig(
        name="optimal-control",
        tag="latest",
        base_image="python:3.10-slim",
        install_jax=True,
        install_gpu_deps=False
    )

    print(f"Image name: {config.name}:{config.tag}")
    print(f"Base image: {config.base_image}")
    print(f"JAX enabled: {config.install_jax}")

    # 2. GPU-enabled Docker configuration
    print("\n2. GPU-Enabled Docker Configuration")
    print("-" * 40)

    gpu_config = DockerImageConfig(
        name="optimal-control-gpu",
        tag="latest",
        cuda_version="11.8.0",
        install_jax=True,
        install_gpu_deps=True
    )

    print(f"Image name: {gpu_config.name}:{gpu_config.tag}")
    print(f"CUDA version: {gpu_config.cuda_version}")
    print(f"GPU dependencies: {gpu_config.install_gpu_deps}")

    # 3. Generate Dockerfile
    print("\n3. Generated Dockerfile Preview")
    print("-" * 40)

    builder = DockerBuilder(config)
    dockerfile = builder.generate_dockerfile()

    # Show first 20 lines
    lines = dockerfile.split('\n')[:20]
    for line in lines:
        print(line)
    print("...")

    # 4. Build instructions
    print("\n4. Docker Build Instructions")
    print("-" * 40)
    print("To build the Docker image:")
    print(f"  docker build -t {config.name}:{config.tag} .")
    print("\nTo run the container:")
    print(f"  docker run -p 8000:8000 {config.name}:{config.tag}")
    print("\nTo push to registry:")
    print(f"  docker push ghcr.io/your-org/{config.name}:{config.tag}")


def demo_kubernetes_deployment():
    """Demonstrate Kubernetes deployment."""
    print("\n" + "=" * 80)
    print("Kubernetes Deployment Demo")
    print("=" * 80)

    # 1. Basic Kubernetes configuration
    print("\n1. Basic Kubernetes Configuration")
    print("-" * 40)

    config = KubernetesConfig(
        name="optimal-control",
        namespace="production",
        replicas=3,
        image="optimal-control:latest",
        ports=[8000]
    )

    print(f"Deployment name: {config.name}")
    print(f"Namespace: {config.namespace}")
    print(f"Replicas: {config.replicas}")
    print(f"Image: {config.image}")

    # 2. Generate deployment manifest
    print("\n2. Deployment Manifest")
    print("-" * 40)

    deployment = KubernetesDeployment(config)
    manifest = deployment.generate_deployment_manifest()

    print(f"Kind: {manifest['kind']}")
    print(f"Replicas: {manifest['spec']['replicas']}")
    print(f"Containers: {len(manifest['spec']['template']['spec']['containers'])}")

    # 3. Generate service manifest
    print("\n3. Service Manifest")
    print("-" * 40)

    service_manifest = deployment.generate_service_manifest(service_type="LoadBalancer")

    print(f"Kind: {service_manifest['kind']}")
    print(f"Type: {service_manifest['spec']['type']}")
    print(f"Ports: {len(service_manifest['spec']['ports'])}")

    # 4. Autoscaling configuration
    print("\n4. Horizontal Pod Autoscaler")
    print("-" * 40)

    autoscale_config = KubernetesConfig(
        name="optimal-control",
        enable_hpa=True,
        min_replicas=2,
        max_replicas=10,
        hpa_cpu_threshold=70
    )

    autoscale_deployment = KubernetesDeployment(autoscale_config)
    hpa_manifest = autoscale_deployment.generate_hpa_manifest()

    if hpa_manifest:
        print(f"Min replicas: {hpa_manifest['spec']['minReplicas']}")
        print(f"Max replicas: {hpa_manifest['spec']['maxReplicas']}")
        print(f"Target CPU: {hpa_manifest['spec']['metrics'][0]['resource']['target']['averageUtilization']}%")

    # 5. GPU deployment
    print("\n5. GPU-Enabled Deployment")
    print("-" * 40)

    gpu_config = KubernetesConfig(
        name="optimal-control-gpu",
        enable_gpu=True,
        replicas=2
    )

    gpu_deployment = KubernetesDeployment(gpu_config)
    gpu_manifest = gpu_deployment.generate_deployment_manifest()

    container_resources = gpu_manifest['spec']['template']['spec']['containers'][0]['resources']
    print(f"GPU resources requested: {container_resources['limits'].get('nvidia.com/gpu', 0)}")

    # 6. Write manifests
    print("\n6. Write Kubernetes Manifests")
    print("-" * 40)

    output_dir = Path("k8s")
    files = deployment.write_manifests(output_dir)

    print(f"Manifests written to: {output_dir}")
    for f in files:
        print(f"  - {f.name}")


def demo_configuration_management():
    """Demonstrate configuration management."""
    print("\n" + "=" * 80)
    print("Configuration Management Demo")
    print("=" * 80)

    # 1. Development configuration
    print("\n1. Development Configuration")
    print("-" * 40)

    dev_config = DeploymentConfig(
        environment="development",
        image_name="optimal-control",
        image_tag="dev",
        replicas=1,
        enable_autoscaling=False
    )

    print(f"Environment: {dev_config.environment}")
    print(f"Image: {dev_config.image_name}:{dev_config.image_tag}")
    print(f"Replicas: {dev_config.replicas}")
    print(f"Autoscaling: {dev_config.enable_autoscaling}")

    # 2. Production configuration
    print("\n2. Production Configuration")
    print("-" * 40)

    prod_config = DeploymentConfig(
        environment="production",
        image_name="optimal-control",
        image_tag="v1.0.0",
        replicas=5,
        enable_autoscaling=True,
        min_replicas=3,
        max_replicas=20
    )

    print(f"Environment: {prod_config.environment}")
    print(f"Image: {prod_config.image_name}:{prod_config.image_tag}")
    print(f"Replicas: {prod_config.replicas}")
    print(f"Autoscaling: {prod_config.enable_autoscaling}")
    print(f"Min/Max replicas: {prod_config.min_replicas}/{prod_config.max_replicas}")

    # 3. Configuration validation
    print("\n3. Configuration Validation")
    print("-" * 40)

    try:
        validate_config(dev_config)
        print("✓ Development config valid")
    except ValueError as e:
        print(f"✗ Development config invalid: {e}")

    try:
        validate_config(prod_config)
        print("✓ Production config valid")
    except ValueError as e:
        print(f"✗ Production config invalid: {e}")

    # 4. Configuration merging
    print("\n4. Configuration Merging")
    print("-" * 40)

    override = {
        "replicas": 10,
        "namespace": "staging"
    }

    merged = merge_configs(prod_config, override)

    print("Base config:")
    print(f"  Replicas: {prod_config.replicas}")
    print(f"  Namespace: {prod_config.namespace}")
    print("\nOverrides:")
    print(f"  Replicas: {override['replicas']}")
    print(f"  Namespace: {override['namespace']}")
    print("\nMerged config:")
    print(f"  Replicas: {merged.replicas}")
    print(f"  Namespace: {merged.namespace}")


def demo_monitoring():
    """Demonstrate monitoring and metrics."""
    print("\n" + "=" * 80)
    print("Monitoring and Metrics Demo")
    print("=" * 80)

    # 1. Create monitoring service
    print("\n1. Initialize Monitoring Service")
    print("-" * 40)

    def alert_handler(alert):
        print(f"  ALERT: {alert.message} (severity: {alert.severity})")

    service = create_monitoring_service(alert_handler=alert_handler)
    print("✓ Monitoring service initialized")

    # 2. Collect system metrics
    print("\n2. Collect System Metrics")
    print("-" * 40)

    service.collect_metrics()

    print("System metrics collected:")
    metrics_summary = service.get_metrics_summary()

    # Show CPU metrics
    if "system.cpu.usage_percent" in metrics_summary:
        cpu_stats = metrics_summary["system.cpu.usage_percent"]
        print(f"  CPU usage: {cpu_stats.get('mean', 0):.1f}% (avg)")

    # Show memory metrics
    if "system.memory.usage_percent" in metrics_summary:
        mem_stats = metrics_summary["system.memory.usage_percent"]
        print(f"  Memory usage: {mem_stats.get('mean', 0):.1f}% (avg)")

    # 3. Track application metrics
    print("\n3. Track Application Metrics")
    print("-" * 40)

    # Simulate API request
    service.app_monitor.record_request_start("req-1")
    time.sleep(0.1)  # Simulate processing
    service.app_monitor.record_request_end("req-1", "/api/solve", 200)

    print("✓ Recorded API request")

    # Simulate solver execution
    service.app_monitor.record_solver_execution(
        solver_type="pmp",
        duration=2.5,
        success=True,
        iterations=100
    )

    print("✓ Recorded solver execution")

    # 4. Health checks
    print("\n4. Health Status")
    print("-" * 40)

    health = service.get_health_status()

    print(f"Overall health: {'HEALTHY' if health.healthy else 'UNHEALTHY'}")
    print("\nIndividual checks:")
    for check_name, result in health.checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")

    if health.details:
        print("\nDetails:")
        for key, value in health.details.items():
            print(f"  {key}: {value}")

    # 5. Custom alerts
    print("\n5. Custom Alert Configuration")
    print("-" * 40)

    # Register custom alert
    custom_alert = Alert(
        name="app.solver.duration_seconds",
        condition=lambda x: x > 5.0,
        message="Solver execution time exceeds 5 seconds",
        severity="warning"
    )

    service.alert_manager.register_alert(custom_alert)
    print("✓ Registered custom alert for solver duration")

    # Check alerts
    triggered = service.alert_manager.check_alerts()
    if triggered:
        print(f"Triggered alerts: {len(triggered)}")
    else:
        print("No alerts triggered")


def demo_ci_cd():
    """Demonstrate CI/CD automation."""
    print("\n" + "=" * 80)
    print("CI/CD Automation Demo")
    print("=" * 80)

    # 1. Version management
    print("\n1. Version Management")
    print("-" * 40)

    current_version = "1.2.3"
    print(f"Current version: {current_version}")

    patch_version = VersionManager.bump_version(current_version, "patch")
    minor_version = VersionManager.bump_version(current_version, "minor")
    major_version = VersionManager.bump_version(current_version, "major")

    print(f"Patch bump: {patch_version}")
    print(f"Minor bump: {minor_version}")
    print(f"Major bump: {major_version}")

    # 2. Build information
    print("\n2. Build Information")
    print("-" * 40)

    project_root = Path.cwd()
    pipeline = CICDPipeline(project_root)

    try:
        build_info = pipeline.build_automation.get_build_info()
        print(f"Version: {build_info.version}")
        print(f"Commit: {build_info.commit_hash}")
        print(f"Branch: {build_info.branch}")
        print(f"Build time: {build_info.build_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Note: Build info requires git repository: {e}")

    # 3. CI pipeline steps
    print("\n3. CI Pipeline Steps")
    print("-" * 40)

    print("CI pipeline would execute:")
    print("  1. Run linting (flake8)")
    print("  2. Run type checking (mypy)")
    print("  3. Run unit tests (pytest)")
    print("  4. Generate coverage report")
    print("  5. Validate all checks pass")

    # 4. CD pipeline steps
    print("\n4. CD Pipeline Steps")
    print("-" * 40)

    print("CD pipeline would execute:")
    print("  1. Get build information")
    print("  2. Build Docker image with version tag")
    print("  3. Push image to container registry")
    print("  4. Apply Kubernetes manifests")
    print("  5. Wait for rollout completion")
    print("  6. Verify deployment health")

    # 5. Release workflow
    print("\n5. Release Workflow")
    print("-" * 40)

    print("Release workflow:")
    print("  1. Run full CI pipeline")
    print("  2. Bump version (patch/minor/major)")
    print("  3. Create git tag")
    print("  4. Build and tag Docker image")
    print("  5. Push to registry")
    print("  6. Deploy to staging")
    print("  7. Run smoke tests")
    print("  8. Deploy to production")


def demo_api_usage():
    """Demonstrate API usage patterns."""
    print("\n" + "=" * 80)
    print("REST API Usage Demo")
    print("=" * 80)

    print("\n1. API Endpoints")
    print("-" * 40)

    endpoints = {
        "GET /": "API information and version",
        "GET /health": "Health check (liveness probe)",
        "GET /ready": "Readiness check",
        "GET /api/solvers": "List available solvers",
        "POST /api/solve": "Submit solver job",
        "GET /api/job/<id>": "Get job status",
        "GET /api/jobs": "List all jobs",
    }

    for endpoint, description in endpoints.items():
        print(f"  {endpoint:30} - {description}")

    print("\n2. Example API Requests")
    print("-" * 40)

    print("\nSubmit PMP solver job:")
    print("""
    POST /api/solve
    {
        "solver_type": "pmp",
        "problem_config": {
            "n_states": 2,
            "n_controls": 1,
            "t0": 0.0,
            "tf": 1.0,
            "initial_state": [0.0, 0.0],
            "target_state": [1.0, 0.0]
        },
        "solver_config": {
            "max_iterations": 100,
            "tolerance": 1e-6
        }
    }
    """)

    print("\nCheck job status:")
    print("""
    GET /api/job/abc123
    Response:
    {
        "job_id": "abc123",
        "status": "completed",
        "progress": 100.0,
        "result": {
            "optimal_control": [...],
            "optimal_trajectory": [...],
            "cost": 0.123
        }
    }
    """)

    print("\n3. Client Usage Example")
    print("-" * 40)

    print("""
    import requests

    # Submit job
    response = requests.post(
        'http://localhost:8000/api/solve',
        json={
            'solver_type': 'pmp',
            'problem_config': {...}
        }
    )
    job_id = response.json()['job_id']

    # Poll for completion
    while True:
        status = requests.get(f'http://localhost:8000/api/job/{job_id}').json()
        if status['status'] in ['completed', 'failed']:
            break
        time.sleep(1)

    # Get result
    result = status['result']
    """)


def demo_full_deployment():
    """Demonstrate full deployment workflow."""
    print("\n" + "=" * 80)
    print("Full Deployment Workflow Demo")
    print("=" * 80)

    print("\nComplete deployment workflow:")
    print("-" * 40)

    steps = [
        ("1. Development", [
            "Write code and tests",
            "Run local tests",
            "Commit and push to feature branch"
        ]),
        ("2. Continuous Integration", [
            "GitHub Actions triggered on push",
            "Run linting and type checking",
            "Run test suite with coverage",
            "Generate test reports"
        ]),
        ("3. Docker Build", [
            "Build multi-stage Docker image",
            "Tag with commit SHA",
            "Run security scanning",
            "Push to container registry (GHCR)"
        ]),
        ("4. Staging Deployment", [
            "Update Kubernetes manifests",
            "Apply to staging namespace",
            "Wait for rollout completion",
            "Run health checks"
        ]),
        ("5. Testing", [
            "Run integration tests",
            "Run smoke tests",
            "Check monitoring metrics",
            "Verify API endpoints"
        ]),
        ("6. Production Deployment", [
            "Create release tag",
            "Apply to production namespace",
            "Perform rolling update",
            "Monitor deployment progress"
        ]),
        ("7. Monitoring", [
            "Track system metrics (CPU, memory, GPU)",
            "Track application metrics (requests, latency)",
            "Set up alerts",
            "Export metrics to monitoring system"
        ]),
        ("8. Rollback (if needed)", [
            "Detect deployment issues",
            "Trigger automatic rollback",
            "Restore previous version",
            "Investigate and fix issues"
        ])
    ]

    for step_name, actions in steps:
        print(f"\n{step_name}")
        for action in actions:
            print(f"  • {action}")

    print("\n" + "=" * 80)


def main():
    """Run all deployment demonstrations."""
    print("\n" + "=" * 80)
    print("DEPLOYMENT INFRASTRUCTURE DEMONSTRATION")
    print("Optimal Control Production Deployment System")
    print("=" * 80)

    try:
        # Run all demos
        demo_docker_containerization()
        demo_kubernetes_deployment()
        demo_configuration_management()
        demo_monitoring()
        demo_ci_cd()
        demo_api_usage()
        demo_full_deployment()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Build Docker image: docker build -t optimal-control .")
        print("2. Run locally: docker run -p 8000:8000 optimal-control")
        print("3. Deploy to Kubernetes: kubectl apply -f k8s/")
        print("4. Monitor: kubectl get pods,svc,hpa")
        print("\nFor production deployment:")
        print("1. Configure CI/CD pipeline in .github/workflows/")
        print("2. Set up container registry credentials")
        print("3. Configure Kubernetes cluster access")
        print("4. Set up monitoring and alerting")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
