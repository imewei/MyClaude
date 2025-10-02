"""Tests for deployment infrastructure.

Author: Nonequilibrium Physics Agents
"""

import pytest
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import deployment modules
from deployment.docker import DockerBuilder, DockerImageConfig, build_docker_image
from deployment.kubernetes import (
    KubernetesDeployment, KubernetesService, KubernetesConfig
)
from deployment.config_manager import (
    DeploymentConfig, EnvironmentConfig, load_config, validate_config, merge_configs
)
from deployment.monitoring import (
    MetricsCollector, SystemMonitor, ApplicationMonitor, HealthChecker,
    Alert, AlertManager, MonitoringService
)
from deployment.ci_cd import (
    VersionManager, BuildAutomation, TestAutomation, DeploymentAutomation,
    CICDPipeline, BuildInfo, TestResult
)


class TestDockerBuilder:
    """Tests for Docker builder."""

    def test_docker_config_creation(self):
        """Test Docker configuration creation."""
        config = DockerImageConfig(
            name="test-image",
            tag="v1.0.0",
            base_image="python:3.10-slim"
        )

        assert config.name == "test-image"
        assert config.tag == "v1.0.0"
        assert config.base_image == "python:3.10-slim"

    def test_docker_config_gpu(self):
        """Test Docker configuration with GPU."""
        config = DockerImageConfig(
            name="test-image",
            cuda_version="11.8.0",
            install_gpu_deps=True
        )

        assert config.cuda_version == "11.8.0"
        assert config.install_gpu_deps is True

    def test_dockerfile_generation(self):
        """Test Dockerfile generation."""
        config = DockerImageConfig(
            name="test-image",
            base_image="python:3.10-slim",
            install_jax=True
        )

        builder = DockerBuilder(config)
        dockerfile = builder.generate_dockerfile()

        assert "FROM python:3.10-slim" in dockerfile
        assert "WORKDIR /app" in dockerfile
        assert "USER appuser" in dockerfile

    def test_dockerfile_generation_gpu(self):
        """Test Dockerfile generation with GPU."""
        config = DockerImageConfig(
            name="test-image",
            cuda_version="11.8.0",
            install_gpu_deps=True
        )

        builder = DockerBuilder(config)
        dockerfile = builder.generate_dockerfile()

        assert "nvidia/cuda" in dockerfile
        assert "11.8.0" in dockerfile

    @patch('subprocess.run')
    def test_docker_build_success(self, mock_run):
        """Test Docker build success."""
        mock_run.return_value = Mock(returncode=0)

        config = DockerImageConfig(name="test-image", tag="latest")
        builder = DockerBuilder(config)

        result = builder.build()
        assert result is True
        assert mock_run.called

    @patch('subprocess.run')
    def test_docker_build_failure(self, mock_run):
        """Test Docker build failure."""
        mock_run.side_effect = Exception("Build failed")

        config = DockerImageConfig(name="test-image", tag="latest")
        builder = DockerBuilder(config)

        result = builder.build()
        assert result is False


class TestKubernetesDeployment:
    """Tests for Kubernetes deployment."""

    def test_kubernetes_config_creation(self):
        """Test Kubernetes configuration creation."""
        config = KubernetesConfig(
            name="test-app",
            namespace="production",
            replicas=5,
            image="test-image:latest"
        )

        assert config.name == "test-app"
        assert config.namespace == "production"
        assert config.replicas == 5
        assert config.image == "test-image:latest"

    def test_deployment_manifest_generation(self):
        """Test deployment manifest generation."""
        config = KubernetesConfig(
            name="test-app",
            replicas=3,
            image="test-image:latest"
        )

        deployment = KubernetesDeployment(config)
        manifest = deployment.generate_deployment_manifest()

        assert manifest["kind"] == "Deployment"
        assert manifest["metadata"]["name"] == "test-app"
        assert manifest["spec"]["replicas"] == 3

    def test_service_manifest_generation(self):
        """Test service manifest generation."""
        config = KubernetesConfig(
            name="test-app",
            ports=[8000, 8080]
        )

        deployment = KubernetesDeployment(config)
        manifest = deployment.generate_service_manifest(service_type="LoadBalancer")

        assert manifest["kind"] == "Service"
        assert manifest["spec"]["type"] == "LoadBalancer"
        assert len(manifest["spec"]["ports"]) == 2

    def test_hpa_manifest_generation(self):
        """Test HPA manifest generation."""
        config = KubernetesConfig(
            name="test-app",
            enable_hpa=True,
            min_replicas=2,
            max_replicas=10
        )

        deployment = KubernetesDeployment(config)
        manifest = deployment.generate_hpa_manifest()

        assert manifest is not None
        assert manifest["kind"] == "HorizontalPodAutoscaler"
        assert manifest["spec"]["minReplicas"] == 2
        assert manifest["spec"]["maxReplicas"] == 10

    def test_gpu_deployment_manifest(self):
        """Test deployment manifest with GPU."""
        config = KubernetesConfig(
            name="test-app",
            enable_gpu=True
        )

        deployment = KubernetesDeployment(config)
        manifest = deployment.generate_deployment_manifest()

        resources = manifest["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "nvidia.com/gpu" in resources["limits"]

    def test_manifest_writing(self):
        """Test writing manifests to files."""
        config = KubernetesConfig(name="test-app")
        deployment = KubernetesDeployment(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            files = deployment.write_manifests(output_dir)

            assert len(files) >= 2  # At least deployment and service
            assert all(f.exists() for f in files)


class TestConfigManager:
    """Tests for configuration management."""

    def test_deployment_config_creation(self):
        """Test deployment config creation."""
        config = DeploymentConfig(
            environment="production",
            image_name="test-image",
            image_tag="v1.0.0",
            replicas=5
        )

        assert config.environment == "production"
        assert config.replicas == 5

    def test_config_validation_success(self):
        """Test config validation success."""
        config = DeploymentConfig(
            environment="production",
            image_name="test-image",
            image_tag="v1.0.0",
            replicas=3,
            min_replicas=2,
            max_replicas=10
        )

        assert validate_config(config) is True

    def test_config_validation_failure_replicas(self):
        """Test config validation failure for replicas."""
        config = DeploymentConfig(
            environment="production",
            image_name="test-image",
            image_tag="v1.0.0",
            replicas=0
        )

        with pytest.raises(ValueError, match="replicas must be >= 1"):
            validate_config(config)

    def test_config_validation_failure_autoscaling(self):
        """Test config validation failure for autoscaling."""
        config = DeploymentConfig(
            environment="production",
            image_name="test-image",
            image_tag="v1.0.0",
            enable_autoscaling=True,
            min_replicas=10,
            max_replicas=5
        )

        with pytest.raises(ValueError, match="max_replicas must be >= min_replicas"):
            validate_config(config)

    def test_config_loading(self):
        """Test config loading from file."""
        config_data = {
            "production": {
                "environment": "production",
                "image_name": "test-image",
                "image_tag": "v1.0.0",
                "replicas": 5
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = load_config(config_path, "production")
            assert config.environment == "production"
            assert config.replicas == 5
        finally:
            config_path.unlink()

    def test_config_merging(self):
        """Test config merging."""
        base = DeploymentConfig(
            environment="staging",
            image_name="test-image",
            image_tag="v1.0.0",
            replicas=3
        )

        override = {"replicas": 10, "namespace": "custom"}
        merged = merge_configs(base, override)

        assert merged.replicas == 10
        assert merged.namespace == "custom"
        assert merged.image_name == "test-image"


class TestMonitoring:
    """Tests for monitoring."""

    def test_metrics_collector_creation(self):
        """Test metrics collector creation."""
        collector = MetricsCollector()
        assert collector is not None

    def test_metric_recording(self):
        """Test metric recording."""
        collector = MetricsCollector()
        collector.record("test.metric", 42.0, labels={"env": "test"}, unit="ms")

        latest = collector.get_latest("test.metric")
        assert latest is not None
        assert latest.value == 42.0
        assert latest.labels["env"] == "test"

    def test_metric_statistics(self):
        """Test metric statistics."""
        collector = MetricsCollector()

        # Record multiple values
        for i in range(100):
            collector.record("test.metric", float(i))

        stats = collector.get_statistics("test.metric")
        assert stats["min"] == 0.0
        assert stats["max"] == 99.0
        assert stats["count"] == 100

    def test_system_monitor_cpu(self):
        """Test system monitor CPU metrics."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)

        monitor.collect_cpu_metrics()

        cpu_metric = collector.get_latest("system.cpu.usage_percent")
        assert cpu_metric is not None
        assert 0.0 <= cpu_metric.value <= 100.0

    def test_system_monitor_memory(self):
        """Test system monitor memory metrics."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)

        monitor.collect_memory_metrics()

        mem_metric = collector.get_latest("system.memory.usage_percent")
        assert mem_metric is not None
        assert 0.0 <= mem_metric.value <= 100.0

    def test_application_monitor_request(self):
        """Test application monitor request tracking."""
        collector = MetricsCollector()
        monitor = ApplicationMonitor(collector)

        # Start and end request
        monitor.record_request_start("req-1")
        monitor.record_request_end("req-1", "/api/solve", 200)

        duration_metric = collector.get_latest("app.request.duration_seconds")
        assert duration_metric is not None
        assert duration_metric.value >= 0.0

    def test_application_monitor_solver(self):
        """Test application monitor solver tracking."""
        collector = MetricsCollector()
        monitor = ApplicationMonitor(collector)

        monitor.record_solver_execution("pmp", 1.5, True, iterations=100)

        duration_metric = collector.get_latest("app.solver.duration_seconds")
        assert duration_metric is not None
        assert duration_metric.value == 1.5

    def test_health_checker(self):
        """Test health checker."""
        collector = MetricsCollector()
        checker = HealthChecker(collector)

        # Record some metrics
        collector.record("system.cpu.usage_percent", 50.0)
        collector.record("system.memory.usage_percent", 60.0)
        collector.record("system.disk.usage_percent", 40.0)

        status = checker.run_all_checks()
        assert status.healthy is True
        assert status.checks["cpu"] is True
        assert status.checks["memory"] is True

    def test_health_checker_unhealthy(self):
        """Test health checker with unhealthy state."""
        collector = MetricsCollector()
        checker = HealthChecker(collector)

        # Record high CPU usage
        collector.record("system.cpu.usage_percent", 95.0)

        status = checker.run_all_checks()
        assert status.checks["cpu"] is False

    def test_alert_manager(self):
        """Test alert manager."""
        collector = MetricsCollector()
        manager = AlertManager(collector)

        # Register alert
        triggered_alerts = []

        def handler(alert):
            triggered_alerts.append(alert)

        manager.register_handler(handler)

        alert = Alert(
            name="test.metric",
            condition=lambda x: x > 50,
            message="Test metric too high"
        )
        manager.register_alert(alert)

        # Record metric that triggers alert
        collector.record("test.metric", 75.0)
        manager.check_alerts()

        assert len(triggered_alerts) == 1

    def test_monitoring_service(self):
        """Test monitoring service."""
        service = MonitoringService()

        # Collect metrics
        service.collect_metrics()

        # Get health status
        status = service.get_health_status()
        assert status is not None

        # Get metrics summary
        summary = service.get_metrics_summary()
        assert isinstance(summary, dict)


class TestCICD:
    """Tests for CI/CD automation."""

    def test_version_parsing(self):
        """Test version parsing."""
        major, minor, patch, prerelease = VersionManager.parse_version("1.2.3")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert prerelease is None

    def test_version_parsing_with_prerelease(self):
        """Test version parsing with prerelease."""
        major, minor, patch, prerelease = VersionManager.parse_version("1.2.3-beta.1")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert prerelease == "beta.1"

    def test_version_bump_patch(self):
        """Test version bump patch."""
        new_version = VersionManager.bump_version("1.2.3", "patch")
        assert new_version == "1.2.4"

    def test_version_bump_minor(self):
        """Test version bump minor."""
        new_version = VersionManager.bump_version("1.2.3", "minor")
        assert new_version == "1.3.0"

    def test_version_bump_major(self):
        """Test version bump major."""
        new_version = VersionManager.bump_version("1.2.3", "major")
        assert new_version == "2.0.0"

    def test_build_info_creation(self):
        """Test build info creation."""
        build_info = BuildInfo(
            version="1.0.0",
            commit_hash="abc123",
            branch="main",
            build_time=datetime.now()
        )

        assert build_info.version == "1.0.0"
        assert build_info.commit_hash == "abc123"

    def test_test_result_creation(self):
        """Test result creation."""
        result = TestResult(
            success=True,
            total_tests=100,
            passed=95,
            failed=5,
            skipped=0,
            duration=10.5,
            coverage_percent=85.0
        )

        assert result.success is True
        assert result.total_tests == 100
        assert result.coverage_percent == 85.0

    @patch('subprocess.run')
    def test_build_automation_docker(self, mock_run):
        """Test build automation Docker build."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            automation = BuildAutomation(Path(tmpdir))
            result = automation.build_docker_image("test-image", "latest")

            assert result is True

    def test_ci_cd_pipeline_creation(self):
        """Test CI/CD pipeline creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = CICDPipeline(Path(tmpdir))

            assert pipeline.version_manager is not None
            assert pipeline.build_automation is not None
            assert pipeline.test_automation is not None


class TestIntegration:
    """Integration tests for deployment infrastructure."""

    def test_full_deployment_workflow(self):
        """Test full deployment workflow."""
        # Create config
        config = KubernetesConfig(
            name="test-app",
            replicas=3,
            image="test-image:latest"
        )

        # Generate manifests
        deployment = KubernetesDeployment(config)
        manifest = deployment.generate_deployment_manifest()

        # Validate manifest structure
        assert manifest["kind"] == "Deployment"
        assert manifest["spec"]["replicas"] == 3

        # Validate config
        deploy_config = DeploymentConfig(
            environment="production",
            image_name="test-image",
            image_tag="latest",
            replicas=3
        )
        assert validate_config(deploy_config) is True

    def test_monitoring_integration(self):
        """Test monitoring integration."""
        service = MonitoringService()

        # Collect system metrics
        service.collect_metrics()

        # Track application metrics
        service.app_monitor.record_request_start("req-1")
        service.app_monitor.record_request_end("req-1", "/api/solve", 200)

        service.app_monitor.record_solver_execution("pmp", 2.5, True, 100)

        # Get health status
        status = service.get_health_status()
        assert status is not None

        # Check metrics were recorded
        summary = service.get_metrics_summary()
        assert len(summary) > 0

    def test_ci_cd_integration(self):
        """Test CI/CD integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = CICDPipeline(Path(tmpdir))

            # Test version management
            new_version = pipeline.version_manager.bump_version("1.0.0", "minor")
            assert new_version == "1.1.0"

            # Test build info
            build_info = pipeline.build_automation.get_build_info()
            assert build_info.branch is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
