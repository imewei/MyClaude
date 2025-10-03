"""CI/CD Automation Utilities.

Provides helpers for continuous integration and deployment:
- Build automation
- Test automation
- Deployment helpers
- Version management
- Release automation

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BuildInfo:
    """Information about a build."""
    version: str
    commit_hash: str
    branch: str
    build_time: datetime
    build_number: Optional[int] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class TestResult:
    """Test execution result."""
    success: bool
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage_percent: Optional[float] = None
    failed_tests: List[str] = None

    def __post_init__(self):
        if self.failed_tests is None:
            self.failed_tests = []


@dataclass
class DeploymentResult:
    """Deployment result."""
    success: bool
    environment: str
    version: str
    deployment_time: datetime
    rollback_available: bool = True
    error: Optional[str] = None


class VersionManager:
    """Manage semantic versioning."""

    VERSION_PATTERN = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?$')

    @staticmethod
    def parse_version(version: str) -> Tuple[int, int, int, Optional[str]]:
        """Parse semantic version string.

        Args:
            version: Version string (e.g., "1.2.3", "v1.2.3-beta")

        Returns:
            Tuple of (major, minor, patch, prerelease)
        """
        match = VersionManager.VERSION_PATTERN.match(version)
        if not match:
            raise ValueError(f"Invalid version format: {version}")

        major, minor, patch, prerelease = match.groups()
        return int(major), int(minor), int(patch), prerelease

    @staticmethod
    def bump_version(version: str, bump_type: str = "patch") -> str:
        """Bump version number.

        Args:
            version: Current version
            bump_type: Type of bump ('major', 'minor', 'patch')

        Returns:
            New version string
        """
        major, minor, patch, _ = VersionManager.parse_version(version)

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")

        return f"{major}.{minor}.{patch}"

    @staticmethod
    def get_latest_tag() -> Optional[str]:
        """Get latest git tag.

        Returns:
            Latest tag or None
        """
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def create_tag(version: str, message: Optional[str] = None) -> bool:
        """Create git tag.

        Args:
            version: Version to tag
            message: Optional tag message

        Returns:
            True if successful
        """
        try:
            cmd = ["git", "tag", "-a", f"v{version}"]
            if message:
                cmd.extend(["-m", message])
            else:
                cmd.extend(["-m", f"Release version {version}"])

            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create tag: {e}")
            return False


class BuildAutomation:
    """Automate build processes."""

    def __init__(self, project_root: Path):
        """Initialize build automation.

        Args:
            project_root: Root directory of project
        """
        self.project_root = project_root

    def get_build_info(self) -> BuildInfo:
        """Get current build information.

        Returns:
            Build information
        """
        # Get git information
        commit_hash = self._run_command(["git", "rev-parse", "HEAD"]).strip()
        branch = self._run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()

        # Get version from tag or default
        version_tag = VersionManager.get_latest_tag()
        if version_tag:
            version = version_tag.lstrip('v')
        else:
            version = "0.1.0"

        return BuildInfo(
            version=version,
            commit_hash=commit_hash[:8],
            branch=branch,
            build_time=datetime.now()
        )

    def build_docker_image(self, image_name: str, tag: str = "latest",
                          dockerfile: Path = None, build_args: Dict[str, str] = None) -> bool:
        """Build Docker image.

        Args:
            image_name: Image name
            tag: Image tag
            dockerfile: Path to Dockerfile
            build_args: Build arguments

        Returns:
            True if successful
        """
        dockerfile = dockerfile or self.project_root / "Dockerfile"

        cmd = ["docker", "build", "-t", f"{image_name}:{tag}", "-f", str(dockerfile)]

        # Add build arguments
        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

        cmd.append(str(self.project_root))

        try:
            self._run_command(cmd, check=True)
            logger.info(f"Successfully built image {image_name}:{tag}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Docker image: {e}")
            return False

    def push_docker_image(self, image_name: str, tag: str = "latest") -> bool:
        """Push Docker image to registry.

        Args:
            image_name: Image name
            tag: Image tag

        Returns:
            True if successful
        """
        try:
            self._run_command(["docker", "push", f"{image_name}:{tag}"], check=True)
            logger.info(f"Successfully pushed {image_name}:{tag}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push Docker image: {e}")
            return False

    def build_python_package(self) -> bool:
        """Build Python package.

        Returns:
            True if successful
        """
        try:
            # Clean previous builds
            dist_dir = self.project_root / "dist"
            if dist_dir.exists():
                import shutil
                shutil.rmtree(dist_dir)

            # Build package
            self._run_command(
                ["python", "-m", "build"],
                cwd=self.project_root,
                check=True
            )
            logger.info("Successfully built Python package")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Python package: {e}")
            return False

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None,
                    check: bool = False) -> str:
        """Run shell command.

        Args:
            cmd: Command to run
            cwd: Working directory
            check: Raise exception on error

        Returns:
            Command output
        """
        result = subprocess.run(
            cmd,
            cwd=cwd or self.project_root,
            capture_output=True,
            text=True,
            check=check
        )
        return result.stdout


class TestAutomation:
    """Automate test execution."""

    def __init__(self, project_root: Path):
        """Initialize test automation.

        Args:
            project_root: Root directory of project
        """
        self.project_root = project_root

    def run_pytest(self, test_path: Optional[Path] = None, coverage: bool = True,
                   markers: Optional[str] = None, verbose: bool = True) -> TestResult:
        """Run pytest tests.

        Args:
            test_path: Path to tests (default: tests/)
            coverage: Enable coverage reporting
            markers: Pytest markers to filter tests
            verbose: Verbose output

        Returns:
            Test results
        """
        test_path = test_path or self.project_root / "tests"

        cmd = ["pytest", str(test_path)]

        if verbose:
            cmd.append("-v")

        if coverage:
            cmd.extend(["--cov=.", "--cov-report=xml", "--cov-report=term"])

        if markers:
            cmd.extend(["-m", markers])

        # Add JSON report for parsing
        json_report = self.project_root / "test-report.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            # Parse JSON report if available
            if json_report.exists():
                with open(json_report) as f:
                    report = json.load(f)

                test_result = TestResult(
                    success=result.returncode == 0,
                    total_tests=report["summary"]["total"],
                    passed=report["summary"].get("passed", 0),
                    failed=report["summary"].get("failed", 0),
                    skipped=report["summary"].get("skipped", 0),
                    duration=report["duration"],
                    failed_tests=[t["nodeid"] for t in report.get("tests", [])
                                if t["outcome"] == "failed"]
                )

                # Get coverage if available
                coverage_file = self.project_root / "coverage.xml"
                if coverage_file.exists():
                    test_result.coverage_percent = self._parse_coverage(coverage_file)

                return test_result
            else:
                # Fallback if JSON report not available
                return TestResult(
                    success=result.returncode == 0,
                    total_tests=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    duration=0.0
                )

        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return TestResult(
                success=False,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                duration=0.0
            )

    def run_linting(self, paths: Optional[List[Path]] = None) -> bool:
        """Run code linting.

        Args:
            paths: Paths to lint (default: all Python files)

        Returns:
            True if no issues found
        """
        paths = paths or [self.project_root]

        # Run flake8
        try:
            subprocess.run(
                ["flake8"] + [str(p) for p in paths],
                cwd=self.project_root,
                check=True
            )
            logger.info("Linting passed")
            return True
        except subprocess.CalledProcessError:
            logger.error("Linting failed")
            return False

    def run_type_checking(self, paths: Optional[List[Path]] = None) -> bool:
        """Run type checking with mypy.

        Args:
            paths: Paths to check (default: all Python files)

        Returns:
            True if no type errors
        """
        paths = paths or [self.project_root]

        try:
            subprocess.run(
                ["mypy"] + [str(p) for p in paths],
                cwd=self.project_root,
                check=True
            )
            logger.info("Type checking passed")
            return True
        except subprocess.CalledProcessError:
            logger.error("Type checking failed")
            return False

    def _parse_coverage(self, coverage_file: Path) -> Optional[float]:
        """Parse coverage from XML file.

        Args:
            coverage_file: Path to coverage.xml

        Returns:
            Coverage percentage or None
        """
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            coverage = root.attrib.get("line-rate")
            if coverage:
                return float(coverage) * 100
        except Exception:
            pass
        return None


class DeploymentAutomation:
    """Automate deployment processes."""

    def __init__(self, project_root: Path):
        """Initialize deployment automation.

        Args:
            project_root: Root directory of project
        """
        self.project_root = project_root

    def deploy_kubernetes(self, manifest_dir: Path, namespace: str = "default") -> DeploymentResult:
        """Deploy to Kubernetes.

        Args:
            manifest_dir: Directory with Kubernetes manifests
            namespace: Kubernetes namespace

        Returns:
            Deployment result
        """
        try:
            # Apply manifests
            subprocess.run(
                ["kubectl", "apply", "-f", str(manifest_dir), "-n", namespace],
                check=True,
                capture_output=True
            )

            # Wait for rollout
            subprocess.run(
                ["kubectl", "rollout", "status", "deployment", "-n", namespace],
                check=True,
                capture_output=True,
                timeout=300
            )

            logger.info(f"Successfully deployed to namespace {namespace}")

            return DeploymentResult(
                success=True,
                environment=namespace,
                version="unknown",  # Would be extracted from manifest
                deployment_time=datetime.now()
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e}")
            return DeploymentResult(
                success=False,
                environment=namespace,
                version="unknown",
                deployment_time=datetime.now(),
                error=str(e)
            )

    def rollback_kubernetes(self, deployment_name: str, namespace: str = "default") -> bool:
        """Rollback Kubernetes deployment.

        Args:
            deployment_name: Name of deployment
            namespace: Kubernetes namespace

        Returns:
            True if successful
        """
        try:
            subprocess.run(
                ["kubectl", "rollout", "undo", f"deployment/{deployment_name}", "-n", namespace],
                check=True
            )
            logger.info(f"Rolled back deployment {deployment_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def scale_deployment(self, deployment_name: str, replicas: int,
                        namespace: str = "default") -> bool:
        """Scale Kubernetes deployment.

        Args:
            deployment_name: Name of deployment
            replicas: Number of replicas
            namespace: Kubernetes namespace

        Returns:
            True if successful
        """
        try:
            subprocess.run(
                ["kubectl", "scale", f"deployment/{deployment_name}",
                 f"--replicas={replicas}", "-n", namespace],
                check=True
            )
            logger.info(f"Scaled {deployment_name} to {replicas} replicas")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Scaling failed: {e}")
            return False


class CICDPipeline:
    """Complete CI/CD pipeline orchestration."""

    def __init__(self, project_root: Path):
        """Initialize CI/CD pipeline.

        Args:
            project_root: Root directory of project
        """
        self.project_root = project_root
        self.version_manager = VersionManager()
        self.build_automation = BuildAutomation(project_root)
        self.test_automation = TestAutomation(project_root)
        self.deployment_automation = DeploymentAutomation(project_root)

    def run_ci_pipeline(self, run_tests: bool = True, run_linting: bool = True,
                       run_type_check: bool = True) -> bool:
        """Run continuous integration pipeline.

        Args:
            run_tests: Run tests
            run_linting: Run linting
            run_type_check: Run type checking

        Returns:
            True if all checks pass
        """
        logger.info("Starting CI pipeline")

        success = True

        # Run linting
        if run_linting:
            logger.info("Running linting...")
            if not self.test_automation.run_linting():
                success = False

        # Run type checking
        if run_type_check:
            logger.info("Running type checking...")
            if not self.test_automation.run_type_checking():
                success = False

        # Run tests
        if run_tests:
            logger.info("Running tests...")
            test_result = self.test_automation.run_pytest()
            if not test_result.success:
                success = False
                logger.error(f"Tests failed: {test_result.failed} failures")

        if success:
            logger.info("CI pipeline completed successfully")
        else:
            logger.error("CI pipeline failed")

        return success

    def run_cd_pipeline(self, environment: str = "staging",
                       image_name: str = "optimal-control",
                       registry: Optional[str] = None) -> DeploymentResult:
        """Run continuous deployment pipeline.

        Args:
            environment: Deployment environment
            image_name: Docker image name
            registry: Container registry

        Returns:
            Deployment result
        """
        logger.info(f"Starting CD pipeline for {environment}")

        # Get build info
        build_info = self.build_automation.get_build_info()
        image_tag = f"{build_info.version}-{build_info.commit_hash}"

        # Build Docker image
        logger.info("Building Docker image...")
        full_image_name = f"{registry}/{image_name}" if registry else image_name

        if not self.build_automation.build_docker_image(full_image_name, image_tag):
            return DeploymentResult(
                success=False,
                environment=environment,
                version=build_info.version,
                deployment_time=datetime.now(),
                error="Docker build failed"
            )

        # Push image
        logger.info("Pushing Docker image...")
        if not self.build_automation.push_docker_image(full_image_name, image_tag):
            return DeploymentResult(
                success=False,
                environment=environment,
                version=build_info.version,
                deployment_time=datetime.now(),
                error="Docker push failed"
            )

        # Deploy to Kubernetes
        logger.info("Deploying to Kubernetes...")
        manifest_dir = self.project_root / "k8s" / environment
        result = self.deployment_automation.deploy_kubernetes(manifest_dir, environment)

        return result

    def create_release(self, bump_type: str = "patch") -> Optional[str]:
        """Create new release.

        Args:
            bump_type: Version bump type ('major', 'minor', 'patch')

        Returns:
            New version string or None if failed
        """
        # Get current version
        current_version = self.version_manager.get_latest_tag()
        if current_version:
            current_version = current_version.lstrip('v')
        else:
            current_version = "0.0.0"

        # Bump version
        new_version = self.version_manager.bump_version(current_version, bump_type)

        # Run CI pipeline
        if not self.run_ci_pipeline():
            logger.error("CI pipeline failed, aborting release")
            return None

        # Create git tag
        if self.version_manager.create_tag(new_version):
            logger.info(f"Created release {new_version}")
            return new_version
        else:
            logger.error("Failed to create release tag")
            return None


def create_ci_cd_pipeline(project_root: Path) -> CICDPipeline:
    """Create CI/CD pipeline.

    Args:
        project_root: Root directory of project

    Returns:
        Configured CI/CD pipeline
    """
    return CICDPipeline(project_root)
