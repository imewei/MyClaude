#!/usr/bin/env python3
"""
Validation Executor - Orchestrates validation runs across projects and scenarios.

This module executes validation scenarios against real-world projects,
collects metrics, and generates comprehensive reports.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.metrics.metrics_collector import MetricsCollector
from validation.metrics.quality_analyzer import QualityAnalyzer
from validation.benchmarks.baseline_collector import BaselineCollector
from validation.benchmarks.regression_detector import RegressionDetector
from validation.reports.report_generator import ReportGenerator


@dataclass
class ValidationResult:
    """Result of a validation run."""
    project_name: str
    scenario_name: str
    success: bool
    duration_seconds: float
    metrics: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProjectContext:
    """Context for a validation project."""
    name: str
    repo_url: str
    local_path: Path
    language: str
    size_category: str
    domain: str


class ValidationExecutor:
    """Executes validation scenarios across multiple projects."""

    def __init__(
        self,
        validation_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        parallel_jobs: int = 3,
    ):
        """Initialize validation executor.

        Args:
            validation_dir: Directory containing validation configuration
            cache_dir: Directory for caching cloned projects
            parallel_jobs: Number of parallel validation jobs
        """
        self.validation_dir = validation_dir or Path(__file__).parent
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "claude_validation_cache"
        self.parallel_jobs = parallel_jobs

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.projects = self._load_projects()
        self.scenarios = self._load_scenarios()

        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.quality_analyzer = QualityAnalyzer()
        self.baseline_collector = BaselineCollector()
        self.regression_detector = RegressionDetector()
        self.report_generator = ReportGenerator()

        # Results storage
        self.results: List[ValidationResult] = []

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_dir = self.validation_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_projects(self) -> Dict[str, Any]:
        """Load validation projects configuration."""
        projects_file = self.validation_dir / "suite" / "validation_projects.yaml"
        with open(projects_file) as f:
            config = yaml.safe_load(f)
        return config.get("validation_projects", {})

    def _load_scenarios(self) -> Dict[str, Any]:
        """Load validation scenarios configuration."""
        scenarios_file = self.validation_dir / "suite" / "validation_scenarios.yaml"
        with open(scenarios_file) as f:
            config = yaml.safe_load(f)
        return config.get("validation_scenarios", {})

    def clone_project(self, project: Dict[str, Any]) -> Optional[Path]:
        """Clone a validation project.

        Args:
            project: Project configuration dictionary

        Returns:
            Path to cloned project or None if clone fails
        """
        project_name = project["name"]
        repo_url = project["repo"]

        # Check cache
        cached_path = self.cache_dir / project_name
        if cached_path.exists():
            self.logger.info(f"Using cached project: {project_name}")
            return cached_path

        self.logger.info(f"Cloning project: {project_name} from {repo_url}")

        try:
            # Shallow clone for faster cloning
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(cached_path)],
                check=True,
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            self.logger.info(f"Successfully cloned: {project_name}")
            return cached_path

        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout cloning {project_name}")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone {project_name}: {e.stderr.decode()}")
            return None

    def run_scenario_step(
        self,
        step: Dict[str, Any],
        project_path: Path,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single scenario step.

        Args:
            step: Step configuration
            project_path: Path to project
            context: Execution context

        Returns:
            Step results dictionary
        """
        action = step["action"]
        command = step.get("command")
        options = step.get("options", [])

        self.logger.info(f"Running step: {action} (command: {command})")

        step_start = time.time()

        try:
            if command:
                # Build command
                cmd_parts = [command] + options + [str(project_path)]

                # Execute command (simulated - would integrate with actual command system)
                result = self._execute_command(cmd_parts, project_path)

                step_duration = time.time() - step_start

                return {
                    "action": action,
                    "command": command,
                    "success": result.get("success", False),
                    "duration": step_duration,
                    "output": result.get("output", ""),
                    "metrics": result.get("metrics", {})
                }
            else:
                # Non-command step (e.g., measure_improvement)
                metrics = step.get("metrics", [])
                collected_metrics = self.metrics_collector.collect(project_path, metrics)

                step_duration = time.time() - step_start

                return {
                    "action": action,
                    "success": True,
                    "duration": step_duration,
                    "metrics": collected_metrics
                }

        except Exception as e:
            self.logger.error(f"Step {action} failed: {e}")
            return {
                "action": action,
                "success": False,
                "error": str(e),
                "duration": time.time() - step_start
            }

    def _execute_command(self, cmd_parts: List[str], cwd: Path) -> Dict[str, Any]:
        """Execute a command (placeholder for actual integration).

        Args:
            cmd_parts: Command parts
            cwd: Working directory

        Returns:
            Command execution results
        """
        # This would integrate with the actual command execution system
        # For now, it's a placeholder that simulates execution

        self.logger.info(f"Executing: {' '.join(cmd_parts)}")

        # Simulate command execution
        time.sleep(0.5)  # Simulate work

        return {
            "success": True,
            "output": f"Executed: {' '.join(cmd_parts)}",
            "metrics": {
                "execution_time": 0.5,
                "memory_mb": 100
            }
        }

    def run_scenario(
        self,
        scenario_name: str,
        scenario_config: Dict[str, Any],
        project_context: ProjectContext
    ) -> ValidationResult:
        """Run a validation scenario on a project.

        Args:
            scenario_name: Name of scenario
            scenario_config: Scenario configuration
            project_context: Project context

        Returns:
            ValidationResult object
        """
        self.logger.info(
            f"Running scenario '{scenario_name}' on project '{project_context.name}'"
        )

        start_time = time.time()
        errors = []
        warnings = []
        all_metrics = {}

        try:
            steps = scenario_config.get("steps", [])
            context = {}

            for step in steps:
                step_result = self.run_scenario_step(
                    step,
                    project_context.local_path,
                    context
                )

                # Update context with step results
                context[step["action"]] = step_result

                # Collect metrics
                if "metrics" in step_result:
                    all_metrics.update(step_result["metrics"])

                # Check for errors
                if not step_result.get("success", False):
                    error_msg = f"Step {step['action']} failed"
                    if "error" in step_result:
                        error_msg += f": {step_result['error']}"
                    errors.append(error_msg)

            # Check success criteria
            success_criteria = scenario_config.get("success_criteria", {})
            criteria_met = self._check_success_criteria(
                success_criteria,
                all_metrics,
                context
            )

            if not criteria_met:
                warnings.append("Not all success criteria were met")

            duration = time.time() - start_time
            success = len(errors) == 0 and criteria_met

            return ValidationResult(
                project_name=project_context.name,
                scenario_name=scenario_name,
                success=success,
                duration_seconds=duration,
                metrics=all_metrics,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            self.logger.error(f"Scenario execution failed: {e}", exc_info=True)
            duration = time.time() - start_time

            return ValidationResult(
                project_name=project_context.name,
                scenario_name=scenario_name,
                success=False,
                duration_seconds=duration,
                metrics=all_metrics,
                errors=[str(e)]
            )

    def _check_success_criteria(
        self,
        criteria: Dict[str, Any],
        metrics: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check if success criteria are met.

        Args:
            criteria: Success criteria dictionary
            metrics: Collected metrics
            context: Execution context

        Returns:
            True if all criteria met
        """
        for criterion, threshold in criteria.items():
            if criterion not in metrics and criterion not in context:
                self.logger.warning(f"Criterion '{criterion}' not found in metrics or context")
                continue

            value = metrics.get(criterion) or context.get(criterion)

            # Parse threshold (e.g., ">= 80", "> 0", "true")
            if not self._evaluate_criterion(value, threshold):
                self.logger.warning(
                    f"Criterion '{criterion}' not met: {value} vs {threshold}"
                )
                return False

        return True

    def _evaluate_criterion(self, value: Any, threshold: str) -> bool:
        """Evaluate a single criterion.

        Args:
            value: Actual value
            threshold: Threshold expression

        Returns:
            True if criterion met
        """
        threshold = str(threshold).strip()

        # Boolean criteria
        if threshold.lower() == "true":
            return bool(value)
        if threshold.lower() == "false":
            return not bool(value)

        # Numeric comparisons
        try:
            if threshold.startswith(">="):
                return float(value) >= float(threshold[2:].strip())
            elif threshold.startswith("<="):
                return float(value) <= float(threshold[2:].strip())
            elif threshold.startswith(">"):
                return float(value) > float(threshold[1:].strip())
            elif threshold.startswith("<"):
                return float(value) < float(threshold[1:].strip())
            elif threshold.startswith("=="):
                return float(value) == float(threshold[2:].strip())
        except (ValueError, TypeError):
            pass

        # String equality
        return str(value) == threshold

    async def run_validation_async(
        self,
        project_filter: Optional[Set[str]] = None,
        scenario_filter: Optional[Set[str]] = None,
        size_filter: Optional[Set[str]] = None
    ) -> List[ValidationResult]:
        """Run validation asynchronously.

        Args:
            project_filter: Set of project names to run (None = all)
            scenario_filter: Set of scenario names to run (None = all)
            size_filter: Set of size categories (small, medium, large, enterprise)

        Returns:
            List of validation results
        """
        tasks = []

        # Build list of validation tasks
        for size_category, projects in self.projects.items():
            if size_filter and size_category not in size_filter:
                continue

            if not isinstance(projects, list):
                continue

            for project in projects:
                project_name = project["name"]

                if project_filter and project_name not in project_filter:
                    continue

                # Clone project
                project_path = self.clone_project(project)
                if not project_path:
                    self.logger.error(f"Skipping {project_name} - clone failed")
                    continue

                # Create project context
                context = ProjectContext(
                    name=project_name,
                    repo_url=project["repo"],
                    local_path=project_path,
                    language=project.get("language", "python"),
                    size_category=size_category,
                    domain=project.get("domain", "unknown")
                )

                # Queue scenario tasks
                for scenario_name, scenario_config in self.scenarios.items():
                    if scenario_filter and scenario_name not in scenario_filter:
                        continue

                    # Create task
                    task = asyncio.create_task(
                        self._run_scenario_async(scenario_name, scenario_config, context)
                    )
                    tasks.append(task)

        # Execute tasks with concurrency limit
        results = []
        semaphore = asyncio.Semaphore(self.parallel_jobs)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        for task in asyncio.as_completed([run_with_semaphore(t) for t in tasks]):
            result = await task
            results.append(result)
            self.results.append(result)

            # Log progress
            total = len(tasks)
            completed = len(results)
            self.logger.info(f"Progress: {completed}/{total} validations completed")

        return results

    async def _run_scenario_async(
        self,
        scenario_name: str,
        scenario_config: Dict[str, Any],
        project_context: ProjectContext
    ) -> ValidationResult:
        """Run scenario asynchronously.

        Args:
            scenario_name: Scenario name
            scenario_config: Scenario configuration
            project_context: Project context

        Returns:
            ValidationResult
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.run_scenario,
            scenario_name,
            scenario_config,
            project_context
        )

    def run_validation(
        self,
        project_filter: Optional[Set[str]] = None,
        scenario_filter: Optional[Set[str]] = None,
        size_filter: Optional[Set[str]] = None
    ) -> List[ValidationResult]:
        """Run validation synchronously.

        Args:
            project_filter: Set of project names to run (None = all)
            scenario_filter: Set of scenario names to run (None = all)
            size_filter: Set of size categories

        Returns:
            List of validation results
        """
        return asyncio.run(
            self.run_validation_async(project_filter, scenario_filter, size_filter)
        )

    def generate_report(
        self,
        output_dir: Optional[Path] = None,
        formats: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """Generate validation reports.

        Args:
            output_dir: Output directory for reports
            formats: List of formats (html, pdf, json, markdown)

        Returns:
            Dictionary mapping format to report path
        """
        if not output_dir:
            output_dir = self.validation_dir / "reports" / datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir.mkdir(parents=True, exist_ok=True)

        formats = formats or ["html", "json", "markdown"]

        return self.report_generator.generate(
            self.results,
            output_dir,
            formats
        )

    def cleanup_cache(self, keep_recent: int = 5) -> None:
        """Clean up old cached projects.

        Args:
            keep_recent: Number of recent projects to keep per project
        """
        self.logger.info("Cleaning up validation cache")

        # For now, simple cleanup - could be more sophisticated
        if self.cache_dir.exists():
            # Get all project directories
            projects = list(self.cache_dir.iterdir())

            # Sort by modification time
            projects.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Keep only recent ones (simplified logic)
            for old_project in projects[keep_recent:]:
                self.logger.info(f"Removing old cache: {old_project.name}")
                shutil.rmtree(old_project)


def main():
    """Main entry point for validation executor."""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Code Validation Executor")
    parser.add_argument(
        "--projects",
        nargs="+",
        help="Specific projects to validate"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Specific scenarios to run"
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "enterprise"],
        nargs="+",
        help="Filter by project size"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of parallel jobs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for reports"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["html", "pdf", "json", "markdown"],
        default=["html", "json", "markdown"],
        help="Report formats to generate"
    )
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Clean up old cached projects"
    )

    args = parser.parse_args()

    # Create executor
    executor = ValidationExecutor(parallel_jobs=args.parallel)

    # Cleanup cache if requested
    if args.cleanup_cache:
        executor.cleanup_cache()

    # Convert filters to sets
    project_filter = set(args.projects) if args.projects else None
    scenario_filter = set(args.scenarios) if args.scenarios else None
    size_filter = set(args.size) if args.size else None

    # Run validation
    print("Starting validation...")
    print(f"Projects: {args.projects or 'all'}")
    print(f"Scenarios: {args.scenarios or 'all'}")
    print(f"Size filter: {args.size or 'all'}")
    print(f"Parallel jobs: {args.parallel}")
    print()

    results = executor.run_validation(
        project_filter=project_filter,
        scenario_filter=scenario_filter,
        size_filter=size_filter
    )

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total validations: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.success)}")
    print(f"Failed: {sum(1 for r in results if not r.success)}")
    print(f"Total duration: {sum(r.duration_seconds for r in results):.2f}s")
    print()

    # Generate reports
    print("Generating reports...")
    report_paths = executor.generate_report(
        output_dir=args.output_dir,
        formats=args.formats
    )

    print("\nReports generated:")
    for format_name, path in report_paths.items():
        print(f"  {format_name}: {path}")

    # Exit with appropriate code
    success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
    sys.exit(0 if success_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()