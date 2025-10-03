#!/usr/bin/env python3
"""
Example Command Implementation
==============================

Complete example showing how to implement a command using the
Unified Command Executor Framework.

This example demonstrates:
- Framework integration
- Agent selection
- Safety features (dry-run, backup)
- Parallel execution
- Caching
- Validation
- Progress tracking

Author: Claude Code Framework
Version: 2.0
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Framework imports
from framework import (
    BaseCommandExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionPhase,
    CommandCategory,
    ValidationRule
)

from agent_system import AgentSelector
from safety_manager import DryRunExecutor, ChangeType
from performance import ParallelExecutor, WorkerTask, ExecutionMode


class ExampleCodeAnalyzerExecutor(BaseCommandExecutor):
    """
    Example command: Analyze Python code quality

    Features demonstrated:
    - Validation rules
    - Agent selection
    - Dry run support
    - Parallel execution
    - Caching
    - Progress tracking
    """

    def __init__(self):
        super().__init__(
            command_name="example-analyzer",
            category=CommandCategory.ANALYSIS,
            version="2.0"
        )

    # ========================================================================
    # Required Methods
    # ========================================================================

    def validate_prerequisites(self, context: ExecutionContext) -> Tuple[bool, List[str]]:
        """
        Validate prerequisites for command execution.

        Checks:
        1. Target path exists
        2. Target contains Python files
        3. Required tools available
        """
        errors = []

        # Check target path
        target = context.args.get('target')
        if not target:
            errors.append("Target path is required")
            return False, errors

        target_path = Path(target)
        if not target_path.exists():
            errors.append(f"Target path does not exist: {target}")
            return False, errors

        # Check for Python files
        python_files = list(target_path.rglob("*.py"))
        if not python_files:
            errors.append(f"No Python files found in {target}")
            return False, errors

        # Store for later use
        context.metadata['python_files'] = python_files
        context.metadata['file_count'] = len(python_files)

        return True, []

    def execute_command(self, context: ExecutionContext) -> ExecutionResult:
        """
        Main execution logic.

        Steps:
        1. Select optimal agents
        2. Analyze files (with caching and parallel execution)
        3. Aggregate results
        4. Generate report
        """
        start_time = time.time()

        # Get files from context
        python_files = context.metadata.get('python_files', [])

        self.logger.info(f"Analyzing {len(python_files)} Python files")

        # Step 1: Select agents
        agents = self._select_agents(context)
        self.logger.info(f"Selected {len(agents)} agents: {[a.name for a in agents]}")

        # Step 2: Analyze files
        if context.parallel:
            results = self._analyze_files_parallel(python_files, context)
        else:
            results = self._analyze_files_sequential(python_files, context)

        # Step 3: Aggregate results
        analysis_summary = self._aggregate_results(results)

        # Step 4: Generate report
        duration = time.time() - start_time

        return ExecutionResult(
            success=True,
            command=self.command_name,
            duration=duration,
            phase=ExecutionPhase.EXECUTION,
            summary=self._format_summary(analysis_summary),
            details={
                "files_analyzed": len(python_files),
                "agents_used": [a.name for a in agents],
                "issues_found": analysis_summary.get('total_issues', 0),
                "analysis": analysis_summary
            },
            metrics={
                "files": len(python_files),
                "issues": analysis_summary.get('total_issues', 0),
                "duration": duration,
                "files_per_second": len(python_files) / duration if duration > 0 else 0
            }
        )

    # ========================================================================
    # Optional Methods
    # ========================================================================

    def get_validation_rules(self) -> List[ValidationRule]:
        """Define validation rules"""
        return [
            ValidationRule(
                name="target_path",
                validator=lambda ctx: (
                    (True, None) if ctx.args.get('target')
                    else (False, "Target path required")
                ),
                severity="error"
            ),
            ValidationRule(
                name="valid_options",
                validator=self._validate_options,
                severity="warning"
            )
        ]

    def pre_execution_hook(self, context: ExecutionContext) -> bool:
        """Pre-execution hook"""
        self.logger.info("Pre-execution: Setting up analysis")

        # If implementing fixes, create backup
        if context.implement and not context.dry_run:
            self.logger.info("Creating backup before implementing fixes")
            backup_id = self.backup_manager.create_backup(
                context.work_dir,
                f"{self.command_name}_fixes"
            )
            context.metadata['backup_id'] = backup_id

        return True

    def post_execution_hook(
        self,
        context: ExecutionContext,
        result: ExecutionResult
    ) -> ExecutionResult:
        """Post-execution hook"""
        self.logger.info("Post-execution: Finalizing analysis")

        # Add recommendations
        if result.success:
            recommendations = self._generate_recommendations(result.details)
            result.details['recommendations'] = recommendations

        # Validate if requested
        if context.validate and result.success:
            self.logger.info("Validating results")
            validation_passed = self._validate_results(result)
            if not validation_passed:
                result.warnings.append("Result validation identified concerns")

        return result

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _select_agents(self, context: ExecutionContext):
        """Select optimal agents based on context"""
        selector = AgentSelector()

        # Build context for agent selection
        agent_context = {
            "task_type": "analysis",
            "work_dir": str(context.work_dir),
            "languages": ["python"],
            "frameworks": self._detect_frameworks(context.work_dir),
            "description": "Python code quality analysis"
        }

        # Select agents (respecting user preference)
        agents_arg = context.args.get('agents', 'auto')
        agents = selector.select_agents(
            agent_context,
            mode=agents_arg,
            max_agents=5
        )

        return agents

    def _detect_frameworks(self, work_dir: Path) -> List[str]:
        """Detect Python frameworks in use"""
        frameworks = []

        # Check requirements.txt
        req_file = work_dir / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text().lower()
            if "django" in content:
                frameworks.append("django")
            if "flask" in content:
                frameworks.append("flask")
            if "fastapi" in content:
                frameworks.append("fastapi")
            if "numpy" in content:
                frameworks.append("numpy")
            if "pandas" in content:
                frameworks.append("pandas")

        return frameworks

    def _analyze_files_sequential(
        self,
        files: List[Path],
        context: ExecutionContext
    ) -> List[Dict[str, Any]]:
        """Analyze files sequentially"""
        results = []

        self.progress_tracker.start("Analyzing files", len(files))

        for i, file_path in enumerate(files):
            # Update progress
            self.progress_tracker.update(f"Analyzing {file_path.name}", i + 1)

            # Analyze file (with caching)
            result = self._analyze_file(file_path, context)
            results.append(result)

        self.progress_tracker.complete()

        return results

    def _analyze_files_parallel(
        self,
        files: List[Path],
        context: ExecutionContext
    ) -> List[Dict[str, Any]]:
        """Analyze files in parallel"""
        self.logger.info(f"Analyzing {len(files)} files in parallel")

        # Create parallel executor
        executor = ParallelExecutor(
            mode=ExecutionMode.PARALLEL_THREAD,
            max_workers=8
        )

        # Create tasks
        tasks = [
            WorkerTask(
                task_id=f"analyze_{i}",
                function=self._analyze_file,
                args=(file_path, context)
            )
            for i, file_path in enumerate(files)
        ]

        # Execute in parallel with progress tracking
        def progress_callback(completed, total):
            self.progress_tracker.update(
                f"Analyzed {completed}/{total} files",
                completed
            )

        self.progress_tracker.start("Analyzing files (parallel)", len(files))

        worker_results = executor.execute_parallel(tasks, progress_callback)

        self.progress_tracker.complete()

        # Extract results
        results = [wr.result for wr in worker_results if wr.success]

        return results

    def _analyze_file(
        self,
        file_path: Path,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Analyze a single file (with caching)"""
        # Generate cache key
        cache_key = f"{file_path}:{file_path.stat().st_mtime}"

        # Check cache
        cached = self.cache_manager.get(cache_key, level="analysis")
        if cached:
            self.logger.debug(f"Cache hit: {file_path.name}")
            return cached

        # Perform analysis
        self.logger.debug(f"Analyzing: {file_path.name}")

        issues = []

        try:
            content = file_path.read_text()

            # Simple analysis (replace with real analysis)
            lines = content.splitlines()

            # Check for long lines
            for i, line in enumerate(lines):
                if len(line) > 120:
                    issues.append({
                        "line": i + 1,
                        "type": "line_too_long",
                        "message": f"Line exceeds 120 characters ({len(line)} chars)",
                        "severity": "warning"
                    })

            # Check for TODO comments
            for i, line in enumerate(lines):
                if "TODO" in line:
                    issues.append({
                        "line": i + 1,
                        "type": "todo_comment",
                        "message": "TODO comment found",
                        "severity": "info"
                    })

            result = {
                "file": str(file_path),
                "lines": len(lines),
                "issues": issues,
                "issue_count": len(issues)
            }

            # Cache result
            self.cache_manager.set(cache_key, result, level="analysis")

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return {
                "file": str(file_path),
                "error": str(e),
                "issues": [],
                "issue_count": 0
            }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate analysis results"""
        summary = {
            "total_files": len(results),
            "total_issues": sum(r.get('issue_count', 0) for r in results),
            "total_lines": sum(r.get('lines', 0) for r in results),
            "files_with_issues": sum(1 for r in results if r.get('issue_count', 0) > 0),
            "issues_by_type": {},
            "issues_by_severity": {
                "error": 0,
                "warning": 0,
                "info": 0
            }
        }

        # Aggregate by type and severity
        for result in results:
            for issue in result.get('issues', []):
                issue_type = issue.get('type', 'unknown')
                severity = issue.get('severity', 'info')

                summary['issues_by_type'][issue_type] = \
                    summary['issues_by_type'].get(issue_type, 0) + 1

                summary['issues_by_severity'][severity] = \
                    summary['issues_by_severity'].get(severity, 0) + 1

        return summary

    def _format_summary(self, analysis: Dict[str, Any]) -> str:
        """Format analysis summary"""
        lines = [
            f"Analyzed {analysis['total_files']} Python files",
            f"Total lines: {analysis['total_lines']}",
            f"Issues found: {analysis['total_issues']}",
            f"Files with issues: {analysis['files_with_issues']}"
        ]

        if analysis['total_issues'] > 0:
            lines.append("\nIssues by severity:")
            for severity, count in analysis['issues_by_severity'].items():
                if count > 0:
                    lines.append(f"  {severity}: {count}")

        return "\n".join(lines)

    def _generate_recommendations(self, details: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        analysis = details.get('analysis', {})
        total_issues = analysis.get('total_issues', 0)

        if total_issues > 0:
            recommendations.append(
                f"Found {total_issues} issues that should be addressed"
            )

        issues_by_type = analysis.get('issues_by_type', {})

        if 'line_too_long' in issues_by_type:
            recommendations.append(
                "Consider using a code formatter like Black to fix line length issues"
            )

        if 'todo_comment' in issues_by_type:
            recommendations.append(
                "Review and address TODO comments before production"
            )

        if not recommendations:
            recommendations.append("Code quality looks good!")

        return recommendations

    def _validate_options(self, context: ExecutionContext) -> Tuple[bool, str]:
        """Validate command options"""
        # Example validation
        parallel = context.args.get('parallel', False)
        file_count = context.metadata.get('file_count', 0)

        if parallel and file_count < 10:
            return True, "Parallel mode may not improve performance for < 10 files"

        return True, ""

    def _validate_results(self, result: ExecutionResult) -> bool:
        """Validate execution results"""
        # Example validation
        details = result.details
        files_analyzed = details.get('files_analyzed', 0)

        if files_analyzed == 0:
            return False

        return True


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Example command execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Example Code Analyzer")
    parser.add_argument('target', help="Target directory to analyze")
    parser.add_argument('--agents', default='auto', help="Agent selection mode")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel execution")
    parser.add_argument('--dry-run', action='store_true', help="Preview mode")
    parser.add_argument('--implement', action='store_true', help="Implement fixes")
    parser.add_argument('--validate', action='store_true', help="Validate results")

    args = parser.parse_args()

    # Create executor
    executor = ExampleCodeAnalyzerExecutor()

    # Execute
    result = executor.execute(vars(args))

    # Print output
    print(executor.format_output(result))

    # Return exit code
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())