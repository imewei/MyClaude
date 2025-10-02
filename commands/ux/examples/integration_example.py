"""
Complete integration example showing how to use all UX components together.

This example demonstrates a realistic command execution scenario with:
- Progress tracking
- Error handling
- Command recommendations
- Live dashboard
"""

import time
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from progress.progress_tracker import ProgressTracker, ProgressStatus
from progress.step_tracker import StepTracker, StepStatus
from progress.live_dashboard import LiveDashboard
from errors.error_formatter import ErrorFormatter, ErrorCategory
from errors.error_recovery import ErrorRecovery
from recommendations.command_recommender import CommandRecommender
from core.ux_manager import UXManager, VerbosityLevel


class EnhancedCommandExecutor:
    """
    Example command executor with full UX integration.

    This demonstrates how to integrate all UX components into
    a command execution framework.
    """

    def __init__(self):
        """Initialize with UX components."""
        self.ux_manager = UXManager()
        self.progress_tracker = ProgressTracker()
        self.dashboard = LiveDashboard()
        self.error_formatter = ErrorFormatter()
        self.error_recovery = ErrorRecovery()
        self.recommender = CommandRecommender()

        # Configure UX
        self.ux_manager.config.verbosity = VerbosityLevel.VERBOSE
        self.console = self.ux_manager.get_console()

    def execute_command(self, command: str, **kwargs):
        """
        Execute a command with full UX integration.

        Args:
            command: Command to execute
            **kwargs: Command arguments
        """
        self.console.print(f"\n[bold cyan]Executing command:[/bold cyan] {command}")

        # Record command for recommendations
        self.recommender.record_command(command, success=False)

        try:
            # Route to appropriate handler
            if command == "optimize":
                result = self._execute_optimize(**kwargs)
            elif command == "refactor":
                result = self._execute_refactor(**kwargs)
            elif command == "test":
                result = self._execute_test(**kwargs)
            else:
                raise ValueError(f"Unknown command: {command}")

            # Mark as successful
            self.recommender.record_command(command, success=True)

            # Show recommendations for next steps
            self._show_recommendations(command)

            return result

        except Exception as e:
            # Format and display error
            formatted = self.error_formatter.format_exception(
                e,
                context={
                    "command": command,
                    "kwargs": kwargs
                }
            )
            self.error_formatter.print_error(formatted)
            raise

    def _execute_optimize(self, **kwargs):
        """Execute optimization command with full UX."""
        with self.dashboard.live():
            self.dashboard.update_command("optimize", "Running")
            self.dashboard.add_agent("Scientific Agent")
            self.dashboard.add_agent("Performance Agent")

            # Create step tracker
            tracker = StepTracker("Code Optimization")
            tracker.add_step("analyze", "Analyze code performance")
            tracker.add_step("identify", "Identify bottlenecks")
            tracker.add_step("optimize", "Apply optimizations")
            tracker.add_step("verify", "Verify improvements")

            try:
                # Step 1: Analyze
                tracker.start_step("analyze")
                self._simulate_analysis()
                tracker.complete_step("analyze")
                self.dashboard.increment_operations(completed=1)

                # Step 2: Identify
                tracker.start_step("identify")
                self._simulate_identification()
                tracker.complete_step("identify")
                self.dashboard.increment_operations(completed=1)

                # Step 3: Optimize
                tracker.start_step("optimize")
                self._simulate_optimization()
                tracker.complete_step("optimize")
                self.dashboard.increment_operations(completed=1)

                # Step 4: Verify
                tracker.start_step("verify")
                self._simulate_verification()
                tracker.complete_step("verify")
                self.dashboard.increment_operations(completed=1)

                tracker.print_summary()

                self.dashboard.remove_agent("Scientific Agent")
                self.dashboard.remove_agent("Performance Agent")
                self.dashboard.update_command("optimize", "Complete")

                return {"status": "success", "optimizations_applied": 15}

            except Exception as e:
                tracker.fail_step(
                    tracker.current_step or "unknown",
                    str(e)
                )
                tracker.print_summary()
                raise

    def _execute_refactor(self, **kwargs):
        """Execute refactoring command with progress tracking."""
        with self.progress_tracker.live_progress():
            # Parent task
            parent = self.progress_tracker.add_task(
                "Code Refactoring",
                total=4
            )

            # Child tasks
            tasks = [
                ("Scanning files", 50),
                ("Analyzing code", 30),
                ("Applying refactorings", 40),
                ("Verifying changes", 20)
            ]

            for i, (desc, total) in enumerate(tasks):
                task = self.progress_tracker.add_task(
                    desc,
                    total=total,
                    parent_id=parent
                )

                # Simulate work
                for j in range(total):
                    time.sleep(0.05)
                    self.progress_tracker.update(task, advance=1)

                    # Simulate cache hits
                    if random.random() > 0.3:
                        self.dashboard.increment_cache_hits()
                    else:
                        self.dashboard.increment_cache_misses()

                self.progress_tracker.complete(task, ProgressStatus.SUCCESS)
                self.progress_tracker.update(parent, advance=1)

            self.progress_tracker.complete(parent, ProgressStatus.SUCCESS)

        self.progress_tracker.print_summary()
        return {"status": "success", "files_refactored": 23}

    def _execute_test(self, **kwargs):
        """Execute test command with error recovery."""

        @self.error_recovery.with_retry(max_attempts=3)
        def run_tests():
            """Run tests with retry."""
            # Simulate flaky tests
            if random.random() < 0.3:
                raise RuntimeError("Test failed due to network timeout")

            return {"status": "success", "tests_passed": 42, "tests_failed": 0}

        with self.dashboard.live():
            self.dashboard.update_command("test", "Running")
            self.dashboard.add_agent("Testing Agent")

            try:
                result = run_tests()

                self.dashboard.update_command("test", "Complete")
                self.dashboard.remove_agent("Testing Agent")

                return result

            except Exception as e:
                self.dashboard.update_command("test", "Failed")
                self.dashboard.remove_agent("Testing Agent")
                raise

    def _show_recommendations(self, last_command: str):
        """Show recommended next commands."""
        self.console.print("\n[bold green]Recommended next steps:[/bold green]")

        recommendations = self.recommender.recommend(
            recent_commands=[last_command],
            limit=3
        )

        for i, rec in enumerate(recommendations, 1):
            confidence_bar = "█" * int(rec.confidence * 10)
            self.console.print(
                f"{i}. [cyan]{rec.command}[/cyan] - {rec.reason} "
                f"[dim]({confidence_bar} {rec.confidence:.0%})[/dim]"
            )

    def _simulate_analysis(self):
        """Simulate code analysis."""
        time.sleep(1.0)
        self.dashboard.increment_cache_hits(5)

    def _simulate_identification(self):
        """Simulate bottleneck identification."""
        time.sleep(0.8)
        self.dashboard.increment_cache_hits(3)

    def _simulate_optimization(self):
        """Simulate optimization."""
        time.sleep(1.2)
        self.dashboard.increment_cache_hits(8)
        self.dashboard.add_execution_time(1.2)

    def _simulate_verification(self):
        """Simulate verification."""
        time.sleep(0.6)
        self.dashboard.increment_cache_hits(2)


def main():
    """Run integration example."""
    print("\n" + "=" * 80)
    print("UX Integration Example - Complete Command Execution")
    print("=" * 80)

    executor = EnhancedCommandExecutor()

    # Example 1: Optimize command
    print("\n[Example 1] Running optimization command...")
    try:
        result = executor.execute_command("optimize", target="src/")
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Command failed: {e}")

    time.sleep(1)

    # Example 2: Refactor command
    print("\n[Example 2] Running refactoring command...")
    try:
        result = executor.execute_command("refactor", scope="project")
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Command failed: {e}")

    time.sleep(1)

    # Example 3: Test command (with retry)
    print("\n[Example 3] Running test command with retry...")
    try:
        result = executor.execute_command("test", coverage=True)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Command failed: {e}")

    # Example 4: Error handling
    print("\n[Example 4] Demonstrating error handling...")
    try:
        result = executor.execute_command("invalid_command")
    except Exception:
        print("Error was handled and displayed beautifully!")

    print("\n" + "=" * 80)
    print("Integration example complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()