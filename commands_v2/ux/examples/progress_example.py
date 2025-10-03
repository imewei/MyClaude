"""
Example demonstrating progress tracking features.

This example shows:
1. Basic progress bars
2. Multi-level hierarchical progress
3. Step-by-step tracking
4. Live dashboard
"""

import time
import random
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from progress.progress_tracker import ProgressTracker, ProgressStatus
from progress.step_tracker import StepTracker, StepStatus
from progress.live_dashboard import LiveDashboard


def example_basic_progress():
    """Example 1: Basic progress tracking."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Progress Tracking")
    print("=" * 60 + "\n")

    tracker = ProgressTracker()

    with tracker.live_progress():
        # Create a task
        task_id = tracker.add_task("Processing files", total=50)

        # Simulate work
        for i in range(50):
            time.sleep(0.05)
            tracker.update(task_id, advance=1)

        tracker.complete(task_id, ProgressStatus.SUCCESS)

    tracker.print_summary()


def example_hierarchical_progress():
    """Example 2: Multi-level hierarchical progress."""
    print("\n" + "=" * 60)
    print("Example 2: Hierarchical Progress")
    print("=" * 60 + "\n")

    tracker = ProgressTracker()

    # Parent task
    parent_id = tracker.add_task("Code Optimization", total=3)

    # Child tasks
    analyze_id = tracker.add_task("Analyzing code", total=100, parent_id=parent_id)
    optimize_id = tracker.add_task("Applying optimizations", total=50, parent_id=parent_id)
    test_id = tracker.add_task("Running tests", total=25, parent_id=parent_id)

    # Simulate analysis
    for i in range(100):
        tracker.update(analyze_id, advance=1)
        time.sleep(0.02)

    tracker.complete(analyze_id, ProgressStatus.SUCCESS)
    tracker.update(parent_id, advance=1)

    # Simulate optimization
    for i in range(50):
        tracker.update(optimize_id, advance=1)
        time.sleep(0.03)

    tracker.complete(optimize_id, ProgressStatus.SUCCESS)
    tracker.update(parent_id, advance=1)

    # Simulate testing
    for i in range(25):
        tracker.update(test_id, advance=1)
        time.sleep(0.04)

    tracker.complete(test_id, ProgressStatus.SUCCESS)
    tracker.update(parent_id, advance=1)

    tracker.complete(parent_id, ProgressStatus.SUCCESS)

    tracker.print_summary()


def example_step_tracking():
    """Example 3: Step-by-step operation tracking."""
    print("\n" + "=" * 60)
    print("Example 3: Step Tracking")
    print("=" * 60 + "\n")

    tracker = StepTracker("Code Quality Improvement")

    # Define steps
    tracker.add_step("analyze", "Analyze code quality", "Running static analysis")
    tracker.add_step("refactor", "Refactor code", "Applying refactoring patterns")
    tracker.add_step("test", "Run tests", "Executing test suite")
    tracker.add_step("lint", "Run linter", "Checking code style")
    tracker.add_step("commit", "Commit changes", "Creating commit")

    # Execute steps
    tracker.start_step("analyze")
    time.sleep(1.0)
    tracker.complete_step("analyze")

    tracker.start_step("refactor")
    time.sleep(1.5)
    tracker.complete_step("refactor")

    tracker.start_step("test")
    time.sleep(1.2)
    tracker.complete_step("test")

    tracker.start_step("lint")
    time.sleep(0.8)
    tracker.complete_step("lint")

    tracker.start_step("commit")
    time.sleep(0.5)
    tracker.complete_step("commit")

    # Print summary
    tracker.print_summary(show_details=True)


def example_live_dashboard():
    """Example 4: Live dashboard."""
    print("\n" + "=" * 60)
    print("Example 4: Live Dashboard")
    print("=" * 60 + "\n")

    dashboard = LiveDashboard()

    with dashboard.live():
        # Simulate command execution
        dashboard.update_command("optimize", "Running")

        # Add agents
        dashboard.add_agent("Scientific Agent")
        time.sleep(1)

        dashboard.add_agent("Quality Agent")
        time.sleep(1)

        # Simulate work
        for i in range(20):
            # Random cache hits/misses
            if random.random() > 0.3:
                dashboard.increment_cache_hits()
            else:
                dashboard.increment_cache_misses()

            # Random operations
            if random.random() > 0.9:
                dashboard.increment_operations(failed=1)
            else:
                dashboard.increment_operations(completed=1)

            dashboard.add_execution_time(0.5)
            time.sleep(0.5)

        # Remove agents
        dashboard.remove_agent("Scientific Agent")
        time.sleep(0.5)
        dashboard.remove_agent("Quality Agent")

        dashboard.update_command("optimize", "Complete")
        time.sleep(2)

    print("\nDashboard session complete!")


def example_progress_with_errors():
    """Example 5: Progress tracking with errors."""
    print("\n" + "=" * 60)
    print("Example 5: Progress with Errors")
    print("=" * 60 + "\n")

    tracker = StepTracker("Deployment Pipeline")

    # Define steps
    tracker.add_step("build", "Build application")
    tracker.add_step("test", "Run tests")
    tracker.add_step("deploy", "Deploy to production")
    tracker.add_step("verify", "Verify deployment")

    # Execute with some failures
    tracker.start_step("build")
    time.sleep(1.0)
    tracker.complete_step("build", StepStatus.COMPLETED)

    tracker.start_step("test")
    time.sleep(0.8)
    tracker.fail_step("test", "3 tests failed")

    tracker.skip_step("deploy", "Skipped due to test failures")
    tracker.skip_step("verify", "Skipped due to test failures")

    # Print summary
    tracker.print_summary(show_details=True)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("UX Progress Tracking Examples")
    print("=" * 60)

    examples = [
        ("Basic Progress", example_basic_progress),
        ("Hierarchical Progress", example_hierarchical_progress),
        ("Step Tracking", example_step_tracking),
        ("Live Dashboard", example_live_dashboard),
        ("Progress with Errors", example_progress_with_errors)
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_fn in examples:
        try:
            example_fn()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\nError in {name}: {e}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()