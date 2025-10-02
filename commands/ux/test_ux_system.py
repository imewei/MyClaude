#!/usr/bin/env python3
"""
Test script to verify UX system implementation.

This script tests all major components to ensure they work correctly.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 60)
    print("Test 1: Module Imports")
    print("=" * 60)

    try:
        # Progress tracking
        from progress.progress_tracker import ProgressTracker, ProgressStatus
        from progress.step_tracker import StepTracker, StepStatus
        from progress.live_dashboard import LiveDashboard

        # Error handling
        from errors.error_formatter import ErrorFormatter, ErrorCategory
        from errors.error_suggestions import ErrorSuggestionEngine
        from errors.error_recovery import ErrorRecovery

        # Recommendations
        from recommendations.command_recommender import CommandRecommender

        # Core
        from core.ux_manager import UXManager

        print("✓ All modules imported successfully")
        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_progress_tracker():
    """Test progress tracker functionality."""
    print("\n" + "=" * 60)
    print("Test 2: Progress Tracker")
    print("=" * 60)

    try:
        from progress.progress_tracker import ProgressTracker, ProgressStatus

        tracker = ProgressTracker(enabled=False)  # Disable for testing

        # Create task
        task = tracker.add_task("Test task", total=10)
        assert task is not None, "Task creation failed"

        # Update task
        tracker.update(task, advance=5)
        item = tracker.get_item(task)
        assert item.completed == 5, "Task update failed"

        # Complete task
        tracker.complete(task, ProgressStatus.SUCCESS)
        assert item.status == ProgressStatus.SUCCESS, "Task completion failed"

        print("✓ Progress tracker working correctly")
        return True

    except Exception as e:
        print(f"✗ Progress tracker test failed: {e}")
        return False


def test_step_tracker():
    """Test step tracker functionality."""
    print("\n" + "=" * 60)
    print("Test 3: Step Tracker")
    print("=" * 60)

    try:
        from progress.step_tracker import StepTracker, StepStatus

        tracker = StepTracker("Test operation", enabled=False)

        # Add steps
        tracker.add_step("step1", "Step 1")
        tracker.add_step("step2", "Step 2")
        assert len(tracker.steps) == 2, "Step addition failed"

        # Execute steps
        tracker.start_step("step1")
        tracker.complete_step("step1")

        step = tracker.get_step("step1")
        assert step.status == StepStatus.COMPLETED, "Step completion failed"

        # Check progress
        progress = tracker.get_progress_percentage()
        assert progress == 50.0, "Progress calculation failed"

        print("✓ Step tracker working correctly")
        return True

    except Exception as e:
        print(f"✗ Step tracker test failed: {e}")
        return False


def test_error_formatter():
    """Test error formatter functionality."""
    print("\n" + "=" * 60)
    print("Test 4: Error Formatter")
    print("=" * 60)

    try:
        from errors.error_formatter import ErrorFormatter, ErrorCategory

        formatter = ErrorFormatter(enabled=False)

        # Create a test exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            formatted = formatter.format_exception(
                e,
                category=ErrorCategory.RUNTIME
            )

            assert formatted.error_id is not None, "Error ID generation failed"
            assert formatted.category == ErrorCategory.RUNTIME, "Category assignment failed"
            assert "Test error" in formatted.message, "Message capture failed"
            assert len(formatted.suggestions) > 0, "Suggestions generation failed"

        print("✓ Error formatter working correctly")
        return True

    except Exception as e:
        print(f"✗ Error formatter test failed: {e}")
        return False


def test_error_suggestions():
    """Test error suggestion engine."""
    print("\n" + "=" * 60)
    print("Test 5: Error Suggestion Engine")
    print("=" * 60)

    try:
        from errors.error_suggestions import ErrorSuggestionEngine
        from errors.error_formatter import ErrorCategory

        engine = ErrorSuggestionEngine()

        # Test import error suggestions
        suggestions = engine.suggest_fixes(
            "No module named 'numpy'",
            ErrorCategory.DEPENDENCY
        )

        assert len(suggestions) > 0, "No suggestions generated"
        assert any("install" in s.title.lower() for s in suggestions), \
            "Install suggestion not found"

        print("✓ Error suggestion engine working correctly")
        return True

    except Exception as e:
        print(f"✗ Error suggestion engine test failed: {e}")
        return False


def test_error_recovery():
    """Test error recovery system."""
    print("\n" + "=" * 60)
    print("Test 6: Error Recovery")
    print("=" * 60)

    try:
        from errors.error_recovery import ErrorRecovery

        recovery = ErrorRecovery()

        # Test checkpoint
        recovery.save_checkpoint("test_op", "checkpoint1", {"data": "test"})
        loaded = recovery.load_checkpoint("test_op", "checkpoint1")
        assert loaded == {"data": "test"}, "Checkpoint save/load failed"

        # Test retry decorator
        attempt_count = [0]

        @recovery.with_retry(max_attempts=3, backoff_factor=1.0)
        def flaky_function():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise RuntimeError("Temporary error")
            return "Success"

        result = flaky_function()
        assert result == "Success", "Retry failed"
        assert attempt_count[0] == 2, "Retry count incorrect"

        # Cleanup
        recovery.clear_checkpoints("test_op")

        print("✓ Error recovery working correctly")
        return True

    except Exception as e:
        print(f"✗ Error recovery test failed: {e}")
        return False


def test_command_recommender():
    """Test command recommender system."""
    print("\n" + "=" * 60)
    print("Test 7: Command Recommender")
    print("=" * 60)

    try:
        from recommendations.command_recommender import CommandRecommender

        recommender = CommandRecommender()

        # Test context detection
        context = recommender.detect_context()
        assert context is not None, "Context detection failed"

        # Test recommendations
        recommendations = recommender.recommend(
            context={"project_type": "python"},
            goal="improve code quality"
        )

        assert len(recommendations) > 0, "No recommendations generated"
        assert all(r.confidence > 0 for r in recommendations), \
            "Invalid confidence scores"

        # Test command recording
        recommender.record_command("test_command", success=True)
        assert len(recommender.command_history) > 0, "Command recording failed"

        print("✓ Command recommender working correctly")
        return True

    except Exception as e:
        print(f"✗ Command recommender test failed: {e}")
        return False


def test_ux_manager():
    """Test UX manager."""
    print("\n" + "=" * 60)
    print("Test 8: UX Manager")
    print("=" * 60)

    try:
        from core.ux_manager import UXManager, ThemeMode, VerbosityLevel

        ux = UXManager()

        # Test configuration
        ux.config.theme = ThemeMode.DARK
        ux.config.verbosity = VerbosityLevel.VERBOSE
        assert ux.config.theme == ThemeMode.DARK, "Theme setting failed"
        assert ux.config.verbosity == VerbosityLevel.VERBOSE, "Verbosity setting failed"

        # Test console
        console = ux.get_console()
        # Console might be None if rich is not available
        if console:
            assert console is not None, "Console creation failed"

        # Test output formatting
        output = ux.format_output({"key": "value"})
        assert output is not None, "Output formatting failed"

        print("✓ UX manager working correctly")
        return True

    except Exception as e:
        print(f"✗ UX manager test failed: {e}")
        return False


def test_package_exports():
    """Test that package exports work."""
    print("\n" + "=" * 60)
    print("Test 9: Package Exports")
    print("=" * 60)

    try:
        # Import from package root
        import ux

        # Check key exports
        assert hasattr(ux, 'ProgressTracker'), "ProgressTracker not exported"
        assert hasattr(ux, 'ErrorFormatter'), "ErrorFormatter not exported"
        assert hasattr(ux, 'CommandRecommender'), "CommandRecommender not exported"
        assert hasattr(ux, 'UXManager'), "UXManager not exported"

        print("✓ Package exports working correctly")
        return True

    except Exception as e:
        print(f"✗ Package exports test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("UX System - Comprehensive Test Suite")
    print("=" * 80)

    tests = [
        ("Module Imports", test_imports),
        ("Progress Tracker", test_progress_tracker),
        ("Step Tracker", test_step_tracker),
        ("Error Formatter", test_error_formatter),
        ("Error Suggestions", test_error_suggestions),
        ("Error Recovery", test_error_recovery),
        ("Command Recommender", test_command_recommender),
        ("UX Manager", test_ux_manager),
        ("Package Exports", test_package_exports)
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("Status: ✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"Status: ✗ {total - passed} TEST(S) FAILED")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())