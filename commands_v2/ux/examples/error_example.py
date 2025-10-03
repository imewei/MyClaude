"""
Example demonstrating error formatting and suggestions.

This example shows:
1. Beautiful error formatting
2. Intelligent error suggestions
3. Error recovery strategies
4. Context-aware error handling
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from errors.error_formatter import ErrorFormatter, ErrorCategory, ErrorSeverity
from errors.error_suggestions import ErrorSuggestionEngine
from errors.error_recovery import ErrorRecovery, retry, fallback


def example_basic_error_formatting():
    """Example 1: Basic error formatting."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Error Formatting")
    print("=" * 60 + "\n")

    formatter = ErrorFormatter()

    try:
        # Simulate an error
        import nonexistent_module
    except ImportError as e:
        formatted = formatter.format_exception(
            e,
            category=ErrorCategory.DEPENDENCY,
            context={"command": "optimize"}
        )
        formatter.print_error(formatted)


def example_file_not_found_error():
    """Example 2: File not found error."""
    print("\n" + "=" * 60)
    print("Example 2: File Not Found Error")
    print("=" * 60 + "\n")

    formatter = ErrorFormatter()

    try:
        with open("/nonexistent/file.txt", 'r') as f:
            content = f.read()
    except FileNotFoundError as e:
        formatted = formatter.format_exception(
            e,
            category=ErrorCategory.FILESYSTEM,
            context={
                "command": "process_file",
                "agent": "Scientific Agent"
            }
        )
        formatter.print_error(formatted)


def example_runtime_error():
    """Example 3: Runtime error."""
    print("\n" + "=" * 60)
    print("Example 3: Runtime Error")
    print("=" * 60 + "\n")

    formatter = ErrorFormatter()

    try:
        # Simulate a runtime error
        result = 10 / 0
    except ZeroDivisionError as e:
        formatted = formatter.format_exception(
            e,
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.ERROR,
            context={"command": "calculate"}
        )
        formatter.print_error(formatted)


def example_error_suggestions():
    """Example 4: Intelligent error suggestions."""
    print("\n" + "=" * 60)
    print("Example 4: Error Suggestions")
    print("=" * 60 + "\n")

    engine = ErrorSuggestionEngine()

    # Test different error patterns
    test_errors = [
        ("No module named 'numpy'", ErrorCategory.DEPENDENCY),
        ("Permission denied: /etc/config.yaml", ErrorCategory.PERMISSION),
        ("Connection timeout: api.example.com", ErrorCategory.NETWORK),
        ("Invalid syntax at line 42", ErrorCategory.SYNTAX)
    ]

    for error_msg, category in test_errors:
        print(f"\nError: {error_msg}")
        print(f"Category: {category.value}")
        print("\nSuggestions:")

        suggestions = engine.suggest_fixes(error_msg, category)

        for i, sugg in enumerate(suggestions, 1):
            print(f"\n{i}. {sugg.title} (Confidence: {sugg.confidence:.0%})")
            print(f"   {sugg.description}")
            if sugg.command:
                print(f"   Command: {sugg.command}")
            elif sugg.action:
                print(f"   Action: {sugg.action}")

        print("\n" + "-" * 60)


def example_retry_recovery():
    """Example 5: Retry-based error recovery."""
    print("\n" + "=" * 60)
    print("Example 5: Retry Recovery")
    print("=" * 60 + "\n")

    recovery = ErrorRecovery()

    @recovery.with_retry(max_attempts=3, backoff_factor=2.0)
    def unstable_operation():
        """Operation that might fail."""
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Network unavailable")
        return "Success!"

    try:
        result = unstable_operation()
        print(f"Operation succeeded: {result}")
    except Exception as e:
        print(f"Operation failed after retries: {e}")


def example_fallback_recovery():
    """Example 6: Fallback error recovery."""
    print("\n" + "=" * 60)
    print("Example 6: Fallback Recovery")
    print("=" * 60 + "\n")

    recovery = ErrorRecovery()

    def fetch_from_cache():
        """Fallback function."""
        return "Cached data"

    @recovery.with_fallback(fallback_fn=fetch_from_cache)
    def fetch_from_api():
        """Primary function that might fail."""
        raise ConnectionError("API unavailable")

    result = fetch_from_api()
    print(f"Result: {result}")


def example_checkpoint_recovery():
    """Example 7: Checkpoint-based recovery."""
    print("\n" + "=" * 60)
    print("Example 7: Checkpoint Recovery")
    print("=" * 60 + "\n")

    recovery = ErrorRecovery()

    try:
        # Save checkpoint before risky operation
        recovery.save_checkpoint(
            "data_processing",
            "before_transform",
            {"records": 1000, "status": "ready"}
        )

        print("Processing data...")
        # Simulate error
        raise ValueError("Invalid data format")

    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Rolling back to checkpoint...")

        # Rollback to checkpoint
        state = recovery.rollback("data_processing", "before_transform")
        print(f"Restored state: {state}")


def example_comprehensive_error_handling():
    """Example 8: Comprehensive error handling."""
    print("\n" + "=" * 60)
    print("Example 8: Comprehensive Error Handling")
    print("=" * 60 + "\n")

    formatter = ErrorFormatter(show_suggestions=True)
    engine = ErrorSuggestionEngine()

    def process_data_with_error():
        """Function that will error."""
        data = {"values": [1, 2, 3]}
        # This will cause a KeyError
        return data["missing_key"]

    try:
        process_data_with_error()
    except Exception as e:
        # Get error suggestions
        suggestions = engine.suggest_fixes(
            str(e),
            ErrorCategory.RUNTIME,
            context={"command": "process"}
        )

        # Format with suggestions
        formatted = formatter.format_exception(
            e,
            category=ErrorCategory.RUNTIME,
            suggestions=suggestions
        )

        # Print formatted error
        formatter.print_error(formatted)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("UX Error Handling Examples")
    print("=" * 60)

    examples = [
        ("Basic Error Formatting", example_basic_error_formatting),
        ("File Not Found Error", example_file_not_found_error),
        ("Runtime Error", example_runtime_error),
        ("Error Suggestions", example_error_suggestions),
        ("Retry Recovery", example_retry_recovery),
        ("Fallback Recovery", example_fallback_recovery),
        ("Checkpoint Recovery", example_checkpoint_recovery),
        ("Comprehensive Error Handling", example_comprehensive_error_handling)
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
            print(f"\nUnexpected error in {name}: {e}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()