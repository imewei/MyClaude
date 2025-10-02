"""
Example demonstrating command recommendation system.

This example shows:
1. Context-based recommendations
2. Command sequence predictions
3. Goal-based workflow suggestions
4. Learning from usage patterns
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommendations.command_recommender import CommandRecommender, ProjectContext


def example_context_recommendations():
    """Example 1: Context-based recommendations."""
    print("\n" + "=" * 60)
    print("Example 1: Context-Based Recommendations")
    print("=" * 60 + "\n")

    recommender = CommandRecommender()

    # Python project context
    context = {
        "project_type": "python",
        "has_tests": True,
        "has_ci": True
    }

    print("Context: Python project with tests and CI")
    print("\nRecommended commands:")

    recommendations = recommender.recommend(context=context, limit=5)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.command} (Confidence: {rec.confidence:.0%})")
        print(f"   {rec.description}")
        print(f"   Reason: {rec.reason}")
        if rec.examples:
            print(f"   Example: {rec.examples[0]}")


def example_sequence_recommendations():
    """Example 2: Command sequence predictions."""
    print("\n" + "=" * 60)
    print("Example 2: Sequence-Based Recommendations")
    print("=" * 60 + "\n")

    recommender = CommandRecommender()

    # Simulate command history
    recent_commands = [
        "git add .",
        "git commit -m 'Update'"
    ]

    # Record these commands
    for cmd in recent_commands:
        recommender.record_command(cmd, success=True)

    print(f"Recent commands: {', '.join(recent_commands)}")
    print("\nNext command suggestions:")

    recommendations = recommender.recommend(recent_commands=recent_commands, limit=3)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.command}")
        print(f"   {rec.description}")
        print(f"   Confidence: {rec.confidence:.0%}")


def example_goal_recommendations():
    """Example 3: Goal-based recommendations."""
    print("\n" + "=" * 60)
    print("Example 3: Goal-Based Recommendations")
    print("=" * 60 + "\n")

    recommender = CommandRecommender()

    goals = [
        "improve code quality",
        "optimize performance",
        "update documentation",
        "fix test failures"
    ]

    for goal in goals:
        print(f"\nGoal: '{goal}'")
        print("Recommended commands:")

        recommendations = recommender.recommend(goal=goal, limit=3)

        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. {rec.command} (Confidence: {rec.confidence:.0%})")
            print(f"     {rec.description}")
            if rec.examples:
                print(f"     Example: {rec.examples[0]}")

        print("\n" + "-" * 60)


def example_workflow_suggestions():
    """Example 4: Complete workflow suggestions."""
    print("\n" + "=" * 60)
    print("Example 4: Workflow Suggestions")
    print("=" * 60 + "\n")

    recommender = CommandRecommender()

    goals = [
        "improve code quality",
        "optimize performance",
        "update documentation"
    ]

    for goal in goals:
        print(f"\nGoal: '{goal}'")
        print("Suggested workflow:")

        workflows = recommender.get_workflow_suggestions(goal)

        for workflow_idx, workflow in enumerate(workflows, 1):
            print(f"\n  Workflow {workflow_idx}:")
            for step_idx, command in enumerate(workflow, 1):
                print(f"    {step_idx}. {command}")

        print("\n" + "-" * 60)


def example_project_detection():
    """Example 5: Project context detection."""
    print("\n" + "=" * 60)
    print("Example 5: Project Context Detection")
    print("=" * 60 + "\n")

    recommender = CommandRecommender()

    # Detect current project context
    context = recommender.detect_context()

    print("Detected project context:")
    print(f"  Project type: {context.project_type or 'Unknown'}")
    print(f"  Languages: {', '.join(context.languages) or 'None detected'}")
    print(f"  Frameworks: {', '.join(context.frameworks) or 'None detected'}")
    print(f"  Has tests: {context.has_tests}")
    print(f"  Has CI: {context.has_ci}")
    print(f"  Has docs: {context.has_docs}")
    print(f"  File count: {context.file_count}")

    print("\nBased on this context, recommended commands:")

    context_dict = {
        "project_type": context.project_type,
        "has_tests": context.has_tests,
        "has_ci": context.has_ci
    }

    recommendations = recommender.recommend(context=context_dict, limit=5)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.command}")
        print(f"   {rec.description}")
        print(f"   Confidence: {rec.confidence:.0%}")


def example_learning_from_usage():
    """Example 6: Learning from usage patterns."""
    print("\n" + "=" * 60)
    print("Example 6: Learning from Usage Patterns")
    print("=" * 60 + "\n")

    recommender = CommandRecommender()

    # Simulate command history
    command_sequences = [
        ["/check-code-quality", "/refactor-clean", "/run-all-tests"],
        ["/optimize", "/run-all-tests", "/update-docs"],
        ["/check-code-quality", "/refactor-clean", "/commit"],
        ["/optimize", "/generate-tests", "/run-all-tests"]
    ]

    print("Recording command sequences...")
    for seq in command_sequences:
        for cmd in seq:
            recommender.record_command(cmd, success=True)
        print(f"  {' -> '.join(seq)}")

    print("\nLearned patterns - After '/check-code-quality', you typically run:")
    recommendations = recommender.recommend(
        recent_commands=["/check-code-quality"],
        limit=3
    )

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.command} ({rec.confidence:.0%})")


def example_comprehensive_recommendations():
    """Example 7: Comprehensive recommendation scenario."""
    print("\n" + "=" * 60)
    print("Example 7: Comprehensive Recommendations")
    print("=" * 60 + "\n")

    recommender = CommandRecommender()

    # Complex scenario
    context = {
        "project_type": "python",
        "has_tests": True,
        "has_ci": True
    }

    recent_commands = [
        "/optimize",
        "/run-all-tests"
    ]

    goal = "improve code quality and performance"

    print("Scenario:")
    print(f"  Context: {context}")
    print(f"  Recent commands: {recent_commands}")
    print(f"  Goal: {goal}")

    print("\nComprehensive recommendations:")

    recommendations = recommender.recommend(
        context=context,
        recent_commands=recent_commands,
        goal=goal,
        limit=5
    )

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.command} (Confidence: {rec.confidence:.0%})")
        print(f"   Description: {rec.description}")
        print(f"   Reason: {rec.reason}")

        if rec.flags:
            print(f"   Suggested flags: {', '.join(rec.flags)}")

        if rec.examples:
            print(f"   Example: {rec.examples[0]}")

        if rec.related_commands:
            print(f"   Related: {', '.join(rec.related_commands)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("UX Command Recommendation Examples")
    print("=" * 60)

    examples = [
        ("Context Recommendations", example_context_recommendations),
        ("Sequence Predictions", example_sequence_recommendations),
        ("Goal-Based Recommendations", example_goal_recommendations),
        ("Workflow Suggestions", example_workflow_suggestions),
        ("Project Detection", example_project_detection),
        ("Learning from Usage", example_learning_from_usage),
        ("Comprehensive Recommendations", example_comprehensive_recommendations)
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