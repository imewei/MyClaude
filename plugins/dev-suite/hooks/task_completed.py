#!/usr/bin/env python3
"""TaskCompleted hook for dev-suite.

Triggers validation reminders and suggests git commit after task completion.
Inspired by Anthropic's "Effective harnesses for long-running agents":
agents should commit progress after each feature/task.
"""

import json
import os
import subprocess
import sys


def has_uncommitted_changes(cwd: str) -> bool:
    """Check if there are uncommitted changes to suggest committing."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main() -> None:
    """Remind about validation and suggest commit after task completion."""
    try:
        task_subject = os.environ.get("TASK_SUBJECT", "unknown task")
        cwd = os.environ.get("PWD", os.getcwd())

        advice = [f"Task completed: {task_subject}."]
        advice.append("Consider running tests and linting before moving on.")

        if has_uncommitted_changes(cwd):
            advice.append(
                "You have uncommitted changes — consider committing this "
                "progress with a descriptive message before starting the next task."
            )

        result = {
            "status": "success",
            "additionalContext": " ".join(advice),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"TaskCompleted hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
