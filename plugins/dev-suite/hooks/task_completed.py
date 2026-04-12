#!/usr/bin/env python3
"""TaskCompleted hook for dev-suite.

Triggers validation reminders when implementation tasks finish.
"""

import json
import os
import sys


def main() -> None:
    """Remind about validation after task completion."""
    try:
        task_subject = os.environ.get("TASK_SUBJECT", "unknown task")
        result = {
            "status": "success",
            "additionalContext": (
                f"Task completed: {task_subject}. "
                "Consider running tests and linting before moving on."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"TaskCompleted hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
