#!/usr/bin/env python3
"""TaskCompleted hook for agent-core plugin.

Fires when a task is marked as complete. Can validate task outputs
or trigger follow-up steps in multi-step workflows.
"""

import json
import sys


def main() -> None:
    """Acknowledge task completion."""
    try:
        result = {
            "status": "success",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump({"status": "error", "message": f"TaskCompleted hook error: {e}"}, sys.stdout)


if __name__ == "__main__":
    main()
