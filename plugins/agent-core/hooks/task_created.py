#!/usr/bin/env python3
"""TaskCreated hook for agent-core plugin.

Fires when a new task is created via TaskCreate. Useful for tracking
task creation patterns and enforcing task hygiene.
"""

import json
import sys


def main() -> None:
    """Log task creation event."""
    try:
        json.load(sys.stdin)  # consume stdin
        result = {
            "status": "success",
            "message": "Task created",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"TaskCreated hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
