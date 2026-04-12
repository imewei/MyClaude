#!/usr/bin/env python3
"""ExecutionError hook for agent-core.

Captures structured error context for reasoning chain failures.
"""

import json
import os
import sys


def main() -> None:
    """Capture execution error context."""
    try:
        error_message = os.environ.get("ERROR_MESSAGE", "unknown error")
        tool_name = os.environ.get("TOOL_NAME", "unknown")
        result = {
            "status": "success",
            "additionalContext": (
                f"Execution error in {tool_name}: {error_message}. "
                "Review reasoning chain for root cause."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"ExecutionError hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
