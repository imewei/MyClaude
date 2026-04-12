#!/usr/bin/env python3
"""ExecutionError hook for dev-suite.

Captures build/test failure diagnostics with structured context.
"""

import json
import os
import sys


def main() -> None:
    """Capture build or test failure context."""
    try:
        error_message = os.environ.get("ERROR_MESSAGE", "unknown error")
        tool_name = os.environ.get("TOOL_NAME", "unknown")

        context = (
            f"Build/test error in {tool_name}: {error_message}. "
            "Check test output, build logs, or dependency state."
        )

        result = {"status": "success", "additionalContext": context}
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"ExecutionError hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
