#!/usr/bin/env python3
"""StopFailure hook for dev-suite.

Captures context when /stop fails mid-operation.
"""

import json
import os
import sys


def main() -> None:
    """Capture stop failure context."""
    try:
        error_message = os.environ.get("ERROR_MESSAGE", "unknown")
        result = {
            "status": "success",
            "additionalContext": (
                f"Stop command failed: {error_message}. "
                "Check for long-running processes or locked resources."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"StopFailure hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
