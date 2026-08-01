#!/usr/bin/env python3
"""StopFailure hook for dev-suite.

Captures context when /stop fails mid-operation.
"""

import json
import sys

from _hook_io import get_field, read_payload, wrap_context


def main() -> None:
    """Capture stop failure context."""
    try:
        error_message = get_field(
            read_payload(),
            "error_message",
            "error",
            "message",
            env_fallback="ERROR_MESSAGE",
        )
        ctx = (
            f"Stop command failed: {error_message}. "
            "Check for long-running processes or locked resources."
        )
        result = {"status": "success", "additionalContext": ctx}
        result.update(wrap_context("StopFailure", ctx))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"StopFailure hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"StopFailure hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
