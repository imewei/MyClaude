#!/usr/bin/env python3
"""StopFailure hook for agent-core plugin.

Fires when the session stops due to an error. The matcher input is
the error type (e.g., "rate_limit", "authentication_failed",
"billing_error", "server_error", "max_output_tokens", "unknown").
Useful for error classification and notification routing.
"""

import json
import sys

import _hook_io

RETRIABLE_ERRORS = ("rate_limit", "server_error")


def main() -> None:
    """Classify and log stop failure event."""
    try:
        payload = _hook_io.read_payload()
        error_type = _hook_io.get_field(
            payload, "error_type", "reason", "matcher_input", default=""
        )

        if not error_type:
            message = "Stop failure: error type not reported"
        elif error_type in RETRIABLE_ERRORS:
            message = f"Stop failure: {error_type} (retriable)"
        else:
            message = f"Stop failure: {error_type} (not a known retriable error)"

        result = {"status": "success", "message": message}
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"StopFailure hook error: {e}", file=sys.stderr)
        error_result = {
            "status": "error",
            "message": f"StopFailure hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
