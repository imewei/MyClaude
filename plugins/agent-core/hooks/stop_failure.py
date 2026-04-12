#!/usr/bin/env python3
"""StopFailure hook for agent-core plugin.

Fires when the session stops due to an error. The matcher input is
the error type (e.g., "rate_limit", "authentication_failed",
"billing_error", "server_error", "max_output_tokens", "unknown").
Useful for error classification and notification routing.
"""

import json
import sys


def main() -> None:
    """Classify and log stop failure event."""
    try:
        input_data = json.load(sys.stdin)
        error_type = input_data.get("matcher_input", "unknown")

        # Classify severity based on error type
        retriable = error_type in ("rate_limit", "server_error")
        result = {
            "status": "success",
            "message": f"Stop failure: {error_type} (retriable={retriable})",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"StopFailure hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
