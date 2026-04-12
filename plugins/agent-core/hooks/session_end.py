#!/usr/bin/env python3
"""SessionEnd hook for agent-core plugin.

Fires when a session ends. The matcher input is the reason for ending
(e.g., "clear", "resume", "logout", "prompt_input_exit", "other").
Useful for session cleanup and summary logging.
"""

import json
import sys


def main() -> None:
    """Log session end event."""
    try:
        input_data = json.load(sys.stdin)
        end_reason = input_data.get("matcher_input", "unknown")
        result = {
            "status": "success",
            "message": f"Session ended: {end_reason}",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"SessionEnd hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
