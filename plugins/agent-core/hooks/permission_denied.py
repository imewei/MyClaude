#!/usr/bin/env python3
"""PermissionDenied hook for agent-core plugin.

Fires when the auto-mode classifier blocks a tool call. Logs denied
actions to surface patterns and help the user adjust permissions.
"""

import json
import os
import sys


def main() -> None:
    """Log permission denial for audit trail."""
    try:
        tool_name = os.environ.get("TOOL_NAME", "unknown")

        result = {
            "status": "success",
            "additionalContext": (
                f"Permission denied for tool '{tool_name}'. "
                "If this is expected, consider adjusting permission mode."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump({"status": "error", "message": f"PermissionDenied hook error: {e}"}, sys.stdout)


if __name__ == "__main__":
    main()
