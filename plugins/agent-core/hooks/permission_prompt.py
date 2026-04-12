#!/usr/bin/env python3
"""PermissionPrompt hook for agent-core.

Logs permission dialog events for debugging permission mode choices.
"""

import json
import os
import sys


def main() -> None:
    """Log permission prompt event."""
    try:
        tool_name = os.environ.get("TOOL_NAME", "unknown")
        result = {
            "status": "success",
            "additionalContext": f"Permission prompt triggered for tool: {tool_name}",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"PermissionPrompt hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
