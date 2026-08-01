#!/usr/bin/env python3
"""PermissionDenied hook for agent-core plugin.

Fires when the auto-mode classifier blocks a tool call. Logs denied
actions to surface patterns and help the user adjust permissions.
"""

import json
import sys

import _hook_io


def main() -> None:
    """Log permission denial for audit trail."""
    try:
        payload = _hook_io.read_payload()
        tool_name = _hook_io.get_field(
            payload,
            "tool_name",
            "matcher_input",
            env_fallback="TOOL_NAME",
            default="",
        )

        result = {"status": "success"}
        # Naming no tool is more honest than asserting a denial for tool 'unknown'.
        if tool_name:
            result["additionalContext"] = (
                f"Permission denied for tool '{tool_name}'. "
                "If this is expected, consider adjusting permission mode."
            )
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"PermissionDenied hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
