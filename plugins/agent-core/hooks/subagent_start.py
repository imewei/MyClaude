#!/usr/bin/env python3
"""SubagentStart hook for agent-core plugin.

Logs when a subagent is dispatched for orchestration telemetry.
Fires when any subagent begins execution. The matcher input is
the agent type name (e.g., "Bash", "Explore", "Plan", or custom).
"""

import json
import sys

import _hook_io


def main() -> None:
    """Log subagent dispatch event."""
    try:
        payload = _hook_io.read_payload()
        agent_type = _hook_io.get_field(
            payload,
            "agent_type",
            "subagent_type",
            "agent_name",
            "matcher_input",
            default="",
        )
        result = {
            "status": "success",
            "message": (
                f"Subagent dispatched: {agent_type}"
                if agent_type
                else "Subagent dispatched (type not reported)"
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"SubagentStart hook error: {e}", file=sys.stderr)
        error_result = {
            "status": "error",
            "message": f"SubagentStart hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
