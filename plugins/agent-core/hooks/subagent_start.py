#!/usr/bin/env python3
"""SubagentStart hook for agent-core plugin.

Logs when a subagent is dispatched for orchestration telemetry.
Fires when any subagent begins execution. The matcher input is
the agent type name (e.g., "Bash", "Explore", "Plan", or custom).
"""

import json
import sys


def main() -> None:
    """Log subagent dispatch event."""
    try:
        input_data = json.load(sys.stdin)
        agent_type = input_data.get("matcher_input", "unknown")
        result = {
            "status": "success",
            "message": f"Subagent dispatched: {agent_type}",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"SubagentStart hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
