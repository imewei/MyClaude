#!/usr/bin/env python3
"""SubagentStop hook for agent-core plugin.

Fires when a subagent finishes execution. Collects agent output
summaries for orchestration awareness.
"""

import json
import os
import sys


def main() -> None:
    """Log subagent completion for orchestration tracking."""
    try:
        agent_name = os.environ.get("AGENT_NAME", "unknown")

        result = {
            "status": "success",
            "additionalContext": f"Subagent '{agent_name}' completed. Check task list for updates.",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
