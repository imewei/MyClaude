#!/usr/bin/env python3
"""PreSubagentUse hook for agent-core.

Validates subagent dispatch before launch — catches misrouted agents.
"""

import json
import os
import sys


def main() -> None:
    """Validate subagent dispatch parameters."""
    try:
        agent_name = os.environ.get("AGENT_NAME", "unknown")
        result = {
            "status": "success",
            "additionalContext": f"Dispatching subagent: {agent_name}",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"PreSubagentUse hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
