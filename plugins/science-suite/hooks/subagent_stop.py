#!/usr/bin/env python3
"""SubagentStop hook for science-suite.

Collects results from parallel science agents (parameter sweeps, etc.).
"""

import json
import os
import sys


def main() -> None:
    """Log science subagent completion."""
    try:
        agent_name = os.environ.get("AGENT_NAME", "unknown")
        result = {
            "status": "success",
            "additionalContext": (
                f"Science agent '{agent_name}' completed. "
                "Check output for numerical validity before proceeding."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
