#!/usr/bin/env python3
"""SubagentStop hook for dev-suite.

Collects test/review results when debugger-pro or quality-specialist finish.
"""

import json
import sys

from _hook_io import get_field, read_payload


def main() -> None:
    """Log subagent completion for dev workflow tracking."""
    try:
        agent_name = get_field(
            read_payload(),
            "agent_name",
            "subagent_type",
            "agent",
            "name",
            env_fallback="AGENT_NAME",
        )

        result = {
            "status": "success",
            "additionalContext": f"Dev-suite agent '{agent_name}' completed.",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
