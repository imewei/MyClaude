#!/usr/bin/env python3
"""SubagentStop hook for dev-suite.

Collects test/review results when quality-specialist finishes.
"""

import json
import sys

from _hook_io import get_field, read_payload, wrap_context


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

        ctx = f"Dev-suite agent '{agent_name}' completed."
        result = {"status": "success", "additionalContext": ctx}
        result.update(wrap_context("SubagentStop", ctx))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
