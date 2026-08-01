#!/usr/bin/env python3
"""SubagentStop hook for agent-core plugin.

Fires when a subagent finishes execution. Collects agent output
summaries for orchestration awareness.
"""

import json
import sys

import _hook_io


def main() -> None:
    """Log subagent completion for orchestration tracking."""
    try:
        payload = _hook_io.read_payload()
        agent_name = _hook_io.get_field(
            payload,
            "agent_name",
            "subagent_type",
            "agent_type",
            "matcher_input",
            default="",
        )

        result = {"status": "success"}
        ctx = (
            f"Subagent '{agent_name}' completed. Check task list for updates."
            if agent_name
            else "A subagent completed. Check task list for updates."
        )
        result["additionalContext"] = ctx
        result.update(_hook_io.wrap_context("SubagentStop", ctx))
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
