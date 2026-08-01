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

        if agent_name:
            result = {"status": "success"}
            ctx = f"Subagent '{agent_name}' completed. Check task list for updates."
            result["additionalContext"] = ctx
            result.update(_hook_io.wrap_context("SubagentStop", ctx))
            json.dump(result, sys.stdout)
        else:
            # No identifiable agent — "a subagent completed, check task list" carries
            # no signal the caller can act on, so stay silent rather than add noise.
            json.dump({"status": "success"}, sys.stdout)
    except Exception as e:
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
