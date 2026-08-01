#!/usr/bin/env python3
"""PreToolUse hook for Task tool calls.

Injects additional context about available agent types and their capabilities
when the Task tool is invoked, helping the orchestrator make better routing decisions.

Capabilities are read from each agent's own frontmatter (plugins/*/agents/*.md)
so this hook can never drift from the agent roster.
"""

import json
import os
import sys
from pathlib import Path

import _hook_io

PLUGINS_ROOT = Path(__file__).resolve().parents[2]


def load_agent_capabilities() -> dict[str, str]:
    """Map agent name -> first sentence of its frontmatter description."""
    capabilities = {}
    for agent_file in PLUGINS_ROOT.glob("*/agents/*.md"):
        name = ""
        description = ""
        try:
            with agent_file.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("name:"):
                        name = line[5:].strip()
                    elif line.startswith("description:"):
                        description = line[12:].strip()
                    if name and description:
                        break
        except OSError as e:
            sys.stderr.write(f"[PreToolUse] Could not read {agent_file}: {e}\n")
            continue
        if name and description:
            capabilities[name] = description.split(". ")[0].rstrip(".")
    return capabilities


def get_tool_input(payload: dict) -> dict:
    """Extract tool_input from the payload, falling back to the TOOL_INPUT env var."""
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        return tool_input
    try:
        legacy = json.loads(os.environ.get("TOOL_INPUT", "{}"))
    except json.JSONDecodeError:
        return {}
    return legacy if isinstance(legacy, dict) else {}


def main() -> None:
    """Provide agent routing context for Task tool calls."""
    try:
        payload = _hook_io.read_payload()
        subagent_type = get_tool_input(payload).get("subagent_type", "")

        result = {"status": "success"}

        capabilities = load_agent_capabilities()
        if subagent_type and subagent_type in capabilities:
            result["additionalContext"] = (
                f"Agent '{subagent_type}' specializes in: "
                f"{capabilities[subagent_type]}. "
                f"Leverage Opus adaptive thinking for complex sub-tasks."
            )

        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"PreToolUse hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
