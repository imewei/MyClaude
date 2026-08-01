#!/usr/bin/env python3
"""SubagentStop hook for science-suite.

Collects results from parallel science agents (parameter sweeps, etc.).
"""

import json
import sys

from _hook_io import get_field, read_payload

# Agents that produce numerical results worth validating. sci-workflow-engineer
# (LLM/RAG tooling) and python-pro (packaging, typing, glue) are excluded.
NUMERICAL_AGENTS = {
    "jax-pro",
    "julia-ml-hpc",
    "julia-pro",
    "ml-expert",
    "neural-network-master",
    "nonlinear-dynamics-expert",
    "pinn-engineer",
    "simulation-expert",
    "statistical-physicist",
}


def main() -> None:
    """Log science subagent completion."""
    try:
        payload = read_payload()
        agent_name = get_field(payload, "agent_name", "subagent_type", "agent_type")

        result: dict[str, str] = {"status": "success"}
        if agent_name in NUMERICAL_AGENTS:
            result["additionalContext"] = (
                f"Science agent '{agent_name}' completed. "
                "Check output for numerical validity before proceeding."
            )

        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
