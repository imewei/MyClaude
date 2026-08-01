#!/usr/bin/env python3
"""SubagentStop hook for research-suite.

Gates on the STOPPED subagent's identity. Only fires artifact-check logic for
research-spark-orchestrator or scientific-review subagents.
All other agent types exit silently with no output.
"""

import json
import sys

import _hook_io

RESEARCH_AGENT_TYPES = {"research-spark-orchestrator", "scientific-review"}

ARTIFACT_CHECK_PROMPT = (
    "A research-spark or scientific-review subagent just finished. "
    "If its transcript shows a research-spark stage completion "
    "(Stage 1-8 marker like '## Stage N:' or 'artifact:'), verify the stage artifact "
    "(problem statement, falsifiable claim, pre-registration, experimental plan, "
    "analysis plan, results, discussion, manuscript) is present and named per convention. "
    "If the subagent was scientific-review, verify the referee report has all required "
    "sections (summary, strengths, weaknesses, major concerns, minor concerns, "
    "recommendation). Report any missing artifacts so the orchestrator can regenerate "
    "them before advancing."
)


def main() -> None:
    try:
        payload = _hook_io.read_payload()
        agent_type = _hook_io.get_field(
            payload,
            "subagent_type",
            "agent_type",
            "agent_name",
            "matcher_input",
            default="",
        ).strip()

        if agent_type in RESEARCH_AGENT_TYPES:
            result = {"status": "success", "additionalContext": ARTIFACT_CHECK_PROMPT}
            result.update(_hook_io.wrap_context("SubagentStop", ARTIFACT_CHECK_PROMPT))
            json.dump(result, sys.stdout)
        else:
            sys.exit(0)
    except Exception as e:
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
