#!/usr/bin/env python3
"""SubagentStop hook for research-suite.

Gates on agent_type: only fires artifact-check logic for
research-spark-orchestrator or scientific-review subagents.
All other agent types exit silently with no output.
"""

import json
import sys

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
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    agent_type = (data.get("agent_type") or "").strip()
    if agent_type not in RESEARCH_AGENT_TYPES:
        # Not a research agent — exit silently, no output.
        sys.exit(0)

    # Research agent: emit the artifact-check prompt as a systemMessage
    # so the orchestrator performs the verification.
    json.dump({"systemMessage": ARTIFACT_CHECK_PROMPT}, sys.stdout)


if __name__ == "__main__":
    main()
