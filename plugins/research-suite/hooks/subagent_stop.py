#!/usr/bin/env python3
"""SubagentStop hook for research-suite.

Gates on the STOPPED subagent's identity. Only fires artifact-check logic for
research-spark-orchestrator or scientific-review subagents.
All other agent types exit silently with no output.
"""

import json
import os
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

LOG_PATH = os.path.expanduser("~/.claude/research-suite-subagent-stop-debug.jsonl")


def main() -> None:
    raw = sys.stdin.read()
    parse_error = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        data = {}
        parse_error = exc.__class__.__name__

    # Log full payload for diagnosis.
    try:
        entry = {
            "stdin_empty": raw == "",
            "stdin_keys": list(data.keys()) if isinstance(data, dict) else repr(type(data)),
            "stdin_data": data,
            "parse_error": parse_error,
        }
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    if not isinstance(data, dict):
        sys.exit(0)

    agent_type = data.get("agent_type")
    if not isinstance(agent_type, str) or not agent_type.strip():
        sys.exit(0)

    if agent_type.strip() in RESEARCH_AGENT_TYPES:
        json.dump({"systemMessage": ARTIFACT_CHECK_PROMPT}, sys.stdout)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
