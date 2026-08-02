#!/usr/bin/env python3
"""SubagentStop hook for dev-suite.

Logs subagent completion for dev workflow tracking (agent_name only — this
hook does not parse or validate any test/review payload; no such payload
is available at SubagentStop).

Also does a real check against three-brain Team mode's "MUST use the Bash
tool to invoke codex/agy" rule (skills/three-brain/references/agent-prompts.md):
if the subagent's own transcript presents itself as a Codex or Agy review but
the transcript contains no Bash invocation of that CLI, flag it — the CLI
requirement was previously prose-only (SKILL.md text, no PreToolUse gate).
This is a text-scan heuristic on the transcript, not a structured tool-call
ledger; upgrade if false negatives show up in practice.
"""

import json
import re
import sys
from pathlib import Path

from _hook_io import get_field, read_payload, wrap_context

# The reviewer subagent's own transcript contains "Codex Code Review" or
# "Codex Content Review" (agent-prompts.md:80,119,168,206 — what the reviewer
# is instructed to write). The bare "Codex Review" heading only appears in
# SKILL.md's Team-Lead *consolidation* format, never in the reviewer
# subagent's own transcript — matching on that string would never fire.
CLAIMS_CODEX_RE = re.compile(r"\bCodex (?:Code|Content) Review\b", re.IGNORECASE)
CLAIMS_AGY_RE = re.compile(r"\bAgy (?:Code|Content) Review\b", re.IGNORECASE)
# `(?:[^"\\]|\\.)*` (not `[^"]*`) so a JSON-escaped quote inside an earlier
# argument (e.g. `codex exec --prompt \"review\"`) doesn't terminate the
# match before reaching "codex"/"agy".
CODEX_CMD_RE = re.compile(r'"command"\s*:\s*"(?:[^"\\]|\\.)*\bcodex\b', re.IGNORECASE)
AGY_CMD_RE = re.compile(r'"command"\s*:\s*"(?:[^"\\]|\\.)*\bagy\b', re.IGNORECASE)


def check_reviewer_transcript(transcript_path: str) -> str | None:
    """Flag a reviewer transcript that claims a Codex/Agy review without
    ever calling the CLI. Returns None if unreadable, irrelevant, or clean."""
    if not transcript_path:
        return None
    path = Path(transcript_path)
    if not path.is_file():
        print(f"SubagentStop: transcript_path not found: {path}", file=sys.stderr)
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        print(f"SubagentStop: could not read transcript {path}: {e}", file=sys.stderr)
        return None

    claims_codex = CLAIMS_CODEX_RE.search(text) is not None
    claims_agy = CLAIMS_AGY_RE.search(text) is not None
    if not (claims_codex or claims_agy):
        return None

    missing = []
    if claims_codex and not CODEX_CMD_RE.search(text):
        missing.append("Codex")
    if claims_agy and not AGY_CMD_RE.search(text):
        missing.append("Agy")
    if not missing:
        return None

    names = "/".join(missing)
    return (
        f"three-brain integrity check: this subagent's transcript presents a {names} "
        f"Review section, but no Bash call to the {names.lower()} CLI was found "
        "in its transcript. Do not present this as an external-model review — "
        "relabel it '[Claude Fallback]' unless the CLI invocation is confirmed."
    )


def main() -> None:
    """Log subagent completion for dev workflow tracking."""
    try:
        payload = read_payload()
        agent_name = get_field(
            payload,
            "agent_name",
            "subagent_type",
            "agent",
            "name",
            env_fallback="AGENT_NAME",
        )
        transcript_path = get_field(payload, "transcript_path", default="")

        ctx = f"Dev-suite agent '{agent_name}' completed."
        flag = check_reviewer_transcript(transcript_path)
        if flag:
            ctx += f"\n\n{flag}"

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
