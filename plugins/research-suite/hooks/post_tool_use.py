#!/usr/bin/env python3
"""PostToolUse hook for research-suite Write operations.

Real completeness check for `scientific-review` deliverables (`./reviews/*.md`
or `.docx`, per that skill's SKILL.md). This replaces the old self-attestation
path in subagent_stop.py — scientific-review is a *skill*, not an agent, so it
never spawns a subagent and SubagentStop can never see it. Write is the only
event that reliably fires when the deliverable is actually produced.
"""

import json
import sys
from pathlib import Path

import _hook_io

# Every review mode's default section order (SKILL.md Phase 2/5) ends in a
# recommendation; that is the one element no review mode omits.
REQUIRED_SECTIONS = ("summary", "recommendation")


def check_review_file(file_path: str) -> str | None:
    path = Path(file_path)
    if "reviews" not in path.parts:
        return None
    if path.suffix not in (".md", ".docx"):
        return None
    if not path.is_file():
        return None

    if path.suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            return (
                f"scientific-review wrote {path.name}, but python-docx isn't "
                "available to verify section completeness — check manually "
                "that Summary and Recommendation sections are present."
            )
        try:
            text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        except Exception:
            return None
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

    lower = text.lower()
    missing = [s for s in REQUIRED_SECTIONS if s not in lower]
    if missing:
        return (
            f"scientific-review wrote {path.name}, but it is missing required "
            f"section(s): {', '.join(missing)}. Do not present this review as "
            "complete until they are added."
        )
    return None


def main() -> None:
    try:
        payload = _hook_io.read_payload()
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        file_path = tool_input.get("file_path") or payload.get("file_path") or ""

        ctx = check_review_file(file_path) if file_path else None
        if ctx:
            result = {"status": "success", "additionalContext": ctx}
            result.update(_hook_io.wrap_context("PostToolUse", ctx))
            json.dump(result, sys.stdout)
        else:
            sys.exit(0)
    except Exception as e:
        print(f"PostToolUse hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"PostToolUse hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
