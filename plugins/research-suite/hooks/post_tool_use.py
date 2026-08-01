#!/usr/bin/env python3
"""PostToolUse hook for research-suite Write and Bash operations.

Real completeness check for `scientific-review` deliverables (`./reviews/*.md`
or `.docx`, per that skill's SKILL.md). This replaces the old self-attestation
path in subagent_stop.py — scientific-review is a *skill*, not an agent, so it
never spawns a subagent and SubagentStop can never see it.

The .md fallback is reliably caught via the Write tool. The primary .docx
deliverable is usually produced by a python-docx/pandoc call run through Bash,
not the Write tool — Write alone can't see it. So this also fires on Bash and,
when no Write file_path is available, falls back to scanning reviews/*.docx
for the most recently modified file (mtime within RECENT_SECONDS). Bounded
heuristic, not a guarantee: a docx written by some other means outside that
window is invisible to this check.
"""

import json
import re
import sys
import time
from pathlib import Path

import _hook_io

RECENT_SECONDS = 120

# Anchored to markdown headings (not arbitrary prose — "in summary, ..."
# elsewhere in the body must not count) and broadened to the synonyms
# SKILL.md's own Phase 2 explicitly allows for journal-adapted reviews
# ("journal-specific criteria" may use "Decision"/"Verdict" instead of the
# default template's "Recommendation").
SUMMARY_RE = re.compile(r"^#{1,6}\s.*\bsummary\b", re.MULTILINE | re.IGNORECASE)
RECOMMENDATION_RE = re.compile(
    r"^#{1,6}\s.*\b(?:recommendation|decision|verdict)\b", re.MULTILINE | re.IGNORECASE
)
REQUIRED_SECTIONS = (("summary", SUMMARY_RE), ("recommendation", RECOMMENDATION_RE))


def missing_sections(text: str) -> list[str]:
    return [name for name, pattern in REQUIRED_SECTIONS if not pattern.search(text)]


def read_docx_text(path: Path) -> str:
    """Returns the docx body text. Raises ImportError if python-docx isn't
    installed, or any other exception on a real parse failure — the caller
    handles both explicitly rather than treating a corrupt/unparsable file
    as a silent clean pass."""
    import docx  # python-docx

    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def check_review_file(path: Path) -> str | None:
    if "reviews" not in path.parts or path.suffix not in (".md", ".docx"):
        return None
    if not path.is_file():
        return None

    if path.suffix == ".docx":
        try:
            text = read_docx_text(path)
        except ImportError:
            return (
                f"scientific-review wrote {path.name}, but python-docx isn't "
                "available to verify section completeness — check manually "
                "that Summary and Recommendation sections are present."
            )
        except Exception as e:
            print(f"PostToolUse: failed to parse {path}: {e}", file=sys.stderr)
            return (
                f"scientific-review wrote {path.name}, but it could not be "
                f"parsed to verify section completeness ({e}) — check "
                "manually that Summary and Recommendation sections are present."
            )
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"PostToolUse: could not read {path}: {e}", file=sys.stderr)
            return None

    missing = missing_sections(text)
    if missing:
        return (
            f"scientific-review wrote {path.name}, but it is missing required "
            f"section(s): {', '.join(missing)}. Do not present this review as "
            "complete until they are added."
        )
    return None


def find_recent_docx(cwd: str) -> Path | None:
    """Bash-triggered fallback: newest reviews/*.docx modified within the
    last RECENT_SECONDS, since python-docx/pandoc writes bypass the Write
    tool entirely."""
    reviews = Path(cwd) / "reviews" if cwd else None
    if not reviews or not reviews.is_dir():
        return None
    try:
        candidates = sorted(
            reviews.glob("*.docx"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except OSError:
        return None
    if not candidates:
        return None
    newest = candidates[0]
    if time.time() - newest.stat().st_mtime > RECENT_SECONDS:
        return None
    return newest


def main() -> None:
    try:
        payload = _hook_io.read_payload()
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        file_path = tool_input.get("file_path") or payload.get("file_path") or ""

        ctx = None
        if file_path:
            ctx = check_review_file(Path(file_path))
        if ctx is None:
            cwd = _hook_io.get_field(payload, "cwd", default="")
            recent = find_recent_docx(cwd) if cwd else None
            if recent:
                ctx = check_review_file(recent)

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
