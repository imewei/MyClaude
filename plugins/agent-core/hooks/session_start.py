#!/usr/bin/env python3
"""Session start hook for agent-core plugin.

Reads git log, progress files, and memory to orient agents at session start.
Inspired by Anthropic's "Effective harnesses for long-running agents":
each new session should quickly understand the state of prior work.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import _hook_io

PROGRESS_HEADER = "## Session ended: "
MAX_PROGRESS_CHARS = 800
MAX_PROGRESS_AGE = timedelta(hours=24)


def read_git_summary(cwd: str) -> str:
    """Get recent git activity summary."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def parse_progress_timestamp(text: str) -> datetime | None:
    """Read the `## Session ended: <ts> UTC` header written by session_end."""
    first_line = text.splitlines()[0] if text else ""
    if not first_line.startswith(PROGRESS_HEADER):
        return None
    stamp = first_line[len(PROGRESS_HEADER) :].strip().removesuffix(" UTC").strip()
    try:
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    except ValueError:
        return None


def read_progress_file(cwd: str) -> str:
    """Read the prior session's progress summary, if present and recent."""
    progress_path = Path(cwd) / ".claude" / "progress" / "agent-core.md"
    if not progress_path.exists():
        return ""
    try:
        text = progress_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        sys.stderr.write(f"[SessionStart] Could not read {progress_path}: {e}\n")
        return ""

    written_at = parse_progress_timestamp(text)
    # No parsable timestamp means we can't tell how old this is — treat as stale.
    if written_at is None or datetime.now(UTC) - written_at > MAX_PROGRESS_AGE:
        return ""

    # Truncate from the head so the timestamp header always survives.
    if len(text) > MAX_PROGRESS_CHARS:
        text = text[:MAX_PROGRESS_CHARS].rsplit("\n", 1)[0] + "\n... (truncated)"
    return text


def read_uncommitted_status(cwd: str) -> str:
    """Check for uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            return f"{len(lines)} uncommitted file(s)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def get_session_context(cwd: str) -> dict:
    """Build session context with orientation from prior work."""

    parts = []

    # Recent git history
    git_log = read_git_summary(cwd)
    if git_log:
        parts.append(f"Recent commits:\n{git_log}")

    # The progress file already lists uncommitted files, so the separate
    # working-tree count is only added when there is no progress block.
    progress = read_progress_file(cwd)
    if progress:
        parts.append(
            "Prior session (agent-core, historical — verify before acting):\n" + progress
        )
    else:
        uncommitted = read_uncommitted_status(cwd)
        if uncommitted:
            parts.append(f"Working tree: {uncommitted}")

    if parts:
        context = "\n---\n".join(parts)
    else:
        context = "Fresh session — no prior work context found."

    ctx = f"Session orientation:\n{context}"
    result = {"status": "success", "additionalContext": ctx}
    result.update(_hook_io.wrap_context("SessionStart", ctx))
    return result


def main() -> None:
    """Output session context as JSON."""
    try:
        payload = _hook_io.read_payload()
        cwd = _hook_io.get_field(payload, "cwd", env_fallback="PWD", default=os.getcwd())
        result = get_session_context(cwd)
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"SessionStart hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
