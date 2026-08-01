#!/usr/bin/env python3
"""SessionEnd hook for agent-core plugin.

Persists a structured progress summary to .claude/progress/agent-core.md so the
next session can quickly orient itself. Inspired by Anthropic's "Effective
harnesses for long-running agents" — each session should leave clear
artifacts for the next.

The path is namespaced per suite: agent-core, dev-suite, and science-suite all
write progress files, and an unnamespaced path made them overwrite each other.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import _hook_io


def get_recent_commits(cwd: str, limit: int = 5) -> str:
    """Get commits made during this session (last N)."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{limit}"],
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
    return "No git history available"


MAX_UNCOMMITTED_CHARS = 2000


def get_uncommitted_files(cwd: str) -> str:
    """List uncommitted changes, capped so the progress file stays small."""
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
            text = result.stdout.strip()
            if len(text) > MAX_UNCOMMITTED_CHARS:
                kept = text[:MAX_UNCOMMITTED_CHARS].rsplit("\n", 1)[0]
                total = len(text.splitlines())
                shown = len(kept.splitlines())
                return f"{kept}\n... ({total - shown} more file(s) truncated)"
            return text
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def write_progress(cwd: str, end_reason: str) -> None:
    """Write structured progress summary for next session."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    commits = get_recent_commits(cwd)
    uncommitted = get_uncommitted_files(cwd)

    lines = [
        f"## Session ended: {timestamp}",
        f"Reason: {end_reason}",
        "",
        "### Recent commits",
        commits,
    ]

    if uncommitted:
        lines.extend(["", "### Uncommitted changes", uncommitted])

    progress_path = Path(cwd) / ".claude" / "progress" / "agent-core.md"
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        # Non-fatal — don't block session end, but don't fail silently either.
        sys.stderr.write(f"[SessionEnd] Could not write {progress_path}: {e}\n")


def main() -> None:
    """Persist progress summary and log session end."""
    try:
        payload = _hook_io.read_payload()
        end_reason = _hook_io.get_field(payload, "reason", "matcher_input")
        cwd = _hook_io.get_field(payload, "cwd", env_fallback="PWD", default=os.getcwd())

        write_progress(cwd, end_reason)

        result = {
            "status": "success",
            "message": (
                f"Session ended: {end_reason}. "
                "Progress saved to .claude/progress/agent-core.md"
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"SessionEnd hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
