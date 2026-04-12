#!/usr/bin/env python3
"""SessionEnd hook for agent-core plugin.

Persists a structured progress summary to .claude-progress.md so the next
session can quickly orient itself. Inspired by Anthropic's "Effective
harnesses for long-running agents" — each session should leave clear
artifacts for the next.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_recent_commits(cwd: str, limit: int = 5) -> str:
    """Get commits made during this session (last N)."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{limit}"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "No git history available"


def get_uncommitted_files(cwd: str) -> str:
    """List uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def write_progress(cwd: str, end_reason: str) -> None:
    """Write structured progress summary for next session."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
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

    progress_path = Path(cwd) / ".claude-progress.md"
    try:
        progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass  # Non-fatal — don't block session end


def main() -> None:
    """Persist progress summary and log session end."""
    try:
        input_data = json.load(sys.stdin)
        end_reason = input_data.get("matcher_input", "unknown")
        cwd = os.environ.get("PWD", os.getcwd())

        write_progress(cwd, end_reason)

        result = {
            "status": "success",
            "message": f"Session ended: {end_reason}. Progress saved to .claude-progress.md",
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
