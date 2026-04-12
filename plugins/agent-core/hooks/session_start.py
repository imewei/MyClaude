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
from pathlib import Path


def read_git_summary(cwd: str) -> str:
    """Get recent git activity summary."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
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


def read_progress_file(cwd: str) -> str:
    """Read the most recent session progress summary if it exists."""
    progress_path = Path(cwd) / ".claude-progress.md"
    if progress_path.exists():
        try:
            text = progress_path.read_text(encoding="utf-8").strip()
            # Limit to last 500 chars to stay within context budget
            if len(text) > 500:
                text = text[-500:]
            return text
        except OSError:
            pass
    return ""


def read_uncommitted_status(cwd: str) -> str:
    """Check for uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            return f"{len(lines)} uncommitted file(s)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def get_session_context() -> dict:
    """Build session context with orientation from prior work."""
    cwd = os.environ.get("PWD", os.getcwd())

    parts = []

    # Recent git history
    git_log = read_git_summary(cwd)
    if git_log:
        parts.append(f"Recent commits:\n{git_log}")

    # Uncommitted work
    uncommitted = read_uncommitted_status(cwd)
    if uncommitted:
        parts.append(f"Working tree: {uncommitted}")

    # Progress file from prior session
    progress = read_progress_file(cwd)
    if progress:
        parts.append(f"Prior session progress:\n{progress}")

    if parts:
        context = "\n---\n".join(parts)
    else:
        context = "Fresh session — no prior work context found."

    return {
        "status": "success",
        "additionalContext": f"Session orientation:\n{context}",
    }


def main() -> None:
    """Output session context as JSON."""
    try:
        result = get_session_context()
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"SessionStart hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
