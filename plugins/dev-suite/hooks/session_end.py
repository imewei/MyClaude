#!/usr/bin/env python3
"""SessionEnd hook for dev-suite.

Persists a structured progress summary including stack context,
recent commits, and uncommitted changes for the next session.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from _hook_io import get_field, read_payload


def get_recent_commits(cwd: str, limit: int = 5) -> str:
    """Get recent git commits."""
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


def get_head(cwd: str) -> str:
    """Short HEAD hash, so session_start can detect a moved-on branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
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


def get_uncommitted_files(cwd: str) -> str:
    """List uncommitted changes, capped so it cannot crowd out the header."""
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
                text = text[:MAX_UNCOMMITTED_CHARS] + "\n... (truncated)"
            return text
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def main() -> None:
    """Persist dev session progress."""
    try:
        payload = read_payload()
        end_reason = get_field(payload, "reason", "matcher_input")
        cwd = get_field(payload, "cwd", env_fallback="PWD", default=os.getcwd())

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        commits = get_recent_commits(cwd)
        uncommitted = get_uncommitted_files(cwd)
        head = get_head(cwd)

        lines = [
            f"## Session ended: {timestamp}",
            f"Reason: {end_reason}",
            f"HEAD: {head}" if head else "HEAD: unknown",
            "",
            "### Recent commits",
            commits,
        ]

        if uncommitted:
            lines.extend(["", "### Uncommitted changes", uncommitted])

        progress_path = Path(cwd) / ".claude" / "progress" / "dev-suite.md"
        saved = False
        try:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            saved = True
        except OSError as e:
            print(f"SessionEnd hook: could not write {progress_path}: {e}", file=sys.stderr)

        if saved:
            json.dump(
                {
                    "status": "success",
                    "message": f"Session ended: {end_reason}. Progress saved.",
                },
                sys.stdout,
            )
        else:
            json.dump(
                {
                    "status": "warning",
                    "message": (
                        f"Session ended: {end_reason}. "
                        f"Progress NOT saved (could not write {progress_path})."
                    ),
                },
                sys.stdout,
            )
    except Exception as e:
        print(f"SessionEnd hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SessionEnd hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
