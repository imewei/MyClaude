#!/usr/bin/env python3
"""SessionEnd hook for science-suite.

Persists a structured progress summary including compute environment
context and recent work for the next session.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from _hook_io import get_field, read_payload

PROGRESS_RELPATH = Path(".claude") / "progress" / "science-suite.md"
MAX_STATUS_CHARS = 2000


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


def get_uncommitted_files(cwd: str) -> str:
    """List uncommitted changes, capped so it cannot crowd out the summary."""
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
            status = result.stdout.strip()
            if len(status) > MAX_STATUS_CHARS:
                status = status[:MAX_STATUS_CHARS].rsplit("\n", 1)[0] + "\n... (truncated)"
            return status
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def main() -> None:
    """Persist science session progress."""
    try:
        payload = read_payload()
        end_reason = get_field(payload, "reason", "matcher_input")
        cwd = get_field(payload, "cwd", env_fallback="PWD", default=os.getcwd())

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

        progress_path = Path(cwd) / PROGRESS_RELPATH
        try:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as e:
            print(f"SessionEnd hook: could not write {progress_path}: {e}", file=sys.stderr)

        json.dump(
            {
                "status": "success",
                "message": f"Session ended: {end_reason}. Progress saved.",
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
