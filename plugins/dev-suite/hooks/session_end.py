#!/usr/bin/env python3
"""SessionEnd hook for dev-suite.

Persists a structured progress summary including stack context,
recent commits, and uncommitted changes for the next session.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_recent_commits(cwd: str, limit: int = 5) -> str:
    """Get recent git commits."""
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


def get_test_status(cwd: str) -> str:
    """Quick check if tests were last passing."""
    # Check for common test result indicators without running tests
    for marker in [".pytest_cache", "node_modules/.cache/jest"]:
        if (Path(cwd) / marker).exists():
            return "Test cache present (run tests to verify)"
    return ""


def main() -> None:
    """Persist dev session progress."""
    try:
        input_data = json.load(sys.stdin)
        end_reason = input_data.get("matcher_input", "unknown")
        cwd = os.environ.get("PWD", os.getcwd())

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        commits = get_recent_commits(cwd)
        uncommitted = get_uncommitted_files(cwd)
        test_status = get_test_status(cwd)

        lines = [
            f"## Session ended: {timestamp}",
            f"Reason: {end_reason}",
            "",
            "### Recent commits",
            commits,
        ]

        if uncommitted:
            lines.extend(["", "### Uncommitted changes", uncommitted])
        if test_status:
            lines.extend(["", "### Test status", test_status])

        progress_path = Path(cwd) / ".claude-progress.md"
        try:
            progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass

        json.dump(
            {
                "status": "success",
                "message": f"Session ended: {end_reason}. Progress saved.",
            },
            sys.stdout,
        )
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SessionEnd hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
