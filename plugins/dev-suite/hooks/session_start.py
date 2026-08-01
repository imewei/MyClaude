#!/usr/bin/env python3
"""SessionStart hook for dev-suite.

Auto-detects project stack: language, framework, test runner, package manager.
"""

import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _hook_io import get_field, read_payload, wrap_context


def detect_stack(cwd: str) -> dict:
    """Detect project stack from file presence."""
    root = Path(cwd)
    stack = {"languages": [], "package_managers": [], "test_runners": []}

    if list(root.glob("*.py")) or (root / "pyproject.toml").exists():
        stack["languages"].append("python")
    if list(root.glob("*.ts")) or (root / "tsconfig.json").exists():
        stack["languages"].append("typescript")
    if list(root.glob("*.js")) or (root / "package.json").exists():
        stack["languages"].append("javascript")
    if list(root.glob("*.go")) or (root / "go.mod").exists():
        stack["languages"].append("go")
    if list(root.glob("*.rs")) or (root / "Cargo.toml").exists():
        stack["languages"].append("rust")

    if (root / "uv.lock").exists():
        stack["package_managers"].append("uv")
    elif (root / "poetry.lock").exists():
        stack["package_managers"].append("poetry")
    if (root / "package-lock.json").exists():
        stack["package_managers"].append("npm")
    elif (root / "yarn.lock").exists():
        stack["package_managers"].append("yarn")
    elif (root / "pnpm-lock.yaml").exists():
        stack["package_managers"].append("pnpm")

    if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
        stack["test_runners"].append("pytest")
    if (root / "jest.config.js").exists() or (root / "jest.config.ts").exists():
        stack["test_runners"].append("jest")
    if (root / "vitest.config.ts").exists():
        stack["test_runners"].append("vitest")

    return stack


MAX_PROGRESS_CHARS = 500
MAX_PROGRESS_AGE = timedelta(hours=24)


def read_progress_file(cwd: str) -> str:
    """Read prior session progress, skipping it if stale or undated."""
    progress_path = Path(cwd) / ".claude" / "progress" / "dev-suite.md"
    try:
        text = progress_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""

    # First line is "## Session ended: YYYY-MM-DD HH:MM UTC" — no parseable
    # timestamp means we cannot tell how old this is, so don't inject it.
    match = re.match(r"## Session ended: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC", text)
    if not match:
        return ""
    written = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    if datetime.now(UTC) - written > MAX_PROGRESS_AGE:
        return ""

    if len(text) > MAX_PROGRESS_CHARS:
        text = text[:MAX_PROGRESS_CHARS] + "\n... (truncated)"
    return text


def main() -> None:
    """Detect project stack and read prior session progress."""
    try:
        cwd = get_field(read_payload(), "cwd", env_fallback="PWD", default=os.getcwd())
        stack = detect_stack(cwd)

        parts = []
        if stack["languages"]:
            parts.append(f"Languages: {', '.join(stack['languages'])}")
        if stack["package_managers"]:
            parts.append(f"Package managers: {', '.join(stack['package_managers'])}")
        if stack["test_runners"]:
            parts.append(f"Test runners: {', '.join(stack['test_runners'])}")

        context = ". ".join(parts) if parts else "No specific stack detected"

        # Read prior session progress
        progress = read_progress_file(cwd)
        if progress:
            context += f"\n\nPrior session progress:\n{progress}"

        ctx = f"Dev environment detected: {context}"
        result = {"status": "success", "additionalContext": ctx}
        result.update(wrap_context("SessionStart", ctx))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"SessionStart hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SessionStart hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
