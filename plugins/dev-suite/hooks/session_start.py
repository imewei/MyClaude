#!/usr/bin/env python3
"""SessionStart hook for dev-suite.

Auto-detects project stack: language, framework, test runner, package manager.
"""

import json
import os
import sys
from pathlib import Path


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


def read_progress_file(cwd: str) -> str:
    """Read prior session progress summary if it exists."""
    progress_path = Path(cwd) / ".claude-progress.md"
    if progress_path.exists():
        try:
            text = progress_path.read_text(encoding="utf-8").strip()
            if len(text) > 500:
                text = text[-500:]
            return text
        except OSError:
            pass
    return ""


def main() -> None:
    """Detect project stack and read prior session progress."""
    try:
        cwd = os.environ.get("PWD", os.getcwd())
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

        result = {
            "status": "success",
            "additionalContext": f"Dev environment detected: {context}",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SessionStart hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
