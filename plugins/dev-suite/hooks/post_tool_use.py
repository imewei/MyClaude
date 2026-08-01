#!/usr/bin/env python3
"""PostToolUse hook for dev-suite Write/Edit operations.

Suggests linting after Python/TypeScript file modifications.
"""

import json
import os
import sys

from _hook_io import read_payload


def main() -> None:
    """Suggest linting after file modifications."""
    try:
        payload = read_payload()

        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            try:
                tool_input = json.loads(os.environ.get("TOOL_INPUT", "{}"))
            except json.JSONDecodeError:
                tool_input = {}
            if not isinstance(tool_input, dict):
                tool_input = {}

        file_path = tool_input.get("file_path") or payload.get("file_path") or ""
        result = {"status": "success"}

        if file_path.endswith(".py"):
            result["additionalContext"] = (
                f"Python file modified: {file_path}. "
                "Consider running ruff check on this file."
            )
        elif file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            result["additionalContext"] = (
                f"JS/TS file modified: {file_path}. "
                "Consider running eslint on this file."
            )

        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"PostToolUse hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
