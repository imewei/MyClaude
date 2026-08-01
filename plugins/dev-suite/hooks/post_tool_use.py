#!/usr/bin/env python3
"""PostToolUse hook for dev-suite Write/Edit operations.

Suggests linting after Python/TypeScript file modifications.
"""

import json
import sys

from _hook_io import read_payload, wrap_context


def main() -> None:
    """Suggest linting after file modifications."""
    try:
        payload = read_payload()

        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {}

        file_path = tool_input.get("file_path") or payload.get("file_path") or ""
        result = {"status": "success"}

        if file_path.endswith(".py"):
            ctx = f"Python file modified: {file_path}. Consider running ruff check on this file."
            result["additionalContext"] = ctx
            result.update(wrap_context("PostToolUse", ctx))
        elif file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            ctx = f"JS/TS file modified: {file_path}. Consider running eslint on this file."
            result["additionalContext"] = ctx
            result.update(wrap_context("PostToolUse", ctx))

        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"PostToolUse hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
