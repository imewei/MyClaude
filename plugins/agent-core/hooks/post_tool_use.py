#!/usr/bin/env python3
"""PostToolUse hook for Write/Edit tool calls.

Fires after a Write or Edit tool succeeds. Can be used to trigger
auto-linting or format checks on modified files.
"""

import json
import sys

import _hook_io


def main() -> None:
    """Log file modifications for potential auto-linting."""
    try:
        payload = _hook_io.read_payload()
        tool_input = payload.get("tool_input")

        file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""

        result = {"status": "success"}

        if file_path and file_path.endswith(".py"):
            ctx = f"Python file modified: {file_path}. Consider running ruff check on this file."
            result["additionalContext"] = ctx
            result.update(_hook_io.wrap_context("PostToolUse", ctx))

        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"PostToolUse hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"PostToolUse hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
