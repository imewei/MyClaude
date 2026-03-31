#!/usr/bin/env python3
"""PostToolUse hook for Write/Edit tool calls.

Fires after a Write or Edit tool succeeds. Can be used to trigger
auto-linting or format checks on modified files.
"""

import json
import os
import sys


def main() -> None:
    """Log file modifications for potential auto-linting."""
    try:
        tool_input = os.environ.get("TOOL_INPUT", "{}")

        try:
            input_data = json.loads(tool_input)
        except json.JSONDecodeError:
            input_data = {}

        file_path = input_data.get("file_path", "")

        result = {"status": "success"}

        if file_path and file_path.endswith(".py"):
            result["additionalContext"] = (
                f"Python file modified: {file_path}. "
                "Consider running ruff check on this file."
            )

        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump({"status": "error", "message": f"PostToolUse hook error: {e}"}, sys.stdout)


if __name__ == "__main__":
    main()
