#!/usr/bin/env python3
"""PostToolUse hook for dev-suite Write/Edit operations.

Suggests linting after Python/TypeScript file modifications.
"""

import json
import os
import sys


def main() -> None:
    """Suggest linting after file modifications."""
    try:
        tool_input = os.environ.get("TOOL_INPUT", "{}")

        try:
            input_data = json.loads(tool_input)
        except json.JSONDecodeError:
            input_data = {}

        file_path = input_data.get("file_path", "")
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
        json.dump({"status": "error", "message": f"PostToolUse hook error: {e}"}, sys.stdout)


if __name__ == "__main__":
    main()
