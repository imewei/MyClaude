#!/usr/bin/env python3
"""PostToolUse hook for science-suite Bash operations.

Checks compute job output for NaN/Inf indicating numerical instability.
"""

import json
import re
import sys

from _hook_io import read_payload

# Only match NaN/Inf in a numeric-literal context (after =, :, [, (, , or before
# , ) ]) so unrelated text like "grep -rn nan src/" does not trigger a warning.
NAN_PATTERN = re.compile(r"[=:\[(,]\s*-?nan\b|\bnan\s*[,)\]]", re.IGNORECASE)
INF_PATTERN = re.compile(
    r"[=:\[(,]\s*-?inf(?:inity)?\b|\binf(?:inity)?\s*[,)\]]", re.IGNORECASE
)


def extract_output(payload: dict) -> str:
    """Concatenate the Bash tool_response text fields."""
    response = payload.get("tool_response")
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        return ""
    parts = [response.get(key) for key in ("stdout", "output", "stderr")]
    return "\n".join(p for p in parts if isinstance(p, str) and p)


def check_numerical_integrity(output: str) -> list:
    """Scan output for NaN/Inf appearing as numeric values."""
    warnings = []
    if NAN_PATTERN.search(output):
        warnings.append("'nan' appears as a numeric value in the command output")
    if INF_PATTERN.search(output):
        warnings.append("'inf' appears as a numeric value in the command output")
    return warnings


def main() -> None:
    """Check Bash output for numerical issues."""
    try:
        output = extract_output(read_payload())
        warnings = check_numerical_integrity(output)

        result = {"status": "success"}
        if warnings:
            result["additionalContext"] = (
                "Numerical integrity check: "
                + "; ".join(warnings)
                + " — verify the computation did not diverge."
            )
        elif re.search(r"\d", output):
            result["additionalContext"] = (
                "Numerical integrity check: no NaN/Inf in command output."
            )

        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"PostToolUse hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"PostToolUse hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
