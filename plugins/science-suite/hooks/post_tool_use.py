#!/usr/bin/env python3
"""PostToolUse hook for science-suite Bash operations.

Checks compute job output for NaN/Inf indicating numerical instability.
"""

import json
import os
import re
import sys


def check_numerical_integrity(output: str) -> list:
    """Scan output for NaN/Inf indicators."""
    warnings = []

    nan_patterns = [r"\bnan\b", r"\bNaN\b", r"\bNAN\b", r"not a number"]
    inf_patterns = [r"\binf\b", r"\bInf\b", r"\bINF\b", r"-inf\b", r"infinity"]

    for pattern in nan_patterns:
        if re.search(pattern, output):
            warnings.append("NaN detected in output — possible numerical instability")
            break

    for pattern in inf_patterns:
        if re.search(pattern, output):
            warnings.append("Inf detected in output — possible overflow or divergence")
            break

    return warnings


def main() -> None:
    """Check Bash output for numerical issues."""
    try:
        tool_output = os.environ.get("TOOL_OUTPUT", "")
        warnings = check_numerical_integrity(tool_output)

        result = {"status": "success"}
        if warnings:
            result["additionalContext"] = (
                "Numerical integrity warning: " + "; ".join(warnings)
            )

        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"PostToolUse hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
