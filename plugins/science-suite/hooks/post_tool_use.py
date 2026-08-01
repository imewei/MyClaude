#!/usr/bin/env python3
"""PostToolUse hook for science-suite Bash operations.

Checks compute job output for NaN/Inf indicating numerical instability.
"""

import json
import re
import sys

from _hook_io import read_payload, wrap_context

# Bare word-boundary match so numpy/Julia/pandas whitespace-delimited output
# (e.g. "[ 1.  nan  2.]", "r_hat  nan") is caught. To avoid flagging
# "grep -rn nan src/", lines that look like a search-command invocation are
# skipped entirely rather than requiring nan/inf to sit next to punctuation.
SEARCH_CMD_RE = re.compile(r"\b(grep|rg|ripgrep|ag|ack|fd|find)\b", re.IGNORECASE)
NAN_TOKEN = re.compile(r"(?<![\w.])-?nan\b", re.IGNORECASE)
INF_TOKEN = re.compile(r"(?<![\w.])-?inf(?:inity)?\b", re.IGNORECASE)
NUMERIC_BOUNDARY_CHARS = set("[](){}=:,.")


def _in_numeric_context(line: str, start: int, end: int) -> bool:
    """A nan/inf token counts only if it sits in a numeric-looking context:
    punctuation/digit/line-boundary immediately adjacent (ignoring
    intervening whitespace), or a 2+ run of whitespace touching the token
    (aligned tabular/array output). This excludes ordinary prose like
    "fix nan handling", which flanks the token with a single space and words.
    """
    left, right = line[:start], line[end:]

    left_stripped = left.rstrip(" \t")
    if (
        not left_stripped
        or left_stripped[-1].isdigit()
        or left_stripped[-1] in NUMERIC_BOUNDARY_CHARS
    ):
        return True
    if len(left) - len(left_stripped) >= 2:
        return True

    right_stripped = right.lstrip(" \t")
    if (
        not right_stripped
        or right_stripped[0].isdigit()
        or right_stripped[0] in NUMERIC_BOUNDARY_CHARS
    ):
        return True
    return len(right) - len(right_stripped) >= 2


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
    """Scan output for NaN/Inf appearing as numeric values, skipping search-command lines."""
    saw_nan = saw_inf = False
    for line in output.splitlines():
        if SEARCH_CMD_RE.search(line):
            continue
        if not saw_nan:
            saw_nan = any(
                _in_numeric_context(line, m.start(), m.end())
                for m in NAN_TOKEN.finditer(line)
            )
        if not saw_inf:
            saw_inf = any(
                _in_numeric_context(line, m.start(), m.end())
                for m in INF_TOKEN.finditer(line)
            )

    warnings = []
    if saw_nan:
        warnings.append("'nan' appears as a numeric value in the command output")
    if saw_inf:
        warnings.append("'inf' appears as a numeric value in the command output")
    return warnings


def main() -> None:
    """Check Bash output for numerical issues."""
    try:
        output = extract_output(read_payload())
        warnings = check_numerical_integrity(output)

        result = {"status": "success"}
        if warnings:
            # A regex scan can warn on a hit; it cannot certify absence, so no
            # "clean" claim is emitted on the no-warnings path.
            ctx = (
                "Numerical integrity check: "
                + "; ".join(warnings)
                + " — verify the computation did not diverge."
            )
            result["additionalContext"] = ctx
            result.update(wrap_context("PostToolUse", ctx))

        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"PostToolUse hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"PostToolUse hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
