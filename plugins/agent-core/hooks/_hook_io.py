"""Shared stdin payload helpers for agent-core hooks.

Claude Code delivers hook payloads as JSON on stdin. Several hooks here
previously read environment variables that nothing ever sets; these helpers
read stdin once and fall back to the old env var so no behavior regresses.
"""

import json
import os
import sys
from typing import Any


def read_payload() -> dict[str, Any]:
    """Parse the hook payload from stdin. Returns {} on empty/invalid input."""
    try:
        # No stdin attached (interactive/no pipe) — reading would block.
        if sys.stdin is None or sys.stdin.isatty():
            return {}
        data = json.load(sys.stdin)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def get_field(
    payload: dict[str, Any],
    *candidate_keys: str,
    env_fallback: str | None = None,
    default: str = "unknown",
) -> str:
    """First non-empty value among payload[key] for each key, then env, then default."""
    for key in candidate_keys:
        value = payload.get(key)
        if value not in (None, "", {}, []):
            return str(value)
    if env_fallback:
        value = os.environ.get(env_fallback)
        if value:
            return value
    return default
