#!/usr/bin/env python3
"""Shared stdin-payload helpers for dev-suite hooks.

Claude Code delivers hook data as a JSON object on stdin. Hooks import this as a
sibling module (``sys.path[0]`` is the hooks dir when run as ``python3 .../foo.py``).
"""

import json
import os
import sys


def read_payload() -> dict:
    """Read the hook JSON payload from stdin. Never raises; returns {} on failure."""
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def get_field(
    payload: dict,
    *candidate_keys: str,
    env_fallback: str | None = None,
    default: str = "unknown",
) -> str:
    """First non-empty value among payload keys, then an env var, then default."""
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if env_fallback:
        value = os.environ.get(env_fallback, "")
        if value.strip():
            return value
    return default


def wrap_context(event_name: str, additional_context: str) -> dict:
    """Correct Claude Code hook-output shape for injecting context.

    A top-level "additionalContext" key is silently ignored by Claude Code —
    only ``hookSpecificOutput.additionalContext`` actually reaches the model.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": additional_context,
        }
    }
