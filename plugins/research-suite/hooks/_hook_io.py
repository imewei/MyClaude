#!/usr/bin/env python3
"""Shared stdin payload and workspace helpers for research-suite hooks.

Hooks run as bare `python3 .../hooks/foo.py`, outside the project venv, so
everything here stays stdlib-only (no yaml import).
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

STATE_FILENAME = "_state.yaml"
CURRENT_STAGE_RE = re.compile(r"^\s*current_stage:\s*(\d+)", re.MULTILINE)


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


def find_state_files(cwd: str) -> list[Path]:
    """Locate research-spark `_state.yaml` files.

    research-spark's SKILL.md puts it at the workspace root; the orchestrator
    nests one level (`<workspace>/<idea-slug>/`). Only those two depths are
    searched — a recursive walk would match vendored trees.
    """
    root = Path(cwd)
    found: list[Path] = []
    if (root / STATE_FILENAME).is_file():
        found.append(root / STATE_FILENAME)
    try:
        for child in sorted(root.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            if (child / STATE_FILENAME).is_file():
                found.append(child / STATE_FILENAME)
    except OSError:
        pass
    return found


def read_current_stage(state_path: Path) -> int | None:
    """Read `current_stage` from a state file. None if absent or unreadable."""
    try:
        match = CURRENT_STAGE_RE.search(state_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return int(match.group(1)) if match else None
