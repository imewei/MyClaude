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
STAGES_COMPLETED_RE = re.compile(r"^\s*stages_completed:\s*\[([^\]]*)\]", re.MULTILINE)


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


def wrap_context(event_name: str, additional_context: str) -> dict[str, Any]:
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


def _immediate_state_children(root: Path) -> list[Path]:
    found: list[Path] = []
    try:
        for child in sorted(root.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            if (child / STATE_FILENAME).is_file():
                found.append(child / STATE_FILENAME)
    except OSError:
        pass
    return found


def find_state_files(cwd: str) -> list[Path]:
    """Locate research-spark `_state.yaml` files.

    Three depths are searched, matching the documented conventions: the
    workspace root itself (`cwd` IS `research-spark/`), one level nested
    (`<workspace>/<idea-slug>/`), and — since research-spark-orchestrator's
    own default workspace is `./research-spark/<idea-slug>/` relative to
    wherever the session cwd actually is (often the repo root, not already
    inside research-spark/) — `cwd/research-spark/<idea-slug>/` as well. No
    recursive walk beyond these three fixed depths — that would match
    vendored trees.
    """
    root = Path(cwd)
    found: list[Path] = []
    if (root / STATE_FILENAME).is_file():
        found.append(root / STATE_FILENAME)
    found.extend(_immediate_state_children(root))
    default_workspace = root / "research-spark"
    if default_workspace.is_dir():
        found.extend(_immediate_state_children(default_workspace))
    return found


def read_current_stage(state_path: Path) -> int | None:
    """Read `current_stage` (the stage IN PROGRESS, not yet completed — see
    stages_completed) from a state file. None if absent or unreadable."""
    try:
        match = CURRENT_STAGE_RE.search(state_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return int(match.group(1)) if match else None


def read_stages_completed(state_path: Path) -> list[int]:
    """Read `stages_completed` (e.g. `[1, 2, 3]`) — the stages whose
    canonical artifact should already exist on disk. Empty list if absent,
    unreadable, or no stage has completed yet."""
    try:
        text = state_path.read_text(encoding="utf-8")
    except OSError:
        return []
    match = STAGES_COMPLETED_RE.search(text)
    if not match:
        return []
    return [int(n) for n in re.findall(r"\d+", match.group(1))]
