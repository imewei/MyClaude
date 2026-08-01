#!/usr/bin/env python3
"""SessionStart hook for science-suite.

Detects computation environment: JAX, GPU, Julia env.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _hook_io import get_field, read_payload

PROGRESS_RELPATH = Path(".claude") / "progress" / "science-suite.md"
PROGRESS_MAX_CHARS = 1500
PROGRESS_MAX_AGE = timedelta(hours=24)
TIMESTAMP_PREFIX = "## Session ended: "
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M UTC"
# Both probes must fit inside the hook's own 10s budget declared in hooks.json.
PROBE_TIMEOUT = 3
JAX_PROBE = (
    "import importlib.util,sys; "
    "sys.exit(0 if importlib.util.find_spec('jax') else 1)"
)


def detect_compute_env() -> dict:
    """Detect available compute resources."""
    env: dict[str, object] = {"jax": False, "gpu": False, "julia": False}

    try:
        # find_spec, not jax.devices() — presence detection must not initialize
        # a CUDA context. Run out-of-process to probe the same python3 as before.
        result = subprocess.run(
            ["python3", "-c", JAX_PROBE],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            check=False,
        )
        env["jax"] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    env["gpu"] = shutil.which("nvidia-smi") is not None

    if shutil.which("julia"):
        env["julia"] = True
        try:
            result = subprocess.run(
                ["julia", "-e", "println(VERSION)"],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT,
                check=False,
            )
            if result.returncode == 0:
                env["julia_version"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return env


def read_progress_file(cwd: str) -> str:
    """Read prior session progress if it exists and is less than a day old."""
    progress_path = Path(cwd) / PROGRESS_RELPATH
    try:
        text = progress_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""

    first_line = text.split("\n", 1)[0]
    if not first_line.startswith(TIMESTAMP_PREFIX):
        return ""
    try:
        saved_at = datetime.strptime(
            first_line[len(TIMESTAMP_PREFIX) :].strip(), TIMESTAMP_FORMAT
        ).replace(tzinfo=UTC)
    except ValueError:
        return ""
    if datetime.now(UTC) - saved_at > PROGRESS_MAX_AGE:
        return ""

    # Truncate from the head so the timestamp line always survives.
    if len(text) > PROGRESS_MAX_CHARS:
        text = text[:PROGRESS_MAX_CHARS].rsplit("\n", 1)[0] + "\n... (truncated)"
    return text


def main() -> None:
    """Detect compute environment and read prior session progress."""
    try:
        payload = read_payload()
        env = detect_compute_env()

        parts = []
        if env["jax"]:
            parts.append("JAX available")
        if env["gpu"]:
            parts.append("GPU detected")
        if env["julia"]:
            version = env.get("julia_version", "unknown")
            parts.append(f"Julia {version}")

        context = ". ".join(parts) if parts else "No scientific compute stack detected"
        sections = [f"Science compute env: {context}"]

        cwd = get_field(payload, "cwd", env_fallback="PWD", default=os.getcwd())
        progress = read_progress_file(cwd)
        if progress:
            sections.append(
                "Prior science-suite session summary (from "
                f"{PROGRESS_RELPATH.as_posix()}, may be stale):\n{progress}"
            )

        result = {
            "status": "success",
            "additionalContext": "\n\n".join(sections),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"SessionStart hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SessionStart hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
