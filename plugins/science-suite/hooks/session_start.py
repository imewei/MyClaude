#!/usr/bin/env python3
"""SessionStart hook for science-suite.

Detects computation environment: JAX devices, GPU, Julia env.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def detect_compute_env() -> dict:
    """Detect available compute resources."""
    env: dict[str, object] = {"jax": False, "gpu": False, "julia": False}

    try:
        result = subprocess.run(
            ["python3", "-c", "import jax; print(jax.devices())"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            env["jax"] = True
            env["jax_devices"] = result.stdout.strip()
            if "gpu" in result.stdout.lower() or "cuda" in result.stdout.lower():
                env["gpu"] = True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if shutil.which("julia"):
        env["julia"] = True
        try:
            result = subprocess.run(
                ["julia", "-e", "println(VERSION)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                env["julia_version"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return env


def read_progress_file(cwd: str) -> str:
    """Read prior session progress summary if it exists."""
    progress_path = Path(cwd) / ".claude-progress.md"
    if progress_path.exists():
        try:
            text = progress_path.read_text(encoding="utf-8").strip()
            if len(text) > 500:
                text = text[-500:]
            return text
        except OSError:
            pass
    return ""


def main() -> None:
    """Detect compute environment and read prior session progress."""
    try:
        env = detect_compute_env()

        parts = []
        if env["jax"]:
            devices = env.get("jax_devices", "unknown")
            parts.append(f"JAX available (devices: {devices})")
        if env["gpu"]:
            parts.append("GPU detected")
        if env["julia"]:
            version = env.get("julia_version", "unknown")
            parts.append(f"Julia {version}")

        context = ". ".join(parts) if parts else "No scientific compute stack detected"

        # Read prior session progress
        cwd = os.environ.get("PWD", os.getcwd())
        progress = read_progress_file(cwd)
        if progress:
            context += f"\n\nPrior session progress:\n{progress}"

        result = {
            "status": "success",
            "additionalContext": f"Science compute env: {context}",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SessionStart hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
