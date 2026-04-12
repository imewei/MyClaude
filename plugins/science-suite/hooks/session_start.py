#!/usr/bin/env python3
"""SessionStart hook for science-suite.

Detects computation environment: JAX devices, GPU, Julia env.
"""

import json
import os
import shutil
import subprocess
import sys


def detect_compute_env() -> dict:
    """Detect available compute resources."""
    env = {"jax": False, "gpu": False, "julia": False}

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


def main() -> None:
    """Detect and report compute environment."""
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
