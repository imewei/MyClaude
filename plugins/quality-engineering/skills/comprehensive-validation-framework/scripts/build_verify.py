#!/usr/bin/env python3
"""
Build verification across different build systems.

Usage:
    python build_verify.py [--clean] [--release]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list) -> int:
    """Run a command and return exit code."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Verify build configuration")
    parser.add_argument("--clean", action="store_true", help="Clean before building")
    parser.add_argument("--release", action="store_true", help="Build in release mode")

    args = parser.parse_args()
    root = Path.cwd()

    # JavaScript/TypeScript
    if (root / "package.json").exists():
        print("🏗️  Building JavaScript/TypeScript project...")
        if args.clean:
            run_command(["npm", "run", "clean"])
        exit_code = run_command(["npm", "run", "build"])
        sys.exit(exit_code)

    # Python
    if (root / "pyproject.toml").exists():
        print("🏗️  Building Python package...")
        if args.clean:
            run_command(["rm", "-rf", "dist", "build", "*.egg-info"])
        exit_code = run_command(["python", "-m", "build"])
        sys.exit(exit_code)

    # Rust
    if (root / "Cargo.toml").exists():
        print("🏗️  Building Rust project...")
        if args.clean:
            run_command(["cargo", "clean"])
        cmd = ["cargo", "build"]
        if args.release:
            cmd.append("--release")
        exit_code = run_command(cmd)
        sys.exit(exit_code)

    # Go
    if (root / "go.mod").exists():
        print("🏗️  Building Go project...")
        if args.clean:
            run_command(["go", "clean"])
        exit_code = run_command(["go", "build", "./..."])
        sys.exit(exit_code)

    print("❌ No build configuration found")
    sys.exit(1)


if __name__ == "__main__":
    main()
