#!/usr/bin/env python3
"""
Run linting and formatting checks across different languages.

Usage:
    python lint_check.py [--fix]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, check: bool = False) -> int:
    """Run a command and return exit code."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return e.returncode


import concurrent.futures

def run_js_checks(root, args):
    """Run JavaScript/TypeScript checks."""
    exit_code = 0
    if (root / "package.json").exists():
        print("📋 Checking JavaScript/TypeScript...")
        if (root / "node_modules" / ".bin" / "eslint").exists():
            cmd = ["npx", "eslint", "."]
            if args.fix: cmd.append("--fix")
            exit_code |= run_command(cmd)

        if (root / "node_modules" / ".bin" / "prettier").exists():
            cmd = ["npx", "prettier", "--check" if not args.fix else "--write", "."]
            exit_code |= run_command(cmd)
    return exit_code

def run_python_checks(root, args):
    """Run Python checks."""
    exit_code = 0
    if (root / "pyproject.toml").exists() or any(root.glob("*.py")):
        print("📋 Checking Python...")
        try:
            cmd = ["ruff", "check", "."]
            if args.fix: cmd.append("--fix")
            exit_code |= run_command(cmd)

            cmd = ["ruff", "format" if args.fix else "format --check", "."]
            exit_code |= run_command(cmd)
        except FileNotFoundError:
            try:
                cmd = ["black", "." if args.fix else "--check ."]
                exit_code |= run_command(cmd)
            except FileNotFoundError:
                print("⚠️  No Python formatter found")
    return exit_code

def run_rust_checks(root, args):
    """Run Rust checks."""
    exit_code = 0
    if (root / "Cargo.toml").exists():
        print("📋 Checking Rust...")
        exit_code |= run_command(["cargo", "clippy"])
        exit_code |= run_command(["cargo", "fmt", "--" if not args.fix else "", "--check" if not args.fix else ""])
    return exit_code

def run_go_checks(root, args):
    """Run Go checks."""
    exit_code = 0
    if (root / "go.mod").exists():
        print("📋 Checking Go...")
        exit_code |= run_command(["gofmt", "-l" if not args.fix else "-w", "."])
    return exit_code

def main():
    parser = argparse.ArgumentParser(description="Run linting and formatting checks (Parallel)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")
    args = parser.parse_args()

    root = Path.cwd()
    exit_code = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_js_checks, root, args),
            executor.submit(run_python_checks, root, args),
            executor.submit(run_rust_checks, root, args),
            executor.submit(run_go_checks, root, args)
        ]

        for future in concurrent.futures.as_completed(futures):
            exit_code |= future.result()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
