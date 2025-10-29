#!/usr/bin/env python3
"""
Accessibility testing for web applications.

Runs axe-core and pa11y accessibility tests.

Usage:
    python accessibility_check.py <url-or-html-file> [--wcag-level AA]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_with_pa11y(target: str, wcag_level: str = "AA"):
    """Run pa11y accessibility check."""
    print(f"♿ Running pa11y accessibility check (WCAG {wcag_level})...")

    cmd = [
        "pa11y",
        target,
        f"--standard=WCAG2{wcag_level}",
        "--reporter", "cli"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"\n❌ Found accessibility issues (WCAG {wcag_level})")
        return 1

    print(f"\n✅ No accessibility issues found (WCAG {wcag_level})")
    return 0


def check_with_axe(target: str):
    """Run axe-core accessibility check."""
    print(f"♿ Running axe-core accessibility check...")

    # This requires axe-cli
    cmd = ["axe", target, "--verbose"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("\n❌ Found accessibility issues")
        return 1

    print("\n✅ No accessibility issues found")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run accessibility testing")
    parser.add_argument("target", help="URL or HTML file to test")
    parser.add_argument("--wcag-level", choices=["A", "AA", "AAA"], default="AA",
                        help="WCAG conformance level")
    parser.add_argument("--tool", choices=["pa11y", "axe", "both"], default="both",
                        help="Tool to use for testing")

    args = parser.parse_args()

    exit_code = 0

    if args.tool in ["pa11y", "both"]:
        exit_code |= check_with_pa11y(args.target, args.wcag_level)

    if args.tool in ["axe", "both"]:
        exit_code |= check_with_axe(args.target)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
