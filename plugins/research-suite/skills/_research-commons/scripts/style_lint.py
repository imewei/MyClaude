#!/usr/bin/env python3
"""
style_lint.py

Flags em dashes and banned vocabulary in markdown or LaTeX files, per the
research-spark stack's writing_constraints.md.

Usage:
    python style_lint.py path/to/file.md
    python style_lint.py path/to/dir/         # recursively lint all .md/.tex files
    python style_lint.py --strict path/to/file.md   # nonzero exit on any finding

Exit codes:
    0: no issues, or issues found in non-strict mode
    1: issues found in strict mode
    2: usage error

The linter prints file:line:col style messages that most editors can parse.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BANNED_VOCAB = [
    "innovative",
    "state-of-the-art",
    "transformative",
    "sustainable",
    "novel",
    "groundbreaking",
    "cutting-edge",
    "revolutionary",
    "paradigm-shifting",
    "seamless",
]

# Em dash characters. We exclude en dashes (—) vs (–) explicitly.
EM_DASH_PATTERN = re.compile(r"\u2014")  # — only

# Word-boundary match for banned vocab. Case-insensitive.
def _banned_pattern(words: list[str]) -> re.Pattern[str]:
    # handle hyphens as part of words; compile once
    escaped = [re.escape(w) for w in words]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


BANNED_PATTERN = _banned_pattern(BANNED_VOCAB)

# File extensions we lint
LINT_EXTENSIONS = {".md", ".tex", ".txt", ".rst"}


def lint_text(text: str, path: str) -> list[str]:
    """Return a list of issue lines for the given text."""
    issues: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        # em dash check
        for m in EM_DASH_PATTERN.finditer(line):
            col = m.start() + 1
            issues.append(
                f"{path}:{lineno}:{col}: em dash found; use commas, colons, "
                "semicolons, parens, or restructure"
            )
        # banned vocabulary check
        for m in BANNED_PATTERN.finditer(line):
            col = m.start() + 1
            word = m.group(0)
            issues.append(
                f"{path}:{lineno}:{col}: banned vocabulary '{word}'; "
                "quantify instead of using marketing adjectives"
            )
    return issues


IGNORE_MARKER = "style_lint:ignore-file"


def lint_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}: could not read as UTF-8; skipped"]
    # Files can opt out by including the marker anywhere (usually in an HTML comment).
    if IGNORE_MARKER in text:
        return []
    return lint_text(text, str(path))


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files: list[Path] = []
        for ext in LINT_EXTENSIONS:
            files.extend(target.rglob(f"*{ext}"))
        return sorted(files)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint markdown/LaTeX for em dashes and banned vocab.")
    parser.add_argument("path", type=Path, help="file or directory to lint")
    parser.add_argument("--strict", action="store_true", help="exit nonzero on any finding")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2

    files = collect_files(args.path)
    if not files:
        print(f"no lintable files found in {args.path}", file=sys.stderr)
        return 0

    all_issues: list[str] = []
    for f in files:
        all_issues.extend(lint_file(f))

    for issue in all_issues:
        print(issue)

    if all_issues:
        print(f"\n{len(all_issues)} issue(s) across {len(files)} file(s)", file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
