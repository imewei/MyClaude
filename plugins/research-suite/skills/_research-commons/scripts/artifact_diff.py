#!/usr/bin/env python3
"""
artifact_diff.py

Compare two versions of an artifact (e.g., 03_claim.md vs 03_claim.v1.md) and
produce a readable summary of what changed. Useful when Stage 8 sends the
pipeline back to Stage 3 and you want to see exactly what revised.

Uses Python's difflib to produce a unified diff, then summarizes by section
(markdown heading).

Usage:
    python artifact_diff.py 03_claim.v1.md 03_claim.md
    python artifact_diff.py 03_claim.v1.md 03_claim.md --summary-only
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return a list of (heading, body) tuples. Text before first heading gets '(preamble)'."""
    sections = []
    current_heading = "(preamble)"
    current_body: list[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if current_body or current_heading != "(preamble)":
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_heading != "(preamble)":
        sections.append((current_heading, "\n".join(current_body)))
    return sections


def section_summary(old_text: str, new_text: str) -> list[dict]:
    """Per-section summary: added / removed / modified heading lists."""
    old_sections = dict(split_into_sections(old_text))
    new_sections = dict(split_into_sections(new_text))

    all_headings = sorted(set(old_sections) | set(new_sections))
    summary = []
    for h in all_headings:
        old_body = old_sections.get(h)
        new_body = new_sections.get(h)
        if old_body is None:
            summary.append({"heading": h, "status": "added", "size_delta": len(new_body or "")})
        elif new_body is None:
            summary.append({"heading": h, "status": "removed", "size_delta": -len(old_body)})
        elif old_body != new_body:
            summary.append({
                "heading": h,
                "status": "modified",
                "size_delta": len(new_body) - len(old_body),
            })
        else:
            summary.append({"heading": h, "status": "unchanged", "size_delta": 0})
    return summary


def print_summary(summary: list[dict]) -> None:
    print("Section-level summary:")
    print("-" * 60)
    for item in summary:
        status = item["status"]
        h = item["heading"]
        delta = item["size_delta"]
        delta_str = f"{delta:+d} chars" if status != "unchanged" else ""
        marker = {"added": "+", "removed": "-", "modified": "~", "unchanged": " "}[status]
        print(f"  {marker} [{status:>9}] {h}  {delta_str}")
    print()
    modified_count = sum(1 for i in summary if i["status"] != "unchanged")
    total = len(summary)
    print(f"{modified_count}/{total} sections differ")


def print_unified_diff(old_text: str, new_text: str, old_name: str, new_name: str) -> None:
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=old_name,
        tofile=new_name,
        n=3,
    )
    sys.stdout.writelines(diff)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two versions of an artifact")
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    for p in (args.old, args.new):
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            return 2

    old_text = args.old.read_text(encoding="utf-8")
    new_text = args.new.read_text(encoding="utf-8")

    summary = section_summary(old_text, new_text)
    print_summary(summary)

    if not args.summary_only:
        print("\n" + "=" * 60)
        print("Unified diff:")
        print("=" * 60)
        print_unified_diff(old_text, new_text, str(args.old), str(args.new))

    return 0


if __name__ == "__main__":
    sys.exit(main())
