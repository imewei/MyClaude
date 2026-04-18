#!/usr/bin/env python3
"""
dimensional_audit.py

Extracts labeled equations from a LaTeX file and reports their locations so the
human (or a later automated check) can verify dimensional consistency.

This script does NOT perform automated dimensional checking (that would require
a symbol-to-dimensions mapping per document, which is problem-specific). It
provides the scaffold: an extracted list of equations with their labels, ready
to be annotated.

Usage:
    python dimensional_audit.py path/to/05_formalism.tex
    python dimensional_audit.py path/to/05_formalism.tex --annotate
        # writes <input>.audit.md with a checklist

Future extension: a YAML sidecar file (05_formalism.dims.yaml) that maps
symbol -> dimensions, which would let the script check automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches labeled display-math environments: equation, align, gather, multline.
# Captures the environment name, optional label, and the body.
DISPLAY_MATH_RE = re.compile(
    r"\\begin\{(equation|align|gather|multline)\*?\}"
    r"(.*?)"
    r"\\end\{\1\*?\}",
    re.DOTALL,
)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def extract_equations(text: str) -> list[dict]:
    """Return a list of dicts with env, label, body, line_start."""
    results = []
    for match in DISPLAY_MATH_RE.finditer(text):
        env = match.group(1)
        body = match.group(2).strip()
        label_match = LABEL_RE.search(body)
        label = label_match.group(1) if label_match else None
        line_start = text.count("\n", 0, match.start()) + 1
        results.append({
            "env": env,
            "label": label,
            "body": body,
            "line_start": line_start,
        })
    return results


def format_annotation(eqs: list[dict], source_path: Path) -> str:
    lines = [
        f"# Dimensional audit for {source_path.name}",
        "",
        "Fill in the units column for each equation. Verify that both sides have the",
        "same units. Flag any equation where the units do not match.",
        "",
        "| # | Label | Line | LHS units | RHS units | OK? |",
        "|---|-------|------|-----------|-----------|-----|",
    ]
    for i, eq in enumerate(eqs, start=1):
        label = eq["label"] or "(unlabeled)"
        lines.append(f"| {i} | `{label}` | {eq['line_start']} | | | |")
    lines.append("")
    lines.append("## Equations extracted")
    lines.append("")
    for i, eq in enumerate(eqs, start=1):
        label = eq["label"] or "(unlabeled)"
        body = eq["body"]
        lines.append(f"### {i}. `{label}` (line {eq['line_start']}, env: {eq['env']})")
        lines.append("```latex")
        lines.append(body)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract equations for dimensional audit.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--annotate", action="store_true", help="write <input>.audit.md")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2

    text = args.path.read_text(encoding="utf-8")
    eqs = extract_equations(text)

    if not eqs:
        print(f"{args.path}: no labeled display-math equations found")
        return 0

    print(f"{args.path}: {len(eqs)} equations found")
    unlabeled = sum(1 for e in eqs if e["label"] is None)
    if unlabeled:
        print(f"  warning: {unlabeled} equation(s) without \\label (cannot be \\cref'd)")
    for eq in eqs:
        label = eq["label"] or "(unlabeled)"
        print(f"  line {eq['line_start']:>4}: [{eq['env']}] {label}")

    if args.annotate:
        out = args.path.with_suffix(args.path.suffix + ".audit.md")
        out.write_text(format_annotation(eqs, args.path), encoding="utf-8")
        print(f"\nAudit checklist written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
