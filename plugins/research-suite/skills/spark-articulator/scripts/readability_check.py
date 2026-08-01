#!/usr/bin/env python3
"""
readability_check.py

Flags potential readability issues in a 01_spark.md articulation:
- Sentences over 30 words
- Jargon-dense regions (>3 field-specific terms per sentence without redefinition)
- High passive-voice concentration (>25% of sentences)

Usage:
    python readability_check.py path/to/01_spark.md

This is a heuristic tool; its output prompts human review rather than gating.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Sentence boundary: period/exclaim/question followed by space/newline.
SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Passive-voice heuristic: auxiliary + past participle. Imprecise but useful as a smell.
PASSIVE_RE = re.compile(
    r"\b(am|is|are|was|were|be|being|been)\s+([a-z]+ed|[a-z]+en)\b",
    re.IGNORECASE,
)

# Common soft-matter / scattering / SciML jargon. Extend as needed.
# The point is not to flag all jargon (impossible); it's to surface clusters.
JARGON_TERMS = {
    "eigenvalue", "operator", "spectral", "tensor", "correlation",
    "stochastic", "fokker-planck", "langevin", "rheological", "oscillatory",
    "xpcs", "saxs", "dma", "laos", "heterodyne", "homodyne",
    "colloidal", "suspension", "flocculation", "vitrimer", "nanocomposite",
    "pinn", "ude", "graybox", "autodiff", "jax",
    "thermodynamic", "non-equilibrium", "vitrification", "glassy",
}


def extract_prose(text: str) -> str:
    """Strip markdown headers, code blocks, and metadata lines."""
    lines = text.splitlines()
    out = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("-") and len(stripped) < 80:
            # short list item, likely metadata
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            continue
        out.append(line)
    return " ".join(out)


def split_sentences(prose: str) -> list[str]:
    sents = SENT_RE.split(prose)
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def count_jargon(s: str) -> list[str]:
    lower = s.lower()
    hits = []
    for term in JARGON_TERMS:
        if term in lower:
            hits.append(term)
    return hits


def is_passive(s: str) -> bool:
    return bool(PASSIVE_RE.search(s))


def analyze(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    prose = extract_prose(text)
    sentences = split_sentences(prose)

    if not sentences:
        print(f"{path}: no prose sentences found")
        return 0

    long_sents = []
    jargon_heavy = []
    passive_count = 0

    for i, s in enumerate(sentences, start=1):
        wc = word_count(s)
        if wc > 30:
            long_sents.append((i, wc, s))
        jargon = count_jargon(s)
        if len(jargon) > 3:
            jargon_heavy.append((i, jargon, s))
        if is_passive(s):
            passive_count += 1

    passive_frac = passive_count / len(sentences)

    issues = 0

    if long_sents:
        issues += len(long_sents)
        print(f"\n{path}: {len(long_sents)} sentence(s) over 30 words:")
        for i, wc, s in long_sents:
            snippet = s[:120] + ("..." if len(s) > 120 else "")
            print(f"  sentence {i} ({wc} words): {snippet}")

    if jargon_heavy:
        issues += len(jargon_heavy)
        print(f"\n{path}: {len(jargon_heavy)} sentence(s) with >3 jargon terms:")
        for i, terms, s in jargon_heavy:
            snippet = s[:120] + ("..." if len(s) > 120 else "")
            print(f"  sentence {i} [{', '.join(terms)}]: {snippet}")

    if passive_frac > 0.25:
        issues += 1
        print(
            f"\n{path}: {passive_frac:.0%} passive-voice rate "
            f"({passive_count}/{len(sentences)}); aim for under 25%"
        )

    if issues == 0:
        print(f"{path}: no readability issues detected across {len(sentences)} sentences")

    return 0  # advisory tool; never gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Check readability of a spark articulation")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if not args.path.exists():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2
    return analyze(args.path)


if __name__ == "__main__":
    sys.exit(main())
