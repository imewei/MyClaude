#!/usr/bin/env python3
"""
concept_extractor.py

Extract candidate key concepts from a Stage 1 spark articulation to seed Stage
2 literature queries. Heuristic: pull capitalized multi-word phrases, technical
noun phrases, and named-entity-like tokens. The user reviews and edits the
list before Stage 2 runs its searches.

This is not a production NLP pipeline. It is a useful starting list so the user
is not staring at a blank search box.

Usage:
    python concept_extractor.py 01_spark.md
    python concept_extractor.py 01_spark.md --top 20
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


# Matches compound technical terms: 2+ words, each starting with a capital,
# or mid-word hyphens/underscores signaling a technical name.
CAPITALIZED_PHRASE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+){1,4})\b"
)

# Matches tokens with mid-word hyphens (e.g., "bond-exchange", "stress-response")
HYPHENATED_RE = re.compile(r"\b([a-z]+-[a-z]+(?:-[a-z]+){0,3})\b")

# Matches acronyms (2-6 uppercase letters)
ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")

# Matches noun phrases of the form "X of Y" or "X for Y"
OF_PHRASE_RE = re.compile(
    r"\b([a-z]+(?:\s[a-z]+)?)\s+(?:of|for)\s+([a-z]+(?:\s[a-z]+)?)\b"
)

# Stop words to filter from the noun-phrase extraction
STOP = {
    "the", "a", "an", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "and", "or", "but", "not", "we", "our",
    "it", "its", "as", "in", "on", "at", "by", "to", "from", "with", "for",
    "of", "about", "very", "much", "many", "some", "any", "all", "each",
    "more", "most", "other", "new", "use", "used", "using", "can", "could",
    "would", "should", "will", "may", "might", "often", "usually",
}


def extract_prose(text: str) -> str:
    """Remove YAML frontmatter, code blocks, and markdown headers."""
    # Strip YAML frontmatter
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # Strip code fences
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Strip inline code
    text = re.sub(r"`[^`]*`", "", text)
    # Strip headers markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Strip bold/italic markers
    text = re.sub(r"[*_]{1,3}", "", text)
    # Strip bracketed placeholders
    text = re.sub(r"\[[^\]]*\]", "", text)
    return text


def extract_candidates(text: str) -> Counter:
    """Return a Counter mapping candidate concepts to frequency."""
    counter: Counter = Counter()

    # Capitalized compound phrases
    for m in CAPITALIZED_PHRASE_RE.finditer(text):
        phrase = m.group(1)
        counter[phrase] += 2  # weight these higher, usually proper names

    # Hyphenated technical terms
    for m in HYPHENATED_RE.finditer(text):
        term = m.group(1)
        if not any(s in term.split("-") for s in STOP):
            counter[term] += 1

    # Acronyms (skip common English ones)
    for m in ACRONYM_RE.finditer(text):
        acronym = m.group(1)
        if acronym not in {"THE", "AND", "FOR", "WITH", "FROM", "NOT", "ARE"}:
            counter[acronym] += 2

    # "X of Y" and "X for Y" noun phrases
    for m in OF_PHRASE_RE.finditer(text):
        x, y = m.group(1).strip(), m.group(2).strip()
        if not any(w in STOP for w in x.split()) and not any(w in STOP for w in y.split()):
            counter[f"{x} of {y}" if m.group(0).split()[1] == "of" else f"{x} for {y}"] += 1

    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract candidate concepts from a spark articulation")
    parser.add_argument("path", type=Path)
    parser.add_argument("--top", type=int, default=15, help="how many candidates to show")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2

    text = args.path.read_text(encoding="utf-8")
    prose = extract_prose(text)
    counter = extract_candidates(prose)

    if not counter:
        print(f"{args.path}: no candidate concepts extracted")
        return 0

    print(f"Top {args.top} candidate concepts from {args.path}:")
    print("(review and edit; not all will be useful as literature queries)")
    print()
    for concept, freq in counter.most_common(args.top):
        print(f"  [{freq:2d}] {concept}")
    print()
    print("Suggested next steps for Stage 2:")
    print("  - pick 3-5 concepts most central to the spark")
    print("  - build Layer 2 (recent) queries from those")
    print("  - ask the user for Layer 1 (foundational) references directly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
