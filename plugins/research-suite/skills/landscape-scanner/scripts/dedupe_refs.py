#!/usr/bin/env python3
"""
dedupe_refs.py

Deduplicate a .bib file by DOI. Entries without DOIs are kept and compared by
a fuzzy title+year+first-author key.

Usage:
    python dedupe_refs.py input.bib output.bib
    python dedupe_refs.py input.bib -            # write to stdout

Reports duplicates to stderr so the user can see what was merged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENTRY_RE = re.compile(r"@\w+\{[^@]*?\n\}", re.DOTALL)
DOI_RE = re.compile(r"doi\s*=\s*\{([^}]+)\}", re.IGNORECASE)
TITLE_RE = re.compile(r"title\s*=\s*\{([^}]+)\}", re.IGNORECASE)
YEAR_RE = re.compile(r"year\s*=\s*\{?(\d{4})\}?", re.IGNORECASE)
AUTHOR_RE = re.compile(r"author\s*=\s*\{([^}]+)\}", re.IGNORECASE)


def normalize_title(t: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def first_author_surname(authors: str) -> str:
    """Extract first author's surname. Handles 'Surname, Given' and 'Given Surname'."""
    first = authors.split(" and ")[0].strip()
    if "," in first:
        return first.split(",")[0].strip().lower()
    parts = first.split()
    return parts[-1].lower() if parts else ""


def key_for_entry(entry: str) -> tuple[str, str]:
    """
    Return (kind, key) where kind is 'doi' or 'fuzzy'.
    """
    doi_m = DOI_RE.search(entry)
    if doi_m:
        return ("doi", doi_m.group(1).strip().lower())
    title_m = TITLE_RE.search(entry)
    year_m = YEAR_RE.search(entry)
    author_m = AUTHOR_RE.search(entry)
    if title_m and year_m and author_m:
        key = (
            first_author_surname(author_m.group(1))
            + "|"
            + year_m.group(1)
            + "|"
            + normalize_title(title_m.group(1))[:50]
        )
        return ("fuzzy", key)
    # Fallback: use the entry's cite-key (e.g., @article{SomeKey,})
    m = re.match(r"@\w+\{([^,\n]+),", entry)
    return ("citekey", m.group(1).strip().lower() if m else entry[:40])


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate .bib by DOI and fuzzy key")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=str, help="output path or - for stdout")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    text = args.input.read_text(encoding="utf-8")
    entries = ENTRY_RE.findall(text)

    seen: dict[tuple[str, str], str] = {}
    duplicates: list[tuple[str, str]] = []
    kept: list[str] = []

    for entry in entries:
        k = key_for_entry(entry)
        if k in seen:
            duplicates.append((k[1], entry[:60].replace("\n", " ")))
        else:
            seen[k] = entry
            kept.append(entry)

    output_text = "\n\n".join(kept) + "\n"

    if args.output == "-":
        sys.stdout.write(output_text)
    else:
        Path(args.output).write_text(output_text, encoding="utf-8")

    print(
        f"Read {len(entries)} entries, kept {len(kept)}, "
        f"dropped {len(duplicates)} duplicate(s)",
        file=sys.stderr,
    )
    for key, snippet in duplicates:
        print(f"  duplicate of {key}: {snippet}...", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
