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

ENTRY_START_RE = re.compile(r"^[ \t]*@(\w+)\s*\{", re.MULTILINE)
ENTRY_ANYWHERE_RE = re.compile(r"@\w+\s*\{")
DOI_RE = re.compile(r"doi\s*=\s*\{([^}]+)\}", re.IGNORECASE)
TITLE_RE = re.compile(r"title\s*=\s*\{([^}]+)\}", re.IGNORECASE)
YEAR_RE = re.compile(r"year\s*=\s*\{?(\d{4})\}?", re.IGNORECASE)
AUTHOR_RE = re.compile(r"author\s*=\s*\{([^}]+)\}", re.IGNORECASE)


def parse_entries(text: str) -> tuple[list[str], list[str]]:
    """Split a .bib into raw entry slices by brace counting.

    A regex cannot do this: `@` inside a note field truncates `[^@]*`, and a
    trailing `}}` on the last field line defeats `\\n\\}`. Both silently drop
    entries. Returns (entries, errors); a non-empty errors list means the file
    was not fully understood and the caller must not write output.
    """
    entries: list[str] = []
    errors: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in ENTRY_START_RE.finditer(text):
        start = match.start()
        i = match.end() - 1  # position of the opening brace
        depth = 0
        end = None
        while i < len(text):
            ch = text[i]
            if ch == "\\":
                i += 2  # escaped character; braces here are literal
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        if end is None:
            snippet = text[start : start + 60].replace("\n", " ")
            errors.append(f"unterminated entry at offset {start}: {snippet}...")
            continue
        entries.append(text[start:end])
        spans.append((start, end))

    # ENTRY_START_RE anchors at line start; an entry opened mid-line would be
    # invisible to both the parser and a guard sharing its regex. Sweep
    # unanchored so that blind spot cannot become another silent drop.
    for match in ENTRY_ANYWHERE_RE.finditer(text):
        if not any(s <= match.start() < e for s, e in spans):
            errors.append(f"entry start not at line start, offset {match.start()}")

    return entries, errors


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
    entries, errors = parse_entries(text)

    # Never write output from a parse we cannot account for: a partial write
    # destroys the bibliography and downstream latex_sanity.py then reports the
    # lost keys as fabricated orphan citations.
    n_starts = len(ENTRY_START_RE.findall(text))
    if errors or len(entries) != n_starts:
        print(
            f"error: parse not verified — {n_starts} entry start marker(s) in "
            f"{args.input}, {len(entries)} fully parsed. Output NOT written.",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

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
        f"Read {len(entries)}/{n_starts} entries (all accounted for), "
        f"kept {len(kept)}, dropped {len(duplicates)} duplicate(s)",
        file=sys.stderr,
    )
    for key, snippet in duplicates:
        print(f"  duplicate of {key}: {snippet}...", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
