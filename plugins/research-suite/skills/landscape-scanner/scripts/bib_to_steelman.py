#!/usr/bin/env python3
"""
bib_to_steelman.py

Convert entries in a .bib file into steelman-note stubs, one stub per entry,
with the three required fields (strongest claim, conditions under which it
breaks, residual uncertainty) left blank for the user to fill in.

Usage:
    python bib_to_steelman.py refs.bib steelman_notes.md
    python bib_to_steelman.py refs.bib - > notes.md      # stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENTRY_RE = re.compile(r"@(\w+)\{([^,\n]+),([^@]*?)\n\}", re.DOTALL)
FIELD_RE = re.compile(r"(\w+)\s*=\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", re.IGNORECASE)


def parse_entry(block: str) -> dict:
    """Return a dict of BibTeX fields."""
    m = ENTRY_RE.match(block)
    if not m:
        return {}
    kind, citekey, body = m.groups()
    out = {"_type": kind, "_citekey": citekey.strip()}
    for fm in FIELD_RE.finditer(body):
        key = fm.group(1).lower()
        value = fm.group(2).strip()
        # Collapse whitespace
        value = re.sub(r"\s+", " ", value)
        out[key] = value
    return out


def parse_bibfile(text: str) -> list[dict]:
    entries = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "@" and depth == 0 and (i == 0 or text[i-1] in "\n\r\t "):
            start = i
            depth = 0
        elif ch == "{" and start is not None:
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                block = text[start:i+1]
                parsed = parse_entry(block)
                if parsed:
                    entries.append(parsed)
                start = None
    return entries


def first_author_surname(authors: str) -> str:
    first = authors.split(" and ")[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    parts = first.split()
    return parts[-1] if parts else "Unknown"


def render_stub(entry: dict) -> str:
    authors = entry.get("author", "Unknown")
    year = entry.get("year", "n.d.")
    title = entry.get("title", "(no title)")
    doi = entry.get("doi", "")
    journal = entry.get("journal", entry.get("booktitle", ""))

    surname = first_author_surname(authors)
    n_authors = len(authors.split(" and "))
    author_short = surname if n_authors == 1 else f"{surname} et al."

    lines = [
        f"### {author_short}, {year}: {title}",
        "",
        f"**Citation key:** `{entry.get('_citekey', '?')}`  ",
        f"**DOI:** {doi or '_not provided_'}  ",
        f"**Journal/venue:** {journal or '_not provided_'}  ",
        "**Layer:** [foundational | recent | adjacent]  ",
        "**Read depth:** [abstract | introduction + conclusion | full read]",
        "",
        "**Strongest claim** (authors' own framing):",
        "> TODO. State the claim as the authors would, at its most ambitious.",
        "",
        "**Conditions under which it breaks:**",
        "> TODO. Specific regime, system, or parameter range. Must be specific.",
        "",
        "**Residual uncertainty:**",
        "> TODO. What the paper leaves unresolved that bears on our spark.",
        "",
        "**Relevance to our spark:**",
        "> TODO. One sentence linking this paper to a specific sub-question.",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .bib entries to steelman-note stubs")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=str, help="path or - for stdout")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    text = args.input.read_text(encoding="utf-8")
    entries = parse_bibfile(text)

    if not entries:
        print(f"warning: no entries parsed from {args.input}", file=sys.stderr)
        return 0

    header = [
        "# Steelman notes",
        "",
        f"_Generated from `{args.input.name}` by `bib_to_steelman.py`. "
        "Every TODO must be filled in before the Stage 2 artifact is considered complete._",
        "",
    ]
    body = [render_stub(e) for e in entries]
    output_text = "\n".join(header) + "\n".join(body)

    if args.output == "-":
        sys.stdout.write(output_text)
    else:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(
            f"Wrote {len(entries)} steelman stub(s) to {args.output}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
