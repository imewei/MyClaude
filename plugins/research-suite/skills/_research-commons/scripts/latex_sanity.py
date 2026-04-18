#!/usr/bin/env python3
"""
latex_sanity.py

Comprehensive sanity check for LaTeX documents. Complements theory-scaffold's
latex_compile_check.sh (which does a full pdflatex compile) by running static
checks that do not require a TeX distribution.

Checks performed:
  1. Undefined references: every \\ref{...} / \\cref{...} / \\eqref{...} points
     to a label that exists in the document.
  2. Unused labels: every \\label{...} is referenced at least once.
  3. Orphan citations: every \\cite{...} key appears in the .bib file (if one
     is provided).
  4. Duplicate labels: no label is defined twice.
  5. Bracket balance: \\begin/\\end pairs match; math mode delimiters balance.
  6. Common typos in standard commands.

When pdflatex is available, this script can also invoke it (--compile flag)
and merge its output with the static findings.

Usage:
    python latex_sanity.py paper.tex
    python latex_sanity.py paper.tex --bib refs.bib
    python latex_sanity.py paper.tex --bib refs.bib --compile
    python latex_sanity.py paper.tex --strict   # nonzero exit on any finding

Exit codes:
    0: clean (or findings but not --strict)
    1: findings in --strict mode
    2: usage error
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from collections import Counter


LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|cref|Cref|eqref|autoref|pageref)\*?\{([^}]+)\}")
CITE_RE = re.compile(r"\\(?:cite|citep|citet|citealt|citealp|citeauthor|citeyear|parencite|textcite)\*?(?:\[[^\]]*\])?\{([^}]+)\}")
BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")
BIB_KEY_RE = re.compile(r"@\w+\{([^,\n]+),")
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
INCLUDE_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def strip_comments(text: str) -> str:
    """Remove LaTeX line comments, preserving escaped percents."""
    return COMMENT_RE.sub("", text)


def load_tex_with_includes(path: Path, seen: set[Path] | None = None) -> str:
    """Recursively inline \\input and \\include commands so checks cover
    multi-file documents."""
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        return ""  # avoid include cycles
    seen.add(path)

    if not path.exists():
        # Try adding .tex extension
        if path.suffix != ".tex" and path.with_suffix(".tex").exists():
            path = path.with_suffix(".tex")
        else:
            return ""

    text = path.read_text(encoding="utf-8")
    text = strip_comments(text)
    base = path.parent

    def replace_include(match: re.Match) -> str:
        fname = match.group(1).strip()
        return load_tex_with_includes(base / fname, seen)

    return INCLUDE_RE.sub(replace_include, text)


def check_references(text: str) -> dict:
    """Return dict with undefined refs, unused labels, duplicate labels."""
    labels = [m.group(1) for m in LABEL_RE.finditer(text)]
    refs = [m.group(1) for m in REF_RE.finditer(text)]

    # Refs can be comma-separated
    ref_set: set[str] = set()
    for r in refs:
        ref_set.update(s.strip() for s in r.split(","))

    label_counts = Counter(labels)
    label_set = set(labels)

    undefined = sorted(ref_set - label_set)
    unused = sorted(label_set - ref_set)
    duplicates = sorted(k for k, v in label_counts.items() if v > 1)

    return {
        "undefined": undefined,
        "unused": unused,
        "duplicates": duplicates,
        "n_labels": len(label_set),
        "n_refs": len(ref_set),
    }


def check_citations(text: str, bib_path: Path | None) -> dict:
    """Return dict with orphan citations (cited but not in bib), unused entries."""
    cited_raw = [m.group(1) for m in CITE_RE.finditer(text)]
    # Citations can be comma-separated in a single \cite{a,b,c}
    cite_set: set[str] = set()
    for c in cited_raw:
        cite_set.update(s.strip() for s in c.split(","))

    if bib_path is None or not bib_path.exists():
        return {
            "orphan": [],
            "unused_entries": [],
            "n_cited": len(cite_set),
            "n_bib": 0,
            "bib_checked": False,
        }

    bib_text = bib_path.read_text(encoding="utf-8")
    bib_keys = {m.group(1).strip() for m in BIB_KEY_RE.finditer(bib_text)}

    orphan = sorted(cite_set - bib_keys)
    unused_entries = sorted(bib_keys - cite_set)

    return {
        "orphan": orphan,
        "unused_entries": unused_entries,
        "n_cited": len(cite_set),
        "n_bib": len(bib_keys),
        "bib_checked": True,
    }


def check_environments(text: str) -> dict:
    """Check that \\begin and \\end pairs match."""
    begins = [m.group(1) for m in BEGIN_RE.finditer(text)]
    ends = [m.group(1) for m in END_RE.finditer(text)]

    begin_counts = Counter(begins)
    end_counts = Counter(ends)

    mismatched: list[tuple[str, int, int]] = []
    envs = set(begin_counts) | set(end_counts)
    for env in sorted(envs):
        b = begin_counts.get(env, 0)
        e = end_counts.get(env, 0)
        if b != e:
            mismatched.append((env, b, e))

    return {"mismatched": mismatched}


def check_math_delimiters(text: str) -> dict:
    """Rough check of math-mode balance. Not perfect but catches common cases."""
    # Count $$ first, then single $ (excluding those in $$)
    double_count = text.count("$$")
    if double_count % 2 != 0:
        return {"unbalanced_double_dollar": True, "unbalanced_single_dollar": False}

    # Remove $$ blocks, then count $
    clean = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    single_count = clean.count("$") - clean.count(r"\$")  # exclude escaped

    return {
        "unbalanced_double_dollar": False,
        "unbalanced_single_dollar": single_count % 2 != 0,
    }


def check_common_typos(text: str) -> list[tuple[int, str]]:
    """Return a list of (line_number, message) for common LaTeX typos."""
    findings = []
    lines = text.splitlines()
    patterns = [
        (re.compile(r"\\beign\{"), "typo '\\beign{' for '\\begin{'"),
        (re.compile(r"\\endd?\{[^}]+\}\{"), "possible doubled brace after \\end"),
        (re.compile(r"\\label\{[^}]*\s[^}]*\}"), "whitespace inside \\label argument"),
        (re.compile(r"\\ref\{[^}]*\s[^}]*\}"), "whitespace inside \\ref argument"),
        (re.compile(r"\\\\[ \t]*\\\\"), "consecutive \\\\ (line breaks) likely a typo"),
        (re.compile(r"\bi\.e\.[^,]"), "'i.e.' usually followed by a comma"),
        (re.compile(r"\be\.g\.[^,]"), "'e.g.' usually followed by a comma"),
    ]
    for i, line in enumerate(lines, start=1):
        for pat, msg in patterns:
            if pat.search(line):
                findings.append((i, msg))
    return findings


def run_pdflatex(tex_path: Path) -> tuple[bool, str]:
    """Invoke pdflatex twice and return (succeeded, log tail)."""
    import shutil
    if not shutil.which("pdflatex"):
        return (False, "pdflatex not on PATH; skipping compilation")

    workdir = tex_path.parent
    base = tex_path.stem
    for _pass in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return (False, result.stdout[-2000:] if result.stdout else result.stderr[-2000:])

    log_path = workdir / f"{base}.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        warn_count = log.count("Warning")
        undef_count = log.count("undefined")
        return (True, f"compile ok ({warn_count} warnings, {undef_count} undefined-like entries in log)")
    return (True, "compile ok")


def format_report(
    tex_path: Path,
    refs: dict,
    cites: dict,
    envs: dict,
    math: dict,
    typos: list,
    compile_result: tuple[bool, str] | None,
) -> tuple[str, int]:
    """Return (report_text, total_findings)."""
    lines = [f"=== latex_sanity report for {tex_path} ==="]
    total = 0

    lines.append(f"\nLabels defined: {refs['n_labels']}; references used: {refs['n_refs']}")
    if refs["undefined"]:
        total += len(refs["undefined"])
        lines.append(f"\n[FAIL] Undefined references ({len(refs['undefined'])}):")
        lines.extend(f"  - {r}" for r in refs["undefined"])
    if refs["duplicates"]:
        total += len(refs["duplicates"])
        lines.append(f"\n[FAIL] Duplicate labels ({len(refs['duplicates'])}):")
        lines.extend(f"  - {r}" for r in refs["duplicates"])
    if refs["unused"]:
        lines.append(f"\n[warn] Unused labels ({len(refs['unused'])}):")
        lines.extend(f"  - {r}" for r in refs["unused"])

    if cites["bib_checked"]:
        lines.append(
            f"\nCitations used: {cites['n_cited']}; bib entries: {cites['n_bib']}"
        )
        if cites["orphan"]:
            total += len(cites["orphan"])
            lines.append(f"\n[FAIL] Orphan citations (cited but not in bib) ({len(cites['orphan'])}):")
            lines.extend(f"  - {c}" for c in cites["orphan"])
        if cites["unused_entries"]:
            lines.append(
                f"\n[warn] Bib entries never cited ({len(cites['unused_entries'])}):"
            )
            lines.extend(f"  - {c}" for c in cites["unused_entries"][:20])
            if len(cites["unused_entries"]) > 20:
                lines.append(f"  ... {len(cites['unused_entries']) - 20} more")
    else:
        lines.append(f"\nCitations used: {cites['n_cited']}; no .bib file checked")

    if envs["mismatched"]:
        total += len(envs["mismatched"])
        lines.append("\n[FAIL] Environment mismatches:")
        for name, b, e in envs["mismatched"]:
            lines.append(f"  - {name}: {b} begin, {e} end")

    if math["unbalanced_double_dollar"]:
        total += 1
        lines.append("\n[FAIL] Unbalanced $$ ... $$ math delimiters")
    if math["unbalanced_single_dollar"]:
        total += 1
        lines.append("\n[FAIL] Unbalanced $ ... $ math delimiters")

    if typos:
        total += len(typos)
        lines.append(f"\n[warn] Common typos and style issues ({len(typos)}):")
        for lineno, msg in typos[:20]:
            lines.append(f"  line {lineno}: {msg}")
        if len(typos) > 20:
            lines.append(f"  ... {len(typos) - 20} more")

    if compile_result is not None:
        ok, info = compile_result
        status = "ok" if ok else "FAIL"
        lines.append(f"\n[compile:{status}] {info}")
        if not ok:
            total += 1

    if total == 0:
        lines.append("\n[PASS] no blocking findings")

    return ("\n".join(lines) + "\n", total)


def main() -> int:
    parser = argparse.ArgumentParser(description="LaTeX sanity checker")
    parser.add_argument("tex", type=Path)
    parser.add_argument("--bib", type=Path, default=None, help="accompanying .bib file")
    parser.add_argument("--compile", action="store_true", help="also invoke pdflatex")
    parser.add_argument("--strict", action="store_true", help="nonzero exit on any finding")
    args = parser.parse_args()

    if not args.tex.exists():
        print(f"error: {args.tex} not found", file=sys.stderr)
        return 2

    text = load_tex_with_includes(args.tex)
    refs = check_references(text)
    cites = check_citations(text, args.bib)
    envs = check_environments(text)
    math = check_math_delimiters(text)
    typos = check_common_typos(text)
    compile_result = run_pdflatex(args.tex) if args.compile else None

    report, total = format_report(args.tex, refs, cites, envs, math, typos, compile_result)
    print(report)

    return 1 if (args.strict and total > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
