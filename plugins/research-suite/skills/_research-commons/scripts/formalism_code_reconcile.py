#!/usr/bin/env python3
"""
formalism_code_reconcile.py

Compare symbols used in a LaTeX formalism (Stage 4-5 output) against symbols
defined in a Python implementation (Stage 6 output). Flags symbols present in
one but not the other.

This is heuristic. Not every LaTeX symbol becomes a Python identifier, and
vice versa; the script produces a list of candidates the human reviews, not
a pass/fail signal.

Usage:
    python formalism_code_reconcile.py 05_formalism.tex src/mypackage/
    python formalism_code_reconcile.py 05_formalism.tex src/mypackage/ --config map.yaml

Config file (optional) maps LaTeX symbol names to Python identifiers and
lists symbols to ignore on each side:

    latex_to_python:
      D_eff: diffusion_effective
      "\\phi": phi
    ignore_latex:
      - "\\mu"              # known to be represented differently in code
    ignore_python:
      - "_private_helper"
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]  # config is optional


# Extract non-trivial LaTeX symbols from $...$, \( \), \[ \], and equation environments.
# We pull out:
#   - commands like \phi, \eta, \nabla
#   - subscripted identifiers like D_eff, \tau_R
#   - multiletter command-like sequences
LATEX_MATH_BLOCK = re.compile(
    r"\$\$(.+?)\$\$|\$(.+?)\$|"
    r"\\\[(.+?)\\\]|\\\((.+?)\\\)|"
    r"\\begin\{(?:equation|align|gather|multline)\*?\}(.+?)\\end\{(?:equation|align|gather|multline)\*?\}",
    re.DOTALL,
)
LATEX_CMD = re.compile(r"\\([a-zA-Z]+)(?![a-zA-Z])")
LATEX_ID = re.compile(r"\b([A-Za-z][A-Za-z_]*)(?:_\{?([A-Za-z0-9]+)\}?)?")

# Standard LaTeX commands that are not physics symbols; ignore these.
LATEX_STDLIB = {
    "begin", "end", "frac", "int", "sum", "prod", "cdot", "times", "pm", "mp",
    "infty", "partial", "nabla", "left", "right", "mathbf", "mathcal", "mathrm",
    "text", "textbf", "textit", "emph", "label", "ref", "cref", "Cref",
    "cite", "citep", "citet", "eqref", "sqrt", "hat", "bar", "vec", "dot",
    "ddot", "tilde", "quad", "qquad", ",", ";", ":", "!", "langle", "rangle",
    "left(", "right)", "left[", "right]", "left\\{", "right\\}",
    "approx", "equiv", "sim", "simeq", "propto", "in", "subset", "leq", "geq",
    "neq", "ne", "ll", "gg", "to", "rightarrow", "leftarrow", "Rightarrow",
    "epsilon", "varepsilon", "phi", "varphi", "psi", "chi", "theta", "vartheta",
    "lambda", "mu", "nu", "alpha", "beta", "gamma", "delta", "sigma", "tau",
    "omega", "Omega", "Phi", "Psi", "Gamma", "Delta", "Lambda", "Sigma", "Theta",
    # Note: Greek letters are included here because they are syntax, but physics
    # code often uses variables named 'phi', 'mu' etc. The reconciliation pass
    # will match those against Python identifiers of the same name anyway.
    # We keep Greek in STDLIB so "\phi {command}" doesn't pollute the symbol
    # set twice; the bare "phi" string can still be tracked via LATEX_ID.
    "textrm", "rm", "mathit", "operatorname", "displaystyle", "textstyle",
    "scriptstyle", "mathop", "limits", "nolimits", "boldsymbol",
}


def extract_latex_symbols(text: str) -> set[str]:
    symbols: set[str] = set()
    for match in LATEX_MATH_BLOCK.finditer(text):
        block = next((g for g in match.groups() if g is not None), "")
        # Commands
        for m in LATEX_CMD.finditer(block):
            cmd = m.group(1)
            if cmd not in LATEX_STDLIB:
                symbols.add(f"\\{cmd}")
        # Identifiers and subscripts
        for m in LATEX_ID.finditer(block):
            ident = m.group(1)
            sub = m.group(2)
            if len(ident) == 1 and ident.isalpha():
                # Single-letter variables are often context-dependent; include
                # with subscript but not bare
                if sub:
                    symbols.add(f"{ident}_{sub}")
            else:
                base = ident if not sub else f"{ident}_{sub}"
                symbols.add(base)
    return symbols


def extract_python_symbols(py_dir: Path) -> set[str]:
    symbols: set[str] = set()
    for py_file in py_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts or "tests" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        symbols.add(target.attr)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                symbols.add(node.target.id)
            elif isinstance(node, ast.FunctionDef):
                symbols.add(node.name)
                for arg in node.args.args:
                    symbols.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                symbols.add(node.name)
    return symbols


def normalize_name(name: str) -> str:
    """Strip LaTeX backslashes and braces for comparison."""
    return name.strip("\\").replace("{", "").replace("}", "").lower()


def reconcile(
    latex_syms: set[str],
    py_syms: set[str],
    mapping: dict[str, str] | None = None,
    ignore_latex: set[str] | None = None,
    ignore_python: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    mapping = mapping or {}
    ignore_latex = ignore_latex or set()
    ignore_python = ignore_python or set()

    # Apply mapping: treat mapped LaTeX symbols as if their Python counterpart
    mapped_from_latex = {mapping.get(s, s) for s in latex_syms}
    norm_py = {normalize_name(s) for s in py_syms}
    norm_mapped = {normalize_name(s) for s in mapped_from_latex}

    in_latex_not_code = []
    for sym in latex_syms:
        if sym in ignore_latex:
            continue
        mapped = mapping.get(sym, sym)
        if normalize_name(mapped) not in norm_py:
            in_latex_not_code.append(sym)

    in_code_not_latex = []
    for sym in py_syms:
        if sym in ignore_python:
            continue
        if normalize_name(sym) not in norm_mapped:
            if len(sym) > 1 and not sym.startswith("_"):
                in_code_not_latex.append(sym)

    return sorted(in_latex_not_code), sorted(in_code_not_latex)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare LaTeX formalism symbols against Python implementation"
    )
    parser.add_argument("tex_file", type=Path)
    parser.add_argument("py_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if not args.tex_file.exists():
        print(f"error: {args.tex_file} not found", file=sys.stderr)
        return 2
    if not args.py_dir.exists():
        print(f"error: {args.py_dir} not found", file=sys.stderr)
        return 2

    mapping: dict[str, str] = {}
    ignore_latex: set[str] = set()
    ignore_python: set[str] = set()
    if args.config and args.config.exists():
        if yaml is None:
            print("warning: pyyaml not installed; ignoring config", file=sys.stderr)
        else:
            cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
            mapping = cfg.get("latex_to_python", {}) or {}
            ignore_latex = set(cfg.get("ignore_latex", []) or [])
            ignore_python = set(cfg.get("ignore_python", []) or [])

    tex = args.tex_file.read_text(encoding="utf-8")
    latex_syms = extract_latex_symbols(tex)
    py_syms = extract_python_symbols(args.py_dir)

    in_latex, in_py = reconcile(latex_syms, py_syms, mapping, ignore_latex, ignore_python)

    print(f"LaTeX symbols extracted: {len(latex_syms)}")
    print(f"Python symbols extracted: {len(py_syms)}")
    print()

    if in_latex:
        print(f"=== In LaTeX but not found in code ({len(in_latex)}) ===")
        print("These symbols appear in the formalism but may not be implemented:")
        for s in in_latex:
            print(f"  {s}")
        print()

    if in_py:
        print(f"=== In code but not found in LaTeX ({len(in_py)}) ===")
        print("These symbols appear in the code but are not in the formalism:")
        for s in in_py:
            print(f"  {s}")
        print()

    if not in_latex and not in_py:
        print("All symbols reconcile.")

    print(
        "\nNote: this is heuristic. Review both lists. Use --config with "
        "a mapping file to handle renames (e.g., D_eff -> diffusion_effective) "
        "and ignore lists for known exceptions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
