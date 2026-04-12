"""Lightweight structural linter for Claude Code command files.

Runs over ``plugins/*/commands/*.md`` to catch the class of structural
bugs that ruff and mypy cannot see but that directly affect how Claude
parses and executes a command prompt:

1. **Unbalanced code fences** — an odd count of ``` lines means the
   file has an unclosed code block, which breaks markdown rendering
   AND confuses the prompt parser.
2. **Heading-level skips** — a jump like H1 → H3 without an intervening
   H2 is a navigation hazard and usually indicates a refactor accident.
3. **Broken internal Step references** — a dispatch tree that points
   to ``Step 3.5`` when no such section exists is a latent parser bug.
4. **Trailing whitespace on non-code lines** — cosmetic but causes
   noise in diffs and breaks some markdown renderers.
5. **Duplicate headings at the same level within the same parent** —
   ambiguous cross-references, harder to maintain.

This is NOT a full markdownlint replacement. It targets command files
specifically because those files encode executable logic via markdown
and a subtle structural bug can silently change runtime behavior.

Usage (standalone):

    python3 tools/validation/command_file_linter.py plugins/agent-core/commands/team-assemble.md
    python3 tools/validation/command_file_linter.py plugins/  # all commands in all plugins

Usage (importable):

    from tools.validation.command_file_linter import lint_command_file, LintIssue
    issues = lint_command_file(pathlib.Path("plugins/agent-core/commands/team-assemble.md"))
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning", "info"]

# Rule identifiers. Keep stable — tests and CI pin against these.
RULE_UNBALANCED_FENCE = "fence-unbalanced"
RULE_HEADING_SKIP = "heading-skip"
RULE_BROKEN_STEP_REF = "step-ref-broken"
RULE_TRAILING_WS = "trailing-whitespace"
RULE_DUPLICATE_HEADING = "heading-duplicate"


@dataclass(frozen=True)
class LintIssue:
    """A single structural issue found in a command file."""

    path: pathlib.Path
    line: int
    rule: str
    severity: Severity
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: [{self.severity}] {self.rule}: {self.message}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_fences(lines: list[str], path: pathlib.Path) -> list[LintIssue]:
    """Report unbalanced triple-backtick fences.

    Counts lines that BEGIN with ``` (possibly after whitespace).
    Indented fences inside lists are counted too.
    """
    fence_lines: list[int] = []
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            fence_lines.append(i)
    if len(fence_lines) % 2 != 0:
        last = fence_lines[-1]
        return [
            LintIssue(
                path=path,
                line=last,
                rule=RULE_UNBALANCED_FENCE,
                severity="error",
                message=(
                    f"unbalanced code fence: found {len(fence_lines)} ``` markers "
                    f"(odd count means an unclosed code block; last fence at line {last})"
                ),
            )
        ]
    return []


def _check_heading_skips(lines: list[str], path: pathlib.Path) -> list[LintIssue]:
    """Report heading-level skips (e.g., H1 → H3 without H2)."""
    issues: list[LintIssue] = []
    previous_level: int | None = None
    in_fence = False
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+\S", line)
        if not match:
            continue
        level = len(match.group(1))
        if previous_level is not None and level > previous_level + 1:
            issues.append(
                LintIssue(
                    path=path,
                    line=i,
                    rule=RULE_HEADING_SKIP,
                    severity="warning",
                    message=(
                        f"heading level skip: H{previous_level} → H{level} "
                        f"(missing H{previous_level + 1})"
                    ),
                )
            )
        previous_level = level
    return issues


def _check_step_references(text: str, path: pathlib.Path) -> list[LintIssue]:
    """Report dispatch-tree references to Step sections that don't exist.

    Only active if the file has at least one ``## Step`` heading.
    """
    existing = set(
        re.findall(r"^## Step (\d+(?:\.\d+)?[a-z]?):", text, re.MULTILINE)
    ) | set(re.findall(r"^### Step (\d+(?:\.\d+)?[a-z]?):", text, re.MULTILINE))
    if not existing:
        return []

    # Find referenced steps. Match bold and non-bold forms; require a
    # word boundary after the number to avoid matching arbitrary
    # decimals in prose.
    issues: list[LintIssue] = []
    seen: set[tuple[int, str]] = set()
    for match in re.finditer(r"\bStep\s+(\d+(?:\.\d+)?[a-z]?)\b", text):
        ref = match.group(1)
        # Normalize: "2.6a" stays as-is, "2.6" stays as-is, but both
        # should match if either is declared.
        if ref in existing:
            continue
        # Also accept a reference to "2.6" when only "2.6a" exists as a
        # sub-step of 2.6, or vice versa — sub-step prefix match.
        if any(e.startswith(ref) or ref.startswith(e) for e in existing):
            continue
        # Compute the line number of the match
        line = text.count("\n", 0, match.start()) + 1
        key = (line, ref)
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            LintIssue(
                path=path,
                line=line,
                rule=RULE_BROKEN_STEP_REF,
                severity="error",
                message=(
                    f"reference to 'Step {ref}' but no such section exists "
                    f"(declared steps: {sorted(existing)})"
                ),
            )
        )
    return issues


def _check_trailing_whitespace(lines: list[str], path: pathlib.Path) -> list[LintIssue]:
    """Report trailing whitespace on non-code, non-empty lines."""
    issues: list[LintIssue] = []
    in_fence = False
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            continue
        if line != line.rstrip():
            issues.append(
                LintIssue(
                    path=path,
                    line=i,
                    rule=RULE_TRAILING_WS,
                    severity="info",
                    message="trailing whitespace on non-code line",
                )
            )
    return issues


def _check_duplicate_headings(lines: list[str], path: pathlib.Path) -> list[LintIssue]:
    """Report duplicate headings at the same level within the same parent.

    Two H3 sections under the same H2 parent with identical text are
    ambiguous. Duplicates under different parents are allowed (e.g.,
    ``### Examples`` under multiple teams).
    """
    issues: list[LintIssue] = []
    # Track last-seen heading text at each level, plus the parent chain
    seen_under_parent: dict[tuple[int, str], tuple[int, str]] = {}
    parent_chain: dict[int, str] = {}
    in_fence = False
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        level = len(match.group(1))
        text = match.group(2).strip()
        parent = parent_chain.get(level - 1, "<root>") if level > 1 else "<root>"
        key = (level, f"{parent}::{text}")
        if key in seen_under_parent:
            prev_line, _ = seen_under_parent[key]
            issues.append(
                LintIssue(
                    path=path,
                    line=i,
                    rule=RULE_DUPLICATE_HEADING,
                    severity="warning",
                    message=(
                        f"duplicate H{level} heading {text!r} under the same "
                        f"parent section (first seen at line {prev_line})"
                    ),
                )
            )
        else:
            seen_under_parent[key] = (i, text)
        parent_chain[level] = text
        # Clear deeper parents when a higher-level heading appears
        for deeper in list(parent_chain):
            if deeper > level:
                del parent_chain[deeper]
        # Clear tracked duplicates under cleared parents
        seen_under_parent = {
            k: v
            for k, v in seen_under_parent.items()
            if k[0] <= level or not k[1].startswith(f"{text}::")
        }
    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lint_command_file(path: pathlib.Path) -> list[LintIssue]:
    """Run all structural checks on a single command file.

    Args:
        path: Path to a markdown command file.

    Returns:
        A flat list of :class:`LintIssue` objects, sorted by line number.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)

    issues: list[LintIssue] = []
    issues.extend(_check_fences(lines, path))
    issues.extend(_check_heading_skips(lines, path))
    issues.extend(_check_step_references(text, path))
    issues.extend(_check_trailing_whitespace(lines, path))
    issues.extend(_check_duplicate_headings(lines, path))

    return sorted(issues, key=lambda issue: (issue.line, issue.rule))


def lint_paths(roots: list[pathlib.Path]) -> list[LintIssue]:
    """Run the linter on every command file under the given paths.

    Each root may be a specific markdown file or a directory to walk.
    Within directories, only ``*/commands/*.md`` files are linted.
    """
    targets: list[pathlib.Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".md":
            targets.append(root)
        elif root.is_dir():
            targets.extend(sorted(root.rglob("commands/*.md")))

    all_issues: list[LintIssue] = []
    for target in targets:
        all_issues.extend(lint_command_file(target))
    return all_issues


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Structural linter for Claude Code command files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=pathlib.Path,
        help="Command file paths or directory roots to scan",
    )
    parser.add_argument(
        "--max-severity",
        choices=["error", "warning", "info"],
        default="error",
        help=(
            "Exit non-zero only when issues at or above this severity "
            "are found (default: error)"
        ),
    )
    args = parser.parse_args(argv)

    issues = lint_paths(args.paths)

    severity_order = {"error": 2, "warning": 1, "info": 0}
    threshold = severity_order[args.max_severity]

    for issue in issues:
        print(issue.format())

    if not issues:
        print("OK: no structural issues found")
        return 0

    hard_issues = sum(
        1 for issue in issues if severity_order[issue.severity] >= threshold
    )
    print(f"\n{len(issues)} total issue(s); {hard_issues} at >= {args.max_severity}")
    return 1 if hard_issues else 0


if __name__ == "__main__":
    sys.exit(main())
