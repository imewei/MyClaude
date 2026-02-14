#!/usr/bin/env python3
"""
Context Budget Checker for Skills (v2.2.1+)

Claude Code v2.1.32+ allocates 2% of the context window for skill content.
With Opus 4.6's 200K context, that's ~4,000 tokens (~3,000 words).
With the 1M beta context, that's ~20,000 tokens (~15,000 words).

This tool checks if SKILL.md files fit within the context budget and
identifies skills that should front-load critical information.

Usage:
    python3 tools/validation/context_budget_checker.py
    python3 tools/validation/context_budget_checker.py --plugins-dir /path/to/plugins
    python3 tools/validation/context_budget_checker.py --context-size 200000
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# Approximate tokens per character ratio for English text
CHARS_PER_TOKEN = 4.0

# Context window sizes
CONTEXT_SIZES = {
    "200k": 200_000,
    "1m": 1_000_000,
}

# Skill budget is 2% of context window
SKILL_BUDGET_PERCENT = 0.02


@dataclass
class SkillBudgetResult:
    """Result of a skill budget check."""

    skill_name: str
    plugin_name: str
    file_path: str
    char_count: int
    estimated_tokens: int
    budget_200k: int
    budget_1m: int
    fits_200k: bool
    fits_1m: bool
    has_frontmatter: bool
    first_section_tokens: int


@dataclass
class BudgetReport:
    """Aggregated budget check report."""

    results: list[SkillBudgetResult] = field(default_factory=list)
    total_skills: int = 0
    fits_200k_count: int = 0
    fits_1m_count: int = 0
    oversized_skills: list[SkillBudgetResult] = field(default_factory=list)
    headroom_warnings: list[SkillBudgetResult] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return int(len(text) / CHARS_PER_TOKEN)


def get_first_section_size(content: str) -> int:
    """Get the size of the first section (before the second ## heading)."""
    lines = content.split("\n")
    in_frontmatter = False
    past_frontmatter = False
    first_heading_found = False
    first_section_lines: list[str] = []

    for line in lines:
        if line.strip() == "---" and not past_frontmatter:
            if in_frontmatter:
                past_frontmatter = True
                in_frontmatter = False
            else:
                in_frontmatter = True
            continue

        if in_frontmatter:
            continue

        if line.startswith("## ") and first_heading_found:
            break

        if line.startswith("# ") or line.startswith("## "):
            first_heading_found = True

        first_section_lines.append(line)

    return estimate_tokens("\n".join(first_section_lines))


def check_skill_budget(skill_path: Path, plugin_name: str) -> SkillBudgetResult | None:
    """Check a single skill's context budget."""
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return None

    content = skill_md.read_text(encoding="utf-8")
    char_count = len(content)
    estimated_tokens = estimate_tokens(content)

    budget_200k = int(CONTEXT_SIZES["200k"] * SKILL_BUDGET_PERCENT)
    budget_1m = int(CONTEXT_SIZES["1m"] * SKILL_BUDGET_PERCENT)

    has_frontmatter = content.strip().startswith("---")
    first_section_tokens = get_first_section_size(content)

    return SkillBudgetResult(
        skill_name=skill_path.name,
        plugin_name=plugin_name,
        file_path=str(skill_md),
        char_count=char_count,
        estimated_tokens=estimated_tokens,
        budget_200k=budget_200k,
        budget_1m=budget_1m,
        fits_200k=estimated_tokens <= budget_200k,
        fits_1m=estimated_tokens <= budget_1m,
        has_frontmatter=has_frontmatter,
        first_section_tokens=first_section_tokens,
    )


def check_all_plugins(plugins_dir: Path) -> BudgetReport:
    """Check all plugin skills for context budget compliance."""
    report = BudgetReport()

    for plugin_dir in sorted(plugins_dir.iterdir()):
        if not plugin_dir.is_dir():
            continue

        plugin_name = plugin_dir.name
        skills_dir = plugin_dir / "skills"
        if not skills_dir.exists():
            continue

        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue

            result = check_skill_budget(skill_dir, plugin_name)
            if result is None:
                continue

            report.results.append(result)
            report.total_skills += 1

            if result.fits_200k:
                report.fits_200k_count += 1
                # Headroom warning: >75% of budget
                if result.estimated_tokens > result.budget_200k * 0.75:
                    report.headroom_warnings.append(result)
            else:
                report.oversized_skills.append(result)

            if result.fits_1m:
                report.fits_1m_count += 1

    return report


def generate_report(report: BudgetReport) -> str:
    """Generate context budget report."""
    lines = [
        "# Skill Context Budget Report (v2.2.1+)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Skills Checked:** {report.total_skills}",
        "",
        "## Context Budget Reference",
        "",
        "| Context Window | Budget (2%) | Approx. Words |",
        "|---------------|-------------|---------------|",
        f"| 200K tokens | {int(CONTEXT_SIZES['200k'] * SKILL_BUDGET_PERCENT):,} tokens | ~{int(CONTEXT_SIZES['200k'] * SKILL_BUDGET_PERCENT * 0.75):,} |",
        f"| 1M tokens | {int(CONTEXT_SIZES['1m'] * SKILL_BUDGET_PERCENT):,} tokens | ~{int(CONTEXT_SIZES['1m'] * SKILL_BUDGET_PERCENT * 0.75):,} |",
        "",
        "## Summary",
        "",
        "| Metric | Count | Rate |",
        "|--------|-------|------|",
        f"| Fits 200K budget | {report.fits_200k_count}/{report.total_skills} | {report.fits_200k_count/max(report.total_skills,1)*100:.0f}% |",
        f"| Fits 1M budget | {report.fits_1m_count}/{report.total_skills} | {report.fits_1m_count/max(report.total_skills,1)*100:.0f}% |",
        f"| Oversized (200K) | {len(report.oversized_skills)}/{report.total_skills} | {len(report.oversized_skills)/max(report.total_skills,1)*100:.0f}% |",
        "",
    ]

    if report.oversized_skills:
        lines.extend([
            "## Oversized Skills (exceed 200K budget)",
            "",
            "| Skill | Plugin | Est. Tokens | Budget | Over By |",
            "|-------|--------|-------------|--------|---------|",
        ])
        for r in sorted(report.oversized_skills, key=lambda x: x.estimated_tokens, reverse=True):
            over_by = r.estimated_tokens - r.budget_200k
            lines.append(
                f"| {r.skill_name} | {r.plugin_name} | "
                f"{r.estimated_tokens:,} | {r.budget_200k:,} | +{over_by:,} |"
            )
        lines.append("")

    if report.headroom_warnings:
        lines.extend([
            "## Headroom Warnings (>75% of 200K budget)",
            "",
            "| Skill | Plugin | Est. Tokens | Budget | Usage |",
            "|-------|--------|-------------|--------|-------|",
        ])
        for r in sorted(report.headroom_warnings, key=lambda x: x.estimated_tokens, reverse=True):
            usage_pct = r.estimated_tokens / r.budget_200k * 100
            lines.append(
                f"| {r.skill_name} | {r.plugin_name} | "
                f"{r.estimated_tokens:,} | {r.budget_200k:,} | {usage_pct:.0f}% |"
            )
        lines.append("")

    # All skills table
    lines.extend([
        "## All Skills by Size",
        "",
        "| Skill | Plugin | Tokens | 200K | 1M | First Section |",
        "|-------|--------|--------|------|-----|--------------|",
    ])
    for r in sorted(report.results, key=lambda x: x.estimated_tokens, reverse=True)[:30]:
        status_200k = "pass" if r.fits_200k else "OVER"
        status_1m = "pass" if r.fits_1m else "OVER"
        lines.append(
            f"| {r.skill_name} | {r.plugin_name} | "
            f"{r.estimated_tokens:,} | {status_200k} | {status_1m} | "
            f"{r.first_section_tokens:,} |"
        )

    if report.total_skills > 30:
        lines.append("| ... | ... | ... | ... | ... | ... |")
        lines.append(f"| *(showing top 30 of {report.total_skills})* | | | | | |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check skill files against context budget constraints"
    )
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=Path("plugins"),
        help="Path to plugins directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output report path (default: stdout)",
    )
    args = parser.parse_args()

    if not args.plugins_dir.exists():
        print(f"Error: Plugins directory not found: {args.plugins_dir}")
        sys.exit(1)

    report = check_all_plugins(args.plugins_dir)
    output = generate_report(report)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)

    # Exit code based on compliance
    if report.oversized_skills:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
