#!/usr/bin/env python3
"""
Skill Inventory Validator

Loads every skill registered in each plugin's ``plugin.json`` and reports
frontmatter problems (missing name/description, loader errors).

This tool deliberately does NOT score skill "triggering". MyClaude skills all
carry ``disable-model-invocation: true`` and are reached either by slash command
or by a hub SKILL.md routing table that names the target file path, so there is
no description-matching dispatch to measure. The previous keyword-scoring
classifier derived its own ground truth from the score it was testing, making
its accuracy/precision/over-trigger table a tautology that reported 100% for any
description text. It was removed rather than repaired.

Usage:
    python3 tools/validation/skill_validator.py
    python3 tools/validation/skill_validator.py --plugins-dir /path/to/plugins
    python3 tools/validation/skill_validator.py --plugin dev-suite
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Allow ad-hoc `python tools/validation/skill_validator.py` CLI runs by adding
# the repo root to sys.path before resolving the `tools` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.common.loader import PluginLoader


@dataclass
class Skill:
    """A skill registered in a plugin manifest."""

    name: str
    description: str
    status: str
    plugin_name: str
    path: Path | None = None


class SkillInventoryValidator:
    """Loads registered skills and validates their frontmatter."""

    # Descriptions shorter than this carry too little routing information for a
    # hub author or a slash-command user to tell skills apart.
    MIN_DESCRIPTION_CHARS = 40

    def __init__(self, plugins_dir: str):
        self.plugins_dir = Path(plugins_dir)
        self.skills: list[Skill] = []
        self.issues: list[str] = []

    def load_skills(self, specific_plugin: str | None = None) -> None:
        """Load skills from all plugins via the shared PluginLoader."""
        print("Loading plugin skills...")

        loader = PluginLoader(self.plugins_dir)
        if specific_plugin:
            metadata = loader.load_plugin(specific_plugin)
            plugins = {specific_plugin: metadata} if metadata else {}
        else:
            plugins = loader.load_all_plugins()

        for plugin_name, metadata in plugins.items():
            if metadata is None or not metadata.skills:
                continue

            for skill_dict in metadata.skills:
                self._add_skill(skill_dict, plugin_name)

            count = len([s for s in self.skills if s.plugin_name == plugin_name])
            print(f"  ✓ {plugin_name}: {count} skills")

        for error in loader.get_errors():
            message = f"loader: {error.message}"
            print(f"  ⚠ {message}")
            self.issues.append(message)

        print(f"\nLoaded {len(self.skills)} total skills")

    def _add_skill(self, skill_data: dict, plugin_name: str) -> None:
        """Create and add a skill object from a normalized dict."""
        name = skill_data.get("name", "")
        if not name:
            self.issues.append(f"{plugin_name}: skill entry with no `name`")
            return

        skill_path = skill_data.get("path")
        skill = Skill(
            name=name,
            description=skill_data.get("description", ""),
            status=skill_data.get("status", "active"),
            plugin_name=plugin_name,
            path=Path(skill_path) if skill_path else None,
        )
        self.skills.append(skill)

        if not skill.description:
            self.issues.append(f"{plugin_name}/{name}: missing `description`")
        elif len(skill.description) < self.MIN_DESCRIPTION_CHARS:
            self.issues.append(
                f"{plugin_name}/{name}: description is only "
                f"{len(skill.description)} chars "
                f"(< {self.MIN_DESCRIPTION_CHARS})"
            )

    def generate_report(self) -> str:
        """Generate the skill inventory report."""
        by_plugin: dict[str, list[Skill]] = {}
        for skill in self.skills:
            by_plugin.setdefault(skill.plugin_name, []).append(skill)

        report = f"""# Skill Inventory Report

**Validation Date:** {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}
**Total Registered Skills:** {len(self.skills)}

Registered skills are the hub skills listed in each `plugin.json`. Sub-skills
are reached through a hub's routing table and are intentionally not registered.

## Registered Skills by Plugin

| Plugin | Skills |
|--------|--------|
"""
        for plugin_name in sorted(by_plugin):
            report += f"| {plugin_name} | {len(by_plugin[plugin_name])} |\n"

        report += "\n## Frontmatter Issues\n\n"
        if self.issues:
            for issue in self.issues:
                report += f"- {issue}\n"
        else:
            report += "None — every registered skill has a name and a description.\n"

        report += "\n## Overall Assessment\n\n"
        if self.issues:
            report += (
                f"**Status:** ❌ {len(self.issues)} issue(s) found in registered "
                "skill frontmatter.\n"
            )
        else:
            report += "**Status:** ✅ All registered skill frontmatter is valid.\n"

        report += (
            "\n_Skill triggering accuracy is not measured here. All MyClaude "
            "skills set `disable-model-invocation: true`, so they are dispatched "
            "by slash command or by an explicit hub routing table, not by "
            "description matching._\n"
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate registered skill frontmatter and report inventory"
    )
    parser.add_argument(
        "--plugins-dir",
        default="plugins",
        help="Path to plugins directory (default: plugins)",
    )
    parser.add_argument("--plugin", help="Validate specific plugin only")
    parser.add_argument(
        "--output",
        default="reports/skill-validation.md",
        help="Output report file (default: reports/skill-validation.md)",
    )

    args = parser.parse_args()

    plugins_dir = Path(args.plugins_dir).absolute()
    if not plugins_dir.exists():
        print(f"Error: Plugins directory not found: {plugins_dir}")
        return 1

    validator = SkillInventoryValidator(str(plugins_dir))
    validator.load_skills(args.plugin)

    report = validator.generate_report()
    print("\n" + "=" * 70)
    print(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n✓ Report saved to: {output_path.absolute()}")

    return 1 if validator.issues else 0


if __name__ == "__main__":
    sys.exit(main())
