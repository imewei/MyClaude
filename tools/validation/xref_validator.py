#!/usr/bin/env python3
"""
Cross-Reference Validator

Validates cross-plugin references by:
- Checking all cross-plugin references
- Validating agent/command/skill mentions
- Identifying broken references
- Generating validation reports

Author: Quality Engineer / Technical Writer
Part of: Plugin Review and Optimization - Task Group 0.4
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import sys

# Allow ad-hoc `python tools/validation/xref_validator.py` CLI runs by adding
# the repo root to sys.path before resolving the `tools` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.common.loader import PluginLoader  # noqa: E402


@dataclass
class CrossReference:
    """A cross-reference from one plugin to another"""

    source_plugin: str
    source_file: str
    source_line: int
    target_plugin: str
    target_type: str  # 'plugin', 'agent', 'command', 'skill'
    target_name: str
    context: str
    is_valid: bool = True
    error_message: str = ""


@dataclass
class XrefResult:
    """Results of cross-reference validation"""

    total_references: int = 0
    valid_references: int = 0
    broken_references: List[CrossReference] = field(default_factory=list)
    warnings: List[CrossReference] = field(default_factory=list)
    plugin_index: Dict[str, dict] = field(default_factory=dict)


class CrossReferenceValidator:
    """Validates cross-references between plugins"""

    # Patterns for detecting cross-references
    REFERENCE_PATTERNS = {
        "plugin_mention": r"\b([a-z]+-[a-z-]+)(?:\s+plugin)?\b",
        "agent_reference": r"\bagent:\s*([a-z]+-[a-z-]+)\b",
        "command_reference": r"/([a-z]+-[a-z-]+)\b",
        "skill_reference": r"\bskill:\s*([a-z]+-[a-z-]+)\b",
        "markdown_link": r"\[([^\]]+)\]\(([^\)]+)\)",
        # Relative skill link inside a plugin: [Name](../sibling-skill/SKILL.md)
        # The captured group is the sibling skill directory name.
        "relative_skill_link": r"\(\.\./([a-z][a-z0-9-]+)/SKILL\.md\)",
        # Absolute resource link via plugins/<plugin>/<kind>/<name>[.md]
        # kind ∈ {agents, commands, skills}; name is the file or directory stem.
        "absolute_resource_link": (
            r"plugins/([a-z][a-z0-9-]+)/(agents|commands|skills)/"
            r"([a-z][a-z0-9-]+)(?:\.md|/SKILL\.md)?"
        ),
    }

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = plugins_dir
        self.result = XrefResult()
        self.references: List[CrossReference] = []
        self._build_plugin_index()

    def _build_plugin_index(self):
        """Build index of all plugins, agents, commands, and skills.

        Manifest-declared hubs come from PluginLoader (which also normalizes
        path-string entries). Sub-skills are discovered from disk by scanning
        ``plugins/<plugin>/skills/<name>/SKILL.md`` — they exist on the
        filesystem but are intentionally absent from ``plugin.json`` per the
        hub-skill routing architecture.
        """
        print("📋 Building plugin index...")

        loader = PluginLoader(self.plugins_dir)
        all_plugins = loader.load_all_plugins()

        for plugin_name, metadata in all_plugins.items():
            plugin_path = Path(metadata.path) if metadata.path else None

            # Manifest-declared components (hubs only for skills).
            agents = {a.get("name"): a.get("description", "") for a in metadata.agents}
            commands = {
                c.get("name"): c.get("description", "") for c in metadata.commands
            }
            skills = {s.get("name"): s.get("description", "") for s in metadata.skills}

            # Disk-discovered sub-skills: any directory under skills/ with a
            # SKILL.md file. Adds the directory name to the skills index so
            # hub→sub-skill ../sibling/SKILL.md links validate cleanly.
            if plugin_path is not None:
                skills_dir = plugin_path / "skills"
                if skills_dir.exists() and skills_dir.is_dir():
                    for entry in skills_dir.iterdir():
                        if entry.is_dir() and (entry / "SKILL.md").exists():
                            skills.setdefault(entry.name, "")

                # Disk-discovered agent and command files. Lets absolute-path
                # references like plugins/<plugin>/agents/<name>.md resolve
                # even when the manifest entry uses a different stem.
                agents_dir = plugin_path / "agents"
                if agents_dir.exists() and agents_dir.is_dir():
                    for entry in agents_dir.glob("*.md"):
                        agents.setdefault(entry.stem, "")

                commands_dir = plugin_path / "commands"
                if commands_dir.exists() and commands_dir.is_dir():
                    for entry in commands_dir.glob("*.md"):
                        commands.setdefault(entry.stem, "")

            self.result.plugin_index[plugin_name] = {
                "version": metadata.version,
                "category": metadata.category,
                "agents": agents,
                "commands": commands,
                "skills": skills,
                "path": str(metadata.path) if metadata.path else "",
            }

        for error in loader.get_errors():
            print(f"⚠️  Warning: {error.message}")

    def validate_all_references(self) -> XrefResult:
        """Validate all cross-references in all plugins"""
        print("🔍 Validating cross-references...")

        # Step 1: Extract all references
        self._extract_all_references()

        # Step 2: Validate each reference
        self._validate_references()

        # Step 3: Calculate statistics
        self.result.total_references = len(self.references)
        self.result.valid_references = sum(1 for ref in self.references if ref.is_valid)
        self.result.broken_references = [
            ref for ref in self.references if not ref.is_valid
        ]

        return self.result

    def _extract_all_references(self):
        """Extract all cross-references from all plugins"""
        for plugin_name in self.result.plugin_index.keys():
            plugin_dir = Path(self.result.plugin_index[plugin_name]["path"])

            # Check README
            readme_path = plugin_dir / "README.md"
            if readme_path.exists():
                self._extract_from_file(readme_path, plugin_name)

            # Check agent documentation
            agents_dir = plugin_dir / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.md"):
                    self._extract_from_file(agent_file, plugin_name)

            # Check command documentation
            commands_dir = plugin_dir / "commands"
            if commands_dir.exists():
                for command_file in commands_dir.glob("*.md"):
                    self._extract_from_file(command_file, plugin_name)

            # Check skill documentation
            skills_dir = plugin_dir / "skills"
            if skills_dir.exists():
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir():
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.exists():
                            self._extract_from_file(skill_file, plugin_name)

    def _extract_from_file(self, file_path: Path, source_plugin: str):
        """Extract cross-references from a file"""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                # Skip code blocks
                if line.strip().startswith("```"):
                    continue

                # Extract plugin mentions
                self._extract_plugin_mentions(line, file_path, source_plugin, line_num)

                # Extract agent references
                self._extract_agent_references(line, file_path, source_plugin, line_num)

                # Extract command references
                self._extract_command_references(
                    line, file_path, source_plugin, line_num
                )

                # Extract skill references
                self._extract_skill_references(line, file_path, source_plugin, line_num)

                # Extract markdown links
                self._extract_markdown_links(line, file_path, source_plugin, line_num)

                # Extract intra-plugin relative skill links: ../sibling/SKILL.md
                self._extract_relative_skill_links(
                    line, file_path, source_plugin, line_num
                )

                # Extract absolute resource links: plugins/<plugin>/<kind>/<name>
                self._extract_absolute_resource_links(
                    line, file_path, source_plugin, line_num
                )

        except Exception as e:
            print(f"⚠️  Warning: Error extracting from {file_path}: {e}")

    def _extract_plugin_mentions(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract plugin name mentions"""
        for plugin_name in self.result.plugin_index.keys():
            if plugin_name == source_plugin:
                continue

            # Look for plugin name with optional "plugin" suffix
            patterns = [
                rf"\b{re.escape(plugin_name)}\s+plugin\b",
                rf"\b`{re.escape(plugin_name)}`\b",
            ]

            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    context = line.strip()[:100]
                    ref = CrossReference(
                        source_plugin=source_plugin,
                        source_file=str(file_path.relative_to(self.plugins_dir)),
                        source_line=line_num,
                        target_plugin=plugin_name,
                        target_type="plugin",
                        target_name=plugin_name,
                        context=context,
                    )
                    self.references.append(ref)
                    break

    def _extract_agent_references(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract agent references"""
        # Look for patterns like "agent: agent-name" or "@agent-name"
        patterns = [
            r"\bagent:\s*([a-z]+-[a-z-]+)\b",
            r"\b@([a-z]+-[a-z-]+)\b",
            r"\bagent\s+`([a-z]+-[a-z-]+)`",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                agent_name = match.group(1)
                context = line.strip()[:100]

                # Try to find which plugin this agent belongs to
                target_plugin = self._find_agent_plugin(agent_name)

                if target_plugin and target_plugin != source_plugin:
                    ref = CrossReference(
                        source_plugin=source_plugin,
                        source_file=str(file_path.relative_to(self.plugins_dir)),
                        source_line=line_num,
                        target_plugin=target_plugin,
                        target_type="agent",
                        target_name=agent_name,
                        context=context,
                    )
                    self.references.append(ref)

    def _extract_command_references(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract command references (slash commands)"""
        # Look for /command-name patterns
        pattern = r"/([a-z]+-[a-z-]+)\b"

        for match in re.finditer(pattern, line):
            command_name = match.group(1)
            context = line.strip()[:100]

            # Try to find which plugin this command belongs to
            target_plugin = self._find_command_plugin(command_name)

            if target_plugin and target_plugin != source_plugin:
                ref = CrossReference(
                    source_plugin=source_plugin,
                    source_file=str(file_path.relative_to(self.plugins_dir)),
                    source_line=line_num,
                    target_plugin=target_plugin,
                    target_type="command",
                    target_name=command_name,
                    context=context,
                )
                self.references.append(ref)

    def _extract_skill_references(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract skill references"""
        # Look for patterns like "skill: skill-name"
        patterns = [
            r"\bskill:\s*([a-z]+-[a-z-]+)\b",
            r"\bskill\s+`([a-z]+-[a-z-]+)`",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                skill_name = match.group(1)
                context = line.strip()[:100]

                # Try to find which plugin this skill belongs to
                target_plugin = self._find_skill_plugin(skill_name)

                if target_plugin and target_plugin != source_plugin:
                    ref = CrossReference(
                        source_plugin=source_plugin,
                        source_file=str(file_path.relative_to(self.plugins_dir)),
                        source_line=line_num,
                        target_plugin=target_plugin,
                        target_type="skill",
                        target_name=skill_name,
                        context=context,
                    )
                    self.references.append(ref)

    def _extract_markdown_links(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract and validate markdown links"""
        pattern = r"\[([^\]]+)\]\(([^\)]+)\)"

        for match in re.finditer(pattern, line):
            link_text = match.group(1)
            link_url = match.group(2)

            # Check if link is to another plugin
            for plugin_name in self.result.plugin_index.keys():
                if plugin_name in link_url or plugin_name in link_text:
                    if plugin_name != source_plugin:
                        context = line.strip()[:100]
                        ref = CrossReference(
                            source_plugin=source_plugin,
                            source_file=str(file_path.relative_to(self.plugins_dir)),
                            source_line=line_num,
                            target_plugin=plugin_name,
                            target_type="link",
                            target_name=link_url,
                            context=context,
                        )
                        self.references.append(ref)
                        break

    def _extract_relative_skill_links(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract intra-plugin relative skill links: ``[Name](../sibling/SKILL.md)``.

        Hub skills wire their sub-skills via ``../<sibling>/SKILL.md`` relative
        links inside ``Core Skills`` sections. The plain markdown-link extractor
        misses these because the URL never contains a plugin name. Treats the
        captured directory name as a skill reference inside the *source* plugin.
        """
        pattern = self.REFERENCE_PATTERNS["relative_skill_link"]

        for match in re.finditer(pattern, line):
            skill_name = match.group(1)
            context = line.strip()[:100]
            ref = CrossReference(
                source_plugin=source_plugin,
                source_file=str(file_path.relative_to(self.plugins_dir)),
                source_line=line_num,
                target_plugin=source_plugin,  # intra-plugin
                target_type="skill",
                target_name=skill_name,
                context=context,
            )
            self.references.append(ref)

    def _extract_absolute_resource_links(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract absolute resource paths: ``plugins/<plugin>/<kind>/<name>``.

        Skill ``Expert Agent`` blocks point at agents using literal paths like
        ``plugins/science-suite/agents/julia-pro.md``. Hub manifests and prose
        sometimes refer to peer skills via the same form. The kind segment is
        ``agents``, ``commands``, or ``skills``; the trailing ``.md`` (for
        agent/command files) or ``/SKILL.md`` (for skill directories) is
        optional in the regex so both shapes match.
        """
        pattern = self.REFERENCE_PATTERNS["absolute_resource_link"]

        # Map manifest kind → CrossReference target type
        kind_to_type = {"agents": "agent", "commands": "command", "skills": "skill"}

        for match in re.finditer(pattern, line):
            target_plugin = match.group(1)
            kind = match.group(2)
            target_name = match.group(3)
            target_type = kind_to_type.get(kind, kind)
            context = line.strip()[:100]
            ref = CrossReference(
                source_plugin=source_plugin,
                source_file=str(file_path.relative_to(self.plugins_dir)),
                source_line=line_num,
                target_plugin=target_plugin,
                target_type=target_type,
                target_name=target_name,
                context=context,
            )
            self.references.append(ref)

    def _find_agent_plugin(self, agent_name: str) -> str:
        """Find which plugin owns an agent"""
        for plugin_name, data in self.result.plugin_index.items():
            if agent_name in data["agents"]:
                return plugin_name
        return ""

    def _find_command_plugin(self, command_name: str) -> str:
        """Find which plugin owns a command"""
        for plugin_name, data in self.result.plugin_index.items():
            if command_name in data["commands"]:
                return plugin_name
        return ""

    def _find_skill_plugin(self, skill_name: str) -> str:
        """Find which plugin owns a skill"""
        for plugin_name, data in self.result.plugin_index.items():
            if skill_name in data["skills"]:
                return plugin_name
        return ""

    def _validate_references(self):
        """Validate all extracted references"""
        for ref in self.references:
            # Check if target plugin exists
            if ref.target_plugin not in self.result.plugin_index:
                ref.is_valid = False
                ref.error_message = f"Plugin '{ref.target_plugin}' not found"
                continue

            # Validate based on type
            plugin_data = self.result.plugin_index[ref.target_plugin]

            if ref.target_type == "agent":
                if ref.target_name not in plugin_data["agents"]:
                    ref.is_valid = False
                    ref.error_message = f"Agent '{ref.target_name}' not found in plugin '{ref.target_plugin}'"

            elif ref.target_type == "command":
                if ref.target_name not in plugin_data["commands"]:
                    ref.is_valid = False
                    ref.error_message = f"Command '{ref.target_name}' not found in plugin '{ref.target_plugin}'"

            elif ref.target_type == "skill":
                if ref.target_name not in plugin_data["skills"]:
                    ref.is_valid = False
                    ref.error_message = f"Skill '{ref.target_name}' not found in plugin '{ref.target_plugin}'"

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate validation report"""
        lines = []

        # Header
        lines.append("# Cross-Reference Validation Report")
        lines.append("")
        lines.append("Validation of cross-plugin references in Claude Code marketplace")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        valid_pct = (
            (self.result.valid_references / self.result.total_references * 100)
            if self.result.total_references > 0
            else 0
        )
        lines.append(f"- **Total References:** {self.result.total_references}")
        lines.append(
            f"- **Valid References:** {self.result.valid_references} ({valid_pct:.1f}%)"
        )
        lines.append(f"- **Broken References:** {len(self.result.broken_references)}")
        lines.append(f"- **Plugins Indexed:** {len(self.result.plugin_index)}")
        lines.append("")

        # Status indicator
        if len(self.result.broken_references) == 0:
            lines.append("✅ **Status:** All cross-references are valid!")
        elif len(self.result.broken_references) < self.result.total_references * 0.05:
            lines.append("⚠️  **Status:** Minor issues found (< 5% broken)")
        else:
            lines.append("❌ **Status:** Significant issues found (>= 5% broken)")
        lines.append("")

        # Reference type breakdown
        lines.append("## Reference Type Distribution")
        lines.append("")
        type_counts: Dict[str, int] = defaultdict(int)
        for ref in self.references:
            type_counts[ref.target_type] += 1

        for ref_type in sorted(type_counts.keys()):
            count = type_counts[ref_type]
            broken_count = sum(
                1
                for ref in self.result.broken_references
                if ref.target_type == ref_type
            )
            status = "✅" if broken_count == 0 else "❌"
            lines.append(
                f"- **{ref_type}**: {count} total, {broken_count} broken {status}"
            )
        lines.append("")

        # Broken references
        if self.result.broken_references:
            lines.append("## Broken References")
            lines.append("")
            lines.append(
                f"Found {len(self.result.broken_references)} broken references:"
            )
            lines.append("")

            # Group by source plugin
            by_plugin = defaultdict(list)
            for ref in self.result.broken_references:
                by_plugin[ref.source_plugin].append(ref)

            for plugin in sorted(by_plugin.keys()):
                refs = by_plugin[plugin]
                lines.append(f"### {plugin} ({len(refs)} broken)")
                lines.append("")

                for ref in refs:
                    lines.append(f"**{ref.target_type}: `{ref.target_name}`**")
                    lines.append(
                        f"- File: `{ref.source_file}` (line {ref.source_line})"
                    )
                    lines.append(f"- Target Plugin: `{ref.target_plugin}`")
                    lines.append(f"- Error: {ref.error_message}")
                    lines.append(f"- Context: {ref.context}")
                    lines.append("")

        # Valid reference summary
        lines.append("## Valid References Summary")
        lines.append("")

        # Group valid references by target plugin
        valid_refs = [ref for ref in self.references if ref.is_valid]
        by_target = defaultdict(list)
        for ref in valid_refs:
            by_target[ref.target_plugin].append(ref)

        lines.append("Most referenced plugins:")
        lines.append("")
        target_counts = [(plugin, len(refs)) for plugin, refs in by_target.items()]
        target_counts.sort(key=lambda x: -x[1])

        for plugin, count in target_counts[:20]:  # Top 20
            lines.append(f"- `{plugin}`: {count} references")
        lines.append("")

        # Plugin index
        lines.append("## Plugin Index")
        lines.append("")
        lines.append(f"Total plugins indexed: {len(self.result.plugin_index)}")
        lines.append("")

        for plugin_name in sorted(self.result.plugin_index.keys()):
            data = self.result.plugin_index[plugin_name]
            lines.append(f"### {plugin_name}")
            lines.append(f"- **Version:** {data['version']}")
            lines.append(f"- **Category:** {data['category']}")
            lines.append(f"- **Agents:** {len(data['agents'])}")
            lines.append(f"- **Commands:** {len(data['commands'])}")
            lines.append(f"- **Skills:** {len(data['skills'])}")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if self.result.broken_references:
            lines.append("### Fix Broken References")
            lines.append("")
            for ref in self.result.broken_references[:10]:  # Top 10
                lines.append(
                    f"- Fix `{ref.source_file}` line {ref.source_line}: {ref.error_message}"
                )
            if len(self.result.broken_references) > 10:
                lines.append(
                    f"- _(... and {len(self.result.broken_references) - 10} more)_"
                )
            lines.append("")

        lines.append("### Best Practices")
        lines.append("")
        lines.append(
            "- Always verify plugin/agent/command/skill names before referencing"
        )
        lines.append("- Use backticks around technical names for clarity")
        lines.append(
            "- Include plugin name when referencing agents/commands/skills from other plugins"
        )
        lines.append("- Regularly run cross-reference validation during development")
        lines.append("")

        report = "\n".join(lines)

        # Write to file if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(f"✅ Validation report saved to: {output_path}")

        return report

    def export_json(self, output_path: Path):
        """Export validation results as JSON"""
        data = {
            "summary": {
                "total_references": self.result.total_references,
                "valid_references": self.result.valid_references,
                "broken_count": len(self.result.broken_references),
                "plugins_indexed": len(self.result.plugin_index),
            },
            "broken_references": [
                {
                    "source_plugin": ref.source_plugin,
                    "source_file": ref.source_file,
                    "source_line": ref.source_line,
                    "target_plugin": ref.target_plugin,
                    "target_type": ref.target_type,
                    "target_name": ref.target_name,
                    "error": ref.error_message,
                    "context": ref.context,
                }
                for ref in self.result.broken_references
            ],
            "plugin_index": self.result.plugin_index,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Validation data exported to: {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate cross-references between Claude Code plugins"
    )
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=Path.cwd() / "plugins",
        help="Path to plugins directory (default: ./plugins)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/xref-validation.md"),
        help="Output file for validation report (default: reports/xref-validation.md)",
    )
    parser.add_argument(
        "--export-json", type=Path, help="Export validation results as JSON"
    )

    args = parser.parse_args()

    # Validate plugins directory
    if not args.plugins_dir.exists():
        print(f"❌ Error: Plugins directory not found: {args.plugins_dir}")
        sys.exit(1)

    # Create validator and validate
    validator = CrossReferenceValidator(args.plugins_dir)
    result = validator.validate_all_references()

    # Generate report
    print("\n📊 Generating validation report...")
    validator.generate_report(args.output)

    # Export JSON if requested
    if args.export_json:
        validator.export_json(args.export_json)

    # Print summary
    print("\n✅ Validation complete!")
    print(f"   Total references: {result.total_references}")
    print(f"   Valid: {result.valid_references}")
    print(f"   Broken: {len(result.broken_references)}")

    # Exit with error code if broken references found
    if result.broken_references:
        print(f"\n⚠️  Found {len(result.broken_references)} broken references!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
