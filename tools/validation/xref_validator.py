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
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
import sys


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
class ValidationResult:
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
        'plugin_mention': r'\b([a-z]+-[a-z-]+)(?:\s+plugin)?\b',
        'agent_reference': r'\bagent:\s*([a-z]+-[a-z-]+)\b',
        'command_reference': r'/([a-z]+-[a-z-]+)\b',
        'skill_reference': r'\bskill:\s*([a-z]+-[a-z-]+)\b',
        'markdown_link': r'\[([^\]]+)\]\(([^\)]+)\)',
    }

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = plugins_dir
        self.result = ValidationResult()
        self.references: List[CrossReference] = []
        self._build_plugin_index()

    def _build_plugin_index(self):
        """Build index of all plugins, agents, commands, and skills"""
        print("📋 Building plugin index...")

        plugin_dirs = [d for d in self.plugins_dir.iterdir() if d.is_dir()]

        for plugin_dir in sorted(plugin_dirs):
            plugin_json = plugin_dir / "plugin.json"
            if not plugin_json.exists():
                continue

            try:
                with open(plugin_json) as f:
                    data = json.load(f)

                plugin_name = data.get("name", plugin_dir.name)

                self.result.plugin_index[plugin_name] = {
                    "version": data.get("version", "unknown"),
                    "category": data.get("category", "uncategorized"),
                    "agents": {
                        a.get("name"): a.get("description", "")
                        for a in data.get("agents", [])
                    },
                    "commands": {
                        c.get("name"): c.get("description", "")
                        for c in data.get("commands", [])
                    },
                    "skills": {
                        s.get("name"): s.get("description", "")
                        for s in data.get("skills", [])
                    },
                    "path": str(plugin_dir)
                }

            except Exception as e:
                print(f"⚠️  Warning: Error loading {plugin_json}: {e}")

    def validate_all_references(self) -> ValidationResult:
        """Validate all cross-references in all plugins"""
        print("🔍 Validating cross-references...")

        # Step 1: Extract all references
        self._extract_all_references()

        # Step 2: Validate each reference
        self._validate_references()

        # Step 3: Calculate statistics
        self.result.total_references = len(self.references)
        self.result.valid_references = sum(
            1 for ref in self.references if ref.is_valid
        )
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
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                # Skip code blocks
                if line.strip().startswith('```'):
                    continue

                # Extract plugin mentions
                self._extract_plugin_mentions(
                    line, file_path, source_plugin, line_num
                )

                # Extract agent references
                self._extract_agent_references(
                    line, file_path, source_plugin, line_num
                )

                # Extract command references
                self._extract_command_references(
                    line, file_path, source_plugin, line_num
                )

                # Extract skill references
                self._extract_skill_references(
                    line, file_path, source_plugin, line_num
                )

                # Extract markdown links
                self._extract_markdown_links(
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
                rf'\b{re.escape(plugin_name)}\s+plugin\b',
                rf'\b`{re.escape(plugin_name)}`\b',
            ]

            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    context = line.strip()[:100]
                    ref = CrossReference(
                        source_plugin=source_plugin,
                        source_file=str(file_path.relative_to(self.plugins_dir)),
                        source_line=line_num,
                        target_plugin=plugin_name,
                        target_type='plugin',
                        target_name=plugin_name,
                        context=context
                    )
                    self.references.append(ref)
                    break

    def _extract_agent_references(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract agent references"""
        # Look for patterns like "agent: agent-name" or "@agent-name"
        patterns = [
            r'\bagent:\s*([a-z]+-[a-z-]+)\b',
            r'\b@([a-z]+-[a-z-]+)\b',
            r'\bagent\s+`([a-z]+-[a-z-]+)`',
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
                        target_type='agent',
                        target_name=agent_name,
                        context=context
                    )
                    self.references.append(ref)

    def _extract_command_references(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract command references (slash commands)"""
        # Look for /command-name patterns
        pattern = r'/([a-z]+-[a-z-]+)\b'

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
                    target_type='command',
                    target_name=command_name,
                    context=context
                )
                self.references.append(ref)

    def _extract_skill_references(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract skill references"""
        # Look for patterns like "skill: skill-name"
        patterns = [
            r'\bskill:\s*([a-z]+-[a-z-]+)\b',
            r'\bskill\s+`([a-z]+-[a-z-]+)`',
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
                        target_type='skill',
                        target_name=skill_name,
                        context=context
                    )
                    self.references.append(ref)

    def _extract_markdown_links(
        self, line: str, file_path: Path, source_plugin: str, line_num: int
    ):
        """Extract and validate markdown links"""
        pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

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
                            target_type='link',
                            target_name=link_url,
                            context=context
                        )
                        self.references.append(ref)
                        break

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

            if ref.target_type == 'agent':
                if ref.target_name not in plugin_data["agents"]:
                    ref.is_valid = False
                    ref.error_message = f"Agent '{ref.target_name}' not found in plugin '{ref.target_plugin}'"

            elif ref.target_type == 'command':
                if ref.target_name not in plugin_data["commands"]:
                    ref.is_valid = False
                    ref.error_message = f"Command '{ref.target_name}' not found in plugin '{ref.target_plugin}'"

            elif ref.target_type == 'skill':
                if ref.target_name not in plugin_data["skills"]:
                    ref.is_valid = False
                    ref.error_message = f"Skill '{ref.target_name}' not found in plugin '{ref.target_plugin}'"

    def generate_report(self, output_path: Path = None) -> str:
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
        valid_pct = (self.result.valid_references / self.result.total_references * 100) if self.result.total_references > 0 else 0
        lines.append(f"- **Total References:** {self.result.total_references}")
        lines.append(f"- **Valid References:** {self.result.valid_references} ({valid_pct:.1f}%)")
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
        type_counts = defaultdict(int)
        for ref in self.references:
            type_counts[ref.target_type] += 1

        for ref_type in sorted(type_counts.keys()):
            count = type_counts[ref_type]
            broken_count = sum(
                1 for ref in self.result.broken_references
                if ref.target_type == ref_type
            )
            status = "✅" if broken_count == 0 else "❌"
            lines.append(f"- **{ref_type}**: {count} total, {broken_count} broken {status}")
        lines.append("")

        # Broken references
        if self.result.broken_references:
            lines.append("## Broken References")
            lines.append("")
            lines.append(f"Found {len(self.result.broken_references)} broken references:")
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
                    lines.append(f"- File: `{ref.source_file}` (line {ref.source_line})")
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
                lines.append(f"- Fix `{ref.source_file}` line {ref.source_line}: {ref.error_message}")
            if len(self.result.broken_references) > 10:
                lines.append(f"- _(... and {len(self.result.broken_references) - 10} more)_")
            lines.append("")

        lines.append("### Best Practices")
        lines.append("")
        lines.append("- Always verify plugin/agent/command/skill names before referencing")
        lines.append("- Use backticks around technical names for clarity")
        lines.append("- Include plugin name when referencing agents/commands/skills from other plugins")
        lines.append("- Regularly run cross-reference validation during development")
        lines.append("")

        report = "\n".join(lines)

        # Write to file if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding='utf-8')
            print(f"✅ Validation report saved to: {output_path}")

        return report

    def export_json(self, output_path: Path):
        """Export validation results as JSON"""
        data = {
            "summary": {
                "total_references": self.result.total_references,
                "valid_references": self.result.valid_references,
                "broken_count": len(self.result.broken_references),
                "plugins_indexed": len(self.result.plugin_index)
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
                    "context": ref.context
                }
                for ref in self.result.broken_references
            ],
            "plugin_index": self.result.plugin_index
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
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
        help="Path to plugins directory (default: ./plugins)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/xref-validation.md"),
        help="Output file for validation report (default: reports/xref-validation.md)"
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Export validation results as JSON"
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
