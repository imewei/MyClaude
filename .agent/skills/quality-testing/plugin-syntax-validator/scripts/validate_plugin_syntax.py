#!/usr/bin/env python3
"""
Comprehensive plugin syntax validator for Claude Code plugins.

Validates:
1. Agent references (plugin:agent format)
2. Skill references (plugin:skill format)
3. Command references
4. File existence
5. Syntax correctness

Usage:
    python validate_plugin_syntax.py [--plugin NAME] [--fix] [--report] [--verbose]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ValidationIssue:
    """Represents a validation issue found."""
    severity: str  # ERROR, WARNING, INFO
    category: str  # syntax, reference, structure
    file: str
    line: int
    message: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False


class PluginValidator:
    """Validates plugin syntax and structure."""

    def __init__(self, plugins_dir: Path, verbose: bool = False):
        self.plugins_dir = plugins_dir
        self.verbose = verbose
        self.issues: List[ValidationIssue] = []
        self.stats = {
            'plugins_scanned': 0,
            'files_scanned': 0,
            'agent_refs_checked': 0,
            'skill_refs_checked': 0,
            'errors': 0,
            'warnings': 0,
            'info': 0
        }

        # Build agent and skill maps
        self.agent_map = self._build_agent_map()
        self.skill_map = self._build_skill_map()

    def _build_agent_map(self) -> Dict[str, Dict[str, Path]]:
        """Build map of all available agents: {plugin: {agent_name: file_path}}"""
        agent_map = defaultdict(dict)

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            agents_dir = plugin_dir / "agents"
            if not agents_dir.exists():
                continue

            for agent_file in agents_dir.glob("*.md"):
                agent_name = agent_file.stem
                agent_map[plugin_dir.name][agent_name] = agent_file

        return dict(agent_map)

    def _build_skill_map(self) -> Dict[str, Dict[str, Path]]:
        """Build map of all available skills: {plugin: {skill_name: dir_path}}"""
        skill_map = defaultdict(dict)

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            skills_dir = plugin_dir / "skills"
            if not skills_dir.exists():
                continue

            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    skill_map[plugin_dir.name][skill_dir.name] = skill_dir

        return dict(skill_map)

    def validate_agent_reference(self, ref: str, file: Path, line_num: int) -> None:
        """Validate an agent reference (plugin:agent format)."""
        self.stats['agent_refs_checked'] += 1

        # Check for double colon
        if "::" in ref:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="syntax",
                file=str(file),
                line=line_num,
                message=f"Double colon (::) in agent reference: '{ref}'",
                suggestion=f"Change to: {ref.replace('::', ':')}",
                auto_fixable=True
            ))
            return

        # Check for colon (namespace)
        if ":" not in ref:
            self.issues.append(ValidationIssue(
                severity="WARNING",
                category="syntax",
                file=str(file),
                line=line_num,
                message=f"Missing namespace in agent reference: '{ref}'",
                suggestion=f"Add plugin namespace: 'plugin-name:{ref}'",
                auto_fixable=False  # Need to know which plugin
            ))
            return

        # Parse plugin:agent
        parts = ref.split(":", 1)
        if len(parts) != 2:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="syntax",
                file=str(file),
                line=line_num,
                message=f"Invalid agent reference format: '{ref}'",
                suggestion="Use format: plugin-name:agent-name"
            ))
            return

        plugin_name, agent_name = parts

        # Check if agent exists
        if plugin_name not in self.agent_map:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="reference",
                file=str(file),
                line=line_num,
                message=f"Plugin not found: '{plugin_name}' in reference '{ref}'",
                suggestion=f"Available plugins: {', '.join(sorted(self.agent_map.keys()))}"
            ))
            return

        if agent_name not in self.agent_map[plugin_name]:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="reference",
                file=str(file),
                line=line_num,
                message=f"Agent not found: '{agent_name}' in plugin '{plugin_name}'",
                suggestion=f"Available agents in {plugin_name}: {', '.join(sorted(self.agent_map[plugin_name].keys()))}"
            ))

    def validate_skill_reference(self, ref: str, file: Path, line_num: int) -> None:
        """Validate a skill reference (plugin:skill format)."""
        self.stats['skill_refs_checked'] += 1

        # Similar validation to agents
        if "::" in ref:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="syntax",
                file=str(file),
                line=line_num,
                message=f"Double colon (::) in skill reference: '{ref}'",
                suggestion=f"Change to: {ref.replace('::', ':')}",
                auto_fixable=True
            ))
            return

        if ":" not in ref:
            self.issues.append(ValidationIssue(
                severity="WARNING",
                category="syntax",
                file=str(file),
                line=line_num,
                message=f"Missing namespace in skill reference: '{ref}'",
                suggestion=f"Add plugin namespace: 'plugin-name:{ref}'"
            ))
            return

        parts = ref.split(":", 1)
        if len(parts) != 2:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="syntax",
                file=str(file),
                line=line_num,
                message=f"Invalid skill reference format: '{ref}'",
                suggestion="Use format: plugin-name:skill-name"
            ))
            return

        plugin_name, skill_name = parts

        if plugin_name not in self.skill_map:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="reference",
                file=str(file),
                line=line_num,
                message=f"Plugin not found: '{plugin_name}' in skill reference '{ref}'"
            ))
            return

        if skill_name not in self.skill_map[plugin_name]:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="reference",
                file=str(file),
                line=line_num,
                message=f"Skill not found: '{skill_name}' in plugin '{plugin_name}'",
                suggestion=f"Available skills in {plugin_name}: {', '.join(sorted(self.skill_map[plugin_name].keys()))}"
            ))

    def validate_command_file(self, file: Path) -> None:
        """Validate a single command markdown file."""
        self.stats['files_scanned'] += 1

        try:
            content = file.read_text()
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                # Find agent references
                agent_matches = re.findall(r'subagent_type\s*=\s*["\']([^"\']+)["\']', line)
                for agent_ref in agent_matches:
                    self.validate_agent_reference(agent_ref, file, line_num)

                # Find skill references (in Skill tool calls or mentions)
                skill_matches = re.findall(r'(?:skill|Skill)\s*[:\(]\s*["\']([^"\']+)["\']', line)
                for skill_ref in skill_matches:
                    if ":" in skill_ref:  # Only validate namespaced references
                        self.validate_skill_reference(skill_ref, file, line_num)

        except Exception as e:
            self.issues.append(ValidationIssue(
                severity="ERROR",
                category="structure",
                file=str(file),
                line=0,
                message=f"Failed to read file: {e}"
            ))

    def validate_plugin(self, plugin_name: str) -> None:
        """Validate all command files in a plugin."""
        plugin_dir = self.plugins_dir / plugin_name

        if not plugin_dir.exists():
            print(f"❌ Plugin not found: {plugin_name}")
            return

        self.stats['plugins_scanned'] += 1

        # Validate commands
        commands_dir = plugin_dir / "commands"
        if commands_dir.exists():
            for cmd_file in commands_dir.glob("*.md"):
                self.validate_command_file(cmd_file)

        # Validate agents (check for broken references within agents)
        agents_dir = plugin_dir / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.md"):
                self.validate_command_file(agent_file)

    def validate_all_plugins(self) -> None:
        """Validate all plugins in the plugins directory."""
        for plugin_dir in sorted(self.plugins_dir.iterdir()):
            if plugin_dir.is_dir() and not plugin_dir.name.startswith('.'):
                self.validate_plugin(plugin_dir.name)

    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        errors = [i for i in self.issues if i.severity == "ERROR"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]
        infos = [i for i in self.issues if i.severity == "INFO"]

        self.stats['errors'] = len(errors)
        self.stats['warnings'] = len(warnings)
        self.stats['info'] = len(infos)

        report = []
        report.append("\n" + "="*80)
        report.append("PLUGIN SYNTAX VALIDATION REPORT")
        report.append("="*80)
        report.append(f"\n📊 Statistics:")
        report.append(f"  Plugins scanned:      {self.stats['plugins_scanned']}")
        report.append(f"  Files scanned:        {self.stats['files_scanned']}")
        report.append(f"  Agent refs checked:   {self.stats['agent_refs_checked']}")
        report.append(f"  Skill refs checked:   {self.stats['skill_refs_checked']}")
        report.append(f"\n📈 Results:")
        report.append(f"  🔴 Errors:   {self.stats['errors']}")
        report.append(f"  🟡 Warnings: {self.stats['warnings']}")
        report.append(f"  🔵 Info:     {self.stats['info']}")

        if not self.issues:
            report.append(f"\n✅ All validations passed!")
            report.append("="*80 + "\n")
            return "\n".join(report)

        # Report errors
        if errors:
            report.append(f"\n{'─'*80}")
            report.append("🔴 ERRORS (Must Fix)")
            report.append(f"{'─'*80}")
            for issue in errors:
                file_short = Path(issue.file).relative_to(self.plugins_dir)
                report.append(f"\n  [{issue.category.upper()}] {file_short}:{issue.line}")
                report.append(f"  {issue.message}")
                if issue.suggestion:
                    report.append(f"  💡 Suggestion: {issue.suggestion}")

        # Report warnings
        if warnings:
            report.append(f"\n{'─'*80}")
            report.append("🟡 WARNINGS (Should Fix)")
            report.append(f"{'─'*80}")
            for issue in warnings:
                file_short = Path(issue.file).relative_to(self.plugins_dir)
                report.append(f"\n  [{issue.category.upper()}] {file_short}:{issue.line}")
                report.append(f"  {issue.message}")
                if issue.suggestion:
                    report.append(f"  💡 Suggestion: {issue.suggestion}")

        report.append("\n" + "="*80)

        if self.stats['errors'] > 0:
            report.append(f"⚠️  Found {self.stats['errors']} error(s) that must be fixed.")
        else:
            report.append("✅ No critical errors found.")

        report.append("="*80 + "\n")

        return "\n".join(report)

    def auto_fix(self) -> int:
        """Automatically fix auto-fixable issues."""
        fixed_count = 0

        # Group issues by file
        issues_by_file = defaultdict(list)
        for issue in self.issues:
            if issue.auto_fixable:
                issues_by_file[issue.file].append(issue)

        for file_path, file_issues in issues_by_file.items():
            try:
                content = Path(file_path).read_text()

                # Fix double colons
                content = re.sub(
                    r'subagent_type\s*=\s*["\']([^:]+)::([^"\']+)["\']',
                    r'subagent_type="\1:\2"',
                    content
                )

                Path(file_path).write_text(content)
                fixed_count += len(file_issues)

                if self.verbose:
                    print(f"✅ Fixed {len(file_issues)} issue(s) in {file_path}")

            except Exception as e:
                print(f"❌ Failed to fix {file_path}: {e}")

        return fixed_count


def main():
    parser = argparse.ArgumentParser(
        description="Validate plugin syntax for Claude Code plugins"
    )
    parser.add_argument(
        "--plugin",
        help="Validate specific plugin only"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix auto-fixable issues"
    )
    parser.add_argument(
        "--report",
        help="Save report to file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=Path.cwd() / "plugins",
        help="Path to plugins directory"
    )

    args = parser.parse_args()

    if not args.plugins_dir.exists():
        print(f"❌ Plugins directory not found: {args.plugins_dir}")
        sys.exit(1)

    validator = PluginValidator(args.plugins_dir, verbose=args.verbose)

    # Run validation
    if args.plugin:
        print(f"🔍 Validating plugin: {args.plugin}")
        validator.validate_plugin(args.plugin)
    else:
        print(f"🔍 Validating all plugins in {args.plugins_dir}")
        validator.validate_all_plugins()

    # Auto-fix if requested
    if args.fix:
        print(f"\n🔧 Attempting to auto-fix issues...")
        fixed = validator.auto_fix()
        print(f"✅ Fixed {fixed} issue(s)")

        # Re-run validation
        print(f"\n🔍 Re-validating...")
        validator = PluginValidator(args.plugins_dir, verbose=args.verbose)
        if args.plugin:
            validator.validate_plugin(args.plugin)
        else:
            validator.validate_all_plugins()

    # Generate report
    report = validator.generate_report()
    print(report)

    # Save report if requested
    if args.report:
        Path(args.report).write_text(report)
        print(f"📄 Report saved to: {args.report}")

    # Exit with error code if errors found
    sys.exit(1 if validator.stats['errors'] > 0 else 0)


if __name__ == "__main__":
    main()
