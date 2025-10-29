#!/usr/bin/env python3
"""
Plugin Review Automation Script

Automates the comprehensive review of Claude Code plugins by validating:
- plugin.json structure and completeness
- Agent/command/skill file existence
- Documentation completeness
- Broken links and missing files
- Generates structured review reports in markdown
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import re
from dataclasses import dataclass, field


@dataclass
class ReviewIssue:
    """Represents a single review issue"""
    section: str
    severity: str  # critical, high, medium, low
    message: str
    details: Optional[str] = None


@dataclass
class ReviewReport:
    """Complete review report for a plugin"""
    plugin_name: str
    plugin_path: Path
    issues: List[ReviewIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    successes: List[str] = field(default_factory=list)

    def add_issue(self, section: str, severity: str, message: str, details: Optional[str] = None):
        """Add an issue to the report"""
        self.issues.append(ReviewIssue(section, severity, message, details))

    def add_warning(self, message: str):
        """Add a warning to the report"""
        self.warnings.append(message)

    def add_success(self, message: str):
        """Add a success to the report"""
        self.successes.append(message)

    def get_issue_count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity"""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts


class PluginReviewer:
    """Main plugin review automation class"""

    REQUIRED_PLUGIN_JSON_FIELDS = [
        "name", "version", "description", "author", "license"
    ]

    RECOMMENDED_PLUGIN_JSON_FIELDS = [
        "agents", "commands", "skills", "keywords", "category"
    ]

    REQUIRED_AGENT_FIELDS = ["name", "description", "status"]
    REQUIRED_COMMAND_FIELDS = ["name", "description", "status"]
    REQUIRED_SKILL_FIELDS = ["name", "description"]

    def __init__(self, plugins_root: Path):
        """Initialize the plugin reviewer"""
        self.plugins_root = Path(plugins_root)

    def review_plugin(self, plugin_name: str) -> ReviewReport:
        """Perform comprehensive review of a plugin"""
        plugin_path = self.plugins_root / plugin_name
        report = ReviewReport(plugin_name=plugin_name, plugin_path=plugin_path)

        # Section 1: Plugin metadata (plugin.json)
        self._review_plugin_json(plugin_path, report)

        # Section 2: Agent documentation
        self._review_agents(plugin_path, report)

        # Section 3: Command documentation
        self._review_commands(plugin_path, report)

        # Section 4: Skill documentation
        self._review_skills(plugin_path, report)

        # Section 5: README completeness
        self._review_readme(plugin_path, report)

        # Section 6: File structure
        self._review_file_structure(plugin_path, report)

        # Section 7: Cross-references
        self._review_cross_references(plugin_path, report)

        return report

    def _review_plugin_json(self, plugin_path: Path, report: ReviewReport):
        """Review plugin.json structure and completeness"""
        plugin_json_path = plugin_path / "plugin.json"

        if not plugin_json_path.exists():
            report.add_issue(
                "plugin.json",
                "critical",
                "plugin.json file not found",
                f"Expected at: {plugin_json_path}"
            )
            return

        try:
            with open(plugin_json_path, 'r', encoding='utf-8') as f:
                plugin_data = json.load(f)
        except json.JSONDecodeError as e:
            report.add_issue(
                "plugin.json",
                "critical",
                "Invalid JSON syntax",
                str(e)
            )
            return
        except Exception as e:
            report.add_issue(
                "plugin.json",
                "critical",
                "Failed to read plugin.json",
                str(e)
            )
            return

        # Check required fields
        for field in self.REQUIRED_PLUGIN_JSON_FIELDS:
            if field not in plugin_data:
                report.add_issue(
                    "plugin.json",
                    "high",
                    f"Missing required field: {field}"
                )
            elif not plugin_data[field]:
                report.add_issue(
                    "plugin.json",
                    "medium",
                    f"Required field is empty: {field}"
                )

        # Check recommended fields
        for field in self.RECOMMENDED_PLUGIN_JSON_FIELDS:
            if field not in plugin_data:
                report.add_warning(f"Missing recommended field in plugin.json: {field}")
            elif isinstance(plugin_data.get(field), list) and len(plugin_data[field]) == 0:
                report.add_warning(f"Recommended field is empty: {field}")

        # Validate semantic versioning
        if "version" in plugin_data:
            version = plugin_data["version"]
            if not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$', version):
                report.add_issue(
                    "plugin.json",
                    "medium",
                    f"Invalid semantic versioning format: {version}",
                    "Expected format: MAJOR.MINOR.PATCH (e.g., 1.0.0)"
                )

        # Validate category
        if "category" in plugin_data:
            category = plugin_data["category"]
            valid_categories = [
                "scientific-computing", "development", "devops",
                "quality-engineering", "infrastructure"
            ]
            if category not in valid_categories:
                report.add_warning(
                    f"Non-standard category: {category}. "
                    f"Consider using: {', '.join(valid_categories)}"
                )

        # Validate agents structure
        if "agents" in plugin_data:
            for idx, agent in enumerate(plugin_data["agents"]):
                self._validate_agent_structure(agent, idx, report)

        # Validate commands structure
        if "commands" in plugin_data:
            for idx, command in enumerate(plugin_data["commands"]):
                self._validate_command_structure(command, idx, report)

        # Validate skills structure
        if "skills" in plugin_data:
            for idx, skill in enumerate(plugin_data["skills"]):
                self._validate_skill_structure(skill, idx, report)

        report.add_success("plugin.json successfully validated")

    def _validate_agent_structure(self, agent: Dict[str, Any], idx: int, report: ReviewReport):
        """Validate individual agent structure"""
        for field in self.REQUIRED_AGENT_FIELDS:
            if field not in agent:
                report.add_issue(
                    "plugin.json/agents",
                    "high",
                    f"Agent {idx}: Missing required field '{field}'",
                    f"Agent name: {agent.get('name', 'unknown')}"
                )

        # Check description length
        if "description" in agent and len(agent["description"]) < 20:
            report.add_issue(
                "plugin.json/agents",
                "medium",
                f"Agent {idx}: Description too short (< 20 chars)",
                f"Agent: {agent.get('name', 'unknown')}"
            )

    def _validate_command_structure(self, command: Dict[str, Any], idx: int, report: ReviewReport):
        """Validate individual command structure"""
        for field in self.REQUIRED_COMMAND_FIELDS:
            if field not in command:
                report.add_issue(
                    "plugin.json/commands",
                    "high",
                    f"Command {idx}: Missing required field '{field}'",
                    f"Command name: {command.get('name', 'unknown')}"
                )

        # Validate command name format (should start with /)
        if "name" in command:
            name = command["name"]
            if not name.startswith("/"):
                report.add_warning(
                    f"Command {idx} ({name}): Name should start with '/'"
                )

    def _validate_skill_structure(self, skill: Dict[str, Any], idx: int, report: ReviewReport):
        """Validate individual skill structure"""
        for field in self.REQUIRED_SKILL_FIELDS:
            if field not in skill:
                report.add_issue(
                    "plugin.json/skills",
                    "high",
                    f"Skill {idx}: Missing required field '{field}'",
                    f"Skill name: {skill.get('name', 'unknown')}"
                )

    def _review_agents(self, plugin_path: Path, report: ReviewReport):
        """Review agent documentation files"""
        agents_dir = plugin_path / "agents"

        if not agents_dir.exists():
            report.add_warning("No agents directory found")
            return

        # Get agent names from plugin.json
        plugin_json_path = plugin_path / "plugin.json"
        if plugin_json_path.exists():
            try:
                with open(plugin_json_path, 'r', encoding='utf-8') as f:
                    plugin_data = json.load(f)
                    agents = plugin_data.get("agents", [])

                    for agent in agents:
                        agent_name = agent.get("name")
                        if agent_name:
                            agent_file = agents_dir / f"{agent_name}.md"
                            if not agent_file.exists():
                                report.add_issue(
                                    "agents",
                                    "high",
                                    f"Agent documentation file not found: {agent_name}.md"
                                )
                            else:
                                self._validate_agent_file(agent_file, agent_name, report)
            except Exception as e:
                report.add_warning(f"Failed to validate agent files: {e}")

    def _validate_agent_file(self, agent_file: Path, agent_name: str, report: ReviewReport):
        """Validate individual agent markdown file"""
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Check minimum content length
                if len(content) < 100:
                    report.add_issue(
                        "agents",
                        "medium",
                        f"Agent file too short: {agent_name}.md",
                        "Should contain comprehensive documentation"
                    )

                # Check for required sections (basic markdown headings)
                if not re.search(r'^#{1,3}\s+', content, re.MULTILINE):
                    report.add_warning(
                        f"Agent {agent_name}.md: No markdown headings found"
                    )
        except Exception as e:
            report.add_warning(f"Failed to read agent file {agent_name}.md: {e}")

    def _review_commands(self, plugin_path: Path, report: ReviewReport):
        """Review command documentation files"""
        commands_dir = plugin_path / "commands"

        if not commands_dir.exists():
            report.add_warning("No commands directory found")
            return

        # Get command names from plugin.json
        plugin_json_path = plugin_path / "plugin.json"
        if plugin_json_path.exists():
            try:
                with open(plugin_json_path, 'r', encoding='utf-8') as f:
                    plugin_data = json.load(f)
                    commands = plugin_data.get("commands", [])

                    for command in commands:
                        command_name = command.get("name", "").lstrip("/")
                        if command_name:
                            command_file = commands_dir / f"{command_name}.md"
                            if not command_file.exists():
                                report.add_issue(
                                    "commands",
                                    "high",
                                    f"Command documentation file not found: {command_name}.md"
                                )
                            else:
                                self._validate_command_file(command_file, command_name, report)
            except Exception as e:
                report.add_warning(f"Failed to validate command files: {e}")

    def _validate_command_file(self, command_file: Path, command_name: str, report: ReviewReport):
        """Validate individual command markdown file"""
        try:
            with open(command_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Check minimum content length
                if len(content) < 100:
                    report.add_issue(
                        "commands",
                        "medium",
                        f"Command file too short: {command_name}.md",
                        "Should contain usage examples and documentation"
                    )

                # Check for code blocks (commands should have examples)
                if "```" not in content:
                    report.add_warning(
                        f"Command {command_name}.md: No code examples found"
                    )
        except Exception as e:
            report.add_warning(f"Failed to read command file {command_name}.md: {e}")

    def _review_skills(self, plugin_path: Path, report: ReviewReport):
        """Review skill documentation files"""
        skills_dir = plugin_path / "skills"

        if not skills_dir.exists():
            report.add_warning("No skills directory found")
            return

        # Get skill names from plugin.json
        plugin_json_path = plugin_path / "plugin.json"
        if plugin_json_path.exists():
            try:
                with open(plugin_json_path, 'r', encoding='utf-8') as f:
                    plugin_data = json.load(f)
                    skills = plugin_data.get("skills", [])

                    for skill in skills:
                        skill_name = skill.get("name")
                        if skill_name:
                            skill_file = skills_dir / f"{skill_name}.md"
                            if not skill_file.exists():
                                report.add_issue(
                                    "skills",
                                    "medium",
                                    f"Skill documentation file not found: {skill_name}.md"
                                )
                            else:
                                self._validate_skill_file(skill_file, skill_name, report)
            except Exception as e:
                report.add_warning(f"Failed to validate skill files: {e}")

    def _validate_skill_file(self, skill_file: Path, skill_name: str, report: ReviewReport):
        """Validate individual skill markdown file"""
        try:
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Check minimum content length
                if len(content) < 100:
                    report.add_issue(
                        "skills",
                        "low",
                        f"Skill file too short: {skill_name}.md",
                        "Should contain patterns and examples"
                    )

                # Check for code blocks (skills should have examples)
                if "```" not in content:
                    report.add_warning(
                        f"Skill {skill_name}.md: No code examples found"
                    )
        except Exception as e:
            report.add_warning(f"Failed to read skill file {skill_name}.md: {e}")

    def _review_readme(self, plugin_path: Path, report: ReviewReport):
        """Review README.md completeness"""
        readme_path = plugin_path / "README.md"

        if not readme_path.exists():
            report.add_issue(
                "README",
                "high",
                "README.md not found"
            )
            return

        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check minimum length
            if len(content) < 500:
                report.add_issue(
                    "README",
                    "medium",
                    "README.md is too short (< 500 chars)",
                    "Should provide comprehensive documentation"
                )

            # Check for required sections
            required_sections = [
                (r'#.*overview', "Overview or similar section"),
                (r'#.*agent', "Agents section"),
                (r'#.*command', "Commands section"),
                (r'#.*skill', "Skills section"),
            ]

            for pattern, description in required_sections:
                if not re.search(pattern, content, re.IGNORECASE):
                    report.add_warning(f"README.md: Missing {description}")

            # Check for code examples
            code_blocks = re.findall(r'```', content)
            if len(code_blocks) < 4:  # At least 2 code blocks (opening and closing)
                report.add_warning("README.md: Insufficient code examples")

            report.add_success("README.md found and contains documentation")

        except Exception as e:
            report.add_issue(
                "README",
                "medium",
                "Failed to read README.md",
                str(e)
            )

    def _review_file_structure(self, plugin_path: Path, report: ReviewReport):
        """Review overall file structure"""
        expected_dirs = ["agents", "commands", "skills"]
        found_dirs = []

        for dir_name in expected_dirs:
            dir_path = plugin_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                found_dirs.append(dir_name)

        if len(found_dirs) < len(expected_dirs):
            missing = set(expected_dirs) - set(found_dirs)
            report.add_warning(
                f"Missing recommended directories: {', '.join(missing)}"
            )

        if len(found_dirs) > 0:
            report.add_success(f"Found directories: {', '.join(found_dirs)}")

    def _review_cross_references(self, plugin_path: Path, report: ReviewReport):
        """Review cross-references within the plugin"""
        readme_path = plugin_path / "README.md"

        if not readme_path.exists():
            return

        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()

            # Find broken local file links
            local_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', readme_content)

            for link_text, link_url in local_links:
                # Skip external URLs
                if link_url.startswith(('http://', 'https://', '#')):
                    continue

                # Check if local file exists
                link_path = plugin_path / link_url
                if not link_path.exists():
                    report.add_issue(
                        "cross-references",
                        "medium",
                        f"Broken link in README: {link_url}",
                        f"Link text: '{link_text}'"
                    )

        except Exception as e:
            report.add_warning(f"Failed to check cross-references: {e}")

    def generate_markdown_report(self, report: ReviewReport) -> str:
        """Generate a markdown formatted report"""
        lines = []
        lines.append(f"# Plugin Review Report: {report.plugin_name}\n")
        lines.append(f"**Plugin Path:** `{report.plugin_path}`\n")

        # Summary
        issue_counts = report.get_issue_count_by_severity()
        total_issues = sum(issue_counts.values())

        lines.append("## Summary\n")
        lines.append(f"- **Total Issues:** {total_issues}")
        lines.append(f"  - Critical: {issue_counts['critical']}")
        lines.append(f"  - High: {issue_counts['high']}")
        lines.append(f"  - Medium: {issue_counts['medium']}")
        lines.append(f"  - Low: {issue_counts['low']}")
        lines.append(f"- **Warnings:** {len(report.warnings)}")
        lines.append(f"- **Successes:** {len(report.successes)}\n")

        # Issues by section
        if report.issues:
            lines.append("## Issues Found\n")

            # Group issues by section
            issues_by_section: Dict[str, List[ReviewIssue]] = {}
            for issue in report.issues:
                if issue.section not in issues_by_section:
                    issues_by_section[issue.section] = []
                issues_by_section[issue.section].append(issue)

            for section, issues in sorted(issues_by_section.items()):
                lines.append(f"### {section}\n")
                for issue in issues:
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🔵"
                    }.get(issue.severity, "⚪")

                    lines.append(f"{severity_emoji} **{issue.severity.upper()}**: {issue.message}")
                    if issue.details:
                        lines.append(f"   - {issue.details}")
                    lines.append("")

        # Warnings
        if report.warnings:
            lines.append("## Warnings\n")
            for warning in report.warnings:
                lines.append(f"⚠️  {warning}")
            lines.append("")

        # Successes
        if report.successes:
            lines.append("## Validation Successes\n")
            for success in report.successes:
                lines.append(f"✅ {success}")
            lines.append("")

        # Overall assessment
        lines.append("## Overall Assessment\n")
        if issue_counts['critical'] > 0:
            lines.append("**Status:** ❌ CRITICAL ISSUES - Immediate attention required")
        elif issue_counts['high'] > 0:
            lines.append("**Status:** ⚠️  HIGH PRIORITY ISSUES - Should be addressed")
        elif issue_counts['medium'] > 0:
            lines.append("**Status:** 🔸 MEDIUM PRIORITY ISSUES - Recommended to fix")
        elif issue_counts['low'] > 0:
            lines.append("**Status:** ✓ MINOR ISSUES - Optional improvements")
        else:
            lines.append("**Status:** ✅ EXCELLENT - No issues found")

        return "\n".join(lines)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python plugin-review-script.py <plugin-name> [plugins-root]")
        print("\nExample:")
        print("  python plugin-review-script.py julia-development")
        print("  python plugin-review-script.py julia-development /path/to/plugins")
        sys.exit(1)

    plugin_name = sys.argv[1]
    plugins_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "plugins"

    if not plugins_root.exists():
        print(f"Error: Plugins directory not found: {plugins_root}")
        sys.exit(1)

    plugin_path = plugins_root / plugin_name
    if not plugin_path.exists():
        print(f"Error: Plugin directory not found: {plugin_path}")
        sys.exit(1)

    reviewer = PluginReviewer(plugins_root)
    report = reviewer.review_plugin(plugin_name)

    # Generate and print markdown report
    markdown_report = reviewer.generate_markdown_report(report)
    print(markdown_report)

    # Optionally save to file
    output_dir = Path("reviews")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{plugin_name}.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    print(f"\n\nReport saved to: {output_file}")

    # Exit with appropriate code
    issue_counts = report.get_issue_count_by_severity()
    if issue_counts['critical'] > 0:
        sys.exit(2)
    elif issue_counts['high'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
