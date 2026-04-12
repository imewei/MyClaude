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
import sys
from pathlib import Path
from typing import Dict, List, Optional
import re
from dataclasses import dataclass, field

# Allow ad-hoc `python tools/validation/plugin_review_script.py` CLI runs by
# adding the repo root to sys.path before resolving the `tools` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.common.models import ValidationIssue, ValidationResult  # noqa: E402
from tools.validation.metadata_validator import MetadataValidator  # noqa: E402


@dataclass
class ReviewReport:
    """Complete review report for a plugin"""

    plugin_name: str
    plugin_path: Path
    _result: ValidationResult = field(init=False)
    successes: List[str] = field(default_factory=list)

    # Severity mapping: review severities → shared model severities
    _SEVERITY_MAP: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._result = ValidationResult(
            plugin_name=self.plugin_name,
            plugin_path=self.plugin_path,
        )
        self._SEVERITY_MAP = {
            "critical": "critical",
            "high": "error",
            "medium": "warning",
            "low": "info",
        }

    def add_issue(
        self, section: str, severity: str, message: str, details: Optional[str] = None
    ) -> None:
        """Add an issue to the report"""
        mapped_severity = self._SEVERITY_MAP.get(severity, "warning")
        suggestion = details
        if mapped_severity in ("critical", "error"):
            self._result.add_error(
                field=section, message=message, suggestion=suggestion
            )
        elif mapped_severity == "warning":
            self._result.add_warning(
                field=section, message=message, suggestion=suggestion
            )
        else:
            self._result.add_info(field=section, message=message)

    def add_warning(self, message: str) -> None:
        """Add a warning to the report"""
        self._result.add_warning(field="general", message=message)

    def add_success(self, message: str) -> None:
        """Add a success to the report"""
        self.successes.append(message)

    @property
    def issues(self) -> list[ValidationIssue]:
        return self._result.issues

    @property
    def warnings(self) -> list[ValidationIssue]:
        return self._result.warnings

    def get_issue_count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity (using original review severity names)"""
        # Map back from shared severities to review severities for report compatibility
        reverse_map = {
            "critical": "critical",
            "error": "high",
            "warning": "medium",
            "info": "low",
        }
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in self._result.issues:
            review_severity = reverse_map.get(issue.severity, "low")
            counts[review_severity] = counts.get(review_severity, 0) + 1
        return counts


class PluginReviewer:
    """Main plugin review automation class"""

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
        """Review plugin.json structure and completeness using MetadataValidator"""
        plugin_json_path = plugin_path / ".claude-plugin" / "plugin.json"

        if not plugin_json_path.exists():
            report.add_issue(
                "plugin.json",
                "critical",
                "plugin.json file not found",
                f"Expected at: {plugin_json_path}",
            )
            return

        validator = MetadataValidator()
        result = validator.validate_plugin_json(plugin_path)

        for error in result.errors:
            report.add_issue(
                f"plugin.json/{error.field}", "high", error.message, error.suggestion
            )

        for warning in result.warnings:
            report.add_issue(
                f"plugin.json/{warning.field}",
                "medium",
                warning.message,
                warning.suggestion,
            )

        if result.is_valid:
            report.add_success("plugin.json structure validated successfully")

    def _review_agents(self, plugin_path: Path, report: ReviewReport):
        """Review agent documentation files"""
        agents_dir = plugin_path / "agents"

        if not agents_dir.exists():
            report.add_warning("No agents directory found")
            return

        # Get agent names from plugin.json
        plugin_json_path = plugin_path / ".claude-plugin" / "plugin.json"
        if plugin_json_path.exists():
            try:
                with open(plugin_json_path, "r", encoding="utf-8") as f:
                    plugin_data = json.load(f)
                    agents = plugin_data.get("agents", [])

                    for agent in agents:
                        # Entries may be file-path strings or objects
                        agent_name = (
                            Path(agent).stem
                            if isinstance(agent, str)
                            else agent.get("name")
                        )
                        if agent_name:
                            agent_file = agents_dir / f"{agent_name}.md"
                            if not agent_file.exists():
                                report.add_issue(
                                    "agents",
                                    "high",
                                    f"Agent documentation file not found: {agent_name}.md",
                                )
                            else:
                                self._validate_agent_file(
                                    agent_file, agent_name, report
                                )
            except Exception as e:
                report.add_warning(f"Failed to validate agent files: {e}")

    def _validate_agent_file(
        self, agent_file: Path, agent_name: str, report: ReviewReport
    ):
        """Validate individual agent markdown file"""
        try:
            with open(agent_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Check minimum content length
                if len(content) < 100:
                    report.add_issue(
                        "agents",
                        "medium",
                        f"Agent file too short: {agent_name}.md",
                        "Should contain comprehensive documentation",
                    )

                # Check for required sections (basic markdown headings)
                if not re.search(r"^#{1,3}\s+", content, re.MULTILINE):
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
        plugin_json_path = plugin_path / ".claude-plugin" / "plugin.json"
        if plugin_json_path.exists():
            try:
                with open(plugin_json_path, "r", encoding="utf-8") as f:
                    plugin_data = json.load(f)
                    commands = plugin_data.get("commands", [])

                    for command in commands:
                        command_name = (
                            Path(command).stem
                            if isinstance(command, str)
                            else command.get("name", "").lstrip("/")
                        )
                        if command_name:
                            command_file = commands_dir / f"{command_name}.md"
                            if not command_file.exists():
                                report.add_issue(
                                    "commands",
                                    "high",
                                    f"Command documentation file not found: {command_name}.md",
                                )
                            else:
                                self._validate_command_file(
                                    command_file, command_name, report
                                )
            except Exception as e:
                report.add_warning(f"Failed to validate command files: {e}")

    def _validate_command_file(
        self, command_file: Path, command_name: str, report: ReviewReport
    ):
        """Validate individual command markdown file"""
        try:
            with open(command_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Check minimum content length
                if len(content) < 100:
                    report.add_issue(
                        "commands",
                        "medium",
                        f"Command file too short: {command_name}.md",
                        "Should contain usage examples and documentation",
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
        plugin_json_path = plugin_path / ".claude-plugin" / "plugin.json"
        if plugin_json_path.exists():
            try:
                with open(plugin_json_path, "r", encoding="utf-8") as f:
                    plugin_data = json.load(f)
                    skills = plugin_data.get("skills", [])

                    for skill in skills:
                        skill_name = (
                            Path(skill).stem
                            if isinstance(skill, str)
                            else skill.get("name")
                        )
                        if skill_name:
                            # Skills live in directories with SKILL.md
                            skill_file = skills_dir / skill_name / "SKILL.md"
                            if not skill_file.exists():
                                report.add_issue(
                                    "skills",
                                    "medium",
                                    f"Skill documentation file not found: {skill_name}/SKILL.md",
                                )
                            else:
                                self._validate_skill_file(
                                    skill_file, skill_name, report
                                )
            except Exception as e:
                report.add_warning(f"Failed to validate skill files: {e}")

    def _validate_skill_file(
        self, skill_file: Path, skill_name: str, report: ReviewReport
    ):
        """Validate individual skill markdown file"""
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Check minimum content length
                if len(content) < 100:
                    report.add_issue(
                        "skills",
                        "low",
                        f"Skill file too short: {skill_name}.md",
                        "Should contain patterns and examples",
                    )

                # Check for code blocks (skills should have examples)
                if "```" not in content:
                    report.add_warning(f"Skill {skill_name}.md: No code examples found")
        except Exception as e:
            report.add_warning(f"Failed to read skill file {skill_name}.md: {e}")

    def _review_readme(self, plugin_path: Path, report: ReviewReport):
        """Review README.md completeness"""
        readme_path = plugin_path / "README.md"

        if not readme_path.exists():
            report.add_issue("README", "high", "README.md not found")
            return

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check minimum length
            if len(content) < 500:
                report.add_issue(
                    "README",
                    "medium",
                    "README.md is too short (< 500 chars)",
                    "Should provide comprehensive documentation",
                )

            # Check for required sections
            required_sections = [
                (r"#.*overview", "Overview or similar section"),
                (r"#.*agent", "Agents section"),
                (r"#.*command", "Commands section"),
                (r"#.*skill", "Skills section"),
            ]

            for pattern, description in required_sections:
                if not re.search(pattern, content, re.IGNORECASE):
                    report.add_warning(f"README.md: Missing {description}")

            # Check for code examples
            code_blocks = re.findall(r"```", content)
            if len(code_blocks) < 4:  # At least 2 code blocks (opening and closing)
                report.add_warning("README.md: Insufficient code examples")

            report.add_success("README.md found and contains documentation")

        except Exception as e:
            report.add_issue("README", "medium", "Failed to read README.md", str(e))

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
            report.add_warning(f"Missing recommended directories: {', '.join(missing)}")

        if len(found_dirs) > 0:
            report.add_success(f"Found directories: {', '.join(found_dirs)}")

    def _review_cross_references(self, plugin_path: Path, report: ReviewReport):
        """Review cross-references within the plugin"""
        readme_path = plugin_path / "README.md"

        if not readme_path.exists():
            return

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()

            # Find broken local file links
            local_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", readme_content)

            for link_text, link_url in local_links:
                # Skip external URLs
                if link_url.startswith(("http://", "https://", "#")):
                    continue

                # Check if local file exists
                link_path = plugin_path / link_url
                if not link_path.exists():
                    report.add_issue(
                        "cross-references",
                        "medium",
                        f"Broken link in README: {link_url}",
                        f"Link text: '{link_text}'",
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
        # Count standalone warnings (add_warning calls, field="general")
        standalone_warnings = [i for i in report.issues if i.field == "general"]
        lines.append(f"- **Warnings:** {len(standalone_warnings)}")
        lines.append(f"- **Successes:** {len(report.successes)}\n")

        # Reverse severity map for display
        display_severity = {
            "critical": "critical",
            "error": "high",
            "warning": "medium",
            "info": "low",
        }
        severity_emoji_map = {
            "critical": "\U0001f534",
            "error": "\U0001f7e0",
            "warning": "\U0001f7e1",
            "info": "\U0001f535",
        }

        # Issues by section (exclude standalone warnings)
        section_issues = [i for i in report.issues if i.field != "general"]
        if section_issues:
            lines.append("## Issues Found\n")

            # Group issues by field (section)
            issues_by_section: Dict[str, list] = {}
            for issue in section_issues:
                if issue.field not in issues_by_section:
                    issues_by_section[issue.field] = []
                issues_by_section[issue.field].append(issue)

            for section, issues in sorted(issues_by_section.items()):
                lines.append(f"### {section}\n")
                for issue in issues:
                    emoji = severity_emoji_map.get(issue.severity, "\u26aa")
                    display_sev = display_severity.get(
                        issue.severity, issue.severity
                    ).upper()

                    lines.append(f"{emoji} **{display_sev}**: {issue.message}")
                    if issue.suggestion:
                        lines.append(f"   - {issue.suggestion}")
                    lines.append("")

        # Standalone warnings
        if standalone_warnings:
            lines.append("## Warnings\n")
            for warning in standalone_warnings:
                lines.append(f"\u26a0\ufe0f  {warning.message}")
            lines.append("")

        # Successes
        if report.successes:
            lines.append("## Validation Successes\n")
            for success in report.successes:
                lines.append(f"✅ {success}")
            lines.append("")

        # Overall assessment
        lines.append("## Overall Assessment\n")
        if issue_counts["critical"] > 0:
            lines.append(
                "**Status:** ❌ CRITICAL ISSUES - Immediate attention required"
            )
        elif issue_counts["high"] > 0:
            lines.append("**Status:** ⚠️  HIGH PRIORITY ISSUES - Should be addressed")
        elif issue_counts["medium"] > 0:
            lines.append("**Status:** 🔸 MEDIUM PRIORITY ISSUES - Recommended to fix")
        elif issue_counts["low"] > 0:
            lines.append("**Status:** ✓ MINOR ISSUES - Optional improvements")
        else:
            lines.append("**Status:** ✅ EXCELLENT - No issues found")

        return "\n".join(lines)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python plugin_review_script.py <plugin-name> [plugins-root]")
        print("\nExample:")
        print("  python plugin_review_script.py julia-development")
        print("  python plugin_review_script.py julia-development /path/to/plugins")
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

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print(f"\n\nReport saved to: {output_file}")

    # Exit with appropriate code
    issue_counts = report.get_issue_count_by_severity()
    if issue_counts["critical"] > 0:
        sys.exit(2)
    elif issue_counts["high"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
