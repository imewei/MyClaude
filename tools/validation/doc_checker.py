#!/usr/bin/env python3
"""
Documentation Completeness Checker

Validates plugin documentation for:
- README.md required sections
- Markdown formatting
- Code block syntax
- Cross-reference accuracy
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Allow ad-hoc `python tools/validation/doc_checker.py` CLI runs by adding
# the repo root to sys.path before resolving the `tools` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.common.models import ValidationResult  # noqa: E402


@dataclass
class DocCheckResult:
    """Results of documentation check, wrapping shared ValidationResult."""

    plugin_name: str
    plugin_path: Path
    _result: ValidationResult = field(init=False)
    stats: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._result = ValidationResult(
            plugin_name=self.plugin_name,
            plugin_path=self.plugin_path,
        )

    def add_issue(
        self,
        file: str,
        severity: str,
        message: str,
        line: Optional[int] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        """Add a documentation issue"""
        if severity == "error":
            self._result.add_error(
                field=file,
                message=message,
                suggestion=suggestion,
                file_path=file,
                line_number=line or 0,
            )
        elif severity == "warning":
            self._result.add_warning(
                field=file,
                message=message,
                suggestion=suggestion,
                file_path=file,
                line_number=line or 0,
            )
        else:
            self._result.add_info(field=file, message=message)

    @property
    def issues(self) -> list:
        return self._result.issues

    def get_issue_count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity"""
        counts = {"error": 0, "warning": 0, "info": 0}
        for issue in self._result.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts


class DocumentationChecker:
    """Documentation completeness checker"""

    # Required sections in README.md
    REQUIRED_README_SECTIONS = [
        (r"#.*overview", "Overview"),
        (r"#.*(agent|expert)", "Agents"),
        (r"#.*command", "Commands"),
        (r"#.*skill", "Skills"),
    ]

    # Recommended sections
    RECOMMENDED_README_SECTIONS = [
        (r"#.*(quick\s*start|getting\s*started)", "Quick Start / Getting Started"),
        (r"#.*(install|setup)", "Installation / Setup"),
        (r"#.*(example|usage)", "Examples / Usage"),
        (r"#.*(integration|workflow)", "Integration / Workflow"),
        (r"#.*license", "License"),
    ]

    # Code block language identifiers
    VALID_CODE_LANGUAGES = {
        "julia",
        "python",
        "javascript",
        "typescript",
        "bash",
        "shell",
        "sh",
        "rust",
        "cpp",
        "c++",
        "c",
        "go",
        "java",
        "json",
        "yaml",
        "toml",
        "html",
        "css",
        "sql",
        "r",
        "matlab",
        "latex",
        "markdown",
        "md",
        "diff",
        "plaintext",
        "text",
        "console",
        "terminal",
    }

    def __init__(self):
        """Initialize the documentation checker"""
        pass

    def check_plugin_documentation(self, plugin_path: Path) -> DocCheckResult:
        """Check all documentation for a plugin"""
        plugin_name = plugin_path.name
        result = DocCheckResult(plugin_name=plugin_name, plugin_path=plugin_path)

        # Check README.md
        readme_path = plugin_path / "README.md"
        if readme_path.exists():
            self._check_readme(readme_path, result)
        else:
            result.add_issue("README.md", "error", "README.md file not found")

        # Check agent documentation
        agents_dir = plugin_path / "agents"
        if agents_dir.exists() and agents_dir.is_dir():
            self._check_markdown_dir(agents_dir, "agent", result)

        # Check command documentation
        commands_dir = plugin_path / "commands"
        if commands_dir.exists() and commands_dir.is_dir():
            self._check_markdown_dir(commands_dir, "command", result)

        # Check skill documentation
        skills_dir = plugin_path / "skills"
        if skills_dir.exists() and skills_dir.is_dir():
            self._check_markdown_dir(skills_dir, "skill", result)

        # Calculate statistics
        result.stats = {
            "total_files_checked": self._count_markdown_files(plugin_path),
            "total_issues": len(result.issues),
            **result.get_issue_count_by_severity(),
        }

        return result

    def _check_markdown_dir(
        self, directory: Path, doc_type: str, result: DocCheckResult
    ):
        """Check all markdown files in a directory"""
        for file_path in directory.glob("*.md"):
            self._check_markdown_file(file_path, doc_type, result)

        return result

    def _check_readme(self, readme_path: Path, result: DocCheckResult):
        """Check README.md file"""
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            result.add_issue("README.md", "error", f"Failed to read file: {e}")
            return

        self._check_file_length(readme_path.name, content, 500, result)
        self._check_required_sections(
            readme_path.name, content, self.REQUIRED_README_SECTIONS, result
        )
        self._check_recommended_sections(
            readme_path.name, content, self.RECOMMENDED_README_SECTIONS, result
        )

        # Run common checks
        self._run_common_checks(readme_path, content, lines, result)

    def _check_markdown_file(
        self, file_path: Path, doc_type: str, result: DocCheckResult
    ):
        """Check individual markdown file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            result.add_issue(file_path.name, "error", f"Failed to read file: {e}")
            return

        # Check minimum content length
        min_length = 100 if doc_type == "skill" else 150
        self._check_file_length(file_path.name, content, min_length, result, doc_type)

        # Check for headings
        if not re.search(r"^#{1,3}\s+.+", content, re.MULTILINE):
            result.add_issue(
                file_path.name,
                "warning",
                f"No markdown headings found in {doc_type} documentation",
                suggestion="Use markdown headings to structure the documentation",
            )

        # Check for code examples
        if doc_type in ["command", "skill"]:
            if "```" not in content:
                result.add_issue(
                    file_path.name,
                    "warning",
                    f"No code examples found in {doc_type} documentation",
                    suggestion="Add code examples to illustrate usage",
                )

        # Run common checks
        self._run_common_checks(file_path, content, lines, result)

    def _run_common_checks(
        self, file_path: Path, content: str, lines: List[str], result: DocCheckResult
    ):
        """Run checks common to all markdown files"""
        self._check_markdown_formatting(file_path, content, lines, result)
        self._check_code_blocks(file_path, content, lines, result)
        self._check_links(file_path, content, lines, result)
        self._check_common_issues(file_path, content, lines, result)

    def _check_file_length(
        self,
        filename: str,
        content: str,
        min_length: int,
        result: DocCheckResult,
        doc_type: str = "README",
    ):
        """Check if file meets minimum length requirements"""
        if len(content) < min_length:
            msg_prefix = (
                f"{doc_type.capitalize()} documentation"
                if doc_type != "README"
                else "README"
            )
            result.add_issue(
                filename,
                "warning",
                f"{msg_prefix} is too short (< {min_length} characters)",
                suggestion="Add more comprehensive documentation",
            )

    def _check_required_sections(
        self,
        filename: str,
        content: str,
        sections: List[Tuple[str, str]],
        result: DocCheckResult,
    ):
        """Check for presence of required sections"""
        missing_sections = []
        for pattern, section_name in sections:
            if not re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                missing_sections.append(section_name)

        if missing_sections:
            result.add_issue(
                filename,
                "error",
                f"Missing required sections: {', '.join(missing_sections)}",
                suggestion="Add these sections to improve documentation structure",
            )

    def _check_recommended_sections(
        self,
        filename: str,
        content: str,
        sections: List[Tuple[str, str]],
        result: DocCheckResult,
    ):
        """Check for presence of recommended sections"""
        missing_recommended = []
        for pattern, section_name in sections:
            if not re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                missing_recommended.append(section_name)

        if missing_recommended:
            result.add_issue(
                filename,
                "warning",
                f"Missing recommended sections: {', '.join(missing_recommended)}",
                suggestion="Consider adding these sections for better documentation",
            )

    def _check_markdown_formatting(
        self, file_path: Path, content: str, lines: List[str], result: DocCheckResult
    ):
        """Check markdown formatting issues"""
        file_name = file_path.name

        # Check for consecutive blank lines (more than 2)
        if re.search(r"\n\n\n\n+", content):
            result.add_issue(
                file_name,
                "info",
                "Found excessive blank lines (3 or more consecutive)",
                suggestion="Use at most 2 consecutive blank lines",
            )

        # Check heading formatting
        for i, line in enumerate(lines, 1):
            # Check for heading without space after #
            if re.match(r"^#{1,6}[^#\s]", line):
                result.add_issue(
                    file_name,
                    "warning",
                    f"Line {i}: Heading should have space after #",
                    line=i,
                    suggestion=f"Change '{line}' to add space after #",
                )

            # Check for inconsistent list markers
            if re.match(r"^\s*[*+-]\s", line):
                # Could check for consistency across document
                pass

        # Check for trailing whitespace
        trailing_whitespace_lines = [
            i for i, line in enumerate(lines, 1) if line and line[-1] in " \t"
        ]
        if trailing_whitespace_lines and len(trailing_whitespace_lines) > 5:
            result.add_issue(
                file_name,
                "info",
                f"Found trailing whitespace on {len(trailing_whitespace_lines)} lines",
                suggestion="Remove trailing whitespace",
            )

    def _check_code_blocks(
        self, file_path: Path, content: str, lines: List[str], result: DocCheckResult
    ):
        """Check code block formatting and syntax"""
        file_name = file_path.name

        # Find all code blocks
        code_blocks = re.finditer(r"```(\w*)\n(.*?)```", content, re.DOTALL)

        code_block_count = 0
        for match in code_blocks:
            code_block_count += 1
            language = match.group(1).lower()
            code = match.group(2)

            # Check if language is specified
            if not language:
                result.add_issue(
                    file_name,
                    "warning",
                    f"Code block {code_block_count}: No language specified",
                    suggestion="Specify language for syntax highlighting (e.g., ```python)",
                )
            elif language not in self.VALID_CODE_LANGUAGES:
                result.add_issue(
                    file_name,
                    "info",
                    f"Code block {code_block_count}: Unusual language identifier '{language}'",
                    suggestion=f"Common languages: {', '.join(sorted(list(self.VALID_CODE_LANGUAGES)[:10]))}",
                )

            # Check for empty code blocks
            if not code.strip():
                result.add_issue(
                    file_name,
                    "warning",
                    f"Code block {code_block_count}: Empty code block",
                    suggestion="Remove empty code blocks or add content",
                )

        # Check for unclosed code blocks
        backtick_count = content.count("```")
        if backtick_count % 2 != 0:
            result.add_issue(
                file_name,
                "error",
                "Unclosed code block detected (odd number of ```)",
                suggestion="Ensure all code blocks are properly closed",
            )

    def _check_links(
        self, file_path: Path, content: str, lines: List[str], result: DocCheckResult
    ):
        """Check links and cross-references in body markdown AND YAML frontmatter.

        Body scan: markdown `[text](url)` links, with code-block stripping to
        avoid false positives on Python `f(x)` expressions and
        markdown-about-markdown examples.

        Frontmatter scan: any string value that looks like a file reference
        (starts with `${CLAUDE_PLUGIN_ROOT}` or `./` or `../`). Frontmatter
        fields like `documentation:` and `references:` historically held
        broken paths that the body-only scan missed.
        """
        file_name = file_path.name
        plugin_path = (
            file_path.parent
            if file_path.name == "README.md"
            else file_path.parent.parent
        )

        # --- Body scan ---
        # Strip fenced code blocks and inline code spans before scanning for
        # links. Otherwise Python expressions like `self.layers[-1](x)` and
        # markdown-about-markdown examples like `[text](url)` get false-flagged.
        # Order matters: triple-fenced first, then double-backtick spans (which
        # may contain literal single backticks), then single-backtick spans.
        body_for_links = content
        if content.startswith("---\n"):
            # Skip YAML frontmatter for the body scan — it's handled separately
            # below. Without this, frontmatter values that happen to contain
            # markdown-link-shaped text would be double-counted.
            end = content.find("\n---\n", 4)
            if end != -1:
                body_for_links = content[end + 5 :]

        stripped = re.sub(r"```.*?```", "", body_for_links, flags=re.DOTALL)
        stripped = re.sub(r"``[^\n]+?``", "", stripped)
        stripped = re.sub(r"`[^`\n]+`", "", stripped)

        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", stripped):
            link_text = match.group(1)
            link_url = match.group(2)
            self._validate_path_ref(
                file_name,
                link_url,
                f"Broken link: {link_url}",
                f"Link text: '{link_text}' - file not found",
                plugin_path,
                result,
            )

        # --- Frontmatter scan ---
        # Walk YAML frontmatter for string values that look like on-disk paths
        # (using ${CLAUDE_PLUGIN_ROOT}, ./, or ../ prefix). Catches the
        # historical pattern where a `documentation:` or `references:` field
        # listed unwritten doc paths.
        if content.startswith("---\n"):
            end = content.find("\n---\n", 4)
            if end != -1:
                frontmatter_text = content[4:end]
                # Match any quoted-or-unquoted string value that starts with
                # one of the path-reference prefixes. Captures up to a closing
                # quote, comma, or end-of-line.
                fm_path_re = re.compile(
                    r"""[:\-]\s*['"]?(\$\{CLAUDE_PLUGIN_ROOT\}/[^'",\n]+|\.\.?/[^'",\n\s]+)['"]?"""
                )
                for m in fm_path_re.finditer(frontmatter_text):
                    ref = m.group(1).strip()
                    self._validate_path_ref(
                        file_name,
                        ref,
                        f"Broken frontmatter reference: {ref}",
                        "frontmatter path does not exist on disk",
                        plugin_path,
                        result,
                    )

    def _validate_path_ref(
        self,
        file_name: str,
        ref: str,
        message: str,
        suggestion: str,
        plugin_path: Path,
        result: DocCheckResult,
    ) -> None:
        """Resolve a file reference and emit an error if the target is missing.

        Skips external URLs, mailto, and in-page anchors.
        """
        if ref.startswith(("http://", "https://", "mailto:", "#")):
            return

        # Resolve ${CLAUDE_PLUGIN_ROOT} to the plugin root before existence
        # check — this is how Claude Code resolves the variable at runtime.
        resolved = ref.replace("${CLAUDE_PLUGIN_ROOT}", str(plugin_path))

        if resolved.startswith("/"):
            link_path = Path(resolved)
        else:
            link_path = plugin_path / resolved

        if not link_path.exists():
            result.add_issue(file_name, "error", message, suggestion=suggestion)

    def _check_common_issues(
        self, file_path: Path, content: str, lines: List[str], result: DocCheckResult
    ):
        """Check for common documentation issues"""
        file_name = file_path.name

        # Check for TODO/FIXME markers
        todos = re.finditer(r"\b(TODO|FIXME|XXX|HACK)\b", content, re.IGNORECASE)
        todo_count = sum(1 for _ in todos)
        if todo_count > 0:
            result.add_issue(
                file_name,
                "info",
                f"Found {todo_count} TODO/FIXME markers",
                suggestion="Consider completing or removing TODO items before release",
            )

        # Check for placeholder text
        placeholders = [
            "lorem ipsum",
            "placeholder",
            "example text",
            "TODO:",
            "FIXME:",
            "coming soon",
            "under construction",
            "work in progress",
        ]
        for placeholder in placeholders:
            if placeholder.lower() in content.lower():
                result.add_issue(
                    file_name,
                    "warning",
                    f"Found placeholder text: '{placeholder}'",
                    suggestion="Replace placeholder text with actual content",
                )
                break

        # Check for very long lines (> 120 chars, excluding code blocks)
        in_code_block = False
        long_lines = []
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            elif not in_code_block and len(line) > 120:
                long_lines.append(i)

        if long_lines and len(long_lines) > 5:
            result.add_issue(
                file_name,
                "info",
                f"Found {len(long_lines)} very long lines (> 120 chars)",
                suggestion="Consider breaking long lines for better readability",
            )

    def _count_markdown_files(self, plugin_path: Path) -> int:
        """Count total markdown files in plugin"""
        count = 0
        for pattern in ["*.md", "**/*.md"]:
            count += len(list(plugin_path.glob(pattern)))
        return count

    def generate_report(self, result: DocCheckResult) -> str:
        """Generate documentation check report"""
        lines = []
        lines.append(f"# Documentation Check Report: {result.plugin_name}\n")
        lines.append(f"**Plugin Path:** `{result.plugin_path}`\n")

        # Summary
        issue_counts = result.get_issue_count_by_severity()
        lines.append("## Summary\n")
        lines.append(
            f"- **Files Checked:** {result.stats.get('total_files_checked', 0)}"
        )
        lines.append(f"- **Total Issues:** {result.stats.get('total_issues', 0)}")
        lines.append(f"  - Errors: {issue_counts['error']}")
        lines.append(f"  - Warnings: {issue_counts['warning']}")
        lines.append(f"  - Info: {issue_counts['info']}\n")

        # Status
        if issue_counts["error"] > 0:
            lines.append("**Status:** ❌ ERRORS FOUND - Documentation needs fixes\n")
        elif issue_counts["warning"] > 0:
            lines.append("**Status:** ⚠️  WARNINGS - Documentation could be improved\n")
        elif issue_counts["info"] > 0:
            lines.append("**Status:** ℹ️  MINOR ISSUES - Documentation is acceptable\n")
        else:
            lines.append("**Status:** ✅ EXCELLENT - Documentation is complete\n")

        # Issues by file
        if result.issues:
            lines.append("## Issues by File\n")

            # Group issues by file
            issues_by_file: Dict[str, list] = {}
            for issue in result.issues:
                file_key = issue.file_path or issue.field
                if file_key not in issues_by_file:
                    issues_by_file[file_key] = []
                issues_by_file[file_key].append(issue)

            for file_name, issues in sorted(issues_by_file.items()):
                lines.append(f"### {file_name}\n")

                # Group by severity
                for severity in ["error", "warning", "info"]:
                    severity_issues = [i for i in issues if i.severity == severity]
                    if not severity_issues:
                        continue

                    severity_emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[
                        severity
                    ]

                    for issue in severity_issues:
                        location = (
                            f" (line {issue.line_number})" if issue.line_number else ""
                        )
                        lines.append(f"{severity_emoji} {issue.message}{location}")
                        if issue.suggestion:
                            lines.append(f"   → {issue.suggestion}")
                        lines.append("")

        # Recommendations
        if result.issues:
            lines.append("## Recommendations\n")
            if issue_counts["error"] > 0:
                lines.append("1. Fix all errors before release")
                lines.append("2. Address warnings to improve documentation quality")
            elif issue_counts["warning"] > 0:
                lines.append("1. Address warnings to improve documentation quality")
                lines.append("2. Review info items for potential improvements")
            else:
                lines.append("1. Review info items for potential improvements")
            lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python doc_checker.py <plugin-path>")
        print("\nExample:")
        print("  python doc_checker.py plugins/science-suite")
        sys.exit(1)

    plugin_path = Path(sys.argv[1])

    if not plugin_path.exists():
        print(f"Error: Plugin path not found: {plugin_path}")
        sys.exit(1)

    checker = DocumentationChecker()
    result = checker.check_plugin_documentation(plugin_path)

    # Generate and print report
    report = checker.generate_report(result)
    print(report)

    # Exit with appropriate code
    issue_counts = result.get_issue_count_by_severity()
    if issue_counts["error"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
