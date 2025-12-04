"""
Base class for markdown report generation.

Provides common report formatting utilities used by multiple tools.
"""

from datetime import datetime
from typing import Optional

from tools.common.models import ValidationResult, ProfileMetric


class ReportGenerator:
    """Base class for generating markdown reports.

    Provides common formatting methods and templates.
    """

    def __init__(self, title: str = "Report") -> None:
        self.title = title
        self.lines: list[str] = []

    def clear(self) -> None:
        """Clear the report content."""
        self.lines = []

    def add_header(
        self,
        title: Optional[str] = None,
        version: Optional[str] = None,
        include_timestamp: bool = True,
    ) -> None:
        """Add report header with title and metadata."""
        self.lines.append(f"# {title or self.title}")
        self.lines.append("")
        if version:
            self.lines.append(f"**Version:** {version}")
        if include_timestamp:
            self.lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.lines.append("")

    def add_section(self, title: str, level: int = 2) -> None:
        """Add a section header."""
        prefix = "#" * level
        self.lines.append(f"{prefix} {title}")
        self.lines.append("")

    def add_text(self, text: str) -> None:
        """Add plain text."""
        self.lines.append(text)
        self.lines.append("")

    def add_bullet(self, text: str) -> None:
        """Add a bullet point."""
        self.lines.append(f"- {text}")

    def add_numbered(self, number: int, text: str) -> None:
        """Add a numbered item."""
        self.lines.append(f"{number}. {text}")

    def add_blank_line(self) -> None:
        """Add a blank line."""
        self.lines.append("")

    def add_separator(self, char: str = "-", width: int = 80) -> None:
        """Add a horizontal separator."""
        self.lines.append(char * width)
        self.lines.append("")

    def add_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        alignments: Optional[list[str]] = None,
    ) -> None:
        """Add a markdown table.

        Args:
            headers: Column headers
            rows: Table data rows
            alignments: Optional alignment per column ('left', 'center', 'right')
        """
        # Header row
        self.lines.append("| " + " | ".join(headers) + " |")

        # Separator row
        if alignments:
            separators = []
            for i, align in enumerate(alignments):
                if align == "center":
                    separators.append(":---:")
                elif align == "right":
                    separators.append("---:")
                else:
                    separators.append("---")
            # Fill remaining columns with default
            while len(separators) < len(headers):
                separators.append("---")
            self.lines.append("| " + " | ".join(separators) + " |")
        else:
            self.lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Data rows
        for row in rows:
            self.lines.append("| " + " | ".join(row) + " |")

        self.lines.append("")

    def add_validation_summary(self, result: ValidationResult) -> None:
        """Add a validation result summary."""
        counts = result.get_issue_count_by_severity()

        self.add_section("Validation Summary")
        self.add_bullet(f"**Plugin:** {result.plugin_name}")
        self.add_bullet(f"**Status:** {'\\u2705 PASS' if result.is_valid else '\\u274c FAIL'}")
        self.add_bullet(f"**Errors:** {counts['critical'] + counts['error']}")
        self.add_bullet(f"**Warnings:** {counts['warning']}")
        self.add_bullet(f"**Info:** {counts['info']}")
        self.add_blank_line()

    def add_issues_table(self, result: ValidationResult) -> None:
        """Add a table of validation issues."""
        if not result.issues:
            self.add_text("No issues found.")
            return

        headers = ["Severity", "Field", "Message", "Suggestion"]
        rows = []

        for issue in result.issues:
            rows.append(
                [
                    f"{issue.emoji} {issue.severity}",
                    issue.field,
                    issue.message,
                    issue.suggestion or "-",
                ]
            )

        self.add_table(headers, rows)

    def add_metrics_table(self, metrics: list[ProfileMetric]) -> None:
        """Add a table of profiling metrics."""
        if not metrics:
            self.add_text("No metrics collected.")
            return

        headers = ["Component", "Time (ms)", "Status", "Details"]
        rows = []

        for metric in metrics:
            rows.append(
                [
                    metric.name,
                    f"{metric.duration_ms:.2f}",
                    metric.status_emoji,
                    metric.details,
                ]
            )

        self.add_table(headers, rows, ["left", "right", "center", "left"])

    def add_code_block(self, code: str, language: str = "") -> None:
        """Add a fenced code block."""
        self.lines.append(f"```{language}")
        self.lines.append(code)
        self.lines.append("```")
        self.lines.append("")

    def add_status_badge(self, status: str, message: str) -> None:
        """Add a status badge line."""
        emoji = {
            "pass": "\\u2705",
            "warn": "\\u26a0\\ufe0f",
            "fail": "\\u274c",
            "error": "\\U0001f534",
            "info": "\\U0001f535",
        }.get(status, "\\u2753")

        self.lines.append(f"**Status:** {emoji} {message}")
        self.lines.append("")

    def get_report(self) -> str:
        """Get the complete report as a string."""
        return "\n".join(self.lines)

    def save(self, path: str) -> None:
        """Save report to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_report())
