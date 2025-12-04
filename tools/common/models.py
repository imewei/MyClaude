"""
Shared data models for plugin validation tools.

Consolidates duplicate dataclass definitions from:
- metadata-validator.py (ValidationError, ValidationResult)
- plugin-review-script.py (ReviewIssue, ReviewReport)
- xref-validator.py (CrossReference, ValidationResult)
- load-profiler.py (LoadMetric, PluginLoadProfile)
- activation-profiler.py (ActivationMetric, ActivationProfile)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ValidationIssue:
    """Represents a validation issue (error or warning).

    Consolidates ValidationError, ReviewIssue, and similar classes.
    """

    field: str
    severity: str  # 'critical', 'error', 'warning', 'info'
    message: str
    suggestion: Optional[str] = None
    file_path: Optional[str] = None
    line_number: int = 0

    @property
    def is_error(self) -> bool:
        """Returns True if this is an error-level issue."""
        return self.severity in ("critical", "error")

    @property
    def emoji(self) -> str:
        """Returns emoji for severity level."""
        return {
            "critical": "\U0001f534",  # Red circle
            "error": "\U0001f7e0",  # Orange circle
            "warning": "\U0001f7e1",  # Yellow circle
            "info": "\U0001f535",  # Blue circle
        }.get(self.severity, "\u26aa")  # White circle


@dataclass
class ValidationResult:
    """Standardized validation result for all validators.

    Consolidates ValidationResult from metadata-validator.py, ReviewReport
    from plugin-review-script.py, etc.
    """

    plugin_name: str
    plugin_path: Optional[Path] = None
    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(
        self,
        field: str,
        message: str,
        suggestion: Optional[str] = None,
        file_path: Optional[str] = None,
        line_number: int = 0,
    ) -> None:
        """Add a validation error."""
        self.issues.append(
            ValidationIssue(
                field=field,
                severity="error",
                message=message,
                suggestion=suggestion,
                file_path=file_path,
                line_number=line_number,
            )
        )
        self.is_valid = False

    def add_warning(
        self,
        field: str,
        message: str,
        suggestion: Optional[str] = None,
        file_path: Optional[str] = None,
        line_number: int = 0,
    ) -> None:
        """Add a validation warning."""
        self.issues.append(
            ValidationIssue(
                field=field,
                severity="warning",
                message=message,
                suggestion=suggestion,
                file_path=file_path,
                line_number=line_number,
            )
        )

    def add_info(self, field: str, message: str) -> None:
        """Add an informational message."""
        self.issues.append(
            ValidationIssue(field=field, severity="info", message=message)
        )

    @property
    def errors(self) -> list[ValidationIssue]:
        """Returns all error-level issues."""
        return [i for i in self.issues if i.severity in ("critical", "error")]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Returns all warning-level issues."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def error_count(self) -> int:
        """Count of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return len(self.warnings)

    def get_issue_count_by_severity(self) -> dict[str, int]:
        """Count issues by severity level."""
        counts: dict[str, int] = {"critical": 0, "error": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts


@dataclass
class PluginMetadata:
    """Plugin metadata extracted from plugin.json.

    Consolidates PluginMetadata from dependency-mapper.py and similar.
    """

    name: str
    version: str
    description: str
    category: str
    path: Optional[Path] = None
    agents: list[dict] = field(default_factory=list)
    commands: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def agent_names(self) -> list[str]:
        """List of agent names."""
        return [a.get("name", "") for a in self.agents if a.get("name")]

    @property
    def command_names(self) -> list[str]:
        """List of command names."""
        return [c.get("name", "") for c in self.commands if c.get("name")]

    @property
    def skill_names(self) -> list[str]:
        """List of skill names."""
        return [s.get("name", "") for s in self.skills if s.get("name")]

    @property
    def agent_count(self) -> int:
        """Number of agents."""
        return len(self.agents)

    @property
    def command_count(self) -> int:
        """Number of commands."""
        return len(self.commands)

    @property
    def skill_count(self) -> int:
        """Number of skills."""
        return len(self.skills)


@dataclass
class ProfileMetric:
    """Timing metric for profiling operations.

    Consolidates LoadMetric and ActivationMetric.
    """

    name: str
    duration_ms: float
    status: str  # 'pass', 'warn', 'fail', 'error'
    details: str = ""

    @property
    def status_emoji(self) -> str:
        """Visual status indicator."""
        return {
            "pass": "\u2705",  # Green checkmark
            "warn": "\u26a0\ufe0f",  # Warning sign
            "fail": "\u274c",  # Red X
            "error": "\U0001f534",  # Red circle
        }.get(self.status, "\u2753")  # Question mark

    @classmethod
    def from_duration(
        cls,
        name: str,
        duration_ms: float,
        pass_threshold: float,
        warn_threshold: float,
        details: str = "",
    ) -> "ProfileMetric":
        """Create metric with automatic status based on thresholds."""
        if duration_ms < pass_threshold:
            status = "pass"
        elif duration_ms < warn_threshold:
            status = "warn"
        else:
            status = "fail"
        return cls(name=name, duration_ms=duration_ms, status=status, details=details)
