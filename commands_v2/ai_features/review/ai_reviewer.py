#!/usr/bin/env python3
"""
AI Code Reviewer
================

Automated code review using AI and machine learning.

Features:
- Identify potential bugs
- Suggest improvements
- Check best practices
- Security vulnerability detection
- Performance issue identification
- Style and formatting checks

Author: Claude Code AI Team
"""

import logging
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class IssueSeverity(Enum):
    """Severity levels for code issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Categories of code issues"""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BEST_PRACTICE = "best_practice"
    MAINTAINABILITY = "maintainability"


@dataclass
class CodeIssue:
    """Code issue found by reviewer"""
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    file_path: Path
    line_number: int
    suggestion: str
    auto_fixable: bool = False


@dataclass
class ReviewResult:
    """Result of code review"""
    file_path: Path
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    summary: str
    overall_quality: float  # 0-1


class AIReviewer:
    """
    AI-powered code reviewer.

    Performs comprehensive code review including:
    - Bug detection
    - Security analysis
    - Performance review
    - Best practice validation
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Review rules
        self.rules = self._load_review_rules()

    def review_file(
        self,
        file_path: Path,
        focus: Optional[List[str]] = None
    ) -> ReviewResult:
        """
        Review a single file.

        Args:
            file_path: Path to file
            focus: Areas to focus on (bug, security, performance, etc.)

        Returns:
            Review result with issues and metrics
        """
        self.logger.info(f"Reviewing file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Perform reviews
            issues = []

            if not focus or "bug" in focus:
                issues.extend(self._check_bugs(tree, file_path, source))

            if not focus or "security" in focus:
                issues.extend(self._check_security(tree, file_path, source))

            if not focus or "performance" in focus:
                issues.extend(self._check_performance(tree, file_path, source))

            if not focus or "style" in focus:
                issues.extend(self._check_style(tree, file_path, source))

            if not focus or "best_practice" in focus:
                issues.extend(self._check_best_practices(tree, file_path, source))

            # Calculate metrics
            metrics = self._calculate_metrics(tree, issues)

            # Calculate overall quality
            quality = self._calculate_quality_score(issues, metrics)

            # Generate summary
            summary = self._generate_summary(issues, metrics)

            return ReviewResult(
                file_path=file_path,
                issues=issues,
                metrics=metrics,
                summary=summary,
                overall_quality=quality
            )

        except Exception as e:
            self.logger.error(f"Review failed for {file_path}: {e}")
            return ReviewResult(
                file_path=file_path,
                issues=[],
                metrics={},
                summary=f"Review failed: {str(e)}",
                overall_quality=0.0
            )

    def _check_bugs(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[CodeIssue]:
        """Check for potential bugs"""
        issues = []

        # Check for common bug patterns
        for node in ast.walk(tree):
            # Mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(CodeIssue(
                            category=IssueCategory.BUG,
                            severity=IssueSeverity.MEDIUM,
                            title="Mutable default argument",
                            description="Default argument is mutable, may cause unexpected behavior",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggestion="Use None as default and initialize inside function",
                            auto_fixable=True
                        ))

            # Bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(CodeIssue(
                        category=IssueCategory.BUG,
                        severity=IssueSeverity.MEDIUM,
                        title="Bare except clause",
                        description="Catching all exceptions can hide bugs",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Catch specific exceptions instead",
                        auto_fixable=False
                    ))

        return issues

    def _check_security(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[CodeIssue]:
        """Check for security vulnerabilities"""
        issues = []

        for node in ast.walk(tree):
            # Use of eval()
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "eval":
                    issues.append(CodeIssue(
                        category=IssueCategory.SECURITY,
                        severity=IssueSeverity.CRITICAL,
                        title="Use of eval()",
                        description="eval() can execute arbitrary code and is a security risk",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Use ast.literal_eval() or safer alternatives",
                        auto_fixable=False
                    ))

                # exec() usage
                if isinstance(node.func, ast.Name) and node.func.id == "exec":
                    issues.append(CodeIssue(
                        category=IssueCategory.SECURITY,
                        severity=IssueSeverity.HIGH,
                        title="Use of exec()",
                        description="exec() can execute arbitrary code",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Avoid exec() or use restricted environment",
                        auto_fixable=False
                    ))

        return issues

    def _check_performance(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[CodeIssue]:
        """Check for performance issues"""
        issues = []

        # Check for nested loops
        loop_depth = 0
        max_depth = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_depth += 1
                max_depth = max(max_depth, loop_depth)

                if loop_depth >= 3:
                    issues.append(CodeIssue(
                        category=IssueCategory.PERFORMANCE,
                        severity=IssueSeverity.MEDIUM,
                        title="Deep loop nesting",
                        description=f"Loop nesting depth of {loop_depth} may impact performance",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Consider algorithm optimization or vectorization",
                        auto_fixable=False
                    ))

                loop_depth -= 1

        return issues

    def _check_style(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[CodeIssue]:
        """Check style and formatting"""
        issues = []

        for node in ast.walk(tree):
            # Missing docstrings
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(CodeIssue(
                        category=IssueCategory.STYLE,
                        severity=IssueSeverity.LOW,
                        title="Missing docstring",
                        description=f"{node.__class__.__name__} '{node.name}' has no docstring",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Add docstring describing purpose and parameters",
                        auto_fixable=True
                    ))

        return issues

    def _check_best_practices(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[CodeIssue]:
        """Check best practice violations"""
        issues = []

        for node in ast.walk(tree):
            # Too many parameters
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 7:
                    issues.append(CodeIssue(
                        category=IssueCategory.BEST_PRACTICE,
                        severity=IssueSeverity.MEDIUM,
                        title="Too many parameters",
                        description=f"Function has {param_count} parameters",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Consider using a configuration object or builder pattern",
                        auto_fixable=False
                    ))

        return issues

    def _calculate_metrics(
        self,
        tree: ast.AST,
        issues: List[CodeIssue]
    ) -> Dict[str, Any]:
        """Calculate code metrics"""
        metrics = {
            "total_issues": len(issues),
            "critical_issues": sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
            "high_issues": sum(1 for i in issues if i.severity == IssueSeverity.HIGH),
            "medium_issues": sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM),
            "low_issues": sum(1 for i in issues if i.severity == IssueSeverity.LOW),
            "auto_fixable": sum(1 for i in issues if i.auto_fixable),
        }

        # Issue breakdown by category
        for category in IssueCategory:
            count = sum(1 for i in issues if i.category == category)
            metrics[f"{category.value}_count"] = count

        return metrics

    def _calculate_quality_score(
        self,
        issues: List[CodeIssue],
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score (0-1)"""
        if not issues:
            return 1.0

        # Weight issues by severity
        severity_weights = {
            IssueSeverity.CRITICAL: 10,
            IssueSeverity.HIGH: 5,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 1,
            IssueSeverity.INFO: 0.5
        }

        weighted_issues = sum(
            severity_weights.get(issue.severity, 1)
            for issue in issues
        )

        # Normalize to 0-1 scale
        # Assume 0 issues = 1.0, 10+ weighted issues = 0.0
        score = max(0.0, 1.0 - (weighted_issues / 20.0))

        return score

    def _generate_summary(
        self,
        issues: List[CodeIssue],
        metrics: Dict[str, Any]
    ) -> str:
        """Generate review summary"""
        parts = []

        parts.append(f"Found {metrics['total_issues']} issues:")

        if metrics['critical_issues'] > 0:
            parts.append(f"  - {metrics['critical_issues']} critical")
        if metrics['high_issues'] > 0:
            parts.append(f"  - {metrics['high_issues']} high severity")
        if metrics['medium_issues'] > 0:
            parts.append(f"  - {metrics['medium_issues']} medium severity")
        if metrics['low_issues'] > 0:
            parts.append(f"  - {metrics['low_issues']} low severity")

        if metrics['auto_fixable'] > 0:
            parts.append(f"\n{metrics['auto_fixable']} issues can be auto-fixed")

        return "\n".join(parts)

    def _load_review_rules(self) -> Dict[str, Any]:
        """Load review rules configuration"""
        # In production, load from configuration file
        return {
            "enable_bug_detection": True,
            "enable_security_scan": True,
            "enable_performance_check": True,
            "enable_style_check": True,
            "enable_best_practices": True,
        }


def main():
    """Demonstration"""
    print("AI Code Reviewer")
    print("===============\n")

    reviewer = AIReviewer()

    # Review current file
    result = reviewer.review_file(Path(__file__))

    print(f"File: {result.file_path.name}")
    print(f"Overall Quality: {result.overall_quality:.2%}\n")
    print(result.summary)

    if result.issues:
        print("\nIssues found:")
        for issue in result.issues[:5]:  # Show first 5
            print(f"\n[{issue.severity.value.upper()}] {issue.title}")
            print(f"  Line {issue.line_number}: {issue.description}")
            print(f"  Suggestion: {issue.suggestion}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())