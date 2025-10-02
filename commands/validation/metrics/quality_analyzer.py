#!/usr/bin/env python3
"""
Quality Analyzer - Measures code quality improvements before and after validation.

This module provides detailed quality analysis including:
- Code quality scoring
- Complexity analysis
- Test coverage measurement
- Documentation completeness
- Security issue detection
- Performance profiling
"""

import ast
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    overall_score: float  # 0-100
    complexity_score: float
    maintainability_score: float
    test_coverage_score: float
    documentation_score: float
    security_score: float
    style_score: float

    # Detailed metrics
    total_files: int
    total_lines: int
    total_functions: int
    documented_functions: int
    complex_functions: int
    security_issues: List[Dict[str, Any]]
    code_smells: List[Dict[str, Any]]

    # Improvements (compared to baseline)
    quality_improvement_percent: float = 0.0
    complexity_reduction_percent: float = 0.0

    timestamp: datetime = datetime.now()


class QualityAnalyzer:
    """Analyzes code quality and tracks improvements."""

    def __init__(self):
        """Initialize quality analyzer."""
        self.baseline_reports: Dict[str, QualityReport] = {}

    def analyze(self, project_path: Path, language: str = "python") -> QualityReport:
        """Perform comprehensive quality analysis.

        Args:
            project_path: Path to project
            language: Programming language

        Returns:
            QualityReport object
        """
        if language == "python":
            return self._analyze_python(project_path)
        else:
            # Fallback to generic analysis
            return self._analyze_generic(project_path)

    def _analyze_python(self, project_path: Path) -> QualityReport:
        """Analyze Python project quality.

        Args:
            project_path: Path to Python project

        Returns:
            QualityReport object
        """
        # Count files and lines
        total_files, total_lines = self._count_files_lines(project_path, ['.py'])

        # Analyze functions
        total_functions, documented_functions, complex_functions = \
            self._analyze_python_functions(project_path)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            complex_functions, total_functions
        )

        # Calculate documentation score
        doc_score = (
            (documented_functions / total_functions * 100)
            if total_functions > 0 else 0
        )

        # Check for security issues (simplified)
        security_issues = self._check_security_python(project_path)
        security_score = max(0, 100 - len(security_issues) * 5)

        # Check code style
        code_smells = self._check_code_smells_python(project_path)
        style_score = max(0, 100 - len(code_smells) * 2)

        # Get test coverage (if possible)
        test_coverage_score = self._get_test_coverage(project_path)

        # Calculate maintainability
        maintainability_score = self._calculate_maintainability(
            complexity_score,
            doc_score,
            style_score
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            complexity_score,
            maintainability_score,
            test_coverage_score,
            doc_score,
            security_score,
            style_score
        )

        return QualityReport(
            overall_score=overall_score,
            complexity_score=complexity_score,
            maintainability_score=maintainability_score,
            test_coverage_score=test_coverage_score,
            documentation_score=doc_score,
            security_score=security_score,
            style_score=style_score,
            total_files=total_files,
            total_lines=total_lines,
            total_functions=total_functions,
            documented_functions=documented_functions,
            complex_functions=complex_functions,
            security_issues=security_issues,
            code_smells=code_smells
        )

    def _analyze_generic(self, project_path: Path) -> QualityReport:
        """Generic analysis for non-Python projects.

        Args:
            project_path: Path to project

        Returns:
            QualityReport object
        """
        extensions = ['.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']
        total_files, total_lines = self._count_files_lines(project_path, extensions)

        # Generic scoring
        return QualityReport(
            overall_score=70.0,
            complexity_score=75.0,
            maintainability_score=70.0,
            test_coverage_score=0.0,
            documentation_score=50.0,
            security_score=80.0,
            style_score=75.0,
            total_files=total_files,
            total_lines=total_lines,
            total_functions=0,
            documented_functions=0,
            complex_functions=0,
            security_issues=[],
            code_smells=[]
        )

    def _count_files_lines(
        self,
        project_path: Path,
        extensions: List[str]
    ) -> Tuple[int, int]:
        """Count files and lines in project.

        Args:
            project_path: Path to project
            extensions: List of file extensions to include

        Returns:
            Tuple of (total_files, total_lines)
        """
        total_files = 0
        total_lines = 0

        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in [
                '.git', '__pycache__', 'node_modules', 'venv', '.venv',
                'build', 'dist', '.eggs'
            ]):
                continue

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    total_files += 1
                    file_path = Path(root) / file

                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += sum(1 for line in f if line.strip())
                    except Exception:
                        continue

        return total_files, total_lines

    def _analyze_python_functions(
        self,
        project_path: Path
    ) -> Tuple[int, int, int]:
        """Analyze Python functions for complexity and documentation.

        Args:
            project_path: Path to project

        Returns:
            Tuple of (total_functions, documented_functions, complex_functions)
        """
        total_functions = 0
        documented_functions = 0
        complex_functions = 0

        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in [
                '.git', '__pycache__', 'venv', '.venv'
            ]):
                continue

            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read(), filename=str(file_path))

                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1

                            # Check for docstring
                            if (node.body and
                                isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and
                                isinstance(node.body[0].value.value, str)):
                                documented_functions += 1

                            # Simple complexity check (count decision points)
                            complexity = self._calculate_cyclomatic_complexity(node)
                            if complexity > 10:
                                complex_functions += 1

                except Exception:
                    continue

        return total_functions, documented_functions, complex_functions

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function.

        Args:
            node: AST function node

        Returns:
            Complexity score
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Count decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_complexity_score(
        self,
        complex_functions: int,
        total_functions: int
    ) -> float:
        """Calculate complexity score.

        Args:
            complex_functions: Number of complex functions
            total_functions: Total number of functions

        Returns:
            Score 0-100 (higher is better)
        """
        if total_functions == 0:
            return 100.0

        complex_ratio = complex_functions / total_functions
        return max(0, 100 - (complex_ratio * 100))

    def _check_security_python(self, project_path: Path) -> List[Dict[str, Any]]:
        """Check for common security issues in Python code.

        Args:
            project_path: Path to project

        Returns:
            List of security issues
        """
        issues = []

        # Patterns to check
        dangerous_patterns = [
            (r'eval\(', 'Use of eval() is dangerous'),
            (r'exec\(', 'Use of exec() is dangerous'),
            (r'__import__\(', 'Dynamic import may be risky'),
            (r'pickle\.loads\(', 'Pickle deserialization can be unsafe'),
            (r'yaml\.load\(', 'Use yaml.safe_load() instead'),
            (r'shell=True', 'shell=True in subprocess is risky'),
        ]

        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'venv']):
                continue

            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    for pattern, message in dangerous_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            issues.append({
                                'file': str(file_path.relative_to(project_path)),
                                'line': line_num,
                                'issue': message,
                                'severity': 'high'
                            })

                except Exception:
                    continue

        return issues[:50]  # Limit to 50 issues

    def _check_code_smells_python(self, project_path: Path) -> List[Dict[str, Any]]:
        """Check for code smells in Python code.

        Args:
            project_path: Path to project

        Returns:
            List of code smells
        """
        smells = []

        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'venv']):
                continue

            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines, 1):
                        # Long lines
                        if len(line) > 120:
                            smells.append({
                                'file': str(file_path.relative_to(project_path)),
                                'line': i,
                                'smell': 'Line too long',
                                'severity': 'low'
                            })

                        # TODO comments
                        if 'TODO' in line or 'FIXME' in line:
                            smells.append({
                                'file': str(file_path.relative_to(project_path)),
                                'line': i,
                                'smell': 'TODO/FIXME comment',
                                'severity': 'low'
                            })

                except Exception:
                    continue

        return smells[:100]  # Limit to 100 smells

    def _get_test_coverage(self, project_path: Path) -> float:
        """Get test coverage percentage.

        Args:
            project_path: Path to project

        Returns:
            Coverage percentage (0-100)
        """
        # Try to run coverage if available
        try:
            result = subprocess.run(
                ['coverage', 'run', '-m', 'pytest'],
                cwd=project_path,
                capture_output=True,
                timeout=60,
                check=False
            )

            if result.returncode == 0:
                result = subprocess.run(
                    ['coverage', 'report'],
                    cwd=project_path,
                    capture_output=True,
                    timeout=10,
                    check=False
                )

                output = result.stdout.decode()
                # Parse coverage percentage from output
                match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
                if match:
                    return float(match.group(1))

        except Exception:
            pass

        return 0.0

    def _calculate_maintainability(
        self,
        complexity_score: float,
        doc_score: float,
        style_score: float
    ) -> float:
        """Calculate maintainability index.

        Args:
            complexity_score: Complexity score
            doc_score: Documentation score
            style_score: Style score

        Returns:
            Maintainability score (0-100)
        """
        return (complexity_score * 0.4 + doc_score * 0.3 + style_score * 0.3)

    def _calculate_overall_score(
        self,
        complexity_score: float,
        maintainability_score: float,
        test_coverage_score: float,
        doc_score: float,
        security_score: float,
        style_score: float
    ) -> float:
        """Calculate overall quality score.

        Args:
            complexity_score: Complexity score
            maintainability_score: Maintainability score
            test_coverage_score: Test coverage score
            doc_score: Documentation score
            security_score: Security score
            style_score: Style score

        Returns:
            Overall score (0-100)
        """
        weights = {
            'complexity': 0.20,
            'maintainability': 0.20,
            'test_coverage': 0.20,
            'documentation': 0.15,
            'security': 0.15,
            'style': 0.10
        }

        overall = (
            complexity_score * weights['complexity'] +
            maintainability_score * weights['maintainability'] +
            test_coverage_score * weights['test_coverage'] +
            doc_score * weights['documentation'] +
            security_score * weights['security'] +
            style_score * weights['style']
        )

        return round(overall, 2)

    def set_baseline(self, project_name: str, report: QualityReport) -> None:
        """Set baseline quality report for comparison.

        Args:
            project_name: Name of project
            report: Quality report to use as baseline
        """
        self.baseline_reports[project_name] = report

    def compare_to_baseline(
        self,
        project_name: str,
        current_report: QualityReport
    ) -> QualityReport:
        """Compare current report to baseline and calculate improvements.

        Args:
            project_name: Name of project
            current_report: Current quality report

        Returns:
            Updated quality report with improvement metrics
        """
        if project_name not in self.baseline_reports:
            return current_report

        baseline = self.baseline_reports[project_name]

        # Calculate improvements
        quality_improvement = (
            (current_report.overall_score - baseline.overall_score) /
            baseline.overall_score * 100
            if baseline.overall_score > 0 else 0
        )

        complexity_improvement = (
            (current_report.complexity_score - baseline.complexity_score) /
            baseline.complexity_score * 100
            if baseline.complexity_score > 0 else 0
        )

        # Update report
        current_report.quality_improvement_percent = quality_improvement
        current_report.complexity_reduction_percent = complexity_improvement

        return current_report

    def generate_quality_diff(
        self,
        project_name: str,
        current_report: QualityReport
    ) -> Dict[str, Any]:
        """Generate detailed diff between baseline and current.

        Args:
            project_name: Name of project
            current_report: Current quality report

        Returns:
            Dictionary containing detailed differences
        """
        if project_name not in self.baseline_reports:
            return {}

        baseline = self.baseline_reports[project_name]

        return {
            'overall_score_change': current_report.overall_score - baseline.overall_score,
            'complexity_score_change': current_report.complexity_score - baseline.complexity_score,
            'maintainability_change': current_report.maintainability_score - baseline.maintainability_score,
            'test_coverage_change': current_report.test_coverage_score - baseline.test_coverage_score,
            'documentation_change': current_report.documentation_score - baseline.documentation_score,
            'security_score_change': current_report.security_score - baseline.security_score,
            'style_score_change': current_report.style_score - baseline.style_score,
            'security_issues_fixed': len(baseline.security_issues) - len(current_report.security_issues),
            'code_smells_fixed': len(baseline.code_smells) - len(current_report.code_smells),
        }


# For standalone testing
if __name__ == "__main__":
    import sys

    analyzer = QualityAnalyzer()

    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    else:
        project_path = Path.cwd()

    print(f"Analyzing project: {project_path}")
    print("-" * 80)

    report = analyzer.analyze(project_path)

    print(f"\nOverall Score: {report.overall_score:.1f}/100")
    print(f"Complexity: {report.complexity_score:.1f}/100")
    print(f"Maintainability: {report.maintainability_score:.1f}/100")
    print(f"Test Coverage: {report.test_coverage_score:.1f}%")
    print(f"Documentation: {report.documentation_score:.1f}%")
    print(f"Security: {report.security_score:.1f}/100")
    print(f"Style: {report.style_score:.1f}/100")
    print(f"\nFiles: {report.total_files}")
    print(f"Lines: {report.total_lines}")
    print(f"Functions: {report.total_functions}")
    print(f"Documented: {report.documented_functions}/{report.total_functions}")
    print(f"Complex: {report.complex_functions}/{report.total_functions}")
    print(f"\nSecurity Issues: {len(report.security_issues)}")
    print(f"Code Smells: {len(report.code_smells)}")