#!/usr/bin/env python3
"""
Regression Detector - Detects performance and quality regressions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from validation.benchmarks.baseline_collector import Baseline, BaselineCollector


@dataclass
class Regression:
    """Detected regression."""
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    severity: str  # 'critical', 'high', 'medium', 'low'

    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.severity.upper()}: {self.metric_name} regressed by "
            f"{self.regression_percent:.1f}% ({self.baseline_value} -> {self.current_value})"
        )


class RegressionDetector:
    """Detects regressions by comparing metrics to baselines."""

    # Thresholds for regression severity
    CRITICAL_THRESHOLD = 50.0  # >50% regression
    HIGH_THRESHOLD = 25.0      # >25% regression
    MEDIUM_THRESHOLD = 10.0    # >10% regression
    LOW_THRESHOLD = 5.0        # >5% regression

    def __init__(self, baseline_collector: Optional[BaselineCollector] = None):
        """Initialize regression detector.

        Args:
            baseline_collector: Baseline collector instance
        """
        self.baseline_collector = baseline_collector or BaselineCollector()

    def detect(
        self,
        project_name: str,
        scenario_name: str,
        current_metrics: Dict[str, Any],
        metrics_to_check: Optional[List[str]] = None
    ) -> List[Regression]:
        """Detect regressions by comparing to baseline.

        Args:
            project_name: Name of project
            scenario_name: Name of scenario
            current_metrics: Current metrics
            metrics_to_check: List of metrics to check (None = all numeric)

        Returns:
            List of detected regressions
        """
        baseline = self.baseline_collector.get(project_name, scenario_name)

        if not baseline:
            return []

        regressions = []

        # Determine which metrics to check
        if metrics_to_check is None:
            metrics_to_check = [
                k for k, v in current_metrics.items()
                if isinstance(v, (int, float))
            ]

        for metric_name in metrics_to_check:
            if metric_name not in baseline.metrics or metric_name not in current_metrics:
                continue

            baseline_value = baseline.metrics[metric_name]
            current_value = current_metrics[metric_name]

            # Skip if not numeric
            if not isinstance(baseline_value, (int, float)) or \
               not isinstance(current_value, (int, float)):
                continue

            # Calculate regression (higher values = worse for most metrics)
            # For metrics like quality_score where higher is better, invert
            if self._is_higher_better(metric_name):
                regression_percent = self._calculate_regression(current_value, baseline_value)
            else:
                regression_percent = self._calculate_regression(baseline_value, current_value)

            # Check if significant regression
            if regression_percent >= self.LOW_THRESHOLD:
                severity = self._determine_severity(regression_percent)

                regressions.append(Regression(
                    metric_name=metric_name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    regression_percent=regression_percent,
                    severity=severity
                ))

        return regressions

    def _calculate_regression(
        self,
        baseline_value: float,
        current_value: float
    ) -> float:
        """Calculate regression percentage.

        Args:
            baseline_value: Baseline value
            current_value: Current value

        Returns:
            Regression percentage (positive = worse)
        """
        if baseline_value == 0:
            return 0.0

        change = current_value - baseline_value
        return (change / baseline_value) * 100

    def _is_higher_better(self, metric_name: str) -> bool:
        """Check if higher values are better for this metric.

        Args:
            metric_name: Name of metric

        Returns:
            True if higher is better
        """
        higher_better_metrics = {
            'quality_score', 'coverage', 'test_coverage',
            'documentation_coverage', 'security_score',
            'maintainability', 'cache_hit_rate',
            'performance_improvement', 'quality_improvement'
        }

        return any(pattern in metric_name.lower() for pattern in higher_better_metrics)

    def _determine_severity(self, regression_percent: float) -> str:
        """Determine severity of regression.

        Args:
            regression_percent: Regression percentage

        Returns:
            Severity level
        """
        if regression_percent >= self.CRITICAL_THRESHOLD:
            return 'critical'
        elif regression_percent >= self.HIGH_THRESHOLD:
            return 'high'
        elif regression_percent >= self.MEDIUM_THRESHOLD:
            return 'medium'
        else:
            return 'low'

    def check_for_regressions(
        self,
        project_name: str,
        scenario_name: str,
        current_metrics: Dict[str, Any],
        fail_on_severity: str = 'high'
    ) -> tuple[bool, List[Regression]]:
        """Check for regressions and determine if validation should fail.

        Args:
            project_name: Name of project
            scenario_name: Name of scenario
            current_metrics: Current metrics
            fail_on_severity: Fail if regression at or above this severity

        Returns:
            Tuple of (should_fail, regressions)
        """
        regressions = self.detect(project_name, scenario_name, current_metrics)

        severity_order = ['low', 'medium', 'high', 'critical']
        fail_threshold = severity_order.index(fail_on_severity)

        should_fail = any(
            severity_order.index(r.severity) >= fail_threshold
            for r in regressions
        )

        return should_fail, regressions

    def generate_regression_report(
        self,
        regressions: List[Regression]
    ) -> str:
        """Generate human-readable regression report.

        Args:
            regressions: List of regressions

        Returns:
            Formatted report string
        """
        if not regressions:
            return "No regressions detected."

        report = ["REGRESSION REPORT", "=" * 80, ""]

        # Group by severity
        by_severity = {}
        for regression in regressions:
            by_severity.setdefault(regression.severity, []).append(regression)

        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in by_severity:
                report.append(f"\n{severity.upper()} Severity ({len(by_severity[severity])}):")
                report.append("-" * 80)
                for r in by_severity[severity]:
                    report.append(str(r))

        return "\n".join(report)