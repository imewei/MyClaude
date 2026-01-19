#!/usr/bin/env python3
"""
Triggering Pattern Report Generator

Aggregates metrics from activation testing, command analysis, and skill validation
to generate comprehensive triggering pattern reports with actionable recommendations.

Usage:
    python3 tools/triggering_reporter.py
    python3 tools/triggering_reporter.py --reports-dir /path/to/reports
    python3 tools/triggering_reporter.py --output comprehensive-report.md
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TriggeringMetrics:
    """Aggregated triggering pattern metrics."""

    # Activation metrics
    activation_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    activation_precision: float = 0.0
    activation_recall: float = 0.0

    # Command metrics
    command_relevance: float = 0.0
    command_timing: float = 0.0
    command_priority: float = 0.0

    # Skill metrics
    skill_accuracy: float = 0.0
    skill_precision: float = 0.0
    skill_over_trigger_rate: float = 0.0
    skill_under_trigger_rate: float = 0.0

    # Plugin-specific metrics
    plugin_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class Issue:
    """Represents a triggering pattern issue."""

    severity: str  # critical, high, medium, low
    category: str  # activation, command, skill
    plugin: Optional[str]
    description: str
    impact: str
    recommendation: str


class TriggeringPatternReporter:
    """Generates comprehensive triggering pattern reports."""

    def __init__(self, reports_dir: str):
        self.reports_dir = Path(reports_dir)
        self.metrics = TriggeringMetrics()
        self.issues: List[Issue] = []

    def load_activation_report(self) -> bool:
        """Load activation accuracy report."""
        report_path = self.reports_dir / "activation-accuracy.md"

        if not report_path.exists():
            print(f"⚠️  Activation report not found: {report_path}")
            return False

        try:
            content = report_path.read_text()

            # Extract metrics using regex
            accuracy_match = re.search(r'Overall Accuracy \| ([\d.]+)%', content)
            if accuracy_match:
                self.metrics.activation_accuracy = float(accuracy_match.group(1))

            precision_match = re.search(r'Precision \| ([\d.]+)%', content)
            if precision_match:
                self.metrics.activation_precision = float(precision_match.group(1))

            recall_match = re.search(r'Recall \| ([\d.]+)%', content)
            if recall_match:
                self.metrics.activation_recall = float(recall_match.group(1))

            fp_match = re.search(r'False Positive Rate \| ([\d.]+)%', content)
            if fp_match:
                self.metrics.false_positive_rate = float(fp_match.group(1))

            fn_match = re.search(r'False Negative Rate \| ([\d.]+)%', content)
            if fn_match:
                self.metrics.false_negative_rate = float(fn_match.group(1))

            # Identify issues
            if self.metrics.false_positive_rate >= 10:
                self.issues.append(Issue(
                    severity="high",
                    category="activation",
                    plugin=None,
                    description=f"High false positive rate: {self.metrics.false_positive_rate:.1f}%",
                    impact="Plugins activating in inappropriate contexts, causing user confusion",
                    recommendation="Review and tighten plugin triggering patterns. Add negative patterns to exclude irrelevant contexts."
                ))
            elif self.metrics.false_positive_rate >= 5:
                self.issues.append(Issue(
                    severity="medium",
                    category="activation",
                    plugin=None,
                    description=f"Elevated false positive rate: {self.metrics.false_positive_rate:.1f}%",
                    impact="Some plugins activating too broadly",
                    recommendation="Review triggering patterns for plugins with high false positive rates."
                ))

            if self.metrics.false_negative_rate >= 10:
                self.issues.append(Issue(
                    severity="high",
                    category="activation",
                    plugin=None,
                    description=f"High false negative rate: {self.metrics.false_negative_rate:.1f}%",
                    impact="Plugins not activating when needed, missing helpful suggestions",
                    recommendation="Expand plugin triggering patterns. Add more keywords and file patterns."
                ))
            elif self.metrics.false_negative_rate >= 5:
                self.issues.append(Issue(
                    severity="medium",
                    category="activation",
                    plugin=None,
                    description=f"Elevated false negative rate: {self.metrics.false_negative_rate:.1f}%",
                    impact="Some plugins not activating when they should",
                    recommendation="Review and expand triggering patterns for affected plugins."
                ))

            print(f"✓ Loaded activation metrics: {self.metrics.activation_accuracy:.1f}% accuracy")
            return True

        except Exception as e:
            print(f"✗ Error loading activation report: {e}")
            return False

    def load_command_report(self) -> bool:
        """Load command suggestion analysis report."""
        report_path = self.reports_dir / "command-analysis.md"

        if not report_path.exists():
            print(f"⚠️  Command report not found: {report_path}")
            return False

        try:
            content = report_path.read_text()

            # Extract metrics
            relevance_match = re.search(r'Relevance Accuracy \| ([\d.]+)%', content)
            if relevance_match:
                self.metrics.command_relevance = float(relevance_match.group(1))

            timing_match = re.search(r'Timing Accuracy \| ([\d.]+)%', content)
            if timing_match:
                self.metrics.command_timing = float(timing_match.group(1))

            priority_match = re.search(r'Priority Accuracy \| ([\d.]+)%', content)
            if priority_match:
                self.metrics.command_priority = float(priority_match.group(1))

            # Identify issues
            if self.metrics.command_relevance < 70:
                self.issues.append(Issue(
                    severity="high",
                    category="command",
                    plugin=None,
                    description=f"Low command relevance: {self.metrics.command_relevance:.1f}%",
                    impact="Commands being suggested in inappropriate contexts",
                    recommendation="Review command keywords and triggering conditions. Add context-specific logic."
                ))
            elif self.metrics.command_relevance < 80:
                self.issues.append(Issue(
                    severity="medium",
                    category="command",
                    plugin=None,
                    description=f"Suboptimal command relevance: {self.metrics.command_relevance:.1f}%",
                    impact="Some commands suggested with low relevance",
                    recommendation="Refine command triggering logic to improve relevance scoring."
                ))

            if self.metrics.command_timing < 75:
                self.issues.append(Issue(
                    severity="medium",
                    category="command",
                    plugin=None,
                    description=f"Poor command timing: {self.metrics.command_timing:.1f}%",
                    impact="Commands suggested at inappropriate times in workflow",
                    recommendation="Implement workflow-aware command suggestions. Consider project maturity."
                ))

            if self.metrics.command_priority < 85:
                self.issues.append(Issue(
                    severity="low",
                    category="command",
                    plugin=None,
                    description=f"Suboptimal priority ranking: {self.metrics.command_priority:.1f}%",
                    impact="Command priorities don't reflect their importance",
                    recommendation="Review and adjust command priority values based on usage patterns."
                ))

            print(f"✓ Loaded command metrics: {self.metrics.command_relevance:.1f}% relevance")
            return True

        except Exception as e:
            print(f"✗ Error loading command report: {e}")
            return False

    def load_skill_report(self) -> bool:
        """Load skill validation report."""
        report_path = self.reports_dir / "skill-validation.md"

        if not report_path.exists():
            print(f"⚠️  Skill report not found: {report_path}")
            return False

        try:
            content = report_path.read_text()

            # Extract metrics
            accuracy_match = re.search(r'Overall Accuracy \| ([\d.]+)%', content)
            if accuracy_match:
                self.metrics.skill_accuracy = float(accuracy_match.group(1))

            precision_match = re.search(r'Precision \| ([\d.]+)%', content)
            if precision_match:
                self.metrics.skill_precision = float(precision_match.group(1))

            over_trigger_match = re.search(r'Over-Trigger Rate \| ([\d.]+)%', content)
            if over_trigger_match:
                self.metrics.skill_over_trigger_rate = float(over_trigger_match.group(1))

            under_trigger_match = re.search(r'Under-Trigger Rate \| ([\d.]+)%', content)
            if under_trigger_match:
                self.metrics.skill_under_trigger_rate = float(under_trigger_match.group(1))

            # Identify issues
            if self.metrics.skill_over_trigger_rate >= 15:
                self.issues.append(Issue(
                    severity="high",
                    category="skill",
                    plugin=None,
                    description=f"High skill over-triggering: {self.metrics.skill_over_trigger_rate:.1f}%",
                    impact="Skills applying too broadly, causing noise",
                    recommendation="Tighten skill pattern matching. Add more specific keywords and context checks."
                ))
            elif self.metrics.skill_over_trigger_rate >= 10:
                self.issues.append(Issue(
                    severity="medium",
                    category="skill",
                    plugin=None,
                    description=f"Elevated skill over-triggering: {self.metrics.skill_over_trigger_rate:.1f}%",
                    impact="Some skills applying more broadly than intended",
                    recommendation="Review skill patterns for affected plugins."
                ))

            if self.metrics.skill_under_trigger_rate >= 15:
                self.issues.append(Issue(
                    severity="high",
                    category="skill",
                    plugin=None,
                    description=f"High skill under-triggering: {self.metrics.skill_under_trigger_rate:.1f}%",
                    impact="Skills not applying when they should",
                    recommendation="Expand skill pattern matching. Add more keywords and broaden context detection."
                ))
            elif self.metrics.skill_under_trigger_rate >= 10:
                self.issues.append(Issue(
                    severity="medium",
                    category="skill",
                    plugin=None,
                    description=f"Elevated skill under-triggering: {self.metrics.skill_under_trigger_rate:.1f}%",
                    impact="Some skills missing relevant contexts",
                    recommendation="Review and expand skill patterns."
                ))

            print(f"✓ Loaded skill metrics: {self.metrics.skill_accuracy:.1f}% accuracy")
            return True

        except Exception as e:
            print(f"✗ Error loading skill report: {e}")
            return False

    def calculate_overall_score(self) -> float:
        """Calculate overall triggering pattern quality score."""
        scores = []

        # Activation score (weight: 40%)
        if self.metrics.activation_accuracy > 0:
            activation_score = (
                self.metrics.activation_accuracy * 0.4 +
                (100 - self.metrics.false_positive_rate) * 0.3 +
                (100 - self.metrics.false_negative_rate) * 0.3
            )
            scores.append((activation_score, 0.4))

        # Command score (weight: 30%)
        if self.metrics.command_relevance > 0:
            command_score = (
                self.metrics.command_relevance * 0.5 +
                self.metrics.command_timing * 0.3 +
                self.metrics.command_priority * 0.2
            )
            scores.append((command_score, 0.3))

        # Skill score (weight: 30%)
        if self.metrics.skill_accuracy > 0:
            skill_score = (
                self.metrics.skill_accuracy * 0.5 +
                (100 - self.metrics.skill_over_trigger_rate) * 0.25 +
                (100 - self.metrics.skill_under_trigger_rate) * 0.25
            )
            scores.append((skill_score, 0.3))

        if not scores:
            return 0.0

        # Calculate weighted average
        total_score = sum(score * weight for score, weight in scores)
        total_weight = sum(weight for _, weight in scores)

        return total_score / total_weight if total_weight > 0 else 0.0

    def generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Sort issues by severity
        critical = [i for i in self.issues if i.severity == "critical"]
        high = [i for i in self.issues if i.severity == "high"]
        medium = [i for i in self.issues if i.severity == "medium"]
        low = [i for i in self.issues if i.severity == "low"]

        if critical:
            recommendations.append("**CRITICAL ACTIONS REQUIRED:**")
            for issue in critical:
                recommendations.append(f"- {issue.recommendation}")

        if high:
            recommendations.append("\n**HIGH PRIORITY:**")
            for issue in high:
                recommendations.append(f"- {issue.recommendation}")

        if medium:
            recommendations.append("\n**MEDIUM PRIORITY:**")
            for issue in medium[:5]:  # Limit to top 5
                recommendations.append(f"- {issue.recommendation}")

        if low and len(recommendations) < 10:
            recommendations.append("\n**LOW PRIORITY:**")
            for issue in low[:3]:  # Limit to top 3
                recommendations.append(f"- {issue.recommendation}")

        # General recommendations based on metrics
        overall_score = self.calculate_overall_score()

        if overall_score < 70:
            recommendations.append("\n**GENERAL RECOMMENDATIONS:**")
            recommendations.append("- Conduct comprehensive review of all triggering patterns")
            recommendations.append("- Implement automated testing for triggering accuracy")
            recommendations.append("- Consider refactoring pattern matching logic")

        return recommendations

    def generate_report(self) -> str:
        """Generate comprehensive triggering pattern report."""
        overall_score = self.calculate_overall_score()

        report = f"""# Comprehensive Triggering Pattern Analysis Report

**Report Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Overall Triggering Quality Score:** {overall_score:.1f}/100

## Executive Summary

"""

        # Determine overall status
        if overall_score >= 90:
            status = "✅ EXCELLENT"
            summary = "Triggering patterns are highly accurate and well-optimized across all categories."
        elif overall_score >= 80:
            status = "✅ GOOD"
            summary = "Triggering patterns are performing well with minor areas for improvement."
        elif overall_score >= 70:
            status = "⚠️ ACCEPTABLE"
            summary = "Triggering patterns are functional but have notable areas needing improvement."
        elif overall_score >= 60:
            status = "⚠️ NEEDS ATTENTION"
            summary = "Triggering patterns have significant issues that should be addressed."
        else:
            status = "❌ CRITICAL"
            summary = "Triggering patterns require immediate attention and optimization."

        report += f"**Status:** {status}\n\n"
        report += f"{summary}\n\n"

        # Key findings
        report += "### Key Findings\n\n"

        findings = []
        if self.metrics.activation_accuracy > 0:
            if self.metrics.activation_accuracy >= 90:
                findings.append(f"✓ Excellent plugin activation accuracy ({self.metrics.activation_accuracy:.1f}%)")
            elif self.metrics.activation_accuracy >= 75:
                findings.append(f"• Good plugin activation accuracy ({self.metrics.activation_accuracy:.1f}%)")
            else:
                findings.append(f"✗ Low plugin activation accuracy ({self.metrics.activation_accuracy:.1f}%)")

        if self.metrics.false_positive_rate > 0:
            if self.metrics.false_positive_rate < 5:
                findings.append(f"✓ False positive rate well below target ({self.metrics.false_positive_rate:.1f}%)")
            elif self.metrics.false_positive_rate < 10:
                findings.append(f"• False positive rate acceptable ({self.metrics.false_positive_rate:.1f}%)")
            else:
                findings.append(f"✗ False positive rate above target ({self.metrics.false_positive_rate:.1f}%)")

        if self.metrics.command_relevance > 0:
            if self.metrics.command_relevance >= 80:
                findings.append(f"✓ High command relevance accuracy ({self.metrics.command_relevance:.1f}%)")
            elif self.metrics.command_relevance >= 70:
                findings.append(f"• Moderate command relevance accuracy ({self.metrics.command_relevance:.1f}%)")
            else:
                findings.append(f"✗ Low command relevance accuracy ({self.metrics.command_relevance:.1f}%)")

        if self.metrics.skill_accuracy > 0:
            if self.metrics.skill_accuracy >= 90:
                findings.append(f"✓ Excellent skill pattern matching ({self.metrics.skill_accuracy:.1f}%)")
            elif self.metrics.skill_accuracy >= 80:
                findings.append(f"• Good skill pattern matching ({self.metrics.skill_accuracy:.1f}%)")
            else:
                findings.append(f"✗ Skill pattern matching needs improvement ({self.metrics.skill_accuracy:.1f}%)")

        for finding in findings:
            report += f"- {finding}\n"

        report += f"""
## Detailed Metrics

### Plugin Activation

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | {self.metrics.activation_accuracy:.1f}% | >90% | {self._status_icon(self.metrics.activation_accuracy > 90)} |
| Precision | {self.metrics.activation_precision:.1f}% | >90% | {self._status_icon(self.metrics.activation_precision > 90)} |
| Recall | {self.metrics.activation_recall:.1f}% | >90% | {self._status_icon(self.metrics.activation_recall > 90)} |
| False Positive Rate | {self.metrics.false_positive_rate:.1f}% | <5% | {self._status_icon(self.metrics.false_positive_rate < 5)} |
| False Negative Rate | {self.metrics.false_negative_rate:.1f}% | <5% | {self._status_icon(self.metrics.false_negative_rate < 5)} |

### Command Suggestions

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Relevance Accuracy | {self.metrics.command_relevance:.1f}% | >80% | {self._status_icon(self.metrics.command_relevance > 80)} |
| Timing Accuracy | {self.metrics.command_timing:.1f}% | >85% | {self._status_icon(self.metrics.command_timing > 85)} |
| Priority Accuracy | {self.metrics.command_priority:.1f}% | >90% | {self._status_icon(self.metrics.command_priority > 90)} |

### Skill Application

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | {self.metrics.skill_accuracy:.1f}% | >90% | {self._status_icon(self.metrics.skill_accuracy > 90)} |
| Precision | {self.metrics.skill_precision:.1f}% | >85% | {self._status_icon(self.metrics.skill_precision > 85)} |
| Over-Trigger Rate | {self.metrics.skill_over_trigger_rate:.1f}% | <10% | {self._status_icon(self.metrics.skill_over_trigger_rate < 10)} |
| Under-Trigger Rate | {self.metrics.skill_under_trigger_rate:.1f}% | <10% | {self._status_icon(self.metrics.skill_under_trigger_rate < 10)} |

## Issues Identified

"""

        # Group issues by severity
        critical = [i for i in self.issues if i.severity == "critical"]
        high = [i for i in self.issues if i.severity == "high"]
        medium = [i for i in self.issues if i.severity == "medium"]
        low = [i for i in self.issues if i.severity == "low"]

        report += f"**Summary:** {len(critical)} critical, {len(high)} high, {len(medium)} medium, {len(low)} low priority issues\n\n"

        if critical:
            report += "### Critical Issues\n\n"
            for issue in critical:
                report += f"**{issue.description}**\n"
                report += f"- Category: {issue.category}\n"
                report += f"- Impact: {issue.impact}\n"
                report += f"- Recommendation: {issue.recommendation}\n\n"

        if high:
            report += "### High Priority Issues\n\n"
            for issue in high:
                report += f"**{issue.description}**\n"
                report += f"- Category: {issue.category}\n"
                report += f"- Impact: {issue.impact}\n"
                report += f"- Recommendation: {issue.recommendation}\n\n"

        if medium:
            report += "### Medium Priority Issues\n\n"
            for issue in medium:
                report += f"- {issue.description}: {issue.recommendation}\n"

        if not self.issues:
            report += "No significant issues identified. All metrics are within acceptable ranges.\n"

        report += """
## Recommendations

"""

        recommendations = self.generate_recommendations()
        if recommendations:
            report += "\n".join(recommendations)
        else:
            report += "Continue monitoring triggering patterns and maintain current quality standards.\n"

        report += """

## Pattern Improvement Suggestions

Based on the analysis, consider these specific improvements:

"""

        suggestions = []

        if self.metrics.false_positive_rate >= 5:
            suggestions.append("**Reduce False Positives:**")
            suggestions.append("  - Add negative patterns to exclude common false positive triggers")
            suggestions.append("  - Increase specificity of file extension patterns")
            suggestions.append("  - Require minimum keyword threshold for activation")

        if self.metrics.false_negative_rate >= 5:
            suggestions.append("**Reduce False Negatives:**")
            suggestions.append("  - Expand keyword sets with synonyms and related terms")
            suggestions.append("  - Add alternative file extension patterns")
            suggestions.append("  - Lower activation thresholds for high-confidence patterns")

        if self.metrics.command_timing < 85:
            suggestions.append("**Improve Command Timing:**")
            suggestions.append("  - Implement workflow-stage detection")
            suggestions.append("  - Add project maturity indicators")
            suggestions.append("  - Context-aware command filtering")

        if self.metrics.skill_over_trigger_rate >= 10:
            suggestions.append("**Reduce Skill Over-Triggering:**")
            suggestions.append("  - Increase specificity of skill patterns")
            suggestions.append("  - Add skill-specific context requirements")
            suggestions.append("  - Implement skill relevance scoring")

        for suggestion in suggestions:
            report += f"{suggestion}\n"

        if not suggestions:
            report += "No specific pattern improvements needed at this time.\n"

        report += """
## Next Steps

1. **Immediate Actions:** Address critical and high priority issues
2. **Short-term:** Implement medium priority improvements
3. **Long-term:** Monitor metrics and iterate on patterns
4. **Continuous:** Run automated triggering tests in CI/CD

## Conclusion

"""

        if overall_score >= 85:
            report += "The triggering pattern system is performing well. Continue monitoring and make incremental improvements.\n"
        elif overall_score >= 70:
            report += "The triggering pattern system is functional but has room for improvement. Focus on addressing identified issues.\n"
        else:
            report += "The triggering pattern system requires significant attention. Prioritize addressing critical and high priority issues.\n"

        return report

    def _status_icon(self, condition: bool) -> str:
        """Return status icon based on condition."""
        return "✅" if condition else "❌"


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive triggering pattern report"
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Path to reports directory (default: reports)"
    )
    parser.add_argument(
        "--output",
        default="reports/triggering-comprehensive-report.md",
        help="Output report file (default: reports/triggering-comprehensive-report.md)"
    )

    args = parser.parse_args()

    # Resolve paths
    reports_dir = Path(args.reports_dir).absolute()

    if not reports_dir.exists():
        print(f"Error: Reports directory not found: {reports_dir}")
        print("Run the triggering analysis tools first:")
        print("  - activation_tester.py")
        print("  - command_analyzer.py")
        print("  - skill_validator.py")
        return 1

    # Generate report
    print("Generating comprehensive triggering pattern report...\n")

    reporter = TriggeringPatternReporter(str(reports_dir))

    # Load individual reports
    activation_loaded = reporter.load_activation_report()
    command_loaded = reporter.load_command_report()
    skill_loaded = reporter.load_skill_report()

    if not (activation_loaded or command_loaded or skill_loaded):
        print("\nError: No reports found. Run the triggering analysis tools first.")
        return 1

    # Generate comprehensive report
    report = reporter.generate_report()
    print("\n" + "=" * 70)
    print(report)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n✓ Comprehensive report saved to: {output_path.absolute()}")

    # Determine exit code
    overall_score = reporter.calculate_overall_score()
    if overall_score >= 80:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
