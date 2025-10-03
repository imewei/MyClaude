#!/usr/bin/env python3
"""
Security gate enforcement.

This script checks security scan results and enforces thresholds.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


class SecurityGate:
    """Enforce security thresholds."""

    SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(
        self,
        max_critical: int = 0,
        max_high: int = 0,
        max_medium: int = 5,
    ):
        self.max_critical = max_critical
        self.max_high = max_high
        self.max_medium = max_medium
        self.violations = []
        self.findings = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}

    def check_vulnerability_report(self, report_file: Path) -> bool:
        """Check Safety/pip-audit vulnerability report."""
        print(f"\nChecking vulnerability report: {report_file.name}")

        with open(report_file) as f:
            data = json.load(f)

        # Parse based on format
        if isinstance(data, list):
            # Safety format
            for vuln in data:
                severity = vuln.get("severity", "UNKNOWN").upper()
                if severity not in self.findings:
                    severity = "MEDIUM"

                self.findings[severity].append({
                    "type": "vulnerability",
                    "package": vuln.get("package", "unknown"),
                    "description": vuln.get("advisory", "No description"),
                    "id": vuln.get("id", ""),
                })
        elif isinstance(data, dict) and "vulnerabilities" in data:
            # pip-audit format
            for vuln in data["vulnerabilities"]:
                severity = vuln.get("severity", "MEDIUM").upper()
                if severity not in self.findings:
                    severity = "MEDIUM"

                self.findings[severity].append({
                    "type": "vulnerability",
                    "package": vuln.get("name", "unknown"),
                    "description": vuln.get("description", "No description"),
                    "id": vuln.get("id", ""),
                })

        return self._check_thresholds()

    def check_sast_report(self, report_file: Path) -> bool:
        """Check Bandit/Semgrep SAST report."""
        print(f"\nChecking SAST report: {report_file.name}")

        with open(report_file) as f:
            data = json.load(f)

        # Parse based on format
        if "results" in data:
            # Bandit format
            for issue in data["results"]:
                severity = issue.get("issue_severity", "MEDIUM").upper()
                if severity not in self.findings:
                    severity = "MEDIUM"

                self.findings[severity].append({
                    "type": "sast",
                    "file": issue.get("filename", "unknown"),
                    "line": issue.get("line_number", 0),
                    "description": issue.get("issue_text", ""),
                    "id": issue.get("test_id", ""),
                })
        elif "results" in data and isinstance(data["results"], list):
            # Semgrep format
            for finding in data["results"]:
                severity = finding.get("extra", {}).get("severity", "MEDIUM").upper()
                if severity == "WARNING":
                    severity = "MEDIUM"
                elif severity == "ERROR":
                    severity = "HIGH"

                self.findings[severity].append({
                    "type": "sast",
                    "file": finding.get("path", "unknown"),
                    "line": finding.get("start", {}).get("line", 0),
                    "description": finding.get("extra", {}).get("message", ""),
                    "id": finding.get("check_id", ""),
                })

        return self._check_thresholds()

    def _check_thresholds(self) -> bool:
        """Check if findings are within thresholds."""
        critical_count = len(self.findings["CRITICAL"])
        high_count = len(self.findings["HIGH"])
        medium_count = len(self.findings["MEDIUM"])

        all_passed = True

        if critical_count > self.max_critical:
            self.violations.append(
                f"Critical issues: {critical_count} (max: {self.max_critical})"
            )
            all_passed = False

        if high_count > self.max_high:
            self.violations.append(
                f"High severity issues: {high_count} (max: {self.max_high})"
            )
            all_passed = False

        if medium_count > self.max_medium:
            self.violations.append(
                f"Medium severity issues: {medium_count} (max: {self.max_medium})"
            )
            all_passed = False

        return all_passed

    def print_findings(self) -> None:
        """Print security findings."""
        print("\n" + "=" * 60)
        print("SECURITY FINDINGS")
        print("=" * 60)

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = len(self.findings[severity])
            if count > 0:
                print(f"\n{severity}: {count}")
                for finding in self.findings[severity][:5]:  # Show first 5
                    print(f"  - [{finding['type']}] {finding['description'][:80]}")
                if count > 5:
                    print(f"  ... and {count - 5} more")

    def print_summary(self) -> None:
        """Print security gate summary."""
        print("\n" + "=" * 60)
        print("SECURITY GATE SUMMARY")
        print("=" * 60)

        total_findings = sum(len(findings) for findings in self.findings.values())
        print(f"Total findings: {total_findings}")
        print(f"  Critical: {len(self.findings['CRITICAL'])} (max: {self.max_critical})")
        print(f"  High: {len(self.findings['HIGH'])} (max: {self.max_high})")
        print(f"  Medium: {len(self.findings['MEDIUM'])} (max: {self.max_medium})")
        print(f"  Low: {len(self.findings['LOW'])}")

        if self.violations:
            print(f"\n❌ SECURITY GATE FAILED ({len(self.violations)} violations)")
            print("\nViolations:")
            for violation in self.violations:
                print(f"  - {violation}")
        else:
            print("\n✅ SECURITY GATE PASSED")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enforce security gates")
    parser.add_argument(
        "--vulnerability-report",
        type=Path,
        help="Vulnerability scan report (Safety/pip-audit JSON)",
    )
    parser.add_argument(
        "--sast-report",
        type=Path,
        help="SAST report (Bandit/Semgrep JSON)",
    )
    parser.add_argument(
        "--max-critical",
        type=int,
        default=0,
        help="Maximum critical issues allowed",
    )
    parser.add_argument(
        "--max-high",
        type=int,
        default=0,
        help="Maximum high severity issues allowed",
    )
    parser.add_argument(
        "--max-medium",
        type=int,
        default=5,
        help="Maximum medium severity issues allowed",
    )

    args = parser.parse_args()

    gate = SecurityGate(
        max_critical=args.max_critical,
        max_high=args.max_high,
        max_medium=args.max_medium,
    )

    try:
        all_passed = True

        if args.vulnerability_report and args.vulnerability_report.exists():
            if not gate.check_vulnerability_report(args.vulnerability_report):
                all_passed = False

        if args.sast_report and args.sast_report.exists():
            if not gate.check_sast_report(args.sast_report):
                all_passed = False

        gate.print_findings()
        gate.print_summary()

        if not all_passed:
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()