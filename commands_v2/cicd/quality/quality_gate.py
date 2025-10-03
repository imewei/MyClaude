#!/usr/bin/env python3
"""
Quality gate enforcement for CI/CD pipeline.

This script checks various quality metrics and enforces thresholds.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


class QualityGate:
    """Enforce quality gates."""

    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

    def check_complexity(
        self,
        complexity_file: Path,
        max_complexity: int = 10,
        max_average: float = 5.0,
    ) -> bool:
        """Check cyclomatic complexity."""
        print(f"\nChecking cyclomatic complexity...")

        with open(complexity_file) as f:
            data = json.load(f)

        violations = []
        total_complexity = 0
        count = 0

        for item in data:
            complexity = item.get("complexity", 0)
            total_complexity += complexity
            count += 1

            if complexity > max_complexity:
                violations.append(
                    f"{item['name']} has complexity {complexity} (max: {max_complexity})"
                )

        average = total_complexity / count if count > 0 else 0

        if violations:
            self.checks_failed.append("Complexity check")
            for v in violations[:10]:  # Show first 10
                print(f"  ❌ {v}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more")
            return False

        if average > max_average:
            self.warnings.append(
                f"Average complexity {average:.2f} exceeds {max_average}"
            )

        self.checks_passed.append("Complexity check")
        print(f"  ✓ Complexity check passed (avg: {average:.2f})")
        return True

    def check_maintainability(
        self,
        maintainability_file: Path,
        min_mi: float = 20.0,
    ) -> bool:
        """Check maintainability index."""
        print(f"\nChecking maintainability index...")

        with open(maintainability_file) as f:
            data = json.load(f)

        violations = []

        for item in data:
            mi = item.get("mi", 0)
            if mi < min_mi:
                violations.append(
                    f"{item['name']} has MI {mi:.2f} (min: {min_mi})"
                )

        if violations:
            self.checks_failed.append("Maintainability check")
            for v in violations[:10]:
                print(f"  ❌ {v}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more")
            return False

        self.checks_passed.append("Maintainability check")
        print(f"  ✓ Maintainability check passed")
        return True

    def generate_report(self, output_file: Path) -> None:
        """Generate quality report."""
        report = {
            "passed": len(self.checks_passed),
            "failed": len(self.checks_failed),
            "warnings": len(self.warnings),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "overall": "PASSED" if not self.checks_failed else "FAILED",
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nQuality report written to {output_file}")

    def print_summary(self) -> None:
        """Print quality gate summary."""
        print("\n" + "=" * 60)
        print("QUALITY GATE SUMMARY")
        print("=" * 60)
        print(f"Passed: {len(self.checks_passed)}")
        print(f"Failed: {len(self.checks_failed)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.checks_failed:
            print("\n❌ QUALITY GATE FAILED")
            print("\nFailed checks:")
            for check in self.checks_failed:
                print(f"  - {check}")
        else:
            print("\n✅ QUALITY GATE PASSED")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enforce quality gates")
    parser.add_argument(
        "--complexity",
        type=Path,
        help="Complexity report file",
    )
    parser.add_argument(
        "--maintainability",
        type=Path,
        help="Maintainability report file",
    )
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=10,
        help="Maximum cyclomatic complexity",
    )
    parser.add_argument(
        "--max-average-complexity",
        type=float,
        default=5.0,
        help="Maximum average complexity",
    )
    parser.add_argument(
        "--min-mi",
        type=float,
        default=20.0,
        help="Minimum maintainability index",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("quality_report.json"),
        help="Output report file",
    )

    args = parser.parse_args()

    gate = QualityGate()

    try:
        all_passed = True

        if args.complexity:
            if not gate.check_complexity(
                args.complexity,
                args.max_complexity,
                args.max_average_complexity,
            ):
                all_passed = False

        if args.maintainability:
            if not gate.check_maintainability(
                args.maintainability,
                args.min_mi,
            ):
                all_passed = False

        gate.generate_report(args.output)
        gate.print_summary()

        if not all_passed:
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()