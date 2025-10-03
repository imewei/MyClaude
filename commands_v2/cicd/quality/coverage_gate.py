#!/usr/bin/env python3
"""
Coverage gate enforcement.

This script checks test coverage and enforces minimum thresholds.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


class CoverageGate:
    """Enforce coverage thresholds."""

    def __init__(self, threshold: float = 90.0):
        self.threshold = threshold
        self.violations = []

    def parse_coverage_xml(self, report_file: Path) -> Tuple[float, Dict[str, float]]:
        """Parse coverage.xml report."""
        tree = ET.parse(report_file)
        root = tree.getroot()

        # Get overall coverage
        coverage_elem = root.find(".//coverage")
        if coverage_elem is not None:
            line_rate = float(coverage_elem.get("line-rate", 0))
            overall = line_rate * 100
        else:
            overall = 0.0

        # Get per-package coverage
        packages = {}
        for package in root.findall(".//package"):
            name = package.get("name", "unknown")
            line_rate = float(package.get("line-rate", 0))
            packages[name] = line_rate * 100

        return overall, packages

    def check_overall_coverage(self, coverage: float) -> bool:
        """Check overall coverage threshold."""
        print(f"\nOverall Coverage: {coverage:.2f}%")

        if coverage < self.threshold:
            print(f"  ❌ Coverage {coverage:.2f}% is below threshold {self.threshold}%")
            self.violations.append(
                f"Overall coverage {coverage:.2f}% < {self.threshold}%"
            )
            return False

        print(f"  ✓ Coverage meets threshold ({self.threshold}%)")
        return True

    def check_package_coverage(
        self,
        packages: Dict[str, float],
        package_threshold: float,
    ) -> bool:
        """Check per-package coverage."""
        print(f"\nPackage Coverage (threshold: {package_threshold}%):")

        all_passed = True

        for name, coverage in sorted(packages.items(), key=lambda x: x[1]):
            if coverage < package_threshold:
                print(f"  ❌ {name}: {coverage:.2f}%")
                self.violations.append(
                    f"Package {name} coverage {coverage:.2f}% < {package_threshold}%"
                )
                all_passed = False
            else:
                print(f"  ✓ {name}: {coverage:.2f}%")

        return all_passed

    def print_summary(self) -> None:
        """Print coverage gate summary."""
        print("\n" + "=" * 60)
        print("COVERAGE GATE SUMMARY")
        print("=" * 60)

        if self.violations:
            print(f"❌ COVERAGE GATE FAILED ({len(self.violations)} violations)")
            print("\nViolations:")
            for violation in self.violations:
                print(f"  - {violation}")
        else:
            print("✅ COVERAGE GATE PASSED")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enforce coverage gates")
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Coverage XML report file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Overall coverage threshold (default: 90)",
    )
    parser.add_argument(
        "--package-threshold",
        type=float,
        default=80.0,
        help="Per-package coverage threshold (default: 80)",
    )

    args = parser.parse_args()

    if not args.report.exists():
        print(f"ERROR: Coverage report not found: {args.report}")
        sys.exit(1)

    gate = CoverageGate(args.threshold)

    try:
        overall, packages = gate.parse_coverage_xml(args.report)

        overall_passed = gate.check_overall_coverage(overall)
        packages_passed = gate.check_package_coverage(
            packages,
            args.package_threshold,
        )

        gate.print_summary()

        if not (overall_passed and packages_passed):
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()