#!/usr/bin/env python3
"""
Cross-language test runner with coverage analysis.

Detects project type and runs appropriate test framework with coverage.

Usage:
    python test_runner.py [--min-coverage 80] [--html-report]
"""

import argparse
import json
import subprocess
import sys
import re
from pathlib import Path


class TestRunner:
    """Runs tests across different project types with coverage."""

    def __init__(self, min_coverage: float = 0.0, html_report: bool = False):
        self.min_coverage = min_coverage
        self.html_report = html_report
        self.root_dir = Path.cwd()

    def detect_project_type(self) -> str:
        """Detect the project type."""
        if (self.root_dir / "package.json").exists():
            return "javascript"
        elif (self.root_dir / "pyproject.toml").exists() or (self.root_dir / "setup.py").exists():
            return "python"
        elif (self.root_dir / "Cargo.toml").exists():
            return "rust"
        elif (self.root_dir / "go.mod").exists():
            return "go"
        return "unknown"

    def run_javascript_tests(self):
        """Run JavaScript/TypeScript tests with coverage."""
        print("Running JavaScript/TypeScript tests with coverage...")

        # Try Jest first
        cmd = ["npm", "test", "--", "--coverage", "--watchAll=false"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Parse coverage from output
        coverage_match = re.search(r"All files.*?\|\s*(\d+\.?\d*)", result.stdout)
        if coverage_match:
            coverage = float(coverage_match.group(1))
            print(f"\n📊 Overall Coverage: {coverage}%")

            if coverage < self.min_coverage:
                print(f"❌ Coverage {coverage}% is below minimum {self.min_coverage}%")
                sys.exit(1)

        return result.returncode

    def run_python_tests(self):
        """Run Python tests with coverage."""
        print("Running Python tests with coverage...")

        cmd = ["pytest", "--cov", "--cov-report=term"]
        if self.html_report:
            cmd.append("--cov-report=html")

        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Parse coverage from pytest-cov output
        coverage_match = re.search(r"TOTAL.*?\s+(\d+)%", result.stdout)
        if coverage_match:
            coverage = int(coverage_match.group(1))
            print(f"\n📊 Overall Coverage: {coverage}%")

            if coverage < self.min_coverage:
                print(f"❌ Coverage {coverage}% is below minimum {self.min_coverage}%")
                sys.exit(1)

        return result.returncode

    def run_rust_tests(self):
        """Run Rust tests."""
        print("Running Rust tests...")

        cmd = ["cargo", "test"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        return result.returncode

    def run_go_tests(self):
        """Run Go tests with coverage."""
        print("Running Go tests with coverage...")

        cmd = ["go", "test", "-cover", "-coverprofile=coverage.out", "./..."]
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Parse coverage
        coverage_match = re.search(r"coverage:\s+(\d+\.?\d*)%", result.stdout)
        if coverage_match:
            coverage = float(coverage_match.group(1))
            print(f"\n📊 Overall Coverage: {coverage}%")

            if coverage < self.min_coverage:
                print(f"❌ Coverage {coverage}% is below minimum {self.min_coverage}%")
                sys.exit(1)

        return result.returncode

    def run(self):
        """Run tests based on project type."""
        project_type = self.detect_project_type()

        if project_type == "javascript":
            return self.run_javascript_tests()
        elif project_type == "python":
            return self.run_python_tests()
        elif project_type == "rust":
            return self.run_rust_tests()
        elif project_type == "go":
            return self.run_go_tests()
        else:
            print("❌ Unknown project type - no tests found")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Run tests with coverage analysis")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="Minimum required coverage percentage")
    parser.add_argument("--html-report", action="store_true",
                        help="Generate HTML coverage report")

    args = parser.parse_args()

    runner = TestRunner(min_coverage=args.min_coverage, html_report=args.html_report)
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
