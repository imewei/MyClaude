#!/usr/bin/env python3
"""
Generate test matrix configurations for CI/CD pipelines.

This script generates test matrix configurations for different Python versions,
operating systems, and dependency versions.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


class TestMatrixGenerator:
    """Generate test matrix configurations."""

    def __init__(self):
        self.python_versions = ["3.10", "3.11", "3.12"]
        self.os_systems = ["ubuntu-latest", "macos-latest", "windows-latest"]
        self.dependency_sets = ["minimal", "standard", "full"]

    def generate_basic_matrix(self) -> Dict[str, List[str]]:
        """Generate basic test matrix with Python versions and OS."""
        return {
            "python-version": self.python_versions,
            "os": self.os_systems,
        }

    def generate_full_matrix(self) -> List[Dict[str, Any]]:
        """Generate full test matrix with all combinations."""
        matrix = []

        for py_version in self.python_versions:
            for os_system in self.os_systems:
                for dep_set in self.dependency_sets:
                    matrix.append({
                        "python-version": py_version,
                        "os": os_system,
                        "dependencies": dep_set,
                        "label": f"py{py_version}-{os_system.split('-')[0]}-{dep_set}",
                    })

        return matrix

    def generate_minimal_matrix(self) -> List[Dict[str, Any]]:
        """Generate minimal test matrix for quick checks."""
        return [
            {
                "python-version": "3.11",
                "os": "ubuntu-latest",
                "dependencies": "standard",
                "label": "quick-check",
            }
        ]

    def generate_coverage_matrix(self) -> List[Dict[str, Any]]:
        """Generate matrix specifically for coverage testing."""
        matrix = []

        for py_version in self.python_versions:
            matrix.append({
                "python-version": py_version,
                "os": "ubuntu-latest",
                "dependencies": "full",
                "coverage": True,
                "label": f"coverage-py{py_version}",
            })

        return matrix

    def generate_performance_matrix(self) -> List[Dict[str, Any]]:
        """Generate matrix for performance testing."""
        return [
            {
                "python-version": "3.11",
                "os": "ubuntu-latest",
                "dependencies": "full",
                "benchmark": True,
                "label": "performance-benchmark",
            }
        ]

    def generate_integration_matrix(self) -> List[Dict[str, Any]]:
        """Generate matrix for integration testing."""
        matrix = []

        for os_system in self.os_systems:
            matrix.append({
                "python-version": "3.11",
                "os": os_system,
                "dependencies": "full",
                "integration": True,
                "label": f"integration-{os_system.split('-')[0]}",
            })

        return matrix

    def export_github_actions_format(
        self, matrix_type: str = "basic"
    ) -> Dict[str, Any]:
        """Export matrix in GitHub Actions format."""
        if matrix_type == "basic":
            matrix = self.generate_basic_matrix()
        elif matrix_type == "full":
            matrix = {"include": self.generate_full_matrix()}
        elif matrix_type == "minimal":
            matrix = {"include": self.generate_minimal_matrix()}
        elif matrix_type == "coverage":
            matrix = {"include": self.generate_coverage_matrix()}
        elif matrix_type == "performance":
            matrix = {"include": self.generate_performance_matrix()}
        elif matrix_type == "integration":
            matrix = {"include": self.generate_integration_matrix()}
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

        return {"matrix": matrix}

    def export_gitlab_ci_format(self, matrix_type: str = "basic") -> List[Dict[str, Any]]:
        """Export matrix in GitLab CI format."""
        if matrix_type == "basic":
            basic = self.generate_basic_matrix()
            matrix = []
            for py_version in basic["python-version"]:
                for os_system in basic["os"]:
                    matrix.append({
                        "PYTHON_VERSION": py_version,
                        "OS": os_system,
                    })
            return matrix
        elif matrix_type == "full":
            matrix = []
            for item in self.generate_full_matrix():
                matrix.append({
                    "PYTHON_VERSION": item["python-version"],
                    "OS": item["os"],
                    "DEPENDENCIES": item["dependencies"],
                })
            return matrix
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test matrix configurations"
    )
    parser.add_argument(
        "--type",
        choices=["basic", "full", "minimal", "coverage", "performance", "integration"],
        default="basic",
        help="Type of matrix to generate",
    )
    parser.add_argument(
        "--format",
        choices=["github", "gitlab"],
        default="github",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    generator = TestMatrixGenerator()

    if args.format == "github":
        result = generator.export_github_actions_format(args.type)
    else:
        result = {"parallel": {"matrix": generator.export_gitlab_ci_format(args.type)}}

    output = json.dumps(result, indent=2)

    if args.output:
        args.output.write_text(output)
        print(f"Matrix configuration written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()