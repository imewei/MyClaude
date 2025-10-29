#!/usr/bin/env python3
"""
Master validation orchestrator that runs all automated checks.

This script coordinates the execution of all validation dimensions:
1. Linting and formatting
2. Type checking
3. Unit tests with coverage
4. Security scanning
5. Build verification
6. Accessibility testing (if applicable)

Usage:
    python run_all_validations.py [--skip-security] [--skip-tests] [--skip-build] [--verbose]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass, asdict


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    duration_seconds: float
    output: str
    error: str = ""
    skipped: bool = False


class ValidationOrchestrator:
    """Orchestrates all validation checks."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.root_dir = Path.cwd()

    def run_command(self, cmd: List[str], name: str, timeout: int = 300) -> ValidationResult:
        """Run a command and capture its result."""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Running: {name}")
            print(f"Command: {' '.join(cmd)}")
            print(f"{'='*80}\n")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.root_dir
            )
            duration = time.time() - start_time

            passed = result.returncode == 0
            output = result.stdout + result.stderr

            return ValidationResult(
                name=name,
                passed=passed,
                duration_seconds=duration,
                output=output if self.verbose else output[:1000],  # Truncate if not verbose
                error="" if passed else result.stderr
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ValidationResult(
                name=name,
                passed=False,
                duration_seconds=duration,
                output="",
                error=f"Command timed out after {timeout} seconds"
            )

        except FileNotFoundError:
            duration = time.time() - start_time
            return ValidationResult(
                name=name,
                passed=False,
                duration_seconds=duration,
                output="",
                error=f"Command not found: {cmd[0]}"
            )

    def detect_project_type(self) -> str:
        """Detect the project type based on files present."""
        if (self.root_dir / "package.json").exists():
            return "javascript"
        elif (self.root_dir / "pyproject.toml").exists() or (self.root_dir / "setup.py").exists():
            return "python"
        elif (self.root_dir / "Cargo.toml").exists():
            return "rust"
        elif (self.root_dir / "go.mod").exists():
            return "go"
        else:
            return "unknown"

    def run_linting(self) -> ValidationResult:
        """Run linting checks based on project type."""
        project_type = self.detect_project_type()

        if project_type == "javascript":
            # Try eslint first, then prettier
            if (self.root_dir / "node_modules" / ".bin" / "eslint").exists():
                return self.run_command(["npx", "eslint", "."], "ESLint Check")
            elif (self.root_dir / "node_modules" / ".bin" / "prettier").exists():
                return self.run_command(["npx", "prettier", "--check", "."], "Prettier Check")

        elif project_type == "python":
            # Try ruff first, then flake8
            try:
                return self.run_command(["ruff", "check", "."], "Ruff Lint Check")
            except:
                try:
                    return self.run_command(["flake8", "."], "Flake8 Check")
                except:
                    pass

        elif project_type == "rust":
            return self.run_command(["cargo", "clippy", "--", "-D", "warnings"], "Cargo Clippy")

        elif project_type == "go":
            return self.run_command(["golangci-lint", "run"], "GolangCI-Lint")

        return ValidationResult(
            name="Linting", passed=True, duration_seconds=0,
            output="No linter configured", skipped=True
        )

    def run_formatting(self) -> ValidationResult:
        """Run formatting checks."""
        project_type = self.detect_project_type()

        if project_type == "javascript":
            if (self.root_dir / "node_modules" / ".bin" / "prettier").exists():
                return self.run_command(["npx", "prettier", "--check", "."], "Prettier Format Check")

        elif project_type == "python":
            try:
                return self.run_command(["black", "--check", "."], "Black Format Check")
            except:
                pass

        elif project_type == "rust":
            return self.run_command(["cargo", "fmt", "--", "--check"], "Cargo Format Check")

        elif project_type == "go":
            return self.run_command(["gofmt", "-l", "."], "Go Format Check")

        return ValidationResult(
            name="Formatting", passed=True, duration_seconds=0,
            output="No formatter configured", skipped=True
        )

    def run_type_checking(self) -> ValidationResult:
        """Run type checking if available."""
        project_type = self.detect_project_type()

        if project_type == "javascript":
            if (self.root_dir / "tsconfig.json").exists():
                return self.run_command(["npx", "tsc", "--noEmit"], "TypeScript Type Check")

        elif project_type == "python":
            if (self.root_dir / "pyproject.toml").exists():
                try:
                    return self.run_command(["mypy", "."], "MyPy Type Check")
                except:
                    pass

        elif project_type == "rust":
            return self.run_command(["cargo", "check"], "Cargo Type Check")

        return ValidationResult(
            name="Type Checking", passed=True, duration_seconds=0,
            output="No type checker configured", skipped=True
        )

    def run_tests(self) -> ValidationResult:
        """Run tests with coverage."""
        project_type = self.detect_project_type()

        if project_type == "javascript":
            if (self.root_dir / "package.json").exists():
                # Try npm test first
                return self.run_command(["npm", "test", "--", "--coverage"], "Tests with Coverage", timeout=600)

        elif project_type == "python":
            # Try pytest with coverage
            try:
                return self.run_command(
                    ["pytest", "--cov", "--cov-report=term", "--cov-report=html"],
                    "Pytest with Coverage",
                    timeout=600
                )
            except:
                try:
                    return self.run_command(["pytest"], "Pytest", timeout=600)
                except:
                    pass

        elif project_type == "rust":
            return self.run_command(["cargo", "test"], "Cargo Tests", timeout=600)

        elif project_type == "go":
            return self.run_command(["go", "test", "-cover", "./..."], "Go Tests with Coverage", timeout=600)

        return ValidationResult(
            name="Tests", passed=True, duration_seconds=0,
            output="No tests configured", skipped=True
        )

    def run_build(self) -> ValidationResult:
        """Run build verification."""
        project_type = self.detect_project_type()

        if project_type == "javascript":
            if (self.root_dir / "package.json").exists():
                with open(self.root_dir / "package.json") as f:
                    package = json.load(f)
                    if "build" in package.get("scripts", {}):
                        return self.run_command(["npm", "run", "build"], "Build", timeout=600)

        elif project_type == "python":
            if (self.root_dir / "pyproject.toml").exists():
                return self.run_command(["python", "-m", "build"], "Python Build", timeout=600)

        elif project_type == "rust":
            return self.run_command(["cargo", "build", "--release"], "Cargo Build", timeout=600)

        elif project_type == "go":
            return self.run_command(["go", "build", "./..."], "Go Build", timeout=600)

        return ValidationResult(
            name="Build", passed=True, duration_seconds=0,
            output="No build configured", skipped=True
        )

    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        skipped = sum(1 for r in self.results if r.skipped)
        failed = total - passed - skipped
        total_duration = sum(r.duration_seconds for r in self.results)

        report = []
        report.append("\n" + "="*80)
        report.append("VALIDATION SUMMARY")
        report.append("="*80)
        report.append(f"\nTotal Checks: {total}")
        report.append(f"✅ Passed: {passed}")
        report.append(f"❌ Failed: {failed}")
        report.append(f"⏭️  Skipped: {skipped}")
        report.append(f"⏱️  Total Duration: {total_duration:.2f}s")
        report.append("\n" + "-"*80)
        report.append("DETAILED RESULTS")
        report.append("-"*80 + "\n")

        for result in self.results:
            status = "✅ PASS" if result.passed else ("⏭️  SKIP" if result.skipped else "❌ FAIL")
            report.append(f"{status:12} {result.name:30} ({result.duration_seconds:.2f}s)")

            if not result.passed and not result.skipped and result.error:
                report.append(f"  Error: {result.error[:200]}")

        report.append("\n" + "="*80)

        if failed == 0:
            report.append("🎉 All validations passed!")
        else:
            report.append(f"⚠️  {failed} validation(s) failed. Review the errors above.")

        report.append("="*80 + "\n")

        return "\n".join(report)

    def run_all(self, skip_security: bool = False, skip_tests: bool = False, skip_build: bool = False):
        """Run all validations."""
        print("\n🚀 Starting Comprehensive Validation\n")

        # Phase 1: Linting and Formatting
        print("📋 Phase 1: Code Quality Checks")
        self.results.append(self.run_linting())
        self.results.append(self.run_formatting())
        self.results.append(self.run_type_checking())

        # Phase 2: Testing
        if not skip_tests:
            print("\n🧪 Phase 2: Testing")
            self.results.append(self.run_tests())

        # Phase 3: Security
        if not skip_security:
            print("\n🔒 Phase 3: Security Scanning")
            print("(Running security_scan.py...)")
            # This will be called separately as it's more complex
            security_result = subprocess.run(
                [sys.executable, Path(__file__).parent / "security_scan.py"],
                capture_output=True,
                text=True
            )
            self.results.append(ValidationResult(
                name="Security Scan",
                passed=security_result.returncode == 0,
                duration_seconds=0,
                output=security_result.stdout,
                error=security_result.stderr
            ))

        # Phase 4: Build
        if not skip_build:
            print("\n🏗️  Phase 4: Build Verification")
            self.results.append(self.run_build())

        # Generate and print report
        report = self.generate_report()
        print(report)

        # Exit with appropriate code
        failed_count = sum(1 for r in self.results if not r.passed and not r.skipped)
        sys.exit(0 if failed_count == 0 else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive validation checks across all dimensions"
    )
    parser.add_argument("--skip-security", action="store_true", help="Skip security scanning")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--skip-build", action="store_true", help="Skip build verification")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    orchestrator = ValidationOrchestrator(verbose=args.verbose)
    orchestrator.run_all(
        skip_security=args.skip_security,
        skip_tests=args.skip_tests,
        skip_build=args.skip_build
    )


if __name__ == "__main__":
    main()
