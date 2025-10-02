#!/usr/bin/env python3
"""
Test Runner Utilities for Command Executors
Provides unified test execution across multiple frameworks
"""

import subprocess
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


class TestFramework(Enum):
    """Supported test frameworks"""
    PYTEST = "pytest"
    JEST = "jest"
    CARGO = "cargo"
    GO_TEST = "go test"
    JULIA = "julia"
    CTEST = "ctest"
    UNKNOWN = "unknown"


class TestResult:
    """Test execution result"""
    def __init__(self, framework: TestFramework):
        self.framework = framework
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.total = 0
        self.duration = 0.0
        self.coverage = 0.0
        self.failures: List[Dict[str, str]] = []
        self.output = ""
        self.success = False

    def to_dict(self) -> Dict:
        return {
            'framework': self.framework.value,
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'total': self.total,
            'duration': self.duration,
            'coverage': self.coverage,
            'failures': self.failures,
            'success': self.success
        }


class TestRunner:
    """Unified test runner for multiple frameworks"""

    def __init__(self, work_dir: Path = None):
        self.work_dir = work_dir or Path.cwd()

    def detect_framework(self) -> TestFramework:
        """
        Auto-detect test framework based on project files

        Returns:
            Detected test framework
        """
        # Check for pytest
        if (self.work_dir / 'pytest.ini').exists() or \
           (self.work_dir / 'pyproject.toml').exists() or \
           list(self.work_dir.rglob('test_*.py')) or \
           list(self.work_dir.rglob('*_test.py')):
            return TestFramework.PYTEST

        # Check for Jest
        if (self.work_dir / 'jest.config.js').exists() or \
           (self.work_dir / 'jest.config.ts').exists() or \
           list(self.work_dir.rglob('*.test.js')) or \
           list(self.work_dir.rglob('*.spec.js')):
            return TestFramework.JEST

        # Check for Cargo (Rust)
        if (self.work_dir / 'Cargo.toml').exists():
            return TestFramework.CARGO

        # Check for Go
        if list(self.work_dir.rglob('*_test.go')):
            return TestFramework.GO_TEST

        # Check for Julia
        if (self.work_dir / 'Project.toml').exists() and \
           (self.work_dir / 'test').is_dir():
            return TestFramework.JULIA

        # Check for CMake/CTest
        if (self.work_dir / 'CMakeLists.txt').exists():
            return TestFramework.CTEST

        return TestFramework.UNKNOWN

    def run_tests(self, framework: Optional[TestFramework] = None,
                  scope: str = 'all', coverage: bool = False,
                  parallel: bool = False, verbose: bool = False) -> TestResult:
        """
        Run tests with specified framework

        Args:
            framework: Test framework (auto-detect if None)
            scope: Test scope (all, unit, integration, performance)
            coverage: Enable coverage reporting
            parallel: Run tests in parallel
            verbose: Enable verbose output

        Returns:
            TestResult object
        """
        if framework is None:
            framework = self.detect_framework()

        if framework == TestFramework.PYTEST:
            return self._run_pytest(scope, coverage, parallel, verbose)
        elif framework == TestFramework.JEST:
            return self._run_jest(scope, coverage, parallel, verbose)
        elif framework == TestFramework.CARGO:
            return self._run_cargo(scope, verbose)
        elif framework == TestFramework.GO_TEST:
            return self._run_go_test(scope, coverage, verbose)
        elif framework == TestFramework.JULIA:
            return self._run_julia(scope, coverage)
        elif framework == TestFramework.CTEST:
            return self._run_ctest(scope, verbose)
        else:
            raise ValueError(f"Unsupported test framework: {framework}")

    def _run_pytest(self, scope: str, coverage: bool, parallel: bool, verbose: bool) -> TestResult:
        """Run pytest tests"""
        args = ['pytest']

        # Scope filtering
        if scope == 'unit':
            args.extend(['-m', 'unit'])
        elif scope == 'integration':
            args.extend(['-m', 'integration'])
        elif scope == 'performance':
            args.extend(['-m', 'performance'])

        # Coverage
        if coverage:
            args.extend(['--cov', '--cov-report=term', '--cov-report=json'])

        # Parallel execution
        if parallel:
            args.extend(['-n', 'auto'])

        # Verbose
        if verbose:
            args.append('-v')

        # JSON output for parsing
        args.extend(['--json-report', '--json-report-file=/tmp/pytest_report.json'])

        result = TestResult(TestFramework.PYTEST)

        try:
            proc = subprocess.run(
                args,
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )

            result.output = proc.stdout + proc.stderr

            # Parse JSON report if available
            report_file = Path('/tmp/pytest_report.json')
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                    result.passed = report['summary'].get('passed', 0)
                    result.failed = report['summary'].get('failed', 0)
                    result.skipped = report['summary'].get('skipped', 0)
                    result.total = report['summary'].get('total', 0)
                    result.duration = report['summary'].get('duration', 0.0)

                    # Extract failures
                    for test in report.get('tests', []):
                        if test['outcome'] == 'failed':
                            result.failures.append({
                                'name': test['nodeid'],
                                'message': test.get('call', {}).get('longrepr', ''),
                                'type': 'failure'
                            })

            # Parse coverage from coverage.json if available
            cov_file = self.work_dir / 'coverage.json'
            if coverage and cov_file.exists():
                with open(cov_file) as f:
                    cov_data = json.load(f)
                    result.coverage = cov_data.get('totals', {}).get('percent_covered', 0.0)

            result.success = proc.returncode == 0

        except Exception as e:
            result.output = str(e)
            result.success = False

        return result

    def _run_jest(self, scope: str, coverage: bool, parallel: bool, verbose: bool) -> TestResult:
        """Run Jest tests"""
        args = ['npm', 'test', '--']

        if coverage:
            args.append('--coverage')

        if verbose:
            args.append('--verbose')

        if not parallel:
            args.append('--runInBand')

        # JSON output
        args.extend(['--json', '--outputFile=/tmp/jest_results.json'])

        result = TestResult(TestFramework.JEST)

        try:
            proc = subprocess.run(
                args,
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )

            result.output = proc.stdout + proc.stderr

            # Parse JSON results
            results_file = Path('/tmp/jest_results.json')
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    result.passed = data.get('numPassedTests', 0)
                    result.failed = data.get('numFailedTests', 0)
                    result.total = data.get('numTotalTests', 0)

            result.success = proc.returncode == 0

        except Exception as e:
            result.output = str(e)
            result.success = False

        return result

    def _run_cargo(self, scope: str, verbose: bool) -> TestResult:
        """Run Cargo (Rust) tests"""
        args = ['cargo', 'test']

        if verbose:
            args.append('--verbose')

        result = TestResult(TestFramework.CARGO)

        try:
            proc = subprocess.run(
                args,
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )

            result.output = proc.stdout + proc.stderr

            # Parse output for test results
            # Cargo output format: "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
            match = re.search(r'(\d+) passed; (\d+) failed', result.output)
            if match:
                result.passed = int(match.group(1))
                result.failed = int(match.group(2))
                result.total = result.passed + result.failed

            result.success = proc.returncode == 0

        except Exception as e:
            result.output = str(e)
            result.success = False

        return result

    def _run_go_test(self, scope: str, coverage: bool, verbose: bool) -> TestResult:
        """Run Go tests"""
        args = ['go', 'test', './...']

        if coverage:
            args.extend(['-cover', '-coverprofile=coverage.out'])

        if verbose:
            args.append('-v')

        result = TestResult(TestFramework.GO_TEST)

        try:
            proc = subprocess.run(
                args,
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )

            result.output = proc.stdout + proc.stderr

            # Parse output for test results
            for line in result.output.splitlines():
                if 'PASS' in line or 'ok' in line:
                    result.passed += 1
                elif 'FAIL' in line:
                    result.failed += 1

            result.total = result.passed + result.failed
            result.success = proc.returncode == 0

        except Exception as e:
            result.output = str(e)
            result.success = False

        return result

    def _run_julia(self, scope: str, coverage: bool) -> TestResult:
        """Run Julia tests"""
        args = ['julia', '--project=.', '-e', 'using Pkg; Pkg.test()']

        result = TestResult(TestFramework.JULIA)

        try:
            proc = subprocess.run(
                args,
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )

            result.output = proc.stdout + proc.stderr
            result.success = proc.returncode == 0

            # Julia test output parsing would go here
            # For now, just check success/failure

        except Exception as e:
            result.output = str(e)
            result.success = False

        return result

    def _run_ctest(self, scope: str, verbose: bool) -> TestResult:
        """Run CTest (CMake) tests"""
        args = ['ctest']

        if verbose:
            args.append('--verbose')

        result = TestResult(TestFramework.CTEST)

        try:
            proc = subprocess.run(
                args,
                cwd=self.work_dir / 'build',  # Assume tests are in build directory
                capture_output=True,
                text=True
            )

            result.output = proc.stdout + proc.stderr
            result.success = proc.returncode == 0

        except Exception as e:
            result.output = str(e)
            result.success = False

        return result

    def get_failed_tests(self, result: TestResult) -> List[str]:
        """Extract list of failed test names"""
        return [f['name'] for f in result.failures]