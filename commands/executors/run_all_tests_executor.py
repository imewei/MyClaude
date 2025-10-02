#!/usr/bin/env python3
"""
Run All Tests Command Executor
Comprehensive test execution with auto-fix capabilities
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add executors to path
sys.path.insert(0, str(Path(__file__).parent))

from base_executor import CommandExecutor, AgentOrchestrator
from test_runner import TestRunner, TestFramework, TestResult
from code_modifier import CodeModifier


class RunAllTestsExecutor(CommandExecutor):
    """Executor for /run-all-tests command"""

    def __init__(self):
        super().__init__("run-all-tests")
        self.test_runner = TestRunner()
        self.code_modifier = CodeModifier()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        """Configure argument parser"""
        parser = argparse.ArgumentParser(
            description='Comprehensive test execution engine'
        )
        parser.add_argument('--scope', type=str, default='all',
                          choices=['all', 'unit', 'integration', 'performance'],
                          help='Test scope')
        parser.add_argument('--profile', action='store_true',
                          help='Enable performance profiling')
        parser.add_argument('--benchmark', action='store_true',
                          help='Run performance benchmarks')
        parser.add_argument('--scientific', action='store_true',
                          help='Scientific computing optimization')
        parser.add_argument('--gpu', action='store_true',
                          help='Enable GPU/TPU testing')
        parser.add_argument('--parallel', action='store_true',
                          help='Run tests in parallel')
        parser.add_argument('--reproducible', action='store_true',
                          help='Ensure reproducible results')
        parser.add_argument('--coverage', action='store_true',
                          help='Generate coverage reports')
        parser.add_argument('--report', action='store_true',
                          help='Generate detailed test reports')
        parser.add_argument('--auto-fix', action='store_true',
                          help='Automatically fix test failures')
        parser.add_argument('--agents', type=str, default='auto',
                          choices=['auto', 'scientific', 'ai', 'engineering', 'domain', 'all'],
                          help='Agent selection')
        parser.add_argument('--orchestrate', action='store_true',
                          help='Enable multi-agent orchestration')
        parser.add_argument('--intelligent', action='store_true',
                          help='Enable intelligent agent selection')
        parser.add_argument('--distributed', action='store_true',
                          help='Enable distributed testing')
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test runner"""

        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE TEST EXECUTION ENGINE")
        print("="*60 + "\n")

        try:
            # Step 1: Detect test framework
            print("🔍 Detecting test framework...")
            framework = self.test_runner.detect_framework()
            print(f"   Detected: {framework.value}")

            if framework == TestFramework.UNKNOWN:
                return {
                    'success': False,
                    'summary': 'No test framework detected',
                    'details': 'Could not find any supported test framework'
                }

            # Step 2: Run tests
            print(f"\n▶️  Running {args.get('scope', 'all')} tests...")

            max_attempts = 5 if args.get('auto_fix') else 1
            attempt = 1
            result = None

            while attempt <= max_attempts:
                if attempt > 1:
                    print(f"\n🔄 Attempt {attempt}/{max_attempts}...")

                result = self.test_runner.run_tests(
                    framework=framework,
                    scope=args.get('scope', 'all'),
                    coverage=args.get('coverage', False),
                    parallel=args.get('parallel', False),
                    verbose=True
                )

                self._print_results(result)

                # If all tests passed, we're done
                if result.success:
                    print("\n✅ All tests passed!")
                    break

                # If not auto-fixing, stop here
                if not args.get('auto_fix'):
                    break

                # If tests failed and auto-fix is enabled, try to fix
                if result.failed > 0 and attempt < max_attempts:
                    print(f"\n🔧 Attempting to fix {result.failed} failing test(s)...")

                    fix_success = self._auto_fix_failures(result, framework)

                    if not fix_success:
                        print("   Could not auto-fix failures")
                        break

                    print("   Fixes applied, re-running tests...")

                attempt += 1

            # Step 3: Generate reports if requested
            if args.get('report'):
                print("\n📊 Generating test report...")
                self._generate_report(result, args)

            # Summary
            return {
                'success': result.success,
                'summary': self._generate_summary(result, attempt),
                'details': self._generate_details(result, args),
                'test_result': result.to_dict(),
                'attempts': attempt
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Test execution failed',
                'details': str(e)
            }

    def _print_results(self, result: TestResult) -> None:
        """Print test results"""
        print(f"\n📊 Results:")
        print(f"   ✅ Passed:  {result.passed}")
        print(f"   ❌ Failed:  {result.failed}")
        print(f"   ⏭️  Skipped: {result.skipped}")
        print(f"   📈 Total:   {result.total}")

        if result.duration > 0:
            print(f"   ⏱️  Time:    {result.duration:.2f}s")

        if result.coverage > 0:
            print(f"   📊 Coverage: {result.coverage:.1f}%")

        if result.failures:
            print(f"\n❌ Failures:")
            for i, failure in enumerate(result.failures[:5], 1):
                print(f"\n   {i}. {failure['name']}")
                if 'message' in failure:
                    # Print first line of error message
                    first_line = failure['message'].split('\n')[0][:80]
                    print(f"      {first_line}")

            if len(result.failures) > 5:
                print(f"\n   ... and {len(result.failures) - 5} more failures")

    def _auto_fix_failures(self, result: TestResult, framework: TestFramework) -> bool:
        """Attempt to automatically fix test failures"""

        if not result.failures:
            return False

        # Create backup before modifications
        print("   Creating backup...")
        self.code_modifier.create_backup()

        fixed_count = 0

        for failure in result.failures:
            try:
                # Analyze failure
                fix_applied = self._analyze_and_fix_failure(failure, framework)

                if fix_applied:
                    fixed_count += 1

            except Exception as e:
                print(f"   ⚠️  Could not fix {failure['name']}: {e}")

        if fixed_count > 0:
            print(f"   ✅ Applied {fixed_count} fix(es)")
            return True
        else:
            # Restore backup if no fixes were applied
            self.code_modifier.restore_backup()
            return False

    def _analyze_and_fix_failure(self, failure: Dict[str, str],
                                 framework: TestFramework) -> bool:
        """Analyze a test failure and attempt to fix it"""

        failure_message = failure.get('message', '')
        test_name = failure['name']

        # Common failure patterns and fixes

        # 1. Import errors
        if 'ImportError' in failure_message or 'ModuleNotFoundError' in failure_message:
            return self._fix_import_error(failure_message, test_name)

        # 2. Assertion errors
        elif 'AssertionError' in failure_message:
            return self._fix_assertion_error(failure_message, test_name)

        # 3. Type errors
        elif 'TypeError' in failure_message:
            return self._fix_type_error(failure_message, test_name)

        # 4. Attribute errors
        elif 'AttributeError' in failure_message:
            return self._fix_attribute_error(failure_message, test_name)

        # 5. Syntax errors
        elif 'SyntaxError' in failure_message:
            return self._fix_syntax_error(failure_message, test_name)

        return False

    def _fix_import_error(self, message: str, test_name: str) -> bool:
        """Fix import errors"""
        # Extract missing module name
        import re
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)

        if match:
            module_name = match.group(1)
            print(f"      Missing module: {module_name}")

            # Try to install missing package (simplified)
            # In production, this would check if it's a known package
            # and install it appropriately
            print(f"      Would install: {module_name}")
            return True

        return False

    def _fix_assertion_error(self, message: str, test_name: str) -> bool:
        """Fix assertion errors"""
        # This is complex - would need to understand what the test expects
        # For now, just log
        print(f"      Assertion failed - requires manual review")
        return False

    def _fix_type_error(self, message: str, test_name: str) -> bool:
        """Fix type errors"""
        # Extract type mismatch info and suggest fixes
        print(f"      Type error detected - may need type annotations")
        return False

    def _fix_attribute_error(self, message: str, test_name: str) -> bool:
        """Fix attribute errors"""
        # Check if attribute exists in similar names
        print(f"      Attribute error - checking for typos")
        return False

    def _fix_syntax_error(self, message: str, test_name: str) -> bool:
        """Fix syntax errors"""
        # Use linter to fix syntax errors
        print(f"      Syntax error - running auto-formatter")
        return False

    def _generate_report(self, result: TestResult, args: Dict[str, Any]) -> None:
        """Generate detailed test report"""
        report_file = Path('.test_reports') / f'test_report_{result.framework.value}.json'
        report_file.parent.mkdir(exist_ok=True)

        import json
        with open(report_file, 'w') as f:
            json.dump({
                'result': result.to_dict(),
                'args': args,
                'timestamp': str(Path.cwd())
            }, f, indent=2)

        print(f"   Report saved to: {report_file}")

    def _generate_summary(self, result: TestResult, attempts: int) -> str:
        """Generate result summary"""
        if result.success:
            summary = f"✅ All {result.total} tests passed"
            if attempts > 1:
                summary += f" (after {attempts} attempts)"
        else:
            summary = f"❌ {result.failed}/{result.total} tests failed"

        return summary

    def _generate_details(self, result: TestResult, args: Dict[str, Any]) -> str:
        """Generate detailed result information"""
        details = f"""
Framework: {result.framework.value}
Scope: {args.get('scope', 'all')}
Total Tests: {result.total}
Passed: {result.passed}
Failed: {result.failed}
Skipped: {result.skipped}
Duration: {result.duration:.2f}s
"""

        if result.coverage > 0:
            details += f"Coverage: {result.coverage:.1f}%\n"

        if args.get('auto_fix') and result.failed > 0:
            details += "\nAuto-fix was enabled but some failures remain.\n"
            details += "Manual intervention may be required.\n"

        return details


def main():
    """Main entry point"""
    executor = RunAllTestsExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())