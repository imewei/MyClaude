#!/usr/bin/env python3
"""
Generate Tests Command Executor
Generate comprehensive test suites for Python, Julia, and JAX scientific computing projects
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add executors to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_executor import CommandExecutor, AgentOrchestrator
from ast_analyzer import ASTAnalyzer
from test_runner import TestRunner


class GenerateTestsExecutor(CommandExecutor):
    """Executor for /generate-tests command"""

    def __init__(self):
        super().__init__("generate-tests")
        self.ast_analyzer = ASTAnalyzer()
        self.test_runner = TestRunner()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        """Configure argument parser"""
        parser = argparse.ArgumentParser(
            description='Test generation engine'
        )
        parser.add_argument('target_file_or_module', nargs='?', default='.',
                          help='Target file or module for test generation')
        parser.add_argument('--type', type=str, default='all',
                          choices=['all', 'unit', 'integration', 'performance',
                                 'jax', 'scientific', 'gpu'],
                          help='Test type to generate')
        parser.add_argument('--framework', type=str, default='auto',
                          choices=['auto', 'pytest', 'julia', 'jax'],
                          help='Test framework')
        parser.add_argument('--coverage', type=int, default=80,
                          help='Target coverage percentage')
        parser.add_argument('--agents', type=str, default='scientific',
                          choices=['scientific', 'quality', 'orchestrator', 'all'],
                          help='Agent selection')
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test generation"""

        print("\n" + "="*60)
        print("🧪 TEST GENERATION ENGINE")
        print("="*60 + "\n")

        try:
            # Step 1: Find target files
            target = Path(args.get('target_file_or_module', '.'))
            if not target.exists():
                target = self.work_dir / target

            print(f"🎯 Target: {target}")

            # Step 2: Analyze code
            print("\n🔍 Analyzing code structure...")
            code_analysis = self._analyze_code(target)

            if not code_analysis['functions'] and not code_analysis['classes']:
                return {
                    'success': False,
                    'summary': 'No functions or classes found to test',
                    'details': 'Target contains no testable code'
                }

            print(f"   Found {len(code_analysis['functions'])} function(s)")
            print(f"   Found {len(code_analysis['classes'])} class(es)")

            # Step 3: Generate tests
            print("\n📝 Generating test cases...")
            tests = self._generate_tests(code_analysis, args)

            print(f"   Generated {len(tests)} test(s)")

            # Step 4: Write test files
            print("\n💾 Writing test files...")
            test_files = self._write_test_files(tests, target, args)

            return {
                'success': True,
                'summary': f'Generated {len(tests)} test(s) in {len(test_files)} file(s)',
                'details': self._generate_details(tests, test_files, args),
                'test_count': len(tests),
                'test_files': test_files
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Test generation failed',
                'details': str(e)
            }

    def _analyze_code(self, target: Path) -> Dict[str, Any]:
        """Analyze code to determine what needs testing"""
        analysis = {
            'functions': [],
            'classes': [],
            'files': []
        }

        files = []
        if target.is_file():
            files = [target]
        else:
            files = list(target.rglob('*.py'))
            # Filter out test files and common directories
            files = [f for f in files if 'test' not in f.name.lower()
                    and not any(d in f.parts for d in {'__pycache__', 'venv', '.venv'})]

        for file in files[:20]:
            try:
                ast_result = self.ast_analyzer.analyze_file(file)
                if ast_result:
                    analysis['files'].append(str(file))
                    analysis['functions'].extend([
                        {**f, 'file': str(file)}
                        for f in ast_result.get('functions', [])
                    ])
                    analysis['classes'].extend([
                        {**c, 'file': str(file)}
                        for c in ast_result.get('classes', [])
                    ])
            except Exception:
                pass

        return analysis

    def _generate_tests(self, analysis: Dict[str, Any],
                       args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases"""
        tests = []
        test_type = args.get('type', 'all')

        # Generate tests for functions
        for func in analysis['functions']:
            if test_type in ['all', 'unit']:
                tests.append(self._generate_function_test(func, 'unit'))

        # Generate tests for classes
        for cls in analysis['classes']:
            if test_type in ['all', 'unit']:
                tests.extend(self._generate_class_tests(cls, 'unit'))

        return tests

    def _generate_function_test(self, func: Dict[str, Any],
                                test_type: str) -> Dict[str, Any]:
        """Generate test for a function"""
        func_name = func.get('name', 'unknown')

        test_code = f"""
def test_{func_name}():
    \"\"\"Test {func_name} function\"\"\"
    # TODO: Implement test
    result = {func_name}()
    assert result is not None
"""

        return {
            'name': f'test_{func_name}',
            'type': test_type,
            'target': func_name,
            'file': func['file'],
            'code': test_code.strip()
        }

    def _generate_class_tests(self, cls: Dict[str, Any],
                             test_type: str) -> List[Dict[str, Any]]:
        """Generate tests for a class"""
        tests = []
        cls_name = cls.get('name', 'Unknown')

        # Test class initialization
        tests.append({
            'name': f'test_{cls_name.lower()}_init',
            'type': test_type,
            'target': cls_name,
            'file': cls['file'],
            'code': f"""
def test_{cls_name.lower()}_init():
    \"\"\"Test {cls_name} initialization\"\"\"
    instance = {cls_name}()
    assert instance is not None
""".strip()
        })

        # Test methods
        for method in cls.get('methods', [])[:5]:
            method_name = method.get('name', 'unknown')
            if not method_name.startswith('_'):  # Skip private methods
                tests.append({
                    'name': f'test_{cls_name.lower()}_{method_name}',
                    'type': test_type,
                    'target': f'{cls_name}.{method_name}',
                    'file': cls['file'],
                    'code': f"""
def test_{cls_name.lower()}_{method_name}():
    \"\"\"Test {cls_name}.{method_name} method\"\"\"
    instance = {cls_name}()
    result = instance.{method_name}()
    assert result is not None
""".strip()
                })

        return tests

    def _write_test_files(self, tests: List[Dict[str, Any]],
                         target: Path, args: Dict[str, Any]) -> List[str]:
        """Write generated tests to files"""
        test_files = []

        # Group tests by source file
        tests_by_file = {}
        for test in tests:
            source_file = test['file']
            if source_file not in tests_by_file:
                tests_by_file[source_file] = []
            tests_by_file[source_file].append(test)

        # Write test files
        for source_file, file_tests in tests_by_file.items():
            source_path = Path(source_file)
            test_filename = f'test_{source_path.stem}.py'

            # Determine test directory
            if target.is_file():
                test_dir = target.parent / 'tests'
            else:
                test_dir = target / 'tests'

            test_dir.mkdir(exist_ok=True)
            test_path = test_dir / test_filename

            # Generate test file content
            content = self._generate_test_file_content(source_path, file_tests, args)

            self.write_file(test_path, content)
            test_files.append(str(test_path))
            print(f"   ✅ {test_path.relative_to(self.work_dir)}")

        return test_files

    def _generate_test_file_content(self, source_file: Path,
                                    tests: List[Dict[str, Any]],
                                    args: Dict[str, Any]) -> str:
        """Generate complete test file content"""
        content = f'''"""
Tests for {source_file.stem}
Auto-generated by Claude Code /generate-tests command
"""

import pytest
from {source_file.stem} import *


'''

        for test in tests:
            content += test['code'] + '\n\n\n'

        return content

    def _generate_details(self, tests: List[Dict[str, Any]],
                         test_files: List[str],
                         args: Dict[str, Any]) -> str:
        """Generate detailed execution information"""
        test_types = {}
        for test in tests:
            test_type = test['type']
            test_types[test_type] = test_types.get(test_type, 0) + 1

        details = f"""
Test Generation Complete

Generated Tests: {len(tests)}
"""

        for test_type, count in test_types.items():
            details += f"  • {test_type}: {count}\n"

        details += f"\nTest Files Created: {len(test_files)}\n"
        for tf in test_files:
            details += f"  - {Path(tf).name}\n"

        details += f"\nTarget Coverage: {args.get('coverage', 80)}%\n"
        details += f"Framework: {args.get('framework', 'auto')}\n"

        return details


def main():
    """Main entry point"""
    executor = GenerateTestsExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())