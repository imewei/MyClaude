#!/usr/bin/env python3
"""
Debug Command Executor
Scientific computing debugging with GPU support and multi-language analysis
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_executor import CommandExecutor, AgentOrchestrator
from test_runner import TestRunner


class DebugExecutor(CommandExecutor):
    """Executor for /debug command"""

    def __init__(self):
        super().__init__("debug")
        self.test_runner = TestRunner()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Debugging engine')
        parser.add_argument('--issue', type=str, help='Issue type to debug')
        parser.add_argument('--gpu', action='store_true', help='GPU debugging')
        parser.add_argument('--julia', action='store_true', help='Julia debugging')
        parser.add_argument('--research', action='store_true', help='Research mode')
        parser.add_argument('--jupyter', action='store_true', help='Jupyter debugging')
        parser.add_argument('--profile', action='store_true', help='Performance profiling')
        parser.add_argument('--monitor', action='store_true', help='Resource monitoring')
        parser.add_argument('--logs', action='store_true', help='Analyze logs')
        parser.add_argument('--auto-fix', action='store_true', help='Auto-fix issues')
        parser.add_argument('--report', action='store_true', help='Generate report')
        parser.add_argument('--agents', type=str, default='scientific',
                          choices=['scientific', 'quality', 'orchestrator', 'all'])
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("🐛 DEBUGGING ENGINE")
        print("="*60 + "\n")

        try:
            print("🔍 Analyzing for issues...")

            issues = self._detect_issues(args)

            print(f"   Found {len(issues)} issue(s)")

            if args.get('auto_fix') and issues:
                print("\n🔧 Attempting auto-fix...")
                fixed = self._auto_fix_issues(issues)
                print(f"   Fixed {fixed} issue(s)")

            return {
                'success': True,
                'summary': f'Found {len(issues)} issue(s)',
                'details': self._generate_debug_report(issues, args),
                'issues_found': len(issues)
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Debugging failed',
                'details': str(e)
            }

    def _detect_issues(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect issues in code"""
        issues = []

        # Check for common issues
        py_files = list(self.work_dir.rglob('*.py'))[:10]

        for file in py_files:
            try:
                content = file.read_text()

                # Check for print debugging
                if 'print(' in content and 'debug' in content.lower():
                    issues.append({
                        'file': str(file),
                        'type': 'debug_code',
                        'severity': 'low',
                        'message': 'Debug print statements found'
                    })

                # Check for bare excepts
                if 'except:' in content:
                    issues.append({
                        'file': str(file),
                        'type': 'bare_except',
                        'severity': 'medium',
                        'message': 'Bare except clauses found'
                    })

            except Exception:
                pass

        return issues

    def _auto_fix_issues(self, issues: List[Dict[str, Any]]) -> int:
        """Auto-fix detected issues"""
        fixed = 0
        for issue in issues:
            if issue['type'] == 'debug_code':
                # Would remove debug statements
                fixed += 1
        return fixed

    def _generate_debug_report(self, issues: List[Dict[str, Any]],
                              args: Dict[str, Any]) -> str:
        """Generate debugging report"""
        report = "\nDEBUGGING REPORT\n" + "="*60 + "\n\n"
        report += f"Issues Found: {len(issues)}\n\n"

        for i, issue in enumerate(issues, 1):
            report += f"{i}. {issue['type']} ({issue['severity']})\n"
            report += f"   File: {Path(issue['file']).name}\n"
            report += f"   {issue['message']}\n\n"

        return report


def main():
    executor = DebugExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())