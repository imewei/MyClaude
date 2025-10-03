#!/usr/bin/env python3
"""
Check Code Quality Command Executor
Code quality analysis for Python, Julia, and JAX ecosystems
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_executor import CommandExecutor, AgentOrchestrator
from ast_analyzer import ASTAnalyzer


class CheckCodeQualityExecutor(CommandExecutor):
    """Executor for /check-code-quality command"""

    def __init__(self):
        super().__init__("check-code-quality")
        self.ast_analyzer = ASTAnalyzer()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Code quality analysis engine')
        parser.add_argument('target_path', nargs='?', default='.',
                          help='Target path for quality check')
        parser.add_argument('--language', type=str, default='auto',
                          choices=['python', 'julia', 'jax', 'auto'])
        parser.add_argument('--analysis', type=str, default='basic',
                          choices=['basic', 'scientific', 'gpu'])
        parser.add_argument('--auto-fix', action='store_true',
                          help='Auto-fix quality issues')
        parser.add_argument('--format', type=str, default='text',
                          choices=['text', 'json'])
        parser.add_argument('--agents', type=str, default='quality',
                          choices=['quality', 'scientific', 'orchestrator', 'all'])
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("📊 CODE QUALITY ANALYSIS ENGINE")
        print("="*60 + "\n")

        try:
            target = Path(args.get('target_path', '.'))
            if not target.exists():
                target = self.work_dir / target

            print(f"🎯 Target: {target.name}")

            # Collect files
            print("\n📂 Collecting files...")
            files = self._collect_files(target, args)
            print(f"   Found {len(files)} file(s)")

            # Analyze quality
            print("\n🔍 Analyzing code quality...")
            quality_report = self._analyze_quality(files, args)

            # Calculate score
            score = self._calculate_quality_score(quality_report)
            print(f"\n📈 Quality Score: {score}/100")

            return {
                'success': True,
                'summary': f'Quality score: {score}/100',
                'details': self._generate_quality_report(quality_report, score, args),
                'score': score,
                'files_analyzed': len(files)
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Quality check failed',
                'details': str(e)
            }

    def _collect_files(self, target: Path, args: Dict[str, Any]) -> List[Path]:
        """Collect files for analysis"""
        if target.is_file():
            return [target]

        language = args.get('language', 'auto')
        pattern = '*.py' if language in ['python', 'jax', 'auto'] else '*.jl'

        files = list(target.rglob(pattern))
        ignore_dirs = {'__pycache__', 'venv', '.venv', 'node_modules'}
        files = [f for f in files if not any(d in f.parts for d in ignore_dirs)]

        return files[:50]

    def _analyze_quality(self, files: List[Path],
                        args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code quality"""
        report = {
            'issues': [],
            'metrics': {
                'total_lines': 0,
                'code_lines': 0,
                'comment_lines': 0,
                'blank_lines': 0,
                'functions': 0,
                'classes': 0,
            },
            'violations': {
                'style': 0,
                'complexity': 0,
                'documentation': 0,
                'security': 0,
            }
        }

        for file in files:
            try:
                content = file.read_text()
                lines = content.split('\n')

                # Count lines
                report['metrics']['total_lines'] += len(lines)
                report['metrics']['blank_lines'] += sum(1 for l in lines if not l.strip())
                report['metrics']['comment_lines'] += sum(1 for l in lines if l.strip().startswith('#'))
                report['metrics']['code_lines'] += (len(lines) -
                                                    report['metrics']['blank_lines'] -
                                                    report['metrics']['comment_lines'])

                # Analyze with AST
                ast_result = self.ast_analyzer.analyze_file(file)
                if ast_result:
                    report['metrics']['functions'] += len(ast_result.get('functions', []))
                    report['metrics']['classes'] += len(ast_result.get('classes', []))

                # Check for quality issues
                # Style violations
                if len(lines) > 1000:
                    report['issues'].append({
                        'file': str(file),
                        'type': 'style',
                        'severity': 'low',
                        'message': 'File too long (>1000 lines)'
                    })
                    report['violations']['style'] += 1

                # Documentation violations
                if 'def ' in content and '"""' not in content:
                    report['issues'].append({
                        'file': str(file),
                        'type': 'documentation',
                        'severity': 'medium',
                        'message': 'Missing docstrings'
                    })
                    report['violations']['documentation'] += 1

                # Complexity violations
                for i, line in enumerate(lines):
                    if line.count('if ') + line.count('and ') + line.count('or ') > 3:
                        report['issues'].append({
                            'file': str(file),
                            'type': 'complexity',
                            'severity': 'medium',
                            'line': i + 1,
                            'message': 'Complex conditional detected'
                        })
                        report['violations']['complexity'] += 1
                        break  # One per file

            except Exception:
                pass

        return report

    def _calculate_quality_score(self, report: Dict[str, Any]) -> int:
        """Calculate overall quality score"""
        base_score = 100

        # Deduct points for violations
        deductions = {
            'style': 2,
            'complexity': 5,
            'documentation': 3,
            'security': 10,
        }

        for violation_type, count in report['violations'].items():
            base_score -= count * deductions.get(violation_type, 1)

        return max(0, min(100, base_score))

    def _generate_quality_report(self, report: Dict[str, Any],
                                 score: int, args: Dict[str, Any]) -> str:
        """Generate quality report"""
        output = "\nCODE QUALITY REPORT\n" + "="*60 + "\n\n"

        output += f"Quality Score: {score}/100\n\n"

        # Metrics
        output += "CODE METRICS:\n"
        for metric, value in report['metrics'].items():
            output += f"  • {metric.replace('_', ' ').title()}: {value}\n"

        output += "\n"

        # Violations
        output += "VIOLATIONS:\n"
        total_violations = sum(report['violations'].values())
        output += f"  Total: {total_violations}\n"

        for vtype, count in report['violations'].items():
            if count > 0:
                output += f"  • {vtype.title()}: {count}\n"

        output += "\n"

        # Top issues
        if report['issues']:
            output += "TOP ISSUES:\n"
            for i, issue in enumerate(report['issues'][:10], 1):
                output += f"\n{i}. {issue['type'].upper()} ({issue['severity']})\n"
                output += f"   File: {Path(issue['file']).name}\n"
                output += f"   {issue['message']}\n"

        return output


def main():
    executor = CheckCodeQualityExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())