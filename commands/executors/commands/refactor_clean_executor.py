#!/usr/bin/env python3
"""
Refactor Clean Command Executor
AI-powered code refactoring with multi-language support and modern patterns
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

# Add executors to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_executor import CommandExecutor, AgentOrchestrator
from ast_analyzer import ASTAnalyzer
from code_modifier import CodeModifier


class RefactorCleanExecutor(CommandExecutor):
    """Executor for /refactor-clean command"""

    def __init__(self):
        super().__init__("refactor-clean")
        self.ast_analyzer = ASTAnalyzer()
        self.code_modifier = CodeModifier()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        """Configure argument parser"""
        parser = argparse.ArgumentParser(
            description='AI-powered code refactoring engine'
        )
        parser.add_argument('target', nargs='?', default='.',
                          help='Target file or directory to refactor')
        parser.add_argument('--language', type=str, default='auto',
                          choices=['python', 'javascript', 'typescript',
                                 'java', 'julia', 'auto'],
                          help='Programming language')
        parser.add_argument('--scope', type=str, default='file',
                          choices=['file', 'project'],
                          help='Refactoring scope')
        parser.add_argument('--patterns', type=str, default='modern',
                          choices=['modern', 'performance', 'security'],
                          help='Refactoring patterns to apply')
        parser.add_argument('--report', type=str, default='summary',
                          choices=['summary', 'detailed'],
                          help='Report detail level')
        parser.add_argument('--implement', action='store_true',
                          help='Implement refactoring suggestions')
        parser.add_argument('--agents', type=str, default='quality',
                          choices=['quality', 'orchestrator', 'all'],
                          help='Agent selection')
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute refactoring analysis and implementation"""

        print("\n" + "="*60)
        print("🔧 CODE REFACTORING ENGINE")
        print("="*60 + "\n")

        try:
            # Step 1: Validate target
            target = Path(args.get('target', '.'))
            if not target.exists():
                target = self.work_dir / target

            if not target.exists():
                return {
                    'success': False,
                    'summary': f'Target not found: {target}',
                    'details': 'Specified target does not exist'
                }

            print(f"🎯 Target: {target.relative_to(self.work_dir) if target.is_relative_to(self.work_dir) else target}")

            # Step 2: Collect files to analyze
            print("\n🔍 Collecting files...")
            files = self._collect_files(target, args)
            print(f"   Found {len(files)} file(s) to analyze")

            if not files:
                return {
                    'success': False,
                    'summary': 'No files found to refactor',
                    'details': 'Target contains no analyzable code files'
                }

            # Step 3: Analyze code for refactoring opportunities
            print("\n📊 Analyzing code quality...")
            issues = self._analyze_files(files, args)

            print(f"   Found {len(issues)} refactoring opportunity(ies)")

            # Step 4: Generate refactoring suggestions
            print("\n💡 Generating refactoring suggestions...")
            suggestions = self._generate_suggestions(issues, args)

            # Step 5: Implement if requested
            applied = []
            if args.get('implement') and suggestions:
                print("\n🔨 Implementing refactorings...")

                # Create backup
                print("   Creating backup...")
                self.code_modifier.create_backup()

                applied = self._apply_refactorings(suggestions, files)
                print(f"   ✅ Applied {len(applied)} refactoring(s)")

            # Step 6: Generate report
            report = self._generate_report(issues, suggestions, applied, args)

            return {
                'success': True,
                'summary': f'Analyzed {len(files)} files, found {len(issues)} issues',
                'details': report,
                'files_analyzed': len(files),
                'issues_found': len(issues),
                'refactorings_applied': len(applied)
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Refactoring failed',
                'details': str(e)
            }

    def _collect_files(self, target: Path, args: Dict[str, Any]) -> List[Path]:
        """Collect files to analyze"""
        language = args.get('language', 'auto')
        scope = args.get('scope', 'file')

        if target.is_file():
            return [target]

        # Collect files based on language
        extensions = {
            'python': ['*.py'],
            'javascript': ['*.js'],
            'typescript': ['*.ts', '*.tsx'],
            'java': ['*.java'],
            'julia': ['*.jl'],
            'auto': ['*.py', '*.js', '*.ts', '*.tsx', '*.java', '*.jl']
        }

        patterns = extensions.get(language, ['*.py'])
        files = []

        for pattern in patterns:
            if scope == 'file':
                files.extend(target.glob(pattern))
            else:
                files.extend(target.rglob(pattern))

        # Filter out common directories to ignore
        ignore_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv'}
        files = [f for f in files if not any(d in f.parts for d in ignore_dirs)]

        return files[:100]  # Limit to prevent slowdown

    def _analyze_files(self, files: List[Path],
                      args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze files for refactoring opportunities"""
        issues = []

        for file in files:
            file_issues = self._analyze_single_file(file, args)
            issues.extend(file_issues)

        return issues

    def _analyze_single_file(self, file: Path,
                             args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze a single file"""
        issues = []
        pattern_type = args.get('patterns', 'modern')

        try:
            content = file.read_text()

            # Python-specific analysis
            if file.suffix == '.py':
                # Check for old-style formatting
                if '%s' in content or '% (' in content:
                    issues.append({
                        'file': str(file),
                        'type': 'outdated_syntax',
                        'pattern': pattern_type,
                        'line': 0,
                        'message': 'Old-style string formatting detected',
                        'suggestion': 'Use f-strings instead'
                    })

                # Check for missing type hints
                func_defs = re.findall(r'def \w+\([^)]*\):', content)
                if func_defs and '->' not in content:
                    issues.append({
                        'file': str(file),
                        'type': 'missing_types',
                        'pattern': pattern_type,
                        'line': 0,
                        'message': 'Missing type hints',
                        'suggestion': 'Add type annotations for better code quality'
                    })

                # Check for long functions
                lines = content.split('\n')
                in_function = False
                func_start = 0
                func_name = ''

                for i, line in enumerate(lines):
                    if line.strip().startswith('def '):
                        in_function = True
                        func_start = i
                        func_name = line.split('def ')[1].split('(')[0]
                    elif in_function and line and not line[0].isspace():
                        func_length = i - func_start
                        if func_length > 50:
                            issues.append({
                                'file': str(file),
                                'type': 'long_function',
                                'pattern': pattern_type,
                                'line': func_start,
                                'message': f'Function {func_name} is {func_length} lines long',
                                'suggestion': 'Consider breaking into smaller functions'
                            })
                        in_function = False

                # Check for complex conditionals
                complex_ifs = re.findall(r'if .* and .* and .* and', content)
                if complex_ifs:
                    issues.append({
                        'file': str(file),
                        'type': 'complex_conditional',
                        'pattern': pattern_type,
                        'line': 0,
                        'message': f'{len(complex_ifs)} complex conditional(s) found',
                        'suggestion': 'Extract to named boolean variables'
                    })

                # Security patterns
                if pattern_type == 'security':
                    # Check for eval/exec usage
                    if 'eval(' in content or 'exec(' in content:
                        issues.append({
                            'file': str(file),
                            'type': 'security_risk',
                            'pattern': pattern_type,
                            'line': 0,
                            'message': 'Unsafe eval/exec usage detected',
                            'suggestion': 'Avoid eval/exec for security reasons'
                        })

                    # Check for hardcoded secrets
                    if re.search(r'password\s*=\s*["\']', content, re.I):
                        issues.append({
                            'file': str(file),
                            'type': 'security_risk',
                            'pattern': pattern_type,
                            'line': 0,
                            'message': 'Possible hardcoded password',
                            'suggestion': 'Use environment variables or secrets management'
                        })

        except Exception as e:
            pass

        return issues

    def _generate_suggestions(self, issues: List[Dict[str, Any]],
                             args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate refactoring suggestions"""
        suggestions = []

        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            issue_type = issue['type']
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

        # Generate suggestions
        for issue_type, type_issues in issues_by_type.items():
            suggestions.append({
                'type': issue_type,
                'count': len(type_issues),
                'files': list(set(i['file'] for i in type_issues)),
                'recommendation': type_issues[0]['suggestion'],
                'priority': self._calculate_priority(issue_type),
                'effort': self._estimate_effort(issue_type, len(type_issues))
            })

        # Sort by priority
        suggestions.sort(key=lambda s: s['priority'], reverse=True)

        return suggestions

    def _calculate_priority(self, issue_type: str) -> int:
        """Calculate refactoring priority"""
        priorities = {
            'security_risk': 10,
            'performance_issue': 8,
            'missing_types': 6,
            'outdated_syntax': 5,
            'long_function': 4,
            'complex_conditional': 4,
        }
        return priorities.get(issue_type, 3)

    def _estimate_effort(self, issue_type: str, count: int) -> str:
        """Estimate refactoring effort"""
        base_effort = {
            'security_risk': 'high',
            'performance_issue': 'medium',
            'missing_types': 'low',
            'outdated_syntax': 'low',
            'long_function': 'medium',
            'complex_conditional': 'low',
        }

        effort = base_effort.get(issue_type, 'medium')

        if count > 20:
            return 'high'
        elif count > 10 and effort == 'low':
            return 'medium'

        return effort

    def _apply_refactorings(self, suggestions: List[Dict[str, Any]],
                           files: List[Path]) -> List[str]:
        """Apply refactoring suggestions"""
        applied = []

        for suggestion in suggestions[:10]:  # Apply top 10 suggestions
            if suggestion['type'] == 'outdated_syntax':
                # Example: Convert old-style string formatting
                for file_path in suggestion['files']:
                    try:
                        file = Path(file_path)
                        content = file.read_text()

                        # Simple replacement (production would be more sophisticated)
                        modified = content.replace('"%" % ', 'f"')

                        if modified != content:
                            file.write_text(modified)
                            applied.append(f"Updated formatting in {file.name}")
                    except Exception:
                        pass

        return applied

    def _generate_report(self, issues: List[Dict[str, Any]],
                        suggestions: List[Dict[str, Any]],
                        applied: List[str],
                        args: Dict[str, Any]) -> str:
        """Generate refactoring report"""
        report = "\n" + "="*60 + "\n"
        report += "REFACTORING ANALYSIS REPORT\n"
        report += "="*60 + "\n\n"

        # Summary
        report += f"Total Issues Found: {len(issues)}\n"
        report += f"Suggestions Generated: {len(suggestions)}\n"

        if applied:
            report += f"Refactorings Applied: {len(applied)}\n"

        report += "\n" + "-"*60 + "\n\n"

        # Top suggestions
        if suggestions:
            report += "TOP REFACTORING SUGGESTIONS:\n\n"

            for i, suggestion in enumerate(suggestions[:10], 1):
                report += f"{i}. {suggestion['type'].replace('_', ' ').title()}\n"
                report += f"   Priority: {suggestion['priority']}/10\n"
                report += f"   Effort: {suggestion['effort']}\n"
                report += f"   Affected Files: {len(suggestion['files'])}\n"
                report += f"   Recommendation: {suggestion['recommendation']}\n\n"

        # Applied refactorings
        if applied:
            report += "\n" + "-"*60 + "\n\n"
            report += "APPLIED REFACTORINGS:\n\n"
            for item in applied:
                report += f"  ✅ {item}\n"

        if args.get('report') == 'detailed':
            report += "\n" + "-"*60 + "\n\n"
            report += "DETAILED ISSUES:\n\n"

            for issue in issues[:20]:
                report += f"File: {issue['file']}\n"
                report += f"Type: {issue['type']}\n"
                report += f"Message: {issue['message']}\n"
                report += f"Suggestion: {issue['suggestion']}\n\n"

        return report


def main():
    """Main entry point"""
    executor = RefactorCleanExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())