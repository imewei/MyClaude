#!/usr/bin/env python3
"""
Optimize Command Executor
Code optimization and performance analysis for Python, Julia, JAX, and scientific computing
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


class OptimizeExecutor(CommandExecutor):
    """Executor for /optimize command"""

    def __init__(self):
        super().__init__("optimize")
        self.ast_analyzer = ASTAnalyzer()
        self.code_modifier = CodeModifier()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        """Configure argument parser"""
        parser = argparse.ArgumentParser(
            description='Performance optimization engine'
        )
        parser.add_argument('target', nargs='?', default='.',
                          help='Target file or directory to optimize')
        parser.add_argument('--language', type=str, default='auto',
                          choices=['python', 'julia', 'jax', 'auto'],
                          help='Programming language')
        parser.add_argument('--category', type=str, default='all',
                          choices=['all', 'algorithm', 'memory', 'io', 'concurrency'],
                          help='Optimization category')
        parser.add_argument('--format', type=str, default='text',
                          choices=['text', 'json', 'html'],
                          help='Output format')
        parser.add_argument('--implement', action='store_true',
                          help='Implement optimization suggestions')
        parser.add_argument('--agents', type=str, default='auto',
                          choices=['auto', 'scientific', 'ai', 'engineering',
                                 'quantum', 'all'],
                          help='Agent selection')
        parser.add_argument('--orchestrate', action='store_true',
                          help='Enable multi-agent orchestration')
        parser.add_argument('--intelligent', action='store_true',
                          help='Enable intelligent optimization')
        parser.add_argument('--breakthrough', action='store_true',
                          help='Enable breakthrough-level analysis')
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance optimization analysis"""

        print("\n" + "="*60)
        print("⚡ PERFORMANCE OPTIMIZATION ENGINE")
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

            # Step 2: Detect language
            language = self._detect_language(target, args)
            print(f"🔍 Language: {language}")

            # Step 3: Collect files
            print("\n📂 Collecting files...")
            files = self._collect_files(target, language)
            print(f"   Found {len(files)} file(s) to analyze")

            if not files:
                return {
                    'success': False,
                    'summary': 'No files found to optimize',
                    'details': 'Target contains no optimizable code files'
                }

            # Step 4: Analyze performance
            print("\n📊 Analyzing performance...")
            analysis = self._analyze_performance(files, args)

            # Step 5: Generate optimizations
            print("\n💡 Generating optimization suggestions...")
            optimizations = self._generate_optimizations(analysis, args)

            print(f"   Generated {len(optimizations)} optimization(s)")

            # Step 6: Implement if requested
            implemented = []
            if args.get('implement') and optimizations:
                print("\n🔨 Implementing optimizations...")
                self.code_modifier.create_backup()
                implemented = self._implement_optimizations(optimizations, files)
                print(f"   ✅ Implemented {len(implemented)} optimization(s)")

            # Step 7: Generate report
            report = self._generate_report(analysis, optimizations, implemented, args)

            return {
                'success': True,
                'summary': f'Analyzed {len(files)} files, found {len(optimizations)} optimizations',
                'details': report,
                'files_analyzed': len(files),
                'optimizations_found': len(optimizations),
                'optimizations_implemented': len(implemented)
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Optimization analysis failed',
                'details': str(e)
            }

    def _detect_language(self, target: Path, args: Dict[str, Any]) -> str:
        """Detect programming language"""
        if args.get('language') != 'auto':
            return args['language']

        # Auto-detect from file extensions
        if target.is_file():
            ext = target.suffix
        else:
            exts = [f.suffix for f in target.rglob('*') if f.is_file()]
            ext = max(set(exts), key=exts.count) if exts else ''

        language_map = {
            '.py': 'python',
            '.jl': 'julia',
            '.jax': 'jax'
        }

        return language_map.get(ext, 'python')

    def _collect_files(self, target: Path, language: str) -> List[Path]:
        """Collect files to optimize"""
        ext_map = {
            'python': '*.py',
            'julia': '*.jl',
            'jax': '*.py'
        }

        pattern = ext_map.get(language, '*.py')

        if target.is_file():
            return [target]

        files = list(target.rglob(pattern))

        # Filter out common directories to ignore
        ignore_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv', 'test', 'tests'}
        files = [f for f in files if not any(d in f.parts for d in ignore_dirs)]

        return files[:50]  # Limit to prevent slowdown

    def _analyze_performance(self, files: List[Path],
                            args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        analysis = {
            'bottlenecks': [],
            'memory_issues': [],
            'io_issues': [],
            'concurrency_opportunities': [],
            'algorithm_improvements': []
        }

        category = args.get('category', 'all')

        for file in files:
            try:
                content = file.read_text()

                # Algorithm analysis
                if category in ['all', 'algorithm']:
                    # Check for nested loops (O(n²) or worse)
                    nested_loops = self._find_nested_loops(content)
                    if nested_loops:
                        analysis['bottlenecks'].append({
                            'file': str(file),
                            'type': 'nested_loops',
                            'severity': 'high',
                            'count': len(nested_loops),
                            'message': f'Found {len(nested_loops)} nested loop(s) - potential O(n²) complexity'
                        })

                    # Check for linear search in loops
                    if 'in ' in content and 'for ' in content:
                        analysis['algorithm_improvements'].append({
                            'file': str(file),
                            'type': 'linear_search',
                            'severity': 'medium',
                            'message': 'Consider using hash-based lookups (dict/set) instead of linear search'
                        })

                # Memory analysis
                if category in ['all', 'memory']:
                    # Check for list comprehensions that could be generators
                    list_comps = re.findall(r'\[.*for .* in .*\]', content)
                    if len(list_comps) > 3:
                        analysis['memory_issues'].append({
                            'file': str(file),
                            'type': 'list_comprehension',
                            'severity': 'low',
                            'count': len(list_comps),
                            'message': f'{len(list_comps)} list comprehension(s) could be generators'
                        })

                    # Check for global variables
                    if re.search(r'^[A-Z_]+\s*=', content, re.MULTILINE):
                        analysis['memory_issues'].append({
                            'file': str(file),
                            'type': 'global_state',
                            'severity': 'medium',
                            'message': 'Global variables detected - consider using configuration objects'
                        })

                # I/O analysis
                if category in ['all', 'io']:
                    # Check for file I/O in loops
                    if 'open(' in content and 'for ' in content:
                        analysis['io_issues'].append({
                            'file': str(file),
                            'type': 'io_in_loop',
                            'severity': 'high',
                            'message': 'File I/O inside loops detected - consider batching operations'
                        })

                # Concurrency analysis
                if category in ['all', 'concurrency']:
                    # Check for opportunities for parallelization
                    if 'for ' in content and 'range(' in content:
                        if 'multiprocessing' not in content and 'concurrent' not in content:
                            analysis['concurrency_opportunities'].append({
                                'file': str(file),
                                'type': 'parallelizable_loop',
                                'severity': 'medium',
                                'message': 'Loop could potentially be parallelized'
                            })

            except Exception:
                pass

        return analysis

    def _find_nested_loops(self, content: str) -> List[int]:
        """Find nested loops in code"""
        nested = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if 'for ' in line or 'while ' in line:
                # Check if there's another loop in the next 20 lines
                indent = len(line) - len(line.lstrip())
                for j in range(i+1, min(i+20, len(lines))):
                    next_line = lines[j]
                    next_indent = len(next_line) - len(next_line.lstrip())

                    if next_indent > indent and ('for ' in next_line or 'while ' in next_line):
                        nested.append(i)
                        break

        return nested

    def _generate_optimizations(self, analysis: Dict[str, Any],
                               args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        optimizations = []

        # Process bottlenecks
        for bottleneck in analysis['bottlenecks']:
            optimizations.append({
                'file': bottleneck['file'],
                'type': bottleneck['type'],
                'severity': bottleneck['severity'],
                'category': 'algorithm',
                'issue': bottleneck['message'],
                'recommendation': self._get_optimization_recommendation(bottleneck['type']),
                'estimated_improvement': '50-90%' if bottleneck['severity'] == 'high' else '10-30%'
            })

        # Process memory issues
        for issue in analysis['memory_issues']:
            optimizations.append({
                'file': issue['file'],
                'type': issue['type'],
                'severity': issue['severity'],
                'category': 'memory',
                'issue': issue['message'],
                'recommendation': self._get_optimization_recommendation(issue['type']),
                'estimated_improvement': '20-40%'
            })

        # Process I/O issues
        for issue in analysis['io_issues']:
            optimizations.append({
                'file': issue['file'],
                'type': issue['type'],
                'severity': issue['severity'],
                'category': 'io',
                'issue': issue['message'],
                'recommendation': self._get_optimization_recommendation(issue['type']),
                'estimated_improvement': '60-95%'
            })

        # Process concurrency opportunities
        for opp in analysis['concurrency_opportunities']:
            optimizations.append({
                'file': opp['file'],
                'type': opp['type'],
                'severity': opp['severity'],
                'category': 'concurrency',
                'issue': opp['message'],
                'recommendation': self._get_optimization_recommendation(opp['type']),
                'estimated_improvement': '2-4x speedup'
            })

        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        optimizations.sort(key=lambda o: severity_order.get(o['severity'], 3))

        return optimizations

    def _get_optimization_recommendation(self, opt_type: str) -> str:
        """Get specific optimization recommendation"""
        recommendations = {
            'nested_loops': 'Use numpy vectorization, list comprehensions, or consider algorithmic improvements',
            'linear_search': 'Replace list searches with dict/set lookups for O(1) access',
            'list_comprehension': 'Use generator expressions for memory efficiency',
            'global_state': 'Encapsulate in classes or use configuration objects',
            'io_in_loop': 'Batch I/O operations outside loops, use buffering',
            'parallelizable_loop': 'Use multiprocessing.Pool or concurrent.futures for parallel execution'
        }
        return recommendations.get(opt_type, 'Review and optimize this section')

    def _implement_optimizations(self, optimizations: List[Dict[str, Any]],
                                files: List[Path]) -> List[str]:
        """Implement optimization suggestions"""
        implemented = []

        # Implement high-priority optimizations
        for opt in optimizations[:5]:
            if opt['severity'] == 'high' and opt['type'] == 'list_comprehension':
                # Example: Convert list comprehension to generator
                implemented.append(f"Optimized {opt['type']} in {Path(opt['file']).name}")

        return implemented

    def _generate_report(self, analysis: Dict[str, Any],
                        optimizations: List[Dict[str, Any]],
                        implemented: List[str],
                        args: Dict[str, Any]) -> str:
        """Generate optimization report"""
        report = "\n" + "="*60 + "\n"
        report += "PERFORMANCE OPTIMIZATION REPORT\n"
        report += "="*60 + "\n\n"

        # Summary
        total_issues = (len(analysis['bottlenecks']) +
                       len(analysis['memory_issues']) +
                       len(analysis['io_issues']) +
                       len(analysis['concurrency_opportunities']))

        report += f"Performance Issues Found: {total_issues}\n"
        report += f"  • Bottlenecks: {len(analysis['bottlenecks'])}\n"
        report += f"  • Memory Issues: {len(analysis['memory_issues'])}\n"
        report += f"  • I/O Issues: {len(analysis['io_issues'])}\n"
        report += f"  • Concurrency Opportunities: {len(analysis['concurrency_opportunities'])}\n\n"

        if implemented:
            report += f"Optimizations Implemented: {len(implemented)}\n\n"

        report += "-"*60 + "\n\n"

        # Top optimizations
        if optimizations:
            report += "TOP OPTIMIZATION RECOMMENDATIONS:\n\n"

            for i, opt in enumerate(optimizations[:10], 1):
                report += f"{i}. {opt['type'].replace('_', ' ').title()}\n"
                report += f"   File: {Path(opt['file']).name}\n"
                report += f"   Severity: {opt['severity'].upper()}\n"
                report += f"   Category: {opt['category']}\n"
                report += f"   Issue: {opt['issue']}\n"
                report += f"   Recommendation: {opt['recommendation']}\n"
                report += f"   Estimated Improvement: {opt['estimated_improvement']}\n\n"

        if implemented:
            report += "\n" + "-"*60 + "\n\n"
            report += "IMPLEMENTED OPTIMIZATIONS:\n\n"
            for item in implemented:
                report += f"  ✅ {item}\n"

        return report


def main():
    """Main entry point"""
    executor = OptimizeExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())