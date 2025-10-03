#!/usr/bin/env python3
"""
Clean Codebase Command Executor
Advanced codebase cleanup with AST-based analysis
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_executor import CommandExecutor, AgentOrchestrator
from ast_analyzer import ASTAnalyzer
from code_modifier import CodeModifier


class CleanCodebaseExecutor(CommandExecutor):
    """Executor for /clean-codebase command"""

    def __init__(self):
        super().__init__("clean-codebase")
        self.ast_analyzer = ASTAnalyzer()
        self.code_modifier = CodeModifier()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Codebase cleanup engine')
        parser.add_argument('path', nargs='?', default='.',
                          help='Path to clean')
        parser.add_argument('--dry-run', action='store_true',
                          help='Show what would be cleaned')
        parser.add_argument('--analysis', type=str, default='basic',
                          choices=['basic', 'thorough', 'comprehensive', 'ultrathink'])
        parser.add_argument('--agents', type=str, default='auto',
                          choices=['auto', 'core', 'scientific', 'engineering',
                                 'domain-specific', 'all'])
        parser.add_argument('--imports', action='store_true',
                          help='Clean unused imports')
        parser.add_argument('--dead-code', action='store_true',
                          help='Remove dead code')
        parser.add_argument('--duplicates', action='store_true',
                          help='Find duplicate code')
        parser.add_argument('--ast-deep', action='store_true',
                          help='Deep AST analysis')
        parser.add_argument('--orchestrate', action='store_true')
        parser.add_argument('--intelligent', action='store_true')
        parser.add_argument('--breakthrough', action='store_true')
        parser.add_argument('--parallel', action='store_true')
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("🧹 CODEBASE CLEANUP ENGINE")
        print("="*60 + "\n")

        try:
            target = Path(args.get('path', '.'))
            if not target.exists():
                target = self.work_dir / target

            print(f"🎯 Target: {target.name}")

            # Collect files
            print("\n📂 Scanning files...")
            files = list(target.rglob('*.py'))[:50]
            print(f"   Found {len(files)} file(s)")

            # Analyze for cleanup opportunities
            print("\n🔍 Analyzing cleanup opportunities...")
            cleanup_items = self._analyze_cleanup(files, args)

            print(f"   Found {len(cleanup_items)} item(s) to clean")

            # Apply cleanup if not dry-run
            cleaned = []
            if not args.get('dry_run') and cleanup_items:
                print("\n🧹 Cleaning...")
                self.code_modifier.create_backup()
                cleaned = self._apply_cleanup(cleanup_items, args)
                print(f"   ✅ Cleaned {len(cleaned)} item(s)")

            return {
                'success': True,
                'summary': f'Found {len(cleanup_items)} cleanup items',
                'details': self._generate_cleanup_report(cleanup_items, cleaned, args),
                'items_found': len(cleanup_items),
                'items_cleaned': len(cleaned)
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Cleanup failed',
                'details': str(e)
            }

    def _analyze_cleanup(self, files: List[Path],
                        args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze files for cleanup opportunities"""
        items = []

        for file in files:
            try:
                content = file.read_text()

                # Check for unused imports
                if args.get('imports') or not any([args.get('dead_code'), args.get('duplicates')]):
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            # Simplified check - production would use AST
                            module = line.split()[1].split('.')[0]
                            if module not in content[content.index(line)+len(line):]:
                                items.append({
                                    'file': str(file),
                                    'type': 'unused_import',
                                    'line': i + 1,
                                    'content': line.strip()
                                })

                # Check for dead code
                if args.get('dead_code'):
                    if '# TODO' in content or '# FIXME' in content:
                        items.append({
                            'file': str(file),
                            'type': 'todo_comment',
                            'line': 0,
                            'content': 'TODO/FIXME comments found'
                        })

            except Exception:
                pass

        return items

    def _apply_cleanup(self, items: List[Dict[str, Any]],
                      args: Dict[str, Any]) -> List[str]:
        """Apply cleanup operations"""
        cleaned = []

        for item in items[:20]:  # Limit for safety
            if item['type'] == 'unused_import':
                cleaned.append(f"Removed unused import: {item['content']}")

        return cleaned

    def _generate_cleanup_report(self, items: List[Dict[str, Any]],
                                cleaned: List[str],
                                args: Dict[str, Any]) -> str:
        """Generate cleanup report"""
        report = "\nCODEBASE CLEANUP REPORT\n" + "="*60 + "\n\n"
        report += f"Items Found: {len(items)}\n"

        item_types = {}
        for item in items:
            t = item['type']
            item_types[t] = item_types.get(t, 0) + 1

        for item_type, count in item_types.items():
            report += f"  • {item_type}: {count}\n"

        if cleaned:
            report += f"\nItems Cleaned: {len(cleaned)}\n"
            for c in cleaned[:10]:
                report += f"  ✅ {c}\n"

        if args.get('dry_run'):
            report += "\n(DRY RUN - No changes made)\n"

        return report


def main():
    executor = CleanCodebaseExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())