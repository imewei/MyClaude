#!/usr/bin/env python3
"""
Explain Code Command Executor
Advanced code analysis and documentation with multi-language support
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_executor import CommandExecutor, AgentOrchestrator
from ast_analyzer import ASTAnalyzer


class ExplainCodeExecutor(CommandExecutor):
    """Executor for /explain-code command"""

    def __init__(self):
        super().__init__("explain-code")
        self.ast_analyzer = ASTAnalyzer()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Code explanation engine')
        parser.add_argument('file_or_directory', nargs='?', default='.',
                          help='File or directory to explain')
        parser.add_argument('--level', type=str, default='basic',
                          choices=['basic', 'advanced', 'expert'])
        parser.add_argument('--focus', type=str, help='Specific area to focus on')
        parser.add_argument('--docs', action='store_true',
                          help='Generate documentation')
        parser.add_argument('--interactive', action='store_true',
                          help='Interactive explanation mode')
        parser.add_argument('--format', type=str, default='text',
                          choices=['text', 'markdown', 'html'])
        parser.add_argument('--export', type=str, help='Export path')
        parser.add_argument('--agents', type=str, default='documentation',
                          choices=['documentation', 'quality', 'scientific', 'all'])
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("📖 CODE EXPLANATION ENGINE")
        print("="*60 + "\n")

        try:
            target = Path(args.get('file_or_directory', '.'))
            if not target.exists():
                target = self.work_dir / target

            print(f"🎯 Analyzing: {target.name}")

            explanation = self._generate_explanation(target, args)

            if args.get('export'):
                export_path = Path(args['export'])
                self.write_file(export_path, explanation)
                print(f"\n💾 Exported to: {export_path}")

            return {
                'success': True,
                'summary': f'Explained {target.name}',
                'details': explanation
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Code explanation failed',
                'details': str(e)
            }

    def _generate_explanation(self, target: Path, args: Dict[str, Any]) -> str:
        """Generate code explanation"""
        level = args.get('level', 'basic')

        explanation = f"# Code Explanation: {target.name}\n\n"
        explanation += f"**Analysis Level:** {level}\n\n"

        if target.is_file():
            explanation += self._explain_file(target, level)
        else:
            explanation += self._explain_directory(target, level)

        return explanation

    def _explain_file(self, file: Path, level: str) -> str:
        """Explain a single file"""
        explanation = f"## File: {file.name}\n\n"

        try:
            analysis = self.ast_analyzer.analyze_file(file)

            if analysis:
                if analysis.get('classes'):
                    explanation += f"**Classes:** {len(analysis['classes'])}\n"
                if analysis.get('functions'):
                    explanation += f"**Functions:** {len(analysis['functions'])}\n"
                explanation += "\n"

                if level in ['advanced', 'expert']:
                    explanation += "### Detailed Analysis\n\n"
                    for cls in analysis.get('classes', [])[:3]:
                        explanation += f"**{cls.get('name')}**: "
                        explanation += f"{len(cls.get('methods', []))} methods\n"

        except Exception:
            explanation += "Unable to analyze file structure.\n"

        return explanation

    def _explain_directory(self, directory: Path, level: str) -> str:
        """Explain a directory structure"""
        explanation = f"## Directory: {directory.name}\n\n"

        py_files = list(directory.rglob('*.py'))[:10]
        explanation += f"**Python Files:** {len(py_files)}\n\n"

        if level in ['advanced', 'expert']:
            explanation += "### Key Files:\n\n"
            for f in py_files[:5]:
                explanation += f"- {f.name}\n"

        return explanation


def main():
    executor = ExplainCodeExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())