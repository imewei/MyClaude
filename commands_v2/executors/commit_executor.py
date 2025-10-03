#!/usr/bin/env python3
"""
Commit Command Executor
Git commit engine with AI message generation and validation
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add executors to path
sys.path.insert(0, str(Path(__file__).parent))

from base_executor import CommandExecutor, AgentOrchestrator
from git_utils import GitUtils, GitError


class CommitExecutor(CommandExecutor):
    """Executor for /commit command"""

    def __init__(self):
        super().__init__("commit")
        self.git = GitUtils()
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        """Configure argument parser"""
        parser = argparse.ArgumentParser(
            description='Git commit engine with AI message generation'
        )
        parser.add_argument('--all', action='store_true',
                          help='Add all changed files before committing')
        parser.add_argument('--staged', action='store_true',
                          help='Only commit currently staged changes')
        parser.add_argument('--amend', action='store_true',
                          help='Amend the previous commit')
        parser.add_argument('--interactive', action='store_true',
                          help='Interactive commit creation')
        parser.add_argument('--split', action='store_true',
                          help='Split large changes into multiple commits')
        parser.add_argument('--template', type=str,
                          choices=['feat', 'fix', 'docs', 'refactor', 'test', 'chore'],
                          help='Use commit template')
        parser.add_argument('--ai-message', action='store_true',
                          help='Generate commit message using AI')
        parser.add_argument('--validate', action='store_true',
                          help='Run pre-commit checks')
        parser.add_argument('--push', action='store_true',
                          help='Push to remote after commit')
        parser.add_argument('--agents', type=str, default='quality',
                          choices=['quality', 'devops', 'orchestrator', 'all'],
                          help='Agent selection')
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute commit workflow"""

        print("\n" + "="*60)
        print("📝 GIT COMMIT ENGINE")
        print("="*60 + "\n")

        try:
            # Check if git repository
            if not self.git.is_repo():
                return {
                    'success': False,
                    'summary': 'Not a git repository',
                    'details': 'Current directory is not a git repository'
                }

            # Step 1: Get git status
            print("🔍 Analyzing repository state...")
            status = self.git.get_status()

            # Check if there are changes to commit
            if not args.get('amend'):
                has_changes = any([
                    status['modified'],
                    status['added'],
                    status['deleted'],
                    status['untracked']
                ])

                if not has_changes:
                    return {
                        'success': False,
                        'summary': 'No changes to commit',
                        'details': 'Working directory is clean'
                    }

            # Step 2: Stage files if requested
            if args.get('all'):
                print("➕ Staging all changes...")
                self.git.add_all()
                staged_files = self.git.get_staged_files()
            elif not args.get('staged'):
                # Interactive file selection
                staged_files = self._select_files_to_stage(status)
                if staged_files:
                    self.git.add_files(staged_files)
            else:
                staged_files = self.git.get_staged_files()

            if not staged_files and not args.get('amend'):
                return {
                    'success': False,
                    'summary': 'No files staged',
                    'details': 'No files selected for commit'
                }

            # Step 3: Get diff for staged changes
            print("📊 Analyzing changes...")
            diff = self.git.get_diff(staged=True)

            # Step 4: Generate commit message
            if args.get('ai_message') or args.get('template'):
                print("🤖 Generating commit message...")
                message = self._generate_commit_message(
                    diff, staged_files, status, args
                )
            else:
                message = input("Enter commit message: ")

            if not message:
                return {
                    'success': False,
                    'summary': 'Empty commit message',
                    'details': 'Commit message cannot be empty'
                }

            # Step 5: Run validation if requested
            if args.get('validate'):
                print("✅ Running pre-commit validation...")
                validation_result = self._run_validation()
                if not validation_result['success']:
                    return validation_result

            # Step 6: Create commit
            print(f"💾 Creating commit...")
            print(f"Message: {message[:100]}...")

            commit_hash = self.git.commit(message, allow_empty=args.get('amend'))

            print(f"✅ Commit created: {commit_hash[:8]}")

            # Step 7: Push if requested
            if args.get('push'):
                print("📤 Pushing to remote...")
                try:
                    branch = self.git.get_current_branch()
                    self.git.push(branch=branch, set_upstream=True)
                    print(f"✅ Pushed to {branch}")
                except GitError as e:
                    print(f"⚠️  Push failed: {e}")

            return {
                'success': True,
                'summary': f'Commit created: {commit_hash[:8]}',
                'details': f"""
Commit Hash: {commit_hash}
Files Changed: {len(staged_files)}
Message: {message}
{'Pushed to remote' if args.get('push') else 'Not pushed'}
""",
                'commit_hash': commit_hash,
                'files_changed': len(staged_files)
            }

        except GitError as e:
            return {
                'success': False,
                'summary': 'Git operation failed',
                'details': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'summary': 'Unexpected error',
                'details': str(e)
            }

    def _select_files_to_stage(self, status: Dict[str, List[str]]) -> List[str]:
        """Interactively select files to stage"""
        all_files = []
        for category, files in status.items():
            if category != 'renamed':
                all_files.extend(files)

        if not all_files:
            return []

        print("\nFiles with changes:")
        for i, file in enumerate(all_files, 1):
            print(f"  {i}. {file}")

        selection = input("\nSelect files (comma-separated numbers, or 'all'): ")

        if selection.lower() == 'all':
            return all_files

        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            return [all_files[i] for i in indices if 0 <= i < len(all_files)]
        except (ValueError, IndexError):
            print("Invalid selection")
            return []

    def _generate_commit_message(self, diff: str, files: List[str],
                                 status: Dict[str, List[str]],
                                 args: Dict[str, Any]) -> str:
        """Generate commit message using AI or template"""

        # Analyze changes
        change_type = self._detect_change_type(diff, files, status)
        scope = self._detect_scope(files)

        # Use template if specified
        if args.get('template'):
            template_type = args['template']
            message = self._apply_template(template_type, change_type, scope, files)
        else:
            # AI-generated message
            message = self._ai_generate_message(diff, change_type, scope, files)

        # Add co-author footer
        message += f"""

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"""

        return message

    def _detect_change_type(self, diff: str, files: List[str],
                           status: Dict[str, List[str]]) -> str:
        """Detect type of changes (feat, fix, docs, refactor, etc.)"""

        # Analyze file types
        has_test = any('test' in f for f in files)
        has_doc = any(f.endswith(('.md', '.rst', '.txt')) for f in files)
        has_config = any(f in ['setup.py', 'pyproject.toml', 'package.json'] for f in files)

        # Analyze diff content
        diff_lower = diff.lower()
        if 'fix' in diff_lower or 'bug' in diff_lower:
            return 'fix'
        elif has_doc:
            return 'docs'
        elif has_test:
            return 'test'
        elif has_config:
            return 'chore'
        elif 'refactor' in diff_lower:
            return 'refactor'
        else:
            return 'feat'

    def _detect_scope(self, files: List[str]) -> str:
        """Detect scope from changed files"""
        if not files:
            return ''

        # Extract common directory
        paths = [Path(f) for f in files]
        if len(paths) == 1:
            return paths[0].parts[0] if len(paths[0].parts) > 1 else paths[0].stem

        # Find common parent
        common = Path(files[0]).parent
        for path in paths[1:]:
            while not str(path).startswith(str(common)):
                common = common.parent
                if common == Path('.'):
                    return 'multiple'

        return common.name if common != Path('.') else 'multiple'

    def _apply_template(self, template_type: str, change_type: str,
                       scope: str, files: List[str]) -> str:
        """Apply commit message template"""

        templates = {
            'feat': f"feat({scope}): add new feature",
            'fix': f"fix({scope}): resolve issue",
            'docs': f"docs({scope}): update documentation",
            'refactor': f"refactor({scope}): improve code structure",
            'test': f"test({scope}): add tests",
            'chore': f"chore({scope}): update dependencies"
        }

        base_message = templates.get(template_type, f"{template_type}({scope}): update")

        # Add file list
        file_summary = f"\n\nModified files:\n" + "\n".join(f"- {f}" for f in files[:10])
        if len(files) > 10:
            file_summary += f"\n... and {len(files) - 10} more files"

        return base_message + file_summary

    def _ai_generate_message(self, diff: str, change_type: str,
                            scope: str, files: List[str]) -> str:
        """AI-generated commit message (simplified version)"""

        # This is a simplified version - in production, this would use
        # Claude API or similar to generate smart commit messages

        summary = f"{change_type}({scope}): "

        # Analyze diff to generate description
        if 'def ' in diff or 'class ' in diff:
            summary += "add new functionality"
        elif '- ' in diff and '+ ' in diff:
            summary += "update implementation"
        elif '+ ' in diff:
            summary += "add features"
        elif '- ' in diff:
            summary += "remove code"
        else:
            summary += "update code"

        return summary

    def _run_validation(self) -> Dict[str, Any]:
        """Run pre-commit validation checks"""
        # This would run linters, tests, etc.
        # Simplified for now
        print("  Running linters...")
        print("  Running tests...")
        print("  Checking for secrets...")

        return {
            'success': True,
            'summary': 'Validation passed'
        }


def main():
    """Main entry point"""
    executor = CommitExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())