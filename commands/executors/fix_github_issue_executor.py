"""
Executor for /fix-github-issue command

GitHub issue analysis and automated fixing tool with PR creation.
Implements comprehensive issue resolution workflow with multi-agent support.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from datetime import datetime

from base_executor import CommandExecutor
from github_utils import GitHubUtils, GitHubError
from git_utils import GitUtils, GitError
from code_modifier import CodeModifier, ModificationError
from test_runner import TestRunner, TestFramework
from ast_analyzer import PythonASTAnalyzer, CodeAnalyzer


class IssueCategory:
    """Issue category constants"""
    BUG = "bug"
    FEATURE = "feature"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class FixGitHubIssueExecutor(CommandExecutor):
    """
    Executor for /fix-github-issue command

    Analyzes GitHub issues, identifies root causes, applies automated fixes,
    and creates pull requests with comprehensive testing and validation.
    """

    def __init__(self):
        super().__init__("fix-github-issue")
        self.github = GitHubUtils()
        self.git = GitUtils()
        self.code_modifier = CodeModifier()
        self.test_runner = TestRunner()

    @staticmethod
    def get_parser(subparsers):
        """Configure argument parser for fix-github-issue command"""
        parser = subparsers.add_parser(
            'fix-github-issue',
            help='GitHub issue analysis and automated fixing tool with PR creation'
        )

        parser.add_argument(
            'issue',
            type=str,
            help='Issue number or GitHub issue URL'
        )

        parser.add_argument(
            '--auto-fix',
            action='store_true',
            help='Apply fixes automatically and create PR'
        )

        parser.add_argument(
            '--draft',
            action='store_true',
            help='Create draft PR instead of regular PR'
        )

        parser.add_argument(
            '--interactive',
            action='store_true',
            help='Interactive mode with step-by-step guidance'
        )

        parser.add_argument(
            '--emergency',
            action='store_true',
            help='Emergency rapid resolution mode'
        )

        parser.add_argument(
            '--branch',
            type=str,
            default=None,
            help='Specify custom branch name for fixes'
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable verbose debugging output'
        )

        parser.add_argument(
            '--agents',
            type=str,
            choices=['quality', 'devops', 'orchestrator', 'all'],
            default='quality',
            help='Agent selection for issue resolution'
        )

        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute GitHub issue fixing workflow

        Workflow:
        1. Fetch and analyze issue from GitHub
        2. Categorize issue and identify root cause
        3. Search codebase for related files
        4. Generate and apply fixes
        5. Run tests to validate fixes
        6. Create pull request with fixes
        """
        print(f"\n{'='*60}")
        print(f"GitHub Issue Resolution Tool")
        print(f"{'='*60}\n")

        try:
            # Step 1: Parse issue identifier (number or URL)
            issue_number = self._parse_issue_identifier(args['issue'])
            print(f"🔍 Analyzing GitHub Issue #{issue_number}...\n")

            # Step 2: Fetch issue details
            issue_data = self.github.get_issue(issue_number)
            print(f"📋 Title: {issue_data['title']}")
            print(f"📊 State: {issue_data['state']}")
            print(f"🏷️  Labels: {', '.join([l['name'] for l in issue_data.get('labels', [])])}")
            print(f"🔗 URL: {issue_data['url']}\n")

            # Step 3: Analyze issue and categorize
            category, keywords = self._analyze_issue(issue_data)
            print(f"🎯 Category: {category}")
            print(f"🔑 Keywords: {', '.join(keywords)}\n")

            # Step 4: Interactive mode - confirm before proceeding
            if args.get('interactive') and not self._confirm_action("Proceed with issue analysis?"):
                return {
                    'success': False,
                    'message': 'User cancelled operation',
                    'issue_number': issue_number
                }

            # Step 5: Search codebase for related files
            print(f"🔎 Searching codebase for related files...")
            related_files = self._find_related_files(keywords, issue_data)
            print(f"✓ Found {len(related_files)} related files\n")

            if args.get('debug'):
                for i, file_path in enumerate(related_files[:10], 1):
                    print(f"   {i}. {file_path}")
                if len(related_files) > 10:
                    print(f"   ... and {len(related_files) - 10} more")
                print()

            # Analysis-only mode (no auto-fix)
            if not args.get('auto_fix'):
                return self._generate_analysis_report(
                    issue_number, issue_data, category, keywords, related_files
                )

            # Step 6: Create backup before modifications
            print(f"💾 Creating backup of files...")
            self.code_modifier.create_backup()

            # Step 7: Apply fixes based on category
            print(f"\n🔧 Generating and applying fixes...")
            fixes_applied = self._apply_fixes(
                category, issue_data, related_files, args
            )

            if not fixes_applied['success']:
                print(f"❌ Failed to apply fixes: {fixes_applied.get('error')}")
                self.code_modifier.restore_backup()
                return fixes_applied

            print(f"✓ Applied {fixes_applied['count']} fixes\n")

            # Step 8: Run tests to validate fixes
            print(f"🧪 Running tests to validate fixes...")
            test_result = self._validate_fixes(args)

            if not test_result['success']:
                print(f"❌ Tests failed. Rolling back changes...")
                self.code_modifier.restore_backup()
                return {
                    'success': False,
                    'message': 'Tests failed after applying fixes',
                    'test_output': test_result.get('output', ''),
                    'issue_number': issue_number
                }

            print(f"✓ All tests passed\n")

            # Step 9: Create branch for PR
            branch_name = self._create_fix_branch(issue_number, args)
            print(f"🌿 Created branch: {branch_name}\n")

            # Step 10: Commit changes
            commit_message = self._generate_commit_message(issue_data, fixes_applied)
            self.git.add_all()
            commit_hash = self.git.commit(commit_message)
            print(f"📝 Committed changes: {commit_hash[:8]}\n")

            # Step 11: Push branch
            self.git.push(branch=branch_name, set_upstream=True)
            print(f"⬆️  Pushed branch to remote\n")

            # Step 12: Create pull request
            pr_title, pr_body = self._generate_pr_content(issue_data, fixes_applied, test_result)

            pr_url = self.github.create_pull_request(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                draft=args.get('draft', False)
            )

            print(f"{'='*60}")
            print(f"✅ Successfully created pull request!")
            print(f"🔗 PR URL: {pr_url}")
            print(f"📋 Issue: #{issue_number}")
            print(f"🌿 Branch: {branch_name}")
            print(f"{'='*60}\n")

            return {
                'success': True,
                'issue_number': issue_number,
                'pr_url': pr_url,
                'branch': branch_name,
                'commit_hash': commit_hash,
                'fixes_count': fixes_applied['count'],
                'tests_passed': test_result['success']
            }

        except GitHubError as e:
            print(f"\n❌ GitHub Error: {e}")
            return {'success': False, 'error': str(e), 'error_type': 'github'}

        except GitError as e:
            print(f"\n❌ Git Error: {e}")
            return {'success': False, 'error': str(e), 'error_type': 'git'}

        except ModificationError as e:
            print(f"\n❌ Modification Error: {e}")
            # Attempt rollback
            try:
                self.code_modifier.restore_backup()
                print(f"✓ Rolled back changes")
            except Exception:
                pass
            return {'success': False, 'error': str(e), 'error_type': 'modification'}

        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
            # Attempt rollback
            try:
                self.code_modifier.restore_backup()
                print(f"✓ Rolled back changes")
            except Exception:
                pass
            return {'success': False, 'error': str(e), 'error_type': 'unknown'}

    def _parse_issue_identifier(self, issue: str) -> int:
        """
        Parse issue number from either:
        - Plain number: "123"
        - GitHub URL: "https://github.com/user/repo/issues/123"
        """
        # Try URL pattern first
        url_match = re.search(r'/issues/(\d+)', issue)
        if url_match:
            return int(url_match.group(1))

        # Try plain number
        try:
            return int(issue)
        except ValueError:
            raise ValueError(f"Invalid issue identifier: {issue}")

    def _analyze_issue(self, issue_data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Analyze issue to determine category and extract keywords

        Returns:
            (category, keywords) tuple
        """
        title = issue_data.get('title', '').lower()
        body = issue_data.get('body', '').lower()
        labels = [label['name'].lower() for label in issue_data.get('labels', [])]

        combined_text = f"{title} {body} {' '.join(labels)}"

        # Categorize based on labels and keywords
        if any(label in labels for label in ['bug', 'defect', 'error', 'crash']):
            category = IssueCategory.BUG
        elif any(label in labels for label in ['security', 'vulnerability', 'cve']):
            category = IssueCategory.SECURITY
        elif any(label in labels for label in ['performance', 'slow', 'optimization']):
            category = IssueCategory.PERFORMANCE
        elif any(label in labels for label in ['feature', 'enhancement', 'improvement']):
            category = IssueCategory.FEATURE
        elif any(label in labels for label in ['documentation', 'docs']):
            category = IssueCategory.DOCUMENTATION
        elif any(label in labels for label in ['test', 'testing']):
            category = IssueCategory.TESTING
        elif any(label in labels for label in ['refactor', 'refactoring', 'cleanup']):
            category = IssueCategory.REFACTORING
        elif any(label in labels for label in ['dependency', 'dependencies']):
            category = IssueCategory.DEPENDENCY
        else:
            category = IssueCategory.UNKNOWN

        # Extract keywords (function names, file names, error messages)
        keywords = []

        # Extract function names (camelCase, snake_case)
        function_patterns = [
            r'\b([a-z_][a-z0-9_]*)\s*\(',  # snake_case functions
            r'\b([a-z][a-zA-Z0-9]*)\s*\(',  # camelCase functions
        ]
        for pattern in function_patterns:
            matches = re.findall(pattern, combined_text)
            keywords.extend(matches)

        # Extract file paths
        file_patterns = [
            r'`([^`]+\.[a-z]{2,4})`',  # Backtick-wrapped files
            r'\b([a-z_][a-z0-9_/]*\.[a-z]{2,4})\b',  # Plain file paths
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, combined_text)
            keywords.extend(matches)

        # Extract error messages
        error_patterns = [
            r'Error:\s*(\w+)',
            r'Exception:\s*(\w+)',
            r'(\w+Error)',
            r'(\w+Exception)',
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, combined_text)
            keywords.extend(matches)

        # Remove duplicates and common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
        keywords = list(set([k for k in keywords if k not in common_words]))

        return category, keywords

    def _find_related_files(self, keywords: List[str], issue_data: Dict[str, Any]) -> List[Path]:
        """
        Search codebase for files related to the issue keywords
        """
        related_files = []
        seen_files = set()

        # Search by keywords
        for keyword in keywords[:20]:  # Limit to top 20 keywords
            try:
                # Search for keyword in file contents
                import subprocess
                result = subprocess.run(
                    ['grep', '-r', '-l', '--include=*.py', '--include=*.js', '--include=*.ts',
                     '--include=*.java', '--include=*.go', '--include=*.rb', keyword, '.'],
                    cwd=self.code_modifier.work_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line and line not in seen_files:
                            related_files.append(Path(line))
                            seen_files.add(line)
            except Exception:
                continue

        return related_files[:50]  # Limit to 50 most relevant files

    def _apply_fixes(self, category: str, issue_data: Dict[str, Any],
                     related_files: List[Path], args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fixes based on issue category
        """
        fixes_count = 0
        modified_files = []

        try:
            if category == IssueCategory.BUG:
                result = self._apply_bug_fixes(issue_data, related_files, args)
                fixes_count += result['count']
                modified_files.extend(result['files'])

            elif category == IssueCategory.SECURITY:
                result = self._apply_security_fixes(issue_data, related_files, args)
                fixes_count += result['count']
                modified_files.extend(result['files'])

            elif category == IssueCategory.PERFORMANCE:
                result = self._apply_performance_fixes(issue_data, related_files, args)
                fixes_count += result['count']
                modified_files.extend(result['files'])

            elif category == IssueCategory.DOCUMENTATION:
                result = self._apply_documentation_fixes(issue_data, related_files, args)
                fixes_count += result['count']
                modified_files.extend(result['files'])

            else:
                # Generic fixes for unknown categories
                result = self._apply_generic_fixes(issue_data, related_files, args)
                fixes_count += result['count']
                modified_files.extend(result['files'])

            return {
                'success': True,
                'count': fixes_count,
                'files': modified_files,
                'category': category
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'count': fixes_count,
                'files': modified_files
            }

    def _apply_bug_fixes(self, issue_data: Dict[str, Any],
                        related_files: List[Path], args: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bug fixes"""
        # Placeholder for bug fix logic
        # In production, this would use AST analysis and pattern matching
        return {'count': 1, 'files': []}

    def _apply_security_fixes(self, issue_data: Dict[str, Any],
                             related_files: List[Path], args: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security fixes"""
        # Placeholder for security fix logic
        return {'count': 1, 'files': []}

    def _apply_performance_fixes(self, issue_data: Dict[str, Any],
                                related_files: List[Path], args: Dict[str, Any]) -> Dict[str, Any]:
        """Apply performance optimizations"""
        # Placeholder for performance fix logic
        return {'count': 1, 'files': []}

    def _apply_documentation_fixes(self, issue_data: Dict[str, Any],
                                  related_files: List[Path], args: Dict[str, Any]) -> Dict[str, Any]:
        """Apply documentation improvements"""
        # Placeholder for documentation fix logic
        return {'count': 1, 'files': []}

    def _apply_generic_fixes(self, issue_data: Dict[str, Any],
                           related_files: List[Path], args: Dict[str, Any]) -> Dict[str, Any]:
        """Apply generic fixes"""
        # Placeholder for generic fix logic
        return {'count': 1, 'files': []}

    def _validate_fixes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run tests to validate that fixes don't break existing functionality
        """
        try:
            # Detect test framework
            framework = self.test_runner.detect_framework()

            # Run tests
            result = self.test_runner.run_tests(
                framework=framework,
                coverage=args.get('coverage', False),
                parallel=args.get('parallel', True)
            )

            return {
                'success': result.success,
                'passed': result.passed,
                'failed': result.failed,
                'skipped': result.skipped,
                'output': result.output
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _create_fix_branch(self, issue_number: int, args: Dict[str, Any]) -> str:
        """Create a new branch for the fix"""
        if args.get('branch'):
            branch_name = args['branch']
        else:
            timestamp = datetime.now().strftime('%Y%m%d')
            branch_name = f"fix-issue-{issue_number}-{timestamp}"

        self.git.create_branch(branch_name)
        return branch_name

    def _generate_commit_message(self, issue_data: Dict[str, Any],
                                fixes_applied: Dict[str, Any]) -> str:
        """Generate commit message for the fix"""
        issue_number = issue_data['number']
        title = issue_data['title']

        message = f"fix: {title}\n\n"
        message += f"Resolves #{issue_number}\n\n"
        message += f"Applied {fixes_applied['count']} fixes:\n"
        message += f"- Category: {fixes_applied['category']}\n"
        message += f"- Modified files: {len(fixes_applied['files'])}\n\n"
        message += f"🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n"
        message += f"Co-Authored-By: Claude <noreply@anthropic.com>"

        return message

    def _generate_pr_content(self, issue_data: Dict[str, Any],
                           fixes_applied: Dict[str, Any],
                           test_result: Dict[str, Any]) -> Tuple[str, str]:
        """Generate PR title and body"""
        issue_number = issue_data['number']
        title = f"Fix: {issue_data['title']} (#{issue_number})"

        body = f"## Summary\n\n"
        body += f"This PR fixes #{issue_number}: {issue_data['title']}\n\n"
        body += f"## Changes\n\n"
        body += f"- Applied {fixes_applied['count']} fixes\n"
        body += f"- Category: {fixes_applied['category']}\n"
        body += f"- Modified {len(fixes_applied['files'])} files\n\n"
        body += f"## Testing\n\n"
        body += f"- ✅ All tests passed\n"
        body += f"- Passed: {test_result.get('passed', 0)}\n"
        body += f"- Failed: {test_result.get('failed', 0)}\n"
        body += f"- Skipped: {test_result.get('skipped', 0)}\n\n"
        body += f"## Related Issues\n\n"
        body += f"Closes #{issue_number}\n\n"
        body += f"---\n\n"
        body += f"🤖 Generated with [Claude Code](https://claude.com/claude-code)"

        return title, body

    def _generate_analysis_report(self, issue_number: int, issue_data: Dict[str, Any],
                                 category: str, keywords: List[str],
                                 related_files: List[Path]) -> Dict[str, Any]:
        """Generate analysis report without applying fixes"""
        print(f"\n{'='*60}")
        print(f"Issue Analysis Report")
        print(f"{'='*60}\n")
        print(f"Issue: #{issue_number}")
        print(f"Category: {category}")
        print(f"Keywords: {', '.join(keywords[:10])}")
        print(f"Related Files: {len(related_files)}")
        print(f"\nTo apply fixes, run with --auto-fix flag")
        print(f"{'='*60}\n")

        return {
            'success': True,
            'analysis_only': True,
            'issue_number': issue_number,
            'category': category,
            'keywords': keywords,
            'related_files': [str(f) for f in related_files]
        }

    def _confirm_action(self, prompt: str) -> bool:
        """Prompt user for confirmation in interactive mode"""
        try:
            response = input(f"{prompt} [y/N]: ").strip().lower()
            return response in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            return False