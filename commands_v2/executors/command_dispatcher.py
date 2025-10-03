#!/usr/bin/env python3
"""
Command Dispatcher for Claude Code Slash Commands
Routes slash command calls to appropriate executors
"""

import sys
import os
from pathlib import Path
from typing import Dict, Type, Optional

# Add executors to path
sys.path.insert(0, str(Path(__file__).parent))

from base_executor import CommandExecutor
from double_check_executor import DoubleCheckExecutor
from think_ultra_executor import ThinkUltraExecutor
from fix_commit_errors_executor import FixCommitErrorsExecutor
from adopt_code_executor import AdoptCodeExecutor
from commit_executor import CommitExecutor
from fix_github_issue_executor import FixGitHubIssueExecutor
from run_all_tests_executor import RunAllTestsExecutor


class CommandDispatcher:
    """Dispatches slash commands to their executors"""

    def __init__(self):
        self.executors: Dict[str, Type[CommandExecutor]] = {}
        self._register_executors()

    def _register_executors(self):
        """Register all available command executors"""
        self.executors['adopt-code'] = AdoptCodeExecutor
        self.executors['commit'] = CommitExecutor
        self.executors['double-check'] = DoubleCheckExecutor
        self.executors['fix-commit-errors'] = FixCommitErrorsExecutor
        self.executors['fix-github-issue'] = FixGitHubIssueExecutor
        self.executors['run-all-tests'] = RunAllTestsExecutor
        self.executors['think-ultra'] = ThinkUltraExecutor

    def dispatch(self, command_name: str, argv: list) -> int:
        """
        Dispatch command to appropriate executor

        Args:
            command_name: Name of the command (without /)
            argv: Command arguments

        Returns:
            Exit code from executor
        """
        if command_name not in self.executors:
            print(f"❌ Unknown command: /{command_name}")
            print(f"Available commands: {', '.join(self.executors.keys())}")
            return 1

        try:
            # Instantiate and run executor
            executor_class = self.executors[command_name]
            executor = executor_class()
            return executor.run(argv)

        except Exception as e:
            print(f"❌ Error executing /{command_name}: {e}")
            if os.environ.get('DEBUG'):
                import traceback
                traceback.print_exc()
            return 1

    def list_commands(self) -> list:
        """List all available commands"""
        return sorted(self.executors.keys())

    def get_command_help(self, command_name: str) -> Optional[str]:
        """Get help text for a command"""
        if command_name not in self.executors:
            return None

        executor_class = self.executors[command_name]
        executor = executor_class()
        parser = executor.get_parser()
        return parser.format_help()


def main():
    """Main entry point for command dispatcher"""

    if len(sys.argv) < 2:
        print("Claude Code Command Dispatcher")
        print("\nUsage: command_dispatcher.py <command-name> [args...]")
        print("\nAvailable commands:")
        dispatcher = CommandDispatcher()
        for cmd in dispatcher.list_commands():
            print(f"  /{cmd}")
        return 1

    command_name = sys.argv[1]
    command_args = sys.argv[2:]

    # Remove leading slash if present
    if command_name.startswith('/'):
        command_name = command_name[1:]

    dispatcher = CommandDispatcher()
    return dispatcher.dispatch(command_name, command_args)


if __name__ == "__main__":
    sys.exit(main())