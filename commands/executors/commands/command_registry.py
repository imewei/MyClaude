#!/usr/bin/env python3
"""
Command Registry
Central registry for all command executors
"""

from typing import Dict, Type, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all executors
from update_docs_executor import UpdateDocsExecutor
from refactor_clean_executor import RefactorCleanExecutor
from optimize_executor import OptimizeExecutor
from generate_tests_executor import GenerateTestsExecutor
from explain_code_executor import ExplainCodeExecutor
from debug_executor import DebugExecutor
from clean_codebase_executor import CleanCodebaseExecutor
from ci_setup_executor import CiSetupExecutor
from check_code_quality_executor import CheckCodeQualityExecutor
from reflection_executor import ReflectionExecutor
from multi_agent_optimize_executor import MultiAgentOptimizeExecutor

# Import from parent directory
from commit_executor import CommitExecutor
from run_all_tests_executor import RunAllTestsExecutor
from fix_github_issue_executor import FixGitHubIssueExecutor
from adopt_code_executor import AdoptCodeExecutor
from fix_commit_errors_executor import FixCommitErrorsExecutor
from think_ultra_executor import ThinkUltraExecutor
from double_check_executor import DoubleCheckExecutor
from base_executor import CommandExecutor


class CommandRegistry:
    """Registry for all command executors"""

    def __init__(self):
        self._commands: Dict[str, Type[CommandExecutor]] = {}
        self._register_all_commands()

    def _register_all_commands(self):
        """Register all available command executors"""

        # Critical Automation (Phase 1)
        self.register('commit', CommitExecutor)
        self.register('run-all-tests', RunAllTestsExecutor)
        self.register('fix-github-issue', FixGitHubIssueExecutor)
        self.register('adopt-code', AdoptCodeExecutor)

        # Code Quality & Testing (Phase 2)
        self.register('fix-commit-errors', FixCommitErrorsExecutor)
        self.register('generate-tests', GenerateTestsExecutor)
        self.register('check-code-quality', CheckCodeQualityExecutor)
        self.register('clean-codebase', CleanCodebaseExecutor)
        self.register('refactor-clean', RefactorCleanExecutor)

        # Advanced Features (Phase 3)
        self.register('optimize', OptimizeExecutor)
        self.register('multi-agent-optimize', MultiAgentOptimizeExecutor)
        self.register('ci-setup', CiSetupExecutor)
        self.register('debug', DebugExecutor)
        self.register('update-docs', UpdateDocsExecutor)
        self.register('reflection', ReflectionExecutor)
        self.register('explain-code', ExplainCodeExecutor)

        # Existing Advanced Commands
        self.register('think-ultra', ThinkUltraExecutor)
        self.register('double-check', DoubleCheckExecutor)

    def register(self, command_name: str, executor_class: Type[CommandExecutor]):
        """
        Register a command executor

        Args:
            command_name: Name of the command (e.g., 'commit', 'run-all-tests')
            executor_class: Executor class to handle the command
        """
        self._commands[command_name] = executor_class

    def get_executor(self, command_name: str) -> Optional[Type[CommandExecutor]]:
        """
        Get executor class for a command

        Args:
            command_name: Name of the command

        Returns:
            Executor class or None if not found
        """
        return self._commands.get(command_name)

    def list_commands(self) -> list:
        """Get list of all registered commands"""
        return sorted(self._commands.keys())

    def has_command(self, command_name: str) -> bool:
        """Check if a command is registered"""
        return command_name in self._commands

    def get_command_categories(self) -> Dict[str, list]:
        """Get commands organized by category"""
        return {
            'Critical Automation': [
                'commit',
                'run-all-tests',
                'fix-github-issue',
                'adopt-code'
            ],
            'Code Quality & Testing': [
                'fix-commit-errors',
                'generate-tests',
                'check-code-quality',
                'clean-codebase',
                'refactor-clean'
            ],
            'Advanced Features': [
                'optimize',
                'multi-agent-optimize',
                'ci-setup',
                'debug',
                'update-docs',
                'reflection',
                'explain-code'
            ],
            'Analysis & Verification': [
                'think-ultra',
                'double-check'
            ]
        }


# Global registry instance
_registry = CommandRegistry()


def get_registry() -> CommandRegistry:
    """Get the global command registry"""
    return _registry


def get_executor_for_command(command_name: str) -> Optional[CommandExecutor]:
    """
    Get an instantiated executor for a command

    Args:
        command_name: Name of the command

    Returns:
        Executor instance or None if not found
    """
    executor_class = _registry.get_executor(command_name)
    if executor_class:
        return executor_class()
    return None


def list_all_commands() -> list:
    """List all available commands"""
    return _registry.list_commands()


def main():
    """Display command registry information"""
    print("Command Registry")
    print("=" * 60)
    print()

    categories = _registry.get_command_categories()

    for category, commands in categories.items():
        print(f"\n{category}:")
        for cmd in commands:
            print(f"  • {cmd}")

    print(f"\n\nTotal Commands: {len(_registry.list_commands())}")


if __name__ == "__main__":
    main()