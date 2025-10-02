#!/usr/bin/env python3
"""
Command-Line Interface for Command Executors
Provides unified CLI access to all command executors
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

from command_registry import get_registry, get_executor_for_command


class CommandCLI:
    """Command-line interface for executors"""

    def __init__(self):
        self.registry = get_registry()

    def run(self, argv: Optional[List[str]] = None) -> int:
        """
        Run command from command line

        Args:
            argv: Command line arguments (defaults to sys.argv[1:])

        Returns:
            Exit code
        """
        if argv is None:
            argv = sys.argv[1:]

        # Create main parser
        parser = argparse.ArgumentParser(
            description='Claude Code Command Executor CLI',
            epilog='Run a command with --help for command-specific options'
        )

        parser.add_argument('command', nargs='?',
                          help='Command to execute')
        parser.add_argument('--list', action='store_true',
                          help='List all available commands')
        parser.add_argument('--categories', action='store_true',
                          help='Show commands by category')

        # Parse just the command name first
        if not argv or argv[0] in ['--list', '--categories', '-h', '--help']:
            args = parser.parse_args(argv)

            if args.list:
                return self._list_commands()
            elif args.categories:
                return self._list_by_category()
            else:
                parser.print_help()
                return 0

        # Get command name
        command_name = argv[0]

        # Check if command exists
        if not self.registry.has_command(command_name):
            print(f"Error: Unknown command '{command_name}'")
            print(f"\nRun with --list to see available commands")
            return 1

        # Get executor and run
        executor = get_executor_for_command(command_name)
        if not executor:
            print(f"Error: Could not load executor for '{command_name}'")
            return 1

        # Run executor with remaining arguments
        try:
            return executor.run(argv[1:])
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            return 130
        except Exception as e:
            print(f"\nError executing command: {e}")
            if '--debug' in argv:
                import traceback
                traceback.print_exc()
            return 1

    def _list_commands(self) -> int:
        """List all available commands"""
        commands = self.registry.list_commands()

        print("\nAvailable Commands:")
        print("=" * 60)

        for cmd in commands:
            print(f"  • {cmd}")

        print(f"\nTotal: {len(commands)} commands")
        print("\nRun any command with --help for detailed options")

        return 0

    def _list_by_category(self) -> int:
        """List commands organized by category"""
        categories = self.registry.get_command_categories()

        print("\nCommand Categories:")
        print("=" * 60)

        for category, commands in categories.items():
            print(f"\n{category}:")
            for cmd in commands:
                print(f"  • {cmd}")

        print(f"\n\nTotal: {len(self.registry.list_commands())} commands")

        return 0


def main():
    """Main entry point"""
    cli = CommandCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())