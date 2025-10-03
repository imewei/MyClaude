#!/usr/bin/env python3
"""
Fix Commit Errors Slash Command Entry Point
Executes GitHub Actions workflow error analysis and fixing
"""

import sys
from pathlib import Path

# Add executors to path
executors_dir = Path(__file__).parent / "executors"
sys.path.insert(0, str(executors_dir))

from command_dispatcher import CommandDispatcher


def main():
    """Entry point for /fix-commit-errors command"""
    dispatcher = CommandDispatcher()
    return dispatcher.dispatch('fix-commit-errors', sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())