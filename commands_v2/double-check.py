#!/usr/bin/env python3
"""
Double-Check Slash Command Entry Point
Executes the verification engine with 5-phase methodology
"""

import sys
from pathlib import Path

# Add executors to path
executors_dir = Path(__file__).parent / "executors"
sys.path.insert(0, str(executors_dir))

from command_dispatcher import CommandDispatcher


def main():
    """Entry point for /double-check command"""
    dispatcher = CommandDispatcher()
    return dispatcher.dispatch('double-check', sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())