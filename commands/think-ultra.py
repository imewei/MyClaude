#!/usr/bin/env python3
"""
Think-Ultra Slash Command Entry Point
Executes multi-agent analytical thinking engine
"""

import sys
from pathlib import Path

# Add executors to path
executors_dir = Path(__file__).parent / "executors"
sys.path.insert(0, str(executors_dir))

from command_dispatcher import CommandDispatcher


def main():
    """Entry point for /think-ultra command"""
    dispatcher = CommandDispatcher()
    return dispatcher.dispatch('think-ultra', sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())