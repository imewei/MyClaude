#!/usr/bin/env python3
"""
Commit Command Executor
Symlink to parent directory executor for backward compatibility
"""

import sys
from pathlib import Path

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from commit_executor import CommitExecutor, main

__all__ = ['CommitExecutor', 'main']

if __name__ == "__main__":
    sys.exit(main())