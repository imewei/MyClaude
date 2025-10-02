#!/usr/bin/env python3
"""
Fix GitHub Issue Command Executor
Symlink to parent directory executor for backward compatibility
"""

import sys
from pathlib import Path

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from fix_github_issue_executor import FixGitHubIssueExecutor, main

__all__ = ['FixGitHubIssueExecutor', 'main']

if __name__ == "__main__":
    sys.exit(main())