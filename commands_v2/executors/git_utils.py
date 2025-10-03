#!/usr/bin/env python3
"""
Git Utilities for Command Executors
Provides common git operations with error handling
"""

import subprocess
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class GitError(Exception):
    """Custom exception for git operations"""
    pass


class GitUtils:
    """Utility class for git operations"""

    def __init__(self, work_dir: Path = None):
        self.work_dir = work_dir or Path.cwd()

    def run_git_command(self, args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run a git command and return the result

        Args:
            args: Git command arguments (e.g., ['status', '--short'])
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess with result

        Raises:
            GitError: If git command fails
        """
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=self.work_dir,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git command failed: {' '.join(args)}\n{e.stderr}")
        except FileNotFoundError:
            raise GitError("Git is not installed or not in PATH")

    def get_status(self) -> Dict[str, List[str]]:
        """
        Get git status categorized by file state

        Returns:
            Dict with keys: modified, added, deleted, untracked
        """
        result = self.run_git_command(['status', '--porcelain'])

        status = {
            'modified': [],
            'added': [],
            'deleted': [],
            'untracked': [],
            'renamed': []
        }

        for line in result.stdout.splitlines():
            if not line:
                continue

            state = line[:2]
            filepath = line[3:]

            if state == '??':
                status['untracked'].append(filepath)
            elif 'M' in state:
                status['modified'].append(filepath)
            elif 'A' in state:
                status['added'].append(filepath)
            elif 'D' in state:
                status['deleted'].append(filepath)
            elif 'R' in state:
                status['renamed'].append(filepath)

        return status

    def get_diff(self, staged: bool = False, file_path: Optional[str] = None) -> str:
        """
        Get git diff

        Args:
            staged: Get staged changes only
            file_path: Specific file to diff

        Returns:
            Diff output as string
        """
        args = ['diff']
        if staged:
            args.append('--staged')
        if file_path:
            args.append(file_path)

        result = self.run_git_command(args)
        return result.stdout

    def add_files(self, files: List[str]) -> None:
        """
        Add files to staging area

        Args:
            files: List of file paths to add
        """
        if not files:
            return

        self.run_git_command(['add'] + files)

    def add_all(self) -> None:
        """Add all changes to staging area"""
        self.run_git_command(['add', '-A'])

    def commit(self, message: str, allow_empty: bool = False) -> str:
        """
        Create a git commit

        Args:
            message: Commit message
            allow_empty: Allow empty commits

        Returns:
            Commit hash
        """
        args = ['commit', '-m', message]
        if allow_empty:
            args.append('--allow-empty')

        result = self.run_git_command(args)

        # Get the commit hash
        hash_result = self.run_git_command(['rev-parse', 'HEAD'])
        return hash_result.stdout.strip()

    def push(self, remote: str = 'origin', branch: Optional[str] = None,
             set_upstream: bool = False) -> None:
        """
        Push commits to remote

        Args:
            remote: Remote name
            branch: Branch name (None for current)
            set_upstream: Set upstream tracking
        """
        args = ['push']
        if set_upstream:
            args.append('-u')
        args.append(remote)
        if branch:
            args.append(branch)

        self.run_git_command(args)

    def get_current_branch(self) -> str:
        """Get current branch name"""
        result = self.run_git_command(['branch', '--show-current'])
        return result.stdout.strip()

    def create_branch(self, branch_name: str, checkout: bool = True) -> None:
        """
        Create a new branch

        Args:
            branch_name: Name of new branch
            checkout: Checkout the branch after creating
        """
        if checkout:
            self.run_git_command(['checkout', '-b', branch_name])
        else:
            self.run_git_command(['branch', branch_name])

    def checkout(self, branch_or_commit: str) -> None:
        """Checkout a branch or commit"""
        self.run_git_command(['checkout', branch_or_commit])

    def get_log(self, max_count: int = 10, format: str = '%H %s') -> List[Dict[str, str]]:
        """
        Get git log

        Args:
            max_count: Maximum number of commits
            format: Git log format string

        Returns:
            List of commit info dicts
        """
        result = self.run_git_command([
            'log',
            f'--max-count={max_count}',
            f'--pretty=format:{format}'
        ])

        commits = []
        for line in result.stdout.splitlines():
            if not line:
                continue
            parts = line.split(' ', 1)
            commits.append({
                'hash': parts[0],
                'message': parts[1] if len(parts) > 1 else ''
            })

        return commits

    def is_repo(self) -> bool:
        """Check if current directory is a git repository"""
        try:
            self.run_git_command(['rev-parse', '--git-dir'])
            return True
        except GitError:
            return False

    def get_remote_url(self, remote: str = 'origin') -> Optional[str]:
        """Get remote URL"""
        try:
            result = self.run_git_command(['remote', 'get-url', remote])
            return result.stdout.strip()
        except GitError:
            return None

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes"""
        status = self.get_status()
        return any(status.values())

    def get_staged_files(self) -> List[str]:
        """Get list of staged files"""
        result = self.run_git_command(['diff', '--cached', '--name-only'])
        return [f for f in result.stdout.splitlines() if f]

    def reset_file(self, file_path: str, hard: bool = False) -> None:
        """
        Reset a file to HEAD

        Args:
            file_path: Path to file
            hard: Use hard reset (discard changes)
        """
        if hard:
            self.run_git_command(['checkout', 'HEAD', '--', file_path])
        else:
            self.run_git_command(['reset', 'HEAD', file_path])

    def stash(self, message: Optional[str] = None) -> None:
        """Stash current changes"""
        args = ['stash', 'push']
        if message:
            args.extend(['-m', message])
        self.run_git_command(args)

    def stash_pop(self) -> None:
        """Pop latest stash"""
        self.run_git_command(['stash', 'pop'])