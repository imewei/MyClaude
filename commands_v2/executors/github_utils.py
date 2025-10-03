#!/usr/bin/env python3
"""
GitHub Utilities for Command Executors
Provides GitHub API operations via gh CLI
"""

import subprocess
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class GitHubError(Exception):
    """Custom exception for GitHub operations"""
    pass


class GitHubUtils:
    """Utility class for GitHub operations using gh CLI"""

    def __init__(self, work_dir: Path = None):
        self.work_dir = work_dir or Path.cwd()

    def run_gh_command(self, args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run a gh CLI command

        Args:
            args: GitHub CLI command arguments
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess with result

        Raises:
            GitHubError: If gh command fails
        """
        try:
            result = subprocess.run(
                ['gh'] + args,
                cwd=self.work_dir,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitHubError(f"GitHub CLI command failed: {' '.join(args)}\n{e.stderr}")
        except FileNotFoundError:
            raise GitHubError("GitHub CLI (gh) is not installed or not in PATH")

    def is_authenticated(self) -> bool:
        """Check if gh CLI is authenticated"""
        try:
            self.run_gh_command(['auth', 'status'])
            return True
        except GitHubError:
            return False

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Get issue details

        Args:
            issue_number: Issue number

        Returns:
            Issue data as dict
        """
        result = self.run_gh_command([
            'issue', 'view', str(issue_number),
            '--json', 'number,title,body,state,labels,assignees,url'
        ])
        return json.loads(result.stdout)

    def create_issue(self, title: str, body: str, labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new issue

        Args:
            title: Issue title
            body: Issue body
            labels: Optional list of labels

        Returns:
            Created issue data
        """
        args = ['issue', 'create', '--title', title, '--body', body]
        if labels:
            args.extend(['--label', ','.join(labels)])

        result = self.run_gh_command(args + ['--json', 'number,url'])
        return json.loads(result.stdout)

    def close_issue(self, issue_number: int, comment: Optional[str] = None) -> None:
        """
        Close an issue

        Args:
            issue_number: Issue number
            comment: Optional closing comment
        """
        args = ['issue', 'close', str(issue_number)]
        if comment:
            args.extend(['--comment', comment])
        self.run_gh_command(args)

    def get_pr(self, pr_number: int) -> Dict[str, Any]:
        """
        Get pull request details

        Args:
            pr_number: PR number

        Returns:
            PR data as dict
        """
        result = self.run_gh_command([
            'pr', 'view', str(pr_number),
            '--json', 'number,title,body,state,url,headRefName,baseRefName,commits,reviews'
        ])
        return json.loads(result.stdout)

    def create_pr(self, title: str, body: str, base: str = 'main',
                  head: Optional[str] = None, draft: bool = False) -> Dict[str, Any]:
        """
        Create a pull request

        Args:
            title: PR title
            body: PR body
            base: Base branch
            head: Head branch (None for current)
            draft: Create as draft PR

        Returns:
            Created PR data
        """
        args = ['pr', 'create', '--title', title, '--body', body, '--base', base]
        if head:
            args.extend(['--head', head])
        if draft:
            args.append('--draft')

        result = self.run_gh_command(args + ['--json', 'number,url'])
        return json.loads(result.stdout)

    def get_workflow_runs(self, workflow: Optional[str] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get workflow runs

        Args:
            workflow: Specific workflow name
            limit: Maximum number of runs

        Returns:
            List of workflow run data
        """
        args = ['run', 'list', '--limit', str(limit), '--json',
                'databaseId,name,status,conclusion,workflowName,headBranch,event,createdAt']
        if workflow:
            args.extend(['--workflow', workflow])

        result = self.run_gh_command(args)
        return json.loads(result.stdout)

    def get_run_logs(self, run_id: int) -> str:
        """
        Get logs for a workflow run

        Args:
            run_id: Workflow run ID

        Returns:
            Log output as string
        """
        result = self.run_gh_command(['run', 'view', str(run_id), '--log'])
        return result.stdout

    def rerun_workflow(self, run_id: int, failed_only: bool = False) -> None:
        """
        Rerun a workflow

        Args:
            run_id: Workflow run ID
            failed_only: Only rerun failed jobs
        """
        args = ['run', 'rerun', str(run_id)]
        if failed_only:
            args.append('--failed')
        self.run_gh_command(args)

    def get_latest_release(self) -> Dict[str, Any]:
        """Get latest release info"""
        result = self.run_gh_command([
            'release', 'view', '--json', 'tagName,name,body,url,createdAt'
        ])
        return json.loads(result.stdout)

    def create_release(self, tag: str, title: str, notes: str,
                       draft: bool = False, prerelease: bool = False) -> Dict[str, Any]:
        """
        Create a release

        Args:
            tag: Release tag
            title: Release title
            notes: Release notes
            draft: Create as draft
            prerelease: Mark as prerelease

        Returns:
            Created release data
        """
        args = ['release', 'create', tag, '--title', title, '--notes', notes]
        if draft:
            args.append('--draft')
        if prerelease:
            args.append('--prerelease')

        result = self.run_gh_command(args + ['--json', 'url,tagName'])
        return json.loads(result.stdout)

    def add_pr_comment(self, pr_number: int, comment: str) -> None:
        """Add comment to PR"""
        self.run_gh_command(['pr', 'comment', str(pr_number), '--body', comment])

    def add_issue_comment(self, issue_number: int, comment: str) -> None:
        """Add comment to issue"""
        self.run_gh_command(['issue', 'comment', str(issue_number), '--body', comment])

    def get_repo_info(self) -> Dict[str, Any]:
        """Get repository information"""
        result = self.run_gh_command([
            'repo', 'view', '--json',
            'name,owner,description,url,isPrivate,defaultBranch'
        ])
        return json.loads(result.stdout)

    def list_issues(self, state: str = 'open', limit: int = 30,
                    labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List issues

        Args:
            state: Issue state (open, closed, all)
            limit: Maximum number of issues
            labels: Filter by labels

        Returns:
            List of issue data
        """
        args = ['issue', 'list', '--state', state, '--limit', str(limit),
                '--json', 'number,title,state,labels,url']
        if labels:
            args.extend(['--label', ','.join(labels)])

        result = self.run_gh_command(args)
        return json.loads(result.stdout)

    def list_prs(self, state: str = 'open', limit: int = 30) -> List[Dict[str, Any]]:
        """
        List pull requests

        Args:
            state: PR state (open, closed, merged, all)
            limit: Maximum number of PRs

        Returns:
            List of PR data
        """
        result = self.run_gh_command([
            'pr', 'list', '--state', state, '--limit', str(limit),
            '--json', 'number,title,state,url,headRefName'
        ])
        return json.loads(result.stdout)

    def approve_pr(self, pr_number: int, comment: Optional[str] = None) -> None:
        """Approve a pull request"""
        args = ['pr', 'review', str(pr_number), '--approve']
        if comment:
            args.extend(['--body', comment])
        self.run_gh_command(args)

    def merge_pr(self, pr_number: int, method: str = 'merge',
                 delete_branch: bool = True) -> None:
        """
        Merge a pull request

        Args:
            pr_number: PR number
            method: Merge method (merge, squash, rebase)
            delete_branch: Delete branch after merge
        """
        args = ['pr', 'merge', str(pr_number), f'--{method}']
        if delete_branch:
            args.append('--delete-branch')
        self.run_gh_command(args)