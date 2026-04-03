---
name: git-workflow
description: Master advanced Git workflows for collaborative development. Covers interactive rebasing, cherry-picking, bisecting for bug discovery, and managing pull requests. Use when resolving merge conflicts, setting up branching strategies, or managing complex Git history.
---

# Git & Workflow Management

## Expert Agent

For Git branching strategies, history management, and collaborative workflows, delegate to:

- **`automation-engineer`**: Enforces branching strategies (trunk-based, GitFlow) and optimizes Git collaboration.
  - *Location*: `plugins/dev-suite/agents/automation-engineer.md`

Expert guide for maintaining a clean, navigable code history and optimizing collaboration.

## 1. History Management

- **Interactive Rebase**: Use `git rebase -i` to clean up commits, squash fixups, and reword messages before merging.
- **Cherry-Pick**: Selectively apply commits across branches for hotfixes or backports.
- **Atomic Commits**: Ensure each commit represents a single logical change with a descriptive message.

## 2. Advanced Discovery & Recovery

- **Git Bisect**: Use binary search to identify the exact commit that introduced a bug.
- **Reflog**: Recover lost commits or reset branches after accidental deletions.
- **Worktrees**: Work on multiple branches simultaneously without needing multiple clones.

## 3. Pull Request Best Practices

- **Code Review**: Focus on logic, architecture, and maintainability. Use automated linting for style.
- **PR Structure**: Include a clear summary, test plan, and link to relevant issues.
- **Force Push**: Use `git push --force-with-lease` when updating rebased PRs to avoid overwriting others' work.

## 4. Git Performance & Scale

- **Parallel Operations**: Use `git fetch --jobs=n` and `git submodule update --jobs=n` for large repositories.
- **LFS**: Use Git Large File Storage for binary assets to keep the repository size manageable.
