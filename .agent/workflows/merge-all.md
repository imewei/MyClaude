---
description: Merge all local branches into main and clean up
triggers:
- /merge-all
- workflow for merge all
version: 1.0.7
allowed-tools: Bash(git:*), Read
argument-hint: '[--skip-commit] [--no-delete] [--force] [--dry-run]'
color: magenta
agents:
  orchestrated: false
---


# Merge All Branches

Commit changes, merge all local branches into main, delete merged branches.

**Arguments:** $ARGUMENTS

## Flags

| Flag | Effect |
|------|--------|
| `--skip-commit` | Fail if uncommitted changes exist |
| `--no-delete` | Keep branches after merging |
| `--force` | Skip confirmation |
| `--dry-run` | Show plan only |

## Execution

### 1. Assess Repository (single command)

```bash
git rev-parse --abbrev-ref HEAD && git status -s && git for-each-ref --format='%(refname:short)' refs/heads/ | grep -E '^(main|master)$' | head -1 && git for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads/ | grep -v -E '^(main|master)\s'
```

Parse output to get: current branch, dirty status, main branch name, other branches with ahead/behind info.

**Abort conditions:**
- No main/master branch → "Create main branch first: `git checkout -b main`"
- Detached HEAD → "Checkout a branch first"
- `--skip-commit` with dirty state → "Commit first or remove --skip-commit"
- No other branches → "Repository already consolidated"

### 2. Commit Changes (if dirty and no --skip-commit)

```bash
git add -A && git diff --cached --stat
```

Generate conventional commit message from changed files. Commit without AI attribution.

### 3. Show Plan

Display branches to merge with commit counts. For `--dry-run`: stop here.

### 4. Confirm (skip if --force)

Use AskUserQuestion: "Merge N branches into main?" with options: "Proceed", "Abort", "Show details"

### 5. Execute Merges

```bash
git checkout main
```

For each branch:
```bash
git merge <branch> --no-ff -m "Merge branch '<branch>'"
```

On conflict: ask user to resolve/skip/abort. Track results.

### 6. Cleanup (skip if --no-delete)

```bash
git branch -d <branch>
```

Use `-d` to ensure fully merged. Track failures.

### 7. Report

Show: branches merged, branches skipped, branches deleted, next steps (push).

## Rollback

- Undo merge: `git reset --hard HEAD~1`
- Restore branch: `git checkout -b <name> <hash>` (find via `git reflog`)
- Abort merge: `git merge --abort`
