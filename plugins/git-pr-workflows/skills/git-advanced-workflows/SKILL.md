---
name: git-advanced-workflows
version: "1.0.7"
maturity: "5-Expert"
specialization: Git History Management
description: Master advanced Git including interactive rebase, cherry-pick, bisect, worktrees, and reflog for clean history and recovery. Use when cleaning up commits before PRs, applying fixes across branches, finding bug-introducing commits, or recovering from Git mistakes.
---

# Git Advanced Workflows

Clean history, cross-branch operations, and recovery techniques.

---

## Interactive Rebase

| Command | Effect |
|---------|--------|
| `pick` | Keep commit as-is |
| `reword` | Change message |
| `edit` | Amend content |
| `squash` | Combine with previous |
| `fixup` | Combine, discard message |
| `drop` | Remove commit |

```bash
# Rebase last 5 commits
git rebase -i HEAD~5

# Rebase all commits on branch
git rebase -i $(git merge-base HEAD main)
```

---

## Cherry-Pick

```bash
# Single commit
git cherry-pick abc123

# Range (exclusive start)
git cherry-pick abc123..def456

# Without committing (stage only)
git cherry-pick -n abc123

# Specific files from commit
git checkout abc123 -- path/to/file
git commit -m "cherry-pick: specific changes"
```

---

## Git Bisect

Binary search to find bug-introducing commit:

```bash
# Manual bisect
git bisect start
git bisect bad HEAD
git bisect good v1.0.0
# Test, then:
git bisect good  # or: git bisect bad
# Repeat until found
git bisect reset

# Automated bisect
git bisect start HEAD v1.0.0
git bisect run npm test
```

---

## Worktrees

Work on multiple branches simultaneously:

```bash
# List worktrees
git worktree list

# Add worktree for existing branch
git worktree add ../project-hotfix hotfix/critical

# Add worktree with new branch
git worktree add -b bugfix/urgent ../project-fix main

# Remove worktree
git worktree remove ../project-hotfix
```

---

## Reflog (Recovery)

```bash
# View reflog
git reflog

# Recover deleted commit
git checkout abc123
git branch recovered-branch

# Recover after bad reset
git reflog
# Find: abc123 HEAD@{1}: commit: my changes
git reset --hard abc123

# Recover deleted branch
git branch deleted-branch abc123
```

---

## Common Workflows

### Clean Up Before PR

```bash
git checkout feature/my-feature
git rebase -i main
# Squash fixups, reword messages
git push --force-with-lease
```

### Apply Hotfix to Multiple Releases

```bash
git checkout main
git commit -m "fix: critical security patch"

git checkout release/2.0 && git cherry-pick abc123
git checkout release/1.9 && git cherry-pick abc123
```

### Multi-Branch Development

```bash
# Create worktree for hotfix
git worktree add ../myapp-hotfix hotfix/critical

# Work in separate directory
cd ../myapp-hotfix
git commit -m "fix: resolve bug"
git push origin hotfix/critical

# Return to main work
cd ~/projects/myapp
git worktree remove ../myapp-hotfix
```

---

## Autosquash Workflow

```bash
# Initial commit
git commit -m "feat: add feature"

# Later, fix something
git commit --fixup HEAD

# Rebase with autosquash
git rebase -i --autosquash main
```

---

## Split Commit

```bash
git rebase -i HEAD~3
# Mark commit with 'edit'

# Reset but keep changes
git reset HEAD^

# Commit in parts
git add file1.py && git commit -m "feat: add validation"
git add file2.py && git commit -m "feat: add error handling"

git rebase --continue
```

---

## Rebase vs Merge

| Situation | Use |
|-----------|-----|
| Local cleanup | Rebase |
| Keeping branch current | Rebase |
| Public/shared branches | Merge |
| Preserving collaboration history | Merge |

```bash
# Update feature branch (rebase)
git fetch origin
git rebase origin/main

# Resolve conflicts
git add .
git rebase --continue
```

---

## Safety Commands

```bash
# Safe force push
git push --force-with-lease

# Create backup before risky operation
git branch backup-branch
git rebase -i main
# If failed:
git reset --hard backup-branch

# Abort operations
git rebase --abort
git cherry-pick --abort
git merge --abort
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| --force-with-lease | Safer than --force |
| Rebase only local | Don't rebase pushed commits |
| Atomic commits | One logical change per commit |
| Backup before rebase | Create safety branch |
| Test after rewrite | Ensure no breakage |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Rebasing public branches | Causes team conflicts |
| --force without lease | Can overwrite others' work |
| Lost work in rebase | Use reflog to recover |
| Orphaned worktrees | Clean with git worktree prune |
| Bisect on dirty tree | Commit or stash first |

---

## Recovery Quick Reference

```bash
# Undo last commit (keep changes)
git reset --soft HEAD^

# Undo last commit (discard changes)
git reset --hard HEAD^

# Restore file from commit
git restore --source=abc123 path/to/file

# Find lost commits
git reflog
git branch recovered abc123
```

---

## Checklist

- [ ] Backup branch created before rebase
- [ ] Commits atomic and well-described
- [ ] Tests pass after history rewrite
- [ ] Using --force-with-lease not --force
- [ ] Reflog checked for recovery options

---

**Version**: 1.0.5
