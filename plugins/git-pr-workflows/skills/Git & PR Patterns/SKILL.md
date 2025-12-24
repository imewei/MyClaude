---
name: Git & PR Patterns
version: "1.0.5"
maturity: "5-Expert"
specialization: Version Control
description: Master Git workflows, branching strategies, and PR best practices. Use when writing git commands, creating branches, committing, resolving conflicts, or managing pull requests.
---

# Git & PR Patterns

Git workflow patterns, branching strategies, and PR best practices.

---

## Branching Strategies

| Strategy | Use Case |
|----------|----------|
| GitHub Flow | Simple, continuous deployment |
| Git Flow | Scheduled releases, multiple versions |
| Trunk-Based | Short-lived branches, frequent merges |

---

## Conventional Commits

| Type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| docs | Documentation |
| refactor | Code restructuring |
| perf | Performance |
| test | Adding tests |
| chore | Maintenance |
| ci | CI/CD changes |

```bash
git commit -m "feat(auth): add OAuth2 login support"
```

---

## GitHub Flow Pattern

```bash
# Create feature branch
git checkout main && git pull origin main
git checkout -b feature/new-feature

# Commit and push
git add . && git commit -m "feat: implement feature"
git push -u origin feature/new-feature

# Create and merge PR
gh pr create --title "Add feature"
gh pr merge --squash
```

---

## Interactive Rebase

```bash
git rebase -i HEAD~5
# pick, squash, reword, drop, edit
git push --force-with-lease  # Feature branches only!
```

---

## Recovery Commands

| Situation | Command |
|-----------|---------|
| Undo last commit | `git reset --soft HEAD~1` |
| Discard changes | `git checkout -- .` |
| Recover branch | `git reflog` + checkout |
| Abort merge | `git merge --abort` |
| Stash changes | `git stash` / `git stash pop` |

---

## PR Template

```markdown
## Summary
Brief description

## Changes
- Added X, Fixed Y, Updated Z

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass

## Checklist
- [ ] Self-review completed
- [ ] Documentation updated
```

---

## Best Practices

| Practice | Guideline |
|----------|-----------|
| PR size | Small, focused changes |
| Titles | Descriptive, follows conventions |
| Issues | Link related tickets |
| CI | Never merge failing builds |
| Feedback | Address all before merge |

---

**Version**: 1.0.5
