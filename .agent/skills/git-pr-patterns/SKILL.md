---
name: "git-pr-patterns & PR Patterns"
version: "1.0.7"
maturity: "5-Expert"
specialization: Version Control
description: Master Git workflows, branching strategies, and PR best practices. Use when writing git commands, creating branches, committing, resolving conflicts, or managing pull requests.
---

# Git & PR Patterns

Git workflow patterns, branching strategies, and PR best practices.

---

<!-- SECTION: BRANCHING -->
## Branching Strategies

| Strategy | Use Case |
|----------|----------|
| GitHub Flow | Simple, continuous deployment |
| Git Flow | Scheduled releases, multiple versions |
| Trunk-Based | Short-lived branches, frequent merges |
<!-- END_SECTION: BRANCHING -->

---

<!-- SECTION: CONVENTIONAL_COMMITS -->
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
<!-- END_SECTION: CONVENTIONAL_COMMITS -->

---

<!-- SECTION: GITHUB_FLOW -->
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
<!-- END_SECTION: GITHUB_FLOW -->

---

<!-- SECTION: REBASE -->
## Interactive Rebase

```bash
git rebase -i HEAD~5
# pick, squash, reword, drop, edit
git push --force-with-lease  # Feature branches only!
```
<!-- END_SECTION: REBASE -->

---

<!-- SECTION: RECOVERY -->
## Recovery Commands

| Situation | Command |
|-----------|---------|
| Undo last commit | `git reset --soft HEAD~1` |
| Discard changes | `git checkout -- .` |
| Recover branch | `git reflog` + checkout |
| Abort merge | `git merge --abort` |
| Stash changes | `git stash` / `git stash pop` |
<!-- END_SECTION: RECOVERY -->

---

<!-- SECTION: PR_TEMPLATE -->
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
<!-- END_SECTION: PR_TEMPLATE -->

---

<!-- SECTION: BEST_PRACTICES -->
## Best Practices

| Practice | Guideline |
|----------|-----------|
| PR size | Small, focused changes |
| Titles | Descriptive, follows conventions |
| Issues | Link related tickets |
| CI | Never merge failing builds |
| Feedback | Address all before merge |
<!-- END_SECTION: BEST_PRACTICES -->

---

**Version**: 1.0.5
