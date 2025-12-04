---
name: Git & PR Patterns
description: Master Git workflows, branching strategies, commit conventions, and pull request best practices for effective version control and team collaboration. Use when writing git commands, creating branches, or managing version control operations. Use when writing commit messages or following conventional commits format (feat, fix, docs, refactor, test, chore). Use when creating pull requests, merge requests, or PR descriptions. Use when resolving merge conflicts, rebasing branches, or cleaning up git history. Use when setting up Git Flow, GitHub Flow, or trunk-based development branching strategies. Use when configuring git hooks, pre-commit hooks, or husky. Use when recovering from git mistakes (reset, revert, reflog, cherry-pick). Use when squashing commits, interactive rebasing, or amending commits. Use when working with .gitignore, .gitattributes, or git configuration files. Use when integrating git with CI/CD pipelines or GitHub Actions workflows. Use when managing release branches, hotfixes, or version tags.
---

# Git & PR Patterns

Comprehensive guide to Git workflow patterns, branching strategies, and pull request best practices for effective team collaboration.

## When to use this skill

- Writing or executing git commands (commit, push, pull, merge, rebase, cherry-pick)
- Creating feature branches, release branches, or hotfix branches
- Writing commit messages following conventional commits format (feat:, fix:, docs:, refactor:, test:, chore:, ci:, perf:)
- Creating pull requests or merge requests with proper descriptions and templates
- Resolving merge conflicts or choosing merge strategies (merge, squash, rebase)
- Setting up branching strategies (Git Flow, GitHub Flow, trunk-based development)
- Performing interactive rebase to clean up commit history
- Recovering from git mistakes using reset, revert, reflog, or cherry-pick
- Configuring git hooks, pre-commit hooks, or husky for automation
- Working with .gitignore, .gitattributes, or git configuration files
- Squashing commits, amending commits, or rewriting history
- Managing git tags for releases and versioning
- Integrating git workflows with GitHub Actions, GitLab CI, or other CI/CD systems
- Setting up PR templates, branch protection rules, or required reviews
- Using GitHub CLI (gh) or GitLab CLI for repository management
- Stashing changes, managing worktrees, or working with submodules

## Core Concepts

### 1. Branching Strategies

- **Git Flow**: Feature, develop, release, hotfix, and main branches
- **GitHub Flow**: Simple feature branch workflow with main
- **Trunk-Based Development**: Short-lived branches, frequent merges
- **Release Branches**: Stable branches for production releases

### 2. Commit Best Practices

- **Atomic Commits**: One logical change per commit
- **Conventional Commits**: Structured commit message format
- **Signed Commits**: GPG-signed for verification
- **Clean History**: Squash, rebase, and organize commits

### 3. Pull Request Workflow

- **Draft PRs**: Work-in-progress visibility
- **Code Review**: Required approvals and checks
- **CI/CD Integration**: Automated testing and validation
- **Merge Strategies**: Merge, squash, or rebase

## Quick Start

```bash
# Create feature branch
git checkout -b feature/user-authentication

# Make changes and commit
git add .
git commit -m "feat(auth): add JWT token validation

- Implement token parsing and validation
- Add expiration checking
- Include refresh token support

Closes #123"

# Push and create PR
git push -u origin feature/user-authentication
gh pr create --title "Add JWT authentication" --body "..."
```

## Branching Patterns

### Pattern 1: GitHub Flow

```bash
# Simple, effective workflow for continuous deployment

# 1. Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/new-feature

# 2. Work and commit
git add .
git commit -m "feat: implement new feature"

# 3. Push and create PR
git push -u origin feature/new-feature
gh pr create

# 4. After review, merge to main
gh pr merge --squash

# 5. Deploy from main (automated)
```

### Pattern 2: Conventional Commits

```bash
# Format: <type>(<scope>): <description>

# Types:
feat:     # New feature
fix:      # Bug fix
docs:     # Documentation only
style:    # Formatting, no code change
refactor: # Code restructuring
perf:     # Performance improvement
test:     # Adding tests
chore:    # Maintenance tasks
ci:       # CI/CD changes

# Examples:
git commit -m "feat(auth): add OAuth2 login support"
git commit -m "fix(api): handle null response from server"
git commit -m "docs(readme): update installation instructions"
```

### Pattern 3: Interactive Rebase

```bash
# Clean up commits before PR
git rebase -i HEAD~5

# Commands in editor:
# pick   - keep commit
# squash - combine with previous
# reword - change message
# drop   - remove commit
# edit   - stop to amend

# Force push after rebase (only on feature branches!)
git push --force-with-lease
```

## PR Best Practices

### Template

```markdown
## Summary
Brief description of changes

## Changes
- Added X functionality
- Fixed Y bug
- Updated Z documentation

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Recovery Patterns

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git checkout -- .

# Recover deleted branch
git reflog
git checkout -b recovered-branch <commit-hash>

# Abort merge conflict
git merge --abort

# Stash changes temporarily
git stash
git stash pop
```

## Best Practices

1. **Small PRs**: Easier to review, faster to merge
2. **Clear Titles**: Descriptive, follows conventions
3. **Link Issues**: Reference related issues/tickets
4. **CI Must Pass**: Never merge failing builds
5. **Review Responses**: Address all feedback before merge
