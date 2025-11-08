# Git Branching Strategies

> **Reference**: Comprehensive guide to Git branching workflows, merge strategies, and release management patterns

## Strategy 1: Trunk-Based Development

### Overview
Trunk-based development emphasizes short-lived feature branches (1-2 days) merged frequently into a single main branch. This strategy maximizes integration speed and minimizes merge conflicts.

### Branch Structure
```
main (production-ready at all times)
  ├── feature/user-auth (1-2 days)
  ├── feature/payment (1-2 days)
  └── hotfix/critical-bug (hours)
```

### Workflow
1. **Create short-lived branch** from main
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/user-authentication
   ```

2. **Commit frequently** with atomic changes
   ```bash
   git add src/auth/
   git commit -m "feat(auth): add JWT token validation"
   ```

3. **Rebase daily** to stay current with main
   ```bash
   git fetch origin
   git rebase origin/main
   ```

4. **Merge via PR** with CI passing
   ```bash
   # After PR approval
   git checkout main
   git merge --ff-only feature/user-authentication
   git push origin main
   ```

### Feature Flags
Enable trunk-based development for incomplete features:

```javascript
// Feature flag pattern
if (featureFlags.isEnabled('new-checkout-flow')) {
  return <NewCheckoutFlow />;
} else {
  return <LegacyCheckoutFlow />;
}
```

### Best Practices
- **Maximum branch age**: 2 days
- **Minimum merge frequency**: Daily
- **CI/CD requirement**: Automated tests on every commit
- **Feature flags**: Use for incomplete features
- **Code review**: Fast-track reviews (<2 hours)

### When to Use
- High-velocity teams (10+ deploys/day)
- Mature CI/CD pipelines with comprehensive testing
- Strong engineering discipline
- Cloud-native applications with easy rollbacks

---

## Strategy 2: Git Flow

### Overview
Git Flow uses long-lived branches for development and releases, with structured merge patterns. Ideal for scheduled releases and version management.

### Branch Structure
```
main (production releases only)
  ├── develop (integration branch)
  │     ├── feature/user-dashboard
  │     ├── feature/api-v2
  │     └── feature/analytics
  ├── release/v2.0.0 (release preparation)
  └── hotfix/security-patch (emergency fixes)
```

### Branch Types

#### 1. Main Branch
- **Purpose**: Production-ready code
- **Lifetime**: Permanent
- **Merge from**: release/*, hotfix/*
- **Merge to**: Never (only receives merges)

#### 2. Develop Branch
- **Purpose**: Integration branch for features
- **Lifetime**: Permanent
- **Merge from**: feature/*, hotfix/*
- **Merge to**: release/*

#### 3. Feature Branches
- **Naming**: feature/short-description
- **Branch from**: develop
- **Merge to**: develop
- **Lifetime**: 1-4 weeks

```bash
# Create feature branch
git checkout develop
git checkout -b feature/user-authentication

# Complete feature
git checkout develop
git merge --no-ff feature/user-authentication
git branch -d feature/user-authentication
git push origin develop
```

#### 4. Release Branches
- **Naming**: release/v1.2.0
- **Branch from**: develop
- **Merge to**: main AND develop
- **Lifetime**: 1-2 weeks

```bash
# Start release
git checkout develop
git checkout -b release/v1.2.0

# Finalize release
git checkout main
git merge --no-ff release/v1.2.0
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge --no-ff release/v1.2.0
git branch -d release/v1.2.0
```

#### 5. Hotfix Branches
- **Naming**: hotfix/critical-bug
- **Branch from**: main
- **Merge to**: main AND develop
- **Lifetime**: Hours

```bash
# Create hotfix
git checkout main
git checkout -b hotfix/security-vulnerability

# Deploy hotfix
git checkout main
git merge --no-ff hotfix/security-vulnerability
git tag -a v1.2.1 -m "Hotfix: security vulnerability"
git push origin main --tags

# Merge to develop
git checkout develop
git merge --no-ff hotfix/security-vulnerability
git branch -d hotfix/security-vulnerability
```

### Automated Git Flow
```bash
# Using git-flow extension
git flow init
git flow feature start user-authentication
git flow feature finish user-authentication

git flow release start 1.2.0
git flow release finish 1.2.0

git flow hotfix start security-patch
git flow hotfix finish security-patch
```

### Best Practices
- **Feature branches**: Keep under 4 weeks, rebase from develop weekly
- **Release branches**: Only bug fixes, no new features
- **Hotfix protocol**: Document in runbook, automate deployment
- **Version tags**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Merge commits**: Use --no-ff to preserve history

### When to Use
- Scheduled releases (monthly, quarterly)
- Multiple versions in production simultaneously
- Desktop/mobile apps with app store review cycles
- Enterprise software with rigorous release processes

---

## Strategy 3: GitHub Flow

### Overview
Simplified workflow with one main branch and feature branches. Everything in main is deployable, features merged via pull requests.

### Branch Structure
```
main (always deployable)
  ├── feature/user-profile
  ├── feature/search-improvements
  └── fix/login-bug
```

### Workflow

#### 1. Create Feature Branch
```bash
git checkout main
git pull origin main
git checkout -b feature/user-profile
```

#### 2. Commit and Push Regularly
```bash
git add src/profile/
git commit -m "feat(profile): add avatar upload"
git push origin feature/user-profile
```

#### 3. Open Pull Request Early
- Create PR when code is ready for feedback (can be draft)
- Describe changes, link issues, tag reviewers
- Update PR with additional commits as needed

#### 4. Deploy from Branch (Optional)
```bash
# Deploy feature branch to staging
git push origin feature/user-profile
# Trigger deployment via CI/CD
```

#### 5. Merge After Review
```bash
# After approval and CI passing
# Merge via GitHub UI (creates merge commit)
# OR via command line:
git checkout main
git merge --no-ff feature/user-profile
git push origin main
```

#### 6. Delete Feature Branch
```bash
git branch -d feature/user-profile
git push origin --delete feature/user-profile
```

### Pull Request Template
```markdown
## Changes
- Added user profile page
- Implemented avatar upload with S3 integration
- Added profile edit functionality

## Testing
- [ ] Unit tests pass (95% coverage)
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Deployed to staging and validated

## Screenshots
[Add screenshots for UI changes]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Best Practices
- **Branch naming**: descriptive, lowercase, hyphens (feature/add-user-search)
- **Commit frequently**: Small, atomic commits
- **PR size**: Target 200-400 lines changed
- **Review turnaround**: <24 hours
- **Deploy after merge**: Continuous deployment to production

### When to Use
- Web applications with continuous deployment
- SaaS products
- Open source projects
- Small to medium teams (5-20 developers)

---

## Strategy 4: GitLab Flow

### Overview
Extends GitHub Flow with environment branches for staging/production deployments. Combines simplicity with deployment control.

### Branch Structure
```
main (development)
  ├── staging (pre-production testing)
  │     ├── feature/analytics
  │     └── feature/reporting
  └── production (live environment)
        └── hotfix/critical-fix
```

### Workflow

#### Development → Staging → Production
```bash
# Feature development
git checkout main
git checkout -b feature/analytics

# Merge to main (development environment)
git checkout main
git merge feature/analytics
git push origin main
# Auto-deploys to development

# Promote to staging
git checkout staging
git merge main
git push origin staging
# Auto-deploys to staging

# Promote to production after testing
git checkout production
git merge staging
git push origin production
# Auto-deploys to production
```

#### Environment-Specific Fixes
```bash
# Fix discovered in staging
git checkout staging
git checkout -b fix/staging-bug
# Fix and test
git checkout staging
git merge fix/staging-bug

# If fix needed in production before next release
git checkout production
git cherry-pick <commit-hash>
git push origin production
```

### CI/CD Integration
```yaml
# .gitlab-ci.yml
stages:
  - test
  - deploy-dev
  - deploy-staging
  - deploy-production

test:
  stage: test
  script:
    - npm test
  only:
    - merge_requests

deploy-dev:
  stage: deploy-dev
  script:
    - deploy-to-dev.sh
  only:
    - main

deploy-staging:
  stage: deploy-staging
  script:
    - deploy-to-staging.sh
  only:
    - staging
  when: manual

deploy-production:
  stage: deploy-production
  script:
    - deploy-to-production.sh
  only:
    - production
  when: manual
```

### Best Practices
- **Environment branches**: Protected, require approvals
- **Deployment gates**: Manual approval for production
- **Rollback strategy**: Revert merge commits on production branch
- **Hotfix process**: Cherry-pick from staging to production if urgent

### When to Use
- Multiple environments (dev, staging, production)
- Controlled deployment gates
- Regulated industries requiring approval workflows
- Medium to large teams with dedicated QA

---

## Merge Strategies

### 1. Merge Commit (--no-ff)
**Creates explicit merge commit preserving branch history**

```bash
git merge --no-ff feature/user-auth
```

**Pros**:
- Preserves complete history
- Groups related commits
- Easy to revert entire feature

**Cons**:
- Cluttered history with many merge commits
- More complex graph visualization

**Use when**: Git Flow, important feature milestones

### 2. Squash and Merge
**Combines all commits into single commit**

```bash
git merge --squash feature/user-auth
git commit -m "feat(auth): implement user authentication"
```

**Pros**:
- Clean linear history
- Removes WIP commits
- Easier to understand history

**Cons**:
- Loses individual commit details
- Makes bisecting harder

**Use when**: GitHub Flow, feature branches with messy history

### 3. Rebase and Merge
**Replays commits on top of base branch**

```bash
git checkout feature/user-auth
git rebase main
git checkout main
git merge --ff-only feature/user-auth
```

**Pros**:
- Clean linear history
- Preserves individual commits
- No merge commits

**Cons**:
- Rewrites history (conflicts if already pushed)
- Can cause confusion for collaborators

**Use when**: Trunk-based, clean commit history desired

### 4. Fast-Forward Merge
**Moves pointer forward without merge commit**

```bash
git merge --ff-only feature/user-auth
```

**Pros**:
- Simplest merge
- Clean linear history

**Cons**:
- Only works when no divergent commits
- Requires rebasing feature branch first

**Use when**: Trunk-based development, short-lived branches

---

## Branch Protection Rules

### Main/Production Branch Protection
```yaml
# GitHub branch protection settings
required_status_checks:
  strict: true
  contexts:
    - continuous-integration/travis-ci
    - code-review/reviewable

enforce_admins: true

required_pull_request_reviews:
  required_approving_review_count: 2
  dismiss_stale_reviews: true
  require_code_owner_reviews: true

restrictions:
  users: []
  teams: [core-team]

required_linear_history: false
allow_force_pushes: false
allow_deletions: false
```

### Develop Branch Protection
```yaml
required_status_checks:
  strict: true
  contexts:
    - ci/tests

required_pull_request_reviews:
  required_approving_review_count: 1

allow_force_pushes: false
```

---

## Release Management

### Semantic Versioning
```
MAJOR.MINOR.PATCH (e.g., 2.1.3)

MAJOR: Breaking changes (2.0.0)
MINOR: New features, backward compatible (2.1.0)
PATCH: Bug fixes, backward compatible (2.1.3)
```

### Version Bumping
```bash
# Using npm version
npm version patch  # 1.0.0 → 1.0.1
npm version minor  # 1.0.1 → 1.1.0
npm version major  # 1.1.0 → 2.0.0

# Manual
echo "2.1.0" > VERSION
git add VERSION
git commit -m "chore: bump version to 2.1.0"
git tag -a v2.1.0 -m "Release v2.1.0"
git push origin v2.1.0
```

### Release Checklist
- [ ] Update CHANGELOG.md
- [ ] Bump version in package.json/VERSION
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create git tag
- [ ] Build release artifacts
- [ ] Deploy to production
- [ ] Monitor error rates
- [ ] Announce release

---

## Conflict Resolution

### Merge Conflict Workflow
```bash
# Attempt merge
git merge feature/user-auth
# CONFLICT (content): Merge conflict in src/auth.js

# View conflicts
git status
# both modified: src/auth.js

# Resolve conflicts in editor
# Look for:
<<<<<<< HEAD
current code
=======
incoming code
>>>>>>> feature/user-auth

# After resolving
git add src/auth.js
git commit -m "merge: resolve conflicts in auth.js"
```

### Rebase Conflict Workflow
```bash
# Attempt rebase
git rebase main
# CONFLICT (content): Merge conflict in src/auth.js

# Resolve conflicts
# Edit file, remove markers

# Continue rebase
git add src/auth.js
git rebase --continue

# If stuck, abort and try merge instead
git rebase --abort
git merge main
```

### Prevention Strategies
1. **Merge frequently**: Pull from main daily
2. **Small PRs**: Reduce surface area for conflicts
3. **Coordinate**: Communicate when working on same files
4. **Feature flags**: Allow parallel work without conflicts
5. **Modular architecture**: Reduce file coupling

---

## Decision Matrix

| Criteria | Trunk-Based | Git Flow | GitHub Flow | GitLab Flow |
|----------|------------|----------|-------------|-------------|
| **Team Size** | 5-50 | 10-100 | 5-30 | 10-50 |
| **Deploy Frequency** | 10+/day | 1/month | 1-5/day | 2-10/day |
| **CI/CD Maturity** | High | Medium | High | High |
| **Release Cadence** | Continuous | Scheduled | Continuous | Controlled |
| **Complexity** | Low | High | Low | Medium |
| **Learning Curve** | Low | High | Low | Medium |
| **Merge Conflicts** | Low | Medium | Low | Low |
| **History Clarity** | High | Medium | High | High |
| **Rollback Ease** | Easy | Medium | Easy | Easy |
| **Best For** | SaaS, APIs | Enterprise | Startups | Multi-env |
