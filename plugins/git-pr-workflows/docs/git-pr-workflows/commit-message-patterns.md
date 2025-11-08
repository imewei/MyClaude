# Commit Message Patterns

> **Reference**: Conventional commit formats, best practices, and anti-patterns for maintainable Git history

## Conventional Commits Specification

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type
**Required**. Describes the category of change:

- **feat**: New feature for the user
- **fix**: Bug fix for the user
- **docs**: Documentation changes
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvement
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI configuration
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Scope
**Optional**. Indicates the area of codebase affected:

```
feat(auth): add OAuth2 support
fix(api): correct response status codes
docs(readme): update installation instructions
```

Common scopes:
- **Component/module names**: `auth`, `api`, `db`, `ui`
- **File/directory names**: `user-service`, `payment`
- **Feature areas**: `checkout`, `search`, `profile`

### Subject
**Required**. Brief description in imperative mood:

✅ **Good**:
```
add user authentication
fix memory leak in cache
update dependencies to latest versions
```

❌ **Bad**:
```
added user authentication    (past tense)
fixing memory leak           (present continuous)
Updated dependencies         (capitalized)
```

**Rules**:
- Use imperative mood ("add" not "adds" or "added")
- Don't capitalize first letter
- No period at the end
- Maximum 50-72 characters
- Be specific and descriptive

### Body
**Optional**. Provides detailed explanation:

```
feat(api): add rate limiting middleware

Implement token bucket algorithm for API rate limiting to
prevent abuse and ensure fair resource allocation. Limits
are configurable per endpoint and user tier.

Configuration:
- Standard users: 100 req/min
- Premium users: 1000 req/min
- Admin users: unlimited

The middleware checks Redis for current usage and blocks
requests that exceed the limit with 429 status code.
```

**Guidelines**:
- Wrap at 72 characters
- Explain "what" and "why", not "how"
- Include context for future maintainers
- Reference issues/tickets
- Describe limitations or known issues

### Footer
**Optional**. References issues, breaking changes:

```
feat(api): migrate to v2 endpoints

BREAKING CHANGE: API v1 endpoints are deprecated.
All clients must migrate to /api/v2/* endpoints.

Migration guide: docs/api-v2-migration.md

Closes #123, #456
Refs #789
```

**Breaking Change** format:
```
BREAKING CHANGE: <description>
```

**Issue References**:
```
Fixes #123           (closes issue)
Closes #123, #456    (closes multiple)
Refs #123            (references without closing)
Related to #123      (related issue)
```

---

## Examples by Type

### feat (Feature)

**Simple feature**:
```
feat(auth): add JWT token refresh endpoint

Implement /auth/refresh endpoint that accepts a refresh
token and returns a new access token. Tokens expire after
15 minutes, refresh tokens last 7 days.

Closes #234
```

**Feature with configuration**:
```
feat(payments): integrate Stripe payment processing

Add Stripe SDK integration for payment processing with
support for one-time charges and subscriptions. Webhook
handling included for payment status updates.

Configuration required:
- STRIPE_SECRET_KEY
- STRIPE_WEBHOOK_SECRET

Closes #567
```

### fix (Bug Fix)

**Simple fix**:
```
fix(auth): prevent SQL injection in login query

Replace string concatenation with parameterized query to
prevent SQL injection vulnerability in authentication flow.

Refs SECURITY-89
```

**Fix with context**:
```
fix(api): resolve race condition in cache invalidation

Add distributed lock using Redis to prevent race condition
where multiple requests simultaneously invalidate and
repopulate cache, causing temporary data inconsistency.

The lock ensures only one process rebuilds the cache while
others wait. Timeout set to 5 seconds to prevent deadlock.

Fixes #890
```

### docs (Documentation)

```
docs(readme): add Docker setup instructions

Include comprehensive Docker setup guide with:
- Docker compose configuration
- Environment variable setup
- Common troubleshooting steps
- Production deployment notes
```

### style (Code Style)

```
style(api): format code with Prettier

Run Prettier on all API route files to ensure consistent
code formatting across the codebase. No functional changes.
```

### refactor (Refactoring)

```
refactor(db): extract query logic into repository pattern

Move database queries from controllers into dedicated
repository classes to improve separation of concerns and
testability. No behavior changes.

Files affected:
- src/repositories/UserRepository.js (new)
- src/controllers/UserController.js (modified)
```

### perf (Performance)

```
perf(api): add Redis caching for user profile queries

Implement Redis cache layer for frequently accessed user
profile data, reducing database load and improving response
times from 200ms to 15ms (92% improvement).

Cache TTL: 5 minutes
Cache invalidation: on profile update

Closes #445
```

### test (Testing)

```
test(auth): add integration tests for OAuth flow

Add comprehensive integration tests covering:
- Successful OAuth authorization
- Invalid state parameter handling
- Token exchange error cases
- Refresh token expiry

Coverage increased from 65% to 92%.
```

### build (Build System)

```
build(deps): upgrade webpack to v5

Migrate from webpack 4 to webpack 5 for better tree-shaking
and faster builds. Update all webpack-related plugins and
loaders to compatible versions.

Build time reduced from 45s to 28s (38% improvement).

BREAKING CHANGE: Node.js 10 no longer supported.
Minimum required version is now Node.js 12.

Closes #678
```

### ci (CI/CD)

```
ci(github-actions): add automated security scanning

Add Snyk security scanning to GitHub Actions workflow to
detect vulnerabilities in dependencies on every PR. Fails
build if critical vulnerabilities found.

Scan runs on: pull_request, push to main
```

### chore (Maintenance)

```
chore(deps): update axios to 1.5.0

Update axios from 1.4.0 to 1.5.0 for bug fixes and security
patches. No API changes affecting our usage.
```

### revert (Revert)

```
revert: feat(api): add rate limiting middleware

Reverts commit a1b2c3d. Rate limiting caused issues with
legitimate high-volume users and needs redesign.

This reverts commit a1b2c3d4e5f6789.
```

---

## Multi-Commit Scenarios

### Multiple Logical Changes (Split)

❌ **Bad** (mixed changes):
```
feat: add user profile and fix login bug

Added user profile page with avatar upload. Also fixed
a bug where users couldn't login with special characters
in password.
```

✅ **Good** (atomic commits):
```
Commit 1:
fix(auth): handle special characters in login password
Encode password input to handle special characters like
@, #, $ properly during authentication.
Fixes #234

Commit 2:
feat(profile): add user profile page with avatar upload
Implement profile page showing user info and avatar.
Avatar upload uses S3 with automatic resizing to 200x200.
Closes #235
```

### Related Changes (Group)

✅ **Good** (grouped related changes):
```
feat(api): implement user search functionality

Add full-text search for users with filters:
- Search by name, email, username
- Filter by role, status, creation date
- Pagination with 20 results per page
- Response time <100ms with indexed queries

Includes:
- SearchController with /api/search/users endpoint
- Database indexes on name, email columns
- Integration tests with 95% coverage

Closes #789
```

---

## Anti-Patterns

### ❌ Vague Messages

**Bad**:
```
fix bug
update code
changes
wip
minor fix
```

**Good**:
```
fix(auth): prevent infinite redirect loop on logout

Add session state check to prevent redirect loop when user
logs out and session cookie is malformed. Returns 401
instead of redirect if session cannot be validated.
```

### ❌ Too Much Detail in Subject

**Bad**:
```
feat(api): add new endpoint /api/v2/users/:id/profile that returns user profile information including avatar URL, bio, social links, and last login timestamp
```

**Good**:
```
feat(api): add user profile endpoint

Implement /api/v2/users/:id/profile endpoint returning:
- Basic info (name, email, username)
- Profile details (avatar, bio, social links)
- Activity (last login, join date)
```

### ❌ Multiple Unrelated Changes

**Bad**:
```
update dependencies and fix login bug and add user search
```

**Good**: Split into 3 commits:
```
chore(deps): update dependencies to latest versions
fix(auth): handle edge case in login validation
feat(search): add user search with filters
```

### ❌ Placeholder Messages

**Bad**:
```
wip
temp
test
asdf
cleanup
```

**Good**: Describe what's actually changed, even for WIP:
```
feat(checkout): implement payment form (WIP)

Add payment form component with Stripe integration.
Still TODO:
- Error handling for failed charges
- Loading states during processing
- Success confirmation page
```

### ❌ Passive Voice

**Bad**:
```
bug was fixed
feature was added
tests were updated
```

**Good**:
```
fix(api): resolve timeout in batch operations
feat(ui): add dark mode toggle
test(auth): increase coverage to 95%
```

---

## Advanced Patterns

### Squash Commit Message

When squashing multiple commits in a PR:

```
feat(payments): implement subscription management

Combines commits:
- Add subscription model and database schema
- Implement create/update/cancel endpoints
- Add Stripe webhook integration
- Add subscription status UI component
- Add integration tests

Provides complete subscription lifecycle management with:
- Monthly and annual billing cycles
- Prorated upgrades/downgrades
- Automatic renewal with retry logic
- Webhook handling for payment failures

Closes #1234, #1235, #1236
```

### Co-Authored Commits

When pairing or collaborating:

```
feat(api): implement GraphQL schema

Add GraphQL schema with Query and Mutation types for user
and post entities. Includes DataLoader for N+1 prevention
and authentication directives.

Co-authored-by: Jane Doe <jane@example.com>
Co-authored-by: Bob Smith <bob@example.com>
```

### Security Fix Pattern

```
fix(sec): patch XSS vulnerability in comment rendering

Sanitize user-generated HTML in comment rendering to
prevent XSS attacks. Use DOMPurify library to strip
dangerous tags and attributes while preserving safe
formatting like bold and italic.

Affected versions: 1.0.0 - 1.5.2
CVE: CVE-2024-1234
Severity: HIGH

SECURITY FIX: Do not backport without approval.
```

### Deprecation Pattern

```
feat(api): add v3 endpoints and deprecate v2

Introduce API v3 with improved response structure and
pagination. Mark v2 endpoints as deprecated.

DEPRECATION NOTICE: API v2 will be removed in version 4.0
(scheduled for 2025-06-01). Clients should migrate to v3.

Migration guide: docs/api-v3-migration.md

Refs #5678
```

---

## Commit Message Template

Create `.git-commit-template.txt`:

```
# <type>(<scope>): <subject> (max 50 chars)
# |<----  Using a Maximum Of 50 Characters  ---->|

# Explain why this change is being made
# |<----   Try To Limit Each Line to a Maximum Of 72 Characters   ---->|

# Provide links or keys to any relevant tickets, articles or other resources
# Example: Closes #123

# --- COMMIT END ---
# Type can be
#    feat     (new feature)
#    fix      (bug fix)
#    refactor (refactoring production code)
#    style    (formatting, missing semi colons, etc)
#    docs     (changes to documentation)
#    test     (adding or refactoring tests)
#    chore    (updating grunt tasks etc)
#    perf     (performance improvement)
#    build    (build system changes)
#    ci       (CI configuration changes)
#    revert   (reverting a previous commit)
# --------------------
# Remember to
#   - Use imperative mood in subject line
#   - Do not end the subject line with a period
#   - Separate subject from body with a blank line
#   - Use the body to explain what and why vs. how
#   - Can use multiple lines with "-" for bullet points in body
# --------------------
```

Configure git to use template:
```bash
git config --global commit.template ~/.git-commit-template.txt
```

---

## Validation Tools

### commitlint Configuration

```javascript
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',
        'fix',
        'docs',
        'style',
        'refactor',
        'perf',
        'test',
        'build',
        'ci',
        'chore',
        'revert'
      ]
    ],
    'subject-case': [2, 'always', 'lower-case'],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
    'footer-max-line-length': [2, 'always', 100]
  }
};
```

### Pre-commit Hook

```bash
#!/bin/sh
# .git/hooks/commit-msg

# commitlint validation
npx --no-install commitlint --edit "$1"
```

### GitHub Action for Commit Validation

```yaml
name: Lint Commit Messages
on: [pull_request]

jobs:
  commitlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      - uses: wagoid/commitlint-github-action@v5
```

---

## Changelog Generation

### Automated from Conventional Commits

```bash
# Using conventional-changelog
npx conventional-changelog-cli -p angular -i CHANGELOG.md -s

# Using standard-version
npx standard-version

# Using semantic-release
npx semantic-release
```

### Generated CHANGELOG.md

```markdown
# Changelog

## [2.1.0] - 2024-11-07

### Features
- **api**: add rate limiting middleware (#234)
- **auth**: implement OAuth2 support (#256)
- **payments**: integrate Stripe subscriptions (#267)

### Bug Fixes
- **auth**: prevent SQL injection in login (#245)
- **api**: resolve race condition in cache (#278)

### Performance
- **db**: add indexes for frequently queried columns (#290)

### Breaking Changes
- **api**: migrate to v2 endpoints. v1 endpoints deprecated. (#301)
```

---

## Best Practices Summary

1. **Atomic commits**: One logical change per commit
2. **Imperative mood**: "add feature" not "added feature"
3. **Descriptive subject**: Explain what changed, not how
4. **Detailed body**: Provide context for future maintainers
5. **Reference issues**: Link to tickets for traceability
6. **Breaking changes**: Clearly mark with BREAKING CHANGE
7. **Consistent format**: Follow team's chosen convention
8. **No WIP commits**: Squash before merging
9. **Meaningful messages**: Avoid "fix", "update", "changes"
10. **Test before commit**: Ensure changes work before committing
