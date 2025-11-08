# Pull Request Review Best Practices

> **Reference**: Code review guidelines, PR templates, review checklists, and communication patterns

## PR Size Guidelines

### Optimal Size
- **Target**: 200-400 lines changed
- **Maximum**: 600 lines (review quality decreases beyond this)
- **Minimum**: 20 lines (too small = overhead)

### When PR is Too Large
**Split strategies**:
1. **Feature-based**: Separate by logical feature boundaries
2. **Layer-based**: Split by backend/frontend/database
3. **Incremental**: Create dependent PRs (PR1 → PR2 → PR3)
4. **Preparatory refactoring**: Refactor first, feature second

```bash
# Example: Large feature split
PR #1: Database schema and migrations (200 lines)
PR #2: Backend API endpoints (300 lines)
PR #3: Frontend components (250 lines)
PR #4: Integration tests (150 lines)
```

---

## PR Description Template

### Standard Template

```markdown
## Summary
Brief 2-3 sentence description of what this PR does and why.

## Changes
- Added user authentication with JWT
- Implemented password reset flow
- Added rate limiting to login endpoint

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Refactoring (no functional changes)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated (coverage: 92%)
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Tested in staging environment

### Test Cases Covered
1. Successful login with valid credentials
2. Failed login with invalid credentials
3. Account lockout after 5 failed attempts
4. Password reset email delivery
5. Token expiration after 1 hour

## Screenshots/Videos
[Add for UI changes]

## Performance Impact
- Login endpoint: 150ms → 85ms (43% improvement)
- Database queries reduced from 5 to 2 per request

## Breaking Changes
None

## Deployment Notes
- Requires environment variables: JWT_SECRET, JWT_EXPIRY
- Run migration: `npm run migrate`
- Restart application after deployment

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No console.log or debug code
- [ ] All tests passing
- [ ] No new warnings
```

### Bug Fix Template

```markdown
## Bug Description
Login fails when password contains special characters like @, #, $

## Root Cause
Password input not being URL-encoded before sending to API, causing
special characters to break the request.

## Fix
- URL-encode password on client side before API call
- Add input sanitization on server side as defense-in-depth

## Testing
- [ ] Verified fix with passwords containing: @, #, $, %, &, =
- [ ] Added unit tests for all special characters
- [ ] Regression tested normal passwords still work

## Affected Versions
v1.0.0 - v1.5.2

## Related Issues
Fixes #234
Related to #235
```

---

## Review Checklist

### Functionality
- [ ] Code does what PR description claims
- [ ] Edge cases handled appropriately
- [ ] Error handling implemented
- [ ] Input validation present
- [ ] No obvious bugs or logic errors

### Code Quality
- [ ] Code is readable and self-documenting
- [ ] Variable/function names are descriptive
- [ ] No duplicate code (DRY principle followed)
- [ ] Functions are small and focused (<50 lines)
- [ ] Complexity is justified
- [ ] Comments explain "why" not "what"

### Testing
- [ ] Tests cover happy path
- [ ] Tests cover edge cases
- [ ] Tests cover error conditions
- [ ] Test names are descriptive
- [ ] No flaky tests introduced
- [ ] Coverage meets project standards (typically 80%+)

### Security
- [ ] No SQL injection vulnerabilities
- [ ] Input is sanitized/validated
- [ ] No XSS vulnerabilities
- [ ] Sensitive data not logged
- [ ] Authentication/authorization checks present
- [ ] No hardcoded secrets or credentials
- [ ] HTTPS enforced for sensitive operations

### Performance
- [ ] No N+1 query problems
- [ ] Database queries are indexed
- [ ] No unnecessary loops or recursion
- [ ] Caching implemented where appropriate
- [ ] Large lists are paginated
- [ ] No memory leaks introduced

### API Design
- [ ] Endpoints follow REST conventions
- [ ] Consistent naming with existing API
- [ ] Proper HTTP status codes used
- [ ] Response format consistent
- [ ] Versioning strategy followed
- [ ] Backward compatibility maintained

### Database
- [ ] Migrations are reversible
- [ ] Indexes added for queried columns
- [ ] Foreign keys properly constrained
- [ ] No data loss in migrations
- [ ] Migration tested on copy of production data

### Documentation
- [ ] README updated if needed
- [ ] API documentation updated
- [ ] Inline comments for complex logic
- [ ] CHANGELOG updated
- [ ] Migration guide provided for breaking changes

---

## Review Communication

### Approval Comments

**✅ Good**:
```
LGTM! Nice work on the optimization - the caching strategy
is elegant and well-tested. Just one minor suggestion about
error messaging, but not blocking approval.
```

**✅ Excellent**:
```
Great PR! I especially like how you:
1. Added comprehensive test coverage (95%)
2. Optimized the N+1 query problem
3. Included clear migration instructions

One question about the cache TTL (see comment on line 47),
but approving since it's not critical.
```

### Requesting Changes

**❌ Avoid**:
```
This doesn't work. Change it.
```

**✅ Better**:
```
The error handling in login.js:45 doesn't account for network
timeouts. Consider adding a timeout handler that retries once
before showing error to user. Example:

```javascript
try {
  await loginAPI(credentials, { timeout: 5000 });
} catch (error) {
  if (error.code === 'TIMEOUT' && !retried) {
    // Retry once
  }
  throw new UserFacingError('Login failed. Please try again.');
}
```

Let me know if you'd like me to clarify!
```

### Asking Questions

**✅ Thoughtful questions**:
```
Question: Why did we choose Redis over memcached for
caching here? I'm curious about the trade-offs considered.

Nit: Consider extracting this validation logic (lines 123-145)
into a separate validator class for reusability?

Suggestion: Could we add a metric to track cache hit rate?
Would be useful for monitoring performance over time.
```

### Providing Context

**✅ Contextual feedback**:
```
Note: We had a similar caching issue in the user service
(see PR #456) where cache invalidation on updates was missed.
Suggest adding a test case that verifies cache is cleared
when user profile is updated.
```

---

## Review Response Guidelines

### For PR Authors

**Respond to all comments**:
- Acknowledge feedback
- Explain decisions
- Ask clarifying questions
- Mark resolved when addressed

**❌ Avoid**:
```
done
fixed
ok
```

**✅ Better**:
```
Good catch! I've added the timeout handler you suggested in
commit abc123. Also added a test case to verify the retry
logic works correctly.

For the validator suggestion - I agree that's cleaner. Created
follow-up issue #789 to refactor across all auth endpoints.
```

### For Reviewers

**Be constructive**:
```
Current approach works but might cause issues at scale. Consider
using pagination for this list (expected to grow to 10,000+ items).
If you'd like, I can pair with you on implementing this.
```

**Distinguish requirements from suggestions**:
```
MUST FIX: Security vulnerability - user input not sanitized.

SUGGESTION: Consider extracting this into a helper function
for reusability (not blocking).

NIT: Typo in comment on line 67 (not blocking).
```

---

## Automated Review Tools

### Linting & Formatting
```yaml
# .github/workflows/lint.yml
name: Lint
on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm run lint
      - run: npm run format:check
```

### Security Scanning
```yaml
# .github/workflows/security.yml
name: Security Scan
on: [pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
```

### Test Coverage
```yaml
# .github/workflows/coverage.yml
name: Test Coverage
on: [pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm test -- --coverage
      - uses: codecov/codecov-action@v3
        with:
          fail_ci_if_error: true
          threshold: 80
```

---

## Review SLA

| PR Type | Response Time | Approval Time |
|---------|--------------|---------------|
| Hotfix/Security | <2 hours | <4 hours |
| Bug Fix | <12 hours | <24 hours |
| Feature | <24 hours | <48 hours |
| Refactor/Docs | <48 hours | <72 hours |

---

## Common Review Pitfalls

### Pitfall 1: Nitpicking Style
**Problem**: Focusing on code style instead of logic
**Solution**: Use automated linters, focus reviews on functionality

### Pitfall 2: Rubber Stamping
**Problem**: Approving without actually reviewing
**Solution**: Set aside dedicated time, use review checklist

### Pitfall 3: Bikeshedding
**Problem**: Endless debate over trivial naming/structure
**Solution**: Follow "rule of three" - if can't agree in 3 exchanges, escalate

### Pitfall 4: Review Overload
**Problem**: Too many reviewers, conflicting feedback
**Solution**: 1-2 required reviewers, others optional

### Pitfall 5: Blocking on Non-Issues
**Problem**: Blocking approval for personal preference
**Solution**: Distinguish MUST FIX from SUGGESTION clearly

---

## Review Best Practices Summary

1. **Review promptly**: Within SLA timeframes
2. **Be constructive**: Focus on improvement, not criticism
3. **Ask questions**: Understand before judging
4. **Provide context**: Explain the "why" behind feedback
5. **Test locally**: For complex changes, pull branch and test
6. **Check tests**: Ensure comprehensive coverage
7. **Security first**: Always check for vulnerabilities
8. **Suggest improvements**: Share knowledge and best practices
9. **Approve generously**: Don't block on nitpicks
10. **Follow up**: Ensure feedback is addressed
