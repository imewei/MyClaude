# Breaking Changes Guide

Comprehensive guide to managing API compatibility, migrations, and rollback strategies.

## What is a Breaking Change?

A breaking change is any modification that requires users to change their code or configuration.

### Examples of Breaking Changes

**API/Library**:
- Removing a public function, method, or endpoint
- Changing function signatures (parameters, return types)
- Changing behavior of existing functionality
- Renaming public interfaces
- Removing or renaming configuration options
- Changing default values that affect behavior

**Database**:
- Removing or renaming columns
- Changing column types
- Adding NOT NULL constraint without default
- Removing tables
- Changing relationships (foreign keys)

**UI**:
- Removing user-facing features
- Changing URL structure
- Modifying data formats in storage

---

## Semantic Versioning (SemVer)

**Format**: MAJOR.MINOR.PATCH (e.g., 2.3.1)

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Examples

- `1.2.3 → 1.2.4`: Bug fix, no breaking changes
- `1.2.4 → 1.3.0`: New feature, backward compatible
- `1.3.0 → 1.0.2`: Breaking change

**Pre-release**: `1.0.2-alpha.1`, `1.0.2-beta.1`, `1.0.2-rc.1`

---

## Avoiding Breaking Changes

### 1. Additive Changes Only

**✅ Good - Add new, deprecate old**:
```python
# v1.0
def get_user(id):
    return db.query(User).filter_by(id=id).first()

# v1.1 - Add new, keep old
def get_user(id):  # Still works
    return _get_user_by_id(id)

def get_user_by_id(id):  # New preferred method
    return _get_user_by_id(id)

def _get_user_by_id(id):
    return db.query(User).filter_by(id=id).first()
```

### 2. Optional Parameters

**✅ Good - New parameter is optional**:
```javascript
// v1.0
function fetchData(url) {
  return fetch(url);
}

// v1.1 - Add optional parameter
function fetchData(url, options = {}) {
  return fetch(url, options);
}
```

**❌ Bad - Required parameter breaks existing code**:
```javascript
// v2.0 - BREAKING
function fetchData(url, options) {  // options now required!
  return fetch(url, options);
}
```

### 3. Extend, Don't Modify

**✅ Good - Extend response**:
```json
// v1.0
{"id": 123, "name": "User"}

// v1.1 - Add new field
{"id": 123, "name": "User", "email": "user@example.com"}
```

**❌ Bad - Change structure**:
```json
// v2.0 - BREAKING
{"user": {"id": 123, "name": "User"}}
```

---

## Deprecation Strategy

### Deprecation Process

1. **Announce**: Mark as deprecated with clear notice
2. **Provide alternative**: Document replacement
3. **Set timeline**: Give users time to migrate (e.g., 6 months)
4. **Monitor usage**: Track deprecated feature usage
5. **Remove**: Remove in next major version

### Deprecation Warnings

**Python**:
```python
import warnings

def old_function(x):
    warnings.warn(
        "old_function is deprecated, use new_function instead. "
        "Will be removed in v3.0.0",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(x)
```

**JavaScript**:
```javascript
function oldFunction(x) {
  console.warn(
    'oldFunction is deprecated, use newFunction instead. ' +
    'Will be removed in v3.0.0'
  );
  return newFunction(x);
}
```

**HTTP API**:
```
Sunset: Sat, 31 Dec 2024 23:59:59 GMT
Deprecation: true
Link: <https://api.example.com/docs/migration>; rel="deprecation"
```

---

## API Versioning Strategies

### 1. URL Path Versioning

```
GET /v1/users
GET /v2/users
```

**Pros**: Clear, easy to route
**Cons**: Duplicate URLs, routing complexity

### 2. Header Versioning

```
GET /users
Accept: application/vnd.example.v1+json
```

**Pros**: Clean URLs
**Cons**: Less visible, harder to test

### 3. Query Parameter

```
GET /users?api_version=2
```

**Pros**: Simple, backward compatible
**Cons**: Can be forgotten, inconsistent

### Recommendation

**Use URL path versioning** for major versions:
```
/v1/  # Version 1.x.x
/v2/  # Version 2.x.x
```

Use headers or parameters for minor version negotiation.

---

## Database Migrations

### Blue-Green Deployment Pattern

**Phase 1**: Deploy code that works with both schemas
```python
# Read from old or new column
def get_user_name(user):
    return user.full_name or user.first_name + ' ' + user.last_name
```

**Phase 2**: Migrate data
```sql
UPDATE users SET full_name = first_name || ' ' || last_name WHERE full_name IS NULL;
```

**Phase 3**: Remove old code/columns
```python
# Only use new column
def get_user_name(user):
    return user.full_name
```

### Making Schema Changes Backwards Compatible

**Adding column** (non-breaking):
```sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
```

**Adding NOT NULL** (two-step, avoid breaking):
```sql
-- Step 1: Add column with default
ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT '';

-- Step 2 (later): Make NOT NULL after data migrated
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
```

**Renaming column** (use views temporarily):
```sql
-- Create view with old name
CREATE VIEW users_legacy AS
SELECT id, email as old_email_name FROM users;

-- Later: Drop view after clients migrated
DROP VIEW users_legacy;
```

### Migration Tools

**Python (Alembic)**:
```python
def upgrade():
    op.add_column('users', sa.Column('email', sa.String(255), nullable=True))

def downgrade():
    op.drop_column('users', 'email')
```

**Node.js (Knex)**:
```javascript
exports.up = function(knex) {
  return knex.schema.table('users', function(table) {
    table.string('email');
  });
};

exports.down = function(knex) {
  return knex.schema.table('users', function(table) {
    table.dropColumn('email');
  });
};
```

**Test migrations**:
```bash
# Test upgrade
npm run migrate:up

# Test downgrade (rollback)
npm run migrate:down

# Test full cycle
npm run migrate:up && npm run migrate:down && npm run migrate:up
```

---

## Feature Flags

### Gradual Rollout

```python
from unleash import UnleashClient

client = UnleashClient(url="http://unleash.example.com/api")

def get_user_profile(user_id):
    if client.is_enabled("new_profile_api", context={"userId": user_id}):
        return new_get_user_profile(user_id)
    else:
        return old_get_user_profile(user_id)
```

### Progressive Rollout

```
Day 1: 5% traffic → Monitor
Day 2: 25% traffic → Monitor
Day 3: 50% traffic → Monitor
Day 5: 100% traffic → Success!
```

---

## Rollback Strategies

### Code Rollback

**Git revert**:
```bash
# Revert last commit
git revert HEAD
git push

# Revert specific commit
git revert abc123
```

**Kubernetes rollback**:
```bash
kubectl rollout undo deployment/myapp
kubectl rollout undo deployment/myapp --to-revision=2
```

### Database Rollback

**Always write reversible migrations**:
```python
def upgrade():
    op.add_column('users', sa.Column('email', sa.String()))

def downgrade():
    op.drop_column('users', 'email')
```

**Test rollback**:
```bash
# Migrate up
alembic upgrade head

# Rollback
alembic downgrade -1

# Should succeed without data loss
```

### Canary Rollback

If canary shows issues, immediately shift traffic back:
```bash
# Shift 100% traffic to stable version
kubectl set image deployment/myapp myapp=stable-version
```

---

## Migration Guides

### Template

```markdown
# Migrating from v1 to v2

## Overview
Brief summary of major changes and why.

## Breaking Changes

### 1. Function Signature Change

**Before (v1)**:
```python
get_user(id)
```

**After (v2)**:
```python
get_user(user_id=id)
```

**Migration**:
```python
# Find and replace
get_user(123)  # Old
get_user(user_id=123)  # New
```

### 2. Configuration Change

**Before (v1)**:
```yaml
database:
  url: postgres://...
```

**After (v2)**:
```yaml
database:
  connection:
    url: postgres://...
```

## Deprecated Features

- `old_function()` → Use `new_function()` instead (removed in v3.0)
- `legacy_config` → Use `new_config` instead (removed in v3.0)

## New Features

- Feature X: Enables improved performance
- Feature Y: Adds support for Z

## Migration Steps

1. Update dependencies
   ```bash
   npm install package@1.0.2
   ```

2. Run automated migration tool
   ```bash
   npx migrate-to-v2
   ```

3. Update configuration
4. Run tests
5. Deploy

## Rollback

If issues arise:
```bash
npm install package@1.9.0
# Revert configuration changes
```

## Support

- GitHub Issues: https://github.com/example/package/issues
- Migration help: migration@example.com
```

---

## Communication Checklist

### Announcing Breaking Changes

- [ ] Document all breaking changes in CHANGELOG
- [ ] Update migration guide
- [ ] Deprecation warnings in previous version
- [ ] Blog post explaining changes
- [ ] Email to users/maintainers
- [ ] GitHub release notes
- [ ] Update documentation
- [ ] Set clear deprecation timeline

---

## Testing for Compatibility

### Contract Testing

**Pact (Consumer-Driven Contracts)**:
```javascript
// Consumer test
const { Pact } = require('@pact-foundation/pact');

it('gets a user', () => {
  const provider = new Pact({ ... });

  await provider.addInteraction({
    state: 'user exists',
    uponReceiving: 'a request for a user',
    withRequest: {
      method: 'GET',
      path: '/users/123'
    },
    willRespondWith: {
      status: 200,
      body: { id: 123, name: 'Test User' }
    }
  });
});
```

### Snapshot Testing

```javascript
// Ensure API response format doesn't change
it('matches snapshot', async () => {
  const response = await api.get('/users/123');
  expect(response.data).toMatchSnapshot();
});
```

---

## Checklist: Before Breaking Changes

- [ ] Is this change absolutely necessary?
- [ ] Can it be done in a backward-compatible way?
- [ ] Have I documented the deprecation?
- [ ] Is there a migration guide?
- [ ] Have users been notified?
- [ ] Is there a transition period (e.g., 6 months)?
- [ ] Are feature flags in place for gradual rollout?
- [ ] Is there a rollback plan?
- [ ] Have I tested the migration path?

---

## References

- [Semantic Versioning](https://semver.org/)
- [API Evolution (Stripe)](https://stripe.com/blog/api-versioning)
- [Expand-Contract Pattern](https://martinfowler.com/bliki/ParallelChange.html)
