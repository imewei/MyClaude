# Fix Strategies

**Version**: 1.0.3
**Command**: `/fix-commit-errors`
**Category**: CI/CD Automation

## Overview

Comprehensive fix strategies for CI/CD errors organized by risk level, with iterative approaches, validation loops, rollback procedures, and prevention strategies.

---

## Fix Strategy Hierarchy

### Level 1: Configuration Fixes (Safest - Auto-Apply)
**Risk**: Low | **Reversibility**: High | **Auto-Apply**: Yes (confidence >0.8)

These fixes modify workflow configurations, package manager flags, or environment variables without changing application code.

#### Strategy 1.1: Workflow YAML Updates

**When to Use**: CI configuration issues, version mismatches, flag additions

**Examples**:
```bash
# Update Node.js version
sed -i 's/node-version: 16/node-version: 18/' .github/workflows/*.yml

# Add --legacy-peer-deps flag for npm
sed -i 's/npm ci/npm ci --legacy-peer-deps/' .github/workflows/*.yml

# Update action versions
sed -i 's/actions\/cache@v2/actions\/cache@v3/' .github/workflows/*.yml

# Add environment variable
sed -i '/env:/a\  NODE_OPTIONS: --max-old-space-size=4096' .github/workflows/*.yml
```

**Validation**:
```bash
# Verify YAML syntax
yamllint .github/workflows/*.yml

# Check workflow locally (with act)
act -l
```

#### Strategy 1.2: Package Manager Fixes

**When to Use**: Dependency conflicts, cache issues, lock file mismatches

**NPM/Yarn**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Update specific package
npm update package-name@latest

# Add missing dependency
npm install package-name --save-dev

# Fix peer dependencies
npm install --legacy-peer-deps
```

**Python/Pip**:
```bash
# Upgrade package
pip install --upgrade package-name

# Clear cache
pip cache purge

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Go**:
```bash
# Tidy dependencies
go mod tidy

# Download missing modules
go mod download

# Update dependency
go get -u package-path@version
```

**Rust/Cargo**:
```bash
# Update dependencies
cargo update

# Clean and rebuild
cargo clean && cargo build
```

#### Strategy 1.3: Cache Management

```bash
# GitHub Actions: Update cache key in workflow
# Old: key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
# New: key: ${{ runner.os }}-node-v2-${{ hashFiles('**/package-lock.json') }}

# Or delete cache via GitHub CLI
gh cache delete <cache-id>

# Or clear all caches
gh cache list | awk '{print $1}' | xargs -I {} gh cache delete {}
```

---

### Level 2: Code Fixes (Moderate Risk - Validate First)
**Risk**: Medium | **Reversibility**: Medium | **Auto-Apply**: With validation

These fixes modify application code, imports, or type definitions.

#### Strategy 2.1: Import Fixes

**Missing Imports**:
```typescript
// Detect missing React import in TSX files
find src -name "*.tsx" -exec grep -l "useState\|useEffect" {} \; | \
  xargs grep -L "import.*React" | \
  xargs sed -i '1i import React from "react";'
```

**Update Import Paths**:
```typescript
// Fix relative imports after directory restructure
find src -name "*.ts" -exec sed -i 's|from "../old-path/|from "../new-path/|g' {} \;
```

#### Strategy 2.2: Type Fixes

**Add Type Definitions**:
```bash
# Install missing @types package
npm install --save-dev @types/node @types/react

# Add type assertion for third-party library
echo "declare module 'untyped-package';" >> src/types/global.d.ts
```

**Fix Type Errors**:
```typescript
// Add explicit type annotations
// Before: const data = getData();
// After: const data: DataType = getData();

# Use sed for simple patterns
sed -i 's/const \(data\) =/const \1: DataType =/' src/file.ts
```

#### Strategy 2.3: Test Fixes

**Update Snapshots**:
```bash
# Jest/Vitest snapshot update
npm test -- -u

# Update specific test file
npm test -- path/to/test.spec.ts -u
```

**Increase Timeouts**:
```bash
# Increase default timeout
sed -i 's/timeout: 5000/timeout: 10000/' test/*.test.js

# Or in jest.config.js
sed -i 's/testTimeout: 5000/testTimeout: 10000/' jest.config.js
```

**Fix Flaky Tests**:
```typescript
// Add retry logic
jest.retryTimes(3);

// Add explicit waits
await waitFor(() => expect(element).toBeInTheDocument(), { timeout: 5000 });

// Mock unstable dependencies
jest.mock('./unstable-module');
```

---

### Level 3: Complex Fixes (Manual Review Required)
**Risk**: High | **Reversibility**: Low | **Auto-Apply**: No

These fixes involve breaking changes, migrations, or architectural changes.

#### Strategy 3.1: Major Version Updates

**Approach**: Create PR with proposed changes, never auto-apply

```bash
# Generate migration PR
git checkout -b fix/major-upgrade-$(date +%s)

# Update package.json
npm install package@latest

# Document breaking changes
echo "## Breaking Changes\n- Feature X removed\n- API Y changed" > UPGRADE.md

# Create PR
git add package.json package-lock.json UPGRADE.md
git commit -m "chore: upgrade package to v2.0

Breaking changes:
- Updated API usage in src/components
- Removed deprecated features

Requires manual review before merge"

gh pr create --title "Upgrade package to v2.0" --body "$(cat UPGRADE.md)"
```

#### Strategy 3.2: API Signature Changes

**Approach**: Identify all usage sites, update systematically

```bash
# Find all usages of old API
git grep -n "oldApiFunction"

# Use codemod for automated refactoring
npx jscodeshift -t transform-old-to-new.js src/

# Verify with tests
npm test
```

#### Strategy 3.3: Database Migrations

**Approach**: NEVER auto-apply, generate migration scripts

```bash
# Generate migration
npm run db:migration:generate -- --name fix-schema-issue

# Review migration SQL
cat migrations/20251106-fix-schema.sql

# Test on staging database
npm run db:migrate -- --env=staging

# Only proceed to production with approval
```

---

## Iterative Fix Approach

### Algorithm

```python
class IterativeFixEngine:
    def __init__(self, max_iterations=5):
        self.max_iterations = max_iterations
        self.iteration = 0
        self.fixes_applied = []

    def execute(self, error, solutions):
        """
        Iteratively apply fixes until error resolved or max iterations reached
        """
        while self.iteration < self.max_iterations:
            self.iteration += 1

            # Select next solution
            solution = self.select_best_solution(solutions, self.fixes_applied)

            if not solution:
                return FixResult(
                    success=False,
                    reason='No more solutions available',
                    iterations=self.iteration
                )

            # Apply fix
            fix_result = self.apply_fix(solution)
            self.fixes_applied.append({
                'solution': solution,
                'result': fix_result,
                'iteration': self.iteration
            })

            # Validate fix
            validation = self.validate_fix()

            if validation.success:
                return FixResult(
                    success=True,
                    solution=solution,
                    iterations=self.iteration,
                    fixes_applied=self.fixes_applied
                )

            # Fix didn't work, rollback if needed
            if solution.risk_level == 'high':
                self.rollback_fix(solution)

        # Max iterations reached
        return FixResult(
            success=False,
            reason='Max iterations reached',
            iterations=self.iteration,
            fixes_applied=self.fixes_applied
        )
```

### Iteration Example

```
Iteration 1:
  ✓ Apply: npm install --legacy-peer-deps
  ✓ Commit: fix(deps): use legacy peer deps
  ✓ Push: trigger new workflow run
  ✗ Result: Still failing (different error)

Iteration 2:
  ✓ Apply: npm update react@18
  ✓ Commit: fix(deps): upgrade react to v18
  ✓ Push: trigger new workflow run
  ✓ Result: SUCCESS

Total Time: 8 minutes
Fixes Applied: 2
```

---

## Validation Loops

### Pre-Apply Validation

```bash
#!/bin/bash
# Validate fix before applying

validate_fix() {
    local fix_type=$1

    case $fix_type in
        "workflow_yaml")
            # Validate YAML syntax
            yamllint .github/workflows/*.yml || return 1
            ;;
        "package_json")
            # Validate package.json
            npm install --dry-run || return 1
            ;;
        "typescript")
            # Type check
            npm run typecheck || return 1
            ;;
        "tests")
            # Run affected tests
            npm test --findRelatedTests || return 1
            ;;
    esac

    return 0
}
```

### Post-Apply Validation

```bash
#!/bin/bash
# Validate after fix applied

post_fix_validation() {
    echo "Running post-fix validation..."

    # 1. Run local tests
    if ! npm test; then
        echo "❌ Tests failed"
        return 1
    fi

    # 2. Run build
    if ! npm run build; then
        echo "❌ Build failed"
        return 1
    fi

    # 3. Run linting
    if ! npm run lint; then
        echo "❌ Linting failed"
        return 1
    fi

    # 4. Check for new errors
    if grep -i "error\|fail" build-output.log; then
        echo "❌ New errors introduced"
        return 1
    fi

    echo "✅ All validations passed"
    return 0
}
```

---

## Rollback Procedures

### Automatic Rollback

```bash
#!/bin/bash
# Automatic rollback on validation failure

apply_fix_with_rollback() {
    local fix_script=$1

    # Save current state
    git stash push -m "Pre-fix state $(date +%s)"
    local stash_ref=$(git stash list | head -1 | cut -d: -f1)

    # Apply fix
    if ! bash "$fix_script"; then
        echo "Fix application failed, rolling back..."
        git stash pop "$stash_ref"
        return 1
    fi

    # Validate
    if ! post_fix_validation; then
        echo "Validation failed, rolling back..."
        git stash pop "$stash_ref"
        return 1
    fi

    # Success, drop stash
    git stash drop "$stash_ref"
    return 0
}
```

### Manual Rollback

```bash
# Rollback last commit
git reset --hard HEAD~1
git push origin $(git branch --show-current) --force

# Revert specific commit
git revert <commit-sha>
git push origin $(git branch --show-current)

# Restore from backup branch
git checkout fix/backup-$(date +%Y%m%d)
git checkout -b fix/attempt-2
```

---

## Prevention Strategies

### 1. Dependency Management

**Lock File Hygiene**:
```bash
# Commit lock files
git add package-lock.json yarn.lock Cargo.lock go.sum
git commit -m "chore: update lock files"

# Use exact versions for critical deps
npm install --save-exact critical-package@1.2.3

# Audit dependencies regularly
npm audit
npm audit fix
```

**Dependency Updates**:
```javascript
// renovate.json - Automated dependency updates
{
  "extends": ["config:base"],
  "automerge": true,
  "automergeType": "pr",
  "packageRules": [
    {
      "matchDepTypes": ["devDependencies"],
      "automerge": true
    },
    {
      "matchPackagePatterns": ["^eslint"],
      "groupName": "eslint packages"
    }
  ]
}
```

### 2. CI/CD Best Practices

**Matrix Testing**:
```yaml
# Test across multiple environments
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    node: [16, 18, 20]
```

**Caching Strategy**:
```yaml
# Proper cache configuration
- uses: actions/cache@v3
  with:
    path: |
      ~/.npm
      ~/.cache
      node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

### 3. Code Quality Gates

**Pre-commit Hooks**:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-added-large-files
```

**Automated Testing**:
```yaml
# Run tests before push
- name: Run tests
  run: npm test -- --coverage --passWithNoTests
```

### 4. Monitoring and Alerting

**Workflow Success Rate Tracking**:
```bash
# Track success rate over time
gh run list --json conclusion --limit 100 | \
  jq '[.[] | select(.conclusion == "success")] | length'

# Alert on degradation
if [ $success_rate -lt 80 ]; then
    echo "::error::Workflow success rate below 80%"
fi
```

---

## Fix Strategy Selection Algorithm

```python
def select_fix_strategy(error, context):
    """Select optimal fix strategy based on error type and context"""

    # Get all applicable strategies
    strategies = get_strategies_for_error(error)

    # Score each strategy
    scored_strategies = []
    for strategy in strategies:
        score = calculate_strategy_score(
            strategy=strategy,
            error=error,
            context=context,
            historical_success_rate=get_historical_success_rate(strategy, error),
            risk_level=strategy.risk_level,
            reversibility=strategy.reversibility,
            time_to_apply=strategy.estimated_time
        )
        scored_strategies.append((strategy, score))

    # Sort by score (highest first)
    scored_strategies.sort(key=lambda x: x[1], reverse=True)

    # Return top strategy that meets confidence threshold
    for strategy, score in scored_strategies:
        if score > 0.7:  # 70% confidence threshold
            return strategy

    return None  # No confident strategy available
```

---

## Success Metrics

### Fix Strategy Performance

| Strategy Level | Avg Success Rate | Avg Time | Auto-Apply |
|----------------|------------------|----------|------------|
| Level 1: Config | 88% | 2-5 min | Yes |
| Level 2: Code | 72% | 5-15 min | With validation |
| Level 3: Complex | 45% | 30-120 min | No (manual review) |

### Iterative Fix Performance

- **1 Iteration Success**: 65%
- **2 Iterations Success**: 85%
- **3 Iterations Success**: 92%
- **4-5 Iterations Success**: 95%

Average iterations to success: 1.8

---

For complete error patterns and root cause analysis, see:
- [error-pattern-library.md](error-pattern-library.md)
- [multi-agent-error-analysis.md](multi-agent-error-analysis.md)
- [knowledge-base-system.md](knowledge-base-system.md)
