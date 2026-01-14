---
name: iterative-error-resolution
version: "1.0.7"
maturity: "5-Expert"
specialization: CI/CD Error Resolution & Knowledge-Based Fixing
description: Iterative CI/CD error resolution with pattern recognition, automated fixes, and learning from outcomes. Use when debugging GitHub Actions, fixing dependency/build/test failures, or implementing automated error resolution loops.
---

# Iterative Error Resolution for CI/CD

Systematic framework for analyzing failures, applying intelligent fixes, and iterating until zero errors.

---

## Error Categories

| Category | Examples | Fix Strategy |
|----------|----------|--------------|
| Dependency | npm ERESOLVE, pip conflicts | Version relaxation, flags |
| Build | TypeScript errors, ESLint | Auto-fix, type corrections |
| Test | Jest failures, pytest | Snapshot update, assertions |
| Runtime | OOM, timeout | Resource limits, retries |
| Network | ETIMEDOUT, ENOTFOUND | Retry logic, fallback |

---

## Fix Patterns

### npm Dependency Fixes

```bash
# ERESOLVE conflicts
sed -i 's/npm install/npm install --legacy-peer-deps/g' .github/workflows/*.yml

# 404 package
npm uninstall "$package_name"
jq "del(.dependencies[\"$package_name\"])" package.json > temp.json && mv temp.json package.json
```

### Python Dependency Fixes

```bash
# Version conflict - relax constraint
sed -i "s/${package}==.*/${package}/g" requirements.txt

# Missing module
pip install "$missing_module" && pip freeze | grep -i "$missing_module" >> requirements.txt
```

### TypeScript Fixes

```typescript
// Object possibly undefined -> add optional chaining
fixed = content.replace(/(\w+)\.(\w+)/g, '$1?.$2');

// Type assertion for unknown properties
fixed = addTypeAssertion(content, line);
```

### Runtime Fixes

```bash
# OOM - increase heap
sed -i '/env:/a\        NODE_OPTIONS: "--max-old-space-size=4096"' .github/workflows/*.yml

# Timeout - increase limit
sed -i 's/timeout-minutes: [0-9]*/timeout-minutes: 60/' .github/workflows/*.yml

# Network - add retry
# Use nick-invision/retry@v2 action with max_attempts: 3
```

---

## Iterative Fix Engine

```python
class IterativeFixEngine:
    def __init__(self, repo: str, workflow: str, max_iterations: int = 5):
        self.repo = repo
        self.workflow = workflow
        self.max_iterations = max_iterations

    def run(self, initial_run_id: str) -> bool:
        current_run_id = initial_run_id

        for iteration in range(1, self.max_iterations + 1):
            errors = self.analyze_run(current_run_id)

            if not errors:
                return True  # Zero errors!

            fixes_applied = []
            for error in self.prioritize_fixes(errors):
                if self.apply_fix(error):
                    fixes_applied.append(error.suggested_fix)

            if not fixes_applied:
                return False  # Manual intervention needed

            self.commit_and_push(fixes_applied, iteration)
            new_run_id = self.trigger_workflow()
            self.wait_for_completion(new_run_id)

            if self.get_run_status(new_run_id) == "success":
                return True

            current_run_id = new_run_id

        return False  # Max iterations reached
```

---

## Knowledge Base

```python
class KnowledgeBase:
    def get_fix_strategy(self, error_type: str) -> str:
        defaults = {
            'npm_eresolve': 'Add --legacy-peer-deps flag',
            'npm_404': 'Remove unavailable package',
            'ts_error': 'Fix TypeScript type errors',
            'eslint_error': 'Run ESLint auto-fix',
            'test_failure': 'Update test snapshots',
            'python_import': 'Install missing module',
            'timeout': 'Increase timeout duration',
            'oom': 'Increase memory allocation'
        }
        return defaults.get(error_type, 'Manual review required')

    def calculate_confidence(self, error_type: str, fix: str) -> float:
        # Recency-weighted success rate
        history = self.success_history.get(f"{error_type}:{fix}", [])
        if len(history) < 3:
            return 0.5
        weights = [2 ** i for i in range(len(history))]
        return sum(w * s for w, s in zip(weights, history)) / sum(weights)
```

---

## Validation & Rollback

```bash
validate_fix() {
    git tag "checkpoint-$(date +%s)"
    npm test || { rollback_fix; return 1; }
    npm run build || { rollback_fix; return 1; }
    return 0
}

rollback_fix() {
    git revert --no-commit HEAD
    git commit -m "fix(ci): rollback failed fix"
    git push
}
```

---

## Integration with /fix-commit-errors

```bash
python3 engine.py "$RUN_ID" \
    --repo "$REPO" \
    --workflow "$WORKFLOW" \
    --max-iterations 5

# Engine will:
# 1. Analyze errors from failed run
# 2. Apply fixes automatically
# 3. Trigger new workflow
# 4. Wait for completion
# 5. Repeat until zero errors or max iterations
# 6. Learn from outcomes
```

---

## Best Practices

| Practice | Rationale |
|----------|-----------|
| High-confidence first | Apply >80% confidence fixes first |
| Validate locally | Run tests before pushing |
| Limit iterations | 3-5 max to prevent infinite loops |
| Learn from failures | Record failed fixes to avoid repeating |
| Rollback on regression | Auto-revert if new errors introduced |
| Manual threshold | Escalate if confidence <50% |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Resolution rate | >80% per iteration |
| Rollback rate | <10% |
| Time to resolution | <30 min |
| Zero-error achievement | >90% of runs |

---

## Error Resolution Checklist

- [ ] Errors categorized by type
- [ ] Fix strategies prioritized by confidence
- [ ] Local validation before push
- [ ] Rollback mechanism ready
- [ ] Knowledge base updated
- [ ] Iteration limit set
- [ ] Success metrics tracked

---

**Version**: 1.0.5
