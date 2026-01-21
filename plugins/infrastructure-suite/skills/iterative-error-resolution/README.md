# Iterative Error Resolution Skill

Comprehensive framework for automatically fixing CI/CD errors through iterative analysis, intelligent fix application, and validation loops until zero errors are achieved.

## Overview

This skill provides a complete engine for resolving GitHub Actions failures automatically by:
- Analyzing error patterns in workflow logs
- Categorizing errors by type (dependency, build, test, runtime)
- Applying proven fix strategies from a knowledge base
- Validating fixes and triggering new workflow runs
- Iterating until all errors are resolved or max iterations reached
- Learning from outcomes to improve future fixes

## Quick Start

### Via /fix-commit-errors Command

The easiest way to use this skill is through the `/fix-commit-errors` command:

```bash
# Iterative fix mode (recommended)
/fix-commit-errors --auto-fix

# Fix specific run
/fix-commit-errors 12345678 --auto-fix
```

### Direct Python Engine

You can also call the engine directly:

```bash
# Basic usage
python3 plugins/cicd-automation/skills/iterative-error-resolution/engine.py \
  12345678 \
  --repo owner/repo \
  --workflow "CI"

# With custom max iterations
python3 plugins/cicd-automation/skills/iterative-error-resolution/engine.py \
  12345678 \
  --repo owner/repo \
  --workflow "CI" \
  --max-iterations 3
```

## Supported Error Types

### 1. Dependency Errors

| Error Type | Example | Fix Strategy |
|------------|---------|--------------|
| npm ERESOLVE | `npm ERR! code ERESOLVE` | Add `--legacy-peer-deps` flag |
| npm 404 | `npm ERR! 404 Not Found` | Remove unavailable package |
| npm peer deps | `peer dep missing` | Update peer dependencies |
| Python import | `ModuleNotFoundError` | Install missing module |
| Python version | `Could not find version` | Relax version constraints |

### 2. Build Errors

| Error Type | Example | Fix Strategy |
|------------|---------|--------------|
| TypeScript | `TS2339: Property does not exist` | Add type assertions |
| ESLint | `error Unexpected token` | Run ESLint auto-fix |
| Webpack | `Module not found` | Install missing module |

### 3. Test Errors

| Error Type | Example | Fix Strategy |
|------------|---------|--------------|
| Snapshot mismatch | `Snapshot does not match` | Update snapshots with `-u` |
| Test timeout | `Timeout exceeded` | Increase timeout duration |
| Assertion failure | `Expected X but got Y` | Analyze and fix assertion |

### 4. Runtime Errors

| Error Type | Example | Fix Strategy |
|------------|---------|--------------|
| Out of Memory | `heap out of memory` | Increase Node heap size |
| Timeout | `exceeded maximum execution time` | Increase job timeout |
| Network error | `ETIMEDOUT`, `ENOTFOUND` | Add retry logic |

## How It Works

### Iterative Fix Loop

```
┌─────────────────────────────────────────────────────────────┐
│ Iteration 1                                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Fetch logs from failed run                              │
│ 2. Parse and categorize errors (e.g., 3 npm, 2 TypeScript) │
│ 3. Prioritize by confidence score                          │
│ 4. Apply top fixes (legacy-peer-deps, type assertions)     │
│ 5. Commit and push                                          │
│ 6. Trigger new workflow run                                │
│ 7. Wait for completion                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [Still have errors?]
                              │
                        ┌─────┴─────┐
                        │           │
                       Yes          No
                        │           │
                        ▼           ▼
              ┌─────────────┐  ✓ Success!
              │ Iteration 2 │  Exit 0
              └─────────────┘
                        │
                        ▼
              [Repeat until max iterations]
```

### Confidence Scoring

Each fix is assigned a confidence score based on:
- **Knowledge base history**: Success rate from previous fixes
- **Error clarity**: How clear/specific the error is
- **Context matching**: How well the error matches known patterns

Fixes are prioritized: High confidence + blocking errors first.

### Knowledge Base Learning

The engine maintains a knowledge base (`.github/fix-knowledge-base.json`) that tracks:
- Success rates for each fix strategy
- Total attempts per error type
- Base confidence levels

Over time, the engine learns which fixes work best for your project.

## Configuration

### Max Iterations

Default: 5 iterations

```bash
# Custom max iterations
--max-iterations 3
```

**Recommendation**:
- 3-5 iterations for most projects
- Lower (2-3) for critical production workflows
- Higher (7-10) for experimental/development workflows

### Confidence Thresholds

Fixes are only applied if confidence > 40%. You can see confidence scores in the output:

```
Fixing: npm ERR! code ERESOLVE
Category: dependency
Confidence: 85%
Strategy: Add --legacy-peer-deps flag
```

## Knowledge Base

### Structure

```json
{
  "npm_eresolve": {
    "base_confidence": 0.85,
    "total_attempts": 12,
    "successes": 10,
    "strategies": [
      {
        "strategy": "Add --legacy-peer-deps flag",
        "success_rate": 0.83
      }
    ]
  }
}
```

### Manual Management

```bash
# View knowledge base
cat .github/fix-knowledge-base.json

# Clear knowledge base (reset learning)
rm .github/fix-knowledge-base.json

# Backup knowledge base
cp .github/fix-knowledge-base.json .github/fix-knowledge-base.backup.json
```

## Output and Reporting

### Console Output

```
==============================================================
ITERATION 1/5
==============================================================

Found 3 error(s) to fix

Fixing: npm ERR! code ERESOLVE
Category: dependency
Confidence: 85%
Strategy: Add --legacy-peer-deps flag

✓ Fix applied successfully

Triggering new workflow run...
New run started: 12345679
Waiting for workflow to complete...
..........

✓ SUCCESS: All errors resolved!
```

### Iteration Summary

At the end, you get a summary:

```
==============================================================
ITERATION SUMMARY
==============================================================

Iteration 1:
  Errors found: 3
  Errors fixed: 3
  Errors remaining: 0
  Status: ✓ SUCCESS
  Fixes applied:
    - Add --legacy-peer-deps flag
    - Run ESLint auto-fix
    - Update test snapshots

Total errors encountered: 3
Total errors fixed: 3
Success rate: 100.0%
```

## Exit Codes

- **0**: All errors resolved successfully
- **1**: Some errors remain after max iterations or manual intervention required

## Best Practices

### 1. Start with Analysis Mode

Before using `--auto-fix`, analyze first:

```bash
# Analyze only (no fixes)
/fix-commit-errors

# Review suggestions, then apply
/fix-commit-errors --auto-fix
```

### 2. Use Lower Iterations for Production

For critical workflows, use fewer iterations to prevent cascading changes:

```bash
--max-iterations 2
```

### 3. Monitor Knowledge Base Growth

Periodically review your knowledge base to see what's being learned:

```bash
cat .github/fix-knowledge-base.json | jq '.[] | {confidence: .base_confidence, successes: .successes, total: .total_attempts}'
```

### 4. Commit Knowledge Base

Consider committing `.github/fix-knowledge-base.json` to share learning across team members.

### 5. Manual Review for Low Confidence

If an iteration repeatedly fails, review the suggested fixes manually:

```bash
# Check what fixes were attempted
git log --grep="automated error resolution"
```

## Limitations

1. **Max Iterations**: Engine stops after max iterations even if errors remain
2. **Complex Errors**: Some errors require human judgment and code refactoring
3. **Flaky Tests**: Engine may struggle with non-deterministic test failures
4. **External Dependencies**: Cannot fix errors caused by external service outages

## Troubleshooting

### Engine Fails to Trigger Workflow

**Cause**: Missing GitHub CLI authentication or permissions

**Fix**:
```bash
gh auth status
gh auth login
```

### Fixes Applied But Tests Still Fail

**Cause**: Fixes may be incomplete or error requires deeper changes

**Fix**: Review iteration summary and manually inspect failed fixes:
```bash
git log --oneline --grep="iteration"
git show <commit-hash>
```

### Knowledge Base Grows Too Large

**Cause**: Tracking too many error types

**Fix**: Periodically clean up old or obsolete entries:
```bash
# Backup and reset
cp .github/fix-knowledge-base.json .github/fix-knowledge-base.backup.json
echo '{}' > .github/fix-knowledge-base.json
```

## Examples

### Example 1: npm Dependency Conflict

**Error**: `npm ERR! code ERESOLVE unable to resolve dependency tree`

**Iteration 1**:
- Adds `--legacy-peer-deps` to all `npm install` commands in workflows
- Commits and pushes
- Triggers new run
- ✓ Success!

### Example 2: Multiple TypeScript Errors

**Error**: 5 TypeScript type errors across different files

**Iteration 1**:
- Runs `tsc --noEmit` to identify errors
- Attempts auto-fixes (type assertions)
- Fixes 3/5 errors
- Commits and re-runs

**Iteration 2**:
- Analyzes remaining 2 errors
- Applies stricter fixes
- Fixes 2/2 errors
- ✓ Success!

### Example 3: Test Snapshot Mismatch

**Error**: `Snapshot does not match stored snapshot`

**Iteration 1**:
- Runs `npm test -- -u` to update snapshots
- Commits new snapshots
- Re-runs tests
- ✓ Success!

## Integration with Other Tools

### Pre-commit Hooks

```bash
# .git/hooks/pre-push
#!/bin/bash
# Auto-fix CI errors before pushing
/fix-commit-errors --auto-fix --max-iterations 2
```

### GitHub Actions Workflow

```yaml
name: Auto-Fix CI Errors
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches: [main]

jobs:
  auto-fix:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Auto-fix errors
        run: |
          python3 plugins/cicd-automation/skills/iterative-error-resolution/engine.py \
            ${{ github.event.workflow_run.id }} \
            --repo ${{ github.repository }} \
            --workflow "CI" \
            --max-iterations 3
```

## Contributing

To add support for new error types:

1. Add pattern to `parse_logs()` in `engine.py`
2. Add category mapping in `get_category()`
3. Implement fix method (e.g., `fix_new_error_type()`)
4. Add default strategy to `KnowledgeBase.get_fix_strategy()`
5. Update documentation in `SKILL.md`

## Support

For issues or questions:
- Review logs: `gh run view <run-id> --log-failed`
- Check knowledge base: `cat .github/fix-knowledge-base.json`
- Open issue: Repository issues page
