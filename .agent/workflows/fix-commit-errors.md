---
description: Automatically analyzes GitHub Actions failures, identifies root causes,
  applies intelligent solutions, validates, and reruns workflows with adaptive learning.
triggers:
- /fix-commit-errors
- workflow for fix commit errors
version: 1.0.7
category: cicd-automation
command: /fix-commit-errors
execution-modes:
  quick-fix: '5-10m: Discovery + Fix'
  standard: '15-30m: Full resolution + learning'
  comprehensive: '30-60m: Deep analysis + correlation'
documentation:
  multi-agent-system: ../docs/cicd-automation/multi-agent-error-analysis.md
  error-patterns: ../docs/cicd-automation/error-pattern-library.md
  fix-strategies: ../docs/cicd-automation/fix-strategies.md
allowed-tools: Bash(gh:*), Bash(git:*), Bash(npm:*), Bash(yarn:*), Bash(pip:*), Bash(cargo:*),
  Bash(go:*)
argument-hint: '[workflow-id|commit-sha|pr-number] [--auto-fix] [--learn] [--mode=quick-fix|standard|comprehensive]'
color: red
---


# Intelligent GitHub Actions Failure Resolution

$ARGUMENTS

**Flags:** `--auto-fix`, `--learn`, `--mode=quick-fix|standard|comprehensive`

## Phase 1: Detection

Target: run ID, commit SHA, PR number, or latest failure

```bash
gh run list --status failure --limit 10
gh run view $RUN_ID --log-failed > error_logs.txt
```

## Phase 2: Pattern Analysis

**Error Categories:**
- Dependency: `npm ERR!`, `ERESOLVE`, `No module named`, `unresolved import`
- Build: `TS[0-9]+:`, `Module not found`, `undefined reference`
- Test: `FAIL`, `AssertionError`, `timeout`, `panic:`
- Runtime: `OOM`, `ECONNREFUSED`, `ETIMEDOUT`
- CI Setup: cache failures, `setup-*` failures

**Root Cause:**
1. Technical: What, why, when
2. Historical: Compare successful runs
3. Correlation: Systemic vs job-specific
4. Environmental: OS/version/timing

## Phase 3: Solution Selection

**Confidence Scoring:**
- HIGH (>80% success): Auto-apply
- MEDIUM (>50%): Test branch first
- LOW (<50%): Manual review

**Risk Levels:**
- L1 (Safe): Config, `--legacy-peer-deps`, `go mod tidy` → Auto-apply
- L2 (Moderate): Code fixes, tests → Validate first
- L3 (Risky): Major upgrades, API changes → PR only

## Phase 4: Apply & Validate

```bash
npm test && npm run build && npm run lint
# Pass → commit/push | Fail → rollback, try next
```

## Phase 5: Re-run

```bash
git push origin $(git branch --show-current)
gh run watch
```

**Auto-fix loop (max 5 iterations):**
1. Analyze errors
2. Apply highest-confidence fix
3. Trigger run
4. Monitor result
5. Update knowledge base

## Phase 6: Knowledge Base

**Location:** `.github/fix-commit-errors/knowledge.json`

```json
{
  "error_patterns": [{
    "pattern": "ERESOLVE.*peer dependency",
    "solutions": [{"action": "npm_install_legacy_peer_deps", "success_rate": 0.85}]
  }]
}
```

## Success Criteria

- Resolution rate: >65%
- Time to fix: <15 min
- Confidence accuracy: >80%
- Regressions: 0

## Safety

- Validate locally before push
- Rollback available for all changes
- Confidence threshold enforcement
- Transparent commit messages

## Examples

```bash
/fix-commit-errors                                      # Analysis only
/fix-commit-errors --auto-fix                           # Fix latest
/fix-commit-errors 12345 --auto-fix --learn            # Specific run
/fix-commit-errors PR#123 --mode=comprehensive          # Deep analysis
```
