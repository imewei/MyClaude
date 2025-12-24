---
version: "1.0.5"
category: "cicd-automation"
command: "/fix-commit-errors"
execution-modes:
  quick-fix:
    description: "Rapid error resolution"
    time: "5-10 minutes"
    phases: "Discovery + Fix only"
  standard:
    description: "Full intelligent resolution with learning"
    time: "15-30 minutes"
    phases: "All 7 phases"
  comprehensive:
    description: "Deep analysis with cross-workflow correlation"
    time: "30-60 minutes"
    phases: "All + knowledge base deep dive"
documentation:
  multi-agent-system: "../docs/cicd-automation/multi-agent-error-analysis.md"
  error-patterns: "../docs/cicd-automation/error-pattern-library.md"
  fix-strategies: "../docs/cicd-automation/fix-strategies.md"
description: Automatically analyzes GitHub Actions failures, identifies root causes, applies intelligent solutions, validates, and reruns workflows with adaptive learning.
allowed-tools: Bash(gh:*), Bash(git:*), Bash(npm:*), Bash(yarn:*), Bash(pip:*), Bash(cargo:*), Bash(go:*)
argument-hint: [workflow-id|commit-sha|pr-number] [--auto-fix] [--learn] [--mode=quick-fix|standard|comprehensive]
color: red
agents:
  primary:
    - devops-security-engineer
  conditional:
    - agent: code-quality
      trigger: pattern "test.*fail|lint.*error|quality"
    - agent: fullstack-developer
      trigger: pattern "npm|yarn|webpack|build.*error"
  orchestrated: false
---

# Intelligent GitHub Actions Failure Resolution

Automatically resolve CI/CD failures through multi-agent analysis, UltraThink reasoning, and adaptive learning.

## Arguments

$ARGUMENTS

**Flags:** `--auto-fix` (apply fixes), `--learn` (update knowledge base), `--mode=quick-fix|standard|comprehensive`

---

## Multi-Agent System

| Agent | Role | Output |
|-------|------|--------|
| Log Parser | Retrieve and structure error logs | Structured error data |
| Pattern Matcher | Classify errors by type | Categories, severity |
| Root Cause Analyzer | Determine underlying causes | Root cause, regression analysis |
| Knowledge Base | Apply proven solutions | Ranked solutions by success rate |
| Solution Generator | Generate fix strategies | Executable fixes with rollback |

---

## Phase 1: Failure Detection

### Target Resolution
- Workflow run ID → Analyze specific run
- Commit SHA → Find runs for commit
- PR number → Analyze PR checks
- Empty → Latest failed run

### Data Collection
```bash
gh run list --status failure --limit 10
gh run view $RUN_ID --log-failed > error_logs.txt
```

---

## Phase 2: Error Pattern Detection

### Error Categories

| Category | Patterns |
|----------|----------|
| **Dependency** | `npm ERR!`, `ERESOLVE`, `No module named`, `unresolved import` |
| **Build** | `TS[0-9]+:`, `Module not found`, `undefined reference` |
| **Test** | `FAIL`, `AssertionError`, `timeout`, `panic:` |
| **Runtime** | `OOM`, `ECONNREFUSED`, `ETIMEDOUT` |
| **CI Setup** | `Failed to restore cache`, `setup-*` failures |

### Root Cause Analysis
1. Technical: What failed? Why? When did it start?
2. Historical: Compare with successful runs
3. Correlation: Systemic vs job-specific
4. Environmental: OS/version/timing-specific

---

## Phase 3: UltraThink Intelligence

### Solution Evaluation

For each candidate solution:

| Aspect | Evaluate |
|--------|----------|
| PROS | Success rate, risk level, complexity |
| CONS | Root cause addressed?, side effects? |
| RISK | Breaking change probability, reversion ease |

### Confidence Scoring

| Condition | Confidence | Action |
|-----------|------------|--------|
| Pattern seen, >80% success | HIGH | Auto-apply |
| >50% success | MEDIUM | Test branch first |
| <50% success | LOW | Manual review |

---

## Phase 4: Fix Application

### By Risk Level

| Level | Fixes | Action |
|-------|-------|--------|
| 1 (Safe) | Config changes, `--legacy-peer-deps`, `go mod tidy` | Auto-apply |
| 2 (Moderate) | Code fixes, test updates, snapshot updates | Validate first |
| 3 (Risky) | Major version updates, API changes | Generate PR |

### Validation Loop
```bash
npm test && npm run build && npm run lint
# Pass → commit and push
# Fail → rollback and try next solution
```

---

## Phase 5: Workflow Re-execution

```bash
git push origin $(git branch --show-current)
gh run watch
```

### Iterative Resolution (--auto-fix)
1. Analyze all errors
2. Apply highest-confidence fix
3. Trigger new run
4. Monitor result
5. Repeat until zero errors or max iterations (5)
6. Update knowledge base

---

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

**Learning:** Query → Apply highest success → Track outcome → Update rates

---

## Phase 7: Reporting

### Progress Output
```
🔍 Analyzing failure... Run #12345
📊 Category: Dependency (95% confidence)
🧠 Knowledge Base: Pattern found (20×), 90% success
🎯 Applying: --legacy-peer-deps
✅ SUCCESS! KB updated: 90% → 91%
```

### Report Contents
- Error classification and severity
- Root cause analysis
- Solution applied with rationale
- Validation results
- Rollback instructions

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Resolution rate | >65% |
| Time to fix | <15 min |
| Confidence accuracy | >80% |
| Regressions | 0 |

---

## Safety Guarantees

- ✅ All fixes validated locally first
- ✅ Rollback available for every change
- ✅ Confidence threshold for auto-apply
- ✅ Full transparency in commit messages
- ✅ Failed fixes decrease confidence

---

## Examples

```bash
# Analysis only
/fix-commit-errors

# Auto-fix latest failure
/fix-commit-errors --auto-fix

# Fix specific run with learning
/fix-commit-errors 12345 --auto-fix --learn

# Comprehensive mode for recurring issues
/fix-commit-errors PR#123 --auto-fix --mode=comprehensive
```
