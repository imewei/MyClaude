---
name: fix-commit-errors
description: Automatically analyzes GitHub Actions failures, identifies root causes, applies intelligent solutions, validates, and reruns workflows with adaptive learning.
category: "dev-suite"
command: "/fix-commit-errors"
execution-modes:
  quick-fix: "5-10m: Discovery + Fix"
  standard: "15-30m: Full resolution + learning"
  comprehensive: "30-60m: Deep analysis + correlation"
allowed-tools: Bash(gh:*), Bash(git:*), Bash(npm:*), Bash(yarn:*), Bash(uv:*), Bash(cargo:*), Bash(go:*), Read, Edit, ScheduleWakeup
argument-hint: "[workflow-id|commit-sha|pr-number] [--auto-fix] [--learn] [--mode=quick-fix|standard|comprehensive]"
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

**Confidence Scoring** (enforced in `engine.py` as `AUTO_APPLY_CONFIDENCE`):
- ≥70%: eligible for auto-apply
- <70%: reported in the plan as manual review, never dispatched

**Risk Levels:**
- L1 (Safe): Config, `go mod tidy`, raising CI timeouts → Auto-apply
- L2 (Moderate): Code fixes, dependency installs → Allowlisted targets only
- L3 (Risky): Dependency removal, major upgrades, API changes → Manual only

**Suppression** — strategies that stop a check failing without addressing the
cause (regenerating snapshots, `--legacy-peer-deps`) are not "safe" merely
because they are low-effort: they make CI green by changing what CI asks.
Snapshot regeneration requires an explicit `--allow-suppression` opt-in, is
never reported as SUCCESS, and downgrades the final "all errors resolved"
claim when used.

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

**Location:** `.github/fix-knowledge-base.json`

A strategy's recorded success rate is ignored until it has at least 3 attempts,
so a single lucky result cannot pin it at 100%.

```json
{
  "error_patterns": [{
    "pattern": "ERESOLVE.*peer dependency",
    "solutions": [{"action": "npm_install_legacy_peer_deps", "success_rate": 0.85}]
  }]
}
```

## Safety

Implemented in `engine.py`:

- **Plan before mutation** — without `--auto-commit` the engine prints what it
  would do and exits having written nothing, installed nothing, and run nothing.
- **Confidence threshold** — errors below `AUTO_APPLY_CONFIDENCE` are not dispatched.
- **Suppression opt-in** — failure-silencing strategies require `--allow-suppression`.
- **No automated dependency removal** — an npm 404 is reported, never uninstalled,
  since registry outages and auth failures produce the same log line.
- **Allowlisted installs** — a missing Python module is only installed if its
  module→package mapping is explicitly known; unmapped names go to a human.
- **Timeouts only increase** — an existing higher `timeout-minutes` is preserved.
- Transparent commit messages.

Not implemented — do not rely on these: automatic local validation before push,
and automatic rollback of applied changes. Review the diff yourself.

## Examples

```bash
/fix-commit-errors                                      # Analysis only
/fix-commit-errors --auto-fix                           # Fix latest
/fix-commit-errors 12345 --auto-fix --learn            # Specific run
/fix-commit-errors PR#123 --mode=comprehensive          # Deep analysis
```
