---
name: fix-commit-errors
description: Automatically analyzes GitHub Actions failures, identifies root causes, applies intelligent solutions, validates, and reruns workflows with adaptive learning.
category: "dev-suite"
command: "/fix-commit-errors"
execution-modes:
  quick-fix: "5-10m: Discovery + Fix"
  standard: "15-30m: Full resolution + learning"
  comprehensive: "30-60m: Deep analysis + correlation"
allowed-tools: Bash(gh:*), Bash(git:*), Bash(npm:*), Bash(yarn:*), Bash(uv:*), Bash(cargo:*), Bash(go:*), Bash(python3:*), Read, Edit, ScheduleWakeup
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

Selection is entirely `engine.py`'s job — this phase is descriptive, not a
separate step to perform by hand:

**Auto-apply eligibility** (`plan_fixes` in `engine.py`):
- `npm_eresolve`, `python_import`, `eslint_error`, `oom`, `timeout` (`LOW_RISK_TYPES`):
  auto-applied by risk tier — each carries its own internal safety net
  (closed allowlists, parse/reparse verification, `max()`-never-lowers), so
  a fresh knowledge base cannot veto them.
- Everything else: needs `AUTO_APPLY_CONFIDENCE` (≥70%) learned confidence to
  auto-apply; below that it is reported in the plan as manual review, never
  dispatched.
- `npm_404` (`MANUAL_REVIEW_TYPES`): always manual — a 404 in a log is also
  what a registry outage or auth failure looks like; never auto-uninstalled.
- `test_failure` (`SUPPRESSION_TYPES`): snapshot regeneration suppresses the
  failure rather than fixing it. Gated behind explicit `--allow-suppression`,
  never reported as SUCCESS, and downgrades the final "all errors resolved"
  claim when used.

## Phase 4: Apply, Commit, Push, Re-trigger

Run the engine directly — it plans, applies, commits, pushes, triggers a new
workflow run, and waits for completion, repeating up to `--max-iterations`:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/iterative-error-resolution/engine.py "$RUN_ID" \
  --repo "$REPO" \
  --workflow "$WORKFLOW" \
  --max-iterations 5 \
  # --auto-commit only with --auto-fix; omit for a plan-only dry run that
  # writes/installs/commits nothing
  # --allow-suppression only if the user explicitly asked to allow snapshot
  # regeneration as a fix
```

There is no separate local-validation-then-rollback step — see Safety below
for what "not implemented" actually means here.

## Phase 5: Knowledge Base

**Location:** `.github/fix-knowledge-base.json`, keyed by error type (the same
key `parse_logs` assigns and `plan_fixes` gates on). A strategy's recorded
success rate is ignored until it has at least 3 attempts (`MIN_SAMPLES`), so a
single lucky result cannot pin it at 100%.

```json
{
  "npm_eresolve": {
    "base_confidence": 0.85,
    "total_attempts": 4,
    "successes": 3
  }
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
