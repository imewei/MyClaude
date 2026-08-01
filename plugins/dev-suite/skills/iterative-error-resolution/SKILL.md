---
name: iterative-error-resolution
description: Reference patterns for CI/CD error classification, fix strategies, and resolution loops. Provides domain knowledge for the /fix-commit-errors command. Use when analyzing GitHub Actions failures, dependency conflicts, or build/test error patterns.
---

# Iterative Error Resolution for CI/CD

## Expert Agent

For CI/CD error diagnosis, automated fix loops, and pipeline troubleshooting, delegate to:

- **`automation-engineer`**: Debugs pipeline failures with pattern recognition and automated resolution.
  - *Location*: `plugins/dev-suite/agents/automation-engineer.md`

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

These are what `engine.py`'s `fix_*` methods actually do — not runnable
snippets, since every one of these has a safety detail a naive `sed`/`jq`
one-liner gets wrong (silently removing a real dependency on a registry
outage, lowering a deliberately-raised timeout, corrupting YAML). Call the
engine; do not hand-reimplement these.

| Error type | What the engine does | Why it's not simpler |
|---|---|---|
| `npm_eresolve` | Appends `--legacy-peer-deps` to `npm install`/`npm ci` in workflow files | Masks the conflict rather than resolving it — treated as a stopgap, not a fix |
| `npm_404` | **Never auto-removes.** Reports the package for manual confirmation | A 404 is also what a registry outage or private-scope auth failure looks like |
| `python_import` | Installs from a closed module→package allowlist (`cv2`→`opencv-python`, `PIL`→`Pillow`, `sklearn`→`scikit-learn`) and appends to `requirements.txt` with a dedup check | An unmapped module name goes to a human — a well-formed name from CI log text is not evidence it's the real package |
| `eslint_error` | Runs `npx eslint . --fix`, reads the real exit code (0=success, 1=partial, ≥2=failed) | Exit 1 means unfixable errors remain; reporting that as SUCCESS anyway is how a broken run gets committed |
| `oom` | Injects `NODE_OPTIONS` into the workflow's top-level `env:` block, parsing and re-parsing the YAML to verify the edit actually landed before writing | A blind `sed -i '/env:/a...'` can append after the wrong `env:` or produce invalid YAML with no verification |
| `timeout` | Raises `timeout-minutes` via `max(existing, 60)` — never lowers a deliberately-set higher value | A flat `sed -i 's/timeout-minutes: [0-9]*/timeout-minutes: 60/'` silently *lowers* an intentional 120 |
| `test_failure` (snapshot) | Regenerates snapshots **only** with `--allow-suppression`, never reports SUCCESS, sets a flag that downgrades the run's final claim | Suppresses the failure rather than fixing it — a rewritten expectation makes the loop's own exit condition true regardless of whether the change was correct |

---

## Iterative Fix Engine

The real loop lives in `engine.py`'s `IterativeFixEngine.run()`. Its actual
shape, not a simplified re-derivation:

1. `analyze_run` fetches and parses logs. A fetch failure returns `None`
   (distinct from an empty list) so it aborts rather than reporting a false
   "zero errors" success; an empty list is only trusted as clean if the run's
   own status also says `success`.
2. `plan_fixes` decides what may be touched **before anything is touched** —
   see Fix Patterns above for the per-type gating — and the plan is printed.
3. Without `--auto-commit`, the loop stops here: nothing was written,
   installed, or run.
4. Applied fixes are committed and pushed; a push failure aborts the
   iteration rather than triggering CI against unpushed code.
5. A new run is triggered and matched by the pushed commit's SHA (not "most
   recent run" — that can return the run already being analyzed), then
   awaited to any terminal status (not just success/failure/cancelled).
6. Iterates up to `--max-iterations`, updating the knowledge base after each
   attempt.

## Knowledge Base

The real `KnowledgeBase` in `engine.py` is a flat per-error-type running
average, not a recency-weighted history:

- Keyed by error type (`npm_eresolve`, `oom`, ...) — the same key
  `parse_logs` assigns, so recording a fix and reading its confidence back
  never mismatch.
- `base_confidence` = `successes / total_attempts`.
- A type's recorded confidence is ignored until it has `MIN_SAMPLES = 3`
  attempts — under that, `calculate_confidence` treats it as neutral (0.5) so
  a single lucky result cannot pin it at 100%.
- Persisted to `.github/fix-knowledge-base.json` after every iteration.

## Validation & Rollback

**Not implemented.** There is no local-validation-before-push step and no
automated rollback — `/fix-commit-errors`' own Safety section says so
explicitly. The plan-before-mutation gate (step 3 above) is the actual safety
mechanism: review the printed plan, then opt in with `--auto-commit`. Review
the diff yourself before trusting a run.

---

## Integration with /fix-commit-errors

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/iterative-error-resolution/engine.py "$RUN_ID" \
    --repo "$REPO" \
    --workflow "$WORKFLOW" \
    --max-iterations 5

# Engine will:
# 1. Analyze errors from failed run (aborts, not "zero errors", if the fetch itself fails)
# 2. Print the plan; apply fixes only if --auto-commit was passed
# 3. Commit and push (aborts the iteration on push failure)
# 4. Trigger new workflow, matched by pushed commit SHA
# 5. Wait for completion (any terminal status, not just success/failure/cancelled)
# 6. Repeat until zero errors or max iterations
# 7. Learn from outcomes, keyed by error type
```

---

## Best Practices

| Practice | Rationale |
|----------|-----------|
| Risk-tier before confidence | `LOW_RISK_TYPES` auto-apply on their own safety net; everything else needs learned confidence |
| Review the plan first | `plan_fixes` prints what will run before `--auto-commit` touches anything |
| Limit iterations | 3-5 max to prevent infinite loops |
| Learn from failures | Record every attempt (keyed by error type) to avoid repeating what doesn't work |
| Review the diff yourself | No local validation or rollback is implemented — a green run after `--auto-commit` is not itself proof; check the diff |
| Manual threshold | Escalate if confidence <50% |

---

## Parallel Resolution Strategies

| Task | Strategy | Benefit |
|------|----------|---------|
| **Validation** | Parallel test runners | Faster confirmation of fixes |
| **Analysis** | Concurrent log parsing | Speed up multi-job failure analysis |
| **Fix Application** | Independent file patches | Apply non-conflicting fixes concurrently |
| **Regression Check** | Parallel matrix builds | Ensure fix works across all environments |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Resolution rate | >80% per iteration |
| Time to resolution | <30 min |
| Zero-error achievement | >90% of runs |

---

## Error Resolution Checklist

- [ ] Errors categorized by type
- [ ] Fix strategies prioritized by confidence/risk tier
- [ ] Plan reviewed before `--auto-commit`
- [ ] Diff reviewed after the run (no automated validation or rollback exists)
- [ ] Knowledge base updated
- [ ] Iteration limit set
- [ ] Success metrics tracked

---

**Version**: 1.0.5
