# MyClaude Migration Notes — v3.3.2 → v3.4.0 (Opus 4.7 / Claude Code v2.1.113)

This document records every component-level change made during the
plugin-forge modernization sweep. The validator (Task 4) cross-checks every
entry here against `.baseline-inventory.txt` to confirm zero functions
were silently lost.

Format: each section corresponds to one task in the workflow.

---

## Task 1 — Plugin manifests & agent frontmatter

**Plugin-architect (`plugin-dev:agent-creator`).** Commit `34462a97`.

### Changes
- All 3 `plugin.json` `keywords` arrays: `"opus-4.6"` → `"opus-4.7"`.

### No-ops
- All 24 agent frontmatter blocks already used canonical tier tokens
  (`opus`/`sonnet`/`haiku`) — no model-ID rewrites needed.
- No agent uses `model: inherit` — nothing to make explicit.
- Schema clean across all 3 manifests; no deprecated fields.

### Deferred (open question) — RESOLVED in follow-up

- `context-specialist` was on `sonnet`. **Elevated to `opus`** in a follow-up
  commit after the v3.4.0 release (see git log for the elevation commit).
  Rationale:
  - The agent already declares `effort: high`, `maxTurns: 35`,
    `background: true`, `memory: project` — every signal points to a
    deep-reasoning workload, not a routine task.
  - Its scope ("dynamic context management, vector databases, knowledge
    graphs, intelligent memory systems, multi-agent workflow orchestration")
    is decision-heavy infrastructure that other agents depend on.
  - Other foundational agents — `orchestrator` and `reasoning-engine` —
    are already on `opus` in agent-core. Keeping `context-specialist` on
    `sonnet` was inconsistent with the same-suite peer pattern.
  - Under Opus 4.7's 1M context, the design space for memory and retrieval
    strategies expands; sub-optimal context-specialist decisions cascade
    to every agent that depends on it.

---

## Task 2 — Hook event schema audit & orphan-script cleanup

**Hook-engineer (`hookify:conversation-analyzer`).** Orchestrator-driven
because the agent could not run shell commands; orchestrator executed the
deletion plan.

### Wired event audit
Confirmed 24/24 hook handlers wired in `hooks.json` files (12 in
agent-core + 7 in dev-suite + 5 in science-suite). All event names valid
under Claude Code v2.1.113 supported set. JSON structure clean.

Note: 2 of the 24 wired handlers are `prompt`-type (not script-backed),
so the script-file count from `hooks.json` references is 22, not 24:
- `dev-suite/hooks/hooks.json` `PreToolUse` — prompt-type guard against
  destructive git/rm Bash commands.
- `science-suite/hooks/hooks.json` `PreToolUse` — prompt-type guard
  against simulation-data overwrite.

### Baseline correction (this sweep, audit finding)
`.baseline-inventory.txt` listed 6 disk-only script filenames including
`pre_task_use.py`. **That was a miscount.** `pre_task_use.py` IS wired —
it's the handler for `agent-core`'s `PreToolUse` event with matcher
`Task` (note: the script is named `pre_task_use.py`, not the more obvious
`pre_tool_use.py` which doesn't exist in `agent-core/hooks/` at all).
Corrected truly-orphan list: 5 unique filenames, 7 total file copies
(because `execution_error.py` exists in 3 suites).

### Deleted (7 files, dead code matching events removed from CC schema)

| File | Event no longer supported by CC v2.1.113 |
|---|---|
| `plugins/agent-core/hooks/context_overflow.py` | `ContextOverflow` |
| `plugins/agent-core/hooks/cost_threshold.py` | `CostThreshold` |
| `plugins/agent-core/hooks/execution_error.py` | `ExecutionError` |
| `plugins/agent-core/hooks/permission_prompt.py` | `PermissionPrompt` |
| `plugins/agent-core/hooks/pre_subagent_use.py` | `PreSubagentUse` |
| `plugins/dev-suite/hooks/execution_error.py` | `ExecutionError` |
| `plugins/science-suite/hooks/execution_error.py` | `ExecutionError` |

Cross-reference grep confirmed no live imports, no `settings.json`
references, and no `hooks.json` `description`-field mentions before
deletion. These match the events CLAUDE.md notes as removed (commit
`148a2df5 fix(hooks): remove unsupported event keys from hooks.json`
removed the wired entries; this sweep removes the orphan scripts).

### Bare-except audit
Grep for `^\s*except\s*:` across all `plugins/*/hooks/*.py` returned
**zero matches**. No fixes needed.

### Opus 4.7 model-ID handling
Grep for `claude-(opus|sonnet|haiku)-4|opus-4\.|sonnet-4\.|haiku-4\.`
across all hook scripts returned **zero matches**. No hook script
branches on model ID — nothing to update.

### Disk script count
- Before: 29 hook scripts on disk (per baseline)
- After: 22 hook scripts on disk (29 − 7 deleted)
- Wired handler count unchanged: **24/24** (12 + 7 + 5)
- Wired script-file count: 22 (24 minus 2 prompt-type handlers)

---

## Task 3 — Skill budgets & hub→sub-skill reachability

**Skill-reviewer (`plugin-dev:skill-reviewer`).** Orchestrator-driven
because the audit is mechanical — run the validators and report.

### Budget audit (under both 200K and 1M context windows)

`context_budget_checker.py` reports against both window sizes simultaneously:

| Metric | Result |
|---|---|
| Total skills checked | 206 |
| Fits 200K budget (4,000 tokens) | 206/206 (100%) |
| Fits 1M budget (20,000 tokens) | 206/206 (100%) |
| Skills oversized at 200K | 0 |
| Skills oversized at 1M | 0 |

**Headroom warnings (>75% of 200K budget):**

| Skill | Plugin | Tokens | 200K usage | 1M usage |
|---|---|---|---|---|
| `thinkfirst` | agent-core | 3,040 | 76% | 15% |

Only one skill brushes the 75% headroom threshold under the 200K window;
under 1M it sits at 15% — comfortable.

### Budget policy decision

**Keep the 4 KB (200K) absolute byte limit, NOT the 20 KB scaled limit.**

Rationale:
- Many MyClaude users still run Opus 4.6 (200K) for cost reasons. Bloating
  skills to 20 KB would silently break those users by overflowing their
  skill listing budget.
- The 4 KB ceiling forces tight, front-loaded prose — a quality property
  worth preserving regardless of model tier.
- The checker already reports BOTH budgets, so headroom under 1M is
  visible without lowering the gate.

This decision is durable: future skills added to MyClaude must continue to
fit the 4 KB budget. The 1M column is informational only.

### Hub → sub-skill reachability

`xref_validator.py` results: **526 cross-references validated, 0 broken.**
All 27 hub skills route correctly into the 179 sub-skills. No orphans.

### Stale prose audit

Grep across all 206 `SKILL.md` files for `200K context | 200,000 |
opus-4\.6 | claude-opus-4-6 | Opus 4\.6` returned **zero matches**. Skill
prose is model-agnostic and forward-compatible.

### Tooling fix

Updated `tools/validation/context_budget_checker.py` docstring:
- Added Opus 4.7 reference for 1M context (was "1M beta", now standard).
- Added explicit policy note that 4K is the gate, 20K is informational.
- Removed stale `--context-size 200000` usage example (flag not implemented).

---

## Follow-up audit (2026-04-18) — orphan-command policy clarified

`/dev-suite:double-check --deep` surfaced that 12 of the 22 unregistered
commands are NOT invoked anywhere in `plugins/` (CLAUDE.md previously
claimed all 22 were skill-invoked). Substantial files (29-318 lines),
not stubs.

**Decision: option (D) — keep them as intentional reference templates,
update CLAUDE.md to be accurate.** No commands deleted, no commands
registered. The 12 commands stay on disk for reference and for users
to copy/adapt; CLAUDE.md `Plugin Suites` section now documents this
explicitly. Net effect: zero functions lost, zero functions added,
documentation now matches reality.

The 12 reference templates: `adopt-code`, `agent-build`, `ai-assistant`,
`c-project`, `deps`, `monitor-setup`, `onboard`, `paper-review`,
`profile-performance`, `run-experiment`, `rust-project`, `scaffold`.

## Task 4 — Regression gate, version bump, UPGRADE_REPORT.md

**Validator (`plugin-dev:plugin-validator`).** Orchestrator-driven.

### Validator suite results
- `metadata_validator.py` per suite: ✅ 0 errors, 0 warnings (all 3).
- `xref_validator.py`: ✅ 526/526 references valid, 0 broken.
- `context_budget_checker.py`: ✅ 206/206 fit both 200K and 1M budgets.
- `skill_validator.py`: ✅ no significant frontmatter issues.
- `pytest tools/tests/ -v`: ✅ **154/154 passed in 5.40 s**.
- `doc_checker.py`: 15 errors + 77 warnings TOTAL across 3 suites — these
  are pre-existing broken `${CLAUDE_PLUGIN_ROOT}/docs/...` links, NOT
  introduced by this sweep. Listed as deferred work in `UPGRADE_REPORT.md`.

### Inventory diff vs `.baseline-inventory.txt`
- Agents: 24/24 ✓
- Registered commands: 14/14 ✓
- Commands on disk: 36/36 ✓
- Hub skills: 27/27 ✓
- Sub-skills: 179/179 ✓
- Total SKILL.md: 206/206 ✓
- Hook handlers wired: 24/24 ✓ (12 + 7 + 5)
- Hook scripts on disk: 22 (was 29; intentional −7 documented in Task 2) ✓
- Pytest tests: 154/154 ✓

**All preservation gates pass.**

### Version bump
- `plugins/agent-core/.claude-plugin/plugin.json`: `3.3.2` → `3.4.0`
- `plugins/dev-suite/.claude-plugin/plugin.json`: `3.3.2` → `3.4.0`
- `plugins/science-suite/.claude-plugin/plugin.json`: `3.3.2` → `3.4.0`
- All 3 bumped together in lockstep.
- Re-validated post-bump: 0 errors, 0 warnings.

### Sign-off
The marketplace is ready for Opus 4.7 (1M context) + CC v2.1.113.
See `UPGRADE_REPORT.md` for the formal release report.

---

## Preservation thresholds (from `.baseline-inventory.txt`)

Post-upgrade counts must equal or exceed:

| Component | Baseline | Notes |
|---|---|---|
| Agents | 24 | unchanged |
| Commands (registered) | 14 | unchanged |
| Commands (on disk) | 36 | unchanged |
| Hub skills | 27 | unchanged |
| Sub-skills | 179 | unchanged |
| Total `SKILL.md` | 206 | unchanged |
| Hook handlers (wired) | 24 | unchanged after Task 2 |
| Hook scripts (on disk) | 29 → 22 | **intentional decrease**, documented above |
| Pytest tests | 154 | re-run in Task 4 |
