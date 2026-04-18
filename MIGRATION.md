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

### Deferred (open question)
- `context-specialist` is currently on `sonnet`. With Opus 4.7's 1M context
  window, a case can be made to elevate it to `opus` (the agent owns memory
  systems, vector search, and multi-agent context management). Not changed
  in this sweep — flagged here for a follow-up decision after validator
  benchmarks in Task 4.

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

## Task 3 — (pending)

Skill context budgets, hub→sub-skill reachability.

---

## Task 4 — (pending)

Regression gate, version bump 3.3.2 → 3.4.0, UPGRADE_REPORT.md.

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
