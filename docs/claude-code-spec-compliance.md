---
orphan: true
---

# Claude Code Plugin Spec Compliance

**Audited against:** Claude Code v2.1.126 (full settings schema)
**Audit method:** Gemini changelog search (4 tool uses) + Codex file audit (1 tool use) + settings schema inspection
**Plugin suite version:** 3.4.1
**Date:** 2026-05-04

---

## Compliance Checklist

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Hook events — deprecated events absent from hooks.json | ✅ Compliant | PreSubagentUse, ExecutionError, PermissionPrompt, ContextOverflow, CostThreshold only appear in README docs, never in hooks.json |
| 2 | Hook events — all active events are valid | ✅ Compliant | All 12 events in use (SessionStart/End, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStart/Stop, PermissionDenied, TaskCreated, TaskCompleted, StopFailure) are confirmed valid in schema |
| 3 | plugin.json required fields present | ✅ Compliant | name, version, description, author, homepage, repository, license, category, keywords present in all 4 manifests |
| 4 | Version consistency across all plugin.json | ✅ Compliant | All at 3.4.1 |
| 5 | Agent frontmatter — only valid fields used | ✅ Compliant | All fields confirmed valid by Codex audit; no unknown fields |
| 6 | SKILL.md frontmatter — required fields only | ✅ Compliant | All skills use only name + description; no unknown fields |
| 7 | Command frontmatter — core fields valid | ✅ Compliant | Standard fields valid; extra fields (category, purpose, tags, external-docs) are unrecognized but benign |
| 8 | No wildcard tool permissions | ✅ Compliant | No `tools: "*"` in any agent |
| 9 | README version references | ✅ Fixed | Updated v2.1.113 → v2.1.126 in agent-core, dev-suite, science-suite READMEs (2026-05-02) |
| 10 | v2.1.113→v2.1.126 delta | ✅ Compliant | Zero breaking changes. No new required fields. Three additive capabilities (see below). |
| 11 | skillListingMaxDescChars compliance (default 1536) | ✅ Compliant | Longest skill description: <1200 chars. Longest agent description: 865 chars (research-expert). Zero truncation risk. |
| 12 | Hook event coverage vs full schema | ✅ Compliant | Using 12/28 valid events — 16 unused events are optional capabilities, not compliance issues |

**Overall: 12/12 compliant. Zero action items required.**

---

## Full Valid Hook Event List (v2.1.126 schema)

Currently used events marked ✅. Unused but available marked ○.

| Event | Status | Notes |
|-------|--------|-------|
| `SessionStart` | ✅ Used | |
| `SessionEnd` | ✅ Used | |
| `PreToolUse` | ✅ Used | Supports `if` conditional filter, `permissionDecision` output |
| `PostToolUse` | ✅ Used | `hookSpecificOutput.updatedToolOutput` now works for ALL tools (v2.1.121) |
| `StopFailure` | ✅ Used | |
| `SubagentStart` | ✅ Used | |
| `SubagentStop` | ✅ Used | |
| `PreCompact` | ✅ Used | |
| `PostCompact` | ✅ Used | |
| `PermissionDenied` | ✅ Used | |
| `TaskCreated` | ✅ Used | |
| `TaskCompleted` | ✅ Used | |
| `Stop` | ○ Available | Fires when Claude stops (including /clear, /compact) |
| `PostToolUseFailure` | ○ Available | Fires when a tool call fails — useful for logging/recovery |
| `PostToolBatch` | ○ Available | Fires after a batch of parallel tool calls completes |
| `UserPromptSubmit` | ○ Available | Fires when user submits a prompt — can inject context |
| `UserPromptExpansion` | ○ Available | |
| `PermissionRequest` | ○ Available | Fires before permission prompt — can auto-allow/deny |
| `Notification` | ○ Available | |
| `Setup` | ○ Available | |
| `TeammateIdle` | ○ Available | Fires when a spawned teammate goes idle |
| `Elicitation` | ○ Available | |
| `ElicitationResult` | ○ Available | |
| `ConfigChange` | ○ Available | Fires when settings change |
| `WorktreeCreate` | ○ Available | Fires when a git worktree is created |
| `WorktreeRemove` | ○ Available | |
| `InstructionsLoaded` | ○ Available | Fires when CLAUDE.md is loaded |
| `CwdChanged` | ○ Available | Fires when working directory changes |
| `FileChanged` | ○ Available | Fires when a watched file changes |

---

## New Capabilities (v2.1.113 → v2.1.126)

| Capability | Version | Description | Potential use in MyClaude |
|-----------|---------|-------------|--------------------------|
| `"type": "mcp_tool"` hook handler | v2.1.118 | Hooks invoke MCP tools directly | PostToolUse → context-mode MCP to auto-index JAX/Julia outputs |
| `themes/` directory in plugin | v2.1.118 | Plugins can ship named custom themes | "scientific-dark" theme for numerical output |
| `PostToolUse.updatedToolOutput` for all tools | v2.1.121 | Rewrite any tool's output before it hits context | Compress verbose JAX debug output in agent-core hooks |
| `claude_code.skill_activated` OTel event | v2.1.126 | Fires on skill activation — observable only | Feed ctx-insight analytics dashboard |
| Hook `if` conditional filter | schema | Permission-rule syntax filter before subprocess spawns | Reduce hook overhead for non-matching commands |
| Hook `once` option | schema | Hook auto-removes after first execution | One-shot setup hooks |
| Hook `async` / `asyncRewake` | schema | Background hook execution; `asyncRewake` can re-wake model | Long-running post-processing without blocking |
| `skillListingMaxDescChars` setting | schema | Caps description length in skill listing (default: 1536) | All 217 MyClaude skills are within limit ✅ |
| `skillListingBudgetFraction` setting | schema | Context fraction for skill listing (default: 0.01) | Could raise slightly given 217 skills compete for 1% |
| `skillOverrides` setting | schema | Per-skill: "name-only", "user-invocable-only", "off" | Could hide rarely-used sub-skills from model routing |

---

## Recommended Next Actions (optional enhancements, not compliance)

1. **Add `$schema` to `.claude/settings.json`** — enables IDE autocomplete for the settings schema
2. **`PostToolUseFailure` hook** — add to agent-core for JAX/Julia tool failures (currently unhandled)
3. **`WorktreeCreate` hook** — auto-set up Julia environments in worktrees (useful for science-suite isolation)
4. **`skillOverrides`** — evaluate hiding general-purpose sub-skills that have been superseded by external plugins
