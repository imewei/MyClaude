---
name: ai-pair
description: |
  AI Pair Collaboration — orchestrate a persistent three-model team (Claude developer/author + Codex reviewer + Gemini reviewer) for iterative code development or content creation with dual-perspective review. Use when: user types /ai-pair, asks to start a "dev team" or "content team", wants an ongoing multi-model review pipeline for a project, or says "team-stop". Also trigger when the user wants Codex + Gemini to collaboratively review ongoing work through multiple iterations — not one-shot review (use three-brain for that). Requires codex and gemini CLIs; degrades gracefully to Claude-only review if either is absent.
---

# AI Pair Collaboration

Coordinate a persistent, semi-automatic team: one creator (developer or author) + two reviewers (Codex + Gemini). The current Claude session acts as Team Lead.

Different AI models look at completely different dimensions. Codex catches bugs, security issues, and edge cases. Gemini surfaces architectural and readability concerns. Running both maximizes coverage without relying on a single model's blind spots.

## Commands

```bash
/ai-pair dev-team [project]     # Code team: developer + codex-reviewer + gemini-reviewer
/ai-pair content-team [topic]   # Content team: author + codex-reviewer + gemini-reviewer
/ai-pair team-stop              # Shut down team and clean up
```

## Team Roles

| Role           | Dev Team                                    | Content Team                                    |
|----------------|---------------------------------------------|-------------------------------------------------|
| Creator        | developer — implements features/fixes       | author — writes articles, scripts, newsletters  |
| Codex reviewer | bugs, security, concurrency, edge cases     | logic, accuracy, structure, fact-checking       |
| Gemini reviewer| architecture, design patterns, alternatives | readability, engagement, style, audience fit    |

## Workflow Loop (Semi-Automatic)

1. **User assigns task** → Team Lead routes to developer/author
2. **Creator completes** → Team Lead shows result to user
3. **User approves** → Team Lead dispatches both reviewers in parallel
4. **Reviewers report** → Team Lead consolidates and presents:
   ```
   ## Codex Review
   {findings}
   ## Gemini Review
   {findings}
   ```
5. **User decides** → "Revise" (loop to step 1) or "Pass" (next task or end)

User controls every transition. No autonomous loops.

## Execution Steps

### 1. Project Detection
1. Explicitly specified → use as-is
2. CWD is inside a project → extract project name from path
3. Ambiguous → ask the user

### 2. Pre-flight CLI Check

```bash
command -v codex && codex --version || echo "CODEX_MISSING"
command -v gemini && gemini --version || echo "GEMINI_MISSING"
```

If either CLI is missing: warn user, offer degraded mode (Claude-only review, clearly labeled) or abort.

### 3. Create Team

```
TeamCreate: team_name = "{project}-dev" or "{topic}-content"
```

### 4. Create Initial Tasks

```
TaskCreate: "Awaiting task assignment" — creator, status: pending
TaskCreate: "Awaiting review" — codex-reviewer, status: pending, blockedBy: task-1
TaskCreate: "Awaiting review" — gemini-reviewer, status: pending, blockedBy: task-1
```

### 5. Launch Agents

Read `references/agent-prompts.md` for the startup prompt templates. The **CLI Invocation Protocol** block below must be included verbatim in each reviewer agent's startup prompt.

Spawn 3 agents via Agent tool with `subagent_type: "general-purpose"` and `mode: "bypassPermissions"` (required — reviewers must execute external CLI commands and read project files).

### 6. Confirm to User

```
Team ready.
Team: {team_name}  Type: {Dev / Content}
Members: developer/author ✓  codex-reviewer ✓  gemini-reviewer ✓
Awaiting your first task.
```

## CLI Invocation Protocol

Include this block verbatim in each reviewer agent's startup prompt:

---
**[Timeout]** All Bash tool calls to codex/gemini MUST set `timeout: 600000` (10 min). External CLIs need 10-15 s to load plus model reasoning time — the default 2-min timeout always fails.

**[File-based content passing]** Write content to a unique temp file before calling the CLI:
```bash
REVIEW_FILE=$(mktemp /tmp/review-XXXXXX.txt)
# write content to $REVIEW_FILE, then reference it in the CLI prompt
rm -f $REVIEW_FILE  # cleanup after capturing output
```
Never pipe via stdin — pipes can truncate or mishandle large inputs.

**[Codex degradation on timeout/failure]** Retry in order: xhigh → high → medium → low → Claude fallback. Append reasoning effort hint to prompt on each retry. Report the current degradation level to team-lead.

**[Gemini degradation]** Retry in order: simplify prompt → reduce analysis dimensions → Claude fallback.

**[Hard rules]** Never skip the CLI call. Never silently self-review. If the CLI is not found, report immediately. Only label `[Claude Fallback — [CLI] four retries all failed]` after all retries are exhausted. Set `NO_COLOR=1` if output has ANSI artifacts.

---

## team-stop Flow

1. `SendMessage` shutdown_request to all agents; wait for confirmations
2. `TeamDelete` to clean up team resources
3. Report: `Team shut down. Closed: developer/author, codex-reviewer, gemini-reviewer. Resources cleaned up.`
