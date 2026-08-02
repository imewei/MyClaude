---
name: three-brain
description: |
  Route work between Claude, Codex, and Agy — either as a single one-shot second opinion, or as a persistent semi-automatic team that stays alive across a multi-round project. Use Route mode (default, one-shot) for second-opinion reviews of Claude's own work, high-risk code paths (auth/billing/migrations/secrets/infra), repeated failures on the same bug, video/audio/PDF/image inspection, long-context repository or document scans, and explicit requests like "ask Codex", "ask Agy", "second opinion", "sanity check", "review your work", or "use all three". Use Team mode (persistent) when the user asks to start a "dev team" or "content team", wants an ongoing multi-model review pipeline for a project, or asks to stop/shut down such a team — also trigger for "pair with codex and agy" or requests for Codex + Agy to collaboratively review ongoing work through multiple iterations. Prefer not to trigger for ordinary Q&A, simple edits, or reviewing user-authored non-code drafts unless the user explicitly asks for another model.
compatibility: Requires `codex` for Codex routes and `agy` for Agy routes. Falls back gracefully — to Claude-only review in Route mode, to a clearly-labeled degraded team in Team mode — when either CLI is missing.
---

# Three-Brain

Use Claude as the driver. Call Codex or Agy only when their different strengths materially improve the result — Codex catches bugs, security issues, concurrency, and edge cases; Agy surfaces architecture, design-pattern, and readability concerns, plus multimodal and long-context perception Claude can't do locally. Keep routes bounded, cite evidence from returned output, and preserve the user's workflow.

## Two Modes

| Mode | Shape | Trigger |
|---|---|---|
| **Route** (default) | One-shot: Claude calls Codex/Agy directly for a single question, then integrates the answer | "ask Codex", "second opinion", "sanity check", high-risk path touched, repeated failure, multimodal/long-context input, explicit "use all three" |
| **Team** | Persistent: a creator (developer/author) plus a codex-reviewer and agy-reviewer stay alive across many tasks via `TeamCreate`/`TaskCreate` | "start a dev team", "start a content team", "pair on this project", "team-stop" |

The dividing line is duration, not model choice — both modes call the same two CLIs. A one-off "does this look right?" is Route. "Keep reviewing everything I build for this project" is Team. If a Team is already active (see below) and a Route-mode trigger fires on the same project, prefer routing through the existing team's reviewer dispatch over spawning a second, uncoordinated Codex/Agy call — two independent reviews of the same change waste tokens and can disagree with no one to reconcile them.

## CLI Invocation Protocol

Both modes call the same two CLIs the same way. Route mode: Claude runs these Bash calls itself. Team mode: paste this whole section verbatim into each reviewer subagent's startup prompt, since a subagent needs the rules self-contained.

**[Timeout]** All Bash tool calls to codex/agy MUST set `timeout: 600000` (10 min). External CLIs need 10-15 s to load plus model reasoning time — the default 2-min timeout always fails.

**[Bypass flags]** Codex calls use `codex exec --dangerously-bypass-approvals-and-sandbox` (add `--skip-git-repo-check` when piping a diff outside a confirmed repo, or `review --commit <SHA>` / `--base <branch>` / `--uncommitted` in place of a free-form prompt). Agy calls use `agy --dangerously-skip-permissions --print-timeout 9m -p` (under the 10-min Bash timeout). Both flags are required — without them the CLI stops on an interactive confirmation prompt that never resolves in a non-interactive call, and it hangs to timeout instead of failing fast.

**[Agy has no @file syntax]** Agy is agentic — it reads files itself once given a path. Name the exact path in the prompt text ("Read and analyze the video at /path/to/video.mp4...") rather than appending `@path` the way gemini did. Use `--add-dir <path>` to widen its workspace when the target lives outside the current directory.

**[File-based content passing]** For arbitrary code/content (not a file that already exists on disk), write it to a unique temp file before calling the CLI:
```bash
REVIEW_FILE=$(mktemp /tmp/review-XXXXXX.txt)
# write content to $REVIEW_FILE, then reference its path in the CLI prompt
rm -f $REVIEW_FILE  # cleanup after capturing output
```
Never pipe via stdin — pipes can truncate or mishandle large inputs.

**[Codex degradation on timeout/failure]** Retry in order: xhigh → high → medium → low → Claude fallback. Append a reasoning-effort hint to the prompt on each retry.

**[Agy degradation]** Retry in order: simplify prompt → reduce analysis dimensions → Claude fallback.

**[Hard rules]** Never skip the CLI call. Never silently self-review. If the CLI is not found, report immediately. Only label a result `[Claude Fallback — [CLI] retries all failed]` after that CLI's ladder is exhausted (Codex: four, Agy: two). Set `NO_COLOR=1` if output has ANSI artifacts.

---

## Route Mode

### Fast Decision Table

| Situation | Route | Why |
| --- | --- | --- |
| User asks to review/check/sanity-check work Claude just produced | Codex review | Avoid same-model blind spots |
| Active edits touch auth, billing, migrations, deployment, secrets, permissions, or infra | Codex review | High blast radius deserves independent scrutiny |
| Same test, command, or bug fails twice on the same path | Codex rescue | Stop repeating the same local approach |
| User provides video, audio, image, scanned PDF, charts, or visual layout to inspect | Agy analysis | Use stronger multimodal perception |
| User asks for broad repository/document discovery over lots of files | Agy long-context scan | Reduce token-heavy local reading |
| User explicitly says ask Codex/Agy/all three/cross-check | Requested model(s) | Follow the user's routing request |
| Ordinary explanation, writing, small edit, local file operation, or user-authored tone review | Claude direct | Extra routing adds cost without clear value |

When uncertain about review of Claude's own output, route to Codex. When uncertain about ordinary user-authored content, stay direct unless the user asked for a second model.

### Startup Check

Run this once per session, only before the first route that needs the tool:

```bash
codex --version 2>&1 | head -1
agy --version 2>&1 | head -1
```

If a CLI is missing, tell the user once and continue with the available route or Claude direct. Do not recheck every turn.

### Codex Routes

Use Codex for independent code review, adversarial reasoning, and rescue after repeated failures.

For tracked repository changes:

```bash
git diff --stat
# Exclude secret-bearing paths from content — see Forced Risk Review below.
# These are exactly the files that must never leave the machine, so the
# trigger list that flags them for review must not also be what pipes their
# contents to an external CLI.
git diff -- . ':(exclude).env*' ':(exclude)secrets/**' \
  | codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "Review this change. Focus on bugs, regressions, security risks, missing tests, and unclear assumptions. Return findings first with file/line references when possible."
```

For untracked files or non-code output, pipe only the relevant file(s) or excerpt. Keep the prompt narrow. Ask Codex for findings, evidence, and recommended fixes; do not ask it to rewrite everything unless that is the task.

After Codex returns:

- Integrate only findings supported by evidence.
- If there are no actionable findings, say so.
- End the response with `(Routed via three-brain -> Codex review.)` when the route was triggered by this skill.

### Agy Routes

Use Agy for perception-heavy and long-context tasks. Ask for structured evidence, not a flat summary.

Video:

```bash
agy --dangerously-skip-permissions --print-timeout 9m -p "Read and analyze the video at /path/to/video.mp4. Return timestamped findings as [MM:SS] event. Cover visible content, on-screen text, speaker/action changes, transitions, and notable issues. Cap at 800 words."
```

Audio:

```bash
agy --dangerously-skip-permissions --print-timeout 9m -p "Read and analyze the audio at /path/to/audio.wav. Return timestamped findings as [MM:SS] event, including speakers if distinguishable, key claims, action items, and uncertainty. Cap at 800 words."
```

PDF or document:

```bash
agy --dangerously-skip-permissions --print-timeout 9m -p "Read /path/to/file.pdf. Extract key claims, tables, chart findings, contradictions, and action items with page-number citations. Cap at 1000 words."
```

Repository or large directory scan:

```bash
agy --dangerously-skip-permissions --print-timeout 9m -p "Search /path/or/directory for every place related to <topic>. Return file:line citations, short purpose, and confidence. Avoid broad summaries."
```

Prefer file, page, or timestamp citations in every Agy prompt.

### Forced Risk Review

Route to Codex when active work touches high-risk targets:

- `src/auth/**`, `**/*OAuth*`
- `src/billing/**`, `**/*Stripe*`
- `migrations/**`
- `deploy/**`, `infra/**`
- `.env*`, `secrets/**` — send **filenames and `git diff --stat` only, never
  content**. These paths are flagged for review precisely because they hold
  secrets; sending their diff content to an external CLI is the one thing
  this rule must not do. Describe the change in prose instead.
- `policy/**`, permissions, roles, or ACL logic

Announce forced routes in one short line before calling the tool so the user can interrupt:

```text
[three-brain] routing to Codex review - risk path: src/auth/
```

Do not announce when the user explicitly asked for the route. If a Team is currently active for this project, send the finding to the team's reviewers via the normal workflow loop instead of firing this route standalone — see "Two Modes" above.

### Failure Counter

Track repeated failures deterministically. If the same command, test, or bug fails twice on the same code path after Claude has attempted a fix, route to Codex rescue:

```text
[three-brain] routing to Codex rescue - same failure repeated twice
```

Give Codex the failing command, exact error, relevant diff, and what was already tried. This avoids wasting tokens on a third local guess.

### Parallel Consensus

Use all three only when the user explicitly requests cross-model consensus or when the decision is high-stakes and the user agrees. Ask each model the same question and require this structure:

```text
Recommendation: <one line>
Blocking risks: <bullets>
Assumptions: <bullets>
Confidence: low / medium / high
Tests required: <bullets>
```

Compare the answers by evidence. Do not average opinions.

### Token And Stability Rules

- Route late enough to have a concrete artifact, error, file, or question.
- Send the smallest useful context: diffs over whole files, exact files over whole repos, bounded excerpts over dumps.
- Cap model outputs in the prompt when the route is exploratory.
- Prefer citations and findings over rewrites.
- Keep Claude responsible for final integration, user communication, and filesystem changes.
- If a route fails, report the failure briefly and continue with the best available local approach.

### Output Filing

When a route produces durable output, write it under:

```text
./three-brain-out/<YYYY-MM-DD>-<short-slug>/
```

Use only the files that apply:

- `input.txt` - user request or routed subquestion
- `codex-review.md` - Codex findings
- `agy-analysis.md` - Agy findings
- `consensus.md` - cross-model comparison
- `log.md` - run-specific summary

Append one root-level line to `./three-brain-out/log.md` for every route:

```text
[YYYY-MM-DD HH:MM] route=<codex-review|codex-rescue|agy-analysis|consensus> target=<short target> status=<ok|partial|failed> duration=<seconds>s outputs=<N> summary="<plain-language result>"
```

Example:

```text
[2026-05-03 04:52] route=codex-review target=auth-middleware status=ok duration=42s outputs=1 summary="Found one missing test and no blocking security issue."
```

---

## Team Mode

Coordinate a persistent, semi-automatic team: one creator (developer or author) + two reviewers (Codex + Agy). The current Claude session acts as Team Lead. This is a skill, not a slash command — there is nothing to type. Route here when the user asks for:

| Request | Setup |
|---------|-------|
| "start a dev team", "pair on this project" | **Dev team** — developer + codex-reviewer + agy-reviewer |
| "start a content team", "help me write this with reviewers" | **Content team** — author + codex-reviewer + agy-reviewer |
| "stop the team", "we're done with the team" | **Shut down** — see team-stop flow below |

### Team Roles

| Role           | Dev Team                                    | Content Team                                    |
|----------------|---------------------------------------------|-------------------------------------------------|
| Creator        | developer — implements features/fixes       | author — writes articles, scripts, newsletters  |
| Codex reviewer | bugs, security, concurrency, edge cases     | logic, accuracy, structure, fact-checking       |
| Agy reviewer   | architecture, design patterns, alternatives | readability, engagement, style, audience fit    |

### Workflow Loop (Semi-Automatic)

1. **User assigns task** → Team Lead routes to developer/author
2. **Creator completes** → Team Lead shows result to user
3. **User approves** → Team Lead dispatches both reviewers in parallel
4. **Reviewers report** → Team Lead consolidates and presents, with the
   actual effort/degradation level each reviewer landed on in its own
   heading — a `low`-effort retry must not read as indistinguishable from an
   `xhigh` first-pass review:
   ```
   ## Codex Review [effort: {level} — {N} retries]
   {findings}
   ## Agy Review [degradation: {level}]
   {findings}
   ```
5. **User decides** → "Revise" (loop to step 1) or "Pass" (next task or end)

User controls every transition. No autonomous loops.

### Execution Steps

#### 1. Project Detection
1. Explicitly specified → use as-is
2. CWD is inside a project → extract project name from path
3. Ambiguous → ask the user

#### 2. Pre-flight CLI Check

```bash
command -v codex && codex --version || echo "CODEX_MISSING"
command -v agy && agy --version || echo "AGY_MISSING"
```

If either CLI is missing: warn user, offer degraded mode (Claude-only review, clearly labeled) or abort.

#### 3. Create Team

```
TeamCreate: team_name = "{project}-dev" or "{topic}-content"
```

#### 4. Create Initial Tasks

```
TaskCreate: "Awaiting task assignment" — creator, status: pending
TaskCreate: "Awaiting review" — codex-reviewer, status: pending, blockedBy: task-1
TaskCreate: "Awaiting review" — agy-reviewer, status: pending, blockedBy: task-1
```

#### 5. Launch Agents

Read `references/agent-prompts.md` for the startup prompt templates. The **CLI Invocation Protocol** section above must be included verbatim in each reviewer agent's startup prompt.

Spawn 3 agents via Agent tool with `subagent_type: "general-purpose"`. Do not
set `mode: "bypassPermissions"` — a skill is not a consent channel, and
reviewers only need Bash (to shell out to codex/agy) and Read (project
files), both of which the normal permission system already grants or prompts
for. Let the user's own permission settings govern these agents like any other.

#### 6. Confirm to User

```
Team ready.
Team: {team_name}  Type: {Dev / Content}
Members: developer/author ✓  codex-reviewer ✓  agy-reviewer ✓
Awaiting your first task.
```

### team-stop Flow

1. `SendMessage` shutdown_request to all agents; wait for confirmations
2. `TeamDelete` to clean up team resources
3. Report: `Team shut down. Closed: developer/author, codex-reviewer, agy-reviewer. Resources cleaned up.`
