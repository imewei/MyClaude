---
disable-model-invocation: true
name: three-brain
description: |
  Route work between Claude, Codex, and Agy as live Herdr panes — either as a single one-shot second opinion, or as a persistent semi-automatic team that stays alive across a multi-round project. Use Route mode (default, one-shot) for second-opinion reviews of Claude's own work, high-risk code paths (auth/billing/migrations/secrets/infra), repeated failures on the same bug, video/audio/PDF/image inspection, long-context repository or document scans, and explicit requests like "ask Codex", "ask Agy", "second opinion", "sanity check", "review your work", or "use all three". Use Team mode (persistent) when the user asks to start a "dev team" or "content team", wants an ongoing multi-model review pipeline for a project, or asks to stop/shut down such a team — also trigger for "pair with codex and agy" or requests for Codex + Agy to collaboratively review ongoing work through multiple iterations. Prefer not to trigger for ordinary Q&A, simple edits, or reviewing user-authored non-code drafts unless the user explicitly asks for another model.
compatibility: Requires Herdr — this skill runs only inside a Herdr-managed pane (`HERDR_ENV=1`) and drives Codex and Agy as named Herdr agents. See https://herdr.dev/docs/agent-skill/. Outside Herdr it stops at the preflight gate.
---

# Three-Brain

Use Claude as the driver. Call Codex or Agy only when their different strengths materially improve the result — Codex catches bugs, security issues, concurrency, and edge cases; Agy surfaces architecture, design-pattern, and readability concerns, plus multimodal and long-context perception Claude can't do locally. Keep routes bounded, cite evidence from returned output, and preserve the user's workflow.

Both modes reach the other models the same way: as **live, named Herdr agents in sibling panes**, prompted through `herdr agent prompt` and read through `herdr agent read`. A pane keeps its context across rounds, so the second review of a file costs far less than the first.

> **This skill is slash-only** (`disable-model-invocation: true`). Nothing here self-fires — including Forced Risk Review and the Failure Counter below. Those describe what to do *once someone has typed* `/dev-suite:three-brain`; they are not ambient triggers.

## Two Modes

| Mode | Shape | Trigger |
|---|---|---|
| **Route** (default) | One-shot: Claude prompts a Codex/Agy pane for a single question, then integrates the answer | "ask Codex", "second opinion", "sanity check", high-risk path touched, repeated failure, multimodal/long-context input, explicit "use all three" |
| **Team** | Persistent: the live `claude`, `codex`, and `agy` panes take creator and reviewer roles across many tasks | "start a dev team", "start a content team", "pair on this project", "team-stop" |

The dividing line is duration, not model choice — both modes drive the same agent kinds. A one-off "does this look right?" is Route. "Keep reviewing everything I build for this project" is Team. Both modes target the same live panes, so a Route-mode trigger during an active Team just prompts the reviewer that is already there. Never start a second agent of a kind that is already running in this directory.

---

## Transport: Herdr

Herdr is the only transport. The Team Lead (this Claude session) is the sole caller — it owns every pane and every prompt. Do not delegate `herdr` commands to subagents: `--current` resolves against the *calling* pane, and a subagent is not a Herdr-managed pane, so its splits land in the wrong place and it cannot answer a `blocked` approval dialog.

### 1. Preflight and discovery

```bash
test "${HERDR_ENV:-}" = 1
```

If this fails, say you are not running inside Herdr and stop. Do not fall back to shelling out to `codex` or `agy` directly — this skill has no non-Herdr path.

**Normally Claude, Codex, and Agy are already running in the session.** Discovery is the main path; starting an agent is the exception. Find what is live and hand it work:

```bash
herdr agent list
```

Each entry is an `AgentInfo`. Four fields decide everything:

| Field | Use |
|---|---|
| `agent` | The **kind** — `codex`, `agy`, `claude`. Match on this, never on the name |
| `name` | The assigned name, **or `null`** for an agent the user started themselves |
| `pane_id` | Always present. The fallback target, and the only target for an unnamed agent |
| `agent_status` | `idle` / `working` / `blocked` / `done` / `unknown` |

Select by kind, then by working directory — one Herdr session often hosts agents for several repos, and `cwd` / `foreground_cwd` are what keep a review pointed at the right one:

```bash
herdr agent list | jq -r --arg cwd "$PWD" '
  .result.agents[]
  | select(.agent == "codex")
  | select((.cwd // .foreground_cwd // "") | startswith($cwd))
  | .name // .pane_id'
```

Use whatever that prints as the target for every later command — **agent commands accept a unique live name or the pane ID hosting the agent**, so an unnamed pane is fully usable as `w1:p3`. Do not rename someone else's agent to make it fit a naming scheme; only name agents you start.

Match `agent_status` before sending:

| Status | Do |
|---|---|
| `idle` / `done` | Ready. Prompt it |
| `working` | Busy on someone else's turn. Wait (`herdr agent wait <target> --timeout …`) or pick another; do not queue a prompt on top |
| `blocked` | An approval dialog is open. Read it, show the user, ask. Never auto-answer |
| `unknown` | Herdr can't classify it. This is **not** proof it is free — `agent read` before assuming |

### 2. Start an agent only when its kind is missing

If no agent of the needed kind is live in this working directory, create one — and record that you created it, because team-stop closes only those.

```bash
herdr pane split --current --direction right --cwd "$PWD" --no-focus
# read the new id from .result.pane.pane_id
herdr agent start cdx-<slug> --kind codex --pane <returned-pane-id>
herdr agent start agy-<slug> --kind agy   --pane <returned-pane-id>
```

Split a wide pane `right` and a narrow or tall pane `down`; check with `herdr pane layout --pane "$HERDR_PANE_ID"`. Always `--no-focus` — the user's focus stays where it was.

**Name scoping applies only to agents you start.** Names are unique among live agents on the server, so a bare `codex-reviewer` collides the moment a second team starts. Suffix a short project slug and stay inside `[a-z][a-z0-9_-]{0,31}`: `cdx-payments`, `agy-payments`, `dev-payments`.

`agent start` requires a pane already sitting at an interactive shell prompt, and never creates layout itself. It returns `agent_not_ready` if the agent is blocked during startup — the name still works for `agent read` and `agent send-keys`, so inspect before re-issuing.

### 3. Prompt and wait

```bash
herdr agent prompt $CODEX "<prompt text>" --wait --timeout 600000
```

`--wait` blocks until the first settled `idle`, `done`, or `blocked`. Do not add `--until` for ordinary work; it is only for waiting on a state-specific condition such as an already-running agent asking for input.

Prompt text goes straight through — no temp file, no stdin pipe, no shell quoting of a diff into an argv. To review a diff, tell the agent where to look (`Review the uncommitted changes in this repo`); these are agentic CLIs sitting in the working directory and they read files themselves.

Distinguish the failure returns, because they mean different things:

| Return | Meaning | Do |
|---|---|---|
| `agent_blocked` | An approval/question dialog was already open; **nothing was sent** | `agent read` the dialog, show it to the user, ask. Never auto-answer |
| `agent_prompt_stalled` | Submitted, but no `working`/`blocked` activity within 5 s | Inspect with `agent get` before deciding; do not blind-resubmit |
| `timeout` | Your timeout expired (submission time counts toward it) | Same — the prompt may well have landed |

A timeout or stall is not proof the prompt was never delivered. Re-sending a review prompt twice gets you two turns of work and a confused pane.

### 4. Read the result — expect two rounds

```bash
herdr agent read $CODEX --source recent-unwrapped --lines 200
```

`recent-unwrapped` joins soft wraps and is the right source for transcripts.

**Codex and Agy render on the terminal's alternate screen, so this often returns a truncated response, and raising `--lines` cannot recover it** — rows that leave the alternate screen never enter Herdr's host scrollback. That is a property of the TUI, not a misconfiguration, so budget for the second round rather than treating it as an error:

```bash
# only after a first read came back truncated
herdr agent prompt $CODEX "Write your complete response as Markdown to a file under /tmp and reply with only the file path." --wait --timeout 300000
herdr agent read $CODEX --source recent-unwrapped --lines 20   # capture just the path
```

Then Read that file directly. Do **not** ask for file output in the initial prompt — first try the direct read, since a short answer comes back whole and one round is cheaper than two.

### 5. Degradation

**Codex effort ladder** — xhigh → high → medium → low. Retry by **re-prompting the same live pane** with a reasoning-effort hint appended, never by restarting the agent: effort is fixed at `agent start` via `-- <agent-args>`, so a restart throws away the context the pane has already built.

**Agy** — simplify prompt → reduce analysis dimensions.

Only after a ladder is exhausted (Codex: four rungs, Agy: two) label the result `[Claude Fallback — <agent> retries all failed]`.

### 6. Hard rules

- Never skip the agent prompt and review it yourself. A same-model review is the one thing this skill exists to avoid.
- Never auto-answer a `blocked` dialog. Surface it and ask.
- Never close a pane, tab, or workspace you did not create. If you reused an existing agent, it is not yours to shut down.
- Never run bare `herdr` (it launches the TUI) and never `herdr server stop`.
- Parse IDs out of the JSON responses. Do not guess `w1:p2` from position.
- Use `--format ansi` only when colour is itself the evidence.

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

### Codex Routes

`$CODEX` and `$AGY` below are the handles discovered in Transport step 1 — a live agent name, or a `pane_id` when the user's agent is unnamed.

Use Codex for independent code review, adversarial reasoning, and rescue after repeated failures. The pane is already in the working directory, so point it at the change rather than shipping content:

```bash
herdr agent prompt $CODEX "Review the uncommitted changes in this repository. Focus on bugs, regressions, security risks, missing tests, and unclear assumptions. Report findings first, with file:line references. Do not modify any files." --wait --timeout 600000
```

For a specific range, name it in the prompt (`Review commit <SHA>`, `Review this branch against main`). Ask for findings, evidence, and recommended fixes; do not ask for a rewrite unless that is the task.

After Codex returns:

- Integrate only findings supported by evidence.
- If there are no actionable findings, say so.
- End the response with `(Routed via three-brain -> Codex review.)` when the route was triggered by this skill.

### Agy Routes

Use Agy for perception-heavy and long-context tasks. Ask for structured evidence, not a flat summary. Agy reads files itself once given a path — name the exact path in the prompt text.

```bash
# video
herdr agent prompt $AGY "Read and analyze the video at /path/to/video.mp4. Return timestamped findings as [MM:SS] event. Cover visible content, on-screen text, speaker/action changes, transitions, and notable issues. Cap at 800 words." --wait --timeout 600000

# audio
herdr agent prompt $AGY "Read and analyze the audio at /path/to/audio.wav. Return timestamped findings as [MM:SS] event, including speakers if distinguishable, key claims, action items, and uncertainty. Cap at 800 words." --wait --timeout 600000

# document
herdr agent prompt $AGY "Read /path/to/file.pdf. Extract key claims, tables, chart findings, contradictions, and action items with page-number citations. Cap at 1000 words." --wait --timeout 600000

# repository scan
herdr agent prompt $AGY "Search /path/or/directory for every place related to <topic>. Return file:line citations, short purpose, and confidence. Avoid broad summaries." --wait --timeout 600000
```

If the target lives outside the pane's working directory, start that pane with `--cwd` at a parent, or pass Agy's own `--add-dir` after `--` at `agent start`.

Prefer file, page, or timestamp citations in every Agy prompt.

### Forced Risk Review

Once this skill is active, route to Codex when active work touches high-risk targets:

- `src/auth/**`, `**/*OAuth*`
- `src/billing/**`, `**/*Stripe*`
- `migrations/**`
- `deploy/**`, `infra/**`
- `.env*`, `secrets/**` — describe the change in **prose only, never content**.
  These paths are flagged for review precisely because they hold secrets.
  A Herdr agent pane runs in your working directory and can open any file it
  is pointed at, so "don't paste the diff" is not sufficient here: do not
  name these paths to the agent at all. Summarize what changed instead.
- `policy/**`, permissions, roles, or ACL logic

Announce forced routes in one short line before prompting so the user can interrupt:

```text
[three-brain] routing to Codex review - risk path: src/auth/
```

Do not announce when the user explicitly asked for the route.

### Failure Counter

If the same command, test, or bug fails twice on the same code path after Claude has attempted a fix, route to Codex rescue:

```text
[three-brain] routing to Codex rescue - same failure repeated twice
```

Give Codex the failing command, exact error, relevant diff, and what was already tried. This avoids wasting tokens on a third local guess.

### Parallel Consensus

Use all three only when the user explicitly requests cross-model consensus or when the decision is high-stakes and the user agrees. Prompt both panes with the same question — they run concurrently, so send both before waiting on either — and require this structure:

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
- Send the smallest useful context: name a diff or path rather than pasting a dump.
- Cap model outputs in the prompt when the route is exploratory.
- Prefer citations and findings over rewrites.
- Keep Claude responsible for final integration, user communication, and filesystem changes.
- Reuse a warm pane across rounds — its retained context is the main saving Herdr buys over one-shot invocation.
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

Coordinate a persistent, semi-automatic team of three Herdr panes: one creator (`claude`) plus two reviewers (`codex`, `agy`). The current Claude session is the Team Lead and the only caller. This is a skill, not a slash command beyond the entry point — there is nothing further to type.

| Request | Setup |
|---------|-------|
| "start a dev team", "pair on this project" | **Dev team** — adopt the live `claude` + `codex` + `agy` agents; start only a missing kind |
| "start a content team", "help me write this with reviewers" | **Content team** — same three agents, content-focused prompts |
| "stop the team", "we're done with the team" | **Shut down** — see team-stop flow below |

### Team Roles

| Role           | Dev Team                                    | Content Team                                    |
|----------------|---------------------------------------------|-------------------------------------------------|
| Creator        | developer — implements features/fixes       | author — writes articles, scripts, newsletters  |
| Codex reviewer | bugs, security, concurrency, edge cases     | logic, accuracy, structure, fact-checking       |
| Agy reviewer   | architecture, design patterns, alternatives | readability, engagement, style, audience fit    |

### Workflow Loop (Semi-Automatic)

1. **User assigns task** → Team Lead prompts the creator pane
2. **Creator completes** → Team Lead reads the pane and shows the result to the user
3. **User approves** → Team Lead prompts both reviewer panes (send both, then wait — they work concurrently)
4. **Reviewers report** → Team Lead reads both panes and consolidates, naming the effort/degradation level each landed on so a `low`-effort retry never reads like an `xhigh` first pass:
   ```
   ## Codex Review [effort: {level} — {N} retries]
   {findings}
   ## Agy Review [degradation: {level}]
   {findings}
   ```
5. **User decides** → "Revise" (loop to step 1) or "Pass" (next task or end)

User controls every transition. No autonomous loops.

### Execution Steps

#### 1. Project detection

1. Explicitly specified → use as-is
2. CWD is inside a project → derive the name, lowercase it, and cut a `<slug>` that keeps every agent name inside 32 characters
3. Ambiguous → ask the user

#### 2. Preflight

```bash
test "${HERDR_ENV:-}" = 1 || echo "NOT_IN_HERDR"
herdr agent list
```

Not in Herdr → say so and stop. Otherwise resolve one handle per kind (`claude`, `codex`, `agy`) scoped to this working directory, exactly as in Transport step 1. Adopted agents are **not yours to close at team-stop** — only ones you start are.

#### 3. Adopt what is live; start only what is missing

For each of the three kinds, take the handle discovery returned. Start a pane **only** for a kind with no live agent in this working directory, and record which ones you created — team-stop closes only those.

```bash
herdr pane split --current --direction right --cwd "$PWD" --no-focus
herdr agent start dev-<slug> --kind claude --pane <id-from-.result.pane.pane_id>
```

Repeat per missing kind (`--kind codex`, `--kind agy`), alternating `right`/`down` per `herdr pane layout` rather than splitting the same direction three times, which leaves unusably narrow columns.

In the common case all three are already running and this step does nothing.

#### 4. Seed each pane with its role

Read `references/agent-prompts.md` for the role-seeding text. Seed a pane you just started. For an **adopted** agent, `herdr agent read <target> --source recent-unwrapped --lines 40` first — it may be mid-conversation on unrelated work, in which case say so and ask the user before repurposing it. Send the seed with `herdr agent prompt <name> "<role text>" --wait --timeout 300000`. There are no reviewer subagents to spawn — the panes *are* the reviewers, and Herdr's own `idle`/`working`/`blocked`/`done` states replace the dispatcher bookkeeping a wrapper agent used to do.

#### 5. Confirm to user

```
Team ready (Herdr).
Team: {slug}  Type: {Dev / Content}
claude → {handle}  codex → {handle}  agy → {handle}
Adopted: {list}   Started this session: {list}
Awaiting your first task.
```

### team-stop Flow

1. For each agent **you started this session**, tell it to wrap up, then `herdr pane close <pane_id>` on the pane you split for it.
2. Leave every adopted agent running and say so — it was the user's before this team existed. In the common all-adopted case, team-stop closes nothing.
3. Report:

```text
Team shut down. Closed: {panes you started}. Left running: {agents you adopted}.
```
