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

Operational detail — preflight and discovery, starting a missing agent kind, prompting and
waiting, reading the two-round reply, degradation, and the hard rules — lives in
`references/herdr-transport.md`. Read it before driving the transport; routing decisions
below do not need it.

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
herdr agent prompt $CODEX "Review the uncommitted changes in this repository, EXCLUDING these paths: .env*, secrets/**, **/*credential*, **/*.pem, **/*.key. Do not open, diff, grep, or quote any file under those paths, and do not report their contents. Focus on bugs, regressions, security risks, missing tests, and unclear assumptions. Report findings first, with file:line references. Do not modify any files." --wait --timeout 600000
```

The exclusion list is **part of the prompt, not a shell filter**. An agent pane runs in your working directory and reads files itself, so `git diff -- ':(exclude)…'` on your side excludes nothing on its side. If the change under review is entirely inside a secret-bearing path, do not send this route at all — see Forced Risk Review.

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

Give Codex the failing command, exact error, relevant diff, and what was already tried — after checking that none of it carries secret material. Stack traces and error strings routinely embed connection strings, tokens, and key paths; redact those before sending, or describe the failure in prose. This avoids wasting tokens on a third local guess.

### Parallel Consensus

Use all three only when the user explicitly requests cross-model consensus or when the decision is high-stakes and the user agrees.

`--wait` blocks until that agent settles, so two `prompt --wait` calls run **sequentially**. To actually parallelise, submit both without `--wait`, then wait on each separately:

```bash
herdr agent prompt $CODEX "<question>" --timeout 60000   # returns after submission
herdr agent prompt $AGY   "<question>" --timeout 60000
herdr agent wait $CODEX --timeout 600000
herdr agent wait $AGY   --timeout 600000
```

Require this structure from each:

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
3. **User approves** → Team Lead submits to both reviewer panes **without** `--wait`, then `herdr agent wait` on each (a `prompt --wait` on the first blocks the second from ever starting — see Parallel Consensus)
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

The five steps — project detection, preflight, adopting what is live,
seeding each pane with its role, and confirming to the user — are in
`references/team-mode-execution.md`.

### team-stop Flow

1. Read `.three-brain/owned-panes.jsonl`. **If it is missing or empty, close nothing** and say so — with no ownership record you cannot prove any pane is yours, and guessing risks killing the user's own agent.
2. For each recorded pane, confirm it still hosts the agent you started (`herdr agent get <pane_id>` — pane IDs are never reused, but the occupant may have been replaced). Tell it to wrap up, then `herdr pane close <pane_id>`.
3. Delete `.three-brain/owned-panes.jsonl` after the closures succeed, so a re-run does not try again.
4. Leave every adopted agent running and say so — it was the user's before this team existed. In the common all-adopted case, team-stop closes nothing.
5. Report:

```text
Team shut down. Closed: {panes you started}. Left running: {agents you adopted}.
```
