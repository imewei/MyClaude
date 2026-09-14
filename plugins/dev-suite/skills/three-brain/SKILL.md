---
name: three-brain
description: |
  Route work between Claude, Codex, and Agy as live Herdr panes — either as a single one-shot second opinion, or as a persistent semi-automatic team that stays alive across a multi-round project. Use Route mode (default, one-shot) for second-opinion reviews of Claude's own work, high-risk code paths (auth/billing/migrations/secrets/infra), repeated failures on the same bug, video/audio/PDF/image inspection, long-context repository or document scans, and explicit requests like "ask Codex", "ask Agy", "second opinion", "sanity check", "review your work", or "use all three". Use Team mode (persistent) when the user asks to start a "dev team" or "content team", wants an ongoing multi-model review pipeline for a project, or asks to stop/shut down such a team — also trigger for "pair with codex and agy" or requests for Codex + Agy to collaboratively review ongoing work through multiple iterations. Prefer not to trigger for ordinary Q&A, simple edits, or reviewing user-authored non-code drafts unless the user explicitly asks for another model.
compatibility: Requires Herdr — this skill runs only inside a Herdr-managed pane (`HERDR_ENV=1`) and drives Codex and Agy as named Herdr agents. See https://herdr.dev/docs/agent-skill/. Outside Herdr it stops at the preflight gate.
---

# Three-Brain

Use Claude as the driver. Call Codex or Agy only when their different strengths materially improve the result — Codex catches bugs, security issues, concurrency, and edge cases; Agy surfaces architecture, design-pattern, and readability concerns, plus multimodal and long-context perception Claude can't do locally. Keep routes bounded, cite evidence from returned output, and preserve the user's workflow.

Both modes reach the other models the same way: as **live, named Herdr agents in sibling panes**, prompted through `herdr agent prompt` and read through `herdr agent read`. A pane keeps its context across rounds, so the second review of a file costs far less than the first.

> **Loading.** Claude may load this skill directly via the Skill tool when the description matches, or the user may type `/dev-suite:three-brain`. Forced Risk Review and the Failure Counter below describe what to do *once the skill is loaded*; they are not ambient triggers outside it.

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

The per-model recipes — what to send Codex for a review or a rescue, what to send Agy for
multimodal analysis or a long-context scan, and what each returns — are in
`references/model-routes.md`. The table above is enough to pick the route.

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

Roles for each pane, the semi-automatic workflow loop, the execution steps, and the
team-stop flow are in `references/team-mode.md` and `references/team-mode-execution.md`.
