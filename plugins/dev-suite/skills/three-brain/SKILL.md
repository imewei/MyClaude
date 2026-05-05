---
name: three-brain
description: |
  Route work between Claude, Codex, and Gemini when another model is likely to improve correctness, coverage, or perception. Use this skill for second-opinion reviews of Claude's own work, high-risk code paths, repeated failures, video/audio/PDF/image inspection, long-context repository or document scans, and explicit requests like "ask Codex", "ask Gemini", "second opinion", "sanity check", "review your work", or "use all three". Prefer not to trigger for ordinary Q&A, simple edits, or reviewing user-authored non-code drafts unless the user explicitly asks for another model.
compatibility: Requires `codex` for Codex routes and `gemini` for Gemini routes. Falls back gracefully when either CLI is missing.
---

# Three-Brain Router

Use Claude as the driver. Call Codex or Gemini only when their different strengths materially improve the result. Keep routes bounded, cite evidence from returned output, and preserve the user's workflow.

## Fast Decision Table

| Situation | Route | Why |
| --- | --- | --- |
| User asks to review/check/sanity-check work Claude just produced | Codex review | Avoid same-model blind spots |
| Active edits touch auth, billing, migrations, deployment, secrets, permissions, or infra | Codex review | High blast radius deserves independent scrutiny |
| Same test, command, or bug fails twice on the same path | Codex rescue | Stop repeating the same local approach |
| User provides video, audio, image, scanned PDF, charts, or visual layout to inspect | Gemini analysis | Use stronger multimodal perception |
| User asks for broad repository/document discovery over lots of files | Gemini long-context scan | Reduce token-heavy local reading |
| User explicitly says ask Codex/Gemini/all three/cross-check | Requested model(s) | Follow the user's routing request |
| Ordinary explanation, writing, small edit, local file operation, or user-authored tone review | Claude direct | Extra routing adds cost without clear value |

When uncertain about review of Claude's own output, route to Codex. When uncertain about ordinary user-authored content, stay direct unless the user asked for a second model.

## Startup Check

Run this once per session, only before the first route that needs the tool:

```bash
codex --version 2>&1 | head -1
gemini --version 2>&1 | head -1
```

If a CLI is missing, tell the user once and continue with the available route or Claude direct. Do not recheck every turn.

## Codex Routes

Use Codex for independent code review, adversarial reasoning, and rescue after repeated failures.

For tracked repository changes:

```bash
git diff --stat
git diff | codex exec --skip-git-repo-check "Review this change. Focus on bugs, regressions, security risks, missing tests, and unclear assumptions. Return findings first with file/line references when possible."
```

For untracked files or non-code output, pipe only the relevant file(s) or excerpt. Keep the prompt narrow. Ask Codex for findings, evidence, and recommended fixes; do not ask it to rewrite everything unless that is the task.

After Codex returns:

- Integrate only findings supported by evidence.
- If there are no actionable findings, say so.
- End the response with `(Routed via three-brain -> Codex review.)` when the route was triggered by this skill.

## Gemini Routes

Use Gemini for perception-heavy and long-context tasks. Ask for structured evidence, not a flat summary.

Video:

```bash
gemini -p "Analyze this video. Return timestamped findings as [MM:SS] event. Cover visible content, on-screen text, speaker/action changes, transitions, and notable issues. Cap at 800 words." @/path/to/video.mp4
```

Audio:

```bash
gemini -p "Analyze this audio. Return timestamped findings as [MM:SS] event, including speakers if distinguishable, key claims, action items, and uncertainty. Cap at 800 words." @/path/to/audio.wav
```

PDF or document:

```bash
gemini -p "Extract key claims, tables, chart findings, contradictions, and action items with page-number citations. Cap at 1000 words." @/path/to/file.pdf
```

Repository or large directory scan:

```bash
gemini -p "Find every place related to <topic>. Return file:line citations, short purpose, and confidence. Avoid broad summaries." @/path/or/archive
```

If Gemini cannot ingest a directory directly, archive or narrow the relevant files first. Prefer file, page, or timestamp citations in every Gemini prompt.

## Forced Risk Review

Route to Codex when active work touches high-risk targets:

- `src/auth/**`, `**/*OAuth*`
- `src/billing/**`, `**/*Stripe*`
- `migrations/**`
- `deploy/**`, `infra/**`
- `.env*`, `secrets/**`
- `policy/**`, permissions, roles, or ACL logic

Announce forced routes in one short line before calling the tool so the user can interrupt:

```text
[three-brain] routing to Codex review - risk path: src/auth/
```

Do not announce when the user explicitly asked for the route.

## Failure Counter

Track repeated failures deterministically. If the same command, test, or bug fails twice on the same code path after Claude has attempted a fix, route to Codex rescue:

```text
[three-brain] routing to Codex rescue - same failure repeated twice
```

Give Codex the failing command, exact error, relevant diff, and what was already tried. This avoids wasting tokens on a third local guess.

## Parallel Consensus

Use all three only when the user explicitly requests cross-model consensus or when the decision is high-stakes and the user agrees. Ask each model the same question and require this structure:

```text
Recommendation: <one line>
Blocking risks: <bullets>
Assumptions: <bullets>
Confidence: low / medium / high
Tests required: <bullets>
```

Compare the answers by evidence. Do not average opinions.

## Token And Stability Rules

- Route late enough to have a concrete artifact, error, file, or question.
- Send the smallest useful context: diffs over whole files, exact files over whole repos, bounded excerpts over dumps.
- Cap model outputs in the prompt when the route is exploratory.
- Prefer citations and findings over rewrites.
- Keep Claude responsible for final integration, user communication, and filesystem changes.
- If a route fails, report the failure briefly and continue with the best available local approach.

## Output Filing

When a route produces durable output, write it under:

```text
./three-brain-out/<YYYY-MM-DD>-<short-slug>/
```

Use only the files that apply:

- `input.txt` - user request or routed subquestion
- `codex-review.md` - Codex findings
- `gemini-analysis.md` - Gemini findings
- `consensus.md` - cross-model comparison
- `log.md` - run-specific summary

Append one root-level line to `./three-brain-out/log.md` for every route:

```text
[YYYY-MM-DD HH:MM] route=<codex-review|codex-rescue|gemini-analysis|consensus> target=<short target> status=<ok|partial|failed> duration=<seconds>s outputs=<N> summary="<plain-language result>"
```

Example:

```text
[2026-05-03 04:52] route=codex-review target=auth-middleware status=ok duration=42s outputs=1 summary="Found one missing test and no blocking security issue."
```
