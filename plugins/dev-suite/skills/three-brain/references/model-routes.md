# Codex and Agy route recipes

How to actually dispatch once the Fast Decision Table in SKILL.md has chosen a model:
the prompts, flags, and result-shapes for each route. The table alone is enough to
decide; read this when carrying the decision out.

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
