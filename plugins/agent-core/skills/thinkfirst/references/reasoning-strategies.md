# Reasoning Strategy Suggestions

After exploring the Seven Dimensions in Phase 2, and before drafting in Phase 3, assess whether the task would benefit from any of the reasoning techniques below. If so, proactively suggest them to the user in a conversational way — explain what the technique does in plain language, give a concrete example of how it would look in their prompt, and ask whether to include it. Do not force these; let the user decide.

If the user accepts a strategy, embed it into the draft using the guidance in the "Embedding Strategies Into the Draft" section at the bottom of this file.

---

## 1. Chain of Thought (CoT)

**What it does:** Forces the AI to work through a problem step by step instead of jumping to a conclusion. Improves quality on anything that involves analysis, weighing options, or multi-factor reasoning.

**When to suggest:** The task requires weighing trade-offs, multi-factor analysis, or arriving at a recommendation. Good signal: the user said things like "I need to decide," "compare X and Y," or "which option is best."

**How to suggest (sample script):**

> "This is the kind of task where AI tends to jump straight to a conclusion. Want me to build in a step-by-step thinking process? For example, first analyze each option individually, then compare across specific criteria, then give a final recommendation with reasoning."

---

## 2. Validation Gates

**What it does:** Breaks a complex multi-stage task into explicit phases, where the AI must complete and present one stage before moving to the next. Prevents compounding errors in long workflows.

**When to suggest:** The task has multiple dependent steps, or the user described a workflow where earlier outputs feed into later ones. Good signal: the user used words like "first," "then," "after that," or described a pipeline.

**How to suggest (sample script):**

> "Since this task has several stages that build on each other, want me to add checkpoints? Like: first do X and confirm the result, then use that to do Y, then finally Z. Adding checkpoints prevents the AI from rushing through and making errors that snowball."

---

## 3. Confidence Signaling

**What it does:** Requires the AI to flag claims it is uncertain about and to produce a verification checklist at the end of its output. Turns a black-box output into something the user can efficiently audit.

**When to suggest:** The task involves data, facts, claims, or any output where being wrong has consequences. Ties directly to Dimension 2 (Stakes) — suggest it whenever stakes are high.

**How to suggest (sample script):**

> "Since accuracy really matters here, want me to add a rule that the AI has to mark anything it is not sure about, and give you a list of things to double-check? That way the review can focus on the uncertain parts instead of re-reading everything."

---

## 4. Advanced Techniques (Suggest When a Clear Fit)

Suggest any of the following when they match the task, using the same conversational approach: explain simply, show how it would look, let the user decide.

- **Multi-persona debate** — Instruct the AI to generate the output from two opposing expert perspectives and then synthesize. Good for strategic or contested topics.
- **Adversarial self-review** — Have the AI produce a draft, then re-read it as a skeptical critic and revise. Good for writing where blind spots are common.
- **Reference class priming** — Give the AI a few real examples of similar outputs ("here are three good versions of this kind of memo") before asking for its draft. Good when tone or genre is hard to describe but easy to recognize.
- **Constrained draft then expand** — Ask for a very short version first (one paragraph), confirm direction, then expand. Good when the user is not sure what they want and iteration cost is high.

---

## Embedding Strategies Into the Draft

When the user accepts a reasoning strategy, weave it into the Phase 3 draft as follows:

- **Chain of Thought** — Add a numbered thinking sequence in the Task section. Example:
  > "Step 1: Analyze each option against the criteria individually. Step 2: Compare the options head-to-head on the three most important criteria. Step 3: Present a final recommendation with the reasoning that drove it."

- **Validation Gates** — Structure the Task as explicit phases with clear outputs per phase. Example:
  > "Phase 1: Produce the outline and present it before proceeding. Phase 2: Once the outline is confirmed, expand each section into full prose. Do not skip ahead to Phase 2 until Phase 1 output has been shown."

- **Confidence Signaling** — Add a constraint requiring the AI to flag uncertain claims inline (e.g., with a `[verify]` marker) and to append a "claims to verify" list at the end of its output. Example:
  > "Constraint: For every factual claim that is not directly stated in the context above, append `[verify]` to the claim. At the end of the output, include a bulleted list titled 'Claims to verify' containing each flagged claim."

- **Multi-persona debate** — Add a Task step: "Generate the draft twice, first from the perspective of [persona A], then from the perspective of [persona B]. Then synthesize a final version that addresses the strongest points from both."

- **Adversarial self-review** — Add a final Task step: "Re-read the draft as a skeptical reviewer looking for weak arguments, unsupported claims, and generic phrasing. Revise the draft in response. Present only the revised version."

- **Reference class priming** — Move the supplied examples into the Examples section of the prompt, wrap them in `<example>` tags, and add a Task instruction: "Match the voice, structure, and level of detail of the examples above."

- **Constrained draft then expand** — Structure the prompt as two phases (similar to Validation Gates): first produce a 150-word summary, then expand after confirmation.

---

## Do Not Force Strategies

More techniques is not better. Each added strategy makes the prompt longer and the AI's job more constrained. Suggest a strategy only when it has a clear justification tied to a specific dimension answer, and when in doubt let the user decide whether to include it. Trust the user's judgment — do not stack strategies reflexively, and do not force any of them onto a task that does not need them.
