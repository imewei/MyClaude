---
name: spark-articulator
description: Stage 1 of the research-spark pipeline. Compresses a rough research idea into a 3-5 sentence articulation that names what the idea is, what is surprising about it, and what observation would confirm it. Triggers on early-stage phrases like "I had this thought about...", "rough idea:", "what if we could...", "I've been wondering whether...", "I want to explore whether X might...", or any time a user shares a research hunch that is not yet tightly scoped. Also triggers when the orchestrator advances to Stage 1. The compression itself is the work; most sparks die here because they will not survive it, which is the point.
---

# spark-articulator

Stage 1. Converts a rough idea into a precise articulation that downstream stages can act on.

## What this stage does and why it comes first

Most research sparks die at articulation, not at execution. They die because the person carrying the spark cannot compress it into a single precise paragraph. Compression exposes three weaknesses that ruin downstream work if left hidden: vagueness (the idea is actually several ideas), absence of novelty (someone has already done this), and absence of a falsifiable core (no observation could confirm or deny it).

The articulation has three required components because each component catches one of those weaknesses:
- *What the idea is* catches vagueness
- *What is surprising* catches prior-art overlap
- *What observation would confirm it* catches unfalsifiability

An articulation missing any of the three is not ready for the pipeline. This is not bureaucratic; it is what distinguishes a hunch from a research claim.

## Workflow

### 1. Hear the rough version

Let the user say whatever they want, at whatever length. Do not interrupt to impose structure. The rough version contains information that premature compression will lose, including the user's own attempt at framing, which is worth knowing.

### 2. Targeted elicitation (only on gaps)

Ask questions only about gaps in the rough version. If the user already stated the system clearly, do not ask "what system?" again. Typical gap categories:

- **Scope.** What is in, what is out. Which regime, which system, which phenomenon.
- **Novelty.** What makes this different from what is already known. Has this been tried, tried and failed, tried and succeeded, or not tried?
- **Verifiability.** What measurement would resolve the question.

One question per category is usually enough. If the user cannot answer the verifiability question, that is information: the spark may not be ready, and manufacturing an answer for them is worse than noting the gap.

### 3. Produce three candidate framings

Write three 3-5-sentence versions of the articulation, each under a different framing:

- **Mechanistic.** "When [conditions], [mechanism] produces [observable]."
- **Predictive.** "Given [inputs], we predict [output] to within [precision]."
- **Design-oriented.** "By [manipulation], we can achieve [target state]."

The user picks one or blends two. Discards go into the final artifact as "Rejected alternatives" because they usually resurface in the paper's Discussion section later.

See `templates/framing_variants.md` for worked examples.

### 4. Elevator test

Read the chosen articulation as if you were a researcher in an adjacent but distinct subfield. Would the point land in one pass? If jargon blocks understanding, rewrite. If structure blocks understanding, reorder. Run `scripts/readability_check.py` as a sanity check on sentence length, jargon density, and passive voice.

### 5. Self-check against the three components

Before writing the artifact, confirm the articulation explicitly contains:

1. What the idea is (one clause, the proposition)
2. What is surprising (one clause, the departure from default expectation)
3. What observation would confirm it (one clause, the falsifiable signature)

If any component is missing or vague, add it or return to elicitation. Producing an artifact that is missing a component sets up downstream failure. The user may want to push through; the right move is to show them what is missing rather than paper over it.

### 6. Write `01_spark.md` and hand off

Use `templates/spark_template.md`. The artifact is short: half a page of prose plus the rejected alternatives. Run the shared style linter (`../_research-commons/scripts/style_lint.py`) on the output. Update `_state.yaml`.

## Failure modes worth naming

These come up often enough that they deserve explicit recognition:

- **Articulations that smuggle their conclusion into the question.** "We will show that X because Y is obviously true" is a paper abstract, not a research claim. Flag and rewrite without the assumed answer.

- **Verifiability answered with "it will be interesting to see."** That is not an observation. Either the user has a falsifiable signature in mind and has not named it yet, or they do not have one. Both outcomes are informative.

- **Framing ambiguity between mechanistic and predictive.** Not every mechanism is a good predictor, and not every predictor has a known mechanism. Forcing the user to pick one sharpens what the downstream work needs to do.

## Templates and scripts bundled here

- `templates/spark_template.md`: artifact structure
- `templates/framing_variants.md`: worked examples of the three framings
- `scripts/readability_check.py`: sentence length, jargon density, passive voice

## What Stage 2 will read from the output

- Named mechanisms or phenomena (for literature queries)
- System type (dense colloidal suspension, vitrimer nanocomposite, etc.)
- Measurement modality if named
- Adjacent-field hooks the user flagged

If any of these are vague in the final artifact, Stage 2 will come back with clarifying questions, which is normal. A too-sharp Stage 1 risks premature commitment; a too-loose Stage 1 makes Stage 2 unnecessarily painful. Aim for "the concepts are nameable" rather than "every term is defined."
