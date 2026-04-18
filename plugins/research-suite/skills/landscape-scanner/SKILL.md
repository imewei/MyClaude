---
name: landscape-scanner
description: Stage 2 of the research-spark pipeline. Conducts a structured three-layer literature scan (foundational / recent / adjacent), steelmanner each paper, builds a gap matrix mapping phenomena against methods, and runs a Reviewer 2 adversarial pass against the proposed research gap. Triggers when the user has a Stage 1 articulation and needs to survey the field, or on phrases like "map the literature on X", "survey what's known about Y", "what's the state of the field on Z", "steelman the prior work on W", "find the gap", or "gap analysis for [project]". Also triggers when the orchestrator advances to Stage 2. The output is not a paper pile but a defendable synthesis, specifically a document the user could hand to the researcher who wrote one of the cited papers.
---

# landscape-scanner

Stage 2. Three-layer literature scan with mandatory steelmanning and an adversarial reviewer pass.

## Why this stage exists and why it is structured this way

Surface searches produce lists of titles. They do not produce an understanding of where the field stands, where it is stuck, and why. The three things that distinguish this stage from a keyword search are:

- A three-layer search (foundational, recent, adjacent) rather than just "recent papers on X." Each layer catches a different kind of prior art.
- A steelman note for every paper, stating its strongest claim *and* the conditions under which it breaks. The break-conditions field is the one that does the work.
- A Reviewer 2 pass against the proposed gap. The reviewer argues the gap is not real, not tractable, or not impact-bearing. A gap that survives this argument is worth investing in.

The product is a synthesis the user can defend in front of someone who wrote one of the cited papers.

## Workflow

### 1. Extract concepts from `01_spark.md`

Pull out the named mechanisms, system type, measurement modalities, and adjacent-field hooks. If these are vague, return a specific clarifying question rather than guessing; Stage 1 produced a spark that was supposed to name these.

### 2. Build a three-layer search plan

Follow `references/search_strategy.md`. Briefly: Layer 1 is the foundational papers the subfield cites without re-reading (often 10-40 years old); Layer 2 is 3-5 recent papers that moved the state of the art on the specific phenomenon; Layer 3 is adjacent fields that encountered the phenomenon from a different angle.

Show the user the search plan before executing. They often know exactly which Layer 1 papers matter, and a quick "the two or three papers that define this subfield" question saves an hour of search.

### 3. Steelman every paper

Use `templates/steelman_note.md`. Three fields are required: strongest claim (as the authors would phrase it, at its most ambitious), conditions under which it breaks (specific regime or parameter range, not generic "this is approximate"), and residual uncertainty bearing on the user's spark.

A break-conditions field that says "fails in extreme cases" has not been steelmanned. Either dig further or mark the paper "not fully surveyed" rather than pretending.

### 4. Build the gap matrix

Rows are phenomena or sub-questions; columns are methods or approaches; cells describe what is known, with a citation. Empty cells are gaps. Use `templates/gap_matrix.md`.

The matrix is a tool for spotting where the user's spark fits. Intersection of an empty row and an empty column is a candidate research question. Verify it is real (not just underreported), tractable (some route to a measurement exists), and impact-bearing (resolving it changes something downstream).

### 5. Reviewer 2 pass against the proposed gap

Use the Stage 2 variant in `../_research-commons/templates/reviewer2_persona.md`. The reviewer argues one of three positions: gap is not real (already filled in the cited literature), not tractable (no route with current tools), or not impact-bearing (resolving it changes nothing downstream).

Append the reviewer transcript to the artifact. Write counter-rebuttals directly below it, paragraph by paragraph. A gap advances only when every reviewer argument has a solid rebuttal or has been absorbed as a revision.

If the reviewer writes soft critique ("interesting work, a few questions"), re-prompt with the harsher calibration in the persona file. The value of this pass is its adversarial bite.

### 6. Synthesis

Two or three paragraphs at the top of the artifact, summarizing what the field knows, what it does not know, and where the user's spark fits. Write this last, not first. A synthesis written before the steelmanning is speculation; written after, it is the summary of work actually done.

### 7. Write, lint, hand off

`artifacts/02_landscape.md`, in this order: synthesis, gap matrix, Reviewer 2 transcript with rebuttals, bibliography with steelman notes, open questions for Stage 3. Style-lint the output. Update `_state.yaml`.

## Depth expectation

Aim for roughly 8 fully steelmanned papers before advancing to Stage 3. This is a default, not a law. If the literature is genuinely thin (a niche subfield, a very new phenomenon) advance with fewer and log the override with a one-line reason. Thin literature is itself informative.

What "fully steelmanned" means: the break-conditions field is specific, and the residual-uncertainty field names something that actually bears on the user's spark. Twelve papers with empty break-conditions fields are less useful than five with filled ones.

## Failure modes worth naming

- **Padding the bibliography with papers that were not read past the abstract.** Better to list fewer papers honestly.
- **A gap matrix where every cell cites the same 2-3 papers.** Either the matrix is too narrow (the rows and columns are not actually distinct), or the literature is thinner than it looks.
- **A Reviewer 2 pass that produces only nitpicks.** Re-prompt harder. The pass is supposed to challenge the gap's existence, not argue about methodology.
- **Skipping Layer 3 because "adjacent fields aren't relevant."** Usually they are, and the user does not know which until they look. Ecology has early-warning-signal literature that rheologists can learn from; dynamical systems has regime-shift work that chemistry can use.

## Templates, references, scripts

- `templates/steelman_note.md`: per-paper format
- `templates/gap_matrix.md`: matrix format with worked example
- `references/search_strategy.md`: three-layer search protocol
- `scripts/dedupe_refs.py`: DOI-based bibliography de-duplication
- `scripts/bib_to_steelman.py`: converts `.bib` entries to steelman-note stubs
- Shared: `../_research-commons/templates/reviewer2_persona.md`

## Handoff to Stage 3

Stage 3 consumes the synthesis and the surviving gap. If the surviving gap is too wide for a single falsifiable claim, Stage 3 will flag it and send the user back here to sub-divide. That is fine and should not feel like failure; wide gaps often need iterative narrowing between stages.
