---
name: falsifiable-claim
description: Stage 3 of the research-spark pipeline. Converts the Stage 1 articulation plus the Stage 2 landscape into a falsifiable claim with an explicit kill criterion, shaped by a Fermi feasibility screen, the Heilmeier catechism, and a Reviewer 2 adversarial pass. Triggers on phrases like "sharpen the hypothesis", "write the Heilmeier", "state what we're actually claiming", "what would falsify this", "tighten the research question", "turn this into something testable", "make this a pre-registerable claim", "is this even measurable", "order-of-magnitude check", or after Stage 2 completes. The output is a claim that names a single measurement that could kill it, pre-specified before any data is taken, on a physically or computationally reachable effect size.
---

# falsifiable-claim

Stage 3. Converts a sharpened idea plus a surveyed landscape into a claim with a kill criterion.

## Why this stage exists

A spark that cannot be falsified by a specific measurement is not yet a research claim. It may be interesting inquiry, but no experiment or simulation can confirm or refute it, which means downstream effort cannot converge. Stage 3 forces the conversion. A project that arrives at Stage 4 with a non-falsifiable claim wastes theoretical and computational work on something that was never going to have a crisp answer.

The four failure modes this stage catches:
- Claims that are trivially true (follow from the assumptions with no empirical content)
- Claims that are trivially unfalsifiable (no measurement could contradict them)
- Claims whose "kill criterion" is actually a consistency check that any theory would pass
- Claims built around an effect that is real in principle but unreachable in practice (below any plausible noise floor, beyond feasible compute): caught by a cheap Fermi estimate before Stage 4-5's derivation is spent on it

## Workflow

### 1. Load inputs and classify the question type

Read `01_spark.md` and `02_landscape.md`, noting the surviving gap and the open questions Stage 2 flagged.

Classify the claim's question type, because this drives the methodological backbone downstream:

- **Mechanism.** What process generates the behavior? (Theory-heavy.)
- **Prediction.** Can we forecast X from Y with specified accuracy? (Prototype-heavy.)
- **Design.** Can we construct a system with target properties? (Experiment-heavy.)
- **Measurement.** Can we resolve a quantity previously inaccessible? (Measurement itself is the contribution.)

Canonical-form variants for each type are in `templates/claim_schema.md`.

### 2. Fermi feasibility screen

Before drafting the claim, run 2-4 back-of-the-envelope estimates for the effect the claim would test: expected signal magnitude, signal-to-noise or contrast ratio, computational complexity bound, or time-to-solution, whichever the question type calls for. Identify at least one governing dimensionless ratio (a timescale ratio, force-balance ratio, or signal-to-background ratio) that defines the regime where the phenomenon should be observable at all.

**Exit gate:** the target phenomenon must occupy a non-zero, reachable domain in that ratio's range. If the estimate puts the effect orders of magnitude below any plausible noise floor, or the compute bound orders of magnitude beyond feasible resources, do not draft a claim around it; return to Stage 1-2 and narrow or reframe. A claim built on an unreachable effect wastes Stage 4-5's derivation on something that was never going to be testable.

This is a screen, not a derivation: Stage 4-5's `dimensional_analysis.md` does the rigorous Buckingham Pi version once the claim is fixed. `templates/fermi_screen.md` has the worksheet and worked examples.

### 3. Draft the claim in canonical form

"Under conditions C, system S exhibits behavior B with measurable signature M."

All four slots filled, specifically. C and S are specific (not "generic colloidal suspensions"; "aqueous silica suspensions at volume fractions 0.45-0.55"). B is a physical behavior, not a metric. M is a measurable quantity with units and expected magnitude.

If any slot is a placeholder, keep drafting. Vague slots cascade forward into vague downstream work.

### 4. Complete the Heilmeier catechism

Use `../_research-commons/templates/heilmeier.md`. Answer in plain language, no jargon that needs a definition.

The Heilmeier is also a cross-check on the claim: if question 1 (what are you trying to do) requires specialist vocabulary, the claim is not yet crisp; if question 3 (what is new) gestures vaguely, the novelty is not yet specific; if question 7 (what are the midterm and final exams) has no quantitative pass/fail, the claim is not yet falsifiable.

### 5. Reviewer 2 challenge

Use the Stage 3 variant in `../_research-commons/templates/reviewer2_persona.md`. The reviewer picks the strongest of three positions: physically impossible (violates a conservation law or thermodynamic bound), mathematically unsound (circular, ill-posed, or trivially derivable), or already solved (equivalent to an existing result).

Append the transcript. Write counter-rebuttals paragraph by paragraph. A claim advances only when every reviewer argument has a solid rebuttal or has been absorbed as a revision.

If the reviewer argues "already solved" and cites a specific paper, cross-check with Stage 2's bibliography. A missing paper means Stage 2 was incomplete; consider a targeted pass back there rather than handwaving.

### 6. Falsifiability check

The kill criterion must pass four properties (see `templates/falsifiability_checklist.md`):

- **Direct.** The measurement contradicts the claim by itself, without auxiliary assumptions.
- **Achievable.** The measurement is possible with current or near-future instrumentation.
- **Pre-specified.** The pass/fail threshold is written down before the measurement is taken, not chosen post-hoc.
- **Distinguishing.** The claim and the null hypothesis (mechanism does not operate) predict different measurement outcomes at the resolution being measured.

A kill criterion that fails a property is a signal to iterate on the criterion, or sometimes on the claim itself. The repair moves are in the checklist.

### 7. Write, lint, hand off

`artifacts/03_claim.md`, in this order: claim in canonical form, question type and rationale, Fermi screen with dimensionless ratio, Heilmeier answers, Reviewer 2 transcript with rebuttals, kill criterion with four-property audit, open questions for Stage 4. Style-lint. Update `_state.yaml`.

## Failure modes worth naming

- **Skipping the Fermi screen because "the effect is obviously there."** The screen is cheap precisely so it can run even on ideas that feel obvious; obvious effects turn out to be three orders of magnitude below the noise floor often enough to be worth the two minutes.
- **Vague "significant correlation" as a kill criterion.** A correlation coefficient with a pre-specified threshold and a power analysis is a kill criterion; "a significant relationship" is not.
- **A Reviewer 2 pass that concedes immediately.** The persona is calibrated adversarial; if the output is soft, re-prompt with the harsher tone in the persona file.
- **Folding the kill criterion into the claim statement.** The claim says what is true; the kill criterion says what would show it false. Conflating them creates circular structure that cannot be tested.
- **Post-hoc threshold selection, framed as "we'll set the threshold after we see the data."** This is how pre-registration dies in practice. If the user resists pre-specification, the usual reason is they do not yet trust the Stage 6 prediction. Address that by sharpening Stage 6, not by waiving pre-registration.

## Templates

- `templates/claim_schema.md`: canonical-form variants per question type
- `templates/fermi_screen.md`: back-of-envelope worksheet and dimensionless-ratio worked examples
- `templates/falsifiability_checklist.md`: four-property audit with repair moves
- Shared: `../_research-commons/templates/heilmeier.md`, `../_research-commons/templates/reviewer2_persona.md`

## Handoff to Stage 4-5

Stage 4-5 reads the claim and builds a theoretical framework whose predictions match the claim's signature M. The kill criterion is the specific prediction that framework must produce. If the framework cannot reproduce the signature even in principle, either the framework is wrong or the claim is; either way, Stage 4-5 will surface the mismatch, which is a useful thing to surface early. The Fermi screen's dimensionless ratio is a head start on Stage 4-5's own Buckingham Pi analysis, not a substitute for it.
