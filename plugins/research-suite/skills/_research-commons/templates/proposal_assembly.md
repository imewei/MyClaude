# Proposal assembly

Run at the core-completion checkpoint, immediately after the hostile
self-audit passes. Synthesizes artifacts `01_spark.md` through
`05_formalism.tex` plus the audit into `proposal_draft.md`, the core
pipeline's terminal deliverable. Not a stage artifact; it lives at the
project root next to `_state.yaml` and `project_log.md` because nothing
downstream reads it by path.

## Write in reverse order

Chronological drafting (Introduction first) invites scope creep: the
background section grows to justify aims that have not been fixed yet.
Draft in this order instead:

1. **Specific Aims** (below), the only section written from a blank page.
2. **Methods & Risk**, pulled from Stage 4-5's formalism and the hostile
   audit's bottleneck routes. Every method traces to a specific aim.
3. **Background & Gap**, pulled from Stage 2's synthesis and gap matrix.
   Now that the aims are fixed, frame the literature to lead directly to
   them, not the reverse.
4. **Abstract**, written last, summarizing the completed architecture in
   one pass.

## Specific Aims

One aim per major claim component from `03_claim.md`. Each aim follows the
same three-part template:

> **Aim N: [one-line title].**
> *Objective*: what this aim establishes, one sentence.
> *Approach*: the minimal model or method from Stage 4-5 (and, if the
> optional extension ran, Stage 6-7); one sentence, no unnecessary detail.
> *Deliverable / falsification metric*: the specific result that would
> confirm the aim, and the result that would refute it, both quantitative.

A typical shape, not a mandate: Aim 1 establishes the theoretical baseline
or minimal model; Aim 2 tests the core hypothesis against the kill
criterion; Aim 3 (if the claim supports one) expands to a broader regime or
real-world validation. Fewer or more aims than three is fine if the claim
does not naturally split into three.

## Line-of-sight check

Every methodological choice in Methods & Risk must trace back to a specific
aim; every aim must trace back to the theoretical gap in `02_landscape.md`.
Read the assembled draft once, end to end, and for each paragraph ask which
aim it serves. A paragraph that serves none is cut, not kept "for context."

$$\text{Theoretical Gap} \to \text{Claim} \to \text{Aim} \to \text{Deliverable} \to \text{Impact}$$

## Value-to-risk calibration

One paragraph in Methods & Risk (not the abstract): name the highest-risk
theoretical or technical leap in the proposal, and state plainly why the
structural insight if it succeeds justifies the risk. Reviewers discount
proposals that hide their biggest bet; naming it and defending it reads as
confidence, not weakness.

## Exit gate

Every aim has a falsification metric (inherited from the claim's kill
criterion and the hostile audit). Every paragraph traces to an aim
(line-of-sight check). The abstract was drafted last, from the finished
aims, not the other way around.

## Optional: solo time-boxing

If working alone against a deadline, a rough split that keeps refinement
from running away: Stage 1-2 (anchoring) ~20% of total time, Stage 3
including the Fermi screen (screening + formulation) ~20%, Stage 4-5
(theory) ~25%, this checkpoint (audit + assembly) ~35%. The checkpoint gets
the largest share deliberately: assembly is where an unconverged idea gets
caught, not papered over with prose.
