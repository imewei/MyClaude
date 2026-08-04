# Hostile self-audit

Run at the core-completion checkpoint, after Stage 4-5 and before assembling
`proposal_draft.md`. Reasoning-only: no code, no instruments, no experiment
plan required. It simulates an adversarial reviewer for a solo researcher
who does not have one standing by.

## 1. The $100k Kill Switch

Ask: "If I were given $100,000 to disprove my primary hypothesis in three
months, how would I do it?"

Write the fatal experiment or edge case as concretely as you can. If it is
easy to name, that is not a weakness in the idea; it is a gift. Fold it
directly into the falsifiable claim's kill criterion (`03_claim.md`) as the
decisive test, rather than leaving it as an external threat.

If no fatal test comes to mind after genuine effort, say so explicitly
rather than padding the answer. A hypothesis nobody can picture disproving
yet is itself useful information for the audit below.

## 2. The Artifact Check

Write three alternative, non-theory explanations for why the expected
result might appear even if the proposed mechanism is false: secondary
numerical dissipation, unmeasured boundary effects, systematic instrument
drift, a confound already flagged in Stage 2's steelman notes, and so on.

For each, name what would distinguish it from the real signal. An
alternative explanation with no distinguishing observation is a permanent
ambiguity baked into the proposal; either find the distinguishing
observation now or flag the ambiguity openly in the risk section.

## 3. The Bottleneck Route

For every aim the proposal will state, complete this sentence:

> "If [technique X] fails to deliver sufficient [resolution / convergence /
> signal], I will pivot to [technique Y], which relaxes assumption [Z]."

One sentence per aim, naming the actual fallback technique and the specific
assumption it relaxes, not "we will investigate alternatives."

## Exit gate

No aim in the assembled proposal may depend on a single unvalidated
technique without a documented alternative pathway (Section 3), and the
kill switch (Section 1) must be either incorporated into the claim's kill
criterion or explicitly acknowledged as unresolved. A proposal that passes
this gate is not risk-free; it is a proposal with its risks on the page
instead of hidden.

## Worked example (rheology/XPCS)

1. **Kill switch.** Run the null control (bond-exchange kinetics artificially
   frozen) at the predicted De ~ O(1) regime. If the spectral gap still
   collapses, the mechanism is not bond-exchange; this becomes the claim's
   kill criterion directly.
2. **Artifact check.** (a) Beam-heating-induced viscosity drift mimicking a
   gap collapse, distinguished by a heating-free dark control. (b) Detector
   saturation at high count rate near the transition, distinguished by
   attenuator sweep showing the signal is intensity-independent. (c) Sample
   aging over the measurement window, distinguished by a repeat run on a
   fresh sample showing the same collapse timing.
3. **Bottleneck route.** "If XPCS temporal resolution cannot reach the
   predicted 10 ms collapse window, I will pivot to bulk oscillatory
   rheology at reduced time resolution, which relaxes the requirement for
   spatially resolved dynamics and tests only the bulk signature of the
   same collapse."
