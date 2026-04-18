# Premortem failure narrative

Write as past-tense prose. Assume the project has failed two years from now. Describe the failure as history.

## Format

Each narrative is 500-800 words and covers:

1. **The opening situation.** What the plan looked like when work started. One paragraph.

2. **The first sign of trouble.** What was seen, when, by whom. This is usually a specific incident: a measurement that did not match prediction, a milestone that slipped, a sample that did not behave. One paragraph.

3. **Why the first sign was not acted on.** Often the most informative part. The sign was rationalized, attributed to noise, or postponed for "after we finish the current run". One paragraph.

4. **The compounding phase.** What made the problem worse over the next months. Often several factors interacting: a schedule slip consumed the buffer, a side project absorbed attention, the rationalization of the first sign made the second harder to recognize. One to two paragraphs.

5. **The end state.** What actually happened. The paper that was not written. The wrong paper that was published. The move to other work. The student who left. One paragraph.

## Worked example (compressed)

> The RheoX spectral-gap project began strongly in month 1. The simulation was running, the theory scaffold produced a clean dimensionless form, and benchtop rheology confirmed the nominal slurry rheology was in the expected range. The first beamtime in month 3 ran smoothly from a logistics standpoint; the Rheo-XPCS cell worked, the shear control was stable, the data were clean.
>
> The first sign of trouble came in the month-3 data reduction. The predicted spectral gap collapse was 1.5 (dimensionless); the measured gap shifted by about 0.3. This was within the parametric uncertainty of the Stage 6 prediction, so it did not immediately register as a problem. We noted it, attributed it to the volume fraction being slightly below target, and planned to re-sample at the next beamtime.
>
> The next beamtime was month 7. The slurry samples for it were prepared by a new postdoc using a slightly modified protocol (they had good reasons; we approved). The month-7 data showed no gap collapse at all. In the debrief, we spent a week convincing ourselves that the sample preparation was the cause, re-characterized the samples, confirmed they matched the specification within our stated tolerance, and re-ran the simulations with the characterized inputs. The re-run simulations still predicted a visible collapse. We scheduled another beamtime for month 10.
>
> Between months 7 and 10, we spent effort rebuilding the Stage 6 prototype with tighter convergence, expecting that a sharper prediction would expose what had gone wrong. The prototype confirmed the prediction. The month-10 beamtime produced results indistinguishable from month 7.
>
> By month 14, the group consensus was that the gray-box closure in Stage 6 had compensated for a theoretical error during training, and the error only showed up when the trained model was applied to the real measurement regime. The theory needed to be redone. At that point the grant cycle was winding down, two group members had moved on, and the follow-up proposal had to be written around a claim that was no longer defensible.
>
> The project ended in month 22 with a reduced-scope paper on the simulation framework itself, no experimental validation of the original spectral-gap claim, and a lingering open question about what the actual relationship between the spectral gap and flocculation onset is. The field moved on; a different group published a measurement-first paper in month 19 showing that the gap signature exists but is weaker and later-arriving than the early-warning framing required.

## What this narrative exposes (example)

- The month-3 deviation of 0.3 vs predicted 1.5 should have triggered a rethink. The rationalization that it was "within parametric uncertainty" was convenient but revealed the prediction was underconstrained.
- The gray-box closure had no out-of-distribution check between training and experimental inference. This was a Stage 4-5 gap, not a Stage 6 gap, even though it surfaced in Stage 6.
- The month-7 debrief spent effort on sample characterization (easy to do) before revisiting the theory (hard to do). This is a recognizable pattern.

## What this narrative suggests as early signals

- A milestone at month 4: any measured observable that falls below 50% of predicted magnitude triggers a mandatory theory review, not a parameter re-characterization.
- A milestone at month 2 (during prototype development): out-of-distribution testing for the gray-box closure, with a specific pass criterion before any experimental data is ingested.
- A milestone at month 6: explicit go/no-go review with the collaborator outside the group, using the one-pager. No silent continuation.

Each of these becomes a line in the revised Stage 7 plan.

## The discipline

Past tense. Specific months. Named incidents. Writing in future conditional ("we might have...") produces generic narratives that do not identify anything actionable.
