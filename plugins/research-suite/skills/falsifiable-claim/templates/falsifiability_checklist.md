# Falsifiability checklist

Every claim has a kill criterion. The kill criterion must satisfy four properties. A claim whose kill criterion fails any property is not yet ready for Stage 4-5.

## Property 1: Direct

The measurement contradicts the claim by itself, without requiring auxiliary assumptions.

**Pass:** Measuring the predicted spectral gap collapse before flocculation onset; if it does not appear, the claim is wrong.

**Fail:** Measuring a correlation between shear history and flocculation onset, then assuming a mechanistic link. The correlation could arise from a different mechanism.

**Why it matters:** Indirect kill criteria let the claim survive auxiliary-assumption revisions forever.

## Property 2: Achievable

The measurement is possible with current or near-future instrumentation.

**Pass:** Two-time correlation function measurement at 10 ms resolution on an XPCS beamline that currently supports 1 ms resolution.

**Fail:** A measurement requiring spatial resolution 100x better than any existing instrument.

**Why it matters:** An unachievable kill criterion is philosophically falsifiable but operationally unfalsifiable. The pipeline cannot proceed.

## Property 3: Pre-specified

The pass/fail threshold is written down before the measurement is taken.

**Pass:** "Spectral gap must drop by more than 1 order of magnitude within 30 seconds of the G'/G'' crossover."

**Fail:** "We will look for a spectral gap signature and assess whether it is consistent with our claim."

**Why it matters:** Post-hoc threshold selection converts any measurement into a confirmation. The most common failure mode in claimed falsifiability.

## Property 4: Distinguishing

The kill criterion distinguishes the claim from the null hypothesis.

**Null hypothesis:** The mechanism proposed in the claim does not operate.

**Pass:** The claim predicts a signature that the null hypothesis does not predict, or predicts a magnitude quantitatively different from the null.

**Fail:** The claim and null hypothesis both predict the measured outcome within experimental error, so the measurement cannot distinguish.

**Test:** Write a two-line summary of what the null hypothesis would predict for the same measurement. If it predicts the same thing, the kill criterion is not distinguishing.

**Why it matters:** Non-distinguishing criteria waste resources confirming effects that any theory would predict. The work contributes nothing.

## Audit format

For each kill criterion, fill in:

```
Kill criterion: [one-sentence statement with threshold]

Property 1 (Direct): [pass / fail, with one-sentence justification]
Property 2 (Achievable): [pass / fail, naming the instrument or method]
Property 3 (Pre-specified): [pass / fail, with the threshold quoted]
Property 4 (Distinguishing): [pass / fail, with the null-hypothesis prediction]
```

If all four pass, the claim advances. If any fails, iterate on the criterion (or the claim).

## Common repair moves

- **Property 1 fails:** The measurement has two interpretations. Either narrow the claim (specify which interpretation is being tested) or add a second measurement that disambiguates.
- **Property 2 fails:** Either rescope the claim to a regime where measurement is possible, or flag as a long-term program where the measurement capability is part of the contribution.
- **Property 3 fails:** Pre-register the threshold. If the user is uncomfortable pre-registering, usually the reason is that they are not confident in the magnitude; address that head-on by doing more preliminary simulation in Stage 6.
- **Property 4 fails:** Usually means the claim's mechanism does not actually predict anything specific. Return to Stage 3 draft and tighten the mechanism's consequences.
