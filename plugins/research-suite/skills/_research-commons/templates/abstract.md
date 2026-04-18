<!-- style_lint:ignore-file -->

# Abstract template

A shared format for abstracts across the research-spark stack: proposal narratives, manuscript drafts, internal summaries. The structure is the same regardless of length; the detail level scales with the target word count.

## Structure

Every abstract, regardless of length, contains the same five moves in this order:

1. **Problem or opportunity.** What is the question? Why does it matter? One sentence. Can name the system (battery-electrode slurries, designed protein folds) and the phenomenon (flocculation onset, de novo fold stability).

2. **Prior limits.** What does the field currently fail to resolve, or do imperfectly? One sentence. Named methods or baselines ("current oscillatory-rheology predictors have zero lead time by construction") work better than generic gestures ("current methods are limited").

3. **Our approach.** What is the specific new idea or method? One to two sentences. Names the mechanism, method, or design move. Quantifies where possible.

4. **Key result.** What did the approach produce, or what is it predicted to produce? One to two sentences with numbers: lead time, precision, effect size, coverage, sample count.

5. **Consequence.** Why this matters downstream. One sentence. Specific enough to be falsifiable; not "advances the field."

## Length variants

### Very short (35-50 words)

Use for social media, grant summaries, conference submission systems with tight limits. Compress by combining moves:

> We show that the spectral gap of the stress-response operator collapses 30-60 seconds before flocculation onset in battery slurries, a lead time unavailable from G' / G'' crossover detection. The predictor, extracted from Rheo-XPCS two-time correlation functions, enables preemptive shear intervention during processing.

### Short (100-150 words)

Common for PRL abstracts, grant one-pagers, first paragraphs of introductions. All five moves distinct:

> Flocculation onset in concentrated battery-electrode slurries is currently detected tens of seconds after the storage-modulus crossover, too late to act on during processing. We show that the first eigenvalue spacing of the linearized stress-response operator, extracted via operator-valued regression on Rheo-XPCS two-time correlation functions, collapses by more than an order of magnitude before the crossover. On aqueous silica suspensions at volume fractions 0.45-0.55, the predictor provides 30-60 s of lead time with a false-positive rate below 5%, enabling shear-history interventions that delay flocculation by tens of seconds at fixed solids loading.

### Standard (200-250 words)

For most manuscript abstracts and Nature-style summaries. Expand moves 3 and 4 with method detail and broader result set.

### Long (300-400 words)

For proposal executive summaries where multiple aims or outcomes need surfacing. Consider breaking move 5 into primary and secondary consequences.

## Claim-type variants

The framing leans on the question type from `03_claim.md`:

- **Mechanism claim.** Moves 3-4 emphasize the mechanism and its signature. Predicate: "produces a measurable signature distinct from alternatives."
- **Prediction claim.** Moves 3-4 emphasize the prediction's precision and margin over baselines. Predicate: "outperforms method X by margin Y under conditions Z."
- **Design claim.** Moves 3-4 emphasize the manipulation and the target state achieved. Predicate: "achieves target T, verified by observable O exceeding threshold Theta."
- **Measurement claim.** Moves 3-4 emphasize the resolving method and precision improvement. Predicate: "resolves quantity Q to precision P, previously accessible only at coarser precision P_prior."

## What to avoid

- **Motivation-heavy opening.** A sentence and a half on why the field matters in general is a luxury the reader may not indulge. Spend that real estate on what is specifically new.
- **Method-first abstracts.** "We used X technique" without first saying what problem X is being applied to asks the reader to do the work of connecting tool to question.
- **Result without context.** "We measured a 30-second lead time" is not a result without the context that current methods have zero lead time.
- **Vague consequence.** "This work opens new possibilities" is empty. Name the downstream thing that changes.
- **Banned vocabulary.** The style linter will flag "innovative", "state-of-the-art", "transformative", "sustainable", "novel", "groundbreaking" and variants. Quantify instead.

## Self-check

Before releasing any abstract:

- Does it contain all five moves?
- Does move 4 have at least one number?
- Does move 5 name a specific downstream change, not a generic gesture?
- Does the abstract stand alone without reference to any figure, table, or section?
- Would a researcher in an adjacent subfield understand it in one pass?
- Does the word count match the target? (Check with `wc -w`, or character count with `wc -m` for venues with character limits.)
