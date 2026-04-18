---
name: experiment-designer
description: Stage 7 of the research-spark pipeline. Translates the Stage 6 predicted observable into an experimental plan with an instrument capability map (3x margin rule on every dimension), pre-registered success metrics, controls, a formal power analysis, a design-of-experiments matrix, and a risk register. Triggers on phrases like "design the experiment", "plan the measurements", "build the DoE", "check instrument capability for this prediction", "compute statistical power for this comparison", "write the risk register", "pre-register the metrics", "what samples do we need", or after Stage 6 completes. The capability map is the central artifact; it is the thing that catches "the measurement cannot actually resolve what we predicted" before beamtime is spent finding that out experimentally.
---

# experiment-designer

Stage 7. Translates the predicted observable into a testable plan, grounded in what the available instruments can actually resolve.

## Why this stage exists

A simulation that predicts an observable is not yet a testable plan. Three things must be added before it becomes one:

1. **The instrument can actually resolve the predicted signal.** This is the capability map, and it is the new element that v2 of the pipeline centers on.
2. **Success is pre-registered.** The pass/fail threshold is fixed before any data is taken, so "we found what we were looking for" is not a retroactive claim.
3. **Ways the plan can fail are named with contingencies.** The risk register; not for paperwork, but so the project does not die on the first surprise.

## Prerequisites

`06_prototype.md` and the predicted observable file must exist. The observable file's uncertainty decomposition (numerical / parametric / statistical) is what the capability map compares against.

## Workflow

### 1. Load inputs

Read `06_prototype.md` and the predicted observable (YAML or JSON per the Stage 6 template). Pull out: observable amplitude, temporal and spatial structure, uncertainty decomposition, noise model, expected SNR.

Also read `03_claim.md` to recover the question type. This determines which statistical test applies, which drives the power analysis.

### 2. Build the instrument capability map

This is the central artifact. For each measurable quantity the plan depends on, compare the predicted signal against the instrument's capability on that dimension, compute the margin, flag the risk.

Template: `templates/instrument_capability_map.md`. Required dimensions: temporal resolution, total duration, signal amplitude vs baseline noise, expected SNR, control variables. Add technique-specific dimensions (coherence length, q-range, count rate for XPCS; torque resolution, shear-rate stability for rheology; etc).

The **3× margin rule** is the default: if margin is below 3× on any dimension, the measurement is high-risk and needs a specific mitigation. The 3× value is a heuristic, not a law. If domain experience with the specific instrument indicates a different threshold is routinely achievable, override with a logged reason. The point is a hard threshold that forces a decision, not a magic number.

For APS Rheo-XPCS work, `references/aps_rheo_xpcs_capabilities.md` has the baseline. Update it after each beamtime run; the file is meant to be a living record of how the capability evolves.

### 3. Pre-register success metrics

Use `templates/metrics_prereg.md`. For each claim component, record: what is measured, sample size (from power analysis), test to be used, PASS threshold, FAIL threshold, what an inconclusive result triggers.

Pre-registration is fixed at this stage. "Inconclusive" is a legitimate outcome and must have its own response plan written down now; waiting to see the data and then deciding what to do defeats pre-registration's purpose.

### 4. Enumerate controls

Three categories, each with a specific sample, expected outcome, and pass criterion:

- **Positive control.** A case where the expected signal should appear strongly, validating that the pipeline can see it.
- **Negative control.** A case where the expected signal should not appear, validating that the pipeline does not hallucinate.
- **Null control.** The measurement with the new physics switched off. Recovers the baseline.

A plan with fewer than three control types has a hole; ask what is being cut and why.

### 5. Power analysis

`scripts/power_analysis.py` handles two-sample t-test, one-way ANOVA, and linear regression (the most common cases). Inputs: effect size (from Stage 6 prediction), noise (from the observable's noise model), test (from question type), desired alpha and power.

Output: required sample size. Compare against what is feasible (beamtime, reagent cost, sample availability). If required N exceeds feasible N, the plan is not ready: either tighten the measurement (reduce noise, increase signal, more efficient test), rescope the claim, or accept the risk explicitly and log it.

Computing "we can achieve N=10 feasibly and we need N=50 for power" is better information than not checking, even if the outcome is uncomfortable.

### 6. Design-of-experiments matrix

`templates/doe_matrix.md` covers selection among full factorial, fractional factorial, response surface, and Latin hypercube. `scripts/doe_generator.py` produces the concrete run list.

Design patterns match question types roughly: mechanism work often benefits from fractional factorial to separate main effects from interactions; design work often benefits from response surface to find an optimum; measurement work often benefits from full factorial on a small set of factors to characterize behavior thoroughly.

After generation, verify: factor balance, randomization, feasibility. A beautifully balanced design that requires 500 runs at 1 shift per run is not feasible beamtime planning; it is a wishlist.

### 7. Multimodal sequencing (when applicable)

If the plan involves Rheo-SAXS-XPCS plus simulation, or any other multi-modality setup, sequence the modalities so early results inform later design. `references/multimodal_sequencing.md` has the principles.

Short version: cheapest and most diagnostic first. Simulation, then benchtop characterization, then benchtop rheology, then static scattering, then the expensive Rheo-XPCS shifts. Decision points between each adjacent pair with a pass criterion for advancing.

### 8. Risk register

`templates/risk_register.md`. Each risk has a probability, impact, mitigation, and crucially an **early signal**: what, in month 1 or 2, would reveal this risk is materializing before it becomes fatal?

The early-signal column is what Stage 8 (premortem) will loop back to. Every H-priority risk should have a signal that can be checked early. A risk with no early signal means the team will not know until the damage is done, which is the worst case.

### 9. Timeline and resource estimate

Beamtime shifts, sample preparation effort, analysis time, writeup time, key dependencies (collaborator inputs, external datasets). Realistic estimates include buffer. 60-70% of scheduled beamtime on the primary measurement is typical; the rest is alignment, calibration, and recovery from the sample issues that the cheap modalities missed.

### 10. Write, lint, hand off

`artifacts/07_plan.md` structure: executive summary (two paragraphs); capability map with margin annotations; pre-registered metrics; controls; power analysis; DoE matrix; multimodal sequencing if applicable; risk register; timeline; open questions for Stage 8. Style-lint. Update `_state.yaml`.

## Failure modes worth naming

- **Setting success metrics after looking at preliminary data.** This is how pre-registration dies; the fix is to set them now, accepting that the Stage 6 prediction is the best available guide.
- **Skipping the capability map because "we know our instrument."** Familiarity bias. The 3× check catches surprises routinely even for well-known instruments, especially when a new sample class or new observable is involved.
- **A risk register of easily-mitigated risks only.** Risks that cannot be fully mitigated are the important ones; omitting them does not make them go away, it just deprives the premortem of its inputs.
- **A DoE matrix that was not generated by the script.** "243 runs in a balanced full factorial" on paper often becomes "about 200 runs if we skip the batch effects" in practice. Generate the concrete list, inspect it, confirm it is what you actually plan to do.

## Templates, scripts, references

- `templates/instrument_capability_map.md`: central artifact; capability-vs-signal with margins
- `templates/metrics_prereg.md`: pre-registration format
- `templates/doe_matrix.md`: DoE layout selection guide
- `templates/risk_register.md`: P × I × mitigation × early-signal
- `scripts/power_analysis.py`: common-test power calculations (scipy-based)
- `scripts/doe_generator.py`: generates DoE tables
- `references/multimodal_sequencing.md`: coordinating rheology / scattering / simulation
- `references/aps_rheo_xpcs_capabilities.md`: 8-ID-I baseline

## Handoff to Stage 8

Stage 8 reads the full plan and writes the failure narrative. Its most important input is the risk register's early-signal column. Stage 8 turns those signals into concrete milestones inserted into this plan's timeline. The feedback loop is what makes the risk register a living document rather than a compliance exercise. If Stage 8 does not cause any revisions to this plan, something went wrong upstream.
