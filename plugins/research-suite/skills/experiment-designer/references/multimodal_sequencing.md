# Multimodal sequencing

When the plan involves more than one measurement modality (for example Rheo-SAXS-XPCS with supporting simulation), the sequence matters. Running expensive modalities first wastes resources on samples that cheap characterization would have filtered out. Running modalities independently rather than adaptively misses the opportunity for early results to sharpen later design.

## The ordering principle

Cheapest and most diagnostic first. Most expensive and most specific last.

A typical order for Rheo-SAXS-XPCS work:

1. **Simulation with pre-measurement parameters** (hours to days, computed locally). Confirms the predicted observable has the expected magnitude given the nominal system parameters. If not, revisit Stage 6 before any physical measurement.

2. **Sample characterization** (hours, bench chemistry). Volume fraction, particle size distribution, zeta potential, viscosity at low shear. Confirms the sample is what you think it is. Samples that fail this gate do not proceed.

3. **Benchtop rheology** (hours per sample). Oscillatory sweep, strain sweep, flow curve. Cheaper than beamtime by orders of magnitude. Confirms the rheological signature the predicted observable depends on is present.

4. **Static SAXS** (if available, hours, often at a dedicated beamline). Confirms structural assumptions underpinning the predicted observable.

5. **Dynamic XPCS or Rheo-XPCS** (the expensive beamtime). The experiment that tests the claim. By now, the sample has been validated, the rheology is known, and the simulation predictions have been sharpened.

6. **Post-hoc simulation with measured parameters** (hours to days). Closes the loop: simulate with the actually-measured inputs and compare to the actually-measured output.

## Decision points between modalities

Between each pair of modalities, place a decision point with an explicit criterion:

- After characterization: does the sample match the nominal spec within tolerance? If not, make a new sample.
- After benchtop rheology: is the rheological signature in the expected range? If not, the claim's premise may be wrong for this sample; reconsider before using beamtime.
- After static SAXS: is the structure factor consistent with the assumption underlying the predicted observable? If not, the observable extraction may not work as predicted.
- After initial XPCS runs: does the signal look like the predicted observable at all? If not, stop the campaign, revisit Stage 6.

## Time budget

Beamtime is the scarcest resource. Budget so that only 60-70% of scheduled shifts are on the primary measurement; the rest is buffer for:
- Alignment and calibration on day 1
- Unexpected sample issues that the cheap modalities missed
- Repeat measurements if signal quality is lower than predicted
- Exploratory measurements if primary results suggest a follow-up

## Parallelism vs. sequencing

Two modalities can run in parallel only when:
- They do not compete for the same sample or instrument
- Early results from one would not change the design of the other
- The second modality's design is robust to plausible outcomes of the first

Otherwise, sequence. Parallelism is tempting because it feels efficient, but it forfeits the sharpening that intermediate results provide.

## For simulation-heavy plans

If the plan is primarily simulation with experimental validation, the analogous ordering is:
1. Analytic and limit-case checks (pencil and paper)
2. Small-scale prototype runs (laptop)
3. Parameter sweep at moderate fidelity (workstation or small cluster)
4. High-fidelity runs at target parameters (HPC)
5. Experimental validation on a small number of samples

The principle is the same: cheapest and most diagnostic first.

## For design-oriented plans

If the plan is building something that should produce a target state, the ordering is:
1. Validation of the control algorithm on a simulated system (cheap)
2. Open-loop demonstration on the real system (moderate)
3. Closed-loop demonstration on the real system (the point)
4. Robustness testing under perturbations (last)

## Writing the sequencing in the plan

In the 07_plan.md artifact, the sequencing goes in its own section with:
- Ordered list of modalities and their estimated duration
- Decision point between each adjacent pair, with the criterion that advances or halts
- Total estimated duration with contingency
- Named stopping conditions (when does the whole plan halt, not just one stage)
