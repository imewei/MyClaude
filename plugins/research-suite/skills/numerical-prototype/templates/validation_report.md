# Validation report

Structure for the three validation passes required by Stage 6. Each pass has a pass/fail outcome and supporting evidence. Failure of any pass blocks advance to Stage 7.

## Pass 1: Analytic-limit recovery

**Limit taken:** [describe; e.g., "interaction_strength = 0, recovering free Brownian motion"]

**Reference solution:** [analytic result; e.g., "MSD = 2 D t in 1D"]

**Numerical result:** [what the solver produced]

**Relative error:** [|numerical - reference| / |reference|]

**Tolerance:** [e.g., 1e-3 for stochastic, 1e-6 for deterministic]

**Outcome:** [PASS / FAIL]

**Evidence:** [path to plot showing numerical vs analytic; timestep sensitivity if relevant]

**If FAIL:** [what is being done about it; do not proceed to Stage 7 until this passes]

---

## Pass 2: Synthetic benchmark

**Benchmark problem:** [describe a problem with a known analytic answer that exercises the new physics, not just the limit]

**Why this benchmark:** [why it is a meaningful test of the implementation, beyond the limit recovery]

**Analytic answer:** [exact result]

**Numerical result:** [what the solver produced, with uncertainty bound]

**Relative error:** [with tolerance]

**Outcome:** [PASS / FAIL]

**Evidence:** [plots, scripts/bench_*.py output]

**If FAIL:** [as above]

---

## Pass 3: Convergence study

**Resolution parameter varied:** [e.g., timestep, grid spacing, particle count]

**Values tested:** [at least three; e.g., dt = 1e-2, 1e-3, 1e-4]

**Richardson extrapolate:** [estimated true value]

**Convergence rate:** [empirical order; compare to expected order from the integrator]

**Observable at finest resolution:** [value ± uncertainty]

**Is the finest-resolution result within the extrapolated uncertainty of the Richardson estimate?** [yes / no]

**Outcome:** [PASS / FAIL]

**Evidence:** [convergence plot showing observable vs resolution, log-log]

**If FAIL:** [as above]

---

## Summary

- Pass 1 (limit recovery): [pass / fail]
- Pass 2 (synthetic benchmark): [pass / fail]
- Pass 3 (convergence): [pass / fail]

All three must pass before the predicted observable is emitted as Stage 6 output.

## Timing and resources

- Wall-clock for full validation suite: [seconds / minutes / hours]
- Hardware used: [CPU / single GPU / etc.]
- Peak memory: [approximate]

This helps Stage 7 estimate whether the prototype can be run at the scales the experimental design implies.
