# Fermi feasibility screen

A cheap check, done before drafting the claim: is the target effect even in a
reachable regime? This is order-of-magnitude reasoning, not derivation:
Stage 4-5's `dimensional_analysis.md` does the rigorous Buckingham Pi version
once the claim is fixed. Budget minutes, not hours.

## Fermi table

Two to four back-of-the-envelope estimates, whichever the question type
calls for:

| Quantity | Estimate | Method |
|---|---|---|
| Expected signal magnitude | order of magnitude, with units | scaling argument or known-analogue value |
| Signal-to-noise / contrast ratio | signal estimate ÷ known noise floor | instrument spec sheet or prior measurement |
| Computational complexity bound | operation count or memory, order of magnitude | naive algorithm complexity × problem size |
| Time-to-solution | wall-clock estimate | complexity bound ÷ known hardware throughput |

**Worked example (rheology/XPCS).** Claim candidate: a spectral gap in the
stress-response operator collapses measurably before flocculation onset.

- Expected signal magnitude: gap collapse ~1 order of magnitude in eigenvalue,
  from linear-stability scaling of the bond-exchange rate against shear rate.
- Signal-to-noise: current XPCS two-time correlation noise floor is ~5% at
  1 ms resolution; a 1-order-of-magnitude gap collapse is >>5%, so the signal
  clears the floor by roughly 20×.
- Time-to-solution (if a prototype is needed later): a 10^4-particle Brownian
  dynamics run at 10^6 steps is ~10^10 particle-steps, a few hours on one GPU.

## Dimensionless ratio

Name at least one governing nondimensional group that defines the regime
where the phenomenon should appear at all: a timescale ratio, force-balance
ratio, or signal-to-background ratio. This is the single most useful output
of the screen: it tells you *where* in parameter space to look, not just
*whether* to look.

**Worked example.** Deborah number De = (bond-exchange relaxation time) /
(shear timescale). The predicted spectral-gap collapse should appear for
De ~ O(1); at De >> 1 the network behaves elastically and the gap should be
absent, at De << 1 it behaves as a simple fluid and there is no gap to
collapse. This pins the experimental regime before Stage 7 designs anything.

## Exit gate

The target phenomenon must occupy a non-zero, reachable domain in the
dimensionless ratio's range, and the Fermi table's signal must clear the
noise floor or fall within feasible compute/time budgets.

```
Effect size vs. noise floor: [estimate] vs [floor] -> margin [Nx]
Compute/time vs. feasible budget: [estimate] vs [budget] -> margin [Nx] or [within/exceeds]
Reachable regime (dimensionless ratio): [name] ~ [range] where the effect should appear
Verdict: [PASS - draft the claim / FAIL - return to Stage 1-2]
```

If margin is comfortably above 1x on every row, proceed to drafting the
claim. A thin margin (close to 1x) is not an automatic fail, but the claim
should say so explicitly rather than presenting the effect as safely
observable.

## Common repair moves

- **Effect below the noise floor by orders of magnitude.** The idea may
  still be real but is not testable as scoped. Either find an amplification
  mechanism (larger sample, longer averaging, different observable) or
  return to Stage 1 and reframe around a larger effect.
- **Compute bound exceeds any feasible budget.** Look for a reduced-order
  model, a smaller representative system, or an analytic approximation that
  captures the mechanism without the full computation; that becomes the
  "minimal non-trivial model" Stage 4-5 formalizes.
- **No clear dimensionless ratio emerges.** Usually means the mechanism is
  still underspecified. Return to Stage 1's mechanistic framing and name the
  competing timescales, forces, or energy scales explicitly.
