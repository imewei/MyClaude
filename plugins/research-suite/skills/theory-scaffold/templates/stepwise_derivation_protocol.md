# Stepwise derivation protocol

One conceptual step per invocation. Each step is verified before the next is allowed. This discipline is inconvenient in the short term and pays back consistently on any derivation longer than five steps.

## Why this protocol exists

Language models (and humans) produce sign errors, dropped terms, and coefficient mistakes most often when asked to leap from A to Z in one step. The verbosity of the intermediate LaTeX hides the error. The stepwise protocol splits the derivation into atomic operations, each of which is small enough to verify.

## The step template

```
### Step N: [one-clause description of the operation]

**Starting point:**
\begin{equation}
    [equation or principle going in]
\end{equation}

**Single conceptual operation:**
[One sentence. Examples: "Take the ensemble average of both sides."
"Substitute the linearized constitutive relation from Eq. (3)."
"Integrate by parts, assuming the boundary term vanishes."]

**Resulting equation:**
\begin{equation}
    [equation coming out]
\end{equation}

**Verification pass** (pick at least one):
- Dimensional check: [show both sides have the same units]
- Limit check: [show the equation reduces correctly in a known limit]
- Sanity argument: [qualitative check on magnitude, sign, or asymptotic behavior]

**Assumptions invoked in this step:**
- [list any assumptions used; cross-reference with the Assumptions section]

**Open questions raised by this step:**
- [anything that became visible at this step and was not visible before]
```

## Allowed operations per step

A "single conceptual operation" includes:

- Substitution of one quantity for another
- Taking an ensemble average, time average, or ergodic limit
- Linearization around a reference state
- Fourier, Laplace, or Mellin transform
- Integration by parts (one application)
- Change of variables (one variable)
- Expansion to a specific order in a small parameter
- Application of a named theorem (Stokes, divergence, fluctuation-dissipation, etc.)

What is *not* a single step:

- "Apply the standard derivation" (not atomic)
- "After some algebra" (hides the operations)
- Combined transform + substitution + linearization in one go
- A new equation with no stated operation connecting it to the previous one

## Verification pass: how much is enough

The verification pass is defense in depth. A dimensional check alone is weak; a limit check plus a sanity argument is strong. Pick verifications that would catch the most likely error for the operation in question.

- After a transform, check dimensions.
- After a linearization, check that the zeroth-order result is preserved.
- After an ensemble average, check that a conserved quantity is still conserved.
- After a change of variables, check that the Jacobian is correct.
- After an expansion, check that the zeroth-order term recovers the unexpanded case.

## Handling long derivations

If the derivation is 20+ steps, it is often worth introducing intermediate "summary" points every 5-7 steps:

- A summary collects the accumulated result so far
- Subsequent steps use the summary equation as their starting point
- The summary does not skip work; it just names what has been done

This keeps the artifact readable without compromising the atomic-step discipline.

## When a step's verification fails

1. Identify which verification failed and what it revealed.
2. Look at the step's inputs: was an assumption invoked that was not on the assumptions list? Add it.
3. Look at the step's operation: was more than one operation performed? Split into two steps.
4. Look at the step's result: is there a sign error, a factor of 2, or a missing term? Fix and re-verify.
5. If the failure cannot be resolved, the step is wrong. Back up one step and try a different operation.

Do not proceed past a failed verification. A failed verification is information; a false claim of success is a time bomb.

## Integration with SymPy

For symbolic derivations, `scripts/limit_check.py` (in the theory-scaffold skill) can automate the limit and dimensional checks. Use it liberally. Automation here is cheap because the checks are objective.
