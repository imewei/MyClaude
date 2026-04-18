---
name: theory-scaffold
description: Stages 4 and 5 of the research-spark pipeline (merged because derivation and formalization interleave in practice). Derives a theoretical framework from first principles, formalizes it in LaTeX, checks known limits, and identifies governing dimensionless groups. Triggers on phrases like "derive the equations", "write down the theory for this claim", "formalize the model", "check the limits", "identify the dimensionless groups", "build the Fokker-Planck / Langevin / GLE / variational structure for this", "work out the math for the claim", or after Stage 3 completes. Also triggers on SciML gray-box specifications where the user needs to draw the boundary between physics-prescribed and learned components. Enforces a stepwise derivation protocol that blocks multi-step mathematical leaps, because that is where symbolic errors concentrate. Produces 04_theory.md plus a compilable 05_formalism.tex.
---

# theory-scaffold

Stages 4-5. Derives the theoretical framework and produces a formalized mathematical statement.

## Why merged, and why this stage exists

Deriving the framework and formalizing it mathematically interleave constantly. A formalization step exposes a missing assumption; the derivation is revisited; a new dimensionless group emerges; the formalization updates. Splitting the two stages creates artificial handoffs. This skill handles both.

Claims without theoretical scaffolding are hard to test because their predictions are imprecise. A spectral gap "should" drop before flocculation, but by how much, over what timescale, under what scaling? Without theory, that is a hunch; with theory, it is a prediction with quantified magnitude and dependencies that Stage 6 can test and Stage 7 can design a measurement for.

## Workflow

### 1. Name the governing framework and the assumptions

From the claim (Stage 3) and the landscape (Stage 2), identify: conservation laws in play (mass, momentum, energy, charge, particle number); symmetries (translation, rotation, time-reversal, gauge); variational structure (action, free energy functional); stochastic framework (Langevin, Fokker-Planck, generalized Langevin with memory, master equation); linear vs nonlinear regime.

Write these at the top of `05_formalism.tex` as the "Assumptions and framework" section. Every downstream step references back.

Enumerate assumptions with explicit validity regimes and the cost of violation (qualitative / quantitative / catastrophic). An assumption without a regime is a trap waiting to be sprung.

### 2. Load and follow the stepwise derivation protocol

`templates/stepwise_derivation_protocol.md` is the most important piece of this skill. It enforces one conceptual step per invocation with a verification pass before the next is allowed. This is inconvenient in the short term and pays back consistently on any derivation longer than five steps.

The short version: one atomic operation (substitute, linearize, transform, integrate by parts, etc.) per step, followed by at least one verification (dimensional check, limit check, or sanity argument). The operation is named explicitly; "after some algebra" is not an operation.

Multi-step leaps hide sign errors and dropped terms in verbose LaTeX. Every stepwise-derivation bug tends to come from trying to save time here.

### 3. Derive the core equations

For each step, emit into `05_formalism.tex` with a LaTeX comment indicating the step number and the operation. `templates/derivation_skeleton.tex` provides the scaffold.

Each step's output is a labeled equation that downstream work (Stage 6 reconciliation, the citation graph in the eventual paper) can reference.

### 4. Known-limit checks

For every core equation, switch off the new physics (or take a known limit) and verify the theory reduces to the reference result. `scripts/limit_check.py` does symbolic limits via SymPy; for limits that are qualitative or by dimensional argument, write the limit and target explicitly.

Failure modes this catches that are hard to spot visually:
- Sign errors that survived the derivation
- Dropped terms
- Coefficient errors (powers of 2 or pi, mainly)
- Implicit assumptions that were not listed in Step 1

If a limit does not reduce correctly, the error is in the derivation or in an un-stated assumption. Find which. Do not proceed until the limit passes; a failed limit that is rationalized propagates forward silently.

### 5. Dimensional analysis

Identify governing dimensionless groups via Buckingham Pi. `templates/dimensional_analysis.md` has the worksheet and worked examples. `scripts/dimensional_audit.py` extracts equations from the LaTeX for an audit table.

The groups are useful downstream because:
- They tell Stage 6 which parameters to sweep independently and which are redundant
- They tell Stage 7 which experimental conditions access which regime
- If the Pi-theorem count does not match the variable count minus the dimension count, a variable was miscounted: hunt down the discrepancy

### 6. Gray-box boundary (SciML only)

If the claim involves learned components, specify the boundary explicitly using `templates/graybox_boundary.md`: what is prescribed from physics, what is learned, how each learned piece is validated standalone before coupling.

Gray-box models without per-component validation fail in ways that are hard to diagnose once everything is coupled, because the loss function cannot distinguish a theoretical error, a learned-component generalization failure, and a data-preprocessing bug. The per-component pass-criterion is the mechanism that separates these.

### 7. Write `04_theory.md` (the prose companion)

`04_theory.md` is the document the user re-reads to remember what the theory says. It points into the `.tex` for details.

Structure: framework and assumptions; core result with physical interpretation; derivation outline with pointers to LaTeX sections; limits verified; dimensionless groups; gray-box spec if SciML; open questions for Stage 6.

### 8. Compile, lint, hand off

Run `scripts/latex_compile_check.sh` on the `.tex` file. Fix undefined references or compile errors. Style-lint `04_theory.md`. Update `_state.yaml`.

## Failure modes worth naming

- **Leaping multiple steps at once because "the algebra is obvious."** This is where sign errors live. The protocol exists for the cases where the algebra feels obvious but is not.
- **Skipping the limit check because "the derivation is clean."** Clean derivations routinely have dropped factors. Limit checks are cheap and catch them.
- **Implicit dimensionless groups.** Write them down even when they feel obvious. The explicit list is what Stage 6 consumes; a mental list does not transfer.
- **Gray-box boundary without per-component validation plans.** A learned component that "learns what the physics does not" is an un-validated component. Specify what would falsify it in isolation.

## Templates and scripts

- `templates/stepwise_derivation_protocol.md`: one-step-at-a-time discipline
- `templates/derivation_skeleton.tex`: LaTeX scaffold
- `templates/dimensional_analysis.md`: Buckingham Pi worksheet with worked examples
- `templates/graybox_boundary.md`: SciML boundary spec
- `scripts/dimensional_audit.py`: extracts equations from LaTeX for audit
- `scripts/limit_check.py`: SymPy-based symbolic limit verification
- `scripts/latex_compile_check.sh`: compile plus undefined-reference check

## Handoff to Stage 6

Stage 6 takes `05_formalism.tex` as the authoritative equation set and implements it in JAX. Stage 6's reconciliation check compares symbols in the code against symbols in the LaTeX. If they disagree, either the theory or the code is wrong; finding out which is the point of running Stage 6 at all.
