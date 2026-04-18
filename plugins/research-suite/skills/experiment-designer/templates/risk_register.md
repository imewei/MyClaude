# Risk register

Structured list of risks, with explicit probability, impact, mitigation, and early-signal columns. The early-signal column is the hook Stage 8 (premortem-critique) will use to insert milestones back into the plan.

## Format

| # | Risk | P | I | Priority | Mitigation | Early signal |
|---|------|---|---|----------|------------|--------------|
| 1 | Sample batch-to-batch variability larger than expected | M | H | HIGH | Pre-characterize every batch; reject batches outside 2σ of reference | Month 1: run 3 batches through characterization; flag if any exceeds range |
| 2 | XPCS detector count rate insufficient at target q-range | M | H | HIGH | Upgrade to faster detector; or accept lower q-range | Month 1: test-shot with reference sample; measure actual count rate vs spec |
| 3 | Gray-box closure fails to generalize beyond training range | M | M | MEDIUM | Include out-of-range test cases in validation; fall back to prescribed physics when confidence is low | Month 2: run inference on synthetic OOD data; report generalization loss |
| 4 | Collaborator sample delivery delayed | L | M | LOW | Parallel internal sample preparation as fallback | Monthly: check status with collaborator |
| 5 | Predicted spectral gap smaller than detectability | L | H | MEDIUM | Increase flux or ensemble more realizations; worst case, rescope claim to a larger-amplitude observable | Month 1: run prototype at conservative parameters; check margin |

## Probability and impact scale

- **P (probability): L / M / H** corresponding roughly to <20%, 20-60%, >60%
- **I (impact): L / M / H** corresponding to "delays but does not kill the project" / "forces substantial rescoping" / "kills the project"

Priority is derived: any H in either P or I makes the row at least MEDIUM; H in both is HIGH. Do not compute a numerical score; the two-letter encoding is enough.

## What belongs here

- Risks that could kill or significantly damage the project
- Risks that would change the plan if they materialized
- Risks that cannot be fully mitigated (these are especially important)

## What does not belong here

- Generic platitudes ("we might be wrong about something")
- Risks with no concrete mitigation AND no early signal (these are either not real risks or need more thought)
- Small process risks ("the shared drive might go down") unless they genuinely threaten the project

## Why the early-signal column is critical

Stage 8 (premortem-critique) writes the failure narrative from the plan. Its output is a list of failure modes with early signals. Those early signals get inserted into the plan as milestones, making the risk register a living document rather than a compliance exercise.

If a risk has no early signal, it means the team will not know the risk materialized until the damage is done. That is the worst case. Every H-priority risk should have an early signal that can be checked in month 1 or 2.

## Update discipline

- The register is updated whenever a risk materializes (note the date it became an issue)
- Risks that are fully mitigated can be marked RESOLVED with a date; do not delete them
- New risks discovered during execution are added, not replaced

The register is append-only in spirit. The historical record matters for learning across projects.
