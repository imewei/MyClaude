# Project log

The orchestrator (`research-spark`) appends to this file on every stage transition, artifact revision, and notable decision.

## Format

One entry per line. YAML-compatible block entries for decisions.

```
[2026-04-18T14:22] stage 3 complete; advanced to stage 4-5
[2026-04-18T15:30] artifact revised: 03_claim.md v2 (reviewer2 pass added)
[2026-04-18T16:45] decision:
    stage: 4
    what: selected Fokker-Planck framework over Langevin
    why: predicted observable is the stationary distribution, not a trajectory
    reversible: yes
```

## What to log

- Stage transitions (forward or backward)
- Artifact version bumps, with one-line reason
- Decisions at named decision points (see `_state.yaml`)
- Abandoned directions (why they were abandoned; useful when the same wall gets hit again in 6 months)
- Every override of a pipeline invariant (e.g., "proceeded to Stage 3 with N=5 steelmanned papers, below default threshold of 8, because adjacent literature genuinely is thin")

## What not to log

- Routine file saves
- Tool-call noise
- User preferences (those go in memory, not here)

## Retention

The log is never truncated. It is append-only. Old entries are the single best source of institutional memory across a project's life.
