# research-suite Expand & Optimize Design

**Date:** 2026-07-30
**Status:** Draft — pending user review
**Builds on:** `2026-05-05-plugin-scicomp-redesign.md` (implemented — 3
research-suite commands already landed) and
`2026-07-30-science-suite-expand-design.md` (approved, adds
`continuum-mechanics-engineer` and extends `statistical-physicist`, both of
which this spec wires into research-suite's cross-suite routing).
**Third and final sub-project** of the marketplace redesign.

---

## 1. Objective

Unlike science-suite, research-suite has no stale version literals and no
mis-tiered agents to fix — both `research-expert` and
`research-spark-orchestrator` are already opus, and that's confirmed correct
(§2). The optimize/expand work here is narrower and different in kind:

1. **Overlap framing** — this machine also loads `ecc:scientific-thinking-
   literature-review`, `ecc:scientific-thinking-scholar-evaluation`,
   `ecc:deep-research`, and `ruflo-goals:deep-researcher`/`research-synthesize`
   — generic research/lit-review tools. research-suite's actual
   differentiator (structured, artifact-gated pipelines tied to JAX/Julia
   implementation — the 8-stage research-spark pipeline, peer-review-to-
   `.docx`, paper-implement's routing to `jax-pro`/`julia-pro`) isn't
   duplicated anywhere, but nothing currently tells a trigger-matcher that.
   Add the same "SEE ALSO" framing dev-suite's `smart-debug`/`test-generate`/
   `eng-feature-dev` already use.
2. **Cross-suite routing tie-in** — science-suite's expand spec adds
   `continuum-mechanics-engineer` and broadens `statistical-physicist`'s
   scope (transient networks, nanocomposites, glass/collective phenomena,
   physical learning). research-suite's paper-reproduction commands
   (`paper-implement`, `replicate`) only know about `jax-pro`/`julia-pro`
   today — they need to route papers in those new domains to the right
   agent, or reproduction attempts in continuum mechanics/rheology/
   materials science will silently fall through to the wrong specialist.

No new domains for research-suite itself — user-confirmed the 3 tracks
(scientific-review, research-spark, research-practice) already cover the
research lifecycle adequately.

## 2. Model tiers: no change

`research-expert` (opus) and `research-spark-orchestrator` (opus) stay as
they are. Considered downgrading the orchestrator to sonnet (it's
state-machine/artifact-gate enforcement, similar to `simulation-expert`'s
orchestration-heavy case in science-suite) — rejected: judging whether an
8-stage pipeline's intermediate artifact (a falsifiable claim, a theory
scaffold) actually clears the gate to the next stage is a substantive
correctness judgment, not mechanical job orchestration like MD/HPC run
management. Recorded here so this isn't re-litigated in the next redesign
pass without a note of why it was already considered.

## 3. Overlap framing additions

Add a "SEE ALSO" line (matching the exact pattern in
`plugins/dev-suite/commands/smart-debug.md`) to:

- `plugins/research-suite/commands/lit-review.md`: "For a general literature
  search without the structured evidence-grading/citation-check pipeline,
  use `ecc:scientific-thinking-literature-review` or `ecc:deep-research`.
  This command produces a structured, artifact-gated review tied to
  research-suite's methodology hub."
- `plugins/research-suite/skills/research-quality-assessment/SKILL.md`: "For
  general scholarly-work feedback, use
  `ecc:scientific-thinking-scholar-evaluation`. This skill applies
  research-suite's specific quality rubrics (PRISMA/GRADE/CONSORT/STROBE)."
- `plugins/research-suite/skills/evidence-synthesis/SKILL.md` and
  `plugins/research-suite/skills/landscape-scanner/SKILL.md`: reference
  `ruflo-goals:deep-researcher`/`research-synthesize` for general multi-source
  research synthesis outside the research-spark pipeline's artifact
  contract.

No functional change — description/routing text only, same pattern already
proven in dev-suite.

## 4. Cross-suite routing tie-in to science-suite's new domains

Update the routing lines identified in `paper-implement.md` and
`replicate.md` (currently: "Routes to `research-expert` ... then
cross-delegates to `jax-pro` (JAX) or `julia-pro` (Julia) for
implementation") to include the two new/extended science-suite
specialists:

- **`continuum-mechanics-engineer`** — for papers whose core method is FEM/
  FEA, constitutive modeling, rheology/DMA, transient-network (CAN/vitrimer)
  theory, or nanocomposite mechanics.
- **`statistical-physicist`** (extended scope) — for papers on glass/
  jamming/collective phenomena or physical/energy-based learning in
  disordered systems, in addition to its existing phase-transition/
  correlation/MCMC-diagnostics routing.

Update `research-expert`'s own delegation list (methodology parsing already
identifies the paper's domain) to route implementation to whichever of
`jax-pro`/`julia-pro`/`continuum-mechanics-engineer`/`statistical-physicist`/
`pinn-engineer`/`simulation-expert` actually matches the paper's method,
rather than the current binary JAX-or-Julia framing which assumes every
reproduced paper is a JAX/Julia numerics paper.

`replicate.md`'s third routing hop (`quality-specialist` for numerical
validation gates) is unaffected — that's a dev-suite agent kept in the trim
spec, still valid regardless of which science-suite specialist did the
implementation.

## 5. Validation / acceptance criteria

- `make validate` passes with zero new errors.
- `PYTHONPATH=. python3 tools/validation/xref_validator.py` — new
  cross-references to `continuum-mechanics-engineer` resolve (agent exists
  once the science-suite spec ships; sequence implementation accordingly —
  see §6).
- `uv run pytest` passes unchanged.
- `paper-implement.md`/`replicate.md`/`research-expert.md` routing tables
  list all 6 candidate implementation specialists (not just
  jax-pro/julia-pro) with a one-line disambiguation each.
- No change to `research-suite`'s agent count, command count, or hub-skill
  registration in `plugin.json` — this spec only edits routing text and adds
  SEE ALSO framing, no structural additions.

## 6. Implementation sequencing note

This spec's §4 changes depend on `continuum-mechanics-engineer` actually
existing (from the science-suite spec). Implementation order: science-suite
spec ships first, then this spec's routing updates, or both land in the same
implementation pass with science-suite's agent file created before
research-suite's routing tables reference it — either works, but
`xref_validator` will correctly fail if research-suite's edits land first in
isolation.

## 7. Out of scope / follow-ups

- No new research-suite domains, tracks, agents, or commands.
- No model-tier changes.
- Deeper trim/dedup pass against ecc's research tools (beyond the SEE ALSO
  framing in §3) — not requested; framing-only was the user's explicit
  choice over a heavier rework.
