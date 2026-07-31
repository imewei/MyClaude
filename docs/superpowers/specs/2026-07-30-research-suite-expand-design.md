# research-suite Expand & Optimize Design

**Date:** 2026-07-30
**Status:** Draft — pending user review
**Builds on:** `2026-05-05-plugin-scicomp-redesign.md` (implemented — 3
research-suite commands already landed) and
`2026-07-30-science-suite-expand-design.md` (Draft — pending user review,
same as this spec; proposes adding `continuum-mechanics-engineer` and
extending `statistical-physicist`, both of which this spec wires into
research-suite's cross-suite routing once that spec ships — see §6 for
sequencing).
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

Add a "SEE ALSO" block in the exact two-line blockquote format used by
`plugins/dev-suite/commands/smart-debug.md`:

```
> **SEE ALSO:** <one-line contrast with the generic tool>
> Use this command/skill for <one-line positive scope statement>.
```

to:

- `plugins/research-suite/commands/lit-review.md`: "For a general literature
  search without structured claim extraction or PRISMA/GRADE synthesis, use
  `ecc:scientific-thinking-literature-review` or `ecc:deep-research`. Use
  this command for research-suite's structured topic-scan pipeline
  (claim extraction, evidence synthesis, gap identification) via the
  `research-practice` hub." (Do not describe `lit-review` as
  "artifact-gated" — that term applies to the separate `research-spark`
  pipeline, not this command; `research-practice`'s own description
  explicitly distinguishes the two.)
- `plugins/research-suite/skills/research-quality-assessment/SKILL.md`: "For
  general scholarly-work feedback, use
  `ecc:scientific-thinking-scholar-evaluation`. This skill applies
  research-suite's specific quality rubrics (PRISMA/GRADE/CONSORT/STROBE)."
- `plugins/research-suite/skills/evidence-synthesis/SKILL.md` and
  `plugins/research-suite/skills/landscape-scanner/SKILL.md`: reference
  `ruflo-goals:deep-researcher`/`research-synthesize` for general multi-source
  research synthesis outside the research-spark pipeline's artifact
  contract.

`ecc:*` and `ruflo-goals:*` are external plugins from other marketplaces —
not part of this repo's `.claude-plugin/marketplace.json` (which lists only
`agent-core`, `dev-suite`, `science-suite`, `research-suite`). Their
names/availability cannot be repo-validated; the SEE ALSO text is
informational framing for environments where they happen to be installed,
not a hard dependency.

No functional change — description/routing text only, same pattern already
proven in dev-suite.

## 4. Cross-suite routing tie-in to science-suite's new domains

The binary JAX-or-Julia framing lives in `paper-implement.md`/`replicate.md`
today — both commands' routing prose ("cross-delegates to `jax-pro` (JAX) or
`julia-pro` (Julia)") and their `argument-hint`/`description` frontmatter
constrain `--framework` to `jax|julia`. (`research-expert.md`'s own
delegation table currently lists `ml-expert`/`simulation-expert`/
`sci-workflow-engineer`/`python-pro` and doesn't mention `jax-pro`/
`julia-pro` at all — it has no existing JAX/Julia framing to fix; this spec
is the first time its delegation list gains implementation-specialist
entries.)

**Framework-independent routing** (no frontmatter/interface change): keep
`--framework` meaning "implementation language for the chosen specialist,"
not "which specialist." `research-expert`'s methodology-parsing step (already
part of its job) determines the paper's *domain* and picks the specialist;
`--framework` (where applicable) only disambiguates JAX vs. Julia within a
numerics-based specialist. This keeps §5's "routing text only, no structural
additions" true — no `argument-hint`/`description` edits to
`paper-implement.md`/`replicate.md` are needed, only their routing-prose
lines and `research-expert.md`'s delegation table.

Update the routing lines in `paper-implement.md` and `replicate.md` (and add
a new delegation table to `research-expert.md`, which has none of these
entries today) to route implementation to whichever specialist matches the
paper's method:

- **`jax-pro`** / **`julia-pro`** — general JAX/Julia numerics papers
  (existing routing, unchanged).
- **`continuum-mechanics-engineer`** — for papers whose core method is FEM/
  FEA, constitutive modeling, rheology/DMA, transient-network (CAN/vitrimer)
  theory, or nanocomposite mechanics.
- **`statistical-physicist`** (extended scope) — for papers on glass/
  jamming/collective phenomena or physical/energy-based learning in
  disordered systems, in addition to its existing phase-transition/
  correlation/MCMC-diagnostics routing.
- **`pinn-engineer`** — for papers whose core method is a physics-informed
  neural network or NeuralPDE-style solve.
- **`simulation-expert`** — for papers whose core method is a physics/HPC
  simulation (MD, agent-based, or similar) rather than a differentiable-
  programming implementation.

`replicate.md`'s third routing hop (`quality-specialist` for numerical
validation gates — confirmed present in `dev-suite`'s `plugin.json`) is
unaffected — that's a dev-suite agent kept in the trim spec, still valid
regardless of which science-suite specialist did the implementation.

## 5. Validation / acceptance criteria

- `make validate` passes with zero new errors. Note: neither `make validate`
  nor `make verify`/`verify-fast` invoke `xref_validator.py` — it is a
  separate, not-currently-make-wired check and must be run explicitly (next
  bullet); passing `make validate` alone does not confirm the new
  cross-references resolve.
- `PYTHONPATH=. python3 tools/validation/xref_validator.py` — new
  cross-references to `continuum-mechanics-engineer` resolve (agent exists
  once the science-suite spec ships; sequence implementation accordingly —
  see §6). Reference it in a form the validator's regexes actually match
  (e.g. `` agent `continuum-mechanics-engineer` `` or the absolute path
  `plugins/science-suite/agents/continuum-mechanics-engineer.md`) — a bare
  `` `continuum-mechanics-engineer` `` mention does not match any of
  `xref_validator.py`'s agent-reference patterns (`agent:\s*name`, `@name`,
  `` agent `name` ``) and will silently fail to register as a
  cross-reference at all, let alone fail loudly.
- `uv run pytest` passes unchanged.
- `paper-implement.md`/`replicate.md`/`research-expert.md` routing tables
  list all 6 candidate implementation specialists (not just
  jax-pro/julia-pro) with a one-line disambiguation each.
- No change to `research-suite`'s agent count, command count, or hub-skill
  registration in `plugin.json` — this spec only edits routing text and adds
  SEE ALSO framing, no structural additions.

## 6. Implementation sequencing note

This spec's §4 changes depend on `continuum-mechanics-engineer` actually
existing (from the science-suite spec, itself still Draft as of this
writing). Implementation order: science-suite spec ships first, then this
spec's routing updates, or both land in the same implementation pass with
science-suite's agent file created before research-suite's routing tables
reference it — either works, but only if the references use a form
`xref_validator` actually recognizes (see §5's note on reference form). Using
a bare backtick mention instead would not "fail loudly" if research-suite's
edits land first in isolation — the validator simply wouldn't register it as
a cross-reference to check, so the dangling reference would pass silently.

## 7. Out of scope / follow-ups

- No new research-suite domains, tracks, agents, or commands.
- No model-tier changes.
- Deeper trim/dedup pass against ecc's research tools (beyond the SEE ALSO
  framing in §3) — not requested; framing-only was the user's explicit
  choice over a heavier rework.
