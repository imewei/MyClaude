# science-suite Expand & Optimize Design

**Date:** 2026-07-30
**Status:** Draft — pending user review
**Builds on:** `2026-05-05-plugin-scicomp-redesign.md` (implemented — model
tiers, description trimming, mode-flag gating, 2 new commands already
landed) and `2026-07-30-agent-core-devsuite-trim-design.md` (approved,
supplies the 5 Python-tooling skills this spec absorbs).
**Second sub-project** of the marketplace redesign; research-suite expand
spec follows separately.

---

## 1. Objective

Two threads, per explicit user direction:

1. **Optimize for Opus 5 / Sonnet 5** — fix stale model-version literals left
   from the Opus-4.7 era, and re-audit all 11 agent model-tier assignments
   now that Sonnet 5 is materially stronger than the Sonnet 4.x generation
   the prior tiering was chosen against.
2. **Expand coverage** — both an accuracy audit of existing skill content
   against the user's pinned toolchain (global CLAUDE.md), and new coverage
   for a specific, user-identified gap cluster: continuum mechanics,
   rheology/DMA, FEM/FEA, constitutive equations, harmonic
   response/superposition, engineering mathematics, and data-driven
   modeling for materials — plus cooperative dynamics/collective
   phenomena/glass problems in soft matter, and graph theory.

## 2. Scope

In scope: `plugins/science-suite/` only. Absorbing the 5 Python-tooling
skills displaced from dev-suite (§3) is in scope; re-litigating dev-suite's
own trim is not (already specced separately).

## 3. Model-version staleness fix

- `plugins/science-suite/.claude-plugin/plugin.json`: drop the `"opus-4.7"`
  keyword; replace with generic tier language that won't go stale again
  (e.g. `"adaptive-model-tiering"` instead of pinning a version number in a
  keyword).
- `plugins/science-suite/README.md`: replace "Optimized for Claude Opus"
  and the `plugin.json` `description` field's "for Claude Opus" with
  language reflecting the actual per-agent opus/sonnet/haiku split (§4) —
  the suite is multi-tier, not opus-only, and hasn't been since the
  2026-05-05 redesign.

## 4. Model-tier re-audit (11 agents)

Confirmed unchanged (9): `jax-pro` (opus), `julia-pro` (opus),
`neural-network-master` (opus), `nonlinear-dynamics-expert` (opus),
`statistical-physicist` (opus), `julia-ml-hpc` (sonnet), `python-pro`
(sonnet), `sci-workflow-engineer` (sonnet), `ml-expert` (haiku).

**Two swaps (user-approved):**

| Agent | Current | New | Rationale |
|---|---|---|---|
| `simulation-expert` | opus | sonnet | MD/HPC orchestration (GROMACS/OpenMM/LAMMPS/multi-node) is engineering-heavy, not novel-math-heavy — Sonnet 5 handles this well. |
| `pinn-engineer` | sonnet | opus | Inverse-PDE and constrained-loss design is comparably deep to `julia-pro`'s SciML work; was under-tiered relative to peers doing similar-difficulty math. |

Net opus count unchanged (6 → 6), reallocated by actual task difficulty
rather than net budget change.

Plus the new agent added in §5 (opus — see rationale there): opus count
becomes 7 of 12 agents.

## 5. New agent: continuum mechanics / FEM / rheology

No existing agent owns weak-form PDE discretization or materials
constitutive modeling — `jax-pro`/`pinn-engineer` cover PDEs via
JAX/neural-operator methods, `simulation-expert` covers particle-based MD,
neither covers classical continuum/FEM methods.

**New agent:** `continuum-mechanics-engineer`
**Model:** opus (constitutive-model correctness is exact-math-critical per
CLAUDE.md principle 1 — theoretically exact math)
**Scope:** continuum mechanics, constitutive equations (linear/nonlinear
viscoelasticity, stress-strain relations), Dynamic Mechanical Analysis
(DMA), rheology (shear and extensional), harmonic response, time-temperature
superposition, Finite Element Modeling/Analysis (FEM/FEA — weak forms, mesh
considerations, convergence), engineering mathematics, and data-driven/
hybrid (physics + ML) modeling for materials. Delegates neural-PDE/
physics-informed approaches to `pinn-engineer`, particle/MD methods to
`simulation-expert`, pure JAX numerics to `jax-pro`.

New hub skill: `continuum-mechanics-and-rheology` (registered in
`plugin.json`, following the existing hub pattern), with sub-skills for
FEM/FEA, constitutive-equations, DMA/rheology, and
harmonic-response-superposition (discovered via the hub's routing table,
not individually registered — per the manifest rule in root CLAUDE.md).

## 6. Extend `statistical-physicist` — glass & collective phenomena

Cooperative dynamics, collective phenomena, and glass problems (jamming,
aging, random landscapes) in soft matter fit `statistical-physicist`'s
existing scope directly (phase transitions, correlations, fluctuation
theorems, soft matter) — no new agent. Extend its `statistical-physics-hub`
and/or `active-matter` skill routing to cover these explicitly; check for
overlap with existing `correlation-*` skill family before adding new
sub-skill files (avoid re-creating content that already exists under a
different name).

## 7. New skill: graph theory

Cross-cutting math (used by GNN work, `network-coupled-dynamics`, and
potentially by FEM mesh/microstructure-network representations in §5). Add
as a new sub-skill, not a new hub — wire routing references from
`neural-network-mathematics`, `network-coupled-dynamics`, and
`continuum-mechanics-and-rheology`'s hubs rather than registering it
top-level. Exact routing wiring is an implementation detail for the
follow-on plan, not a further design decision.

## 8. Python-tooling consolidation (absorbing dev-suite's 5 skills)

Checked against existing science-suite skills — this is a **merge-and-delete**,
not a copy-in; no new files needed:

| Incoming (from dev-suite trim) | Existing science-suite home | Action |
|---|---|---|
| `async-python-patterns` | `python-development` (already covers TaskGroups/structured concurrency) | Fold any unique content in, delete incoming file |
| `python-toolchain` | `python-development` (already covers uv-based packaging) | Fold any unique content in, delete incoming file |
| `python-packaging` | `python-packaging-advanced` (already covers uv/pyproject.toml/workspaces) | Fold any unique content in, delete incoming file |
| `uv-package-manager` | `python-packaging-advanced` | Fold any unique content in, delete incoming file |
| `python-performance-optimization` | `performance-tuning` (existing science-suite skill) | Fold any unique content in, delete incoming file |

Diff each incoming/existing pair during implementation — only port content
that's genuinely additive (e.g. a specific flag or workflow the existing
skill doesn't mention). Do not blindly concatenate.

## 9. Accuracy audit against CLAUDE.md toolchain

Before/alongside the above, spot-check existing skill content for staleness
against the user's pinned specifics (not a full line-by-line rewrite —
targeted verification):

- `consensus-mcmc-pigeons` / `turing-model-design`: reflect Pigeons DEV
  override (PR #409, commit `4d981068`) and DynamicPPL v0.40 correctly.
- `modeling-toolkit` / `sciml-modern-stack`: reflect the MTK v9 (`@sciml`/
  `@bayes`) vs MTK v11 (`@pinn`) split accurately, not a single version.
- `julia-graph-neural-networks`: reflect GNNLux/GNNGraphs/GNNlib DEV
  overrides against monorepo master, not a released version.
- `bifurcation-analysis`: must NOT recommend `BifurcationKit` (blocked on
  Julia 1.12, per CLAUDE.md prohibited list) — confirm it correctly routes
  to Python `tick` or existing skill guidance instead.
- `point-processes` equivalent content: confirm no reference to a Julia
  `PointProcesses` package (registry tagging pending, not usable) without
  the caveat.
- `julia-hpc-distributed` / `parallel-computing`: confirm no
  `SequentialMonteCarlo` recommendation in any multi-package Julia env
  (RNGPool SIGABRT on 1.12, per CLAUDE.md prohibited list).
- Any Flux.jl recommendation for new SciML neural closures should be
  flagged/corrected to Lux.jl per CLAUDE.md.

## 10. plugin.json updates

- Add `continuum-mechanics-engineer` to `agents` array.
- Add `continuum-mechanics-and-rheology` to `skills` array (new hub).
- Remove the 5 absorbed dev-suite skill names if they were ever
  cross-referenced by name anywhere in science-suite docs (they weren't
  registered here before, so likely a no-op, but check).
- Update `keywords` (drop `opus-4.7`, add nothing that pins a version
  number).
- Agent count: 11 → 12. Version bump: minor (new agent + new hub skill +
  tier changes).

## 11. Validation / acceptance criteria

- `make validate` passes with zero new errors.
- `PYTHONPATH=. python3 tools/validation/context_budget_checker.py` — new
  `continuum-mechanics-and-rheology` sub-skills stay under the 4,000-token
  (200K/2%) budget per file, consistent with all 223 currently-checked
  skills passing.
- `PYTHONPATH=. python3 tools/validation/xref_validator.py` — no dangling
  references from the 5 deleted dev-suite-origin skill names.
- `uv run pytest` passes unchanged.
- No skill content contradicts a CLAUDE.md "prohibited" item (§9's audit
  list) after changes.
- `plugin.json` agent/skill/keyword arrays match §5–§10 exactly.

## 12. Out of scope / follow-ups

- Full line-by-line rewrite of all ~130 skills against CLAUDE.md — §9 is a
  targeted spot-check of named risk areas, not exhaustive.
- research-suite expand spec — next sub-project, separate.
- Exact graph-theory routing wiring — implementation-time detail.
