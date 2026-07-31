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
   response/superposition, engineering mathematics, data-driven modeling
   for materials, transient networks (physical and covalent adaptable
   networks), and adaptive materials/nanocomposites — plus cooperative
   dynamics/collective phenomena/glass problems and physical/energy-based
   learning problems in disordered and soft-matter systems, and graph
   theory.

## 2. Scope

In scope: `plugins/science-suite/` only. Absorbing the 5 Python-tooling
skills displaced from dev-suite (§8) is in scope; re-litigating dev-suite's
own trim is not (already specced separately). Ownership split: this spec
owns folding the incoming content into science-suite; physical removal of
the 5 directories under `plugins/dev-suite/skills/` and the
`python-toolchain` manifest entry are executed by the sibling trim spec
(`2026-07-30-agent-core-devsuite-trim-design.md`), not duplicated here.

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
- Same README pass, independent staleness to fix while the file is open:
  the agent table currently mislabels `jax-pro`/`julia-pro` as sonnet
  (actually opus) and `ml-expert` as sonnet (actually haiku), and the
  summary line "11 specialized agents (4 opus, 7 sonnet)" is wrong today
  (actual current split is 6 opus / 4 sonnet / 1 haiku). Rewrite both to
  the **post-change** state (12 agents: 7 opus / 4 sonnet / 1 haiku, per
  §4/§5/§10) rather than restoring today's already-incorrect numbers. Also
  fix "This suite registers zero slash commands" (README.md:52 — false;
  `plugin.json` registers 2) and "14 hubs → 112 sub-skills" (README.md:54
  — doesn't match `plugin.json`'s 29 registered hubs); update both to
  actual counts at implementation time, since §5/§7 will shift the totals
  again.

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
hybrid (physics + ML) modeling for materials — including **transient
networks** (physical networks and covalent adaptable networks/vitrimers:
bond-exchange kinetics, sticky Rouse and Green-Tobolsky transient-network
rheology, stress relaxation via reversible/exchangeable crosslinks) and
**adaptive materials/nanocomposites** (filler-matrix constitutive modeling,
effective-medium theory, percolation-based property prediction,
self-healing/responsive composite behavior). Delegates neural-PDE/
physics-informed approaches to `pinn-engineer`, particle/MD methods to
`simulation-expert`, pure JAX numerics to `jax-pro`; cross-references
`statistical-physicist`'s percolation/correlation content (§6) for the
microstructure-statistics side of nanocomposite filler networks rather than
duplicating it. Note: no percolation content currently exists anywhere in
science-suite (checked the `correlation-*` family and
`statistical-physics-hub` — zero matches) — §6 must author this content,
not merely cross-reference existing material.

New hub skill: `continuum-mechanics-and-rheology` (registered in
`plugin.json`, following the existing hub pattern), with sub-skills for
FEM/FEA, constitutive-equations, DMA/rheology,
harmonic-response-superposition, transient-networks-and-can (covalent
adaptable networks), and nanocomposites-and-adaptive-materials (discovered
via the hub's routing table, not individually registered — per the manifest
rule in root CLAUDE.md).

## 6. Extend `statistical-physicist` — glass, collective phenomena & physical learning

Cooperative dynamics, collective phenomena, and glass problems (jamming,
aging, random landscapes) in soft matter fit `statistical-physicist`'s
existing scope directly (phase transitions, correlations, fluctuation
theorems, soft matter) — no new agent. Extend its `statistical-physics-hub`
and/or `active-matter` skill routing to cover these explicitly; check for
overlap with existing `correlation-*` skill family before adding new
sub-skill files (avoid re-creating content that already exists under a
different name). Nanocomposite filler-network percolation/correlation
statistics (§5) live here too, cross-referenced from
`continuum-mechanics-and-rheology` — this is new content to author (no
existing skill currently covers percolation thresholds or filler-network
statistics), not a routing-only change.

Also extend to **physical/energy-based learning problems in disordered and
soft-matter systems** — not limited to soft matter, per user clarification:
coupled learning and contrastive Hebbian learning in physical (mechanical/
electrical) networks, plasticity and memory formation in disordered
materials, energy-based learning frameworks (Hopfield-style) applied to
physical substrates. This is "learning" as a physical/statistical-mechanics
phenomenon (a material or network adapting its own structure), distinct
from classical ML — keep it out of `ml-expert`/`neural-network-master`'s
scope, which cover learning *algorithms*, not learning *as physics*.
Cross-reference `nonlinear-dynamics-expert` for the network-dynamics side of
physical learning substrates (attractor formation, plasticity rules as
dynamical systems).

## 7. New skill: graph theory

Cross-cutting math (used by GNN work, `network-coupled-dynamics`, and
potentially by FEM mesh/microstructure-network representations in §5). Add
as a new sub-skill, not a new hub — wire routing references from
`neural-network-mathematics`, `network-coupled-dynamics`, and
`continuum-mechanics-and-rheology`'s hubs rather than registering it
top-level. Exact routing wiring is an implementation detail for the
follow-on plan, not a further design decision.

## 8. Python-tooling consolidation (absorbing dev-suite's 5 skills)

Checked against existing science-suite skills — this is a **merge**, not a
copy-in: content folds into existing science-suite skills; the source
directories under `plugins/dev-suite/skills/` are removed by the sibling
trim spec (§2), not by this task:

| Incoming (from dev-suite trim) | Existing science-suite home | Action |
|---|---|---|
| `async-python-patterns` | `python-development` (already covers TaskGroups/structured concurrency) | Fold any unique content in |
| `python-toolchain` | `python-development` (already covers uv-based packaging) | Fold any unique content in |
| `python-packaging` | `python-packaging-advanced` (already covers uv/pyproject.toml/workspaces) | Fold any unique content in |
| `uv-package-manager` | `python-packaging-advanced` | Fold any unique content in |
| `python-performance-optimization` | No clean match — see note below | Add a new Python-profiling section to `python-development`; do not fold into `performance-tuning` |

Diff each incoming/existing pair during implementation — only port content
that's genuinely additive (e.g. a specific flag or workflow the existing
skill doesn't mention). Do not blindly concatenate.

`performance-tuning`'s title ("Julia Performance Tuning"), description
(`@code_warntype`, `@profview`, BenchmarkTools.jl), and Expert Agent
pointer (`julia-pro`) are Julia-specific throughout — folding
`python-performance-optimization`'s Python profiling content
(`cProfile`, `line_profiler`, `py-spy`, `memory_profiler`) there would
contradict its own framing and confuse skill-matching. `python-development`
only briefly mentions `py-spy` in a checklist item, with no profiling
depth — it's the correct destination, but needs a real new section, not a
one-line fold.

Also: the one confirmed cross-reference to an absorbed name —
`research-and-domains/SKILL.md:51` names "dev-suite python-toolchain
hub" — must be repointed at the new science-suite home as part of this
task (a repo-wide check found no other cross-references to the remaining
4 incoming names inside science-suite).

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
- `bifurcation-analysis`: currently recommends `BifurcationKit.jl` as
  primary throughout (description line, Quick Start section, and its
  Python-escape-hatch table, which marks `juliacall` → `BifurcationKit` as
  "Recommended") — this needs actual correction, not just confirmation,
  since BifurcationKit is blocked on Julia 1.12 (per CLAUDE.md prohibited
  list). The skill's own escape-hatch table already lists the right
  fallback: demote the `juliacall → BifurcationKit` row from "Recommended"
  and promote `AUTO-07p` (Fortran, already documented, Julia-free)
  instead — Python `tick` is a Hawkes/point-process library, not a
  numerical-continuation substitute, and should not be the routing target
  here. `julia-pro.md` carries the same BifurcationKit-as-primary content
  in several places (domain capability table, Domain 6 continuation
  section, skill-routing table, decision tree) and needs the same
  correction throughout; §4's "confirmed unchanged" list is about model
  tier only, not skill/agent content, so this is a separate content fix.
- `point-processes`: currently lists `PointProcesses.jl` as "Best starting
  point in Julia" with no caveat that it isn't usable (registry tagging
  pending, per CLAUDE.md prohibited list) — add the caveat or demote the
  recommendation; confirming the absence of a caveat isn't sufficient,
  since one is in fact missing today.
- `julia-hpc-distributed` / `parallel-computing`: confirm no
  `SequentialMonteCarlo` recommendation in any multi-package Julia env
  (RNGPool SIGABRT on 1.12, per CLAUDE.md prohibited list).
- Any Flux.jl recommendation for new SciML neural closures should be
  flagged/corrected to Lux.jl per CLAUDE.md.

## 10. plugin.json & doc updates

- Add `continuum-mechanics-engineer` to `agents` array.
- Add `continuum-mechanics-and-rheology` to `skills` array (new hub).
- The `research-and-domains/SKILL.md:51` cross-reference to the absorbed
  `python-toolchain` name is handled in §8, not repeated here — it is not
  a no-op (a repo-wide check confirmed the hit).
- Update `keywords` (drop `opus-4.7`, add nothing that pins a version
  number).
- Agent count: 11 → 12. Version bump: per repo convention all 4 plugin
  manifests and `pyproject.toml` stay version-synced (`make validate`
  enforces this) — this is one coordinated minor bump shared with the
  sibling trim spec's changes, not a science-suite-only bump.
- Sphinx docs enumerating science-suite agents and skill/hub counts
  (`docs/suites/science-suite.rst`, which has an `.. agent::` directive per
  agent plus a summary line currently reading "11 Agents ... 17 Hubs → 110
  Sub-skills"; `docs/agent-teams-guide.md`, which has a per-agent routing
  table) need a new `continuum-mechanics-engineer` entry and corrected
  counts — locate the exact tables at implementation time.

## 11. Validation / acceptance criteria

- `make validate` passes with zero new errors.
- `PYTHONPATH=. python3 tools/validation/context_budget_checker.py` — new
  `continuum-mechanics-and-rheology` sub-skills stay under the 4,000-token
  (200K/2%) budget per file, consistent with the checker's current
  repo-wide baseline of 223/223 passing (223 is the total across all 4
  plugins, not science-suite alone — science-suite alone currently has 127
  skill files).
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
