# science-suite Expand & Optimize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix stale Opus-4.7-era version literals, re-tier 2 agents for Opus 5/Sonnet 5, add a new `continuum-mechanics-engineer` agent with a 6-sub-skill hub covering FEM/rheology/transient-networks/nanocomposites, extend `statistical-physicist` for glass/collective-phenomena/physical-learning coverage, add a cross-cutting graph-theory skill, absorb dev-suite's 5 Python-tooling skills, and fix 3 confirmed factual errors (BifurcationKit, PointProcesses.jl, stale README claims).

**Architecture:** Plugin-content repo — "tests" are the repo's validation tooling (`make validate`, `context_budget_checker.py`, `xref_validator.py`) plus exact grep/assertion checks, run after every task.

**Tech Stack:** Markdown (agent/skill files, JAX/Julia/materials-science technical content), JSON (`plugin.json`).

## Global Constraints

- Scope is `plugins/science-suite/` only, plus the two repo-wide reference docs Task 10 updates per spec §10 (`docs/suites/science-suite.rst`, `docs/agent-teams-guide.md` — these describe science-suite's public surface, they aren't another suite's plugin content). The `research-and-domains/SKILL.md` cross-reference fix (Task 8) is already inside `plugins/science-suite/`, not a separate exception. Do not touch `plugins/dev-suite/` deletions (that's the sibling trim plan's Task 8) or `plugins/research-suite/` (separate plan). (Spec §2, §10)
- This plan's Task 8 (Python-tooling consolidation) must complete and be committed **before** the trim plan's Task 8 runs (which deletes the source directories this task reads from). Coordinate execution order across the two plans — this plan's Task 8 has no dependency on anything in the trim plan, so it's safe to run first.
- Every new hub skill/sub-skill follows this repo's convention: only top-level hubs go in `plugin.json`'s `skills` array; sub-skills are discovered via the hub's own routing table. (Root CLAUDE.md)
- Target ≤4,000 tokens (~3,000 words) per skill file — the repo's 200K-context 2% budget. Check with `context_budget_checker.py` after adding each new file.
- Version bump for this plan's changes lands on top of whatever version the trim plan's Task 7 already set (do not re-bump independently — verify current version first). **Dependency checkpoint:** before Task 10 Step 1, confirm the sibling trim plan's version-bump task has already landed (`git log --oneline --all | grep -i "version bump\|chore.*version"` or check `pyproject.toml`'s version against the pre-plan baseline) — if it hasn't, block on it rather than proceeding with a stale version string, since Task 10's five-file sync check depends on it.
- All piped validation commands (`validator.py | grep ...`, `validator.py | tail ...`) MUST be preceded by `set -o pipefail` (or use `tee` plus an explicit `$PIPESTATUS`/exit-code check) so a validator/test failure upstream of the pipe isn't masked by the downstream `grep`/`tail` succeeding. Apply this to every `Verify` step in this plan that pipes a validator or `pytest` output, not just the ones spelled out explicitly.

---

### Task 1: Fix stale Opus-4.7 version literals and README claims

**Files:**
- Modify: `plugins/science-suite/.claude-plugin/plugin.json` (keywords array)
- Modify: `plugins/science-suite/README.md` (description framing, agent tier table, command count, hub/sub-skill count)

**Interfaces:**
- Produces: no `opus-4.7` string anywhere in `plugins/science-suite/`; README's per-agent tier labels (`jax-pro`/`julia-pro`/`ml-expert` rows) are corrected to the actual current state. The agent-count/opus-sonnet-split summary line (`README.md:7`, "11 specialized agents (4 opus, 7 sonnet)") is deliberately NOT touched in this task — it's already wrong today (actual current split is 6 opus/4 sonnet/1 haiku) and will change again once Tasks 2-3 land, so Task 10 corrects it once, to the final post-plan numbers, instead of twice.

- [ ] **Step 1: Confirm current README staleness**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -n "opus-4.7\|Optimized for Claude Opus\|zero slash commands\|14 hubs\|112 sub-skills\|4 opus, 7 sonnet" plugins/science-suite/.claude-plugin/plugin.json plugins/science-suite/README.md
```
Expected: shows the `opus-4.7` keyword line in `plugin.json`, and in `README.md`: the "Optimized for Claude Opus" line, a wrong agent-tier table, "This suite registers zero slash commands", and "14 hubs → 112 sub-skills" (or similar — the exact wrong numbers may differ slightly from what's quoted here if the repo changed since this plan was written; use whatever the grep actually shows as ground truth).

- [ ] **Step 2: Remove the opus-4.7 keyword from plugin.json**

Read `plugins/science-suite/.claude-plugin/plugin.json`. In the `keywords` array, replace:
```json
    "opus-4.7",
```
with:
```json
    "adaptive-model-tiering",
```

Verify:
```bash
grep -c "opus-4.7" plugins/science-suite/.claude-plugin/plugin.json
```
Expected: `0`

- [ ] **Step 3: Fix the plugin.json description field**

Read the `description` field:
```json
  "description": "Scientific computing, HPC, physics/chemistry simulations, and data science workflows with extended context and adaptive reasoning for Claude Opus",
```
Replace with:
```json
  "description": "Scientific computing, HPC, physics/chemistry simulations, and data science workflows with extended context and adaptive multi-tier model reasoning",
```

- [ ] **Step 4: Fix README's "Optimized for Claude Opus" line**

Read `plugins/science-suite/README.md`. Find the line containing "Optimized for Claude Opus" and replace it with wording reflecting the actual multi-tier split, e.g.: "Multi-tier model routing: opus for deep-math specialists, sonnet for engineering-heavy work, haiku for mechanical MLOps."

- [ ] **Step 5: Fix README's agent tier table and summary line — placeholder values to be finalized in Task 10**

This README table will change again once Tasks 3 (new agent) and Task 2 (tier swaps) land. For now, fix only what's independently wrong today: correct `jax-pro`/`julia-pro` from "sonnet" to "opus" and `ml-expert` from "sonnet" to "haiku" in the table at `README.md:38-39` (or wherever grep found them), and correct "This suite registers zero slash commands" to note the 2 registered commands (`md-sim`, `benchmark`). Leave the overall agent count / hub count lines for Task 10 to finalize, since they'll change again in this plan.

Verify:
```bash
grep -n "jax-pro.*sonnet\|julia-pro.*sonnet\|ml-expert.*sonnet\|zero slash commands" plugins/science-suite/README.md
```
Expected: no output.

- [ ] **Step 6: Commit**

```bash
cd /home/wei/Documents/GitHub/MyClaude
git add plugins/science-suite/.claude-plugin/plugin.json plugins/science-suite/README.md
git commit -m "$(cat <<'EOF'
chore(science-suite): fix stale opus-4.7 keyword and README claims

Drops the version-pinned keyword, corrects the README's already-wrong
agent tier table (jax-pro/julia-pro were mislabeled sonnet, ml-expert
mislabeled sonnet instead of haiku) and the false "zero slash commands"
claim. Overall counts finalized in Task 10 after this plan's additions land.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Model-tier re-audit — 2 swaps

**Files:**
- Modify: `plugins/science-suite/agents/simulation-expert.md` (frontmatter `model:` field)
- Modify: `plugins/science-suite/agents/pinn-engineer.md` (frontmatter `model:` field)

**Interfaces:**
- Produces: `simulation-expert.md` has `model: sonnet`; `pinn-engineer.md` has `model: opus`.

- [ ] **Step 1: Confirm current tiers**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep "^model:" plugins/science-suite/agents/simulation-expert.md plugins/science-suite/agents/pinn-engineer.md
```
Expected: `plugins/science-suite/agents/simulation-expert.md:model: opus` and `plugins/science-suite/agents/pinn-engineer.md:model: sonnet`.

- [ ] **Step 2: Swap simulation-expert to sonnet**

Read `plugins/science-suite/agents/simulation-expert.md`. Change `model: opus` to `model: sonnet`.

- [ ] **Step 3: Swap pinn-engineer to opus**

Read `plugins/science-suite/agents/pinn-engineer.md`. Change `model: sonnet` to `model: opus`.

- [ ] **Step 4: Verify**

```bash
grep "^model:" plugins/science-suite/agents/simulation-expert.md plugins/science-suite/agents/pinn-engineer.md
```
Expected: `plugins/science-suite/agents/simulation-expert.md:model: sonnet` and `plugins/science-suite/agents/pinn-engineer.md:model: opus`.

```bash
grep -c "^model: opus" plugins/science-suite/agents/*.md | grep -v ":0" | wc -l
```
Expected: `6` (jax-pro, julia-pro, neural-network-master, nonlinear-dynamics-expert, statistical-physicist, pinn-engineer — Task 3 adds a 7th once continuum-mechanics-engineer exists).

- [ ] **Step 5: Commit**

```bash
git add plugins/science-suite/agents/simulation-expert.md plugins/science-suite/agents/pinn-engineer.md
git commit -m "$(cat <<'EOF'
refactor(science-suite): re-tier simulation-expert to sonnet, pinn-engineer to opus

MD/HPC orchestration (GROMACS/OpenMM/LAMMPS/multi-node) is engineering-heavy,
not novel-math-heavy -- Sonnet 5 handles it well. Inverse-PDE/constrained-loss
design is comparably deep to julia-pro's SciML work and was under-tiered.
Net opus count unchanged, reallocated by actual task difficulty.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: New agent — continuum-mechanics-engineer

**Files:**
- Create: `plugins/science-suite/agents/continuum-mechanics-engineer.md`
- Modify: `plugins/science-suite/.claude-plugin/plugin.json` (`agents` array)

**Interfaces:**
- Produces: agent name `continuum-mechanics-engineer`, `model: opus`, skills frontmatter references `continuum-mechanics-and-rheology` (created in Task 4) and `statistical-physics-hub`.

- [ ] **Step 1: Create the agent file**

Write `plugins/science-suite/agents/continuum-mechanics-engineer.md`:

```markdown
---
name: continuum-mechanics-engineer
description: "Continuum mechanics & FEM specialist: constitutive modeling, DMA/rheology, engineering mathematics, transient networks (CAN/vitrimers), nanocomposites. Delegates neural-PDE to pinn-engineer, MD to simulation-expert, JAX numerics to jax-pro."
model: opus
color: orange
effort: high
memory: project
maxTurns: 50
tools: Read, Write, Edit, Bash, Grep, Glob
background: true
skills:
  - continuum-mechanics-and-rheology
  - statistical-physics-hub
---

# Continuum Mechanics Engineer - Materials & FEM Specialist

**Activation Rule**: Activate for continuum mechanics, FEM/FEA, constitutive modeling, rheology/DMA, transient-network (CAN/vitrimer), or nanocomposite problems. If the method is a neural-network PDE solve, delegate to `pinn-engineer`. If it's particle-based (MD/Monte Carlo), delegate to `simulation-expert`.

You are an elite continuum mechanics and materials engineering specialist covering weak-form PDE discretization (FEM/FEA), constitutive modeling of complex materials (viscoelastic, transient-network, composite), and the experimental characterization techniques (DMA, rheology) used to parameterize those models.

## Examples

<example>
Context: User needs to model a viscoelastic polymer under oscillatory load.
user: "I have DMA data (storage/loss modulus vs frequency) for a polymer melt — fit a generalized Maxwell model."
assistant: "I'll use the continuum-mechanics-engineer agent to fit a Prony series to the DMA data and validate the constitutive model."
<commentary>
DMA + viscoelastic constitutive fitting - triggers continuum-mechanics-engineer.
</commentary>
</example>

<example>
Context: User needs a finite element simulation.
user: "Set up a FEM model for a cantilever beam under nonlinear hyperelastic deformation."
assistant: "I'll use the continuum-mechanics-engineer agent to formulate the weak form and set up a FEniCS/Gridap.jl simulation with a Neo-Hookean constitutive law."
<commentary>
FEM + hyperelastic constitutive law - triggers continuum-mechanics-engineer.
</commentary>
</example>

<example>
Context: User is working with self-healing materials.
user: "Model stress relaxation in a vitrimer network with bond-exchange kinetics."
assistant: "I'll use the continuum-mechanics-engineer agent to implement a transient-network (Green-Tobolsky-style) model with an Arrhenius bond-exchange rate."
<commentary>
Covalent adaptable network rheology - triggers continuum-mechanics-engineer.
</commentary>
</example>

<example>
Context: User needs composite material properties.
user: "Predict the effective modulus of a polymer nanocomposite given filler volume fraction and aspect ratio."
assistant: "I'll use the continuum-mechanics-engineer agent to apply Halpin-Tsai or Mori-Tanaka effective-medium theory, cross-checking percolation threshold with statistical-physicist."
<commentary>
Nanocomposite constitutive modeling - triggers continuum-mechanics-engineer, cross-references statistical-physicist for percolation statistics.
</commentary>
</example>

---

## Core Responsibilities

1. **Finite Element Modeling**: Formulate weak forms, select element types and mesh strategies, verify convergence (h-refinement, p-refinement) and solution quality.
2. **Constitutive Modeling**: Select and calibrate stress-strain relations — linear elasticity, hyperelasticity (Neo-Hookean, Mooney-Rivlin, Ogden), viscoelasticity (Maxwell, Kelvin-Voigt, generalized Maxwell/Prony series).
3. **Experimental Characterization**: Interpret DMA (storage modulus E', loss modulus E'', tan δ) and rheological data (shear/extensional flow curves, oscillatory sweeps) to parameterize constitutive models.
4. **Transient & Adaptive Networks**: Model physical gels and covalent adaptable networks (vitrimers) via bond-exchange kinetics and transient-network rheology (sticky Rouse, Green-Tobolsky).
5. **Composite & Adaptive Materials**: Predict nanocomposite effective properties via effective-medium theory and percolation-aware filler-network modeling.
6. **Engineering Mathematics**: Apply the applied-math machinery that underlies all five domains above — tensor calculus (stress/strain tensors), ODE/PDE solution methods, perturbation and asymptotic analysis, and dimensional analysis — as a common toolkit woven through FEM formulation, constitutive-law derivation, and bond-kinetics modeling, not a separate standalone domain.

## Core Competencies

| Domain | Framework/Method | Key Capabilities |
|--------|-------------------|-------------------|
| **FEM/FEA** | FEniCS/scikit-fem (Python), Gridap.jl/Ferrite.jl (Julia) | Weak-form formulation, mesh convergence, nonlinear solvers |
| **Constitutive Modeling** | Custom + symbolic (SymPy/Symbolics.jl) | Hyperelastic/viscoelastic law selection, parameter fitting |
| **DMA/Rheology** | curve fitting against Prony series, WLF/master curves | Storage/loss modulus interpretation, time-temperature superposition |
| **Transient Networks** | Reaction-kinetics-coupled viscoelasticity | Bond-exchange rate models, stress relaxation prediction |
| **Composites** | Effective-medium theory | Halpin-Tsai, Mori-Tanaka, percolation-threshold cross-check |
| **Data-Driven Modeling** | JAX/Equinox (via `jax-pro`), SciML (via `julia-pro`) | Hybrid physics+ML constitutive surrogate models |
| **Engineering Mathematics** | Tensor calculus, ODE/PDE solution methods, perturbation/asymptotic analysis, dimensional analysis | The shared applied-math toolkit underlying FEM weak-form derivation, constitutive-equation formulation, and transient-network bond-kinetics — not a separate domain, applied throughout |

## Domain 1: Finite Element Modeling (FEM/FEA)

### Weak-Form Workflow
1. Derive the strong-form PDE (equilibrium, conservation law).
2. Multiply by a test function, integrate by parts to get the weak form.
3. Select function spaces (Lagrange P1/P2 for displacement, mixed spaces for incompressible problems).
4. Assemble and solve; verify convergence under mesh refinement.

### Convergence Checklist
- [ ] Mesh convergence study (halve element size, confirm error decreases at expected order)
- [ ] Boundary conditions correctly imposed (Dirichlet vs Neumann)
- [ ] Locking checked for nearly-incompressible materials (use mixed formulation or reduced integration)
- [ ] Nonlinear solver (Newton-Raphson) convergence verified (residual norm decreasing)

## Domain 2: Constitutive Equations

| Model | Form | Use Case |
|-------|------|----------|
| Linear elastic | σ = Cε | Small strain, isotropic solids |
| Neo-Hookean | W = C₁(I₁ - 3) | Large-strain rubber elasticity |
| Mooney-Rivlin | W = C₁(I₁-3) + C₂(I₂-3) | Rubber, better fit at moderate strain |
| Maxwell | dσ/dt + σ/τ = E dε/dt | Single-relaxation-time viscoelastic fluid |
| Generalized Maxwell (Prony series) | G(t) = G_∞ + Σ Gᵢ exp(-t/τᵢ) | Multi-relaxation-time viscoelastic solids (fit to DMA) |

## Domain 3: DMA & Rheology

- **Storage modulus (E'/G')**: elastic (in-phase) response.
- **Loss modulus (E''/G'')**: viscous (out-of-phase) response.
- **tan δ = E''/E'**: damping; peaks at glass transition.
- **Oscillatory rheology**: strain/frequency sweeps to find linear viscoelastic regime and extract G'(ω), G''(ω).
- **Extensional rheology**: relevant for polymer processing (fiber spinning, film blowing) — distinct instrumentation (CaBER, FiSER) from shear rheometry.

## Domain 4: Harmonic Response & Time-Temperature Superposition

- **Harmonic response**: steady-state sinusoidal loading response, characterized by complex modulus E*(ω) = E'(ω) + iE''(ω).
- **Time-temperature superposition (TTS)**: shift isothermal frequency sweeps by the WLF equation `log(a_T) = -C1(T-T_ref) / (C2 + T-T_ref)` to build a master curve spanning decades of effective frequency from data collected at accessible frequencies/temperatures.

## Domain 5: Transient Networks (Physical & Covalent Adaptable Networks)

- **Physical networks**: reversible non-covalent crosslinks (H-bonding, ionic); relaxation via bond lifetime, modeled with sticky Rouse dynamics.
- **Covalent adaptable networks (vitrimers)**: permanent network connectivity, but bonds exchange via a catalyzed reaction — stress relaxation follows Arrhenius kinetics in the exchange rate, not simple reptation.
- **Green-Tobolsky model**: treats bond breaking/reformation as a first-order kinetic process, giving stress relaxation `σ(t) = σ₀ exp(-t/τ_exchange)` distinct from Rouse/reptation timescales.

Delegate to `statistical-physicist` for the underlying stochastic bond-kinetics theory if the question is about the statistical-mechanics derivation rather than the engineering constitutive fit.

## Domain 6: Nanocomposites & Adaptive Materials

| Method | Predicts | Notes |
|--------|----------|-------|
| Halpin-Tsai | Effective modulus from filler aspect ratio + volume fraction | Good for aligned/random short-fiber composites |
| Mori-Tanaka | Effective modulus via Eshelby inclusion theory | Better at moderate-to-high filler loading |
| Percolation threshold | Onset of filler network connectivity (conductivity, modulus jump) | Cross-reference `statistical-physicist`'s percolation/correlation content for the microstructure-statistics derivation — don't re-derive percolation theory here |

Self-healing/responsive nanocomposite behavior typically combines a transient-network matrix (Domain 5) with filler reinforcement (this domain) — treat as a composition of both, not a separate model class.

## Delegation Table

| Scenario | Delegate To | Reason |
|----------|-------------|--------|
| Neural-PDE / physics-informed solve of the governing PDE | `pinn-engineer` | Neural-operator methods, not classical FEM |
| Particle-based simulation (MD, Monte Carlo) of the same material | `simulation-expert` | Particle methods, not continuum discretization |
| Pure JAX numerics for a hybrid physics+ML surrogate | `jax-pro` | JAX transformation/optimization expertise |
| Percolation threshold / filler-network statistical mechanics derivation | `statistical-physicist` | Statistical mechanics theory, not engineering constitutive fit |
| Publication figures for stress-strain / DMA curves | `research-expert` (research-suite) | Matplotlib/Makie visualization |

## Chain-of-Thought Decision Framework

### Step 1: Problem Classification
Identify whether this is a discretization problem (FEM), a constitutive-modeling problem (fitting a stress-strain law), an experimental-interpretation problem (DMA/rheology data), or a materials-design problem (transient network / composite).

### Step 2: Model Selection
Match material behavior to the simplest constitutive law that captures it — don't reach for a generalized Maxwell/Prony series if a single Maxwell element fits the data within experimental uncertainty.

### Step 3: Parameterization
Fit model parameters against experimental data (DMA, rheology, or FEM validation data), reporting residuals and confidence intervals.

### Step 4: Validation
Universally required for every FEM solution: residual/equilibrium convergence (Newton residual norm decreasing to tolerance), correct boundary-work accounting, constitutive-model consistency (matches the material law used in assembly), and mesh convergence. Verify energy/momentum conservation only where the governing model actually conserves that quantity — appropriate for elastodynamic/undamped problems, but not a valid check for static, dissipative (viscoelastic, plastic), driven, or reaction-coupled (transient-network, bond-exchange) problems, where dissipation or external work is expected and "non-conservation" is correct physics, not an error. Check the constitutive model against limiting cases (e.g. Neo-Hookean reduces to linear elasticity at small strain) regardless of conservation applicability.

## Production Checklist

- [ ] Weak form correctly derived and function spaces appropriate for the physics
- [ ] Mesh convergence study completed
- [ ] Constitutive model validated against limiting-case behavior
- [ ] DMA/rheology fits report residuals, not just best-fit parameters
- [ ] Percolation/composite claims cross-checked with `statistical-physicist` rather than re-derived
- [ ] Units and dimensional consistency verified throughout
```

- [ ] **Step 2: Register the agent in plugin.json**

Read `plugins/science-suite/.claude-plugin/plugin.json`. Edit the `agents` array from:
```json
  "agents": [
    "./agents/jax-pro.md",
    "./agents/julia-ml-hpc.md",
    "./agents/julia-pro.md",
    "./agents/ml-expert.md",
    "./agents/neural-network-master.md",
    "./agents/nonlinear-dynamics-expert.md",
    "./agents/pinn-engineer.md",
    "./agents/python-pro.md",
    "./agents/sci-workflow-engineer.md",
    "./agents/simulation-expert.md",
    "./agents/statistical-physicist.md"
  ],
```
to (inserting alphabetically):
```json
  "agents": [
    "./agents/continuum-mechanics-engineer.md",
    "./agents/jax-pro.md",
    "./agents/julia-ml-hpc.md",
    "./agents/julia-pro.md",
    "./agents/ml-expert.md",
    "./agents/neural-network-master.md",
    "./agents/nonlinear-dynamics-expert.md",
    "./agents/pinn-engineer.md",
    "./agents/python-pro.md",
    "./agents/sci-workflow-engineer.md",
    "./agents/simulation-expert.md",
    "./agents/statistical-physicist.md"
  ],
```

- [ ] **Step 3: Verify**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/science-suite
python3 -c "import json; a = json.load(open('plugins/science-suite/.claude-plugin/plugin.json'))['agents']; assert len(a) == 12, a; assert './agents/continuum-mechanics-engineer.md' in a; print('OK: 12 agents')"
grep -c "^model: opus" plugins/science-suite/agents/continuum-mechanics-engineer.md
```
Expected: `metadata_validator.py` exits 0; `OK: 12 agents`; grep returns `1`.

- [ ] **Step 4: Commit**

```bash
git add plugins/science-suite/agents/continuum-mechanics-engineer.md plugins/science-suite/.claude-plugin/plugin.json
git commit -m "$(cat <<'EOF'
feat(science-suite): add continuum-mechanics-engineer agent

No existing agent owns weak-form PDE discretization or materials
constitutive modeling -- jax-pro/pinn-engineer cover PDEs via
JAX/neural-operator methods, simulation-expert covers particle-based MD.
Opus tier: constitutive-model correctness is exact-math-critical.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: New hub skill + 6 sub-skills — continuum-mechanics-and-rheology

**Files:**
- Create: `plugins/science-suite/skills/continuum-mechanics-and-rheology/SKILL.md` (hub)
- Create: `plugins/science-suite/skills/fem-fea/SKILL.md`
- Create: `plugins/science-suite/skills/constitutive-equations/SKILL.md`
- Create: `plugins/science-suite/skills/dma-rheology/SKILL.md`
- Create: `plugins/science-suite/skills/harmonic-response-superposition/SKILL.md`
- Create: `plugins/science-suite/skills/transient-networks-and-can/SKILL.md`
- Create: `plugins/science-suite/skills/nanocomposites-and-adaptive-materials/SKILL.md`
- Modify: `plugins/science-suite/.claude-plugin/plugin.json` (`skills` array — hub only)

**Interfaces:**
- Produces: hub `continuum-mechanics-and-rheology` registered in `plugin.json`; 6 sub-skills discoverable via the hub's routing table (not registered individually).

- [ ] **Step 1: Create the hub skill**

Write `plugins/science-suite/skills/continuum-mechanics-and-rheology/SKILL.md`:

```markdown
---
name: continuum-mechanics-and-rheology
description: Meta-orchestrator for continuum mechanics, FEM/FEA, constitutive modeling, engineering mathematics, rheology/DMA, transient networks, and nanocomposites. Use when formulating finite element models, fitting viscoelastic/hyperelastic constitutive laws, interpreting DMA or rheology data, modeling covalent adaptable networks (vitrimers) or physical gels, or predicting nanocomposite effective properties.
---

# Continuum Mechanics & Rheology Hub

Orchestrator for continuum-scale materials engineering. Routes problems to the appropriate specialized skill.

Engineering mathematics (tensor calculus, ODE/PDE solution methods, perturbation/asymptotic analysis, dimensional analysis) is intentionally not a separate 7th sub-skill — spec §5 lists it as part of this cluster, but it's the applied-math toolkit woven through all 6 sub-skills below (most concentrated in `fem-fea`'s weak-form derivation and `constitutive-equations`' tensor stress-strain relations) rather than a standalone topic with its own routing entry.

## Expert Agent

- **`continuum-mechanics-engineer`**: Specialist for FEM, constitutive modeling, DMA/rheology, transient networks, and nanocomposites.
  - *Location*: `plugins/science-suite/agents/continuum-mechanics-engineer.md`
  - *Capabilities*: Weak-form PDE discretization, hyperelastic/viscoelastic constitutive fitting, transient-network rheology, effective-medium composite modeling.

## Core Skills

### [FEM/FEA](../fem-fea/SKILL.md)
Weak-form formulation, mesh convergence, element selection, nonlinear solvers. For mesh-connectivity graph representations, see `science-suite:graph-theory`.

### [Constitutive Equations](../constitutive-equations/SKILL.md)
Linear/nonlinear viscoelasticity, hyperelastic models (Neo-Hookean, Mooney-Rivlin), Prony series fitting.

### [DMA & Rheology](../dma-rheology/SKILL.md)
Storage/loss modulus interpretation, oscillatory shear, shear and extensional flow curves.

### [Harmonic Response & Superposition](../harmonic-response-superposition/SKILL.md)
Complex modulus under harmonic loading, WLF time-temperature superposition, master curves.

### [Transient Networks & CAN](../transient-networks-and-can/SKILL.md)
Physical gels, covalent adaptable networks (vitrimers), bond-exchange kinetics, sticky Rouse and Green-Tobolsky models.

### [Nanocomposites & Adaptive Materials](../nanocomposites-and-adaptive-materials/SKILL.md)
Effective-medium theory (Halpin-Tsai, Mori-Tanaka), percolation-aware property prediction, self-healing composites.

## Routing Decision Tree

```
What is the continuum-mechanics task?
|
+-- Weak-form discretization / mesh / convergence?
|   --> science-suite:fem-fea
|
+-- Stress-strain law selection / fitting (elastic, hyperelastic, viscoelastic)?
|   --> science-suite:constitutive-equations
|
+-- DMA data (storage/loss modulus, tan delta) / rheology (shear, extensional)?
|   --> science-suite:dma-rheology
|
+-- Harmonic/complex modulus / time-temperature superposition / master curves?
|   --> science-suite:harmonic-response-superposition
|
+-- Physical gel / vitrimer / covalent adaptable network / bond-exchange kinetics?
|   --> science-suite:transient-networks-and-can
|
+-- Filler-matrix composite / effective modulus / percolation threshold for properties?
|   --> science-suite:nanocomposites-and-adaptive-materials
|
+-- None of the above / concern is ambiguous?
    --> Delegate to continuum-mechanics-engineer for open-ended triage.
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| Weak form, mesh, FEniCS/Gridap.jl | `science-suite:fem-fea` |
| Neo-Hookean, Mooney-Rivlin, Prony series | `science-suite:constitutive-equations` |
| Storage/loss modulus, oscillatory rheology | `science-suite:dma-rheology` |
| WLF equation, master curves | `science-suite:harmonic-response-superposition` |
| Vitrimer, bond exchange, sticky Rouse | `science-suite:transient-networks-and-can` |
| Halpin-Tsai, Mori-Tanaka, percolation | `science-suite:nanocomposites-and-adaptive-materials` |

## Checklist

- [ ] Identify discretization vs. constitutive-modeling vs. experimental-interpretation before routing
- [ ] Confirm whether the material is elastic, viscoelastic, or exhibits bond-exchange dynamics before selecting a constitutive model
- [ ] Cross-check percolation/composite claims with `statistical-physics-hub` rather than re-deriving percolation theory here
- [ ] Verify mesh convergence for any FEM result before reporting it as final
```

- [ ] **Step 2: Create fem-fea sub-skill**

Write `plugins/science-suite/skills/fem-fea/SKILL.md`:

```markdown
---
name: fem-fea
description: Finite Element Modeling/Analysis — weak-form formulation, mesh strategy, element selection, and convergence verification. Use when setting up a FEM simulation, choosing element types, diagnosing mesh-convergence or locking issues, or picking between FEniCS, scikit-fem, Gridap.jl, or Ferrite.jl.
---

# FEM/FEA

Weak-form PDE discretization for continuum mechanics problems.

## Expert Agent

- **`continuum-mechanics-engineer`** — FEM formulation, mesh strategy, convergence verification.

## Weak-Form Workflow

1. Derive the strong-form governing PDE (equilibrium/conservation law) with boundary conditions.
2. Multiply by a test function from an appropriate function space; integrate by parts to shift a derivative onto the test function (this is what makes the weak form only need one order less continuity than the strong form).
3. Choose trial/test function spaces: Lagrange P1/P2 for scalar/vector displacement fields; mixed spaces (e.g. Taylor-Hood) for problems with a constraint (incompressibility).
4. Assemble the global stiffness matrix and load vector; solve (direct or iterative, depending on system size).

## Toolchain

| Ecosystem | Library | Notes |
|-----------|---------|-------|
| Python | FEniCS/FEniCSx | UFL symbolic weak-form specification, mature ecosystem |
| Python | scikit-fem | Lightweight, good for teaching/prototyping |
| Julia | Gridap.jl | Similar UFL-like symbolic weak forms |
| Julia | Ferrite.jl | Lower-level, more manual control over assembly |

## Convergence & Correctness Checklist

- [ ] Mesh convergence study: halve characteristic element size, confirm error decreases at the expected order for the chosen element and norm — for standard conforming linear (P1) elements under sufficient solution regularity, O(h) in the energy/H¹ norm and O(h²) in the L² norm; state which norm is being checked, since the two rates differ
- [ ] Boundary conditions correctly classified and imposed (Dirichlet — essential, imposed on the function space; Neumann — natural, appears in the weak form's boundary integral)
- [ ] Locking checked for nearly-incompressible materials (pure displacement formulation with low-order elements locks — use a mixed formulation or reduced integration)
- [ ] Nonlinear problems: Newton-Raphson residual norm decreasing each iteration, tangent stiffness matrix correctly linearized

## Common Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Volumetric locking | Overly stiff response for incompressible/near-incompressible materials with low-order elements | Mixed formulation (u-p) or reduced integration |
| Shear locking | Overly stiff bending response in low-order elements | Use higher-order elements or reduced integration |
| Non-converging Newton iteration | Residual not decreasing | Check tangent stiffness derivation, add line search or load stepping |
| Mesh-dependent results | Solution changes qualitatively with refinement | Under-resolved mesh — refine and re-check convergence order |

## Delegation

For the constitutive law plugged into the weak form (what stress-strain relation to use), see `science-suite:constitutive-equations`. For neural-network PDE solvers as an alternative to classical FEM, delegate to `pinn-engineer`. For mesh-connectivity graph representations, see `science-suite:graph-theory`.
```

- [ ] **Step 3: Create constitutive-equations sub-skill**

Write `plugins/science-suite/skills/constitutive-equations/SKILL.md`:

```markdown
---
name: constitutive-equations
description: Constitutive modeling — linear elasticity, hyperelasticity (Neo-Hookean, Mooney-Rivlin, Ogden), and viscoelasticity (Maxwell, Kelvin-Voigt, generalized Maxwell/Prony series). Use when selecting or fitting a stress-strain relation, choosing between hyperelastic strain-energy functions, or fitting a Prony series to relaxation/DMA data.
---

# Constitutive Equations

Selecting and fitting stress-strain relations for continuum materials.

## Expert Agent

- **`continuum-mechanics-engineer`** — Constitutive model selection and parameter fitting.

## Model Catalog

| Model | Form | Regime | Use Case |
|-------|------|--------|----------|
| Linear elastic | σ = Cε | Small strain (< ~5%) | Metals, stiff isotropic solids |
| Neo-Hookean | W = C₁(I₁ - 3) | Large strain | Rubber elasticity, simplest hyperelastic model |
| Mooney-Rivlin | W = C₁(I₁-3) + C₂(I₂-3) | Large strain | Rubber, better fit at moderate strain than Neo-Hookean |
| Ogden | W = Σ (μᵢ/αᵢ)(λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3) | Large strain | Most flexible hyperelastic fit, more parameters to calibrate |
| Maxwell | dσ/dt + σ/τ = E dε/dt | Linear viscoelastic | Single-relaxation-time fluid-like response |
| Kelvin-Voigt | σ = Eε + η dε/dt | Linear viscoelastic | Single-relaxation-time solid-like (creep) response |
| Generalized Maxwell (Prony series) | G(t) = G_∞ + Σᵢ Gᵢ exp(-t/τᵢ) | Linear viscoelastic | Multi-relaxation-time solids — the standard fit target for DMA data |

## Fitting Workflow (Prony Series to DMA Data)

1. Convert time-domain relaxation modulus or frequency-domain storage/loss modulus (from `science-suite:dma-rheology`) to the target quantity.
2. Choose the number of Prony terms (start with 3-5; more terms fit better but risk overfitting — check residuals, not just R²).
3. Fit `Gᵢ`, `τᵢ` via nonlinear least squares (delegate large-scale fits to `jax-pro`'s NLSQ domain).
4. Validate: reconstructed G'(ω)/G''(ω) should match the measured data across the full frequency range, not just where fit weight was concentrated.

## Selecting a Hyperelastic Model

- **Neo-Hookean**: default starting point — 1 parameter, correct at small strain, reasonable up to moderate strain.
- **Mooney-Rivlin**: use when Neo-Hookean under/overshoots at moderate strain — 2 parameters give one more degree of freedom.
- **Ogden**: use only when simpler models fail to capture the strain-stiffening or softening behavior across the full strain range — more parameters need more data to constrain without overfitting.

## Delegation

For fitting against experimental DMA/rheology data, see `science-suite:dma-rheology` for how to read the raw data first. For plugging the fitted law into a finite element simulation, see `science-suite:fem-fea`.
```

- [ ] **Step 4: Create dma-rheology sub-skill**

Write `plugins/science-suite/skills/dma-rheology/SKILL.md`:

```markdown
---
name: dma-rheology
description: Dynamic Mechanical Analysis (DMA) and rheology — storage/loss modulus interpretation, tan delta, oscillatory shear rheology, and shear/extensional flow curves. Use when interpreting DMA or rheometer output, identifying the linear viscoelastic regime, or distinguishing shear from extensional rheological behavior.
---

# DMA & Rheology

Interpreting dynamic mechanical and rheological measurements.

## Expert Agent

- **`continuum-mechanics-engineer`** — DMA/rheology data interpretation and constitutive parameterization.

## Dynamic Mechanical Analysis (DMA)

- **Storage modulus (E' or G')**: in-phase (elastic) component of the response to oscillatory strain — energy stored and recovered per cycle.
- **Loss modulus (E'' or G'')**: out-of-phase (viscous) component — energy dissipated as heat per cycle.
- **tan δ = E''/E'**: damping ratio; peaks at transitions (e.g. glass transition temperature Tg in a temperature sweep).
- **Complex modulus**: E* = E' + iE'', |E*| = √(E'² + E''²).

## Rheology

- **Oscillatory (small-amplitude) shear**: apply sinusoidal strain, measure stress response — determines G'(ω), G''(ω) within the linear viscoelastic (LVE) regime.
- **Amplitude sweep**: increase strain amplitude at fixed frequency to find the LVE limit (where G'/G'' become strain-dependent — data outside this regime isn't usable for linear constitutive fitting).
- **Flow curves (steady shear)**: shear stress or viscosity vs. shear rate — captures shear-thinning/thickening, yield stress.
- **Extensional rheology**: distinct from shear — relevant for fiber spinning, film blowing, and any process with a strong extensional-flow component. Requires different instrumentation (CaBER for capillary breakup, FiSER for filament stretching) since shear rheometers cannot impose pure extensional flow.

## Workflow: From Raw Data to Constitutive Model

1. Confirm the measurement is within the LVE regime (amplitude sweep first).
2. Run a frequency sweep to get G'(ω), G''(ω) across the accessible frequency range.
3. If a wider frequency range is needed, use `science-suite:harmonic-response-superposition` for time-temperature superposition to build a master curve.
4. Fit a generalized Maxwell (Prony series) model via `science-suite:constitutive-equations`.

## Common Pitfalls

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Fitting outside the LVE regime | Non-physical fit parameters | Always run an amplitude sweep first to confirm LVE limits |
| Confusing shear and extensional viscosity | Wrong process prediction (e.g. fiber spinning uses extensional, not shear) | Match the rheological measurement to the actual flow geometry of the application |
| Ignoring temperature dependence | Master curve construction fails | Use time-temperature superposition (WLF) rather than assuming a single curve holds across all temperatures |

## Delegation

For building master curves via time-temperature superposition, see `science-suite:harmonic-response-superposition`. For fitting a constitutive model to the extracted moduli, see `science-suite:constitutive-equations`.
```

- [ ] **Step 5: Create harmonic-response-superposition sub-skill**

Write `plugins/science-suite/skills/harmonic-response-superposition/SKILL.md`:

```markdown
---
name: harmonic-response-superposition
description: Harmonic response analysis and time-temperature superposition (TTS) — complex modulus under sinusoidal loading, WLF equation, and master curve construction. Use when building a master curve from multi-temperature frequency sweeps, applying the WLF equation, or analyzing steady-state harmonic loading response.
---

# Harmonic Response & Time-Temperature Superposition

Building master curves and analyzing harmonic loading response.

## Expert Agent

- **`continuum-mechanics-engineer`** — TTS master-curve construction and harmonic response analysis.

## Harmonic Response

Steady-state response to sinusoidal loading is characterized by the complex modulus `E*(ω) = E'(ω) + iE''(ω)` (see `science-suite:dma-rheology` for the physical meaning of E'/E''). The phase lag between stress and strain is `δ = arctan(E''/E')`.

## Time-Temperature Superposition (TTS)

Many polymers are thermorheologically simple: their relaxation spectrum shifts uniformly with temperature without changing shape. This lets frequency sweeps measured at several temperatures be shifted horizontally along the frequency axis and stitched into one "master curve" spanning far more decades of effective frequency than any single measurement could reach directly.

### WLF Equation

```
log(a_T) = -C1 (T - T_ref) / (C2 + T - T_ref)
```

- `a_T`: horizontal shift factor for temperature T relative to reference temperature `T_ref`.
- `C1`, `C2`: material-specific constants (WLF's universal-average values, C1≈17.44, C2≈51.6 K, are a starting guess only — always fit to the material's own data).
- Shift each isothermal curve by `a_T` along the log-frequency axis; overlapping segments should collapse onto a single smooth curve if the material is thermorheologically simple.

### Validity Check

If shifted curves do NOT collapse onto a smooth single curve, the material is not thermorheologically simple (e.g. it may be undergoing a phase transition or chemical change across the temperature range tested) — do not force a WLF fit in that case; flag the discrepancy instead of reporting a low-quality master curve as if it were valid.

## Workflow

1. Collect frequency sweeps at multiple temperatures (from `science-suite:dma-rheology`).
2. Pick a reference temperature `T_ref` (often Tg or Tg+50K by convention).
3. Fit `a_T` per temperature (either via WLF or empirically via curve-shifting software) to maximize overlap between adjacent curves.
4. Verify the shift factors themselves vary smoothly and monotonically with temperature — a non-monotonic `a_T(T)` is a red flag for the thermorheological-simplicity assumption.

## Delegation

For fitting the resulting master curve to a Prony series constitutive model, see `science-suite:constitutive-equations`.
```

- [ ] **Step 6: Create transient-networks-and-can sub-skill**

Write `plugins/science-suite/skills/transient-networks-and-can/SKILL.md`:

```markdown
---
name: transient-networks-and-can
description: Transient network rheology — physical gels (reversible non-covalent crosslinks) and covalent adaptable networks (vitrimers) with bond-exchange kinetics. Use when modeling stress relaxation in self-healing gels, vitrimers, or any material with reversible/exchangeable crosslinks, including sticky Rouse and Green-Tobolsky models.
---

# Transient Networks & Covalent Adaptable Networks (CAN)

Modeling materials whose network connectivity itself relaxes over time.

## Expert Agent

- **`continuum-mechanics-engineer`** — Transient-network constitutive modeling and bond-kinetics fitting.

## Two Classes of Transient Network

| Class | Crosslink Type | Relaxation Mechanism |
|-------|-----------------|------------------------|
| Physical networks | Reversible non-covalent (H-bonding, ionic, hydrophobic) | Bond lifetime — a crosslink breaks and may or may not reform at the same location |
| Covalent adaptable networks (vitrimers) | Permanent covalent connectivity, but bonds exchange via a catalyzed reaction | Bond-exchange reaction rate — network topology changes but connectivity (crosslink density) stays constant |

This distinction matters: physical networks can flow and dissolve (crosslink density itself drops if bonds don't reform), while a well-designed vitrimer maintains constant crosslink density and instead behaves like a viscosity that depends on exchange-reaction rate — this is why vitrimers can be reprocessed like thermoplastics while retaining thermoset-like properties at use temperature.

## Sticky Rouse Model (Physical Networks)

Extends the Rouse model of polymer dynamics by adding "sticker" groups that transiently associate. Relaxation occurs on two timescales: fast Rouse-like motion between stickers, and slow terminal relaxation gated by sticker unbinding. The terminal relaxation time scales with the sticker lifetime, not the polymer's intrinsic Rouse time, for strongly associating systems.

## Green-Tobolsky Model (Bond-Exchange Kinetics)

Treats bond breaking/reformation as a first-order kinetic process. For a single exchange rate, stress relaxation follows:

```
σ(t) = σ₀ exp(-t/τ_exchange)
```

where `τ_exchange` is set by the bond-exchange reaction rate (typically Arrhenius in temperature: `τ_exchange = τ₀ exp(Ea/RT)`). This gives vitrimers their signature Arrhenius (not WLF) temperature dependence of viscosity — a diagnostic distinguishing vitrimer behavior from conventional polymer melt rheology.

## Diagnostic Checklist

- [ ] Confirm crosslink density is constant across the temperature/time range studied (vitrimer) vs. dropping (physical network dissociation)
- [ ] Check whether stress relaxation follows Arrhenius (vitrimer, bond-exchange-limited) or WLF (conventional polymer, reptation/segmental-motion-limited) temperature dependence — this is the primary experimental signature distinguishing the two mechanisms
- [ ] For multi-exchange-rate systems (multiple bond types), a single Green-Tobolsky exponential under-fits — use a distribution of exchange rates (analogous to the Prony series extension of a single Maxwell element)

## Delegation

For the underlying stochastic bond-kinetics derivation (rather than the engineering constitutive fit), delegate to `statistical-physicist`. For fitting the resulting relaxation data to a broader constitutive framework, see `science-suite:constitutive-equations`.
```

- [ ] **Step 7: Create nanocomposites-and-adaptive-materials sub-skill**

Write `plugins/science-suite/skills/nanocomposites-and-adaptive-materials/SKILL.md`:

```markdown
---
name: nanocomposites-and-adaptive-materials
description: Nanocomposite and adaptive-material modeling — effective-medium theory (Halpin-Tsai, Mori-Tanaka), percolation-aware property prediction, and self-healing/responsive composite behavior. Use when predicting effective modulus/conductivity from filler volume fraction and aspect ratio, or modeling self-healing composites.
---

# Nanocomposites & Adaptive Materials

Predicting effective properties of filler-reinforced and adaptive composite materials.

## Expert Agent

- **`continuum-mechanics-engineer`** — Effective-medium modeling and composite constitutive prediction.

## Effective-Medium Theories

| Method | Predicts | Best For |
|--------|----------|----------|
| Halpin-Tsai | Effective modulus from filler aspect ratio + volume fraction | Aligned or randomly-oriented short-fiber/platelet composites, low-to-moderate loading |
| Mori-Tanaka | Effective modulus via Eshelby inclusion theory, accounts for filler-filler interaction | Moderate-to-high filler loading where Halpin-Tsai's dilute assumption breaks down |

Both are mean-field approximations — they assume filler is well-dispersed. Neither captures percolation-driven property jumps (see below); use them for modulus prediction below the percolation threshold, not across it.

## Percolation Threshold

Near a critical filler volume fraction (the percolation threshold `φc`), the filler network transitions from disconnected to fully connected, causing a sharp (often orders-of-magnitude) jump in properties governed by connectivity — most dramatically electrical/thermal conductivity, but also a stiffness upturn beyond what Halpin-Tsai/Mori-Tanaka predict.

**Do not derive percolation theory (critical exponents, universality class, lattice vs. continuum percolation) here** — cross-reference `science-suite:statistical-physics-hub`'s percolation/collective-phenomena content (see `glass-and-collective-dynamics`) for that derivation. This skill's job is applying the percolation threshold as an input to composite property prediction, not deriving it from first principles.

## Self-Healing & Responsive Composites

Self-healing nanocomposites typically combine:
1. A transient-network matrix (see `science-suite:transient-networks-and-can`) providing the reversible bond-breaking/reforming mechanism.
2. Filler reinforcement (this skill's effective-medium methods) providing mechanical stiffness.

Model these as a composition: fit the matrix's transient-network behavior independently, then apply effective-medium theory using the matrix's (rate-dependent) modulus as the base-material modulus input — not as a single combined model from scratch.

## Workflow

1. Determine filler volume fraction, aspect ratio, and whether the system is below or near the percolation threshold.
2. Below threshold: apply Halpin-Tsai (dilute) or Mori-Tanaka (concentrated) for modulus prediction.
3. Near/above threshold: flag the effective-medium prediction as unreliable and delegate the connectivity-driven property jump to `statistical-physics-hub`.
4. For adaptive/self-healing systems, decompose into transient-network matrix + filler reinforcement per the composition approach above.

## Delegation

For percolation-threshold statistical mechanics, delegate to `statistical-physicist` via `science-suite:statistical-physics-hub`. For the matrix's own transient-network rheology, see `science-suite:transient-networks-and-can`.
```

- [ ] **Step 8: Register the hub in plugin.json**

Read `plugins/science-suite/.claude-plugin/plugin.json`. Add `"./skills/continuum-mechanics-and-rheology",` to the `skills` array (insert alphabetically, after `"./skills/bayesian-ude-workflow",` and before `"./skills/correlation-analysis",`).

- [ ] **Step 9: Verify**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/science-suite
PYTHONPATH=. python3 tools/validation/skill_validator.py plugins/science-suite 2>&1 | grep -A3 -E "continuum-mechanics|fem-fea|constitutive-equations|dma-rheology|harmonic-response|transient-networks|nanocomposites"
PYTHONPATH=. python3 tools/validation/context_budget_checker.py 2>&1 | grep -E "continuum-mechanics-and-rheology|fem-fea|constitutive-equations|dma-rheology|harmonic-response-superposition|transient-networks-and-can|nanocomposites-and-adaptive-materials"
python3 -c "
import json
skills = json.load(open('plugins/science-suite/.claude-plugin/plugin.json'))['skills']
assert './skills/continuum-mechanics-and-rheology' in skills, skills
assert len(skills) == 30, len(skills)
print('OK: 30 hub skills')
"
```
Expected: validators show no errors for the 7 new files; `context_budget_checker.py` shows all 7 under the 4,000-token budget (`pass`); `OK: 30 hub skills`.

- [ ] **Step 10: Commit**

```bash
git add plugins/science-suite/skills/continuum-mechanics-and-rheology/ plugins/science-suite/skills/fem-fea/ \
  plugins/science-suite/skills/constitutive-equations/ plugins/science-suite/skills/dma-rheology/ \
  plugins/science-suite/skills/harmonic-response-superposition/ plugins/science-suite/skills/transient-networks-and-can/ \
  plugins/science-suite/skills/nanocomposites-and-adaptive-materials/ plugins/science-suite/.claude-plugin/plugin.json
git commit -m "$(cat <<'EOF'
feat(science-suite): add continuum-mechanics-and-rheology hub + 6 sub-skills

Covers FEM/FEA, constitutive equations, DMA/rheology, harmonic response and
time-temperature superposition, transient networks (physical + covalent
adaptable networks/vitrimers), and nanocomposites/adaptive materials --
none of this was covered anywhere in science-suite before.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Wire the new hub into science-hub's top-level routing

**Files:**
- Modify: `plugins/science-suite/skills/science-hub/SKILL.md` (description, Expert Agents list, Hub Skills list, routing decision tree, routing table)

**Interfaces:**
- Consumes: `continuum-mechanics-engineer` agent (Task 3), `continuum-mechanics-and-rheology` hub (Task 4).
- Produces: `science-hub`'s routing surface includes the new agent and hub so top-level queries about FEM/rheology/materials reach them.

- [ ] **Step 1: Add the agent to the Expert Agents list**

Read `plugins/science-suite/skills/science-hub/SKILL.md`. In "## Expert Agents", add (in alphabetical position among the existing entries):
```markdown
- **`continuum-mechanics-engineer`** — FEM/FEA, constitutive modeling, DMA/rheology, transient networks, nanocomposites
```

- [ ] **Step 2: Add the hub to the Hub Skills list**

In "## Hub Skills", add:
```markdown
- [Continuum Mechanics & Rheology](../continuum-mechanics-and-rheology/SKILL.md) — FEM/FEA, constitutive modeling, DMA/rheology, transient networks, nanocomposites
```

- [ ] **Step 3: Add a branch to the routing decision tree**

In the `## Routing Decision Tree` code block, add a new branch (position it near the simulation/nonlinear-dynamics branches since it's physically adjacent domain territory):
```
+-- FEM/FEA, constitutive modeling, DMA/rheology, transient networks, nanocomposites?
|   --> science-suite:continuum-mechanics-and-rheology
|
```

- [ ] **Step 4: Add a row to the routing table**

In `## Routing Table`, add:
```
| FEM, FEA, finite element, weak form, constitutive equation, hyperelastic, viscoelastic, DMA, storage modulus, loss modulus, rheology, shear rheology, extensional rheology, WLF, time-temperature superposition, master curve, vitrimer, covalent adaptable network, transient network, nanocomposite, Halpin-Tsai, Mori-Tanaka, percolation composite | science-suite:continuum-mechanics-and-rheology |
```

- [ ] **Step 5: Update the description field**

Append to the end of the `description` block (before the closing quote/dash): `; continuum-mechanics/FEM/FEA/constitutive equations/DMA/rheology/transient networks/vitrimers/nanocomposites`. Use the hyphenated form `continuum-mechanics` (not "continuum mechanics") so it counts toward Step 6's grep verification below.

- [ ] **Step 6: Verify**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -c "continuum-mechanics" plugins/science-suite/skills/science-hub/SKILL.md
```
Expected: at least `5` (description, Expert Agents, Hub Skills, decision tree, routing table).

```bash
PYTHONPATH=. python3 tools/validation/skill_validator.py plugins/science-suite 2>&1 | grep -A3 "science-hub"
```
Expected: no description-length errors (science-hub's description is already long — verify it still fits within `skillListingMaxDescChars`; if it doesn't, trim an existing less-critical clause rather than the new addition).

- [ ] **Step 7: Commit**

```bash
git add plugins/science-suite/skills/science-hub/SKILL.md
git commit -m "$(cat <<'EOF'
refactor(science-suite): wire continuum-mechanics-and-rheology into science-hub

Top-level router now discovers the new hub and agent for FEM/rheology/
materials-science queries.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Extend statistical-physicist — glass/collective phenomena + physical learning

**Files:**
- Create: `plugins/science-suite/skills/glass-and-collective-dynamics/SKILL.md`
- Create: `plugins/science-suite/skills/physical-learning-systems/SKILL.md`
- Modify: `plugins/science-suite/skills/statistical-physics-hub/SKILL.md` (Core Skills list, routing decision tree, routing table)

**Interfaces:**
- Consumes: nothing from earlier tasks (independent of Tasks 3-5, but shares the `statistical-physicist` agent).
- Produces: 2 new sub-skills discoverable via `statistical-physics-hub`'s routing table; `nanocomposites-and-adaptive-materials` (Task 4) cross-references `glass-and-collective-dynamics` for percolation content — confirm that reference resolves once this task lands.

- [ ] **Step 0: Check for overlap with the existing `correlation-*` skill family**

```bash
cd /home/wei/Documents/GitHub/MyClaude
ls plugins/science-suite/skills | grep -i correlation
grep -l "percolation\|jamming\|glass transition\|aging\|dynamical heterogeneity\|coupled learning\|contrastive Hebbian" plugins/science-suite/skills/correlation-*/SKILL.md 2>/dev/null
```
Expected: shows the 5 existing `correlation-*` files (`correlation-analysis`, `correlation-computational-methods`, `correlation-experimental-data`, `correlation-math-foundations`, `correlation-physical-systems`). Read each. Per spec §5's finding (checked at design time: zero percolation matches in the `correlation-*` family or `statistical-physics-hub`), these should show no overlap with glass/jamming/aging/percolation/physical-learning content — but re-confirm at implementation time rather than trusting the design-time check, since the corpus may have changed. Document the outcome (overlap found vs. none) in the Task 6 commit message. If genuine overlap is found, fold the new content into the existing `correlation-*` file instead of creating a duplicate.

- [ ] **Step 1: Create glass-and-collective-dynamics**

Write `plugins/science-suite/skills/glass-and-collective-dynamics/SKILL.md`:

```markdown
---
name: glass-and-collective-dynamics
description: Glass physics, jamming, and collective phenomena in disordered/soft-matter systems — random landscapes, aging, cooperative dynamics, and percolation theory. Use when analyzing glassy relaxation, jamming transitions, aging dynamics, cooperative/collective particle motion, or percolation thresholds in filler networks or disordered media.
---

# Glass & Collective Dynamics

Cooperative and collective phenomena in disordered and glassy soft-matter systems.

## Expert Agent

- **`statistical-physicist`** — Glass physics, jamming, percolation, and collective-dynamics theory.

## Glass & Jamming

- **Glass transition**: dynamical arrest without long-range structural order — relaxation times diverge (often described phenomenologically by Vogel-Fulcher-Tammann, `τ = τ₀ exp(DT₀/(T-T₀))`) as temperature approaches Tg from above.
- **Jamming**: analogous arrest driven by density/packing rather than temperature — a jammed packing is mechanically rigid despite being disordered, characterized by the jamming point `φⱼ` (random close packing for frictionless spheres, ≈0.64 in 3D).
- **Aging**: glassy/jammed systems never fully equilibrate on experimental timescales — their properties (relaxation time, mechanical response) continue to evolve ("age") with waiting time since preparation/quench, breaking time-translation invariance.

## Cooperative & Collective Dynamics

Dynamics in dense/disordered systems are often dominated by cooperative rearrangements (groups of particles moving together) rather than independent single-particle motion — this is the microscopic origin of the dramatic slowdown near the glass transition despite only modest structural changes. Dynamical heterogeneity (spatially varying local relaxation rates) is the key diagnostic, measured via multi-point correlation functions (e.g. the four-point susceptibility χ₄).

## Random (Rugged) Energy Landscapes

Glassy and jammed systems are naturally described as slow dynamics on a rugged, high-dimensional energy landscape with an exponentially large number of local minima separated by barriers — rather than a single well-defined ground state. Two conceptual pictures recur: (1) mean-field spin-glass models (random-energy model, p-spin models), where the landscape structure connects directly to replica-symmetry-breaking (RSB) — the free-energy landscape itself splits into a hierarchy of metastable states below a dynamical transition temperature; (2) the potential-energy-landscape (inherent-structure) framework for structural glasses, where aging and slow relaxation correspond to the system searching a landscape whose accessible minima become progressively deeper/rarer as temperature or waiting time increases. Both pictures explain why glassy relaxation is non-exponential and history-dependent (aging) rather than a simple activated process over a single barrier.

## Percolation Theory

- **Percolation threshold `pc`**: the critical occupation/connection probability at which a system-spanning cluster first appears.
- **Universality**: near `pc`, cluster-size distributions and connectivity properties follow power laws with universal critical exponents depending only on dimensionality and percolation type (site vs. bond), not microscopic details.
- **Application to filler networks**: a nanocomposite's filler particles form a percolating network above their own geometric percolation threshold — this is the statistical-mechanics basis for the property jump referenced (but not derived) in `science-suite:nanocomposites-and-adaptive-materials`.

## Routing Within This Skill

| Question | Answer here covers |
|----------|----------------------|
| "Why does relaxation slow down approaching Tg?" | Glass transition phenomenology, VFT law |
| "What's the jamming point for this packing?" | Jamming transition, random close packing |
| "Why hasn't this system's properties stabilized?" | Aging, breaking of time-translation invariance |
| "Are these particles moving cooperatively?" | Dynamical heterogeneity, four-point susceptibility |
| "Why does this system have so many metastable states?" | Rugged energy landscapes, RSB, inherent-structure framework |
| "What filler fraction gives a percolating network?" | Percolation threshold, critical exponents |

## Delegation

For applying a percolation threshold to nanocomposite property prediction (not deriving the threshold itself), see `science-suite:nanocomposites-and-adaptive-materials`. For active-matter collective phenomena specifically (self-propelled particles, flocking), see `science-suite:active-matter` — that skill covers a different (non-equilibrium, driven) mechanism for collective behavior than the equilibrium/quenched-disorder mechanisms here.
```

- [ ] **Step 2: Create physical-learning-systems**

Write `plugins/science-suite/skills/physical-learning-systems/SKILL.md`:

```markdown
---
name: physical-learning-systems
description: Physical and energy-based learning in disordered and soft-matter systems — coupled learning, contrastive Hebbian learning in physical (mechanical/electrical) networks, plasticity and memory formation in disordered materials, and Hopfield-style energy-based learning applied to physical substrates. Distinct from classical ML algorithms — this is learning as a physical/statistical-mechanics phenomenon, not a learning algorithm implemented in software.
---

# Physical Learning Systems

Learning as a physical phenomenon: materials and networks that adapt their own structure in response to training signals, not software implementing a learning algorithm.

## Expert Agent

- **`statistical-physicist`** — Physical/energy-based learning theory in disordered and soft-matter systems.

## Scope Boundary

This skill covers systems where the *physical substrate itself* (a mechanical network's spring stiffnesses, an electrical network's conductances, a material's internal disorder) is the thing being trained — the learning rule is implemented by physics (local rules acting on physical degrees of freedom), not by a digital algorithm. For learning algorithms implemented in software (backprop, gradient descent, classical ML), see `science-suite:machine-learning` or `science-suite:deep-learning-hub` — those are a different subject even when the vocabulary overlaps.

## Coupled Learning

A physical network (e.g. a network of variable resistors or elastic springs) is subjected to two boundary conditions: a "free" state (task input only) and a "clamped" state (task input plus a nudge toward the desired output). Local physical elements adjust (e.g. resistance decreases where current differs most between the two states) to reduce the discrepancy — this is a physically-realizable analog of contrastive Hebbian learning, requiring no global error backpropagation since each element only needs locally-available information (the two states of its own local variables).

## Contrastive Hebbian Learning & Energy-Based Models

Hopfield-style energy-based models store patterns as local minima of an energy function; learning adjusts the energy landscape so desired patterns become minima. Contrastive Hebbian learning trains via the difference between a "free" phase (network relaxes without a target) and a "clamped" phase (network relaxes with the target imposed) — mathematically related to coupled learning above but originating from the neural-network/statistical-mechanics literature (Boltzmann machines) rather than the materials-physics literature.

## Plasticity & Memory in Disordered Materials

Disordered materials (glasses, granular packings, amorphous solids) can exhibit memory effects: their response to a perturbation depends on their loading history in ways that let them "remember" prior training protocols. Cyclic loading can create memories analogous to those studied in the glass/jamming literature (see `science-suite:glass-and-collective-dynamics`) — this is the connection between glassy physics and physical learning: both concern how disordered systems' internal degrees of freedom retain a trace of past states.

## Delegation

For the underlying glassy/jamming dynamics that produce memory effects, see `science-suite:glass-and-collective-dynamics`. For the network-dynamics/dynamical-systems side of plasticity rules (treating them as attractor formation in a dynamical system), delegate to `nonlinear-dynamics-expert`. For classical ML learning algorithms (the software kind), see `science-suite:machine-learning`.
```

- [ ] **Step 3: Wire both into statistical-physics-hub**

Read `plugins/science-suite/skills/statistical-physics-hub/SKILL.md`. Add to "## Core Skills" (after the existing 7 entries):
```markdown
### [Glass & Collective Dynamics](../glass-and-collective-dynamics/SKILL.md)
Glass transition, jamming, aging, dynamical heterogeneity, and percolation theory.

### [Physical Learning Systems](../physical-learning-systems/SKILL.md)
Coupled learning, contrastive Hebbian learning in physical networks, and plasticity/memory in disordered materials.
```

Add to the routing decision tree (before the "None of the above" fallback branch):
```
+-- Glass transition / jamming / aging / dynamical heterogeneity / percolation?
|   --> science-suite:glass-and-collective-dynamics
|
+-- Physical/energy-based learning / coupled learning / contrastive Hebbian learning in physical networks?
|   --> science-suite:physical-learning-systems
|
```

Add to the "## Skill Selection Table":
```
| Glass transition, jamming, aging, percolation | `science-suite:glass-and-collective-dynamics` |
| Coupled learning, physical/energy-based learning, plasticity in disordered materials | `science-suite:physical-learning-systems` |
```

- [ ] **Step 4: Verify**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/science-suite
PYTHONPATH=. python3 tools/validation/skill_validator.py plugins/science-suite 2>&1 | grep -A3 -E "glass-and-collective-dynamics|physical-learning-systems|statistical-physics-hub"
PYTHONPATH=. python3 tools/validation/context_budget_checker.py 2>&1 | grep -E "glass-and-collective-dynamics|physical-learning-systems"
grep -c "glass-and-collective-dynamics" plugins/science-suite/skills/nanocomposites-and-adaptive-materials/SKILL.md
```
Expected: no validator errors, both new skills pass the context budget, and the last grep returns `1` (confirms Task 4's forward-reference to this skill now resolves to an existing file).

- [ ] **Step 5: Commit**

```bash
git add plugins/science-suite/skills/glass-and-collective-dynamics/ plugins/science-suite/skills/physical-learning-systems/ \
  plugins/science-suite/skills/statistical-physics-hub/SKILL.md
git commit -m "$(cat <<'EOF'
feat(science-suite): add glass/collective-dynamics and physical-learning skills

Extends statistical-physicist's scope per user request: jamming/aging/
percolation (also the derivation backing nanocomposite property jumps) and
physical/energy-based learning in disordered systems (coupled learning,
contrastive Hebbian learning) -- distinct from classical ML algorithms.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: New graph-theory skill + cross-hub routing wiring

**Files:**
- Create: `plugins/science-suite/skills/graph-theory/SKILL.md`
- Modify: `plugins/science-suite/skills/network-coupled-dynamics/SKILL.md` (routing reference — direct content edit; this is its own standalone skill file, discovered via `nonlinear-dynamics/SKILL.md`'s routing table)
- Modify: `plugins/science-suite/skills/nonlinear-dynamics/SKILL.md` (routing-table entry — **this is the actual registered hub** in `plugin.json`'s `skills` array; `network-coupled-dynamics` is a sub-skill reached through this hub's routing table, not a hub itself, so wiring only the sub-skill file leaves the hub's own routing table undiscoverable for graph-theory queries that land on the hub first)
- Modify: `plugins/science-suite/skills/neural-network-mathematics/SKILL.md` (routing reference — spec §7 requires wiring from this hub too; it currently has zero graph-theory/spectral content)

**Interfaces:**
- Produces: `graph-theory` is a discoverable sub-skill (not registered in `plugin.json`), referenced from at least 4 existing files (`continuum-mechanics-and-rheology` and `fem-fea` already reference it from Task 4's content; `network-coupled-dynamics`, `nonlinear-dynamics` (its parent hub), and `neural-network-mathematics` wired in this task per spec §7).

- [ ] **Step 1: Confirm network-coupled-dynamics and neural-network-mathematics are the wiring targets**

```bash
cd /home/wei/Documents/GitHub/MyClaude
find plugins/science-suite/skills -iname "*network-coupled*"
grep -rl "network-coupled-dynamics" plugins/science-suite/skills/*/SKILL.md
grep -n "network-coupled-dynamics\|nonlinear-dynamics" plugins/science-suite/.claude-plugin/plugin.json
grep -c "graph-theory\|Laplacian\|spectral" plugins/science-suite/skills/neural-network-mathematics/SKILL.md
```
Expected: confirms `network-coupled-dynamics` is its own top-level skill directory (`plugins/science-suite/skills/network-coupled-dynamics/SKILL.md`), but `plugin.json`'s `skills` array registers only `./skills/nonlinear-dynamics` as the hub — `network-coupled-dynamics` is discovered through `nonlinear-dynamics/SKILL.md`'s routing table, not registered itself. Spec §7 names `network-coupled-dynamics` as a wiring target, but per this repo's own hub convention (root CLAUDE.md) the routing table that actually needs the new entry lives in the hub file, `nonlinear-dynamics/SKILL.md` — so Step 3a wires the sub-skill file directly (cheap, harmless) AND Step 3a2 adds the routing-table entry in `nonlinear-dynamics/SKILL.md` (the file a top-level query actually resolves through first). The `neural-network-mathematics` grep should show no existing graph-theory/spectral content (if the repo has changed and it already has some, adapt Step 3b's wording accordingly rather than duplicating it).

- [ ] **Step 2: Create the graph-theory skill**

Write `plugins/science-suite/skills/graph-theory/SKILL.md`:

```markdown
---
name: graph-theory
description: Graph theory foundations — spectral graph theory, graph algorithms, and network topology metrics. Use when the task requires reasoning about a graph/network structure itself (spectral properties, centrality, connectivity, community structure) rather than a specific application domain (GNNs, coupled oscillators, FEM meshes) that happens to use a graph representation.
---

# Graph Theory

Cross-cutting graph-theoretic foundations used across GNN, network-dynamics, and mesh-representation work.

## Expert Agents

Consulted by multiple agents depending on the application: `neural-network-master` (GNN architecture), `nonlinear-dynamics-expert` (coupled-oscillator network topology), `continuum-mechanics-engineer` (FEM mesh connectivity).

## Core Concepts

| Concept | Definition | Application |
|---------|------------|-------------|
| Adjacency/Laplacian matrix | `A` (connections), `L = D - A` (degree-adjacency) | Spectral clustering, GNN message passing, diffusion on networks |
| Spectral graph theory | Eigenvalues/eigenvectors of `L` | Graph partitioning (Fiedler vector), synchronization thresholds in coupled-oscillator networks |
| Centrality measures | Degree, betweenness, eigenvector centrality | Identifying influential nodes in a network |
| Connectivity | Vertex/edge connectivity, connected components | Percolation threshold (structural side — see `science-suite:glass-and-collective-dynamics` for the statistical-mechanics side), network robustness |
| Community structure | Modularity, spectral/hierarchical clustering | Detecting mesoscale structure in large networks |

## Spectral Graph Theory Quick Reference

- For the combinatorial Laplacian `L = D - A` of an undirected graph, the smallest eigenvalue is always 0 (with eigenvector = all-ones); the number of zero eigenvalues equals the number of connected components. This does not hold unqualified for directed graphs or normalized-Laplacian variants (`L_sym`, `L_rw`) — state the convention being used before relying on this property.
- The second-smallest eigenvalue (algebraic connectivity, "Fiedler value") governs how well-connected a graph is — small values indicate near-disconnection or bottlenecks; its eigenvector (Fiedler vector) gives a natural graph bipartition for spectral clustering.
- For coupled-oscillator networks, the Laplacian eigenvalue spectrum directly determines synchronization stability (via the master stability function) — this is the graph-theory input `nonlinear-dynamics-expert` needs for synchronization analysis.

## Application-Specific Delegation

| Application | Delegate To | This skill provides |
|-------------|-------------|------------------------|
| GNN architecture (message passing, pooling) | `neural-network-master` | Adjacency/Laplacian formalism underlying message-passing operators |
| Coupled-oscillator synchronization | `nonlinear-dynamics-expert` | Laplacian spectrum for master-stability-function analysis |
| FEM mesh connectivity/quality | `continuum-mechanics-engineer` | Graph representation of mesh topology, connectivity checks |
| Julia-native GNN implementation | `julia-ml-hpc` | GNNGraphs.jl's graph representation conventions |

This skill is deliberately thin — it holds the shared mathematical vocabulary so the 4 application-specific agents above don't each re-explain spectral graph theory independently. Route here first when a question is about the graph structure itself, then to the application-specific agent for what to do with it.
```

- [ ] **Step 3a: Wire into network-coupled-dynamics**

Read `plugins/science-suite/skills/network-coupled-dynamics/SKILL.md`. Add, near its existing "Graph Laplacian Construction" section:
```markdown
For the underlying spectral graph theory (Laplacian eigenvalues, algebraic connectivity, Fiedler vector), see `science-suite:graph-theory`.
```

- [ ] **Step 3a2: Wire into nonlinear-dynamics (the actual registered hub)**

Read `plugins/science-suite/skills/nonlinear-dynamics/SKILL.md`. This is the hub registered in `plugin.json` — `network-coupled-dynamics` is one of its routed sub-skills, so the hub's own routing table also needs a graph-theory entry for queries that resolve to the hub first. Add a row to its routing table (near the existing `network-coupled-dynamics` entry):
```markdown
| Graph/network spectral properties (Laplacian eigenvalues, algebraic connectivity, Fiedler vector) underlying coupled-network dynamics | `science-suite:graph-theory` |
```

Verify:
```bash
grep -c "graph-theory" plugins/science-suite/skills/nonlinear-dynamics/SKILL.md
```
Expected: at least `1`.

- [ ] **Step 3b: Wire into neural-network-mathematics**

Read `plugins/science-suite/skills/neural-network-mathematics/SKILL.md` (spec §7 explicitly requires this hub as a wiring target, alongside `network-coupled-dynamics`). Add, near its existing linear-algebra/spectral content:
```markdown
For the graph-theoretic foundations underlying GNN message passing (adjacency/Laplacian matrices, spectral graph theory), see `science-suite:graph-theory`.
```

- [ ] **Step 4: Confirm the continuum-mechanics-and-rheology reference from Task 4 resolves**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -c "graph-theory" plugins/science-suite/skills/continuum-mechanics-and-rheology/SKILL.md plugins/science-suite/skills/fem-fea/SKILL.md
```
Expected: at least `1` in each (Task 4 already added these references pointing forward to this skill; this step just confirms they now resolve to a real file).

- [ ] **Step 5: Verify**

```bash
PYTHONPATH=. python3 tools/validation/skill_validator.py plugins/science-suite 2>&1 | grep -A3 "graph-theory"
PYTHONPATH=. python3 tools/validation/context_budget_checker.py 2>&1 | grep "graph-theory"
grep -rc "science-suite:graph-theory" plugins/science-suite/skills/*/SKILL.md | grep -v ":0" | wc -l
```
Expected: no validator errors; `graph-theory` passes the context budget; at least `5` files reference it (`continuum-mechanics-and-rheology` and `fem-fea` from Task 4, `network-coupled-dynamics` from Step 3a, `nonlinear-dynamics` from Step 3a2, `neural-network-mathematics` from Step 3b).

- [ ] **Step 6: Commit**

```bash
git add plugins/science-suite/skills/graph-theory/
git add plugins/science-suite/skills/network-coupled-dynamics/SKILL.md plugins/science-suite/skills/nonlinear-dynamics/SKILL.md plugins/science-suite/skills/neural-network-mathematics/SKILL.md
git commit -m "$(cat <<'EOF'
feat(science-suite): add graph-theory skill, wire into GNN/network/FEM hubs

Cross-cutting spectral-graph-theory vocabulary shared by GNN architecture
work (neural-network-mathematics), coupled-oscillator synchronization
analysis (network-coupled-dynamics, wired both directly and via its actual
parent hub nonlinear-dynamics), and FEM mesh connectivity -- added as a
sub-skill, not a new top-level hub. Wires the hub named in spec section 7
as network-coupled-dynamics via its actual plugin.json-registered parent
(nonlinear-dynamics), plus neural-network-mathematics and
continuum-mechanics-and-rheology.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Python-tooling consolidation (absorbs dev-suite's 5 skills)

**⚠️ This task reads from `plugins/dev-suite/skills/{async-python-patterns,python-packaging,python-performance-optimization,python-toolchain,uv-package-manager}/` — run this BEFORE the trim plan's Task 8, which deletes those directories.**

**Files:**
- Modify: `plugins/science-suite/skills/python-development/SKILL.md` (absorb `async-python-patterns`, `python-toolchain`, and a new profiling section from `python-performance-optimization`)
- Modify: `plugins/science-suite/skills/python-packaging-advanced/SKILL.md` (absorb `python-packaging`, `uv-package-manager`)
- Modify: `plugins/science-suite/skills/research-and-domains/SKILL.md` (fix the `dev-suite python-toolchain hub` cross-reference)

**Interfaces:**
- Produces: `python-development` and `python-packaging-advanced` contain all genuinely-additive content from the 5 source files; `research-and-domains` no longer points at a soon-to-be-deleted dev-suite hub.

- [ ] **Step 1: Read all 5 source files and both destination files**

```bash
cd /home/wei/Documents/GitHub/MyClaude
cat plugins/dev-suite/skills/async-python-patterns/SKILL.md
cat plugins/dev-suite/skills/python-toolchain/SKILL.md
cat plugins/dev-suite/skills/python-packaging/SKILL.md
cat plugins/dev-suite/skills/uv-package-manager/SKILL.md
cat plugins/dev-suite/skills/python-performance-optimization/SKILL.md
cat plugins/science-suite/skills/python-development/SKILL.md
cat plugins/science-suite/skills/python-packaging-advanced/SKILL.md
```
Read all 7 outputs before making any edit — the merge decisions in the following steps depend on knowing exactly what's already covered vs. genuinely new.

- [ ] **Step 2: Fold async-python-patterns into python-development**

Compare `async-python-patterns`'s content against `python-development`'s existing coverage of structured concurrency/TaskGroups. For each distinct concept in `async-python-patterns` NOT already present in `python-development` (e.g. a specific `asyncio` pattern, a specific error-handling idiom for async code, a specific testing pattern for async functions), append it as a new subsection under a `## Async Patterns` heading in `python-development/SKILL.md` (create the heading if it doesn't exist; if `python-development` already has an async section, add to it instead of creating a duplicate heading). Do not port content that's already covered — check by grep for the concept's distinctive terms first.

- [ ] **Step 3: Fold python-toolchain into python-development**

Same process: compare `python-toolchain`'s uv-based packaging / general toolchain content (type hints, error handling, legacy migration guidance per its dev-hub routing table entry) against what `python-development` already covers. Port genuinely new material under appropriate existing or new subsections (e.g. `## Type Hints`, `## Legacy Migration` if those don't already exist and the source file has substantive content on them).

- [ ] **Step 4: Add a new Python-profiling section to python-development**

Per the spec's explicit finding: `performance-tuning` is Julia-only (title "Julia Performance Tuning", `@code_warntype`/`@profview`/BenchmarkTools.jl content, `julia-pro` as its Expert Agent) — do NOT fold Python profiling content there. Instead, read `python-performance-optimization`'s full content (cProfile, line_profiler, py-spy, memory_profiler) and add it as a new `## Python Profiling` section in `python-development/SKILL.md`, since that file currently only briefly mentions `py-spy` in a checklist item with no depth.

- [ ] **Step 5: Fold python-packaging and uv-package-manager into python-packaging-advanced**

Compare both source files against `python-packaging-advanced`'s existing uv/pyproject.toml/workspace/monorepo coverage. Port genuinely new material (any uv subcommand, workflow, or packaging pattern not already documented) into appropriate subsections.

- [ ] **Step 6: Fix the research-and-domains cross-reference**

Read `plugins/science-suite/skills/research-and-domains/SKILL.md`. Find:
```
+-- Python systems / packaging / performance?
|   (These are co-located here for scientific Python workflows;
|    for general Python toolchain, see dev-suite python-toolchain hub)
|   --> science-suite:python-development / science-suite:python-packaging-advanced
|   --> science-suite:rust-extensions / science-suite:type-driven-design / science-suite:modern-concurrency
|   --> science-suite:robust-testing (Hypothesis property-based tests, mutation testing)
```
Replace the parenthetical with one reflecting that this is now the canonical Python-tooling home in this marketplace, not a scientific-specific subset of a broader dev-suite home:
```
+-- Python systems / packaging / performance?
|   (Canonical Python-tooling home for this marketplace as of the
|    dev-suite trim -- covers general uv/packaging/async/profiling
|    as well as scientific-specific concerns)
|   --> science-suite:python-development / science-suite:python-packaging-advanced
|   --> science-suite:rust-extensions / science-suite:type-driven-design / science-suite:modern-concurrency
|   --> science-suite:robust-testing (Hypothesis property-based tests, mutation testing)
```

Verify:
```bash
grep -c "dev-suite python-toolchain" plugins/science-suite/skills/research-and-domains/SKILL.md
```
Expected: `0`

- [ ] **Step 7: Validate**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/skill_validator.py plugins/science-suite 2>&1 | grep -A3 -E "python-development|python-packaging-advanced|research-and-domains"
PYTHONPATH=. python3 tools/validation/context_budget_checker.py 2>&1 | grep -E "python-development|python-packaging-advanced"
```
Expected: no validator errors; both destination files still pass the context budget after the merge (if either now exceeds 4,000 tokens, split the newly-added profiling/async content into a small new sub-skill rather than leaving the hub oversized — check the checker's output before deciding). If the fallback triggers: split `python-development/SKILL.md`'s new `## Python Profiling` section into `plugins/science-suite/skills/python-profiling/SKILL.md` (frontmatter `name: python-profiling`, `description:` summarizing cProfile/line_profiler/py-spy/memory_profiler usage, discovered via a routing-table line added to `python-development/SKILL.md`: `For deep Python profiling workflows (cProfile, line_profiler, py-spy, memory_profiler), see `science-suite:python-profiling`.`) rather than `python-packaging-advanced`'s content, since profiling is the section most likely to push the token count over budget. If this fallback fires, also add `python-profiling` to Task 10 Step 1's directory recount so the final `ls`-based count includes it (the `plugin.json` hub-skill count is unaffected either way, since sub-skills aren't registered there).

- [ ] **Step 8: Commit**

```bash
git add plugins/science-suite/skills/python-development/SKILL.md plugins/science-suite/skills/python-packaging-advanced/SKILL.md \
  plugins/science-suite/skills/research-and-domains/SKILL.md
git commit -m "$(cat <<'EOF'
feat(science-suite): absorb dev-suite's 5 Python-tooling skills

Merge, not copy-in: folds genuinely-additive content from
async-python-patterns/python-toolchain (-> python-development),
python-packaging/uv-package-manager (-> python-packaging-advanced), and
python-performance-optimization (-> new Python Profiling section in
python-development, NOT the Julia-only performance-tuning skill). Fixes
the research-and-domains cross-reference to the dev-suite hub these skills
came from -- that hub is removed by the sibling trim plan.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Accuracy audit fixes (BifurcationKit, PointProcesses.jl, SequentialMonteCarlo, Flux)

**Files:**
- Modify: `plugins/science-suite/skills/bifurcation-analysis/SKILL.md`
- Modify: `plugins/science-suite/agents/julia-pro.md`
- Modify: `plugins/science-suite/agents/nonlinear-dynamics-expert.md` (BifurcationKit-as-primary content — confirmed present in decision tree, capability table, and skill-selection table)
- Check/modify: `plugins/science-suite/skills/nonlinear-dynamics/SKILL.md`, `plugins/science-suite/skills/chaos-attractors/SKILL.md`, `plugins/science-suite/skills/differential-equations/SKILL.md`, `plugins/science-suite/skills/julia-mastery/SKILL.md`, `plugins/science-suite/skills/jax-julia-interop/SKILL.md` (repo-wide BifurcationKit audit — confirmed present in all 5, not just `bifurcation-analysis`/`julia-pro`)
- Modify: `plugins/science-suite/skills/point-processes/SKILL.md` (or wherever the `PointProcesses.jl` recommendation lives — confirm exact filename first)
- Check (no guaranteed edit): `plugins/science-suite/skills/julia-hpc-distributed/SKILL.md` (or `parallel-computing/SKILL.md`) for `SequentialMonteCarlo` mentions
- Check (no guaranteed edit): any Flux.jl recommendation for new SciML neural closures, including in `julia-ml-hpc.md` (do not blanket-exclude this file)
- Modify: `plugins/science-suite/skills/consensus-mcmc-pigeons/SKILL.md` (Pigeons DEV override / DynamicPPL v0.40 accuracy check — spec §9)
- Modify: `plugins/science-suite/skills/turing-model-design/SKILL.md` (same Pigeons DEV override / DynamicPPL v0.40 accuracy check — spec §9 names both files explicitly)
- Modify: `plugins/science-suite/skills/modeling-toolkit/SKILL.md` (MTK v9/v11 split accuracy check — spec §9)
- Check + correct-if-needed: `plugins/science-suite/skills/sciml-modern-stack/SKILL.md` — spec §9 names this file explicitly alongside `modeling-toolkit`; this file's own header marks it FROZEN at 78% context budget ("Do not add new content"), which blocks *additions* but not *corrections* to existing wrong text — read it and fix any existing MTK-version claim that contradicts the v9/v11 split in place (edit, don't append); if no existing claim needs correction, leave the file untouched and note that explicitly in the commit message rather than silently skipping it
- Modify: `plugins/science-suite/skills/julia-graph-neural-networks/SKILL.md` (GNNLux/GNNGraphs/GNNlib DEV-override accuracy check — spec §9)

**Interfaces:**
- Produces: zero recommendations of `BifurcationKit` as "Recommended"/primary anywhere in science-suite (not just the `juliacall` escape hatch in 2 files — 7 files total carry BifurcationKit-as-primary content); zero unhedged `PointProcesses.jl` recommendations; zero `SequentialMonteCarlo` recommendations in multi-package Julia envs; zero Flux.jl recommendations for new SciML neural closures (Lux.jl only), audited across all files including `julia-ml-hpc.md`; `consensus-mcmc-pigeons` AND `turing-model-design` both state the Pigeons DEV override (PR #409, `4d981068`) / DynamicPPL v0.40 pairing; `modeling-toolkit` states the MTK v9 (`@sciml`/`@bayes`) vs MTK v11 (`@pinn`) split, and `sciml-modern-stack` is confirmed free of a contradicting version claim (or corrected in place); `julia-graph-neural-networks` states the GNNLux/GNNGraphs/GNNlib DEV-override-against-monorepo-master status.

- [ ] **Step 1: Fix bifurcation-analysis.md's BifurcationKit recommendation**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -n "BifurcationKit\|AUTO-07p\|tick" plugins/science-suite/skills/bifurcation-analysis/SKILL.md
```

Read the file and locate: (a) the description line naming BifurcationKit, (b) the Quick Start section, (c) the Python-escape-hatch table marking `juliacall → BifurcationKit` as "Recommended". For each:

- In the description and Quick Start, if BifurcationKit is presented as the primary/default tool, add or strengthen a caveat that it's blocked on Julia 1.12 (per this repo's CLAUDE.md prohibited-package list) and point to `AUTO-07p` (Fortran, already documented in the skill per the spec's finding) as the working alternative.
- In the escape-hatch table, change the `juliacall → BifurcationKit` row's status from "Recommended" to something like "Blocked (MiniQhull ≥0.4 build failure on Julia 1.12)" and promote the `AUTO-07p` row (or add one if it's missing from that specific table) to "Recommended".
- Do NOT introduce `tick` as a bifurcation-continuation alternative anywhere — it's a Hawkes/point-process library, unrelated to numerical continuation.

Verify:
```bash
grep -n "BifurcationKit" plugins/science-suite/skills/bifurcation-analysis/SKILL.md | grep -i "recommend"
```
Expected: no output (no remaining "Recommended" label attached to BifurcationKit).

- [ ] **Step 2: Fix julia-pro.md's BifurcationKit-as-primary content**

```bash
grep -n "BifurcationKit" plugins/science-suite/agents/julia-pro.md
```
Read each hit's surrounding context (the spec identifies: domain capability table, Domain 6 continuation section, skill-routing table, decision tree — confirm these are still the actual locations, the file may have changed). Apply the same correction pattern as Step 1: demote BifurcationKit, promote AUTO-07p, in every one of these locations — this file carries the same wrong content in multiple places, not just one.

Verify:
```bash
grep -n "BifurcationKit" plugins/science-suite/agents/julia-pro.md
```
Expected: every remaining mention (if any) is in a context explicitly marked as blocked/not-recommended, not presented as the primary tool.

- [ ] **Step 2b: Repo-wide BifurcationKit audit — 6 more files beyond bifurcation-analysis and julia-pro**

Confirmed at plan-authoring time (`grep -rl "BifurcationKit" plugins/science-suite/`) that BifurcationKit-as-primary content also exists in: `nonlinear-dynamics/SKILL.md` (description line, routing table, and an escape-hatch table row explicitly marking `juliacall → BifurcationKit.jl` as the "**recommended practical path**"), `chaos-attractors/SKILL.md` (a "Notes" line recommending `juliacall` + BifurcationKit as the replacement for unmaintained `PyDSTool`), `differential-equations/SKILL.md` (a direct "Use BifurcationKit.jl for parameter continuation" recommendation with a code sample), `julia-mastery/SKILL.md` (lists BifurcationKit.jl among `julia-pro`'s core tools, no caveat), `jax-julia-interop/SKILL.md` (multiple mentions including a worked JAX/BifurcationKit interop example), and `nonlinear-dynamics-expert.md` (decision tree step "Is the task symbolic continuation or branch tracking? --> Julia-first (BifurcationKit.jl)", plus a capability table and workflow table). Re-run the grep at implementation time to confirm this list is current (files may have changed), then read each hit's context and apply the same correction pattern as Steps 1-2: demote BifurcationKit from "recommended"/primary status, note the Julia 1.12 MiniQhull build-failure block, and point to `AUTO-07p` (or, where a Julia-free path isn't the point of the passage — e.g. `julia-mastery`'s tool inventory — simply add a caveat rather than restructuring the whole entry). Allowlist exception: a mention that is explicitly historical/contextual (e.g. "BifurcationKit.jl exists but is currently blocked on Julia 1.12") needs no further correction — the goal is zero *unqualified* recommendations, not zero mentions of the package name.

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -rln "BifurcationKit" plugins/science-suite/
```
Expected: same 8 files as the plan-authoring-time check (2 already fixed in Steps 1-2, 6 fixed in this step) — investigate any new hit not in that list.

Verify:
```bash
grep -rn "BifurcationKit" plugins/science-suite/ | grep -iE "recommended|primary|best|use.*for parameter continuation" | grep -v "AUTO-07p\|blocked\|Julia 1.12\|MiniQhull"
```
Expected: no output (any remaining "recommended"/"primary"/"use X" language attached to BifurcationKit is now paired with the blocked/Julia-1.12 caveat on the same line or immediately adjacent).

- [ ] **Step 3: Fix the point-processes PointProcesses.jl caveat**

```bash
find plugins/science-suite/skills -iname "*point-process*"
grep -n "PointProcesses" plugins/science-suite/skills/*/SKILL.md
```
Read the file that names `PointProcesses.jl` as "Best starting point in Julia". Add an explicit caveat immediately after that claim: registry tagging is pending for this package (per this repo's CLAUDE.md prohibited-package list) — it is not currently installable/usable, and Python `tick` or the skill's existing Julia-free guidance should be used instead.

Verify:
```bash
grep -A2 "PointProcesses" <the file found above> | grep -i "pending\|not.*install\|caveat\|unavailable"
```
Expected: the caveat text is present immediately after the `PointProcesses.jl` mention.

- [ ] **Step 4: Check for SequentialMonteCarlo mentions**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -rln "SequentialMonteCarlo" plugins/science-suite/
```
For each file found, read the context. If it's recommended for use in any multi-package Julia environment without a warning, add a caveat: `SequentialMonteCarlo` causes a `RNGPool.__init__` threading race on Julia 1.12 (SIGABRT, reproducible on every OS) per this repo's CLAUDE.md — do not recommend it in any multi-package env.

Verify:
```bash
grep -A2 "SequentialMonteCarlo" plugins/science-suite/skills/*/SKILL.md 2>/dev/null | grep -i "sigabrt\|race\|caveat\|avoid"
```
Expected: any remaining mention is accompanied by the warning (empty output if there were no mentions to begin with — check Step 4's initial grep result to know which case applies).

- [ ] **Step 5: Check for Flux.jl recommendations for new SciML neural closures**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -rn "Flux\.jl\|Flux\b" plugins/science-suite/skills/*/SKILL.md plugins/science-suite/agents/*.md | grep -v "GNNLux"
```
Read every hit's context, **including all hits in `julia-ml-hpc.md`** — do not filter that file out of the search entirely (a blanket `grep -v julia-ml-hpc` would hide every Flux mention in that file, not just the acceptable general-ML ones, and would miss a genuine new-SciML-closure recommendation if one exists there). `julia-ml-hpc.md`'s own description explicitly says "Lux/Flux" as general ML/HPC coverage — that specific mention is fine as-is. Flag and fix only cases, in any file including `julia-ml-hpc.md`, where Flux is recommended specifically for a *new* SciML neural closure (UDE/neural-ODE right-hand-side network) rather than general Julia ML — per CLAUDE.md, those should recommend Lux.jl instead.

Verify: re-run the same grep after any fix and confirm remaining hits are legitimate (general ML context, not new-SciML-closure context).

- [ ] **Step 6: Check Pigeons DEV override / DynamicPPL v0.40 accuracy (spec §9)**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -n "Pigeons\|PR #409\|4d981068\|DynamicPPL" plugins/science-suite/skills/consensus-mcmc-pigeons/SKILL.md plugins/science-suite/skills/turing-model-design/SKILL.md
```
Read `consensus-mcmc-pigeons/SKILL.md`. Per this repo's global CLAUDE.md (§2b), the `@bayes` env runs a Pigeons DEV override (PR #409, commit `4d981068`) paired with DynamicPPL v0.40 — the skill currently states neither, so a reader could assume a stock released Pigeons version. Add a brief note (e.g. near the top, alongside the existing "When to use Pigeons" framing) stating this skill targets the `@bayes` env's Pigeons DEV override / DynamicPPL v0.40 pairing, not a released Pigeons version, since a DEV branch's API can drift from the last tagged release. Do not fabricate detail beyond what CLAUDE.md states.

Verify:
```bash
grep -c "DEV override\|4d981068\|DynamicPPL v0.40\|@bayes" plugins/science-suite/skills/consensus-mcmc-pigeons/SKILL.md
```
Expected: at least `1`.

Spec §9 names `turing-model-design` alongside `consensus-mcmc-pigeons` for this same check — apply the identical fix there, not just a read. Read `plugins/science-suite/skills/turing-model-design/SKILL.md` and add the same brief DEV-override/DynamicPPL v0.40 note (adapted to Turing.jl model-design framing rather than Pigeons-specific framing, e.g. near any existing version/environment context in the file).

Verify:
```bash
grep -c "DEV override\|4d981068\|DynamicPPL v0.40\|@bayes" plugins/science-suite/skills/turing-model-design/SKILL.md
```
Expected: at least `1`.

- [ ] **Step 7: Check MTK v9/v11 split accuracy (spec §9)**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -n "v9\|v11\|@sciml\|@bayes\|@pinn\|Symbolics" plugins/science-suite/skills/modeling-toolkit/SKILL.md
```
Read `modeling-toolkit/SKILL.md`. Per CLAUDE.md §2b, MTK content spans two incompatible generations: MTK v9/Symbolics v6 in `@sciml`/`@bayes`, and MTK v11/Symbolics v7 in `@pinn` (isolated because NeuralPDE needs the newer MTK). The skill currently names neither version, so a reader could assume one MTK version applies everywhere. Add a short note (e.g. after the intro paragraph) naming both versions and which env each lives in.

Spec §9 names `sciml-modern-stack` alongside `modeling-toolkit` for this same check — its own header marks it FROZEN at 78% context budget ("Do not add new content"), which blocks additive new sections but does not exempt it from the audit. Read `sciml-modern-stack/SKILL.md` and grep it for any MTK version claim:
```bash
grep -n "MTK\|ModelingToolkit\|v9\|v11" plugins/science-suite/skills/sciml-modern-stack/SKILL.md
```
If it states or implies a single MTK version without the v9/v11 split (i.e. contains a factual claim that contradicts the split), correct that existing line in place — an in-place correction of already-wrong text is not the "new content" the freeze forbids. If it makes no MTK-version claim at all (most likely, given the freeze), leave the file untouched and note in the Step 9 commit message that `sciml-modern-stack` was checked and required no change, so the deferral is an explicit, verified decision rather than a silent skip.

Verify:
```bash
grep -c "MTK v9\|MTK v11\|@pinn" plugins/science-suite/skills/modeling-toolkit/SKILL.md
```
Expected: at least `1`.

- [ ] **Step 8: Check GNNLux/GNNGraphs/GNNlib DEV-override accuracy (spec §9)**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -n "GNNLux\|GNNGraphs\|GNNlib\|monorepo\|DEV override" plugins/science-suite/skills/julia-graph-neural-networks/SKILL.md
```
Read `julia-graph-neural-networks/SKILL.md`. Per CLAUDE.md §2b, the `@gnn` env runs GNNLux/GNNGraphs/GNNlib as DEV overrides against the monorepo `master` (commit `493a3f8`), not released versions — the skill currently presents them as ordinary packages with no such caveat. Add a brief note (e.g. near the existing "Package layering" section) stating these are DEV-override dependencies tracking monorepo master, so API details may drift from the last tagged release.

Verify:
```bash
grep -c "DEV override\|monorepo master\|493a3f8" plugins/science-suite/skills/julia-graph-neural-networks/SKILL.md
```
Expected: at least `1`.

- [ ] **Step 9: Commit**

```bash
git add -u plugins/science-suite/
git commit -m "$(cat <<'EOF'
fix(science-suite): correct prohibited-package recommendations and
accuracy-audit staleness

8 files (bifurcation-analysis, julia-pro, nonlinear-dynamics-expert,
nonlinear-dynamics, chaos-attractors, differential-equations, julia-mastery,
jax-julia-interop) recommended BifurcationKit as primary despite it being
blocked on Julia 1.12 (MiniQhull build failure) -- demoted in favor of the
already-documented AUTO-07p fallback across all of them. point-processes
lacked the required PointProcesses.jl unavailability caveat. Checked for
SequentialMonteCarlo and Flux-for-new-SciML-closures (including
julia-ml-hpc, not blanket-excluded) per CLAUDE.md's prohibited list. Also
covers the remaining spec section 9 audit items: Pigeons DEV
override/DynamicPPL v0.40 (consensus-mcmc-pigeons AND turing-model-design),
MTK v9/v11 split (modeling-toolkit, plus a verified check of
sciml-modern-stack), and GNNLux/GNNGraphs/GNNlib DEV overrides against
monorepo master (julia-graph-neural-networks).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Final doc updates and validation

**Files:**
- Modify: `plugins/science-suite/README.md` (finalize agent count, hub/sub-skill counts, command count)
- Modify: `docs/suites/science-suite.rst` (add `continuum-mechanics-engineer` `.. agent::` entry; sync `simulation-expert`/`pinn-engineer` `:model:` fields with Task 2's swap; fix the "11 Agents ... 17 Hubs → 110 Sub-skills" summary line; fix the separately-stale "14 hubs routing to 112 sub-skills. Optimized for Claude Opus 4.7" sentence near the top of the file)
- Modify: `docs/agent-teams-guide.md` (add `continuum-mechanics-engineer` to the science-suite rows of the Agent Inventory table — columns are `Suite | Agent | subagent_type | Specialization`, not a model-tier table)

**Interfaces:**
- Produces: all doc counts match the final state — 12 agents (7 opus / 4 sonnet / 1 haiku), 30 hub skills, 2 commands.

- [ ] **Step 0: Verify the five-file version sync landed**

Per Global Constraints' dependency checkpoint: this plan does not itself bump the version — the sibling trim plan does, and this plan's changes ride on top of it. Confirm that bump actually happened and is synced across all 4 plugin manifests and `pyproject.toml` before proceeding:
```bash
cd /home/wei/Documents/GitHub/MyClaude
set -o pipefail
grep -h '"version"' plugins/*/.claude-plugin/plugin.json
grep -n "^version" pyproject.toml
python3 -c "
import json, re
versions = set()
for p in ['plugins/agent-core/.claude-plugin/plugin.json', 'plugins/dev-suite/.claude-plugin/plugin.json', 'plugins/research-suite/.claude-plugin/plugin.json', 'plugins/science-suite/.claude-plugin/plugin.json']:
    versions.add(json.load(open(p))['version'])
pyproject_version = re.search(r'^version\s*=\s*\"([^\"]+)\"', open('pyproject.toml').read(), re.M).group(1)
versions.add(pyproject_version)
assert len(versions) == 1, f'version drift across manifests: {versions}'
print('OK: all 5 files synced at', versions.pop())
"
```
Expected: all 5 version strings match. If they don't, STOP — do not proceed with Task 10 until the sibling trim plan's version-bump task has landed and synced; do not bump the version from this plan to paper over the mismatch (spec §10 requires one coordinated bump, not a science-suite-only one).

- [ ] **Step 1: Recount the final state**

```bash
cd /home/wei/Documents/GitHub/MyClaude
python3 -c "
import json
d = json.load(open('plugins/science-suite/.claude-plugin/plugin.json'))
print('agents:', len(d['agents']))
print('commands:', len(d['commands']))
print('hub skills:', len(d['skills']))
"
grep -c "^model: opus" plugins/science-suite/agents/*.md | grep -v ":0" | wc -l
grep -c "^model: sonnet" plugins/science-suite/agents/*.md | grep -v ":0" | wc -l
grep -c "^model: haiku" plugins/science-suite/agents/*.md | grep -v ":0" | wc -l
ls -d plugins/science-suite/skills/*/ | wc -l
```
Expected: `agents: 12`, `commands: 2`, `hub skills: 30`; opus count `7`, sonnet count `4`, haiku count `1`; total skill directories `127 + 9 (7 continuum + 2 statistical-physics extensions) + 1 (graph-theory) = 137`. Note: this arithmetic assumes Task 8's Step 7 context-budget fallback (splitting oversized profiling/async content into a new sub-skill) was NOT triggered — that fallback would add sub-skill directories not counted here without changing the hub-skill count (`plugin.json`'s `skills` array only tracks hubs). Trust the actual `ls` output over this arithmetic if they disagree.

- [ ] **Step 2: Finalize README.md**

Read `plugins/science-suite/README.md`. Update the agent tier table to add a `continuum-mechanics-engineer | opus` row, correct the summary line to "12 specialized agents (7 opus, 4 sonnet, 1 haiku)", and correct the hub/sub-skill count line to match Step 1's actual counts (30 hubs, and the sub-skill count = total directories minus 30 hubs).

Verify:
```bash
grep -n "12 specialized agents\|7 opus, 4 sonnet, 1 haiku" plugins/science-suite/README.md
```
Expected: shows the corrected line.

- [ ] **Step 3: Update docs/suites/science-suite.rst**

Read `docs/suites/science-suite.rst`. Four fixes in this file (confirm exact current text with a fresh grep first, since the plan may have gone stale — the repo state quoted below is what was confirmed when this plan was written):

1. Add a new `.. agent::` directive for `continuum-mechanics-engineer`, following the existing format (each entry has `:description:`, `:model:`, `:version:`):
```rst
.. agent:: continuum-mechanics-engineer
   :description: Expert in FEM/FEA, constitutive modeling, DMA/rheology, transient networks (CAN/vitrimers), and nanocomposites.
   :model: opus
   :version: <current version string, from pyproject.toml/plugin.json>
```
2. Sync the `simulation-expert` directive's `:model: opus` to `:model: sonnet`, and the `pinn-engineer` directive's `:model: sonnet` to `:model: opus`, matching Task 2's swap — these directives currently still show the pre-swap tiers and this task is the only one that touches this file.
3. Fix the summary line currently reading `**Version:** 3.5.2 | **11 Agents** | **2 Registered Commands** | **17 Hubs → 110 Sub-skills** | **5 Hook Events**` to the Step 1 counts (12 agents, 30 hubs, sub-skill count = total dirs minus 30).
4. Fix the separate, independently-stale sentence near the top of the file — `Uses the :term:`Hub Skill` architecture with 14 hubs routing to 112 sub-skills. Optimized for Claude Opus 4.7 with extended context and adaptive reasoning.` — this states a *different* wrong hub/sub-skill count than item 3's line and repeats the same stale "Opus 4.7" framing fixed in Task 1's README pass. Update the counts and drop the version-pinned "Opus 4.7" language here too.

Verify:
```bash
grep -c "continuum-mechanics-engineer" docs/suites/science-suite.rst
grep -A2 "agent:: simulation-expert" docs/suites/science-suite.rst | grep ":model:"
grep -A2 "agent:: pinn-engineer" docs/suites/science-suite.rst | grep ":model:"
grep -c "Opus 4.7\|14 hubs\|112 sub-skills\|17 Hubs\|110 Sub-skills" docs/suites/science-suite.rst
```
Expected: first grep at least `1`; `simulation-expert` shows `:model: sonnet`; `pinn-engineer` shows `:model: opus`; last grep `0`.

- [ ] **Step 4: Update docs/agent-teams-guide.md**

Read `docs/agent-teams-guide.md`. Find the "Agent Inventory" table's `**science**` rows — the table's columns are `Suite | Agent | subagent_type | Specialization` (there is no model-tier column) — and add:
```markdown
| | continuum-mechanics-engineer | `science-suite:continuum-mechanics-engineer` | FEM/FEA, constitutive modeling, DMA/rheology, transient networks, nanocomposites |
```

Verify:
```bash
grep -c "continuum-mechanics-engineer" docs/agent-teams-guide.md
```
Expected: at least `1`.

- [ ] **Step 5: Full validation suite**

```bash
cd /home/wei/Documents/GitHub/MyClaude
set -o pipefail
make validate
PYTHONPATH=. python3 tools/validation/context_budget_checker.py 2>&1 | tail -30
PYTHONPATH=. python3 tools/validation/xref_validator.py 2>&1 | tail -30
uv run pytest 2>&1 | tail -20
```
`set -o pipefail` ensures a failure in `context_budget_checker.py`, `xref_validator.py`, or `pytest` is not masked by `tail`'s own success — check `$?` after each piped command, not just that `tail` printed something.

Expected: `make validate` exits 0. `context_budget_checker.py` shows all science-suite skills passing (0 oversized), with a higher total skill count than the 223-file baseline (10 new science-suite skill files added this plan — 7 from Task 4, 2 from Task 6, 1 from Task 7, matching the `127 + 7 + 2 + 1 = 137` arithmetic in Step 1, plus the Task 8 fallback's `python-profiling` if it triggered; 0 removed here — the 5 dev-suite files removed are the sibling trim plan's concern). `xref_validator.py` shows no dangling references, including the forward-reference from `nanocomposites-and-adaptive-materials` to `glass-and-collective-dynamics` (now resolved since Task 6 landed) and the `research-and-domains` fix from Task 8. `uv run pytest` passes.

- [ ] **Step 6: Commit**

```bash
git add plugins/science-suite/README.md docs/suites/science-suite.rst docs/agent-teams-guide.md
git commit -m "$(cat <<'EOF'
docs(science-suite): finalize agent/hub/sub-skill counts after expand

12 agents (7 opus, 4 sonnet, 1 haiku), 30 hub skills. Adds
continuum-mechanics-engineer to the suite reference docs and agent-teams
routing guide.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** §3 (version staleness) → Task 1. §4 (tier re-audit) → Task 2. §5 (new agent) → Tasks 3-5. §6 (statistical-physicist extension) → Task 6. §7 (graph theory) → Task 7. §8 (Python consolidation) → Task 8. §9 (accuracy audit) → Task 9. §10 (plugin.json & doc updates) → Tasks 3-5, 10. §11 (validation) → Task 10 plus per-task validator runs. §12 (out of scope) → not actioned, correctly.
- **Placeholder scan:** every new skill file has complete, technically real content (no "TBD" or "add appropriate X"); every merge task (8, 9) uses a concrete read-compare-port pattern with exact verification greps rather than "reconcile as needed."
- **Type consistency:** agent/skill names used in cross-references (`continuum-mechanics-engineer`, `glass-and-collective-dynamics`, `graph-theory`, etc.) are spelled identically everywhere they appear across Tasks 3-10 — verified via the exact grep commands in each task's Verify step. Task 7's forward-reference check (Step 4) confirms Task 4's early references to `graph-theory` resolve once Task 7 lands.
