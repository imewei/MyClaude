# Plugin Scientific Computing Redesign
**Date:** 2026-05-05
**Status:** Approved — ready for implementation
**Approach:** Parallel multi-agent (Approach B)

---

## Objective

Refocus all 4 plugin suites on scientific computing (Python JAX, Julia SciML, physics-informed AI, MD simulation) and research development. Optimize opus/sonnet/haiku model tiers for efficiency and effectiveness. Reduce token usage via description trimming and SKILL.md mode-flag gating. Align with Claude Code v2.1.128.

---

## Section 1: Agent Restructuring

### 1a. Model Tier Changes (science-suite)

| Agent | Current | New | Rationale |
|---|---|---|---|
| `jax-pro` | sonnet | opus | Custom VJPs, distributed pmap, GPU kernel design require deep reasoning |
| `julia-pro` | sonnet | opus | UDE sensitivity, ModelingToolkit v9 symbolic reasoning, package interop |
| `ml-expert` | sonnet | haiku | Classical MLOps (scikit-learn, XGBoost) is largely mechanical |

All other model tiers unchanged. Science-suite opus count: 4 → 6.

### 1b. Agent Repurposing

**`ai-engineer` → `pinn-engineer`**
- File: `plugins/science-suite/agents/pinn-engineer.md` (rename from `ai-engineer.md`)
- Model: sonnet
- Description: "Use when building physics-informed neural networks, solving PDEs with neural methods, implementing BPINN/BNNODE, or designing physics-constrained loss functions. Delegates JAX implementation to jax-pro, Julia NeuralPDE to julia-pro."
- Focus: NeuralPDE.jl, DeepXDE, BPINN/BNNODE, physics-constrained loss functions, inverse problem solving, domain decomposition

**`prompt-engineer` → `sci-workflow-engineer`**
- File: `plugins/science-suite/agents/sci-workflow-engineer.md` (rename from `prompt-engineer.md`)
- Model: sonnet
- Description: "Use when integrating LLMs into scientific pipelines, designing codegen prompts for JAX/Julia, building experiment description templates, or automating numerical workflow steps with Claude."
- Focus: LLM-assisted scientific workflows, JAX/Julia codegen prompting, experiment description templating, structured output for simulation pipelines, Claude API integration in scientific apps

### 1c. Description Trimming

All 11 science-suite agent descriptions trimmed to ≤180 chars using trigger-phrase format. Library lists move into sub-skill SKILL.md files (loaded on demand). Example:

Before: "Also covers NumPyro, NLSQ, JAX-MD/CFD, Lineax/Optimistix solvers, and interpax interpolation. Handles distributed training, custom VJPs, and GPU kernels."

After: "Core JAX stack: JIT, vmap/pmap, custom VJPs, GPU kernels. Covers NumPyro, JAX-MD, Optimistix. Delegates bifurcation to nonlinear-dynamics-expert."

Target: ≤180 chars per description, well inside `skillListingMaxDescChars`.

---

## Section 2: Skill Compression & Token Optimization

### 2a. Mode-Flag Gating

Extend the `scientific-review --mode` pattern to all science-suite and research-suite hub skills.

**Gating tiers:**
- `--mode quick`: Routing tree + agent delegation table only (always loaded)
- `--mode standard` (default): + core sub-skill descriptions (~3-4 items)
- `--mode deep`: + full library reference, constraint tables, worked examples

**Priority targets for gating:**

| Hub Skill | Content gated to `--mode deep` |
|---|---|
| `simulation-and-hpc` | Sub-skills: `md-simulation-setup`, `trajectory-analysis`, `advanced-simulations` |
| `sciml-and-diffeq` | MTK v9 constraint table, DiffEq solver matrix |
| `bayesian-inference` | NumPyro diagnostics reference (R-hat, ESS, BFMI tables) |
| `julia-language` | 5-env split table (to `--mode standard`); Manifest.toml rules (to `--mode deep`) |
| `research-practice` | PRISMA/GRADE checklists |

### 2b. Routing Tree Format Compression

Replace verbose prose routing blocks with compact decision matrices.

Format:
```
KEYWORD/KEYWORD/KEYWORD    → agent-name (model)
```

Example:
```
JAX/vmap/pmap/VJP/GPU          → jax-pro (opus)
Julia/DiffEq/SciML/UDE/MTK     → julia-pro (opus)
bifurcation/chaos/Lyapunov     → nonlinear-dynamics-expert
PINN/NeuralPDE/BPINN/physics-loss → pinn-engineer
MD/GROMACS/OpenMM/Monte-Carlo  → simulation-expert
```

Target: ~40% line reduction per routing tree across 14 science-suite hub skills.

### 2c. Prompt Caching Hook

New `PreCompact` hook in agent-core pins the 4 most frequently loaded hub skills into prompt cache before compaction fires.

- Hook file: `plugins/agent-core/hooks/pre_compact_cache_pin.py`
- Hook event: `PreCompact`
- Pinned skills: `agent-systems`, `jax-computing`, `julia-language`, `simulation-and-hpc`
- Registration: add entry to `plugins/agent-core/hooks/hooks.json`

### 2d. Context Pruning — `.claudeignore`

Create `.claudeignore` at repo root:
```
graphify-out/
docs/_build/
**/__pycache__/
**/*.pyc
tools/tests/__pycache__/
plugins/*/skills/**/*.json
```

### 2e. Structured Output Constraints

Add `## Output Format` section to all 14 science-suite hub SKILL.md files:
- Return diffs, not full rewrites, when modifying existing code
- Cap explanation prose at 3 sentences before switching to code
- Use structured headers (`### Step N`) for multi-step derivations

---

## Section 3: New Commands

### Science-Suite Commands (2 new)

**`/md-sim`** (`plugins/science-suite/commands/md-sim.md`)
- Purpose: Set up and run a molecular dynamics simulation — topology prep, force field selection, equilibration protocol, trajectory analysis
- Argument hint: `[--engine gromacs|openmm|jax-md] [--system path/to/pdb] [--steps N]`
- Routes to: `simulation-expert` via `science-suite:simulation-and-hpc` hub
- Token strategy: Engine-specific reference sections gated to `--mode deep`
- Allowed tools: `Read`, `Write`, `Bash`, `Edit`, `Glob`

**`/benchmark`** (`plugins/science-suite/commands/benchmark.md`)
- Purpose: Profile JAX/Julia/HPC code — wall time, memory, GPU utilization, JIT overhead — and suggest optimizations
- Argument hint: `[--target path/to/script] [--backend jax|julia|cuda] [--profile memory|time|both]`
- Routes to: `jax-pro` (JAX/CUDA), `julia-pro` (Julia), `systems-engineer` (C/Fortran HPC)
- Token strategy: Profiling templates loaded on `--profile` flag only
- Allowed tools: `Read`, `Bash`, `Glob`

### Research-Suite Commands (3 new)

**`/paper-implement`** (`plugins/research-suite/commands/paper-implement.md`)
- Purpose: Reproduce a paper's core methods in JAX or Julia — parse equations, scaffold implementation, wire up experiment
- Argument hint: `[--paper path/to/pdf|arxiv-id] [--framework jax|julia] [--section methods|experiments|all]`
- Routes to: `research-expert` → `jax-pro` / `julia-pro` via cross-suite delegation
- Token strategy: Section-gated loading
- Allowed tools: `Read`, `Write`, `Edit`, `Bash`, `WebFetch`

**`/lit-review`** (`plugins/research-suite/commands/lit-review.md`)
- Purpose: Structured literature review — topic scan, claim extraction, evidence synthesis, gap identification
- Argument hint: `[--topic "query"] [--scope narrow|broad] [--output summary|table|annotated-bib]`
- Routes to: `research-expert` via `research-suite:research-practice` hub
- Token strategy: PRISMA/GRADE templates gated to `--mode deep`
- Allowed tools: `Read`, `Write`, `WebSearch`, `WebFetch`

**`/replicate`** (`plugins/research-suite/commands/replicate.md`)
- Purpose: End-to-end replication pipeline — fetch paper, extract claims, implement in JAX/Julia, validate against reported numbers
- Argument hint: `[--paper arxiv-id|doi] [--tolerance 0.01] [--framework jax|julia]`
- Routes to: `research-expert` (claim extraction) → `research-spark-orchestrator` (pipeline) → `jax-pro`/`julia-pro`
- Token strategy: Claim extraction and implementation are separate turns
- Allowed tools: `Read`, `Write`, `Edit`, `Bash`, `WebFetch`, `WebSearch`

---

## Section 4: v2.1.128 Compliance Audit

### Audit Sequence

```bash
python3 tools/validation/metadata_validator.py plugins/agent-core
python3 tools/validation/metadata_validator.py plugins/dev-suite
python3 tools/validation/metadata_validator.py plugins/research-suite
python3 tools/validation/metadata_validator.py plugins/science-suite
python3 tools/validation/context_budget_checker.py
python3 tools/validation/skill_validator.py
```

### Verification Checklist

- [ ] All 25 agent descriptions ≤ `skillListingMaxDescChars` after trimming
- [ ] All 24 hook handlers map to supported events (no deprecated events)
- [ ] `$schema` field present in all 4 `plugin.json` files, pointing to v2.1.128 spec
- [ ] 5 new command `.md` files pass `metadata_validator.py`
- [ ] `pinn-engineer` and `sci-workflow-engineer` referenced in ≥1 hub routing tree each
- [ ] All sub-skills reachable (no orphans introduced by agent renames)
- [ ] Version string identical across all 4 `plugin.json` files (bump to 3.5.0)

### Version Bump

All 4 `plugin.json` files: `"version": "3.4.1"` → `"version": "3.5.0"` (minor bump: new agents + commands + hooks).

---

## Section 5: Implementation Coordination

### Workstream Ownership

| # | Worker | Files Owned | Task |
|---|---|---|---|
| WS-1 | Gemini | All 217 SKILL.md (read-only) | Compression targets, redundant routing, orphan sub-skills, llm-and-ai / llm-engineering overlap |
| WS-2 | Codex | `plugins/science-suite/agents/*.md` | Model tiers + description trimming + pinn-engineer + sci-workflow-engineer rewrites |
| WS-3 | Codex | `plugins/science-suite/commands/`, `plugins/research-suite/commands/` | 5 new command `.md` files |
| WS-4 | Claude Code | `plugins/*/plugin.json`, `plugins/*/hooks/`, `.claudeignore` | Compliance audit + schema patches + manifest updates + PreCompact hook + .claudeignore |

### Merge Sequence

```
WS-1 (Gemini)   ──────────────────────────────────────┐
WS-2 (Codex)    ─────────────────────────────────────┐ │
WS-3 (Codex)    ─────────────────────────────────────┤ │ → merge
WS-4 (CC)       ─────────────────────────────────────┘ │   → apply Gemini WS-1 findings to SKILL.md
                                                         │   → uv run pytest (188 tests)
                                                         │   → context_budget_checker.py
                                                         └   → Gemini final cross-suite review
```

### Gemini WS-1 Questions

Gemini reads all 217 SKILL.md files across `plugins/science-suite/skills/` and `plugins/research-suite/skills/`. Output is a ranked list of edit targets applied in the second phase (after WS-2/3/4 merge) to those same directories.

1. Which hub routing trees have redundant blocks duplicated across multiple skills?
2. Which sub-skills exceed 80% of the 2% context budget (gating candidates)?
3. Which sub-skills have ≤1 inbound routing reference (orphan candidates)?
4. Where do `llm-and-ai` (science-suite) and `llm-engineering` (agent-core) overlap — what to deduplicate?

### Final Gate

```bash
uv run pytest tools/tests/ -v
python3 tools/validation/context_budget_checker.py
python3 tools/validation/metadata_validator.py plugins/agent-core
python3 tools/validation/metadata_validator.py plugins/dev-suite
python3 tools/validation/metadata_validator.py plugins/research-suite
python3 tools/validation/metadata_validator.py plugins/science-suite
```

All 188 tests pass + zero validator errors = merge gate cleared.

---

## Out of Scope

- dev-suite: no changes (general SDLC coverage intentional; scientific computing is handled by science-suite agents)
- agent-core: only the new PreCompact hook added; no agent or skill changes
- research-suite agents: both already on opus, no model tier changes needed
- New hub skills: none added in this redesign (existing routing covers new agents)
