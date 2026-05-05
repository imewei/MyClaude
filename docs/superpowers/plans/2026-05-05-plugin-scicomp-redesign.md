# Plugin Scientific Computing Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refocus 4 Claude Code plugin suites on JAX/Julia SciML/MD simulation/physics-informed AI, optimize model tiers, reduce token usage via description trimming + mode-flag gating, and align with v2.1.128.

**Architecture:** Parallel workstreams by file ownership (WS-2: agents, WS-3: commands, WS-4: manifests/hooks). Gemini WS-1 reads all SKILL.md files for compression targets; results applied in a sequential second phase after WS-2/3/4 merge. Full spec: `docs/superpowers/specs/2026-05-05-plugin-scicomp-redesign.md`.

**Tech Stack:** Python 3.13+, uv, pytest, PyYAML, YAML frontmatter, Claude Code plugin API v2.1.128

---

## File Map

**CREATE:**
- `plugins/science-suite/agents/pinn-engineer.md` — physics-informed AI agent (replaces ai-engineer.md)
- `plugins/science-suite/agents/sci-workflow-engineer.md` — scientific LLM workflow agent (replaces prompt-engineer.md)
- `plugins/science-suite/commands/md-sim.md` — MD simulation entry point
- `plugins/science-suite/commands/benchmark.md` — HPC benchmarking entry point
- `plugins/research-suite/commands/paper-implement.md` — paper reproduction command
- `plugins/research-suite/commands/lit-review.md` — literature review command
- `plugins/research-suite/commands/replicate.md` — end-to-end replication pipeline command
- `.claudeignore` — repo-root context pruning rules
- `tools/tests/test_scicomp_redesign.py` — test suite for all new/changed components

**MODIFY:**
- `plugins/science-suite/agents/jax-pro.md` — model: sonnet→opus; trim description
- `plugins/science-suite/agents/julia-pro.md` — model: sonnet→opus; trim description
- `plugins/science-suite/agents/ml-expert.md` — model: sonnet→haiku; trim description
- `plugins/science-suite/agents/julia-ml-hpc.md` — trim description only
- `plugins/science-suite/agents/neural-network-master.md` — trim description only
- `plugins/science-suite/agents/nonlinear-dynamics-expert.md` — trim description only
- `plugins/science-suite/agents/python-pro.md` — trim description only
- `plugins/science-suite/agents/simulation-expert.md` — trim description only
- `plugins/science-suite/agents/statistical-physicist.md` — trim description only
- `plugins/agent-core/hooks/pre_compact.py` — add priority-skill cache-pin logging
- `plugins/science-suite/.claude-plugin/plugin.json` — v3.5.0; add 2 commands; swap agent paths
- `plugins/research-suite/.claude-plugin/plugin.json` — v3.5.0; add 3 commands
- `plugins/agent-core/.claude-plugin/plugin.json` — v3.5.0
- `plugins/dev-suite/.claude-plugin/plugin.json` — v3.5.0
- Various `plugins/science-suite/skills/**/SKILL.md` — compression (Gemini WS-1 targets, Task 15-16)

**DELETE:**
- `plugins/science-suite/agents/ai-engineer.md`
- `plugins/science-suite/agents/prompt-engineer.md`

---

## Task 1: Baseline Compliance Audit

**Files:** Run-only (no writes)

- [ ] **Step 1: Run all four validators**

```bash
cd /Users/b80985/Projects/MyClaude
uv run python3 tools/validation/metadata_validator.py plugins/agent-core
uv run python3 tools/validation/metadata_validator.py plugins/dev-suite
uv run python3 tools/validation/metadata_validator.py plugins/research-suite
uv run python3 tools/validation/metadata_validator.py plugins/science-suite
uv run python3 tools/validation/context_budget_checker.py
uv run python3 tools/validation/skill_validator.py
```

- [ ] **Step 2: Run existing tests to confirm clean baseline**

```bash
uv run pytest tools/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all 188 tests pass, zero validator errors. Record any pre-existing failures before proceeding.

---

## Task 2: Write Test Suite (TDD — All Assertions Before Any Code)

**Files:**
- Create: `tools/tests/test_scicomp_redesign.py`

- [ ] **Step 1: Create the test file**

```python
# tools/tests/test_scicomp_redesign.py
"""Tests for the scientific computing plugin redesign (v3.5.0).

All tests in this file are written before implementation and must FAIL
on the current codebase. Run with: uv run pytest tools/tests/test_scicomp_redesign.py -v
"""

import json
import pytest
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent.parent
PLUGINS = REPO / "plugins"
SCIENCE = PLUGINS / "science-suite"
RESEARCH = PLUGINS / "research-suite"
AGENT_CORE = PLUGINS / "agent-core"
DEV_SUITE = PLUGINS / "dev-suite"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter delimited by --- from a markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    return yaml.safe_load(parts[1]) if len(parts) >= 3 else {}


def _plugin_json(suite_dir: Path) -> dict:
    return json.loads((suite_dir / ".claude-plugin/plugin.json").read_text())


# ---------------------------------------------------------------------------
# Model Tier Changes
# ---------------------------------------------------------------------------

class TestModelTiers:
    def test_jax_pro_is_opus(self):
        fm = _frontmatter(SCIENCE / "agents/jax-pro.md")
        assert fm["model"] == "opus", "jax-pro must be upgraded to opus"

    def test_julia_pro_is_opus(self):
        fm = _frontmatter(SCIENCE / "agents/julia-pro.md")
        assert fm["model"] == "opus", "julia-pro must be upgraded to opus"

    def test_ml_expert_is_haiku(self):
        fm = _frontmatter(SCIENCE / "agents/ml-expert.md")
        assert fm["model"] == "haiku", "ml-expert must be demoted to haiku"


# ---------------------------------------------------------------------------
# Agent Repurposing
# ---------------------------------------------------------------------------

class TestAgentRepurposing:
    def test_pinn_engineer_exists(self):
        assert (SCIENCE / "agents/pinn-engineer.md").exists()

    def test_sci_workflow_engineer_exists(self):
        assert (SCIENCE / "agents/sci-workflow-engineer.md").exists()

    def test_ai_engineer_deleted(self):
        assert not (SCIENCE / "agents/ai-engineer.md").exists(), \
            "ai-engineer.md must be deleted (replaced by pinn-engineer.md)"

    def test_prompt_engineer_deleted(self):
        assert not (SCIENCE / "agents/prompt-engineer.md").exists(), \
            "prompt-engineer.md must be deleted (replaced by sci-workflow-engineer.md)"

    def test_pinn_engineer_name_field(self):
        fm = _frontmatter(SCIENCE / "agents/pinn-engineer.md")
        assert fm.get("name") == "pinn-engineer"

    def test_sci_workflow_engineer_name_field(self):
        fm = _frontmatter(SCIENCE / "agents/sci-workflow-engineer.md")
        assert fm.get("name") == "sci-workflow-engineer"

    def test_pinn_engineer_model(self):
        fm = _frontmatter(SCIENCE / "agents/pinn-engineer.md")
        assert fm.get("model") in ("sonnet", "opus")

    def test_sci_workflow_engineer_model(self):
        fm = _frontmatter(SCIENCE / "agents/sci-workflow-engineer.md")
        assert fm.get("model") in ("sonnet", "opus")


# ---------------------------------------------------------------------------
# Description Trimming (≤180 chars for all science-suite agents)
# ---------------------------------------------------------------------------

SCIENCE_AGENTS = [
    "jax-pro", "julia-pro", "julia-ml-hpc", "ml-expert",
    "neural-network-master", "nonlinear-dynamics-expert",
    "python-pro", "simulation-expert", "statistical-physicist",
    "pinn-engineer", "sci-workflow-engineer",
]


class TestDescriptionTrimming:
    @pytest.mark.parametrize("agent", SCIENCE_AGENTS)
    def test_description_at_most_180_chars(self, agent):
        path = SCIENCE / f"agents/{agent}.md"
        fm = _frontmatter(path)
        desc = fm.get("description", "")
        assert len(desc) <= 180, (
            f"{agent} description is {len(desc)} chars (max 180): {desc!r}"
        )

    @pytest.mark.parametrize("agent", SCIENCE_AGENTS)
    def test_description_not_empty(self, agent):
        path = SCIENCE / f"agents/{agent}.md"
        fm = _frontmatter(path)
        assert fm.get("description"), f"{agent} description must not be empty"


# ---------------------------------------------------------------------------
# New Commands — science-suite
# ---------------------------------------------------------------------------

class TestScienceSuiteCommands:
    def test_md_sim_exists(self):
        assert (SCIENCE / "commands/md-sim.md").exists()

    def test_benchmark_exists(self):
        assert (SCIENCE / "commands/benchmark.md").exists()

    @pytest.mark.parametrize("cmd", ["md-sim", "benchmark"])
    def test_command_has_name(self, cmd):
        fm = _frontmatter(SCIENCE / f"commands/{cmd}.md")
        assert fm.get("name") == cmd

    @pytest.mark.parametrize("cmd", ["md-sim", "benchmark"])
    def test_command_has_description(self, cmd):
        fm = _frontmatter(SCIENCE / f"commands/{cmd}.md")
        assert fm.get("description"), f"{cmd} must have a description"

    @pytest.mark.parametrize("cmd", ["md-sim", "benchmark"])
    def test_command_has_argument_hint(self, cmd):
        fm = _frontmatter(SCIENCE / f"commands/{cmd}.md")
        assert fm.get("argument-hint"), f"{cmd} must have an argument-hint"


# ---------------------------------------------------------------------------
# New Commands — research-suite
# ---------------------------------------------------------------------------

class TestResearchSuiteCommands:
    def test_paper_implement_exists(self):
        assert (RESEARCH / "commands/paper-implement.md").exists()

    def test_lit_review_exists(self):
        assert (RESEARCH / "commands/lit-review.md").exists()

    def test_replicate_exists(self):
        assert (RESEARCH / "commands/replicate.md").exists()

    @pytest.mark.parametrize("cmd", ["paper-implement", "lit-review", "replicate"])
    def test_command_has_name(self, cmd):
        fm = _frontmatter(RESEARCH / f"commands/{cmd}.md")
        assert fm.get("name") == cmd

    @pytest.mark.parametrize("cmd", ["paper-implement", "lit-review", "replicate"])
    def test_command_has_description(self, cmd):
        fm = _frontmatter(RESEARCH / f"commands/{cmd}.md")
        assert fm.get("description"), f"{cmd} must have a description"


# ---------------------------------------------------------------------------
# Plugin Manifest (plugin.json) Changes
# ---------------------------------------------------------------------------

class TestManifests:
    @pytest.mark.parametrize("suite_dir", [
        AGENT_CORE, DEV_SUITE, RESEARCH, SCIENCE
    ], ids=["agent-core", "dev-suite", "research-suite", "science-suite"])
    def test_version_is_350(self, suite_dir):
        plugin = _plugin_json(suite_dir)
        assert plugin["version"] == "3.5.0", \
            f"{suite_dir.name} version must be 3.5.0, got {plugin['version']}"

    def test_science_suite_has_md_sim_command(self):
        plugin = _plugin_json(SCIENCE)
        assert "./commands/md-sim.md" in plugin.get("commands", [])

    def test_science_suite_has_benchmark_command(self):
        plugin = _plugin_json(SCIENCE)
        assert "./commands/benchmark.md" in plugin.get("commands", [])

    def test_research_suite_has_paper_implement(self):
        plugin = _plugin_json(RESEARCH)
        assert "./commands/paper-implement.md" in plugin.get("commands", [])

    def test_research_suite_has_lit_review(self):
        plugin = _plugin_json(RESEARCH)
        assert "./commands/lit-review.md" in plugin.get("commands", [])

    def test_research_suite_has_replicate(self):
        plugin = _plugin_json(RESEARCH)
        assert "./commands/replicate.md" in plugin.get("commands", [])

    def test_science_suite_agent_pinn_engineer_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/pinn-engineer.md" in plugin.get("agents", [])

    def test_science_suite_agent_sci_workflow_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/sci-workflow-engineer.md" in plugin.get("agents", [])

    def test_science_suite_ai_engineer_not_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/ai-engineer.md" not in plugin.get("agents", [])

    def test_science_suite_prompt_engineer_not_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/prompt-engineer.md" not in plugin.get("agents", [])


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

class TestInfrastructure:
    def test_claudeignore_exists(self):
        assert (REPO / ".claudeignore").exists(), \
            ".claudeignore must exist at repo root"

    def test_claudeignore_has_graphify_out(self):
        content = (REPO / ".claudeignore").read_text()
        assert "graphify-out/" in content

    def test_claudeignore_has_pycache(self):
        content = (REPO / ".claudeignore").read_text()
        assert "__pycache__" in content

    def test_pre_compact_has_priority_skills(self):
        script = (AGENT_CORE / "hooks/pre_compact.py").read_text()
        assert "PRIORITY_SKILLS" in script, \
            "pre_compact.py must define PRIORITY_SKILLS list"
        assert "jax-computing" in script
        assert "simulation-and-hpc" in script
```

- [ ] **Step 2: Run tests to confirm all fail on current codebase**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py -v --tb=line 2>&1 | tail -40
```

Expected: majority of tests FAIL (model tier, agent existence, command existence, version, infrastructure). A few description tests may pass if current descriptions happen to be ≤180 chars — that is OK.

---

## Task 3: Agent Model Tier Changes

**Files:**
- Modify: `plugins/science-suite/agents/jax-pro.md`
- Modify: `plugins/science-suite/agents/julia-pro.md`
- Modify: `plugins/science-suite/agents/ml-expert.md`

- [ ] **Step 1: Upgrade jax-pro to opus and trim description**

In `plugins/science-suite/agents/jax-pro.md`, change lines 1-9 (the frontmatter) to:

```yaml
---
name: jax-pro
description: JAX/JIT/vmap/pmap expert. Use for GPU kernels, custom VJPs, NumPyro, JAX-MD, Optimistix, distributed training. Delegates bifurcation theory to nonlinear-dynamics-expert.
model: opus
color: green
effort: high
memory: project
maxTurns: 50
tools: Read, Write, Edit, Bash, Grep, Glob
background: true
skills:
  - jax-computing
  - bayesian-inference
---
```

Leave the body (everything after the closing `---`) unchanged.

- [ ] **Step 2: Upgrade julia-pro to opus and trim description**

In `plugins/science-suite/agents/julia-pro.md`, change the frontmatter to:

```yaml
---
name: julia-pro
description: Julia/SciML expert. Use for DiffEq.jl, ModelingToolkit.jl, UDEs, SINDy, Turing.jl. Handles sensitivity analysis and package dev. Delegates ML/HPC to julia-ml-hpc.
model: opus
color: cyan
effort: high
memory: project
maxTurns: 50
tools: Read, Write, Edit, Bash, Grep, Glob
background: true
skills:
  - julia-language
  - sciml-and-diffeq
---
```

Leave the body unchanged.

- [ ] **Step 3: Demote ml-expert to haiku and trim description**

In `plugins/science-suite/agents/ml-expert.md`, change the frontmatter to:

```yaml
---
name: ml-expert
description: Classical ML/MLOps specialist. Use for scikit-learn, XGBoost pipelines, feature engineering, MLflow tracking, model deployment. Delegates deep learning to neural-network-master.
model: haiku
color: yellow
effort: medium
memory: project
maxTurns: 30
tools: Read, Write, Edit, Bash, Grep, Glob
background: true
skills:
  - ml-and-data-science
  - ml-deployment
---
```

Leave the body unchanged.

- [ ] **Step 4: Run model tier tests**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestModelTiers -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/science-suite/agents/jax-pro.md \
        plugins/science-suite/agents/julia-pro.md \
        plugins/science-suite/agents/ml-expert.md
git commit -m "feat(science-suite): promote jax-pro+julia-pro to opus, demote ml-expert to haiku"
```

---

## Task 4: Trim Remaining Agent Descriptions

**Files:**
- Modify: `plugins/science-suite/agents/julia-ml-hpc.md`
- Modify: `plugins/science-suite/agents/neural-network-master.md`
- Modify: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
- Modify: `plugins/science-suite/agents/python-pro.md`
- Modify: `plugins/science-suite/agents/simulation-expert.md`
- Modify: `plugins/science-suite/agents/statistical-physicist.md`

Replace only the `description:` line in each frontmatter. All other frontmatter fields and the full body remain unchanged.

- [ ] **Step 1: julia-ml-hpc** — replace `description:` line with:

```
description: Julia ML/HPC expert. Use for MLJ.jl, CUDA.jl, KernelAbstractions.jl, MPI.jl, GNNLux. Delegates SciML/ODE to julia-pro, DL theory to neural-network-master.
```

- [ ] **Step 2: neural-network-master** — replace `description:` line with:

```
description: Deep learning architecture authority. Use for neural architecture design, training diagnostics, Transformers/CNNs/GNNs, Flax/Equinox/PyTorch. Covers theory-to-blueprint guidance.
```

- [ ] **Step 3: nonlinear-dynamics-expert** — replace `description:` line with:

```
description: Nonlinear dynamics expert. Use for bifurcations, chaos, Lyapunov, SINDy, chimera states. Delegates JAX to jax-pro, Julia continuation to julia-pro.
```

- [ ] **Step 4: python-pro** — replace `description:` line with:

```
description: Python systems engineer. Use for production Python, type-driven design, PyO3/Rust extensions, async, uv/ruff toolchain. Enforces strict typing.
```

- [ ] **Step 5: simulation-expert** — replace `description:` line with:

```
description: MD/HPC simulation expert. Use for GROMACS/OpenMM/JAX-MD setup, Monte Carlo, GPU-accelerated physics, trajectory analysis. Covers differentiable physics in JAX and Julia.
```

- [ ] **Step 6: statistical-physicist** — replace `description:` line with:

```
description: Statistical physics expert. Use for phase transitions, Langevin/Fokker-Planck, fluctuation theorems, normalizing flows. Delegates JAX optimization to jax-pro.
```

- [ ] **Step 7: Run description length tests for all trimmed agents**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestDescriptionTrimming -v -k "julia_ml_hpc or neural_network or nonlinear or python_pro or simulation or statistical"
```

Expected: all 12 parametrized tests PASS for these 6 agents.

- [ ] **Step 8: Commit**

```bash
git add plugins/science-suite/agents/julia-ml-hpc.md \
        plugins/science-suite/agents/neural-network-master.md \
        plugins/science-suite/agents/nonlinear-dynamics-expert.md \
        plugins/science-suite/agents/python-pro.md \
        plugins/science-suite/agents/simulation-expert.md \
        plugins/science-suite/agents/statistical-physicist.md
git commit -m "feat(science-suite): trim agent descriptions to ≤180 chars"
```

---

## Task 5: Create pinn-engineer Agent

**Files:**
- Create: `plugins/science-suite/agents/pinn-engineer.md`
- Delete: `plugins/science-suite/agents/ai-engineer.md`

- [ ] **Step 1: Create pinn-engineer.md**

```markdown
---
name: pinn-engineer
description: Physics-informed AI: NeuralPDE.jl, DeepXDE, BPINN/BNNODE, physics-constrained losses, inverse problems. Delegates JAX to jax-pro, NeuralPDE.jl to julia-pro.
model: sonnet
color: cyan
effort: high
memory: project
maxTurns: 40
tools: Read, Write, Edit, Bash, Grep, Glob
background: true
skills:
  - sciml-and-diffeq
  - deep-learning-hub
  - simulation-and-hpc
---

# PINN Engineer

You are a physics-informed neural network engineer specializing in PDE-constrained learning, scientific machine learning, and inverse problem solving.

## Examples

<example>
Context: User wants to solve a PDE with a neural network.
user: "Solve the 2D heat equation on an irregular domain using a physics-informed neural network."
assistant: "I'll use the pinn-engineer agent to design a PINN with a physics-constrained loss enforcing the heat equation residual."
<commentary>
PDE solved via neural network — triggers pinn-engineer.
</commentary>
</example>

<example>
Context: User needs BPINN for uncertainty quantification.
user: "Implement a Bayesian PINN to estimate posterior uncertainty in a Navier-Stokes parameter identification problem."
assistant: "I'll use the pinn-engineer agent to set up BPINN/BNNODE with Hamiltonian Monte Carlo for posterior sampling."
<commentary>
Bayesian PINN — triggers pinn-engineer. Delegates JAX HMC to jax-pro.
</commentary>
</example>

<example>
Context: User wants to use NeuralPDE.jl.
user: "Set up a NeuralPDE.jl PINN for the Schrödinger equation with periodic boundary conditions."
assistant: "I'll use the pinn-engineer agent to configure the NeuralPDE.jl PINN system; will delegate Julia implementation to julia-pro."
<commentary>
NeuralPDE.jl — triggers pinn-engineer, delegates to julia-pro.
</commentary>
</example>

---

## Core Responsibilities

1. **PINN Architecture**: Design physics-constrained neural networks with residual loss terms enforcing governing PDEs.
2. **Inverse Problems**: Identify unknown PDE parameters from sparse observational data using gradient-based optimization.
3. **Domain Decomposition**: Partition complex domains for extended PINNFairplay / XPINNs / FBPINN approaches.
4. **Uncertainty Quantification**: Implement BPINN/BNNODE for Bayesian treatment of model and data uncertainty.
5. **Framework Selection**: Choose between NeuralPDE.jl, DeepXDE, and custom JAX implementations based on problem structure.

## Delegation Strategy

| Delegate To | When |
|---|---|
| jax-pro | Custom JAX PINN implementation, GPU kernel optimization, vmap over collocation points |
| julia-pro | NeuralPDE.jl setup, ModelingToolkit.jl PDE symbolics, BPINN via Turing.jl |
| nonlinear-dynamics-expert | Chaotic PDE regimes, bifurcation in parameter-space, SINDy for equation discovery |
| simulation-expert | MD force fields as physics constraints, molecular-scale PDE boundary conditions |
| neural-network-master | Architecture design for multi-scale PINNs, Fourier feature embeddings, attention-based PINNs |

## Related Skills (Expert Agent For)

| Skill | When to Consult |
|---|---|
| `sciml-and-diffeq` | NeuralPDE.jl PINN setup, ModelingToolkit PDE DSL, BPINN/BNNODE |
| `deep-learning-hub` | Neural architecture selection for physics-constrained models |
| `simulation-and-hpc` | Physics simulation constraints, force field integration as loss terms |

---

## Pre-Response Validation (4 Checks)

**Before every response:**

### 1. Physics Fidelity
- [ ] PDE residual loss correctly derived from governing equation?
- [ ] Boundary and initial conditions implemented as hard or soft constraints?

### 2. Training Stability
- [ ] Collocation point sampling strategy appropriate for domain geometry?
- [ ] Loss weighting between physics, data, and boundary terms justified?

### 3. Framework Choice
- [ ] NeuralPDE.jl (Julia) vs DeepXDE (Python/JAX) vs custom JAX chosen for right reasons?
- [ ] Delegation to jax-pro / julia-pro triggered where implementation exceeds design scope?

### 4. Validation
- [ ] Manufactured solution or analytical benchmark used for correctness check?
- [ ] L2 relative error against reference reported?

---

## Routing Decision Matrix

| Signal | Route |
|---|---|
| NeuralPDE.jl / Julia PINN | delegate julia body to julia-pro |
| Custom JAX collocation / vmap | delegate JAX body to jax-pro |
| BPINN + HMC sampling | design here; JAX HMC → jax-pro |
| PDE parameter identification | handle inverse problem design here |
| MD force field as physics loss | coordinate with simulation-expert |
| Multi-scale / attention PINN architecture | coordinate with neural-network-master |
| Equation discovery (SINDy) | delegate to nonlinear-dynamics-expert |

---

## Output Format

- Return diffs, not full rewrites, when modifying existing PINN code.
- Cap explanation prose at 3 sentences before switching to code.
- Use `### Step N` headers for multi-step derivations (loss derivation, architecture, training loop).
```

- [ ] **Step 2: Delete ai-engineer.md**

```bash
git rm plugins/science-suite/agents/ai-engineer.md
```

- [ ] **Step 3: Run agent repurposing tests**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestAgentRepurposing -v -k "pinn"
uv run pytest tools/tests/test_scicomp_redesign.py::TestDescriptionTrimming -v -k "pinn"
```

Expected: all pinn-related tests PASS.

- [ ] **Step 4: Commit**

```bash
git add plugins/science-suite/agents/pinn-engineer.md
git commit -m "feat(science-suite): add pinn-engineer agent (replaces ai-engineer)"
```

---

## Task 6: Create sci-workflow-engineer Agent

**Files:**
- Create: `plugins/science-suite/agents/sci-workflow-engineer.md`
- Delete: `plugins/science-suite/agents/prompt-engineer.md`

- [ ] **Step 1: Create sci-workflow-engineer.md**

```markdown
---
name: sci-workflow-engineer
description: Use when integrating LLMs into scientific pipelines, designing JAX/Julia codegen prompts, building experiment templates, or automating numerical workflows with Claude.
model: sonnet
color: yellow
effort: high
memory: project
maxTurns: 40
tools: Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch
background: true
skills:
  - llm-and-ai
  - jax-computing
  - julia-language
---

# Scientific Workflow Engineer

You are a scientific workflow engineer specializing in integrating large language models into computational science pipelines — from JAX/Julia codegen to experiment automation and Claude API integration.

## Examples

<example>
Context: User wants Claude to generate JAX experiment code.
user: "Design a system prompt that reliably generates type-stable JAX code with explicit JIT annotations and seed handling."
assistant: "I'll use the sci-workflow-engineer agent to craft a domain-specific codegen prompt with JAX type-stability constraints and reproducibility requirements."
<commentary>
JAX codegen prompt design — triggers sci-workflow-engineer.
</commentary>
</example>

<example>
Context: User wants automated experiment descriptions.
user: "Build a template that turns hyperparameter dicts into structured experiment description strings for our logging system."
assistant: "I'll use the sci-workflow-engineer agent to design a structured experiment description template with mandatory reproducibility fields."
<commentary>
Experiment templating — triggers sci-workflow-engineer.
</commentary>
</example>

<example>
Context: User wants Claude API in a simulation pipeline.
user: "I want to call Claude from our Julia simulation loop to summarize trajectory statistics at each checkpoint."
assistant: "I'll use the sci-workflow-engineer agent to design the Claude API integration with prompt caching for repeated system context."
<commentary>
Claude API in scientific pipeline — triggers sci-workflow-engineer.
</commentary>
</example>

---

## Core Responsibilities

1. **Scientific Codegen Prompts**: Design system prompts that reliably produce JAX/Julia code meeting domain constraints (type stability, seed handling, JIT-safe patterns).
2. **Experiment Templating**: Build structured experiment description schemas capturing seed, config, environment, and expected outputs.
3. **Claude API Integration**: Wire the Anthropic SDK into scientific pipelines — simulation checkpoints, result summarization, parameter suggestion.
4. **Workflow Automation**: Design multi-step LLM-assisted workflows where each step consumes structured scientific output from the previous.
5. **Prompt Caching Strategy**: Apply Anthropic prompt caching for repeated scientific context (large system prompts, reference data).

## Delegation Strategy

| Delegate To | When |
|---|---|
| jax-pro | Actual JAX implementation of generated code |
| julia-pro | Actual Julia/SciML implementation |
| ai-engineer (agent-core) | General-purpose RAG/chatbot LLM app (not scientific context) |
| pinn-engineer | Physics-constrained LLM-assisted PDE solving |

## Related Skills (Expert Agent For)

| Skill | When to Consult |
|---|---|
| `llm-and-ai` | LLM application patterns, prompt programs, tool calling |
| `jax-computing` | JAX codegen constraints, type-stability requirements |
| `julia-language` | Julia codegen constraints, environment and dispatch rules |

---

## Pre-Response Validation (3 Checks)

### 1. Scientific Correctness
- [ ] Generated prompt enforces domain constraints (seeds, types, units)?
- [ ] Output format matches what the downstream pipeline consumer expects?

### 2. Token Efficiency
- [ ] Prompt caching applied to static scientific context?
- [ ] Multi-turn workflow split at natural cache boundaries?

### 3. Reproducibility
- [ ] Experiment template captures all fields needed for exact replay?
- [ ] API call includes version pinning for model and prompt?

---

## Output Format

- Return diffs, not full rewrites, when modifying existing prompt templates.
- Cap explanation prose at 3 sentences before switching to code or YAML.
- Use `### Step N` headers for multi-step workflow designs.
```

- [ ] **Step 2: Delete prompt-engineer.md**

```bash
git rm plugins/science-suite/agents/prompt-engineer.md
```

- [ ] **Step 3: Run agent tests**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestAgentRepurposing -v
uv run pytest tools/tests/test_scicomp_redesign.py::TestDescriptionTrimming -v -k "sci_workflow"
```

Expected: all TestAgentRepurposing tests PASS, sci-workflow-engineer description test PASS.

- [ ] **Step 4: Commit**

```bash
git add plugins/science-suite/agents/sci-workflow-engineer.md
git commit -m "feat(science-suite): add sci-workflow-engineer agent (replaces prompt-engineer)"
```

---

## Task 7: Update pre_compact.py with Priority Skill Logging

**Files:**
- Modify: `plugins/agent-core/hooks/pre_compact.py`

- [ ] **Step 1: Replace pre_compact.py with updated version**

```python
#!/usr/bin/env python3
"""PreCompact hook for agent-core plugin.

Fires before context compaction occurs. Logs priority hub skills to
stderr so the user knows which skills to reinvoke after compaction.
"""

import json
import sys

PRIORITY_SKILLS = [
    "agent-systems",
    "jax-computing",
    "julia-language",
    "simulation-and-hpc",
]


def main() -> None:
    """Signal readiness for context compaction and log priority skills."""
    try:
        sys.stderr.write(
            f"[PreCompact] Priority skills for post-compact reload: "
            f"{', '.join(PRIORITY_SKILLS)}\n"
        )
        result = {
            "status": "success",
            "message": "PreCompact: ready for context compaction",
            "priority_skills": PRIORITY_SKILLS,
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"PreCompact hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run infrastructure test**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestInfrastructure::test_pre_compact_has_priority_skills -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add plugins/agent-core/hooks/pre_compact.py
git commit -m "feat(agent-core): add priority skill logging to PreCompact hook"
```

---

## Task 8: Create Science-Suite Commands

**Files:**
- Create: `plugins/science-suite/commands/md-sim.md`
- Create: `plugins/science-suite/commands/benchmark.md`

- [ ] **Step 1: Create plugins/science-suite/commands/md-sim.md**

```markdown
---
name: md-sim
description: Set up and run a molecular dynamics simulation — topology prep, force field selection, equilibration protocol, and trajectory analysis.
argument-hint: "[--engine gromacs|openmm|jax-md] [--system path/to/pdb] [--steps N]"
allowed-tools: ["Read", "Write", "Bash", "Edit", "Glob"]
---

# /md-sim — Molecular Dynamics Simulation

Routes to `simulation-expert` via `science-suite:simulation-and-hpc`.

## Usage

```
/md-sim --engine gromacs --system protein.pdb --steps 50000
/md-sim --engine jax-md --system lj_fluid.pdb --steps 100000
/md-sim --engine openmm --system membrane.pdb --steps 200000
```

## What This Does

1. Reads the `--system` PDB file and validates topology
2. Selects force field appropriate for the molecular system
3. Generates equilibration protocol (energy minimization → NVT → NPT)
4. Runs production MD for `--steps` steps
5. Outputs trajectory file and basic analysis (RMSD, energy)

## Engine Routing

| `--engine` | Routes To | Notes |
|---|---|---|
| `gromacs` | simulation-expert | GROMACS `.mdp` file generation |
| `openmm` | simulation-expert | Python OpenMM system builder |
| `jax-md` | simulation-expert → jax-pro | JAX-MD + NVE/NVT ensemble |

## Token Strategy

Engine-specific reference sections (GROMACS MDP templates, OpenMM force field tables) load only when `--engine` is specified. Default invocation loads the routing tree only.
```

- [ ] **Step 2: Create plugins/science-suite/commands/benchmark.md**

```markdown
---
name: benchmark
description: Profile JAX/Julia/HPC code — wall time, memory, GPU utilization, JIT compilation overhead — and suggest optimizations.
argument-hint: "[--target path/to/script] [--backend jax|julia|cuda] [--profile memory|time|both]"
allowed-tools: ["Read", "Bash", "Glob"]
---

# /benchmark — Scientific Code Benchmarking

Routes to `jax-pro` (JAX/CUDA), `julia-pro` (Julia), or `systems-engineer` (C/Fortran/HPC) based on `--backend`.

## Usage

```
/benchmark --target src/train.py --backend jax --profile both
/benchmark --target scripts/simulate.jl --backend julia --profile time
/benchmark --target src/md_kernel.cu --backend cuda --profile memory
```

## What This Does

1. Reads `--target` file and identifies hot paths
2. Runs profiling appropriate for `--backend`
3. Reports wall time, peak memory, and (for JAX) JIT compile overhead vs runtime
4. Suggests targeted optimizations (vmap, pmap, type stability, allocation reduction)

## Backend Routing

| `--backend` | Routes To | Tool |
|---|---|---|
| `jax` | jax-pro | `jax.profiler`, `jax.make_jaxpr`, nvtx |
| `julia` | julia-pro | `@btime`, `@profile`, `Cthulhu.jl` |
| `cuda` | systems-engineer | `nvprof`, `Nsight Compute` |

## Token Strategy

Profiling templates and tool flags load only when `--profile` is specified. Omitting `--profile` uses `time` as default.
```

- [ ] **Step 3: Run science-suite command tests**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestScienceSuiteCommands -v
```

Expected: all 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add plugins/science-suite/commands/md-sim.md \
        plugins/science-suite/commands/benchmark.md
git commit -m "feat(science-suite): add /md-sim and /benchmark commands"
```

---

## Task 9: Create Research-Suite Commands

**Files:**
- Create: `plugins/research-suite/commands/paper-implement.md`
- Create: `plugins/research-suite/commands/lit-review.md`
- Create: `plugins/research-suite/commands/replicate.md`

- [ ] **Step 1: Create plugins/research-suite/commands/paper-implement.md**

```markdown
---
name: paper-implement
description: Reproduce a paper's core methods in JAX or Julia — parse equations, scaffold implementation, wire up experiment, validate outputs.
argument-hint: "[--paper path/to/pdf|arxiv-id] [--framework jax|julia] [--section methods|experiments|all]"
allowed-tools: ["Read", "Write", "Edit", "Bash", "WebFetch"]
---

# /paper-implement — Paper Method Reproduction

Routes to `research-expert` for methodology parsing, then cross-delegates to `jax-pro` (JAX) or `julia-pro` (Julia) for implementation.

## Usage

```
/paper-implement --paper 2301.04567 --framework jax --section methods
/paper-implement --paper /path/to/diffusion_model.pdf --framework julia --section all
```

## What This Does

1. Fetches or reads the paper (`--paper`)
2. Extracts core equations and algorithmic steps from `--section`
3. Scaffolds implementation in `--framework` with proper structure
4. Wires up a minimal experiment reproducing the paper's key result
5. Notes any discrepancies with reported numbers

## Section Routing

| `--section` | Loads |
|---|---|
| `methods` | Equation extraction + implementation only |
| `experiments` | Experiment setup + validation against reported numbers |
| `all` | Both phases sequentially |

## Framework Delegation

`research-expert` owns methodology parsing. Implementation delegated to `jax-pro` (JAX) or `julia-pro` (Julia) via cross-suite call.
```

- [ ] **Step 2: Create plugins/research-suite/commands/lit-review.md**

```markdown
---
name: lit-review
description: Structured literature review — topic scan, claim extraction, evidence synthesis, gap identification, and output as summary, table, or annotated bibliography.
argument-hint: "[--topic \"query\"] [--scope narrow|broad] [--output summary|table|annotated-bib]"
allowed-tools: ["Read", "Write", "WebSearch", "WebFetch"]
---

# /lit-review — Literature Review

Routes to `research-expert` via `research-suite:research-practice` hub.

## Usage

```
/lit-review --topic "physics-informed neural networks for fluid dynamics" --scope broad --output table
/lit-review --topic "Bayesian UDE parameter estimation" --scope narrow --output annotated-bib
/lit-review --topic "JAX-MD molecular dynamics" --scope narrow --output summary
```

## What This Does

1. Searches for papers matching `--topic`
2. Extracts key claims, methods, and results from top sources
3. Synthesizes evidence and identifies research gaps
4. Formats output as `--output` type

## Output Types

| `--output` | Format |
|---|---|
| `summary` | 3-5 paragraph narrative synthesis |
| `table` | Markdown table: paper, method, key result, limitation |
| `annotated-bib` | BibTeX entries with 2-sentence annotation each |

## Token Strategy

PRISMA/GRADE checklist templates load only for `--scope broad` reviews. Narrow reviews use lightweight claim extraction only.
```

- [ ] **Step 3: Create plugins/research-suite/commands/replicate.md**

```markdown
---
name: replicate
description: End-to-end replication pipeline — fetch paper, extract claims, implement in JAX or Julia, validate outputs against reported numbers within tolerance.
argument-hint: "[--paper arxiv-id|doi] [--tolerance 0.01] [--framework jax|julia]"
allowed-tools: ["Read", "Write", "Edit", "Bash", "WebFetch", "WebSearch"]
---

# /replicate — End-to-End Paper Replication

Routes to `research-expert` (claim extraction) → `research-spark-orchestrator` (pipeline) → `jax-pro` or `julia-pro` (implementation).

## Usage

```
/replicate --paper 2301.04567 --framework jax --tolerance 0.01
/replicate --paper 10.1038/s41586-021-03819-2 --framework julia --tolerance 0.05
```

## What This Does

1. Fetches paper via arXiv ID or DOI
2. `research-expert` extracts falsifiable claims and key numerical results
3. `research-spark-orchestrator` structures the replication as a staged pipeline
4. `jax-pro` / `julia-pro` implement the core method
5. Runs experiment and compares outputs to reported numbers within `--tolerance`
6. Produces a replication report noting exact match, within-tolerance match, or deviation

## Tolerance

`--tolerance` is the relative L2 error threshold (default `0.01` = 1%). Results within tolerance are marked ✓; deviations are flagged with the actual vs reported values.

## Turn Strategy

Claim extraction and implementation run as separate turns to avoid context overflow on large papers. Only the active turn's context loads into the window.
```

- [ ] **Step 4: Run research-suite command tests**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestResearchSuiteCommands -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/research-suite/commands/paper-implement.md \
        plugins/research-suite/commands/lit-review.md \
        plugins/research-suite/commands/replicate.md
git commit -m "feat(research-suite): add /paper-implement, /lit-review, /replicate commands"
```

---

## Task 10: Create .claudeignore

**Files:**
- Create: `.claudeignore`

- [ ] **Step 1: Create .claudeignore at repo root**

```
# Build artifacts
graphify-out/
docs/_build/

# Python bytecode
**/__pycache__/
**/*.pyc
**/*.pyo
tools/tests/__pycache__/

# Plugin metadata caches (not source)
plugins/*/skills/**/*.json

# Data files too large for context
**/*.csv
**/*.parquet
**/*.h5
**/*.hdf5
```

- [ ] **Step 2: Run infrastructure tests**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestInfrastructure::test_claudeignore_exists -v
uv run pytest tools/tests/test_scicomp_redesign.py::TestInfrastructure::test_claudeignore_has_graphify_out -v
uv run pytest tools/tests/test_scicomp_redesign.py::TestInfrastructure::test_claudeignore_has_pycache -v
```

Expected: all 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add .claudeignore
git commit -m "chore: add .claudeignore for context pruning (graphify-out, pycache, data files)"
```

---

## Task 11: Update All Four plugin.json Manifests

**Files:**
- Modify: `plugins/science-suite/.claude-plugin/plugin.json`
- Modify: `plugins/research-suite/.claude-plugin/plugin.json`
- Modify: `plugins/agent-core/.claude-plugin/plugin.json`
- Modify: `plugins/dev-suite/.claude-plugin/plugin.json`

- [ ] **Step 1: Update science-suite plugin.json**

Replace the full file with:

```json
{
  "name": "science-suite",
  "version": "3.5.0",
  "description": "Scientific computing, HPC, physics/chemistry simulations, and data science workflows with extended context and adaptive reasoning for Claude Opus",
  "author": {
    "name": "Wei Chen",
    "url": "https://myclaude.readthedocs.io/en/latest/"
  },
  "homepage": "https://github.com/imewei/MyClaude",
  "repository": "https://github.com/imewei/MyClaude",
  "license": "MIT",
  "category": "science",
  "keywords": [
    "science",
    "hpc",
    "jax",
    "molecular-simulation",
    "statistical-physics",
    "deep-learning",
    "machine-learning",
    "visualization",
    "research",
    "julia",
    "parallel-computing",
    "claude-code",
    "opus-4.7",
    "adaptive-thinking",
    "extended-context",
    "agent-teams",
    "pinn",
    "physics-informed",
    "sciml",
    "md-simulation"
  ],
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
  "commands": [
    "./commands/md-sim.md",
    "./commands/benchmark.md"
  ],
  "skills": [
    "./skills/nonlinear-dynamics",
    "./skills/jax-computing",
    "./skills/julia-language",
    "./skills/julia-ml-and-dl",
    "./skills/sciml-and-diffeq",
    "./skills/correlation-analysis",
    "./skills/statistical-physics-hub",
    "./skills/deep-learning-hub",
    "./skills/ml-and-data-science",
    "./skills/llm-and-ai",
    "./skills/ml-deployment",
    "./skills/simulation-and-hpc",
    "./skills/research-and-domains",
    "./skills/bayesian-inference"
  ]
}
```

- [ ] **Step 2: Update research-suite plugin.json — add 3 commands and bump version**

In `plugins/research-suite/.claude-plugin/plugin.json`, make exactly two changes:
1. `"version": "3.4.1"` → `"version": "3.5.0"`
2. `"commands": []` → `"commands": ["./commands/paper-implement.md", "./commands/lit-review.md", "./commands/replicate.md"]`

All other fields (agents, skills, description, keywords) remain unchanged.

- [ ] **Step 3: Update agent-core plugin.json — bump version only**

Change `"version": "3.4.1"` → `"version": "3.5.0"`. No other changes.

- [ ] **Step 4: Update dev-suite plugin.json — bump version only**

Change `"version": "3.4.1"` → `"version": "3.5.0"`. No other changes.

Also add `"commit": "./commands/commit.md"` to dev-suite if it is missing from the current commands array (check the file — if `./commands/commit.md` is already registered, skip).

- [ ] **Step 5: Run manifest tests**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py::TestManifests -v
```

Expected: all 11 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add plugins/science-suite/.claude-plugin/plugin.json \
        plugins/research-suite/.claude-plugin/plugin.json \
        plugins/agent-core/.claude-plugin/plugin.json \
        plugins/dev-suite/.claude-plugin/plugin.json
git commit -m "chore: bump all plugin versions to 3.5.0; register new commands and agents"
```

---

## Task 12: Full Test Run and Compliance Audit

**Files:** Run-only

- [ ] **Step 1: Run the full redesign test suite**

```bash
uv run pytest tools/tests/test_scicomp_redesign.py -v
```

Expected: **all tests PASS**. If any fail, fix the specific file before continuing.

- [ ] **Step 2: Run the existing 188-test suite**

```bash
uv run pytest tools/tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all 188 original tests still PASS (plus new tests from Task 2).

- [ ] **Step 3: Run v2.1.128 compliance validators**

```bash
uv run python3 tools/validation/metadata_validator.py plugins/agent-core
uv run python3 tools/validation/metadata_validator.py plugins/dev-suite
uv run python3 tools/validation/metadata_validator.py plugins/research-suite
uv run python3 tools/validation/metadata_validator.py plugins/science-suite
uv run python3 tools/validation/context_budget_checker.py
uv run python3 tools/validation/skill_validator.py
```

Expected: zero errors. If any validator reports errors, fix the specific file and re-run that validator.

- [ ] **Step 4: Verify pinn-engineer and sci-workflow-engineer are reachable from hub skills**

```bash
grep -r "pinn-engineer" plugins/science-suite/skills/
grep -r "sci-workflow-engineer" plugins/science-suite/skills/
```

If either returns no results, add a routing entry for that agent in the most relevant hub SKILL.md:
- `pinn-engineer` → add to `plugins/science-suite/skills/sciml-and-diffeq/SKILL.md` routing tree
- `sci-workflow-engineer` → add to `plugins/science-suite/skills/llm-and-ai/SKILL.md` routing tree

- [ ] **Step 5: Commit merge-gate checkpoint**

```bash
git add -A
git commit -m "test: add scicomp redesign test suite; all validators pass at v3.5.0"
```

---

## Task 13: Gemini WS-1 — SKILL.md Large-Context Analysis

**Files:** Read-only analysis pass (writes happen in Task 14)

- [ ] **Step 1: Launch Gemini agent with full SKILL.md corpus**

Dispatch the `cc-gemini-plugin:gemini-agent` with this prompt:

> You are doing a large-context read of all SKILL.md files in a Claude Code plugin system. Your task is analysis only — no file writes.
>
> Read every SKILL.md file under `plugins/science-suite/skills/` and `plugins/research-suite/skills/` (all sub-directories).
>
> Answer these four questions with specific file paths and line references:
>
> 1. **Redundant routing blocks**: Which hub routing trees have blocks duplicated across multiple skills? List each duplicated block, which files contain it, and which one to keep.
>
> 2. **Compression targets**: Which sub-skills exceed 80% of a 2% context budget (approximately 4,000 tokens = ~16,000 chars)? Rank by size descending. For each, identify which content section is the largest and could be gated to `--mode deep`.
>
> 3. **Orphan sub-skills**: Which sub-skills have ≤1 inbound routing reference from any hub SKILL.md? List file paths.
>
> 4. **llm-and-ai vs llm-engineering overlap**: Compare `plugins/science-suite/skills/llm-and-ai/` with `plugins/agent-core/skills/llm-engineering/`. List exact sections that are semantically duplicated and a recommendation for which suite should own each section.
>
> Format your output as four numbered sections with bullet points. Be specific: file paths, approximate character counts, exact content to remove or gate.

- [ ] **Step 2: Save Gemini output to a working file**

```bash
# After Gemini returns output, save it:
cat > /tmp/gemini_skill_analysis.md << 'GEMINI_OUTPUT'
[paste Gemini output here]
GEMINI_OUTPUT
```

---

## Task 14: Apply Gemini SKILL.md Compression Findings

**Files:** `plugins/science-suite/skills/**/SKILL.md`, `plugins/research-suite/skills/**/SKILL.md`

- [ ] **Step 1: Address redundant routing blocks**

For each duplicated block identified by Gemini in Task 13, remove the duplicate from the non-canonical file. Keep the version in the file Gemini identified as canonical.

After each removal, verify the routing tree still has a path to every sub-skill it previously covered:

```bash
grep -n "→\|->\\|route\|delegate" <modified_skill_file>
```

- [ ] **Step 2: Gate large content sections to --mode deep**

For each sub-skill Gemini flagged as >80% context budget, follow the **existing pattern** from `plugins/research-suite/skills/scientific-review/SKILL.md` (the reference implementation for mode-flag gating):

1. Read `plugins/research-suite/skills/scientific-review/SKILL.md` to see the exact `--mode` flag structure used there.
2. Apply the same structure to the flagged sub-skill: add a `## Mode Flag` section near the top listing what loads at each tier, and mark the heavy reference sections as `deep`-only using the same prose pattern (not HTML comments — the gating is instruction-based, not parsed markup).
3. The three tiers must be:
   - `--mode quick`: routing table + agent delegation only
   - `--mode standard` (default): + core sub-skill descriptions (3-4 items)
   - `--mode deep`: + full reference tables, constraint lists, worked examples

- [ ] **Step 3: Remove orphaned sub-skills**

For each orphan sub-skill Gemini identified (≤1 inbound reference):

1. Confirm it truly has no callers: `grep -r "<skill-name>" plugins/`
2. If confirmed orphan: `git rm plugins/<suite>/skills/<skill-name>/SKILL.md`
3. If the directory only contained SKILL.md: `git rm -r plugins/<suite>/skills/<skill-name>/`

- [ ] **Step 4: Deduplicate llm-and-ai vs llm-engineering**

Apply Gemini's recommendation for which sections to keep in each suite. Remove the duplicate section from the non-owning suite's SKILL.md and add a cross-reference line:

```markdown
> See `agent-core:llm-engineering` for [section topic].
```

- [ ] **Step 5: Run full validation after compression**

```bash
uv run pytest tools/tests/ -v --tb=short 2>&1 | tail -20
uv run python3 tools/validation/context_budget_checker.py
uv run python3 tools/validation/skill_validator.py
```

Expected: all tests pass, no skills over budget, no orphan sub-skills.

- [ ] **Step 6: Commit compression changes**

```bash
git add plugins/science-suite/skills/ plugins/research-suite/skills/
git commit -m "feat(skills): apply Gemini WS-1 SKILL.md compression — mode-flag gating, dedup routing, orphan removal"
```

---

## Task 15: Final Validation Gate

**Files:** Run-only

- [ ] **Step 1: Run complete test suite**

```bash
uv run pytest tools/tests/ -v 2>&1 | tail -30
```

Expected: zero failures.

- [ ] **Step 2: Run all validators**

```bash
uv run python3 tools/validation/metadata_validator.py plugins/agent-core
uv run python3 tools/validation/metadata_validator.py plugins/dev-suite
uv run python3 tools/validation/metadata_validator.py plugins/research-suite
uv run python3 tools/validation/metadata_validator.py plugins/science-suite
uv run python3 tools/validation/context_budget_checker.py
uv run python3 tools/validation/skill_validator.py
```

Expected: zero errors across all validators.

- [ ] **Step 3: Verify version sync across all 4 manifests**

```bash
grep '"version"' plugins/*/".claude-plugin"/plugin.json
```

Expected: all four lines show `"version": "3.5.0"`.

- [ ] **Step 4: Verify science-suite has exactly 11 agents registered**

```bash
python3 -c "
import json
p = json.load(open('plugins/science-suite/.claude-plugin/plugin.json'))
agents = p['agents']
print(f'Agents ({len(agents)}):')
for a in sorted(agents): print(' ', a)
"
```

Expected: 11 agents, including `pinn-engineer` and `sci-workflow-engineer`, excluding `ai-engineer` and `prompt-engineer`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: plugin scientific computing redesign v3.5.0

- science-suite: jax-pro + julia-pro promoted to opus; ml-expert → haiku
- science-suite: ai-engineer → pinn-engineer (physics-informed AI)
- science-suite: prompt-engineer → sci-workflow-engineer (scientific LLM workflows)
- science-suite: all 11 agent descriptions trimmed to ≤180 chars
- science-suite: /md-sim + /benchmark commands added
- research-suite: /paper-implement + /lit-review + /replicate commands added
- agent-core: PreCompact hook logs priority skills for post-compact reload
- skills: mode-flag gating applied to heavy sub-skills (Gemini WS-1)
- chore: .claudeignore added; all versions bumped to 3.5.0

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Spec Coverage Self-Review

| Spec Requirement | Task |
|---|---|
| jax-pro → opus | Task 3 |
| julia-pro → opus | Task 3 |
| ml-expert → haiku | Task 3 |
| ai-engineer → pinn-engineer | Task 5 |
| prompt-engineer → sci-workflow-engineer | Task 6 |
| Description trimming ≤180 chars | Tasks 3, 4, 5, 6 |
| Mode-flag gating for heavy skills | Task 14 |
| Routing tree compression | Task 14 |
| PreCompact cache-pin logging | Task 7 |
| .claudeignore | Task 10 |
| /md-sim command | Task 8 |
| /benchmark command | Task 8 |
| /paper-implement command | Task 9 |
| /lit-review command | Task 9 |
| /replicate command | Task 9 |
| v2.1.128 compliance audit | Tasks 1, 12 |
| Version bump to 3.5.0 | Task 11 |
| Manifest agent path updates | Task 11 |
| Gemini WS-1 SKILL.md analysis | Task 13 |
| Apply Gemini findings | Task 14 |
| Final validation gate | Task 15 |
| pinn-engineer reachable from hub | Task 12 Step 4 |
| sci-workflow-engineer reachable from hub | Task 12 Step 4 |
