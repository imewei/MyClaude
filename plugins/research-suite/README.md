# Research Suite

## Overview

Scientific research workflows for Claude Code. Three complementary tracks:

1. **`scientific-review`** — produce a rigorous, journal-ready peer review as a `.docx` (for reviewing *other people's* manuscripts).
2. **`research-spark` stack** — refine a rough research idea into a fundable plan through eight artifact-gated stages (for scoping *your own* project).
3. **`research-practice` hub** — general research methodology: study design, paper reproduction, quality assessment, scientific writing (IMRaD), and evidence synthesis (PRISMA, meta-analysis, GRADE).

All three prioritize domain rigor, adversarial critique, and explicit handoffs over freeform chat.

## Quick Start

```bash
# Peer review (scientific-review skill, one-shot)
"Review this manuscript for Physical Review Letters"

# Research spark pipeline (orchestrator, multi-stage)
"I have a rough idea about spectral gaps as early warning for flocculation —
 can we run it through research-spark?"

# Direct stage invocation, skipping the orchestrator
"Premortem the current plan in ./research-spark/my-project/"

# Research methodology (research-practice hub routes)
"Help me design a power analysis for this RCT"
"Reproduce the results of https://arxiv.org/abs/2310.12345"
"Write a systematic review of diffusion-model generalization papers"
```

## Agents

| Agent | Model | Specialization |
|-------|-------|----------------|
| `research-expert` | opus | Research methodology, literature synthesis, scientific communication — one-off tasks |
| `research-spark-orchestrator` | opus | Autonomous driver for the 8-stage research-spark pipeline; owns state, enforces artifact contract, fans out to sub-agents |

Use `research-expert` for discrete methodology tasks (power analysis, systematic review, IMRaD structuring). Use `research-spark-orchestrator` when you have a rough research idea you want to walk through the full articulation → premortem pipeline with artifact-gated handoffs.

## Commands

This plugin ships **zero registered slash commands** — every workflow is skill-driven. The legacy `/paper-review` command was removed because `scientific-review` auto-triggers on "review this paper" phrasings and produces a strictly better deliverable (.docx with journal-specific adaptation, Six-Lens analysis, Confidential Comments to Editor). If a new command ever gets added, it will appear here.

## Skills

### Stand-alone workflows

| Skill | Role |
|-------|------|
| `scientific-review` | Six-lens peer review → `.docx` |

### research-spark stack (pipeline)

| Skill | Role |
|-------|------|
| `research-spark` | Orchestrator for the 8-stage refinement pipeline |
| `spark-articulator` | **Stage 1** — rough idea → 3-to-5-sentence articulation |
| `landscape-scanner` | **Stage 2** — three-layer literature scan + Reviewer 2 pass |
| `falsifiable-claim` | **Stage 3** — claim + Heilmeier doc + kill criterion |
| `theory-scaffold` | **Stages 4–5** — stepwise derivation → LaTeX formalism |
| `numerical-prototype` | **Stage 6** — JAX prototype + three validation passes |
| `experiment-designer` | **Stage 7** — DoE + instrument capability map + power analysis |
| `premortem-critique` | **Stage 8** — failure narratives + simulated reviewers |
| `_research-commons` | Shared style, code architecture, templates, scripts (not invoked directly) |

### research-practice hub (methodology)

| Skill | Role |
|-------|------|
| `research-practice` | Hub routing to the five methodology sub-skills below |
| `research-methodology` | Study design, hypothesis formulation, literature review strategy |
| `research-paper-implementation` | Reproduce published results from methods sections |
| `research-quality-assessment` | Evaluate statistical rigor, reproducibility, sample-size justification |
| `scientific-communication` | IMRaD structure, technical reports, posters |
| `evidence-synthesis` | PRISMA, meta-analysis, GRADE |

> **Figures:** publication-quality scientific visualization is provided by `scientific-visualization` in `science-suite` — cross-suite because it's useful beyond research (ML training curves, physics sweeps). Load it when figures matter.

### The research-spark artifact contract

Each stage writes one canonical artifact that the next consumes. This is the load-bearing invariant of the pipeline.

| Stage | Artifact | Format |
|-------|----------|--------|
| 1 | `01_spark.md` | 3–5 sentence articulation with rejected variants |
| 2 | `02_landscape.md` | Bibliography, steelman notes, gap matrix, Reviewer 2 transcript |
| 3 | `03_claim.md` | Claim, Heilmeier doc, kill criterion, Reviewer 2 challenge |
| 4–5 | `04_theory.md` + `05_formalism.tex` | Derivation, limits, dimensionless groups, gray-box spec |
| 6 | `06_prototype.md` + `code/` | Running JAX implementation, validation, predicted observable |
| 7 | `07_plan.md` | DoE matrix, capability map, power analysis, risk register |
| 8 | `08_premortem.md` | Failure narratives, early signals, reviewer critique |

State lives in `_state.yaml` at the project root; the orchestrator reads it before every action.

## Three adversarial patterns worth knowing

- **Reviewer 2 persona** (stages 2 and 3): an adversarial reviewer argues the gap isn't real / tractable / impact-bearing, or the claim is physically impossible / mathematically unsound / already solved. Each rebuttal must cite a specific paper.
- **Stepwise derivation protocol** (stages 4–5): one conceptual step per invocation, with a verification pass (dimensional check, limit check, sanity argument) before the next step. Blocks multi-step symbolic leaps where errors concentrate.
- **Instrument capability margin** (stage 7): for each measurable quantity, compute the margin between the predicted signal and instrument capability on each axis. Margin < 3× → high-risk measurement requiring explicit mitigation before the plan advances.

## Requirements

### scientific-review
- `python-docx` for `.docx` generation (markdown fallback available).
- `pandoc` for DOCX ingestion; `pdftotext` / `pypdf` / `pymupdf` for PDF.

### research-spark stack
- Python 3.12+, `uv` for dependency resolution.
- Stage-specific: `sympy` (theory-scaffold), `scipy` + `pyyaml` (experiment-designer), `jax` + `jaxlib` (numerical-prototype), `pdflatex` (latex_compile_check.sh).
- All scripts install locally via `uv add`, never globally.

## Project directory layout (research-spark)

```text
<workspace>/<idea-slug>/
├── _state.yaml
├── project_log.md
├── artifacts/
│   ├── 01_spark.md
│   └── ... 08_premortem.md
└── code/                    # emerges at Stage 6
    ├── pyproject.toml
    ├── src/<slug>/
    └── tests/
```

## Installation

```bash
/plugin enable research-suite
```

## Design notes

- **Three tracks, one plugin.** `scientific-review` evaluates *other people's* work; `research-spark` refines *your own* idea into a scoped project; `research-practice` covers the broader research lifecycle (design, reproduce, assess, write, synthesize). All three require the same mental discipline — adversarial critique, explicit falsifiability, artifact-gated handoffs.
- **One agent, zero commands.** The `research-expert` agent handles multi-step research delegations. Skills drive everything else — the old `/paper-review` command was removed because `scientific-review` auto-triggers on "review this paper" phrasings and produces a strictly better deliverable (.docx with journal-specific adaptation, Six-Lens analysis, Confidential Comments to Editor).
- **Phase-aligned sub-skills under `research-practice`.** Each methodology sub-skill owns one phase (design / evaluate / reproduce / write / synthesize) with explicit scope boundaries. See each skill's "Scope boundary" section to understand why the splits are drawn where they are.
- **Sub-agent fan-out.** The research-spark orchestrator can parallelize across Claude Code sub-agents at natural points: Stage 2 literature layers, Stage 6 validation passes, Stage 8 reviewer archetypes. The orchestrator itself is a skill, not an agent.
- **Style enforcement.** Every emitted markdown passes `../_research-commons/scripts/style_lint.py`: no em dashes, no banned vocabulary (*innovative, state-of-the-art, transformative, novel, groundbreaking*, etc.), quantified language preferred.
- **Trust the model, explain the why.** The skills lean on theory of mind rather than rigid MUSTs; they explain *why* a step matters so Claude can apply judgment at the edges.
- **Moved from science-suite (2026-04-18).** The `research-expert` agent and 5 methodology skills originated in `science-suite`. They moved here because peer review / methodology / writing are *evaluative* research activities, whereas `science-suite` focuses on *computational* work (JAX, Julia, simulations, ML).

## License

MIT — see repository root.
