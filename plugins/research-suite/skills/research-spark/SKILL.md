---
name: research-spark
description: Orchestrator for a structured research-refinement pipeline. Moves a research spark through eight stages from rough idea to executable plan, each stage producing a canonical artifact the next consumes. Triggers on phrases like "work on my idea about X", "refine this research spark", "let's scope this project", "continue the project on Y", "turn this into a fundable plan", "walk this idea through research-spark", or any description of a rough research idea the user wants to sharpen into a testable program. Also triggers when the user resumes work on a prior project (even implicitly, by saying "back to the X work") or asks to enter a specific stage by name. The orchestrator itself does not do the stage work; it figures out where the user is and loads the right specialist skill (spark-articulator, landscape-scanner, falsifiable-claim, theory-scaffold, numerical-prototype, experiment-designer, or premortem-critique).
---

# research-spark

The dispatcher for an eight-stage research-refinement pipeline. Detects stage, loads specialist, enforces the artifact contract, keeps state.

## The pipeline at a glance

| Stage | Skill | Artifact |
|-------|-------|----------|
| 1 | spark-articulator | `01_spark.md` |
| 2 | landscape-scanner | `02_landscape.md` |
| 3 | falsifiable-claim | `03_claim.md` |
| 4-5 | theory-scaffold | `04_theory.md` + `05_formalism.tex` |
| 6 | numerical-prototype | `06_prototype.md` + `code/` |
| 7 | experiment-designer | `07_plan.md` |
| 8 | premortem-critique | `08_premortem.md` |

Each artifact filename is canonical. Specialists write to these paths; they do not invent new names. This is the single most important thing to preserve, because downstream stages read by path.

## How routing works

Three situations cover almost all invocations.

**New spark.** The user describes a rough idea with no prior artifacts in the conversation. Propose a short slug from the user's phrasing, create a project directory, initialize `_state.yaml` at stage 1, and load spark-articulator. Ask if the proposed location and slug are fine before creating files.

**Resuming.** The user refers to a project the orchestrator has seen before, explicitly ("back to the RheoX work") or implicitly (arriving mid-conversation with shared context). Read `_state.yaml`, summarize where things left off (current stage, last decision point, open questions), and ask whether to continue where they left off or jump somewhere specific.

**Jumping to a stage.** The user names a stage directly ("let's premortem the current plan", "redo Stage 3"). Check that the prior-stage artifact exists; if not, explain what is missing and offer to run the missing stage first. If it does exist, load the named specialist and log the jump.

For re-entry to a stage that was previously completed, preserve the old artifact as `NN_name.v1.md` before writing the new version. Downstream artifacts are not invalidated automatically; ask the user whether they want to re-run downstream stages given the revision, rather than either silently keeping stale downstream or silently discarding it.

## State file

`_state.yaml` lives at the project root. Its job is to make resume and re-entry reliable across sessions. Minimum content:

```yaml
idea_slug: rheox_spectral_gap
title: "Spectral gap early warning for rheological transitions"
current_stage: 4
stages_completed: [1, 2, 3]
artifacts:
  stage_1: artifacts/01_spark.md
  stage_2: artifacts/02_landscape.md
  stage_3: artifacts/03_claim.md
last_updated: 2026-04-18T14:22:00
next_decision_point: "selection of stochastic framework for bond-exchange kinetics"
open_questions:
  - "Does the gray-box boundary sit at the memory kernel or the stress response?"
overrides:
  - "Stage 2 advanced with N=5 steelmanned papers (default 8); adjacent literature genuinely thin"
```

The state file is the single source of truth. If in-memory stage tracking disagrees with `_state.yaml`, trust the file.

## Principles the orchestrator upholds

**Artifact trail integrity.** Each stage requires its prior-stage artifact as input. If the user tries to skip, name the missing stage and offer to run it. A stage started without its input is cargo-culting the pipeline.

**Overrides are legitimate but logged.** Depth gates and other defaults (like "8 steelmanned papers") exist because skipping them usually costs later. Sometimes they should be skipped anyway. When that happens, record the override in `_state.yaml` and in `project_log.md` with a one-line reason. Future self will want to know why.

**Canonical paths.** Specialist skills that want to write somewhere non-canonical are either confused or wrong. If a specialist emits to the wrong path, move it to canonical and log the correction. Do not invent parallel directories.

**Re-entry does not silently invalidate downstream.** If Stage 3 is revised, Stage 4 might still be valid, or might not. Ask.

## Default workspace

If the user has not specified one, use `./research-spark/<idea-slug>/`. Default layout:

```
<workspace>/
├── _state.yaml
├── project_log.md
├── artifacts/
│   ├── 01_spark.md
│   └── ...
└── code/          # emerges at Stage 6
```

## First-run example

```
User: I had this idea about using spectral gaps of the stress-response operator
      as an early warning for flocculation transitions in battery slurries.

Claude: [detects a fresh spark; proposes slug "spectral-gap-flocculation"]
        Before I create any files, a quick check: I'd like to set up a project
        at ./research-spark/spectral-gap-flocculation/ and walk you through
        Stage 1 (articulation). Sound right, or do you want a different slug
        or location?

        [on confirmation, creates directory, initializes _state.yaml at stage 1,
        loads spark-articulator, begins the elicitation]
```

## What breaks and how to recover

**Missing prior artifact.** Do not proceed. Tell the user which stage is missing and offer to run it. Skipping produces garbage downstream.

**Corrupt `_state.yaml`.** Do not overwrite silently. Surface the specific problem (parse error, missing field) and ask whether to repair in place or recreate from scratch.

**Artifact written to wrong path.** This is almost always a specialist-skill bug, not a user-facing issue. Move the file to the canonical path, log the move in `project_log.md`, continue.

**User pushes to advance when a depth gate says no.** Allow it with an explicit override logged. The purpose of the gate is to make the decision deliberate, not to block it absolutely.

## Multi-agent note (Claude Code)

In Claude Code with sub-agents, the orchestrator can delegate stages to sub-agents with scoped prompts. Each specialist's SKILL.md documents its fan-out points. In single-agent contexts (Claude.ai chat), stages run sequentially in one conversation; the fan-out notes are aspirational and do not change behavior.
