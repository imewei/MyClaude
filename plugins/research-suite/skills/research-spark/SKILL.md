---
name: research-spark
description: Orchestrator for a research-refinement pipeline. A five-stage core (spark, landscape, claim, theory) turns a rough idea into a testable, fundable research proposal; three further stages (numerical prototype, experiment design, premortem) are an optional extension toward execution, not required for the proposal itself. Triggers on phrases like "work on my idea about X", "refine this research spark", "let's scope this project", "continue the project on Y", "turn this into a fundable plan", "walk this idea through research-spark", or any description of a rough research idea the user wants to sharpen into a testable proposal. Also triggers when the user resumes work on a prior project (even implicitly, by saying "back to the X work") or asks to enter a specific stage by name. The orchestrator itself does not do the stage work; it figures out where the user is, loads the right specialist skill (spark-articulator, landscape-scanner, falsifiable-claim, theory-scaffold, numerical-prototype, experiment-designer, or premortem-critique), and at Stage 5 runs a hostile self-audit and assembles a proposal draft before asking whether to stop or continue into the optional extension.
---

# research-spark

The dispatcher for the research-refinement pipeline. Detects stage, loads specialist, enforces the artifact contract, keeps state. A five-stage **core** turns a rough idea into a testable, fundable proposal; three further stages are an **optional extension** toward execution.

## The pipeline at a glance

**Core (required).** The pipeline's job is done here: a Stage 5 artifact plus a passed checkpoint is a complete, testable, fundable research proposal.

| Stage | Skill | Artifact |
|-------|-------|----------|
| 1 | [spark-articulator](../spark-articulator/SKILL.md) | `01_spark.md` |
| 2 | [landscape-scanner](../landscape-scanner/SKILL.md) | `02_landscape.md` |
| 3 | [falsifiable-claim](../falsifiable-claim/SKILL.md) | `03_claim.md` |
| 4-5 | [theory-scaffold](../theory-scaffold/SKILL.md) | `04_theory.md` + `05_formalism.tex` |

**Extension (optional).** Only for users who want to carry the proposal toward execution: numerical validation, experiment design, red-teaming. Never auto-entered; the orchestrator asks first.

| Stage | Skill | Artifact |
|-------|-------|----------|
| 6 | [numerical-prototype](../numerical-prototype/SKILL.md) | `06_prototype.md` + `code/` |
| 7 | [experiment-designer](../experiment-designer/SKILL.md) | `07_plan.md` |
| 8 | [premortem-critique](../premortem-critique/SKILL.md) | `08_premortem.md` |

Each artifact filename is canonical. Specialists write to these paths; they do not invent new names. This is the single most important thing to preserve, because downstream stages read by path.

## Routing Decision Tree

```
What triggered the invocation?
|
+-- Rough new idea (no prior artifacts)?
|   --> Stage 1: research-suite:spark-articulator
|
+-- Resuming a project (explicit "back to X" or implicit mid-conversation)?
|   --> Read _state.yaml; summarize state; ask whether to continue or jump.
|       Then dispatch to the current stage's skill using the dispatch table below.
|
+-- User names a stage directly ("redo Stage 3", "premortem the plan")?
|   +-- Prior-stage artifact exists?
|   |   --> Load named specialist per dispatch table below; log jump.
|   +-- Prior-stage artifact missing?
|       --> Refuse; offer to run the missing stage first.
|
+-- Stage 5 complete (theory exists, proposal not yet assembled)?
|   --> Core-completion checkpoint: run the hostile self-audit, then
|       assemble proposal_draft.md (reverse-order draft). Only then ask:
|       stop here with the proposal, or continue into the optional
|       extension (Stage 6)? Record `core_complete: true` in _state.yaml.
|       Never auto-advance into Stage 6 the way Stages 1-5 auto-advance
|       into each other.
|
+-- Stage N complete, advancing to N+1 (N != 5)?
|   --> Verify N's canonical artifact exists; load next specialist per dispatch table.
|
+-- None of the above / invocation is ambiguous?
|   --> Delegate to research-spark-orchestrator for open-ended triage, or clarify
|       the current stage and re-enter the routing decision tree.
|
By stage (canonical dispatch table):
  1   -> research-suite:spark-articulator     (elicit 3-line spark)               [core]
  2   -> research-suite:landscape-scanner     (prior art + gap + Reviewer 2 pass) [core]
  3   -> research-suite:falsifiable-claim     (testable claim + kill criterion)   [core]
  4-5 -> research-suite:theory-scaffold       (narrative theory + formalism.tex)  [core]
  6   -> research-suite:numerical-prototype   (computational existence check)     [optional]
  7   -> research-suite:experiment-designer   (DoE, power, pre-registration)      [optional]
  8   -> research-suite:premortem-critique    (red-team before execution)         [optional]
```

## How routing works

Four situations cover almost all invocations.

**Completing the core.** When Stage 5 finishes, the theory exists but the proposal is not yet assembled. Two required passes close the core, in order:

1. **Hostile self-audit**: `../_research-commons/templates/hostile_self_audit.md`. Reasoning-only: the $100k Kill Switch, the Artifact Check, the Bottleneck Route. Exit gate: no aim may rest on a single unvalidated technique without a documented fallback.
2. **Proposal assembly**: `../_research-commons/templates/proposal_assembly.md`. Synthesizes `01_spark.md` through `05_formalism.tex` plus the audit into `proposal_draft.md`, drafted in reverse order (Aims → Methods & Risk → Background → Abstract). Exit gate: every paragraph traces to a specific aim (line-of-sight check).

Only after both pass is the core complete. Record `core_complete: true` in `_state.yaml`, then ask explicitly: stop here, or continue into the optional extension (Stage 6 onward)? Do not auto-advance into Stage 6 the way Stages 1-5 advance into each other, and do not treat stopping here as unfinished work: a proposal that passed both gates is done, not partial.

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
core_complete: false   # set true once the hostile audit passes AND proposal_draft.md is assembled, not just when Stage 5 finishes
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

**The core is a valid endpoint.** A Stage 5 theory plus a passed hostile audit plus an assembled `proposal_draft.md` is a complete research proposal. Stopping there is success, not a partial run. Only enter Stage 6 onward after the user explicitly asks to continue.

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
├── proposal_draft.md   # written at the core-completion checkpoint, after Stage 5
├── artifacts/
│   ├── 01_spark.md
│   └── ...
└── code/          # emerges only if the optional extension (Stage 6) runs
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
