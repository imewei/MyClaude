---
name: dev-workflows
description: Meta-orchestrator for development workflows and tooling. Routes to Git workflow, documentation standards, data pipeline orchestration (Airflow), and systematic debugging skills. Use when managing Git branches and commits, maintaining documentation workflow standards, orchestrating data pipelines with Airflow, or systematically debugging runtime issues. For writing scientific software docs or tutorials, use documentation-expert or /docs.
---

# Dev Workflows

Orchestrator for day-to-day development workflows and tooling. Routes to the appropriate specialized skill based on the version control, documentation, pipeline orchestration, or debugging need.

## Expert Agent

- **`debugger-pro`**: Specialist for systematic debugging, workflow root cause analysis, and pipeline failure diagnosis.
  - *Location*: `plugins/dev-suite/agents/debugger-pro.md`
  - *Capabilities*: Git conflict resolution, documentation audits, Airflow DAG debugging, and structured debugging methodology.

## Core Skills

### [Git Workflow](../git-workflow/SKILL.md)
Branch strategies, commit conventions, rebase vs merge, and conflict resolution patterns.

### [Documentation Standards](../documentation-standards/SKILL.md)
README structure, API docs, ADRs, and documentation-as-code with automated publishing.

### [Airflow Scientific Workflows](../airflow-scientific-workflows/SKILL.md)
Airflow DAG design, task dependencies, sensor patterns, and data pipeline orchestration. Covers both scientific and general-purpose Airflow workflows.

### [Debugging Toolkit](../debugging-toolkit/SKILL.md)
Systematic debugging methodology, profiler-guided diagnosis, and root cause analysis frameworks.

### [Three-Brain Router](../three-brain/SKILL.md)
Multi-model routing between Claude, Codex, and Gemini. Use for second-opinion code review, high-risk path scrutiny, repeated-failure rescue, multimodal analysis (video/audio/PDF/images), and long-context repository scans.

### [AI Pair](../ai-pair/SKILL.md)
Persistent three-model team (Claude developer/author + Codex reviewer + Gemini reviewer) for iterative code development or content creation. Use for ongoing multi-round projects where creation + dual-perspective review repeats across tasks. Distinct from three-brain: ai-pair runs a sustained team through multiple tasks; three-brain routes a single request.

## Routing Decision Tree

```
What is the workflow concern?
|
+-- Branch strategy / commits / merge conflicts?
|   --> git-workflow
|
+-- README / API docs / ADRs / doc-as-code?
|   --> documentation-standards
|
+-- Airflow DAG / task / sensor / pipeline?
|   --> airflow-scientific-workflows
|
+-- Bug diagnosis / profiler / root cause?
|   --> debugging-toolkit
|
+-- Second opinion / Codex review / Gemini scan (one-shot)?
|   --> three-brain
|
+-- /ai-pair / dev-team / content-team / multi-round team review?
    --> ai-pair
```

## Routing Table

| Trigger                                               | Sub-skill                       |
|-------------------------------------------------------|---------------------------------|
| Git branch, rebase, merge, conflict, tags             | git-workflow                    |
| README, ADR, docstring, Sphinx, MkDocs                | documentation-standards         |
| Airflow, DAG, task, XCom, sensor                      | airflow-scientific-workflows    |
| Debugger, pdb, breakpoint, root cause                 | debugging-toolkit               |
| Codex, Gemini, second opinion, sanity check, all three (one-shot) | three-brain        |
| /ai-pair, dev-team, content-team, team-stop, multi-round review   | ai-pair            |

## Checklist

- [ ] Identify whether the concern is version control, documentation, orchestration, or debugging
- [ ] Confirm Git branching strategy aligns with team size and release cadence
- [ ] Verify documentation is co-located with code and updated in the same PR
- [ ] Check Airflow DAGs have idempotent tasks before scheduling in production
- [ ] Validate debugging sessions start with hypothesis formation, not random changes
- [ ] Ensure debugging findings are documented to prevent regression
- [ ] For high-risk paths (auth, billing, migrations, infra) or repeated failures, route via three-brain
- [ ] For sustained multi-task projects needing iterative creation + dual review, use ai-pair
