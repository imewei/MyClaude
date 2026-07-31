---
name: dev-workflows
description: Meta-orchestrator for development workflows and tooling. Routes to Git workflow, documentation standards, data pipeline orchestration (Airflow), and systematic debugging skills. Use when managing Git branches and commits, maintaining documentation workflow standards, orchestrating data pipelines with Airflow, or systematically debugging runtime issues. For writing scientific software docs or tutorials, use documentation-expert or /docs.
---

# Dev Workflows

Orchestrator for day-to-day development workflows and tooling. Routes to the appropriate specialized skill based on the version control, documentation, pipeline orchestration, or debugging need.

## Expert Agent

No dev-suite agent specializes in general debugging directly. For systematic pre-fix
debugging, defer to `superpowers:systematic-debugging` or `mattpocock-skills:diagnosing-bugs`;
for scientific-computing-specific debugging, see [Debugging Toolkit](../debugging-toolkit/SKILL.md).

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
|   --> dev-suite:git-workflow
|
+-- README / API docs / ADRs / doc-as-code?
|   --> dev-suite:documentation-standards
|
+-- Airflow DAG / task / sensor / pipeline?
|   --> dev-suite:airflow-scientific-workflows
|
+-- Bug diagnosis / profiler / root cause?
|   --> dev-suite:debugging-toolkit
|
+-- Second opinion / Codex review / Gemini scan (one-shot)?
|   --> dev-suite:three-brain
|
+-- /ai-pair / dev-team / content-team / multi-round team review?
|   --> dev-suite:ai-pair
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to software-architect for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger                                               | Sub-skill                       |
|-------------------------------------------------------|---------------------------------|
| Git branch, rebase, merge, conflict, tags             | dev-suite:git-workflow                    |
| README, ADR, docstring, Sphinx, MkDocs                | dev-suite:documentation-standards         |
| Airflow, DAG, task, XCom, sensor                      | dev-suite:airflow-scientific-workflows    |
| Debugger, pdb, breakpoint, root cause                 | dev-suite:debugging-toolkit               |
| Codex, Gemini, second opinion, sanity check, all three (one-shot) | dev-suite:three-brain        |
| /ai-pair, dev-team, content-team, team-stop, multi-round review   | dev-suite:ai-pair            |

## Checklist

- [ ] Identify whether the concern is version control, documentation, orchestration, or debugging
- [ ] Confirm Git branching strategy aligns with team size and release cadence
- [ ] Verify documentation is co-located with code and updated in the same PR
- [ ] Check Airflow DAGs have idempotent tasks before scheduling in production
- [ ] Validate debugging sessions start with hypothesis formation, not random changes
- [ ] Ensure debugging findings are documented to prevent regression
- [ ] For high-risk paths (auth, billing, migrations, infra) or repeated failures, route via dev-suite:three-brain
- [ ] For sustained multi-task projects needing iterative creation + dual review, use dev-suite:ai-pair
