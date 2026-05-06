---
name: research-hub
description: >-
  Top-level router for scientific research workflows. Use for: reviewing or critiquing a manuscript/paper/peer review/referee report/journal submission/novelty assessment/statistical claims/data integrity; refining a rough research idea through structured stages (research spark, falsifiable claim, theory scaffold, experiment design, premortem); open-ended research methodology questions — "how should I design this study", "is this paper trustworthy", "reproduce this result", "help me write this up", power analysis/DoE/hypothesis planning/CONSORT/STROBE/PRISMA/meta-analysis/GRADE/IMRaD/literature synthesis; reproducing or implementing a published paper.
---

# Research Suite (research-hub)

Meta-router. Identifies the user's research task and delegates to the correct specialist skill.

## Expert Agents

- **`research-expert`**: Methodology, literature synthesis, scientific communication.
  - *Location*: `plugins/research-suite/agents/research-expert.md`
- **`research-spark-orchestrator`**: Artifact-gated eight-stage pipeline from rough idea to fundable plan.
  - *Location*: `plugins/research-suite/agents/research-spark-orchestrator.md`

## Hub Skills

- [**scientific-review**](../scientific-review/SKILL.md) — Formal peer review of *other people's* manuscripts; produces a .docx referee report with Confidential Comments to Editor.
- [**research-spark**](../research-spark/SKILL.md) — Eight-stage artifact-gated pipeline (spark → landscape → claim → theory → prototype → plan → premortem) for refining *your own* research idea into a fundable plan.
- [**research-practice**](../research-practice/SKILL.md) — Methodology hub: study design, paper reproduction, quality assessment, scientific writing, evidence synthesis.

## Routing Decision Tree

```
Whose manuscript / whose idea?
|
+-- Reviewing SOMEONE ELSE's paper → referee report needed?
|   --> scientific-review
|
+-- Developing YOUR OWN research idea into a fundable plan?
|   --> research-spark
|
+-- Methodology help (design, reproduce, evaluate, write, synthesize)?
    --> research-practice
        |
        +-- Design a study before data collection      → research-methodology
        +-- Evaluate an existing paper / grant         → research-quality-assessment
        +-- Reproduce a published paper in code        → research-paper-implementation
        +-- Draft a manuscript / report / poster       → scientific-communication
        +-- Systematic review / meta-analysis / GRADE  → evidence-synthesis
```

## Routing Table

| Trigger | Skill |
|---|---|
| Peer review, referee report, review someone's paper | `scientific-review` |
| Refine my idea, fundable plan, research spark pipeline | `research-spark` |
| Study design, reproduce paper, write manuscript, evidence synthesis | `research-practice` |

## Checklist

- [ ] Confirm whose work is in focus: *other people's manuscript* → `scientific-review`; *user's own idea* → `research-spark`.
- [ ] If the user is mid-pipeline in `research-spark`, resume at the correct stage rather than restarting.
- [ ] For `research-practice`, identify the lifecycle phase (design / evaluate / reproduce / write / synthesize) before loading a sub-skill.
- [ ] Never route to `_research-commons` directly — it is an internal shared resource, not a user-facing skill.
- [ ] For any figures or visualization needs, defer to `scientific-visualization` in `science-suite`.
