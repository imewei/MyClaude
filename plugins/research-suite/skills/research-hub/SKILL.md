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
- **`research-spark-orchestrator`**: Artifact-gated pipeline from rough idea to fundable proposal — 5-stage core + optional extension.
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
|   --> research-suite:scientific-review
|
+-- Developing YOUR OWN research idea into a fundable plan?
|   --> research-suite:research-spark
|
+-- Methodology help (design, reproduce, evaluate, write, synthesize)?
|   --> research-suite:research-practice
|       |
|       +-- Design a study before data collection      → research-suite:research-methodology
|       +-- Evaluate an existing paper / grant         → research-suite:research-quality-assessment
|       +-- Reproduce a published paper in code        → research-suite:research-paper-implementation
|       +-- Draft a manuscript / report / poster       → research-suite:scientific-communication
|       +-- Systematic review / meta-analysis / GRADE  → research-suite:evidence-synthesis
|
+-- None of the above / concern is ambiguous?
    --> Delegate to research-expert for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger | Skill |
|---|---|
| Peer review, referee report, review someone's paper, critique manuscript, novelty assessment, statistical claims, data integrity check, journal submission, assess preprint | `research-suite:scientific-review` |
| Refine my idea, fundable plan, research spark pipeline, continue project, resume stage, back to X work, redo stage N, premortem the plan | `research-suite:research-spark` |
| Study design, reproduce paper, write manuscript, evidence synthesis, "is this trustworthy", CONSORT, STROBE, PRISMA, p-hacking, HARKing, power analysis, IMRaD, GRADE, meta-analysis, literature review | `research-suite:research-practice` |

## Checklist

- [ ] Confirm whose work is in focus: *other people's manuscript* → `research-suite:scientific-review`; *user's own idea* → `research-suite:research-spark`.
- [ ] If the user is mid-pipeline in `research-suite:research-spark`, resume at the correct stage rather than restarting.
- [ ] For `research-suite:research-practice`, identify the lifecycle phase (design / evaluate / reproduce / write / synthesize) before loading a sub-skill.
- [ ] Never route to `_research-commons` directly — it is an internal shared resource, not a user-facing skill.
- [ ] For any figures or visualization needs, defer to `scientific-visualization` in `science-suite`.
