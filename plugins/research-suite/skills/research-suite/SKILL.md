---
name: research-suite
description: Top-level entry point for all research tasks — peer review of manuscripts, developing your own research idea into a fundable plan, and methodology help (study design, paper reproduction, scientific writing, evidence synthesis).
---

# Research Suite

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
