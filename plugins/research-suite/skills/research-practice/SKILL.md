---
name: research-practice
description: Meta-orchestrator for the research lifecycle. Routes to specialized skills for *designing* experiments (power analysis, DoE, hypothesis planning), *evaluating* existing work (CONSORT/STROBE/PRISMA, red-flag detection), *reproducing* published papers, *writing* manuscripts (IMRaD, reports), and *synthesizing* literature (PRISMA, meta-analysis, GRADE). This skill should be used when the user asks any open-ended methodology question — "how should I design this study", "is this paper trustworthy", "reproduce this result", "help me write this up", "what does the literature say on X" — and the task is neither a structured artifact-gated pipeline (use `research-spark`) nor a formal journal peer review (use `scientific-review`).
---

# Research Practice

Orchestrator for the research lifecycle. Routes problems to the appropriate specialized skill based on the phase of work.

## Expert Agent

- **`research-expert`**: Specialist for research methodology, literature synthesis, and scientific communication.
  - *Location*: `plugins/research-suite/agents/research-expert.md`
  - *Capabilities*: Study design, paper implementation, quality assessment, scientific writing, evidence synthesis.

## Phase-aligned sub-skills

Each sub-skill owns one phase of research. The boundaries are deliberate — see each skill's "Scope boundary" section for the specific contract.

### [Research Methodology](../research-methodology/SKILL.md) — *design phase*
Hypothesis formulation, power analysis, sample-size justification, ablation planning, statistical-test selection, pre-registration. Used *before* data collection.

### [Research Quality Assessment](../research-quality-assessment/SKILL.md) — *evaluate phase*
Score methodology against CONSORT/STROBE/PRISMA guidelines, detect red flags (p-hacking, HARKing, selective reporting, circular analysis), check reproducibility. Used on *existing* manuscripts, preprints, or grant proposals.

### [Research Paper Implementation](../research-paper-implementation/SKILL.md) — *reproduce phase*
Translate a published paper's methods section into runnable code. Recover missing hyperparameters, map notation to variables, validate against reported benchmarks.

### [Scientific Communication](../scientific-communication/SKILL.md) — *write-up phase*
IMRaD structure, technical reports, posters. Load `_research-commons/templates/abstract.md` for abstracts.

### [Evidence Synthesis](../evidence-synthesis/SKILL.md) — *synthesize phase*
PRISMA systematic reviews, meta-analysis (effect-size pooling, heterogeneity via I²/Q), GRADE evidence grading.

## Related skills in this suite

Two stand-alone workflows sit *outside* this hub's routing because they are complete pipelines, not single-phase specialists:

- **`scientific-review`** — peer-review workflow for reviewing *other people's* manuscripts. Produces a .docx referee report with Confidential Comments to Editor. Use this instead of research-quality-assessment when the deliverable is a journal peer-review report.
- **`research-spark` pipeline** — eight-stage artifact-gated refinement of your own rough research idea into a fundable plan. Use this instead of research-methodology when you want structured handoffs (spark → landscape → claim → theory → prototype → plan → premortem) with state tracked in `_state.yaml`.

### Phase ↔ research-spark stage mapping

When the user is already inside an active research-spark project, the pipeline stage supersedes this hub's generic sub-skill — the stage version enforces tighter artifact contracts.

| Phase (this hub) | research-spark Stage |
|------------------|----------------------|
| Design (research-methodology) | Stage 7 — `experiment-designer` (DoE + capability margin + pre-registered metrics) |
| Reproduce (research-paper-implementation) | Stage 6 — `numerical-prototype` (when the target is the user's own formalism, not another paper) |
| Synthesize (evidence-synthesis) | Stage 2 — `landscape-scanner` (three-layer scan + Reviewer 2 pass, not PRISMA) |
| Write up (scientific-communication) | Stage 1 — `spark-articulator` (for the rough-idea compression) + `_research-commons/templates/heilmeier.md` (for the grant framing) |
| Evaluate (research-quality-assessment) | Stage 8 — `premortem-critique` (internal red-team of *your* plan, not audit of someone else's) |

## Figures

Publication-quality scientific visualization is provided by `scientific-visualization` in `science-suite` (cross-suite because it also serves ML and physics workflows). Load it when figures matter.

## Routing Decision Tree

```
What is the task category?
|
+-- Writing a journal peer-review report (.docx)?
|   --> (out of hub) scientific-review
|
+-- Refining a rough idea into a fundable plan with artifact handoffs?
|   --> (out of hub) research-spark
|
+-- Designing an experiment before data collection?
|   --> research-methodology
|
+-- Evaluating existing work (manuscript, grant, preprint)?
|   +-- Need a scored rubric / red-flag audit?    --> research-quality-assessment
|   +-- Need a journal peer-review .docx?         --> scientific-review
|
+-- Translating a published paper into code?
|   --> research-paper-implementation
|
+-- Drafting a manuscript / report / poster?
|   --> scientific-communication
|
+-- Systematic review / meta-analysis / GRADE evidence grading?
|   --> evidence-synthesis
```

## Checklist

- [ ] For paper implementation, read the methods section *and* appendix fully before writing code — critical details often live in the appendix.
- [ ] For quality assessment, check sample-size justification and multiple-comparison handling *before* commenting on effect sizes.
- [ ] For manuscript drafting, align terminology with the target journal's conventions and run `_research-commons/scripts/style_lint.py` before finalizing.
- [ ] For evidence synthesis, pre-register the inclusion and exclusion criteria on OSF or PROSPERO.
- [ ] For study design, identify the null and alternative hypotheses, the effect size of interest, and the kill criterion before running power analysis.
