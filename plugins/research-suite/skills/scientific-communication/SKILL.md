---
name: scientific-communication
description: Structure scientific arguments and write technical reports (IMRaD), abstracts, posters, and grant-style one-pagers. This skill should be used when the user asks to "structure the Introduction", "write the abstract", "IMRaD outline", "turn this into a paper", "draft the methods section", "make a one-pager", "Heilmeier catechism", "tighten the writing", "write a poster", or any request to organize or clarify scientific prose. For peer-review reports targeting a specific journal, use `scientific-review`; for designing the study the methods section will describe, use `research-methodology`; for the Stage 1 compression inside a research-spark project, use `spark-articulator`.
---

# Scientific Communication

Principles for clear, precise, and effective scientific writing and presentation.

## Expert Agent

For drafting and refining scientific content, delegate to:

- **`research-expert`**: Unified specialist for Scientific Communication.
  - *Location*: `plugins/research-suite/agents/research-expert.md`
  - *Capabilities*: IMRaD structuring, clarity optimization, and technical reporting.

## IMRaD Structure

The standard format for original research papers.

| Section | Purpose | Key Content |
|---------|---------|-------------|
| **Introduction** | Why did you do it? | Context, gap, objective, hypothesis. |
| **Methods** | How did you do it? | Materials, procedure, analysis (reproducible). |
| **Results** | What did you find? | Data, figures, tables (no interpretation). |
| **Discussion** | What does it mean? | Interpretation, limitations, context, future work. |

## Writing Principles

### Clarity & Precision
- **Be Specific**: Avoid "very", "significantly" (without stats), "good". Use numbers.
- **Active Voice**: "We measured the temperature" (Active) vs "The temperature was measured" (Passive). Active is generally preferred for clarity.
- **Simple Language**: Don't use a $10 word when a $1 word will do. "Utilize" $\to$ "Use".

### Paragraph Structure
1.  **Topic Sentence**: The main idea of the paragraph.
2.  **Supporting Evidence**: Data, citations, logic.
3.  **Concluding Sentence**: Summary or transition to the next paragraph.

## Technical Reports

### Structure
1.  **Executive Summary**: Key findings and recommendations (1 page).
2.  **Background**: Problem statement and context.
3.  **Methodology**: Approach taken.
4.  **Findings**: Detailed results.
5.  **Conclusions & Recommendations**: Actionable items.

## Citation Management

- **BibTeX**: Standard for LaTeX.
- **Zotero/Mendeley**: Reference managers.
- **Style**: Follow journal/conference guidelines (IEEE, APA, Nature).

## Shared templates

For the structural details of specific artifacts, consult the shared commons templates:

- `../_research-commons/templates/abstract.md` — five-move abstract structure with length variants (35-word pitch, 150-word conference, 250-word journal, 400-word structured).
- `../_research-commons/templates/heilmeier.md` — six-question catechism for grant-style framings.
- `../_research-commons/templates/onepage.md` — one-page executive summary format.
- `../_research-commons/style/writing_constraints.md` — banned vocabulary and formatting rules (em-dash prohibition, quantified-over-qualitative, etc.) enforced across the research stack.
- `../_research-commons/style/citation_style.md` — APS PRL-baseline citation conventions.

## Related skills

- `research-methodology` — for writing the methods section after design lock-in.
- `evidence-synthesis` — for reporting systematic reviews or meta-analyses (PRISMA, GRADE).
- `research-quality-assessment` — for self-review before submission.
- `spark-articulator` (research-spark Stage 1) — when the goal is to compress a rough idea into a 3–5 sentence articulation, not draft a paper.
- `scientific-review` — for writing peer-review comments on *another* author's manuscript rather than drafting your own.

## Checklist

- [ ] Narrative flow is logical and coherent.
- [ ] Jargon is defined or avoided.
- [ ] Figures are referenced in text.
- [ ] Citations are accurate and formatted correctly.
- [ ] Abstract summarizes the entire paper (Background, Methods, Results, Conclusion).
- [ ] Banned-vocabulary lint passed via `../_research-commons/scripts/style_lint.py` before submission.
