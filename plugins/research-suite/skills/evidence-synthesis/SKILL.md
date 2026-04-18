---
name: evidence-synthesis
description: Conduct systematic literature reviews (PRISMA), meta-analyses, and evidence grading (GRADE). This skill should be used when the user asks to "systematic review", "meta-analysis", "PRISMA flow diagram", "grade this evidence", "pool effect sizes", "compute I-squared / Q-statistic", "GRADE summary table", "publication-bias funnel plot", or any request to aggregate findings across multiple studies or evaluate evidence quality at the field level. For a structured three-layer literature scan inside an active research-spark project, use `landscape-scanner`; for peer-reviewing a single manuscript, use `scientific-review`.
---

# Evidence Synthesis

Systematic methods for aggregating and evaluating scientific evidence.

## Expert Agent

For complex reviews and meta-analyses, delegate to:

- **`research-expert`**: Unified specialist for Evidence Synthesis.
  - *Location*: `plugins/research-suite/agents/research-expert.md`
  - *Capabilities*: PRISMA workflows, bias assessment, and GRADE evaluation.

## Systematic Reviews (PRISMA)

### Workflow
1.  **Protocol Registration**: Define inclusion/exclusion criteria (PROSPERO).
2.  **Search Strategy**: Boolean logic across databases (PubMed, IEEE Xplore, arXiv).
3.  **Screening**: Title/Abstract screening followed by Full-text review.
4.  **Extraction**: Standardized data extraction forms.
5.  **Synthesis**: Qualitative or quantitative (meta-analysis) synthesis.

### PRISMA Checklist
- [ ] Rationale and objectives defined.
- [ ] Eligibility criteria specified.
- [ ] Information sources listed.
- [ ] Search strategy documented (reproducible).
- [ ] Selection process described.
- [ ] Data collection process defined.

## Meta-Analysis

### Effect Size Calculation
- **Continuous**: Cohen's d, Hedges' g.
- **Binary**: Odds Ratio (OR), Relative Risk (RR).

### Heterogeneity
- **Q-statistic**: Test for heterogeneity.
- **I² statistic**: Percentage of variation due to heterogeneity.
  - Low: < 25%
  - Moderate: 25-50%
  - High: > 50%

### Fixed vs Random Effects
- **Fixed Effects**: Assumes one true effect size.
- **Random Effects**: Assumes distribution of effect sizes (generally preferred).

## Evidence Grading (GRADE)

| Level | Definition |
|-------|------------|
| **High** | Very confident that the true effect lies close to the estimate. |
| **Moderate** | Moderately confident in the effect estimate. |
| **Low** | Confidence is limited; true effect may be substantially different. |
| **Very Low** | Very little confidence in the effect estimate. |

### Downgrading Factors
1.  **Risk of Bias**: Poor study design/execution.
2.  **Inconsistency**: Unexplained heterogeneity ($I^2$).
3.  **Indirectness**: Population/intervention differs from PICO.
4.  **Imprecision**: Wide confidence intervals.
5.  **Publication Bias**: Missing negative studies.

## Related skills

- `landscape-scanner` (research-spark Stage 2) — when the goal is a gap-finding literature scan inside an active research-spark project rather than a formal PRISMA review. Uses three-layer scan (foundational / recent / adjacent) + Reviewer 2 pass, not PRISMA flow diagrams.
- `research-quality-assessment` — when evaluating a *single* manuscript's rigor rather than aggregating across many.
- `scientific-review` — when producing a .docx peer-review deliverable for a specific journal submission.
