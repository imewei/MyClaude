---
name: research-quality-assessment
description: Evaluate scientific research quality against CONSORT/STROBE/PRISMA/MOOSE reporting guidelines, score methodology and statistical rigor, and detect red flags (p-hacking, HARKing, selective reporting, circular analysis). This skill should be used when the user asks to "assess this manuscript", "audit this preprint", "score the methodology", "check for p-hacking", "is this reproducible", "red-flag this grant proposal", "run a CONSORT / STROBE / PRISMA / MOOSE check", "evaluate statistical rigor", or wants a scored rubric on *existing* work without producing a journal-ready referee report. For a .docx peer-review deliverable, use `scientific-review`; for *designing* a new study before data collection, use `research-methodology`.
---

# Research Quality Assessment

Systematic framework for evaluating research quality *after* the work exists — manuscripts, grant proposals, preprints, or internal review. Design-phase planning belongs in `research-methodology`; journal peer-review reports with .docx output belong in `scientific-review`.

## Expert Agent

For research quality evaluation, methodology assessment, and publication readiness, delegate to:

- **`research-expert`**: Research methodology, quality assessment, and scientific rigor.
  - *Location*: `plugins/research-suite/agents/research-expert.md`

## Scope boundary

| Task | Skill |
|------|-------|
| **Score rigor / detect red flags** in existing work | research-quality-assessment ← *this skill* |
| **Design** a new experiment before data collection | research-methodology |
| **Write a peer-review report** (.docx) for a journal submission | scientific-review |
| **Systematic review or meta-analysis** | evidence-synthesis |

---

## Assessment Dimensions

| Dimension | Weight | Key Criteria |
|-----------|--------|--------------|
| Methodology | 20% | Design selection, controls, reproducibility |
| Experimental Design | 20% | Power analysis, randomization, blinding |
| Data Quality | 15% | Completeness, missing data handling |
| Statistical Rigor | 20% | Appropriate tests, effect sizes, corrections |
| Result Validity | 15% | Reproducibility, practical significance |
| Publication Readiness | 10% | Methods detail, data availability |

---

## Scoring Rubric

| Score | Level | Description |
|-------|-------|-------------|
| 9-10 | Exceptional | Exceeds all standards, exemplary |
| 7-8 | Strong | Meets standards, minor improvements |
| 5-6 | Adequate | Minimum standards, improvements needed |
| 3-4 | Weak | Significant issues, major revisions |
| 1-2 | Poor | Fundamental flaws, rejection likely |

---

## Reporting Guidelines

| Study Type | Guideline | Key Requirements |
|------------|-----------|------------------|
| Clinical Trials | CONSORT | Randomization, flow diagram, ITT |
| Observational | STROBE | Selection, confounding, bias |
| Systematic Reviews | PRISMA | Search strategy, selection, synthesis |
| Meta-Analysis | MOOSE | Heterogeneity, publication bias |

---

## Red Flags

| Issue | Detection |
|-------|-----------|
| Underpowered | Sample too small for effect size claimed |
| P-hacking | Multiple tests without correction |
| HARKing | Hypothesis after results known |
| Selective reporting | Cherry-picked outcomes |
| Circular analysis | Same data for discovery and validation |

---

## Statistical Adherence Check

When evaluating existing work, verify each element appears in the manuscript:

- [ ] Appropriate test reported for data type
- [ ] Test assumptions explicitly checked (normality, homoscedasticity, independence)
- [ ] Multiple-comparison correction applied where multiple tests were run
- [ ] Effect sizes reported alongside p-values, with confidence intervals
- [ ] Power analysis referenced (or retrospectively computable from reported numbers)
- [ ] Missing-data handling documented
- [ ] Sensitivity or robustness analyses included

For designing these elements into a new study (vs checking that an existing study included them), see `research-methodology`.

---

## Reproducibility Checklist

- [ ] Methods sufficiently detailed
- [ ] Data available or access described
- [ ] Analysis code/scripts available
- [ ] Materials accessible
- [ ] Protocol pre-registered (if applicable)

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Pre-registration | Register hypotheses before data collection |
| Blinding | Blind analysts to conditions |
| Internal replication | Include validation cohort |
| Transparency | Share data, code, materials |
| Reporting standards | Follow field-specific guidelines |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Missing power analysis | Calculate before data collection |
| No effect sizes | Report Cohen's d or equivalent |
| Ignoring assumptions | Check residuals, use robust methods |
| Causal overclaims | Match claims to study design |
| Poor documentation | Detailed methods, supplementary info |

---

## Related skills

- `scientific-review` — produces a journal-ready .docx peer-review deliverable (Six-Lens analysis, Confidential Comments to Editor) instead of an internal scoring rubric.
- `research-methodology` — when *designing* the study that would avoid these red flags in the first place.
- `evidence-synthesis` — when scoring the quality of a *corpus* of studies (PRISMA + GRADE) rather than a single one.
- `premortem-critique` (research-spark Stage 8) — for internal red-teaming of your own plan *before* data collection rather than auditing someone else's finished work.

## Checklist

- [ ] Research question clearly stated
- [ ] Design appropriate for question
- [ ] Sample size justified
- [ ] Appropriate statistical methods
- [ ] Effect sizes reported
- [ ] Limitations acknowledged
- [ ] Reproducibility materials provided
- [ ] Reporting guidelines followed
