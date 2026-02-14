---
name: research-methodology
version: "2.2.1"
description: Systematic framework for scientific research, covering experimental design, statistical rigor, quality assessment, and publication readiness.
---

# Research Methodology

Expert guide for designing, executing, and evaluating scientific research with high rigor and reproducibility.

## Expert Agent

For systematic reviews, rigorous experimental design, and publication-quality reporting, delegate to the expert agent:

- **`research-expert`**: Unified specialist for Research Methodology and Evidence Synthesis.
  - *Location*: `plugins/science-suite/agents/research-expert.md`
  - *Capabilities*: Systematic literature reviews (PRISMA), meta-analysis, evidence grading (GRADE), and technical writing.

## Core Skills

### [Evidence Synthesis](./evidence-synthesis/SKILL.md)
Systematic reviews, meta-analyses, and GRADE evidence evaluation.

### [Scientific Communication](./scientific-communication/SKILL.md)
IMRaD structuring, technical writing principles, and reporting standards.

### [Research Quality Assessment](./research-quality-assessment/SKILL.md)
Evaluating rigor, reproducibility, and statistical validity.

## 1. Experimental Design

### Core Principles
- **Power Analysis**: Conduct power analysis (target 0.80) to justify sample sizes before data collection.
- **Ablation Studies**: Systematically remove components to quantify their individual contributions to the overall result.
- **Baseline Comparisons**: Compare against multiple state-of-the-art baselines to ensure fair and rigorous evaluation.

### Checklist
- [ ] Research question clearly stated.
- [ ] Sample size justified via power analysis.
- [ ] Systematic parameter space exploration (grid or random search).
- [ ] Interaction effects considered and tested.

## 2. Statistical Rigor

### Test Selection & Assumptions
- Verify assumptions (normality, homoscedasticity) before selecting parametric tests.
- Use non-parametric alternatives (e.g., Mann-Whitney U, Kruskal-Wallis) when assumptions are violated.
- Apply multiple testing corrections (Bonferroni, FDR) when conducting multiple comparisons.

### Uncertainty Quantification
- Always report 95% confidence intervals and effect sizes (Cohen's d, $R^2$).
- Include error bars on all plots and clarify if they represent standard deviation (SD) or standard error (SE).

## 3. Quality Assessment & Reproducibility

### Reporting Guidelines
- **Clinical/Experimental**: Follow CONSORT or STROBE guidelines.
- **Systematic Reviews**: Follow PRISMA.
- **Transparency**: Share data, analysis scripts, and materials in public repositories.

### Red Flags to Avoid
- **P-hacking**: Conducting multiple tests without correction to find significant results.
- **HARKing**: Hypothesizing after results are known.
- **Circular Analysis**: Using the same data for both discovery and validation.

## 4. Publication Readiness

- [ ] Methods described with sufficient detail for replication.
- [ ] Data and code availability statements included.
- [ ] Limitations and potential biases acknowledged.
- [ ] Figure formats and DPI meet journal standards.
