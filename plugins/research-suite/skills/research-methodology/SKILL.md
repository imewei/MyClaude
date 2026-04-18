---
name: research-methodology
description: Design-phase guide for scientific research — experimental design, hypothesis formulation, power analysis, sample-size justification, ablation planning, and statistical-test selection *before* data collection. This skill should be used when the user asks to "run a power analysis", "justify this sample size", "pick a statistical test", "design an ablation study", "pre-register this experiment", "how many samples do I need", "is a t-test or Mann-Whitney right here", "plan a randomized design", or any design-phase methodology question asked before data is collected. For evaluating existing work, use `research-quality-assessment`; for writing up the finished study, use `scientific-communication`; for systematic reviews or meta-analysis, use `evidence-synthesis`; for reproducing a specific paper, use `research-paper-implementation`. Inside an active research-spark project, Stage 7's `experiment-designer` handles DoE, instrument capability mapping, and pre-registered metrics under stricter artifact conventions.
---

# Research Methodology

Guide for *designing* experiments and studies with high statistical rigor and reproducibility, before data collection begins.

## Expert Agent

For systematic reviews, rigorous experimental design, and publication-quality reporting, delegate to:

- **`research-expert`**: Specialist for research methodology, literature synthesis, and scientific writing.
  - *Location*: `plugins/research-suite/agents/research-expert.md`

## Scope boundary

This skill covers the *design* phase — what to plan before running the experiment. Sister skills in the `research-practice` hub cover adjacent phases:

| Phase | Skill |
|-------|-------|
| **Design** (hypothesis, power, DoE) | research-methodology ← *this skill* |
| **Evaluate** (rigor, reproducibility, red flags) | research-quality-assessment |
| **Write up** (IMRaD, reports) | scientific-communication |
| **Synthesize** (PRISMA, meta-analysis) | evidence-synthesis |
| **Reproduce a paper** | research-paper-implementation |

The research-spark stack (`research-spark` → 8 stages) is a structured pipeline that embeds these phases; use it when you want artifact-gated handoffs between stages rather than free-form methodology guidance.

## 1. Experimental Design

### Core Principles
- **Power Analysis**: Conduct power analysis (target 0.80) to justify sample sizes *before* data collection. Post-hoc power is not informative.
- **Ablation Studies**: Systematically remove components to quantify their individual contributions to the overall result.
- **Baseline Comparisons**: Compare against multiple state-of-the-art baselines to ensure fair and rigorous evaluation.
- **Pre-registration**: Register hypotheses, analysis plan, and success criteria on OSF or PROSPERO before looking at data. Downstream this blocks HARKing and p-hacking without needing discipline.

### Design Checklist
- [ ] Research question clearly stated as a testable hypothesis (null + alternative).
- [ ] Sample size justified via formal power analysis for the target effect size.
- [ ] Independent, dependent, and control variables explicitly listed.
- [ ] Randomization and (where feasible) blinding plan specified.
- [ ] Systematic parameter-space exploration plan (grid, random, or Bayesian optimization).
- [ ] Interaction effects explicitly modeled, not assumed additive.
- [ ] Pre-registration filed where field conventions permit.

## 2. Statistical Test Planning

### Test Selection

Choose the test *before* seeing the data. Adjusting the test to the data distribution after the fact is a form of p-hacking.

- **Continuous outcome, two groups**: t-test if normality + equal variance hold; Mann-Whitney U otherwise.
- **Continuous outcome, >2 groups**: ANOVA (+ Tukey HSD post-hoc) if assumptions hold; Kruskal-Wallis otherwise.
- **Categorical outcome**: Chi-squared if expected counts ≥ 5; Fisher's exact otherwise.
- **Paired measurements**: paired t-test or Wilcoxon signed-rank.
- **Correlation**: Pearson if linear + normal; Spearman or Kendall for monotonic but nonlinear.
- **Regression**: linear/logistic/mixed-effects/survival depending on outcome type and data structure.

### Assumption Checks

Plan in the methods section how assumptions will be verified:

- Normality: Shapiro-Wilk (small n), Q-Q plots (visual), or robust methods that don't require it.
- Homoscedasticity: Levene's test, residual plots.
- Independence: ensure experimental units are independent; account for clustering with mixed-effects models.

### Multiple-Comparison Correction

When the design involves multiple tests, commit to the correction scheme in advance:

- **Bonferroni**: conservative; family-wise error rate.
- **Benjamini-Hochberg (FDR)**: less conservative; controls false discovery rate; appropriate when many tests are exploratory.
- Commit in the pre-registration which outcomes are primary (no correction) vs secondary (corrected).

## 3. Effect Size and Uncertainty

Report effect sizes *and* confidence intervals, not p-values alone.

- **Standardized effect sizes**: Cohen's d (two groups), η² (ANOVA), R² (regression), odds ratio (logistic).
- **95% confidence intervals** on all primary estimates.
- **Error bars**: clarify SD vs SEM vs CI explicitly in figure legends — these are routinely conflated.

A small p-value with a tiny effect size is a statistical fact, not a scientific finding. The effect size carries the practical meaning.

## Related skills

- `research-quality-assessment` — evaluating existing work against CONSORT/STROBE/PRISMA.
- `scientific-communication` — writing the methods section once the design is locked in.
- `evidence-synthesis` — if the study is itself a systematic review or meta-analysis.
- `experiment-designer` (research-spark Stage 7) — when working inside an active research-spark project, use this instead: it produces an instrument-capability map (3× margin rule), a DoE matrix, a pre-registered metrics file, and a risk register as canonical artifacts.
- `scientific-review` — for peer-reviewing someone else's completed study rather than designing your own.

## Checklist

- [ ] Hypothesis stated with null and alternative.
- [ ] Sample size justified by power analysis for the target effect.
- [ ] Statistical test chosen and committed to *before* data collection.
- [ ] Multiple-comparison correction scheme specified for secondary tests.
- [ ] Effect-size measure selected (not just p-value).
- [ ] Pre-registration filed where applicable.
