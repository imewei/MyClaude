---
name: statistical-analysis-fundamentals
version: "1.0.7"
maturity: "5-Expert"
specialization: Statistical Analysis
description: Comprehensive statistical analysis with scipy.stats, statsmodels, and PyMC3 including hypothesis testing, Bayesian methods, regression, experimental design, and causal inference. Use when conducting A/B tests, power analysis, or treatment effect estimation.
---

# Statistical Analysis Fundamentals

Systematic frameworks for hypothesis testing, experimental design, and causal inference.

---

## Test Selection Guide

| Scenario | Test | Python |
|----------|------|--------|
| Compare 2 means | t-test | `scipy.stats.ttest_ind` |
| Compare 3+ means | ANOVA | `scipy.stats.f_oneway` |
| Categorical association | Chi-square | `scipy.stats.chi2_contingency` |
| Non-parametric means | Mann-Whitney U | `scipy.stats.mannwhitneyu` |
| Regression | OLS/Logistic | `statsmodels`, `sklearn` |
| Bayesian inference | MCMC | `pymc3`, `numpyro` |
| Time series | ARIMA | `statsmodels.tsa` |
| Causal inference | DiD, PSM | `statsmodels`, `econml` |

---

## Hypothesis Testing Framework

```python
from scipy import stats
import numpy as np

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)

# Effect size (Cohen's d)
pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
effect_size = (group_a.mean() - group_b.mean()) / pooled_std

# Confidence interval
ci = stats.t.interval(0.95, len(group_a)-1, loc=group_a.mean(), scale=stats.sem(group_a))
```

---

## Sample Size Calculation

```python
from statsmodels.stats.power import tt_ind_solve_power

n_per_group = tt_ind_solve_power(
    effect_size=0.2,  # Small effect (Cohen's d)
    alpha=0.05,
    power=0.8
)
```

---

## Bayesian A/B Testing

```python
import pymc3 as pm

with pm.Model():
    p_a = pm.Beta('p_a', 1, 1)
    p_b = pm.Beta('p_b', 1, 1)
    obs_a = pm.Binomial('obs_a', n=trials_a, p=p_a, observed=conversions_a)
    obs_b = pm.Binomial('obs_b', n=trials_b, p=p_b, observed=conversions_b)
    delta = pm.Deterministic('delta', p_b - p_a)
    trace = pm.sample(2000)

prob_b_wins = (trace['delta'] > 0).mean()
```

---

## Causal Inference Methods

| Method | Use Case | Assumption |
|--------|----------|------------|
| Difference-in-Differences | Before/after with control | Parallel trends |
| Propensity Score Matching | Observational treatment | No unmeasured confounders |
| Instrumental Variables | Endogeneity present | Valid instrument |
| Regression Discontinuity | Threshold-based assignment | Continuity at cutoff |

---

## Multiple Testing Correction

```python
from statsmodels.stats.multitest import multipletests

reject, p_corrected, _, _ = multipletests(
    p_values, alpha=0.05, method='fdr_bh'  # Benjamini-Hochberg
)
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Pre-register hypotheses | Before data collection |
| Report effect sizes | Cohen's d, odds ratios |
| Include confidence intervals | 95% CI with point estimates |
| Check assumptions | Normality, homoscedasticity |
| Correct for multiple tests | Bonferroni or FDR |
| Distinguish statistical vs practical | Large n → small p doesn't mean important |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Underpowered studies | Power analysis before collecting data |
| P-hacking | Pre-registration, multiple testing correction |
| Ignoring effect size | Always report Cohen's d or equivalent |
| Assumption violations | Check residuals, use robust methods |
| Causal claims from correlation | Use appropriate causal methods |

---

## Checklist

- [ ] Hypothesis and alpha level pre-specified
- [ ] Sample size justified with power analysis
- [ ] Appropriate test selected for data type
- [ ] Assumptions checked and documented
- [ ] Effect size and CI reported
- [ ] Multiple testing corrected if applicable
- [ ] Statistical vs practical significance distinguished

---

**Version**: 1.0.5
