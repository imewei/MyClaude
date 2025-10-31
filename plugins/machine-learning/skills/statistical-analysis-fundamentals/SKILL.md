---
name: statistical-analysis-fundamentals
description: Comprehensive statistical analysis workflows using scipy.stats, statsmodels, and PyMC3 including hypothesis testing (t-tests, ANOVA, chi-square), Bayesian methods (MCMC, posterior inference), regression analysis (OLS, logistic, time series), experimental design, A/B testing, power analysis, and causal inference (DiD, propensity score matching). Use when writing or editing statistical analysis scripts (`.py`), Jupyter notebooks (`.ipynb`) for experiments, or A/B test evaluation code. Apply this skill when conducting hypothesis tests (t-test, ANOVA, Mann-Whitney U), calculating effect sizes and confidence intervals, performing regression analysis with assumption checking, implementing Bayesian A/B tests with posterior distributions, designing experiments with sample size calculations, analyzing A/B test results with statistical significance, implementing causal inference methods (difference-in-differences, propensity score matching), performing time series analysis (ARIMA, seasonal decomposition), handling multiple testing with Bonferroni or FDR corrections, or validating statistical assumptions (normality, homoscedasticity, independence).
---

# Statistical Analysis Fundamentals

Systematic frameworks for statistical analysis, hypothesis testing, experimental design, and causal inference in data science applications.

## When to Use

- Conducting hypothesis testing and statistical significance analysis
- Designing A/B tests or multivariate experiments
- Performing regression analysis (linear, logistic, time series)
- Applying Bayesian methods for probabilistic modeling
- Causal inference and treatment effect estimation
- Sample size and power analysis for experiments
- Validating assumptions and checking model diagnostics

## Core Statistical Methods

### 1. Hypothesis Testing Framework

Follow this systematic approach:
```
1. State null (H₀) and alternative (H₁) hypotheses
2. Choose significance level (α, typically 0.05)
3. Select appropriate test statistic
4. Calculate p-value
5. Compute effect size (Cohen's d, odds ratio)
6. Report confidence intervals
7. Make decision with business context
```

**Common Tests:**
- t-test: Compare means between groups
- ANOVA: Compare means across 3+ groups
- Chi-square: Test independence in categorical data
- Mann-Whitney U: Non-parametric alternative
- Kolmogorov-Smirnov: Test distribution equality

**Python Template:**
```python
from scipy import stats
import numpy as np

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)

# Effect size (Cohen's d)
pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
effect_size = (group_a.mean() - group_b.mean()) / pooled_std

# Confidence interval
ci = stats.t.interval(0.95, len(group_a)-1,
                      loc=group_a.mean(),
                      scale=stats.sem(group_a))

print(f"p-value: {p_value:.4f}, Effect size: {effect_size:.3f}")
```

### 2. Regression Analysis

**Linear Regression with Diagnostics:**
```python
import statsmodels.api as sm

# Fit model
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())  # R², coefficients, p-values

# Check assumptions:
# 1. Linearity: Residual plot
# 2. Independence: Durbin-Watson ~2
# 3. Homoscedasticity: Breusch-Pagan test
# 4. Normality: Shapiro-Wilk, Q-Q plot
```

**Logistic Regression:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Interpret coefficients as odds ratios
odds_ratios = np.exp(model.coef_[0])
```

### 3. Bayesian Methods

**Bayesian A/B Testing:**
```python
import pymc3 as pm

with pm.Model() as model:
    # Prior beliefs (uniform)
    p_a = pm.Beta('p_a', alpha=1, beta=1)
    p_b = pm.Beta('p_b', alpha=1, beta=1)

    # Observed data
    obs_a = pm.Binomial('obs_a', n=trials_a, p=p_a,
                        observed=conversions_a)
    obs_b = pm.Binomial('obs_b', n=trials_b, p=p_b,
                        observed=conversions_b)

    # Treatment effect
    delta = pm.Deterministic('delta', p_b - p_a)

    trace = pm.sample(2000)

# Probability B > A
prob_b_wins = (trace['delta'] > 0).mean()
```

### 4. Experimental Design

**Sample Size Calculation:**
```python
from statsmodels.stats.power import tt_ind_solve_power

# Inputs
effect_size = 0.2  # Small effect (Cohen's d)
alpha = 0.05
power = 0.8

n_per_group = tt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power
)
```

**Stratified Randomization:**
```python
# Ensure balance across important covariates
for stratum in df['stratum'].unique():
    mask = df['stratum'] == stratum
    treatment = df[mask].sample(frac=0.5).index
    df.loc[treatment, 'group'] = 'treatment'
```

### 5. Causal Inference

**Difference-in-Differences:**
```python
import statsmodels.formula.api as smf

# Y = β₀ + β₁*Treatment + β₂*Post + β₃*(Treatment*Post)
# β₃ is the causal effect estimate

model = smf.ols('outcome ~ treatment * post', data=df).fit()
did_estimate = model.params['treatment:post']
```

**Propensity Score Matching:**
```python
from sklearn.linear_model import LogisticRegression

# Step 1: Estimate propensity scores
ps_model = LogisticRegression()
ps_model.fit(X_covariates, treatment)
propensity_scores = ps_model.predict_proba(X_covariates)[:, 1]

# Step 2: Match treated to control (1:1 nearest neighbor)
# Step 3: Estimate average treatment effect on matched sample
```

### 6. Time Series Analysis

**ARIMA Modeling:**
```python
from statsmodels.tsa.arima.model import ARIMA

# Determine order (p,d,q) using ACF/PACF plots
model = ARIMA(ts_data, order=(p, d, q))
fitted = model.fit()

# Forecast
forecast = fitted.forecast(steps=30)
```

## Best Practices

### Multiple Testing Correction

When running multiple hypothesis tests:
```python
from statsmodels.stats.multitest import multipletests

# Benjamini-Hochberg (FDR) correction
reject, p_corrected, _, _ = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)
```

### Report Both Statistical and Practical Significance

```python
# Statistical significance
if p_value < 0.05:
    status = "statistically significant"

# Practical significance (effect size thresholds)
if abs(cohen_d) > 0.5:
    impact = "medium to large practical effect"
```

### Assumption Checking Template

```python
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan

# Normality
stat, p = shapiro(residuals)
print(f"Normality test p-value: {p:.4f}")

# Homoscedasticity
lm_stat, lm_p, _, _ = het_breuschpagan(residuals, X)
print(f"Heteroscedasticity test p-value: {lm_p:.4f}")
```

## Quick Reference

| Task | Method | Python Library |
|------|--------|----------------|
| Compare 2 means | t-test | scipy.stats.ttest_ind |
| Compare 3+ means | ANOVA | scipy.stats.f_oneway |
| Categorical association | Chi-square | scipy.stats.chi2_contingency |
| Regression | OLS/Logistic | statsmodels, sklearn |
| Bayesian inference | MCMC | pymc3, stan |
| Time series | ARIMA | statsmodels.tsa |
| Causal inference | DiD, PSM | statsmodels, econml |

---

*Apply rigorous statistical methods to validate hypotheses, design experiments, and draw causal inferences from data.*
