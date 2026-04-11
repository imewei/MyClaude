---
name: extreme-value-statistics
description: Fit extreme-value distributions to tail data with block-maxima GEV (Gumbel / Fréchet / Weibull) and peaks-over-threshold GPD models. Covers maximum-likelihood, L-moments, Hill, Pickands, and moment estimators for the tail index; return-level and return-period computation; threshold selection via mean-residual-life and parameter-stability plots; non-stationary EVT with covariate-dependent parameters; and the boundary between power-law (SOC) analyses and heavy-tail EVT analyses. Use when fitting tail distributions to extreme magnitudes (floods, gusts, earthquake magnitudes, financial drawdowns, material failure stresses, avalanche sizes in the heavy-tail regime), estimating return levels, computing exceedance probabilities, or quantifying tail index. Use proactively when the user mentions extreme value, EVT, GEV, GPD, generalized Pareto, generalized extreme value, Gumbel, Fréchet, Weibull, block maxima, peaks over threshold, POT, return level, return period, tail index, Hill estimator, Pickands, exceedance, heavy tail, Pareto tail, or `pyextremes`.
---

# Extreme Value Statistics

Fit and interpret the tails of a distribution when the rare / extreme part of a dataset is what matters. EVT is the canonical machinery for rainfall extremes, flood return levels, stress-failure thresholds, financial drawdowns, earthquake-magnitude distributions, and the large-amplitude end of avalanche / crackling-noise spectra.

## Expert Agents

- **`statistical-physicist`** — asymptotic theory (Fisher-Tippett-Gnedenko, Pickands-Balkema-de Haan), tail-index interpretation, and the bridge to large-deviation theory.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
- **`ml-expert`** (secondary) — applied Python workflow, `pyextremes` / `POT` / `scipy.stats` tooling, and feature engineering for non-stationary covariates.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`

---

## When to reach for EVT vs. SOC / rare-event sampling

| Situation | Right tool |
|-----------|------------|
| You already have many observed extremes (annual maxima, peaks above a threshold) and want to fit a distribution to them | **EVT** (this skill) |
| The process is critical / self-organized and you want to establish a power-law exponent with model-independent confidence intervals | `rare-events-sampling` (SOC, avalanche size distributions) |
| Direct simulation never reaches the tail and you need to generate extremes first | `rare-events-sampling` (FFS / WE / TPS) |
| You have a point-process view (when do extremes occur?) rather than a magnitude view | `point-processes` (Hawkes, Cox, inhomogeneous Poisson) |

The crucial distinction: **EVT fits a distribution to observed magnitudes**, whereas SOC / rare-events-sampling are about *producing* or *exponent-identifying* the tail. The two skills are complementary — fit the tail with EVT, interpret the exponent with SOC theory.

---

## Two paradigms

### Block-maxima (GEV)

Partition the record into fixed blocks (year, decade, etc.), keep only the maximum in each block, fit a **Generalized Extreme Value** distribution:

$$F(x; \mu, \sigma, \xi) = \exp\\Big\\{ -\\big[1 + \xi (x-\mu)/\sigma\\big]_+^{-1/\xi}\\Big\\}$$

The shape parameter $\xi$ classifies the tail:

| $\xi$ | Type | Tail behavior | Examples |
|------|------|---------------|----------|
| $\xi = 0$ | Gumbel | Exponential decay | Thermal noise maxima, some hydrological series |
| $\xi > 0$ | Fréchet | Power-law, unbounded | Financial losses, earthquake magnitudes |
| $\xi < 0$ | Weibull | Bounded support | Material strength, wind gusts with physical cap |

### Peaks-over-threshold (GPD)

Pick a high threshold $u$, keep all exceedances $y_i = x_i - u$ where $x_i > u$, fit a **Generalized Pareto** distribution:

$$F_u(y; \sigma_u, \xi) = 1 - \\big[1 + \xi y / \sigma_u\\big]_+^{-1/\xi}$$

POT is nearly always more data-efficient than block-maxima — blocks waste the second-, third-, ...-largest observations.

---

## Python workflow

### scipy.stats (low-level)

```python
from scipy.stats import genextreme, genpareto

# Block-maxima GEV (note scipy's sign convention: c = -xi)
c, loc, scale = genextreme.fit(block_maxima)
xi = -c                       # convert to standard EVT convention

# Peaks-over-threshold GPD
threshold = np.quantile(x, 0.95)
excesses = x[x > threshold] - threshold
c, _, scale = genpareto.fit(excesses, floc=0)
xi = c                        # same sign convention as EVT
```

### pyextremes (high-level)

```python
import pyextremes

model = pyextremes.EVA(data=series)
model.get_extremes(method="BM", block_size="365.2425D")  # or "POT"
model.fit_model(distribution="genextreme")
model.plot_diagnostic(alpha=0.95)

# Return level for 100-year return period
return_level = model.get_summary(return_period=[10, 50, 100])
```

### POT (optional)

The `POT` package specializes in peaks-over-threshold workflows with automated threshold selection via mean-residual-life and parameter-stability plots.

---

## Estimators and when to use them

| Estimator | Best for | Caveats |
|-----------|----------|---------|
| **Maximum likelihood** | Default; asymptotically efficient | Fails for $\xi < -0.5$; small-sample bias |
| **L-moments** | Small samples, robust to outliers | No uncertainty quantification out of the box |
| **Probability-weighted moments (PWM)** | Hydrology tradition | Equivalent to L-moments for GEV |
| **Hill estimator** | Pure Fréchet tail, $\xi > 0$ only | Threshold-sensitive; plot Hill plot and pick stable region |
| **Pickands estimator** | Sign-agnostic tail index | Higher variance than Hill |
| **Moment estimator (Dekkers-Einmahl-de Haan)** | Robust to $\xi$ sign | Middle ground between Hill and Pickands |

**Practical rule:** fit with MLE first, cross-check with L-moments. For the tail index alone, look at the Hill plot *together* with the Pickands and moment estimators — agreement across methods is the check that the threshold is high enough.

---

## Return levels

The $T$-period return level $z_T$ is the value expected to be exceeded once every $T$ blocks:

$$z_T = \mu + (\sigma / \xi) \\big[(-\log(1 - 1/T))^{-\xi} - 1\\big] \\quad (\xi \\ne 0)$$

Always report a confidence interval — profile likelihood is preferred over the delta method because return-level likelihoods are asymmetric in the tail.

---

## Threshold selection (POT)

Two visual diagnostics drive threshold choice:

1. **Mean residual life plot** — plot $\\mathbb{E}[X - u \\mid X > u]$ against $u$. The GPD is valid where the plot is approximately linear.
2. **Parameter stability plot** — refit the GPD at a grid of thresholds and plot $\\xi$ and $\\sigma^*$ against $u$. Pick the lowest $u$ where both are stable.

Automated selection (e.g., Northrop-Coleman score) is available in `pyextremes` but should be cross-checked against the visual diagnostics.

---

## Non-stationary EVT

When the process is non-stationary (climate trends, macroeconomic regime changes), let the GEV / GPD parameters depend on covariates:

$$\mu(t) = \mu_0 + \\beta_1 \\cdot t, \\quad \\sigma(t) = \exp(\\sigma_0 + \\beta_2 \\cdot t)$$

Always keep $\\sigma > 0$ through a log-link; fit by joint MLE. `pyextremes` and the R `ismev` package both support this pattern. When the covariate structure is uncertain, route to `bayesian-inference` and use a NumPyro / PyMC model with a prior on the trend coefficient.

---

## Connection to SOC, rare events, and time series

- **SOC / power-law tails** — if the process is self-organized critical, the magnitude distribution is a pure power law and a Fréchet ($\xi > 0$) fit will recover the exponent. Cross-check with the Clauset-Shalizi-Newman MLE for power laws (`powerlaw` Python package) before committing to EVT. See `rare-events-sampling`.
- **Rare-event rate estimation** — combining EVT with importance-sampling estimators (e.g., FFS + POT GPD on the flux histogram) is a standard pattern. See `rare-events-sampling`.
- **Regular-grid time series** — for stationarity / ARIMA / forecasting see `time-series-analysis`; for event-time rates see `point-processes`.
- **Large-deviation theory** — the Fisher-Tippett-Gnedenko theorem is the statistical shadow of the large-deviation rate function; `statistical-physicist` agent can connect the two.

---

## Composition with neighboring skills

- **Rare events sampling** — produce the tail samples; EVT fits a distribution to them. See `rare-events-sampling`.
- **Statistical physics hub** — parent routing for SOC, avalanches, and non-equilibrium physics. See `statistical-physics-hub`.
- **Time series analysis** — stationarity tests before applying block-maxima EVT. See `time-series-analysis`.
- **Point processes** — temporal clustering of exceedance events (above a POT threshold). See `point-processes`.
- **Bayesian inference hub** — Bayesian EVT with NumPyro / PyMC priors. See `bayesian-inference`.

---

## Checklist

- [ ] Verified the data are (approximately) independent and identically distributed over blocks / exceedances; declustered where needed
- [ ] Chose between block-maxima and POT based on data volume and block homogeneity
- [ ] Cross-checked threshold with mean-residual-life plot **and** parameter-stability plot (POT)
- [ ] Fit with MLE and cross-checked with L-moments / PWM
- [ ] Looked at Hill, Pickands, and moment estimators together for the tail index
- [ ] Reported return levels with profile-likelihood confidence intervals, not delta method
- [ ] Checked non-stationarity before pooling: stationarity test, fit trend model if needed
- [ ] Distinguished EVT ($\xi$) from a pure SOC power-law exponent before interpretation
- [ ] Validated the fit with a QQ / PP plot, a return-level plot, and a residual-density plot
