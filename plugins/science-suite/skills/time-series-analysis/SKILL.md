---
name: time-series-analysis
description: "Analyze and forecast time series data with statsmodels, Prophet, and neural methods including ARIMA/SARIMA, GARCH, STL decomposition, stationarity testing (ADF, KPSS, Phillips-Perron), change-point detection (PELT, BinSeg, dynamic programming via `ruptures`), anomaly detection, and deep learning forecasting (N-BEATS, TFT). Use when building forecasting models, testing stationarity, detecting change points or anomalies in regularly-sampled temporal data, or analyzing seasonal patterns. For irregular event timestamps (Hawkes, Cox, renewal), route to point-processes."
---

# Time Series Analysis

Forecast, decompose, and detect anomalies in temporal data.

## Expert Agent

For ML pipeline design and model selection for time series tasks, delegate to the expert agent:

- **`ml-expert`**: Classical and applied ML specialist for feature engineering, model selection, and evaluation.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: Pipeline design, cross-validation strategies, model comparison, hyperparameter optimization.

## Time Series Decomposition

```python
import pandas as pd
from statsmodels.tsa.seasonal import STL

def decompose_series(series: pd.Series, period: int = 12) -> dict:
    """STL decomposition into trend, seasonal, and residual."""
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "strength_seasonal": 1 - result.resid.var() / (result.seasonal + result.resid).var(),
    }
```

## ARIMA / SARIMA

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
        "critical_values": result[4],
    }

def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
) -> dict:
    """Fit SARIMA model and return diagnostics."""
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return {
        "aic": results.aic,
        "bic": results.bic,
        "params": results.params.to_dict(),
        "forecast": results.get_forecast(steps=12),
        "summary": results.summary(),
    }
```

## Prophet Forecasting

```python
from prophet import Prophet

def prophet_forecast(
    df: pd.DataFrame,
    periods: int = 30,
    freq: str = "D",
    seasonality_mode: str = "multiplicative",
) -> pd.DataFrame:
    """Fit Prophet and generate forecast with uncertainty intervals."""
    # df must have columns: ds (datetime), y (value)
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
```

## Anomaly Detection

```python
import numpy as np

def detect_anomalies_zscore(
    series: pd.Series, window: int = 30, threshold: float = 3.0
) -> pd.Series:
    """Rolling Z-score anomaly detection."""
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    z_scores = (series - rolling_mean) / rolling_std
    return z_scores.abs() > threshold

```

## Feature Engineering for Time Series

```python
def create_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    """Create lagged features for time series ML."""
    result = df.copy()
    for lag in lags:
        result[f"{target}_lag_{lag}"] = result[target].shift(lag)
    return result

def create_rolling_features(
    df: pd.DataFrame, target: str, windows: list[int]
) -> pd.DataFrame:
    """Create rolling statistics features."""
    result = df.copy()
    for w in windows:
        result[f"{target}_rolling_mean_{w}"] = result[target].rolling(w).mean()
        result[f"{target}_rolling_std_{w}"] = result[target].rolling(w).std()
    return result
```

## Neural Forecasting (N-BEATS)

```python
# Using pytorch-forecasting or neuralforecast
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, TFT

def neural_forecast(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Train N-BEATS and TFT models, return ensemble forecast."""
    models = [
        NBEATS(h=horizon, input_size=2 * horizon, max_steps=500),
        TFT(h=horizon, input_size=2 * horizon, max_steps=500),
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=df)  # df: columns [unique_id, ds, y]
    return nf.predict()
```

## Nonlinear & Information-Theoretic Time Series

For nonlinear dynamics, causality, and information flow — complementing the linear/ML methods above:

> **Julia-first codebase?** All the Python packages below can be called from Julia via `PythonCall.jl` — see the handoff pattern in `chaos-attractors` for the canonical `pyimport` idiom, a concrete `nolds.lyap_r` example, and caveats (GIL, array marshaling, PyCall.jl vs PythonCall.jl).

| Package | Role | Key API |
|---------|------|---------|
| **`arch`** | GARCH / ARCH / EGARCH / FIGARCH; ADF / DF-GLS / PhillipsPerron unit roots; Engle-Granger / Phillips-Ouliaris cointegration; stationary / moving-block bootstraps | `arch_model`, `ADF`, `engle_granger`, `IIDBootstrap`, `StationaryBootstrap` |
| **`statsmodels.tsa`** | State-space SARIMAX with missing-data handling, VAR / VARMAX, UnobservedComponents, DynamicFactor, Markov switching | `SARIMAX`, `VAR`, `VARMAX`, `UnobservedComponents`, `MarkovRegression` |
| **`pyEDM`** | Empirical dynamic modeling: simplex projection, S-map (nonlinearity θ), convergent cross mapping for weak causality | `Simplex`, `SMap`, `CCM`, `EmbedDimension` |
| **`IDTxl`** | Greedy multivariate transfer entropy, active information storage, partial information decomposition (KSG/Kraskov + discrete estimators) | `MultivariateTE`, `BivariateTE`, `ActiveInformationStorage`, `PartialInformationDecomposition` |
| **`nolds`** | Largest Lyapunov (Rosenstein), full spectrum (Eckmann), Hurst, DFA, sample entropy, correlation dimension | `lyap_r`, `lyap_e`, `hurst_rs`, `dfa`, `sampen`, `corr_dim` |
| **`antropy`** | Permutation / sample / spectral / SVD entropy, Higuchi / Katz fractal dimension, DFA, Lempel-Ziv complexity, Hjorth parameters — EEG-style feature set | `perm_entropy`, `higuchi_fd`, `detrended_fluctuation`, `lziv_complexity` |
| **`EntropyHub`** | 40+ entropy estimators with multiscale (coarse / composite / refined), cross-entropy, multivariate multiscale, bidimensional entropy | `SampEn`, `PermEn`, `DispEn`, `MvMSEn` |
| **`ewstools`** | Rolling variance / AC(1) / skewness as critical-slowing-down indicators, Kendall tau significance, deep-learning bifurcation classifiers | `TimeSeries`, `compute_var`, `compute_auto`, `compute_ktau`, `apply_classifier` |

### Data assimilation / state estimation

| Package | Role |
|---------|------|
| **`filterpy`** | Linear Kalman, EKF, UKF (Julier / Merwe sigma points), IMM, particle filter, H-infinity; `batch_filter` for offline sequences. NumPy, CPU. |
| **`DAPPER`** | Benchmark harness for ensemble DA: stochastic / deterministic EnKF, ETKF, LETKF, iEnKS, PartFilt, 3D / 4D-Var; built-in toy models (Lorenz 63 / 84 / 96, QG); averaged diagnostics with confidence intervals. NumPy, CPU. |
| **`dynamax`** | **JAX-native** state-space + HMM library (Kevin Murphy / Probabilistic ML group). `GaussianHMM`, `LinearGaussianSSM`, `NonlinearGaussianSSM`, `CategoricalHMM`, `PoissonHMM`, `AutoregressiveHMM`; Kalman filter / smoother, EKF, UKF, particle filter; `fit_em` / stochastic gradient / fully-Bayesian HMC parameter learning; Viterbi decoding; parallel-associative scan → **O(log T) on GPU / TPU** for long sequences. Integrates with NumPyro — the `dynamax` log-prob is a JAX callable, so it composes inside a `numpyro.sample` block for Bayesian hyperparameter inference. |

```python
# dynamax: LGSSM with Kalman smoother + EM
import jax.random as jr
from dynamax.linear_gaussian_ssm import LinearGaussianSSM

state_dim, obs_dim = 2, 2
ssm = LinearGaussianSSM(state_dim=state_dim, emission_dim=obs_dim)
params, props = ssm.initialize(jr.PRNGKey(0))           # random init
params, lps  = ssm.fit_em(params, props, emissions)     # EM parameter learning
smoothed     = ssm.smoother(params, emissions)          # RTS smoother
```

> **Framework selection**: use `filterpy` for textbook Kalman / EKF / UKF / IMM on a single machine with NumPy arrays. Use `DAPPER` for research benchmarks on ensemble-DA methods with ready-made Lorenz / QG toy models. Use **`dynamax`** when the state-space inference is the hot path inside a JAX pipeline — it's the only JAX-native option, so the choice is made for you if the surrounding code already runs on GPU / TPU. For fully-Bayesian state-space inference, compose `dynamax` log-probs with a NumPyro model (see `numpyro-core-mastery`).

```python
# arch: GARCH volatility + unit root test
from arch import arch_model
from arch.unitroot import ADF
am = arch_model(returns, vol="Garch", p=1, q=1, dist="skewt")
res = am.fit(disp="off")
adf = ADF(returns).summary()

# pyEDM: convergent cross mapping to detect causality
from pyEDM import CCM
df = CCM(dataFrame=data, E=3, Tp=0, columns="X", target="Y",
         libSizes="10 70 10", sample=100)
```

> **Install notes**: `EntropyHub`, `ewstools`, and `IDTxl` have active repos but out-of-date or absent PyPI wheels — prefer `pip install git+https://github.com/<org>/<repo>.git` for current features.

## Point processes & self-exciting event data

When the data is irregular event *timestamps* rather than a regular grid (earthquake aftershocks, trade arrivals, neuron spikes, social-media cascades), ARMA/GARCH don't apply. Reach for Hawkes / multivariate Hawkes / Bayesian HSGP-background Hawkes / Julia `PointProcesses.jl`. See the dedicated **[point-processes](../point-processes/SKILL.md)** skill for the `tick` reference stack, NumPyro + HSGP Bayesian pattern, Julia ecosystem, and rescaled-residuals diagnostics.

## Evaluation Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| MAE | mean(\|y - y_hat\|) | Robust to outliers |
| RMSE | sqrt(mean((y - y_hat)^2)) | Penalizes large errors |
| MAPE | mean(\|y - y_hat\| / \|y\|) | Scale-independent |
| sMAPE | mean(2\|y - y_hat\| / (\|y\| + \|y_hat\|)) | Symmetric percentage |

## Forecasting Checklist

- [ ] Check stationarity (ADF test, KPSS test)
- [ ] Examine ACF/PACF plots for model order selection
- [ ] Use time-based train/test split (never random split)
- [ ] Compare naive baseline (seasonal naive, last value)
- [ ] Cross-validate with expanding or sliding window
- [ ] Report prediction intervals, not just point forecasts
- [ ] Test residuals for autocorrelation (Ljung-Box test)
