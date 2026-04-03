---
name: time-series-analysis
description: "Analyze and forecast time series data with statsmodels, Prophet, and neural methods including ARIMA/SARIMA, decomposition, anomaly detection, change point detection, and deep learning forecasting (N-BEATS, TFT). Use when building forecasting models, detecting anomalies in temporal data, or analyzing seasonal patterns."
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
