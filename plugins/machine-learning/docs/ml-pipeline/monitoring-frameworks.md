# Monitoring Frameworks Guide

Comprehensive guide to ML model monitoring, drift detection, and observability for production systems.

## Table of Contents
- [Model Performance Monitoring](#model-performance-monitoring)
- [Data Drift Detection](#data-drift-detection)
- [Concept Drift Detection](#concept-drift-detection)
- [System Observability](#system-observability)
- [Cost Tracking](#cost-tracking--optimization)

---

## Model Performance Monitoring

### Key Metrics to Track

**Prediction Quality Metrics**:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Mean Absolute Error (MAE), RMSE
- Custom business metrics

**Operational Metrics**:
- Latency (p50, p95, p99)
- Throughput (predictions/second)
- Error rate
- Resource utilization (CPU, memory, GPU)

**Implementation**:
```python
from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Define metrics
predictions_total = Counter('ml_predictions_total', 'Total predictions', ['model_version', 'status'])
prediction_latency = Histogram('ml_prediction_latency_seconds', 'Prediction latency', buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
model_accuracy = Gauge('ml_model_accuracy', 'Rolling window accuracy')
prediction_confidence = Summary('ml_prediction_confidence', 'Prediction confidence scores')

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        prediction = model.predict(request.features)
        confidence = model.predict_proba(request.features).max()

        # Record metrics
        predictions_total.labels(model_version="v2.1.3", status="success").inc()
        prediction_latency.observe(time.time() - start_time)
        prediction_confidence.observe(confidence)

        # Log low-confidence predictions for review
        if confidence < 0.7:
            logger.warning(f"Low confidence prediction: {confidence:.2f}", extra={
                'prediction': prediction,
                'features': request.features
            })

        return {"prediction": prediction, "confidence": confidence}

    except Exception as e:
        predictions_total.labels(model_version="v2.1.3", status="error").inc()
        raise
```

### Grafana Dashboards

**Model Performance Dashboard**:
```json
{
  "dashboard": {
    "title": "ML Model Performance",
    "panels": [
      {
        "title": "Predictions per Second",
        "targets": [{"expr": "rate(ml_predictions_total[5m])"}]
      },
      {
        "title": "Latency Percentiles",
        "targets": [
          {"expr": "histogram_quantile(0.50, ml_prediction_latency_seconds_bucket)", "legendFormat": "p50"},
          {"expr": "histogram_quantile(0.95, ml_prediction_latency_seconds_bucket)", "legendFormat": "p95"},
          {"expr": "histogram_quantile(0.99, ml_prediction_latency_seconds_bucket)", "legendFormat": "p99"}
        ]
      },
      {
        "title": "Error Rate",
        "targets": [{"expr": "rate(ml_predictions_total{status='error'}[5m]) / rate(ml_predictions_total[5m])"}]
      },
      {
        "title": "Model Accuracy (24h rolling)",
        "targets": [{"expr": "ml_model_accuracy"}]
      }
    ]
  }
}
```

---

## Data Drift Detection

**Statistical Tests for Drift**:

1. **Kolmogorov-Smirnov Test** (Continuous features)
2. **Chi-Square Test** (Categorical features)
3. **Population Stability Index (PSI)** (All features)

### Implementation

**KS Test for Continuous Features**:
```python
from scipy.stats import ks_2samp
import pandas as pd

def detect_ks_drift(reference_data, current_data, threshold=0.05):
    """Kolmogorov-Smirnov test for feature drift"""

    drift_report = []

    for column in reference_data.select_dtypes(include=['float64', 'int64']).columns:
        # Perform KS test
        statistic, p_value = ks_2samp(
            reference_data[column].dropna(),
            current_data[column].dropna()
        )

        drifted = p_value < threshold

        drift_report.append({
            'feature': column,
            'test': 'ks_test',
            'statistic': statistic,
            'p_value': p_value,
            'drifted': drifted,
            'severity': 'high' if p_value < 0.01 else 'medium' if drifted else 'low'
        })

    return pd.DataFrame(drift_report)
```

**Population Stability Index (PSI)**:
```python
import numpy as np

def calculate_psi(reference_data, current_data, bins=10):
    """Calculate Population Stability Index"""

    # Create bins based on reference data
    breakpoints = np.percentile(reference_data, np.linspace(0, 100, bins+1))

    # Calculate distributions
    ref_counts, _ = np.histogram(reference_data, bins=breakpoints)
    curr_counts, _ = np.histogram(current_data, bins=breakpoints)

    ref_percents = ref_counts / len(reference_data)
    curr_percents = curr_counts / len(current_data)

    # PSI calculation
    psi_values = (curr_percents - ref_percents) * np.log((curr_percents + 1e-10) / (ref_percents + 1e-10))
    psi = np.sum(psi_values)

    # Interpretation
    if psi < 0.1:
        status = "No significant change"
    elif psi < 0.2:
        status = "Moderate change - investigate"
    else:
        status = "Significant change - retrain model"

    return {
        'psi': psi,
        'status': status,
        'requires_action': psi >= 0.2
    }
```

**Automated Drift Monitoring**:
```python
import schedule
import time

def daily_drift_check():
    """Run daily drift detection"""

    # Load reference data (training set)
    reference_data = load_training_data()

    # Load recent production data (last 7 days)
    current_data = load_production_data(days=7)

    # Detect drift
    drift_report = detect_ks_drift(reference_data, current_data)

    # Check for drifted features
    drifted_features = drift_report[drift_report['drifted'] == True]

    if len(drifted_features) > 0:
        # Alert data science team
        send_alert(
            title="Data Drift Detected",
            message=f"{len(drifted_features)} features showing drift",
            details=drifted_features.to_dict('records')
        )

        # Trigger retraining pipeline if critical features drifted
        critical_features = ['customer_tenure', 'total_spend_30d']
        if any(feat in drifted_features['feature'].values for feat in critical_features):
            trigger_retraining_pipeline()

    # Log results
    log_drift_metrics(drift_report)

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(daily_drift_check)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

---

## Concept Drift Detection

**Definition**: When the statistical properties of the target variable change over time.

### Detection Methods

**1. Performance Degradation Monitoring**:
```python
def detect_concept_drift(model, recent_data_with_labels, baseline_accuracy=0.94, threshold=0.05):
    """Detect concept drift via performance degradation"""

    # Evaluate model on recent labeled data
    predictions = model.predict(recent_data_with_labels['features'])
    current_accuracy = accuracy_score(recent_data_with_labels['labels'], predictions)

    # Check for significant degradation
    accuracy_drop = baseline_accuracy - current_accuracy

    if accuracy_drop > threshold:
        return {
            'concept_drift_detected': True,
            'baseline_accuracy': baseline_accuracy,
            'current_accuracy': current_accuracy,
            'accuracy_drop': accuracy_drop,
            'action': 'retrain_model'
        }
    else:
        return {
            'concept_drift_detected': False,
            'current_accuracy': current_accuracy
        }
```

**2. Prediction Distribution Monitoring**:
```python
def monitor_prediction_distribution(historical_predictions, current_predictions):
    """Check if prediction distribution has changed"""

    # Historical distribution
    hist_positive_rate = (historical_predictions == 1).mean()

    # Current distribution
    curr_positive_rate = (current_predictions == 1).mean()

    # Significant change indicates concept drift
    change = abs(curr_positive_rate - hist_positive_rate)

    return {
        'historical_positive_rate': hist_positive_rate,
        'current_positive_rate': curr_positive_rate,
        'change': change,
        'drift_detected': change > 0.1  # 10% threshold
    }
```

---

## System Observability

### Distributed Tracing

**OpenTelemetry Integration**:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configure tracer
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    with tracer.start_as_current_span("ml_prediction") as span:
        # Feature extraction
        with tracer.start_as_current_span("feature_extraction"):
            features = extract_features(request.raw_data)
            span.set_attribute("feature_count", len(features))

        # Model inference
        with tracer.start_as_current_span("model_inference"):
            prediction = model.predict(features)
            span.set_attribute("prediction", prediction)

        # Post-processing
        with tracer.start_as_current_span("post_processing"):
            result = post_process(prediction)

        return result
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }

        if hasattr(record, 'prediction'):
            log_data['prediction'] = record.prediction
        if hasattr(record, 'features'):
            log_data['features'] = record.features
        if hasattr(record, 'latency'):
            log_data['latency_ms'] = record.latency

        return json.dumps(log_data)

# Configure logger
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger('ml_service')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Prediction made", extra={
    'prediction': 'churn',
    'confidence': 0.85,
    'latency': 45.2
})
```

---

## Cost Tracking & Optimization

### Infrastructure Cost Monitoring

```python
import boto3
from datetime import datetime, timedelta

def get_ml_infrastructure_costs(days=7):
    """Get AWS costs for ML infrastructure"""

    ce_client = boto3.client('ce', region_name='us-east-1')

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date.isoformat(),
            'End': end_date.isoformat()
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'Tags': {
                'Key': 'Project',
                'Values': ['ml-pipeline']
            }
        },
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'SERVICE'},
            {'Type': 'TAG', 'Key': 'Environment'}
        ]
    )

    return response['ResultsByTime']
```

### Cost per Prediction Tracking

```python
def calculate_cost_per_prediction(infrastructure_cost, num_predictions):
    """Calculate unit economics for ML model"""

    cost_per_prediction = infrastructure_cost / num_predictions

    return {
        'total_cost': infrastructure_cost,
        'total_predictions': num_predictions,
        'cost_per_prediction': cost_per_prediction,
        'cost_per_1k_predictions': cost_per_prediction * 1000
    }

# Example: Monitor daily
daily_cost = get_ml_infrastructure_costs(days=1)
daily_predictions = get_prediction_count(days=1)
metrics = calculate_cost_per_prediction(daily_cost, daily_predictions)

# Alert if cost per prediction increases significantly
if metrics['cost_per_1k_predictions'] > 1.50:  # $1.50 threshold
    send_cost_alert(metrics)
```

---

## Best Practices

1. **Monitor Proactively**: Set up alerts before issues become critical
2. **Track Business Metrics**: Don't just monitor technical metrics
3. **Automate Drift Detection**: Run daily/weekly drift checks
4. **Set Clear Thresholds**: Define when to retrain based on performance degradation
5. **Log Everything**: Structured logging for debugging and analysis
6. **Use Distributed Tracing**: Understand end-to-end request flow
7. **Optimize Costs**: Track cost per prediction, use spot instances
8. **Document Runbooks**: Clear procedures for handling alerts
