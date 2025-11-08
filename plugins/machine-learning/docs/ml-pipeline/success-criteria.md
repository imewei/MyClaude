# ML Pipeline Success Criteria

Quantified success metrics for evaluating production ML pipeline quality across data, model, operations, development, and cost dimensions.

## Table of Contents
- [Data Pipeline Success Metrics](#1-data-pipeline-success)
- [Model Performance Metrics](#2-model-performance)
- [Operational Excellence Metrics](#3-operational-excellence)
- [Development Velocity Metrics](#4-development-velocity)
- [Cost Efficiency Metrics](#5-cost-efficiency)

---

## 1. Data Pipeline Success

### Data Quality
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Data quality issues** | <0.1% | (Failed validations / Total records) × 100 |
| **Schema validation pass rate** | >99.9% | (Valid batches / Total batches) × 100 |
| **Missing value rate** | <5% for critical features | (Null count / Total count) × 100 per feature |
| **Duplicate record rate** | <0.01% | (Duplicates / Total records) × 100 |
| **Data freshness** | <1 hour lag | Time between data generation and availability |

**Example Implementation**:
```python
def calculate_data_quality_metrics(data):
    """Calculate data quality KPIs"""

    total_records = len(data)

    metrics = {
        'total_records': total_records,
        'duplicate_rate': data.duplicated().sum() / total_records,
        'quality_issues': 0,
        'feature_missing_rates': {}
    }

    # Check each feature
    for column in data.columns:
        missing_rate = data[column].isnull().sum() / total_records

        metrics['feature_missing_rates'][column] = missing_rate

        # Flag quality issue if critical feature missing
        if column in CRITICAL_FEATURES and missing_rate > 0.05:
            metrics['quality_issues'] += 1

    metrics['quality_issue_rate'] = metrics['quality_issues'] / len(data.columns)

    return metrics
```

### Data Lineage & Governance
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Lineage tracking coverage** | 100% | (Tables with lineage / Total tables) × 100 |
| **Data versioning adoption** | 100% | All datasets tracked in DVC/lakeFS |
| **PII compliance** | 100% | All PII fields properly masked/encrypted |
| **Access audit logs** | 100% | All data access logged with retention >1 year |

### Feature Engineering Performance
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Feature serving latency (p99)** | <1 second | 99th percentile of feature retrieval time |
| **Feature computation success rate** | >99.9% | (Successful computations / Total requests) × 100 |
| **Feature store availability** | >99.99% | Uptime percentage over month |

---

## 2. Model Performance

### Prediction Quality
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Meets or exceeds baseline** | Yes | Current metric ≥ Baseline metric |
| **Model accuracy** | >85% (project-specific) | (Correct predictions / Total predictions) × 100 |
| **Precision** | >80% | True Positives / (True Positives + False Positives) |
| **Recall** | >75% | True Positives / (True Positives + False Negatives) |
| **F1-Score** | >0.80 | 2 × (Precision × Recall) / (Precision + Recall) |
| **AUC-ROC** | >0.85 | Area under ROC curve |

**Example Monitoring**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def monitor_model_performance(y_true, y_pred, y_pred_proba, baseline_metrics):
    """Monitor model performance against baselines"""

    current_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
    }

    # Check against baselines
    alerts = []
    for metric, value in current_metrics.items():
        baseline = baseline_metrics.get(metric, 0)
        degradation = baseline - value

        if degradation > 0.02:  # 2% degradation threshold
            alerts.append({
                'metric': metric,
                'current': value,
                'baseline': baseline,
                'degradation': degradation
            })

    return current_metrics, alerts
```

### Model Stability
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Performance degradation threshold** | <5% before retraining | Max acceptable drop from baseline before triggering retrain |
| **Prediction consistency** | >95% | (Same predictions on retry / Total) × 100 |
| **Model drift detection latency** | <24 hours | Time to detect significant model drift |

### Experimentation
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **A/B test statistical significance** | p-value <0.05 | Statistical test result |
| **Minimum sample size met** | Yes | Actual samples ≥ Required samples for power=0.80 |
| **Test runtime** | <2 weeks | Time from test start to decision |

---

## 3. Operational Excellence

### Availability & Reliability
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Model serving uptime** | >99.9% | (Uptime minutes / Total minutes) × 100 per month |
| **API availability** | >99.95% | (Successful health checks / Total checks) × 100 |
| **Error rate** | <0.1% | (Failed predictions / Total predictions) × 100 |
| **Incident MTTR** | <30 minutes | Mean time to restore service |

**SLA Calculation**:
```python
from datetime import datetime, timedelta

def calculate_sla_compliance(uptime_events, total_time_minutes):
    """Calculate SLA compliance"""

    downtime_minutes = sum(event['duration_minutes'] for event in uptime_events if event['status'] == 'down')
    uptime_minutes = total_time_minutes - downtime_minutes

    sla_percentage = (uptime_minutes / total_time_minutes) * 100

    # 99.9% SLA = 43.2 minutes downtime per month
    sla_target = 99.9
    allowed_downtime = total_time_minutes * (1 - sla_target / 100)

    return {
        'uptime_percentage': sla_percentage,
        'uptime_minutes': uptime_minutes,
        'downtime_minutes': downtime_minutes,
        'allowed_downtime_minutes': allowed_downtime,
        'sla_met': sla_percentage >= sla_target,
        'downtime_budget_remaining': allowed_downtime - downtime_minutes
    }
```

### Performance
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Prediction latency (p50)** | <50ms | Median response time |
| **Prediction latency (p95)** | <150ms | 95th percentile response time |
| **Prediction latency (p99)** | <200ms | 99th percentile response time |
| **Throughput** | >1000 predictions/sec | Requests per second capacity |
| **Concurrent requests** | >100 | Maximum simultaneous requests handled |

### Deployment
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Deployment success rate** | >99% | (Successful deployments / Total deployments) × 100 |
| **Rollback time** | <5 minutes | Time from issue detection to rollback completion |
| **Automated rollback success** | >95% | (Successful auto-rollbacks / Total auto-rollbacks) × 100 |
| **Zero-downtime deployments** | 100% | All deployments complete without service interruption |

### Monitoring & Alerting
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Alert response time** | <1 minute | Time from issue to alert delivery |
| **False positive rate** | <10% | (False alerts / Total alerts) × 100 |
| **Monitoring coverage** | 100% | All critical components have health checks |
| **Log retention** | >90 days | Days of logs retained for debugging |

---

## 4. Development Velocity

### CI/CD Pipeline
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Commit to production time** | <1 hour | Time from git commit to production deployment |
| **Build success rate** | >95% | (Successful builds / Total builds) × 100 |
| **Test pass rate** | 100% | All tests must pass for deployment |
| **Pipeline execution time** | <15 minutes | Total time for CI/CD pipeline |

**Example Tracking**:
```python
from datetime import datetime

def track_deployment_metrics(commit_sha):
    """Track deployment velocity metrics"""

    commit_time = get_commit_timestamp(commit_sha)
    build_start = get_build_start_time(commit_sha)
    build_end = get_build_end_time(commit_sha)
    deployment_time = get_deployment_time(commit_sha)

    metrics = {
        'commit_sha': commit_sha,
        'commit_time': commit_time,
        'build_duration_minutes': (build_end - build_start).total_seconds() / 60,
        'commit_to_production_minutes': (deployment_time - commit_time).total_seconds() / 60,
        'build_success': check_build_status(commit_sha) == 'success'
    }

    # Alert if taking too long
    if metrics['commit_to_production_minutes'] > 60:
        alert_slow_deployment(metrics)

    return metrics
```

### Experimentation & Iteration
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Parallel experiment capacity** | >10 experiments | Simultaneous experiments supported |
| **Experiment setup time** | <30 minutes | Time to launch new experiment |
| **Reproducibility rate** | 100% | (Reproducible runs / Total runs) × 100 |
| **Model training time** | <4 hours | Time to train production model |

### Automation
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Automated testing coverage** | >80% | Code coverage percentage |
| **Manual deployment steps** | 0 | All deployments fully automated |
| **Self-service model deployment** | Yes | Data scientists can deploy without DevOps |
| **Automated retraining** | Yes | Model retrains automatically on data drift |

---

## 5. Cost Efficiency

### Infrastructure Costs
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Infrastructure waste** | <20% | (Unused resources / Total resources) × 100 |
| **Cost per 1K predictions** | <$0.50 | Total infrastructure cost / (Predictions / 1000) |
| **Training cost per model** | <$100 | Total compute cost for model training |
| **Storage cost optimization** | >30% savings | Savings from lifecycle policies and compression |

**Cost Tracking**:
```python
def calculate_cost_efficiency_metrics(infrastructure_cost, predictions_count, training_runs):
    """Calculate cost efficiency KPIs"""

    metrics = {
        'total_monthly_cost': infrastructure_cost,
        'total_predictions': predictions_count,
        'cost_per_1k_predictions': (infrastructure_cost / predictions_count) * 1000,
        'training_cost_per_model': infrastructure_cost / training_runs if training_runs > 0 else 0
    }

    # Cost per prediction target: $0.50 per 1K predictions
    if metrics['cost_per_1k_predictions'] > 0.50:
        metrics['cost_alert'] = f"Cost per 1K predictions (${metrics['cost_per_1k_predictions']:.2f}) exceeds target ($0.50)"

    return metrics
```

### Resource Optimization
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **CPU utilization** | 60-80% | Average CPU usage across instances |
| **Memory utilization** | 60-80% | Average memory usage |
| **GPU utilization (training)** | >85% | GPU usage during training jobs |
| **Spot instance usage** | >60% | (Spot instances / Total instances) × 100 |
| **Auto-scaling effectiveness** | >90% | Correct scaling decisions percentage |

### Data Storage
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Storage growth rate** | <10% per month | Month-over-month storage increase |
| **Archived data percentage** | >70% of 90-day+ data | Data moved to cold storage |
| **Compression ratio** | >5:1 for Parquet | Original size / Compressed size |
| **Duplicate data** | <1% | (Duplicate records / Total records) × 100 |

---

## Comprehensive Success Criteria Dashboard

```python
def evaluate_ml_pipeline_success():
    """Evaluate overall ML pipeline success across all dimensions"""

    criteria = {
        'data_pipeline': {
            'quality_issue_rate': {'actual': 0.08, 'target': 0.1, 'met': True},
            'validation_pass_rate': {'actual': 99.95, 'target': 99.9, 'met': True},
            'feature_latency_p99': {'actual': 0.8, 'target': 1.0, 'met': True}
        },
        'model_performance': {
            'accuracy': {'actual': 0.942, 'target': 0.85, 'met': True},
            'degradation_from_baseline': {'actual': 0.018, 'target': 0.05, 'met': True},
            'auc_roc': {'actual': 0.91, 'target': 0.85, 'met': True}
        },
        'operational': {
            'uptime': {'actual': 99.97, 'target': 99.9, 'met': True},
            'latency_p99_ms': {'actual': 145, 'target': 200, 'met': True},
            'error_rate': {'actual': 0.05, 'target': 0.1, 'met': True},
            'rollback_time_minutes': {'actual': 3.2, 'target': 5, 'met': True}
        },
        'development_velocity': {
            'commit_to_prod_minutes': {'actual': 42, 'target': 60, 'met': True},
            'pipeline_duration_minutes': {'actual': 12, 'target': 15, 'met': True},
            'experiment_capacity': {'actual': 15, 'target': 10, 'met': True}
        },
        'cost_efficiency': {
            'infrastructure_waste_pct': {'actual': 15, 'target': 20, 'met': True},
            'cost_per_1k_predictions': {'actual': 0.38, 'target': 0.50, 'met': True},
            'spot_instance_usage_pct': {'actual': 65, 'target': 60, 'met': True}
        }
    }

    # Calculate overall success rate
    total_criteria = 0
    met_criteria = 0

    for category, metrics in criteria.items():
        for metric, data in metrics.items():
            total_criteria += 1
            if data['met']:
                met_criteria += 1

    overall_success_rate = (met_criteria / total_criteria) * 100

    return {
        'criteria': criteria,
        'overall_success_rate': overall_success_rate,
        'total_criteria': total_criteria,
        'met_criteria': met_criteria,
        'pipeline_ready_for_production': overall_success_rate >= 95
    }
```

---

## Red Flags & Immediate Actions

### Critical Issues Requiring Immediate Action

| Red Flag | Threshold | Immediate Action |
|----------|-----------|------------------|
| **Uptime drops below 99%** | <99.0% | Trigger incident response, rollback if recent deployment |
| **Error rate spike** | >1% | Rollback deployment, investigate root cause |
| **Model accuracy drops >5%** | <Baseline - 5% | Retrain model immediately, investigate data quality |
| **Latency p99 >500ms** | >500ms | Scale up resources, investigate performance bottleneck |
| **Data quality issues >1%** | >1% | Halt pipeline, investigate data source |
| **Cost spike >50%** | >150% of baseline | Investigate resource utilization, check for runaway processes |
| **Security breach detected** | Any incident | Immediately restrict access, alert security team |

---

## Monthly Success Report Template

```markdown
# ML Pipeline Monthly Success Report - [Month Year]

## Executive Summary
- Overall Success Rate: [X]% ([Y]/[Z] criteria met)
- Pipeline Status: [Production-Ready / Needs Improvement]
- Key Achievements: [List top 3]
- Critical Issues: [List any red flags]

## Data Pipeline
- Quality Issue Rate: [X]% (Target: <0.1%) ✅/❌
- Validation Pass Rate: [X]% (Target: >99.9%) ✅/❌
- Feature Latency p99: [X]ms (Target: <1000ms) ✅/❌

## Model Performance
- Accuracy: [X]% (Target: >[Y]%) ✅/❌
- Degradation: [X]% (Target: <5%) ✅/❌
- AUC-ROC: [X] (Target: >0.85) ✅/❌

## Operational Excellence
- Uptime: [X]% (Target: >99.9%) ✅/❌
- Latency p99: [X]ms (Target: <200ms) ✅/❌
- Error Rate: [X]% (Target: <0.1%) ✅/❌

## Development Velocity
- Commit-to-Prod Time: [X] min (Target: <60 min) ✅/❌
- Build Success Rate: [X]% (Target: >95%) ✅/❌

## Cost Efficiency
- Cost per 1K Predictions: $[X] (Target: <$0.50) ✅/❌
- Infrastructure Waste: [X]% (Target: <20%) ✅/❌
- Spot Instance Usage: [X]% (Target: >60%) ✅/❌

## Action Items
1. [Action item 1 with owner and deadline]
2. [Action item 2 with owner and deadline]
```
