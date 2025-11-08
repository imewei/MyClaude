# Deployment Strategies Guide

Comprehensive guide to ML model deployment strategies for zero-downtime, safe production rollouts.

## Table of Contents
- [Canary Deployments](#canary-deployments)
- [Blue-Green Deployments](#blue-green-deployments)
- [Shadow Deployments](#shadow-deployments)
- [A/B Testing](#ab-testing-for-ml-models)
- [Feature Flags](#feature-flags-for-gradual-rollout)

---

## Canary Deployments

**Overview**: Gradually shift traffic from old model to new model while monitoring performance.

**Traffic Split Strategy**:
```
Stage 1: 90% old, 10% canary (Monitor 1 hour)
Stage 2: 75% old, 25% canary (Monitor 2 hours)
Stage 3: 50% old, 50% canary (Monitor 4 hours)
Stage 4: 0% old, 100% canary (Full rollout)
```

**Implementation (Kubernetes + Istio)**:
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-model-canary
spec:
  hosts:
  - ml-model-service
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: ml-model-service
        subset: canary
      weight: 100
  - route:
    - destination:
        host: ml-model-service
        subset: stable
      weight: 90
    - destination:
        host: ml-model-service
        subset: canary
      weight: 10
```

**Monitoring Criteria**:
- Error rate <0.5% increase
- Latency p99 <10% increase
- Model accuracy within 2% of baseline
- No increase in invalid predictions

**Rollback Trigger**:
```python
def should_rollback_canary(metrics):
    """Automated rollback decision"""
    return (
        metrics['error_rate'] > 0.01 or
        metrics['latency_p99'] > 500 or
        metrics['accuracy_drop'] > 0.02
    )
```

---

## Blue-Green Deployments

**Overview**: Maintain two identical environments (blue=current, green=new), switch traffic instantly.

**Benefits**:
- Instant rollback capability
- Full testing in production environment
- Zero downtime
- Easy to understand and implement

**Implementation**:
```yaml
# Deployment for Blue (current production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-blue
  labels:
    version: blue
spec:
  replicas: 5
  selector:
    matchLabels:
      app: ml-model
      version: blue
  template:
    metadata:
      labels:
        app: ml-model
        version: blue
    spec:
      containers:
      - name: model
        image: ml-model:v1.0.0

---
# Deployment for Green (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-green
  labels:
    version: green
spec:
  replicas: 5
  selector:
    matchLabels:
      app: ml-model
      version: green
  template:
    metadata:
      labels:
        app: ml-model
        version: green
    spec:
      containers:
      - name: model
        image: ml-model:v2.0.0

---
# Service (switch by updating selector)
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
    version: blue  # Change to 'green' to switch traffic
  ports:
  - port: 80
    targetPort: 8000
```

**Deployment Process**:
1. Deploy green environment with new model
2. Run smoke tests against green
3. Update service selector from `version: blue` to `version: green`
4. Monitor for 24 hours
5. If stable, decommission blue; else revert selector

---

## Shadow Deployments

**Overview**: Route copy of production traffic to new model without affecting users, compare predictions.

**Use Cases**:
- Validate new model in production without risk
- Compare multiple model versions
- Test infrastructure changes
- Benchmark performance under real load

**Implementation**:
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-model-shadow
spec:
  hosts:
  - ml-model-service
  http:
  - match:
    - uri:
        prefix: "/predict"
    route:
    - destination:
        host: ml-model-stable
      weight: 100
    mirror:
      host: ml-model-shadow
    mirrorPercentage:
      value: 100.0  # Mirror 100% of traffic
```

**Prediction Comparison**:
```python
import logging

def compare_shadow_predictions(stable_pred, shadow_pred, features):
    """Log prediction differences for analysis"""

    difference = abs(stable_pred - shadow_pred)

    if difference > 0.1:  # 10% threshold
        logging.warning(
            f"Shadow prediction differs by {difference:.2%}",
            extra={
                'stable_prediction': stable_pred,
                'shadow_prediction': shadow_pred,
                'features': features
            }
        )

    # Store in metrics database for analysis
    metrics_db.insert({
        'timestamp': datetime.now(),
        'stable_pred': stable_pred,
        'shadow_pred': shadow_pred,
        'difference': difference
    })
```

---

## A/B Testing for ML Models

**Overview**: Randomly assign users to different model versions, measure business impact.

**Statistical Framework**:
```python
from scipy.stats import chi2_contingency
import pandas as pd

def analyze_ab_test(group_a_data, group_b_data, metric='conversion_rate'):
    """Analyze A/B test results with statistical significance"""

    # Calculate metrics
    a_metric = group_a_data[metric].mean()
    b_metric = group_b_data[metric].mean()

    # Chi-square test for significance
    contingency_table = pd.crosstab(
        group_a_data['converted'],
        group_b_data['converted']
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Calculate effect size
    effect_size = (b_metric - a_metric) / a_metric

    return {
        'group_a_metric': a_metric,
        'group_b_metric': b_metric,
        'effect_size': effect_size,
        'p_value': p_value,
        'statistically_significant': p_value < 0.05,
        'sample_size_a': len(group_a_data),
        'sample_size_b': len(group_b_data)
    }
```

**Traffic Splitting**:
```python
import hashlib

def assign_to_variant(user_id, variants={'A': 0.5, 'B': 0.5}):
    """Consistent user assignment to A/B test variants"""

    # Hash user_id for consistent assignment
    hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    random_value = (hash_value % 10000) / 10000.0

    # Assign based on cumulative weights
    cumulative = 0
    for variant, weight in variants.items():
        cumulative += weight
        if random_value < cumulative:
            return variant

    return list(variants.keys())[-1]

# Usage
user_id = "customer-12345"
variant = assign_to_variant(user_id)  # Returns 'A' or 'B' consistently

if variant == 'A':
    prediction = model_a.predict(features)
else:
    prediction = model_b.predict(features)
```

**Sample Size Calculation**:
```python
from statsmodels.stats.power import tt_ind_solve_power

def calculate_required_sample_size(
    baseline_rate=0.10,
    minimum_detectable_effect=0.02,  # 2% absolute increase
    alpha=0.05,  # Significance level
    power=0.80   # Statistical power
):
    """Calculate required sample size per variant"""

    effect_size = minimum_detectable_effect / baseline_rate

    n_per_group = tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )

    return int(n_per_group)

# Example: Need 3,841 samples per group to detect 2% effect
sample_size = calculate_required_sample_size()
```

---

## Feature Flags for Gradual Rollout

**Overview**: Control model deployment via configuration flags, enable gradual rollout without code deployment.

**LaunchDarkly Integration**:
```python
import launchdarkly

ld_client = launchdarkly.Client(sdk_key=os.getenv("LD_SDK_KEY"))

def get_model_version(user_id):
    """Get model version based on feature flag"""

    user = {
        "key": user_id,
        "custom": {
            "segment": get_user_segment(user_id)
        }
    }

    # Check feature flag
    use_new_model = ld_client.variation("ml-model-v2", user, default=False)

    if use_new_model:
        return "v2.0.0"
    else:
        return "v1.0.0"

# Usage in prediction endpoint
@app.post("/predict")
async def predict(user_id: str, features: dict):
    model_version = get_model_version(user_id)
    model = load_model(model_version)
    prediction = model.predict(features)
    return {"prediction": prediction, "model_version": model_version}
```

**Rollout Strategy**:
```yaml
# Feature flag configuration
flags:
  ml-model-v2:
    enabled: true
    rollout:
      - variation: true
        percentage: 5    # Start with 5% of users
        targets:
          - segment: "beta-users"  # Enable for beta testers first

# Day 2: Increase to 25%
# Day 5: Increase to 50%
# Day 7: Increase to 100% (full rollout)
```

---

## Comparison Matrix

| Strategy | Downtime | Rollback Speed | Resource Cost | Complexity | Best For |
|----------|----------|----------------|---------------|------------|----------|
| **Canary** | Zero | Fast (seconds) | Low (gradual scaling) | Medium | Gradual validation |
| **Blue-Green** | Zero | Instant | High (2x resources) | Low | Instant rollback needs |
| **Shadow** | Zero | N/A (no user impact) | Medium | Medium | Risk-free testing |
| **A/B Testing** | Zero | Fast | Low | High | Business impact measurement |
| **Feature Flags** | Zero | Instant | Low | Low | Fine-grained control |

---

## Best Practices

1. **Start with Shadow Deployments**: Validate without risk first
2. **Use Canary for Gradual Rollout**: Monitor at each stage
3. **Keep Blue-Green for Critical Systems**: Instant rollback capability
4. **Run A/B Tests for Business Impact**: Measure actual value
5. **Combine Strategies**: Use feature flags + canary for maximum control
6. **Automate Rollback**: Define clear criteria and automate decisions
7. **Monitor Business Metrics**: Don't just track technical metrics
8. **Document Rollback Procedures**: Practice incident response
