# Deployment Strategies

Comprehensive guide to modern deployment strategies for feature rollouts.

## Table of Contents
- [Direct Deployment](#direct-deployment)
- [Canary Deployment](#canary-deployment)
- [Feature Flag Deployment](#feature-flag-deployment)
- [Blue-Green Deployment](#blue-green-deployment)
- [A/B Testing Deployment](#ab-testing-deployment)
- [Rollback Procedures](#rollback-procedures)
- [Strategy Selection Guide](#strategy-selection-guide)

---

## Direct Deployment

### Overview
Immediate rollout to all users simultaneously. Simplest but highest risk strategy.

### When to Use
- ✅ Low-risk features (UI tweaks, copy changes)
- ✅ Internal tools with small user base
- ✅ Hot fixes for critical bugs
- ✅ Non-production environments

### Implementation

```yaml
# .github/workflows/deploy-direct.yml
name: Direct Deployment

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build
        run: npm run build

      - name: Test
        run: npm test

      - name: Deploy to Production
        run: |
          aws s3 sync ./dist s3://production-bucket
          aws cloudfront create-invalidation --distribution-id $DIST_ID --paths "/*"
```

### Risks
- ⚠️ All users affected immediately if bugs exist
- ⚠️ No gradual rollout for monitoring
- ⚠️ Difficult to rollback quickly

### Mitigation
- Comprehensive testing in staging
- Smoke tests after deployment
- Quick rollback procedure ready
- Monitoring alerts configured

---

## Canary Deployment

### Overview
Gradual rollout starting with small percentage of traffic, progressively increasing.

### Traffic Distribution Phases
```
Phase 1: 5% traffic   (30 min) → Monitor
Phase 2: 25% traffic  (1 hour) → Monitor
Phase 3: 50% traffic  (2 hours) → Monitor
Phase 4: 100% traffic (Full rollout)
```

### Implementation

**Kubernetes Canary with Istio**:
```yaml
# canary-deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: feature-service
spec:
  selector:
    app: feature-service
  ports:
    - port: 80

---
# Stable deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-service-stable
spec:
  replicas: 9  # 90% traffic
  selector:
    matchLabels:
      app: feature-service
      version: stable
  template:
    metadata:
      labels:
        app: feature-service
        version: stable
    spec:
      containers:
      - name: app
        image: myapp:v1.0

---
# Canary deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-service-canary
spec:
  replicas: 1  # 10% traffic
  selector:
    matchLabels:
      app: feature-service
      version: canary
  template:
    metadata:
      labels:
        app: feature-service
        version: canary
    spec:
      containers:
      - name: app
        image: myapp:v2.0  # New version

---
# Istio VirtualService for traffic splitting
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: feature-service
spec:
  hosts:
    - feature-service
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: feature-service
        subset: canary
      weight: 100
  - route:
    - destination:
        host: feature-service
        subset: stable
      weight: 95
    - destination:
        host: feature-service
        subset: canary
      weight: 5  # Start with 5% canary traffic
```

**Progressive Rollout Script**:
```bash
#!/bin/bash
# progressive-canary.sh

PERCENTAGES=(5 25 50 100)
WAIT_TIMES=(1800 3600 7200 0)  # 30min, 1hr, 2hr

for i in "${!PERCENTAGES[@]}"; do
  PERCENT=${PERCENTAGES[$i]}
  WAIT=${WAIT_TIMES[$i]}

  echo "Rolling out to $PERCENT% of traffic..."

  kubectl patch virtualservice feature-service --type=json \
    -p="[{'op': 'replace', 'path': '/spec/http/1/route/1/weight', 'value': $PERCENT}]"

  if [ $PERCENT -lt 100 ]; then
    echo "Monitoring for $WAIT seconds..."
    sleep $WAIT

    # Check metrics
    ERROR_RATE=$(get_error_rate_from_prometheus)
    if (( $(echo "$ERROR_RATE > 1.0" | bc -l) )); then
      echo "Error rate too high ($ERROR_RATE%), rolling back!"
      rollback_canary
      exit 1
    fi
  fi
done

echo "Canary deployment complete!"
```

### Monitoring Dashboard
```yaml
# prometheus-rules.yaml
groups:
  - name: canary-monitoring
    rules:
      - alert: CanaryHighErrorRate
        expr: |
          sum(rate(http_requests_total{version="canary", status=~"5.."}[5m])) /
          sum(rate(http_requests_total{version="canary"}[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Canary error rate > 1%"

      - alert: CanarySlowResponseTime
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket{version="canary"}[5m])
          ) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Canary p95 latency > 500ms"
```

---

## Feature Flag Deployment

### Overview
Deploy code to production but control activation via feature toggles. Allows instant rollback without redeployment.

### Feature Flag Providers
- **LaunchDarkly**: Enterprise feature management
- **Split.io**: A/B testing + feature flags
- **Unleash**: Open-source feature toggles
- **AWS AppConfig**: AWS-native feature flags
- **Custom**: Redis/database-backed flags

### Implementation

**Backend (Node.js + LaunchDarkly)**:
```typescript
import LaunchDarkly from 'launchdarkly-node-server-sdk';

const ldClient = LaunchDarkly.init(process.env.LAUNCHDARKLY_SDK_KEY);

app.get('/api/features', async (req, res) => {
  await ldClient.waitForInitialization();

  const user = {
    key: req.user.id,
    email: req.user.email,
    custom: {
      tier: req.user.tier,
      betaTester: req.user.betaTester
    }
  };

  // Check feature flag
  const newFeatureEnabled = await ldClient.variation('new-feature-enabled', user, false);

  if (newFeatureEnabled) {
    // New feature code path
    const result = await executeNewFeature(req);
    res.json({ result, version: 'new' });
  } else {
    // Old code path (fallback)
    const result = await executeOldFeature(req);
    res.json({ result, version: 'old' });
  }
});

// Gradual rollout rules in LaunchDarkly dashboard:
// - 5% of users (random)
// - 100% of beta testers
// - Specific user segments
```

**Frontend (React + LaunchDarkly)**:
```typescript
import { useLDClient, useFlags } from 'launchdarkly-react-client-sdk';

function FeatureComponent() {
  const { newFeatureEnabled } = useFlags();
  const ldClient = useLDClient();

  // Track feature exposure
  useEffect(() => {
    if (newFeatureEnabled) {
      ldClient?.track('new-feature-exposed');
    }
  }, [newFeatureEnabled]);

  if (newFeatureEnabled) {
    return <NewFeatureUI />;
  }

  return <OldFeatureUI />;
}
```

**Feature Flag Configuration**:
```yaml
# feature-flags.yaml
features:
  new-checkout-flow:
    enabled: true
    rollout:
      strategy: percentage
      percentage: 10  # Start with 10%
    targeting:
      rules:
        - name: Beta Testers
          clauses:
            - attribute: betaTester
              operator: equals
              value: true
          variation: true
        - name: Premium Users
          clauses:
            - attribute: tier
              operator: in
              values: [premium, enterprise]
          rollout:
            percentage: 50  # 50% of premium users
    variations:
      true: { enabled: true }
      false: { enabled: false }
```

### Kill Switch Pattern
```typescript
// Instant rollback without deployment
async function emergencyDisable(flagKey: string) {
  await featureFlagService.update(flagKey, {
    enabled: false,
    reason: 'Emergency disable due to production issues'
  });

  // Flag change propagates to all servers within seconds
  console.log(`Feature ${flagKey} disabled globally`);
}

// Usage
await emergencyDisable('new-checkout-flow');
```

---

## Blue-Green Deployment

### Overview
Maintain two identical production environments (Blue = current, Green = new). Switch traffic instantly with zero downtime.

### Infrastructure Setup

**AWS with ELB**:
```terraform
# blue-green-infrastructure.tf
resource "aws_lb_target_group" "blue" {
  name     = "feature-service-blue"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
  }
}

resource "aws_lb_target_group" "green" {
  name     = "feature-service-green"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

resource "aws_lb_listener" "main" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.blue.arn  # Initially route to blue
  }
}

# Auto Scaling Groups for both environments
resource "aws_autoscaling_group" "blue" {
  name                 = "feature-service-blue-asg"
  max_size             = 10
  min_size             = 2
  desired_capacity     = 3
  target_group_arns    = [aws_lb_target_group.blue.arn]
  launch_configuration = aws_launch_configuration.blue.id
}

resource "aws_autoscaling_group" "green" {
  name                 = "feature-service-green-asg"
  max_size             = 10
  min_size             = 0  # Starts at 0, scale up during deployment
  desired_capacity     = 0
  target_group_arns    = [aws_lb_target_group.green.arn]
  launch_configuration = aws_launch_configuration.green.id
}
```

**Deployment Script**:
```bash
#!/bin/bash
# blue-green-deploy.sh

# Deploy new version to Green environment
echo "Deploying to Green environment..."
terraform apply -target=aws_autoscaling_group.green \
  -var="green_desired_capacity=3" \
  -var="green_ami=ami-new-version"

# Wait for Green to be healthy
echo "Waiting for Green health checks..."
aws elbv2 wait target-in-service \
  --target-group-arn $GREEN_TARGET_GROUP_ARN

# Run smoke tests against Green
echo "Running smoke tests on Green..."
./smoke-tests.sh https://green.internal.example.com

if [ $? -ne 0 ]; then
  echo "Smoke tests failed, aborting deployment"
  exit 1
fi

# Switch traffic from Blue to Green
echo "Switching traffic to Green..."
aws elbv2 modify-listener \
  --listener-arn $LISTENER_ARN \
  --default-actions Type=forward,TargetGroupArn=$GREEN_TARGET_GROUP_ARN

echo "Traffic switched to Green. Monitoring..."
sleep 300  # Monitor for 5 minutes

# Check error rates
ERROR_RATE=$(get_error_rate)
if (( $(echo "$ERROR_RATE > 1.0" | bc -l) )); then
  echo "High error rate detected, rolling back!"
  rollback_to_blue
  exit 1
fi

# Success - scale down Blue (keep for quick rollback)
echo "Deployment successful. Scaling down Blue..."
terraform apply -target=aws_autoscaling_group.blue \
  -var="blue_desired_capacity=1"  # Keep 1 instance for quick rollback

echo "Blue-Green deployment complete!"
```

### Database Migrations
```sql
-- Blue-Green compatible migrations (backward compatible)

-- Phase 1: Deploy Green with new column (nullable)
ALTER TABLE users ADD COLUMN phone_number VARCHAR(20) NULL;

-- Phase 2: After Green is live, backfill data
UPDATE users SET phone_number = legacy_phone WHERE phone_number IS NULL;

-- Phase 3: After backfill, make required (next deployment)
-- ALTER TABLE users ALTER COLUMN phone_number SET NOT NULL;
```

---

## A/B Testing Deployment

### Overview
Split traffic between multiple variants to measure impact on metrics.

### Implementation

**Variant Assignment (Consistent Hashing)**:
```typescript
import crypto from 'crypto';

function assignVariant(userId: string, experimentName: string): 'control' | 'variant_a' | 'variant_b' {
  const hash = crypto.createHash('md5')
    .update(`${experimentName}:${userId}`)
    .digest('hex');

  const hashInt = parseInt(hash.substring(0, 8), 16);
  const bucket = hashInt % 100;  // 0-99

  // 33% control, 33% variant_a, 34% variant_b
  if (bucket < 33) return 'control';
  if (bucket < 66) return 'variant_a';
  return 'variant_b';
}

app.get('/api/checkout', async (req, res) => {
  const variant = assignVariant(req.user.id, 'checkout-flow-test');

  // Track assignment
  await analytics.track({
    userId: req.user.id,
    event: 'experiment_assigned',
    properties: {
      experiment: 'checkout-flow-test',
      variant
    }
  });

  switch (variant) {
    case 'control':
      return res.json(await originalCheckout(req));
    case 'variant_a':
      return res.json(await checkoutVariantA(req));
    case 'variant_b':
      return res.json(await checkoutVariantB(req));
  }
});
```

**Statistical Analysis**:
```python
# analyze-ab-test.py
import pandas as pd
from scipy import stats

def analyze_ab_test(data: pd.DataFrame):
    """
    Analyze A/B test results for statistical significance.

    data columns: user_id, variant, converted (bool)
    """
    control = data[data['variant'] == 'control']['converted']
    variant_a = data[data['variant'] == 'variant_a']['converted']

    # Calculate conversion rates
    control_rate = control.mean()
    variant_rate = variant_a.mean()
    lift = (variant_rate - control_rate) / control_rate * 100

    # Chi-squared test
    contingency_table = pd.crosstab(
        data['variant'].isin(['variant_a']),
        data['converted']
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Results
    return {
        'control_conversion_rate': f"{control_rate:.2%}",
        'variant_conversion_rate': f"{variant_rate:.2%}",
        'lift': f"{lift:+.1f}%",
        'p_value': p_value,
        'statistically_significant': p_value < 0.05,
        'confidence': f"{(1 - p_value) * 100:.1f}%",
        'recommendation': 'Ship variant' if p_value < 0.05 and lift > 0 else 'Keep control'
    }

# Usage
results = analyze_ab_test(experiment_data)
print(results)
# {
#   'control_conversion_rate': '12.34%',
#   'variant_conversion_rate': '14.56%',
#   'lift': '+18.0%',
#   'p_value': 0.003,
#   'statistically_significant': True,
#   'confidence': '99.7%',
#   'recommendation': 'Ship variant'
# }
```

---

## Rollback Procedures

### 1. Feature Flag Rollback (< 1 minute)
```typescript
// Instant disable via feature flag
await featureFlagService.disable('new-feature');
```

### 2. Blue-Green Rollback (< 5 minutes)
```bash
# Switch load balancer back to Blue
aws elbv2 modify-listener \
  --listener-arn $LISTENER_ARN \
  --default-actions Type=forward,TargetGroupArn=$BLUE_TARGET_GROUP_ARN
```

### 3. Kubernetes Rollback (< 5 minutes)
```bash
# Rollback to previous deployment
kubectl rollout undo deployment/feature-service
kubectl rollout status deployment/feature-service
```

### 4. Full Deployment Rollback (< 15 minutes)
```bash
# Revert code and redeploy
git revert $BAD_COMMIT_SHA
git push origin main
# CI/CD automatically deploys reverted version
```

### 5. Database Migration Rollback
```bash
# Run down migration
npm run migrate:down
# or
alembic downgrade -1
```

---

## Strategy Selection Guide

| Criteria | Direct | Canary | Feature Flag | Blue-Green | A/B Test |
|----------|--------|--------|--------------|------------|----------|
| **Deployment Speed** | Fastest | Slow | Medium | Fast | Medium |
| **Rollback Speed** | Slow | Medium | Instant | Instant | Instant |
| **Risk Level** | High | Low | Very Low | Low | Medium |
| **Infrastructure Cost** | Low | Medium | Low | High (2x) | Medium |
| **Monitoring Complexity** | Low | High | Medium | Medium | High |
| **Best For** | Low-risk changes | New features | Experiments | Zero-downtime | Optimization |

### Decision Tree

```
Is this a critical production system?
├─ No → Direct Deployment
└─ Yes → Do you need to measure business impact?
    ├─ Yes → A/B Testing Deployment
    └─ No → Do you need instant rollback?
        ├─ Yes → Feature Flag or Blue-Green
        └─ No → Canary Deployment
```

---

## Complete Rollout Example

```yaml
# complete-rollout-playbook.yml
deployment:
  feature: new-checkout-flow
  strategy: feature-flag + canary

  phase-1-deploy:
    - name: Deploy code to production
      actions:
        - Merge feature branch to main
        - CI/CD builds and deploys
        - Feature flag set to 0% (disabled for all)
      validation:
        - Smoke tests pass
        - No errors in logs

  phase-2-beta:
    - name: Enable for beta users
      actions:
        - Set feature flag targeting: betaTester = true → 100%
        - Monitor for 24 hours
      validation:
        - < 0.1% error rate
        - Positive beta user feedback
        - No performance degradation

  phase-3-canary:
    - name: Gradual rollout
      actions:
        - 5% random users (monitor 30min)
        - 25% random users (monitor 1hr)
        - 50% random users (monitor 2hr)
        - 100% users
      validation-at-each-step:
        - Error rate < 0.5%
        - p95 latency < 200ms
        - Conversion rate >= baseline

  phase-4-complete:
    - name: Full rollout
      actions:
        - Feature flag set to 100%
        - Monitor for 48 hours
        - Remove feature flag code (tech debt cleanup)
      validation:
        - All metrics stable
        - No user complaints
        - Conversion rate improved

  rollback-triggers:
    - Error rate > 1%
    - p95 latency > 500ms
    - Conversion rate drops > 5%
    - Critical bug discovered
```

---

## References

- [Feature Development Command](../../commands/feature-development.md)
- [Phase Templates](./phase-templates.md)
- [Best Practices](./best-practices.md)
