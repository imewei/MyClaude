# Phase 4: Deployment & Monitoring

## Overview

Phase 4 safely deploys the improved agent to production with monitoring and rollback capabilities.

**Duration**: 5-7 days (staged rollout)
**Prerequisites**: Phase 3 passed with go decision
**Outputs**: Production deployment, monitoring dashboard, post-deployment report

---

## Quick Start

```bash
# Execute Phase 4 with staged rollout
/improve-agent <agent-name> --phase=4 --canary=10%

# Monitor and increase traffic gradually
/improve-agent <agent-name> --phase=4 --increase-to=50%
/improve-agent <agent-name> --phase=4 --increase-to=100%
```

---

## Step 4.1: Staged Rollout Strategy

### Traffic Allocation

```
Day 1-2: Alpha (10% traffic)
├─ Internal team testing
├─ Automated monitoring
└─ Manual spot checks

Day 3-4: Beta (20% → 50%)
├─ Gradual traffic increase
├─ Continuous monitoring
└─ User feedback collection

Day 5-7: Full (100%)
├─ Complete rollout
├─ 7-day observation
└─ Post-deployment analysis
```

### Implementation

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RolloutStage:
    name: str
    traffic_percentage: int
    duration_hours: int
    success_criteria: dict

def create_rollout_plan(agent_name: str, new_version: str):
    return [
        RolloutStage(
            name="alpha",
            traffic_percentage=10,
            duration_hours=48,
            success_criteria={
                'min_tasks': 50,
                'success_rate_drop_threshold': 0.05,
                'max_critical_errors': 0
            }
        ),
        RolloutStage(
            name="beta_20",
            traffic_percentage=20,
            duration_hours=24,
            success_criteria={
                'min_tasks': 100,
                'success_rate_vs_baseline': -0.03,
                'user_complaints': 2
            }
        ),
        RolloutStage(
            name="beta_50",
            traffic_percentage=50,
            duration_hours=48,
            success_criteria={
                'min_tasks': 250,
                'success_rate_vs_baseline': 0.0,
                'latency_increase_max': 0.15
            }
        ),
        RolloutStage(
            name="full",
            traffic_percentage=100,
            duration_hours=168,  # 7 days
            success_criteria={
                'sustained_improvement': True
            }
        )
    ]
```

---

## Step 4.2: Monitoring Setup

### Real-Time Dashboard

**Key Metrics to Track**:
```python
dashboard_metrics = {
    'performance': {
        'success_rate_1h': rolling_average(1, 'hours'),
        'success_rate_24h': rolling_average(24, 'hours'),
        'p95_latency': percentile(95, window='1h'),
        'error_rate': error_count / total_tasks
    },
    'quality': {
        'user_corrections': corrections_per_task,
        'tool_efficiency': correct_tools / total_tool_calls,
        'user_satisfaction': avg_rating
    },
    'safety': {
        'constraint_violations': violation_count,
        'critical_errors': critical_error_count
    },
    'cost': {
        'tokens_per_task': avg_tokens,
        'cost_per_task': avg_cost_usd
    }
}
```

### Alert Configuration

```yaml
alerts:
  critical:
    - metric: success_rate_1h
      condition: < 0.80  # vs baseline 0.87
      action: auto_rollback

    - metric: critical_errors
      condition: > 0
      action: immediate_notification

  warning:
    - metric: success_rate_1h
      condition: < 0.85
      action: notify_team

    - metric: p95_latency
      condition: > 600ms  # +33% vs baseline 450ms
      action: investigate

    - metric: user_complaints
      condition: ≥ 3 in 1 hour
      action: manual_review

  info:
    - metric: rollout_stage_complete
      action: proceed_to_next_stage
```

---

## Step 4.3: Automatic Rollback

### Trigger Conditions

```python
def should_rollback(current_metrics, baseline_metrics) -> tuple[bool, str]:
    """Determine if automatic rollback needed"""

    # Critical: Success rate drops significantly
    if current_metrics['success_rate'] < baseline_metrics['success_rate'] - 0.10:
        return True, "Success rate dropped >10pp"

    # Critical: New critical errors
    if current_metrics['critical_errors'] > 0:
        return True, f"Critical errors detected: {current_metrics['critical_errors']}"

    # Critical: Safety violations
    if current_metrics['safety_violations'] > 0:
        return True, "Safety constraint violated"

    # Warning: Sustained degradation
    if current_metrics['success_rate_24h'] < baseline_metrics['success_rate'] - 0.05:
        return True, "Sustained 5pp degradation over 24h"

    return False, ""
```

### Rollback Procedure

```python
def execute_rollback(agent_name: str, from_version: str, to_version: str):
    """Immediately switch traffic back to previous version"""

    print(f"⚠️  ROLLBACK INITIATED: {agent_name}")
    print(f"   From: {from_version}")
    print(f"   To: {to_version}")

    # 1. Switch traffic routing (immediate)
    update_traffic_config(agent_name, active_version=to_version)

    # 2. Notify team
    send_alert(
        severity="critical",
        message=f"Auto-rollback executed for {agent_name}",
        details=get_rollback_reason()
    )

    # 3. Generate incident report
    create_incident_report(
        agent=agent_name,
        failed_version=from_version,
        metrics_at_rollback=get_current_metrics(),
        reason=get_rollback_reason()
    )

    # 4. Log for analysis
    log_rollback_event(agent_name, from_version, to_version, timestamp=datetime.now())

    print("✅ Rollback complete. Traffic restored to {to_version}")
```

---

## Step 4.4: Post-Deployment Analysis

### 7-Day Report

```markdown
# Post-Deployment Report: customer-support v1.1.0

**Deployment Period**: June 5-12, 2025
**Rollout**: Staged (alpha → beta → full over 7 days)
**Status**: ✅ Successful deployment

## Performance vs Baseline

| Metric | Baseline (v1.0.0) | Production (v1.1.0) | Change | Target Met? |
|--------|-------------------|---------------------|--------|-------------|
| Success rate | 87.0% | 93.5% | +6.5pp | ✅ Yes (+15% target) |
| Avg corrections | 2.3 | 1.5 | -0.8 (-35%) | ✅ Yes (-25% target) |
| Tool efficiency | 72% | 86% | +14pp | ✅ Yes (+10pp target) |
| User satisfaction | 8.2 | 8.5 | +0.3 | ✅ Yes (+0.2 target) |
| Response latency (p95) | 450ms | 445ms | -5ms | ✅ Yes (<10% increase) |
| Cost per task | $0.042 | $0.044 | +$0.002 | ✅ Yes (<5% increase) |

## Rollout Timeline

- **Day 1-2 (Alpha 10%)**: No issues, metrics stable
- **Day 3 (Beta 20%)**: Success rate +7pp, proceeded
- **Day 4 (Beta 50%)**: User satisfaction +0.4, proceeded
- **Day 5 (Full 100%)**: Full rollout, continued monitoring
- **Day 6-12**: Sustained improvement, no regressions

## User Feedback

**Positive (87%)**:
- "Responses are more helpful and to the point" (34 mentions)
- "Got my answer faster" (28 mentions)
- "Better understanding of complex questions" (19 mentions)

**Negative (13%)**:
- "Slightly longer responses" (8 mentions - addressed in v1.1.1)
- Minor UI feedback (unrelated to agent)

## Incidents

- **None**: No rollbacks, no critical errors, no safety violations

## ROI Analysis

**Investment**:
- Engineering time: 16 hours
- Testing/validation: 7 days
- Rollout monitoring: 7 days

**Returns** (monthly, extrapolated):
- User satisfaction improvement: +0.3 points
- Reduced corrections: 2000 hours saved (user + support time)
- Increased success rate: 750 fewer escalations
- Estimated value: $35,000/month

**ROI**: $35,000 / $4,800 ≈ 7.3x return

## Lessons Learned

1. **Tool decision tree** had highest impact (+14pp tool efficiency)
2. **Pricing examples** eliminated most pricing query failures
3. **Conciseness guideline** slightly increased response length (monitor)

## Next Optimization Cycle

Planned for: July 2025
Focus areas:
- Response verbosity (8 user comments)
- Edge case handling for international shipping
- Integration with new returns API
```

---

## Step 4.5: Continuous Monitoring

### Weekly Review

```python
def generate_weekly_report(agent_name: str):
    """Automated weekly performance summary"""
    last_week = get_metrics(agent_name, days=7)
    prev_week = get_metrics(agent_name, days=14, offset=7)

    report = {
        'week_ending': datetime.now().date(),
        'total_tasks': len(last_week['tasks']),
        'metrics': {
            'success_rate': {
                'current': last_week['success_rate'],
                'previous': prev_week['success_rate'],
                'trend': 'improving' if last_week['success_rate'] > prev_week['success_rate'] else 'declining'
            },
            # ... other metrics
        },
        'anomalies': detect_anomalies(last_week),
        'recommendations': generate_recommendations(last_week, prev_week)
    }

    return report
```

---

## Checklist

- [ ] Rollout plan created with stages
- [ ] Monitoring dashboard configured
- [ ] Alert thresholds set
- [ ] Rollback procedure tested
- [ ] Team notified of deployment schedule
- [ ] Alpha stage (10%) completed successfully
- [ ] Beta stages (20%, 50%) completed
- [ ] Full rollout (100%) deployed
- [ ] 7-day monitoring period completed
- [ ] Post-deployment report generated
- [ ] Lessons learned documented
- [ ] Next optimization cycle planned

---

## Common Issues

**Issue**: Metrics degradation in alpha
**Solution**: Roll back immediately, analyze root cause, iterate on prompt

**Issue**: Inconsistent metrics across stages
**Solution**: Check sample size, wait for statistical significance

**Issue**: User complaints despite good metrics
**Solution**: Qualitative analysis, manual review of flagged cases

---

**See also**:
- [Phase 3: Testing](phase-3-testing.md) - Previous phase
- [Continuous Improvement](agent-optimization-guide.md#continuous-improvement) - Next steps
- [Success Metrics](success-metrics.md) - Metric definitions
