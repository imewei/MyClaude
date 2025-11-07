# Phase 3: Testing & Validation

## Overview

Phase 3 validates that prompt improvements actually work through rigorous testing and A/B comparison.

**Duration**: 3-7 days (includes data collection)
**Prerequisites**: Phase 2 completed, new prompt version ready
**Outputs**: Test results, statistical analysis, go/no-go decision

---

## Quick Start

```bash
# Execute Phase 3 testing
/improve-agent <agent-name> --phase=3

# Runs A/B test comparing current vs new version
# Collects data for minimum 100 tasks per variant
# Generates statistical analysis report
```

---

## Step 3.1: Test Suite Development

### Test Categories

**Golden Path Tests** (smoke tests):
```json
{
  "test_id": "pricing_simple_001",
  "category": "golden_path",
  "input": "How much does the Pro plan cost?",
  "expected_output_includes": ["$29/month", "Pro plan"],
  "expected_tools": ["get_pricing_info"],
  "max_latency_ms": 3000
}
```

**Regression Tests** (previously failed):
```json
{
  "test_id": "regression_tool_selection_123",
  "category": "regression",
  "input": "Where is my order ORD-123456?",
  "expected_tools": ["get_order_status"],
  "should_not_use_tools": ["get_product_info", "search_docs"],
  "original_failure": "Used wrong tool (get_product_info)"
}
```

**Edge Cases**:
```json
{
  "test_id": "edge_empty_input_001",
  "category": "edge_case",
  "input": "",
  "expected_behavior": "politely_ask_for_clarification",
  "should_not": "error_or_crash"
}
```

---

## Step 3.2: A/B Testing Framework

### Setup

```python
import random
from dataclasses import dataclass

@dataclass
class ABTest:
    test_id: str
    variant_a: str  # "customer-support-v1.0.0"
    variant_b: str  # "customer-support-v1.1.0"
    traffic_split: float = 0.5
    min_sample_size: int = 100
    max_duration_days: int = 7

def assign_variant(user_id: str, test: ABTest) -> str:
    """Consistent assignment based on user_id"""
    hash_value = hash(f"{test.test_id}:{user_id}")
    if (hash_value % 100) / 100 < test.traffic_split:
        return test.variant_a
    return test.variant_b
```

### Data Collection

```python
def collect_ab_metrics(test: ABTest, days: int = 7):
    results = {
        'variant_a': {'tasks': [], 'successes': 0, 'failures': 0},
        'variant_b': {'tasks': [], 'successes': 0, 'failures': 0}
    }

    # Collect tasks for both variants
    for task in get_agent_tasks(test.test_id, days=days):
        variant = task['variant']
        results[variant]['tasks'].append(task)
        if task['status'] == 'success':
            results[variant]['successes'] += 1
        else:
            results[variant]['failures'] += 1

    return results
```

---

## Step 3.3: Statistical Analysis

### Success Rate Comparison

```python
from scipy import stats
import numpy as np

def analyze_ab_test(results):
    """Two-proportion z-test for success rates"""
    n_a = len(results['variant_a']['tasks'])
    n_b = len(results['variant_b']['tasks'])
    successes_a = results['variant_a']['successes']
    successes_b = results['variant_b']['successes']

    # Success rates
    p_a = successes_a / n_a
    p_b = successes_b / n_b

    # Two-proportion z-test
    z_stat, p_value = stats.proportions_ztest(
        [successes_a, successes_b],
        [n_a, n_b]
    )

    # Effect size (Cohen's h)
    cohens_h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))

    # Determine significance
    is_significant = p_value < 0.05
    effect_interpretation = interpret_effect_size(cohens_h)

    return {
        'variant_a': {'n': n_a, 'success_rate': p_a},
        'variant_b': {'n': n_b, 'success_rate': p_b},
        'improvement': p_b - p_a,
        'p_value': p_value,
        'is_significant': is_significant,
        'cohens_h': cohens_h,
        'effect_size': effect_interpretation
    }

def interpret_effect_size(h):
    """Cohen's h interpretation"""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"
```

### Example Results

```python
{
  "variant_a": {"n": 127, "success_rate": 0.87},
  "variant_b": {"n": 131, "success_rate": 0.94},
  "improvement": 0.07,  # 7 percentage points
  "p_value": 0.023,     # p < 0.05 (significant!)
  "is_significant": true,
  "cohens_h": 0.42,
  "effect_size": "small-to-medium"
}
```

**Interpretation**: New version improves success rate from 87% to 94% with statistical significance (p=0.023). Effect size is small-to-medium, which is meaningful for production deployment.

---

## Step 3.4: Human Evaluation

### Evaluation Rubric

```python
@dataclass
class EvaluationScore:
    correctness: int       # 0-5: Is the answer correct?
    helpfulness: int       # 0-5: Does it help the user?
    format_compliance: int # 0-5: Matches format requirements?
    tone: int             # 0-5: Professional and empathetic?

    @property
    def overall(self) -> float:
        return (
            self.correctness * 0.4 +
            self.helpfulness * 0.3 +
            self.format_compliance * 0.2 +
            self.tone * 0.1
        )
```

### Blind Evaluation Protocol

```python
def prepare_blind_evaluation(variant_a_tasks, variant_b_tasks):
    """Mix and shuffle for blind evaluation"""
    all_tasks = []

    for task in variant_a_tasks[:20]:  # 20 samples per variant
        all_tasks.append({
            'task_id': task['id'],
            'input': task['user_message'],
            'output': task['agent_response'],
            'variant': 'A',  # Evaluator sees only A/B, not version numbers
            'actual_variant': 'v1.0.0'
        })

    for task in variant_b_tasks[:20]:
        all_tasks.append({
            'task_id': task['id'],
            'input': task['user_message'],
            'output': task['agent_response'],
            'variant': 'B',
            'actual_variant': 'v1.1.0'
        })

    random.shuffle(all_tasks)
    return all_tasks
```

### Inter-Rater Reliability

```python
from sklearn.metrics import cohen_kappa_score

def check_inter_rater_reliability(rater1_scores, rater2_scores):
    """Cohen's Kappa for agreement between evaluators"""
    kappa = cohen_kappa_score(rater1_scores, rater2_scores)

    if kappa > 0.8:
        return "excellent agreement"
    elif kappa > 0.6:
        return "substantial agreement"
    elif kappa > 0.4:
        return "moderate agreement - refine rubric"
    else:
        return "poor agreement - retrain evaluators"
```

---

## Go/No-Go Decision

### Decision Criteria

**Deploy new version if ALL true**:
- ✅ Statistical significance: p < 0.05
- ✅ Success rate improves ≥5 percentage points
- ✅ No regressions on critical metrics (latency, safety)
- ✅ Human evaluation shows improvement
- ✅ No unexpected edge case failures

**Iterate (don't deploy) if**:
- ⚠️ Improvement not statistically significant
- ⚠️ Effect size too small (Cohen's h < 0.2)
- ⚠️ Regressions detected (some metrics worse)

**Rollback immediately if**:
- ❌ Success rate decreases
- ❌ Safety violations occur
- ❌ Critical errors increase

### Decision Tree

```
Sample size ≥100 per variant?
├─ No → Continue collecting data
└─ Yes
   └─ Is improvement significant (p<0.05)?
      ├─ No → No-go (iterate on prompt)
      └─ Yes
         └─ Effect size ≥ small (h≥0.2)?
            ├─ No → No-go (improvement too small)
            └─ Yes
               └─ Any critical regressions?
                  ├─ Yes → No-go (fix regressions first)
                  └─ No → ✅ GO (proceed to Phase 4)
```

---

## Test Report Template

```markdown
# A/B Test Results: customer-support v1.0.0 vs v1.1.0

**Test Period**: May 28 - June 4, 2025 (7 days)
**Sample Size**: 127 (v1.0.0), 131 (v1.1.0)

## Key Metrics

| Metric | v1.0.0 | v1.1.0 | Change | p-value | Significant? |
|--------|--------|--------|--------|---------|--------------|
| Success rate | 87.0% | 94.0% | +7.0pp | 0.023 | ✅ Yes |
| Avg corrections | 2.3 | 1.4 | -0.9 | 0.008 | ✅ Yes |
| Tool efficiency | 72% | 87% | +15pp | <0.001 | ✅ Yes |
| User satisfaction | 8.2 | 8.6 | +0.4 | 0.041 | ✅ Yes |
| Response latency (p95) | 450ms | 425ms | -25ms | 0.312 | ❌ No |

## Effect Sizes

- Success rate: Cohen's h = 0.42 (small-to-medium)
- Tool efficiency: Cohen's h = 0.68 (medium)

## Human Evaluation (n=40, blind)

| Dimension | v1.0.0 | v1.1.0 | Winner |
|-----------|--------|--------|--------|
| Correctness | 4.1/5 | 4.6/5 | ✅ v1.1.0 |
| Helpfulness | 3.9/5 | 4.4/5 | ✅ v1.1.0 |
| Format | 4.3/5 | 4.5/5 | ≈ Tie |
| Tone | 4.5/5 | 4.6/5 | ≈ Tie |

Inter-rater reliability: κ=0.74 (substantial agreement)

## Decision: ✅ GO

**Rationale**:
- All primary metrics improved significantly
- No critical regressions
- Effect sizes meaningful for production
- Human evaluation confirms improvement

**Next Steps**:
1. Proceed to Phase 4 (staged deployment)
2. Start with 10% canary rollout
3. Monitor for 48 hours before full deployment
```

---

## Checklist

- [ ] Test suite created (≥50 test cases)
- [ ] A/B test configured (50/50 split)
- [ ] Collected ≥100 samples per variant
- [ ] Statistical analysis completed
- [ ] Human evaluation conducted (≥20 samples per variant)
- [ ] Go/no-go decision made
- [ ] Test report generated and reviewed
- [ ] If go: Phase 4 deployment plan ready
- [ ] If no-go: Iteration plan documented

---

**See also**:
- [Phase 2: Prompt Engineering](phase-2-prompts.md) - Previous phase
- [Phase 4: Deployment](phase-4-deployment.md) - Next phase
- [Success Metrics](success-metrics.md) - Metric definitions
