# Phase 1: Performance Analysis

## Overview

Phase 1 establishes the quantitative baseline for agent performance. This phase is critical—you can't improve what you don't measure.

**Duration**: 1-2 hours
**Prerequisites**: Agent must have ≥7 days of production usage
**Outputs**: Baseline metrics report, failure mode classification, improvement recommendations

---

## Step 1.1: Gather Performance Data

### Using context-manager Agent

```bash
# Invoke context-manager to analyze 30 days of agent performance
/improve-agent <agent-name> --mode=check
```

**What it collects**:
- Task completion rate (successful vs failed tasks)
- Response accuracy (requires evaluation rubric)
- Tool usage efficiency (correct tool / total tool calls)
- User satisfaction indicators (ratings, corrections, retries)
- Response latency (p50, p95, p99)
- Token consumption per task

### Manual Data Collection (if context-manager unavailable)

**From logs**:
```python
import json
from datetime import datetime, timedelta

def analyze_agent_logs(agent_name, days=30):
    cutoff = datetime.now() - timedelta(days=days)

    tasks = load_agent_logs(agent_name, since=cutoff)

    metrics = {
        'total_tasks': len(tasks),
        'successful': sum(1 for t in tasks if t['status'] == 'success'),
        'failed': sum(1 for t in tasks if t['status'] == 'failed'),
        'avg_latency': np.mean([t['latency_ms'] for t in tasks]),
        'avg_tokens': np.mean([t['tokens_used'] for t in tasks]),
        'user_corrections': sum(t.get('corrections', 0) for t in tasks)
    }

    metrics['success_rate'] = metrics['successful'] / metrics['total_tasks']
    metrics['avg_corrections_per_task'] = metrics['user_corrections'] / metrics['total_tasks']

    return metrics
```

**From user feedback**:
```sql
-- Query user satisfaction ratings
SELECT
    AVG(rating) as avg_satisfaction,
    COUNT(*) as total_ratings,
    SUM(CASE WHEN rating >= 8 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as promoter_score
FROM user_feedback
WHERE agent_name = 'customer-support'
  AND created_at >= NOW() - INTERVAL '30 days';
```

---

## Step 1.2: User Feedback Pattern Analysis

### Identify Correction Patterns

**What to look for**:
- Where users consistently modify agent outputs
- Common clarification requests
- Frequent task abandonments
- Repetitive follow-up questions

**Analysis script**:
```python
def analyze_correction_patterns(agent_logs):
    corrections = [t for t in agent_logs if t.get('user_edited_output')]

    # Group by correction type
    correction_types = {}
    for task in corrections:
        task_type = task['type']
        correction_reason = classify_correction(task['original'], task['edited'])

        if task_type not in correction_types:
            correction_types[task_type] = []
        correction_types[task_type].append(correction_reason)

    # Find patterns
    patterns = []
    for task_type, reasons in correction_types.items():
        most_common = Counter(reasons).most_common(3)
        patterns.append({
            'task_type': task_type,
            'top_issues': most_common,
            'frequency': len(reasons) / len(corrections)
        })

    return sorted(patterns, key=lambda x: x['frequency'], reverse=True)
```

**Example output**:
```json
{
  "correction_patterns": [
    {
      "task_type": "pricing_query",
      "top_issues": [
        ["misunderstood_complex_pricing", 12],
        ["missed_discount_eligibility", 8],
        ["incorrect_currency_conversion", 5]
      ],
      "frequency": 0.15
    }
  ]
}
```

---

## Step 1.3: Failure Mode Classification

### Taxonomy

**1. Instruction Misunderstanding (25-35% of failures)**
- **Symptoms**: Agent does wrong task despite clear instructions
- **Examples**:
  - User asks for order status, agent provides product info
  - User wants refund, agent explains return policy (doesn't initiate refund)
- **Root cause**: Ambiguous role definition, unclear prioritization

**2. Output Format Errors (15-20%)**
- **Symptoms**: Correct content, wrong structure
- **Examples**:
  - Returns plain text when JSON expected
  - Markdown table when CSV requested
  - Missing required fields in structured output
- **Root cause**: Inconsistent examples, vague format specs

**3. Context Loss (10-15%)**
- **Symptoms**: Forgets earlier conversation context
- **Examples**:
  - Asks for information already provided
  - Contradicts previous statements
  - Loses track of conversation goal
- **Root cause**: Long prompts, poor summarization, no memory management

**4. Tool Misuse (20-25%)**
- **Symptoms**: Wrong tool selected or incorrect parameters
- **Examples**:
  - Uses search when should use direct lookup
  - Calls tool with missing required parameters
  - Doesn't call tool when should
- **Root cause**: Unclear tool descriptions, insufficient examples

**5. Constraint Violations (5-10%)**
- **Symptoms**: Violates safety, business logic, or preferences
- **Examples**:
  - Promises features not yet released
  - Shares confidential information
  - Violates company policies
- **Root cause**: Constraints not prominent, conflicts unresolved

**6. Edge Case Failures (5-10%)**
- **Symptoms**: Fails on unusual inputs
- **Examples**:
  - Empty input handling
  - Extreme values (very large numbers)
  - Unexpected data types
- **Root cause**: Training only on common cases

### Classification Script

```python
def classify_failure(task):
    """Classify failure into one of 6 categories"""

    if has_wrong_tool_usage(task):
        return "tool_misuse"
    elif has_format_error(task):
        return "output_format_error"
    elif has_context_loss(task):
        return "context_loss"
    elif violates_constraints(task):
        return "constraint_violation"
    elif is_edge_case(task):
        return "edge_case_failure"
    else:
        return "instruction_misunderstanding"

def analyze_failures(failed_tasks):
    classification = {}

    for task in failed_tasks:
        category = classify_failure(task)
        if category not in classification:
            classification[category] = []
        classification[category].append(task)

    # Generate report
    report = {
        'total_failures': len(failed_tasks),
        'by_category': {}
    }

    for category, tasks in classification.items():
        report['by_category'][category] = {
            'count': len(tasks),
            'percentage': len(tasks) / len(failed_tasks) * 100,
            'examples': [t['task_id'] for t in tasks[:3]]
        }

    return report
```

---

## Step 1.4: Baseline Performance Report

### Generate Comprehensive Report

**Report Structure**:
```json
{
  "agent_name": "customer-support",
  "version": "1.0.0",
  "evaluation_period": {
    "start": "2025-05-01",
    "end": "2025-05-31",
    "total_tasks": 1247,
    "total_users": 342
  },
  "performance_metrics": {
    "success_rate": 0.87,
    "avg_corrections_per_task": 2.3,
    "tool_call_efficiency": 0.72,
    "user_satisfaction": 8.2,
    "response_latency": {
      "p50": 285,
      "p95": 450,
      "p99": 780
    },
    "token_consumption": {
      "avg_per_task": 1250,
      "total": 1558750
    }
  },
  "failure_modes": {
    "instruction_misunderstanding": {
      "count": 41,
      "percentage": 25.3,
      "top_examples": [
        "Pricing queries misinterpreted (15 cases)",
        "Refund vs return confusion (12 cases)",
        "Account access vs info requests (8 cases)"
      ]
    },
    "tool_misuse": {
      "count": 33,
      "percentage": 20.4,
      "top_examples": [
        "get_product_info instead of get_order_status (18 cases)",
        "Missing order_id parameter (10 cases)",
        "search_docs when direct tool available (5 cases)"
      ]
    }
  },
  "user_feedback_themes": [
    {
      "theme": "Overly verbose responses",
      "frequency": 47,
      "sentiment": "negative"
    },
    {
      "theme": "Helpful and empathetic",
      "frequency": 213,
      "sentiment": "positive"
    }
  ],
  "recommendations": [
    {
      "priority": "high",
      "issue": "Tool selection accuracy (72% vs target 90%)",
      "fix": "Add tool decision tree and 5 few-shot examples",
      "expected_impact": "+15% tool efficiency"
    },
    {
      "priority": "high",
      "issue": "Complex pricing query handling (15% failure rate)",
      "fix": "Add 3 few-shot examples for pricing scenarios",
      "expected_impact": "+10% success rate"
    },
    {
      "priority": "medium",
      "issue": "Response verbosity (47 user complaints)",
      "fix": "Add conciseness instruction with example",
      "expected_impact": "+0.3 satisfaction points"
    }
  ]
}
```

### Visualization

**Create dashboard**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_baseline(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Success rate over time
    axes[0, 0].plot(metrics['daily_success_rate'])
    axes[0, 0].set_title('Success Rate Trend')
    axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='Target')

    # Failure mode distribution
    failure_data = metrics['failure_modes']
    axes[0, 1].pie(
        [f['count'] for f in failure_data.values()],
        labels=failure_data.keys(),
        autopct='%1.1f%%'
    )
    axes[0, 1].set_title('Failure Mode Distribution')

    # Response latency distribution
    axes[1, 0].hist(metrics['latencies'], bins=50)
    axes[1, 0].axvline(x=metrics['latency_p95'], color='r', label='p95')
    axes[1, 0].set_title('Response Latency Distribution')

    # User satisfaction trend
    axes[1, 1].plot(metrics['daily_satisfaction'])
    axes[1, 1].set_title('User Satisfaction Trend')

    plt.tight_layout()
    plt.savefig('baseline_metrics.png')
```

---

## Checklist

Before moving to Phase 2, ensure:

- [ ] Collected ≥30 days of performance data
- [ ] Calculated all primary metrics (success rate, corrections, tool efficiency, satisfaction, latency, tokens)
- [ ] Analyzed ≥100 user feedback instances
- [ ] Classified all failures into 6 categories
- [ ] Identified top 3 improvement opportunities
- [ ] Generated baseline report (JSON + visualization)
- [ ] Saved baseline to `.metrics/<agent>-baseline-YYYY-MM-DD.json`
- [ ] Stakeholders reviewed and approved baseline

---

## Common Pitfalls

1. **Insufficient data**: <100 tasks → not statistically significant
2. **Biased sampling**: Only analyzing successful tasks
3. **Ignoring seasonality**: Holiday traffic patterns differ
4. **No user feedback**: Metrics without qualitative context
5. **Premature optimization**: Optimizing before understanding root causes

---

## Tools

- **context-manager agent**: Automated performance analysis
- **Jupyter notebooks**: Interactive exploration
- **SQL**: Query production databases
- **Python libraries**: pandas, numpy, matplotlib, scipy
- **A/B testing platforms**: Optimizely, LaunchDarkly
- **Analytics**: Amplitude, Mixpanel, custom dashboards

---

## Next Steps

Once baseline is established:
1. Review with team to prioritize improvements
2. Proceed to [Phase 2: Prompt Engineering](phase-2-prompts.md)
3. Track improvement over baseline throughout optimization

---

**See also**:
- [Agent Optimization Guide](agent-optimization-guide.md) - Complete methodology
- [Success Metrics](success-metrics.md) - Metric definitions and targets
- [Phase 2: Prompt Engineering](phase-2-prompts.md) - Next phase
