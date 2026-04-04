---
name: agent-evaluation
description: Evaluate AI agent performance with benchmark design, metrics collection, A/B testing, regression detection, and quality scoring. Use when measuring agent accuracy, building eval suites, or tracking agent performance over time.
---

# Agent Evaluation

## Expert Agent

For evaluation orchestration and performance analysis, delegate to:

- **`orchestrator`**: Coordinates evaluation pipelines, manages test execution, and aggregates results.
  - *Location*: `plugins/agent-core/agents/orchestrator.md`

Comprehensive guide for measuring, benchmarking, and improving AI agent performance systematically.

---

## 1. Evaluation Framework Design

### Core Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **Test Cases** | Define inputs and expected outputs | Task descriptions with gold answers |
| **Metrics** | Quantify performance dimensions | Accuracy, latency, cost |
| **Judges** | Score agent outputs | Automated scorers, LLM judges, humans |
| **Pipeline** | Orchestrate end-to-end runs | CI-triggered evaluation suite |

### Framework Architecture

```python
from dataclasses import dataclass, field

@dataclass
class TestCase:
    id: str
    input_data: dict
    expected: dict
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy | medium | hard

@dataclass
class EvalResult:
    test_id: str
    agent_output: dict
    scores: dict[str, float]
    latency_ms: float
    token_count: int
    passed: bool
```

---

## 2. Metric Types

### Task Completion Metrics

| Metric | Definition | Range |
|--------|-----------|-------|
| **Success Rate** | Tasks completed correctly / total | 0-100% |
| **Partial Credit** | Weighted score for partially correct | 0-1.0 |
| **Step Accuracy** | Correct intermediate steps / total steps | 0-100% |
| **Tool Accuracy** | Correct tool calls / total tool calls | 0-100% |

### Quality Metrics

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| **Factual Accuracy** | Claims verified against ground truth | NLI model or human |
| **Coherence** | Logical flow and consistency | LLM judge (1-5 scale) |
| **Helpfulness** | Addresses the user need | Human rating (1-5) |
| **Safety** | No harmful or policy-violating content | Classifier + human |

### Efficiency Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Latency** | Time from request to final response | < 10s for interactive |
| **Token Usage** | Total input + output tokens | Minimize for cost |
| **Tool Call Count** | Number of tool invocations | Minimize for speed |
| **Retry Rate** | Failed attempts / total attempts | < 5% |

---

## 3. Benchmark Creation

### Benchmark Design Process

| Step | Action | Output |
|------|--------|--------|
| 1. Define scope | Choose tasks representative of production use | Task taxonomy |
| 2. Collect examples | Gather real user queries + expert solutions | Raw dataset |
| 3. Label ground truth | Expert annotation with inter-rater agreement | Gold standard |
| 4. Stratify | Balance by difficulty, category, edge cases | Balanced test set |
| 5. Validate | Run baseline agent, verify scores are meaningful | Calibrated benchmark |

### Stratification Rules

- Minimum 10 test cases per category and difficulty level.
- Count tag distribution and flag under-represented categories.
- Balance easy/medium/hard at roughly 20%/50%/30%.

### Anti-Patterns in Benchmarks

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Leaked test data | Inflated scores from memorization | Use held-out, date-gated data |
| Narrow coverage | Misses real failure modes | Stratify by category and difficulty |
| Static benchmarks | Scores saturate over time | Rotate 20% of cases quarterly |
| No baselines | Cannot interpret scores | Include naive + SOTA baselines |

---

## 4. Automated Evaluation Pipelines

### CI-Integrated Pipeline

Trigger evaluations on PRs that modify `prompts/`, `agents/`, or `tools/`. Steps:

1. Run evaluation suite against the PR branch.
2. Compare results to the baseline (main branch scores).
3. Flag regressions exceeding 5% on any primary metric.
4. Post a summary comment on the PR with pass/fail status.

### Regression Detection

Compare each metric between current and baseline. Flag any metric where `(baseline - current) / baseline > threshold`. Default threshold is 5%. Report the metric name, old value, new value, and percentage drop.

---

## 5. Human-in-the-Loop Evaluation

### When to Use Human Judges

| Scenario | Reason |
|----------|--------|
| Creative or open-ended tasks | No single correct answer |
| Safety-critical outputs | Automated classifiers miss nuance |
| New task categories | No automated metric exists yet |
| Calibrating LLM judges | Need human ground truth to tune |

### Human Evaluation Protocol

1. **Blind review**: Evaluators do not see which agent variant produced the output.
2. **Rubric scoring**: Provide a 1-5 scale rubric with concrete examples for each level.
3. **Inter-rater agreement**: Require >= 0.7 Cohen's kappa before trusting aggregated scores.
4. **Sampling**: Evaluate a random 10-20% subset when full review is too expensive.

---

## 6. A/B Testing for Agents

### Experiment Design

| Element | Requirement |
|---------|-------------|
| **Hypothesis** | "Prompt v2 improves success rate by >= 5%" |
| **Metric** | Primary: success rate. Secondary: latency, cost |
| **Split** | Random 50/50 user allocation |
| **Duration** | Minimum 500 samples per variant |

---

## 7. Evaluation Checklist

- [ ] Test cases cover all major task categories
- [ ] Ground truth labels validated by domain experts
- [ ] At least 3 metric types measured (completion, quality, efficiency)
- [ ] Baseline scores established for comparison
- [ ] Regression detection threshold defined (default 5%)
- [ ] CI pipeline triggers evaluation on prompt/agent changes
- [ ] Human evaluation protocol documented with rubric
- [ ] Benchmark rotated quarterly to prevent overfitting

---

## Related Skills

- `multi-agent-coordination` -- Agent workflows whose performance evaluation measures
- `agent-performance-optimization` -- Optimization techniques informed by evaluation results
