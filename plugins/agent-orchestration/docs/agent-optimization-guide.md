# Agent Optimization Guide

## Overview

This comprehensive guide covers systematic improvement of AI agents through data-driven performance analysis, advanced prompt engineering, rigorous testing, and safe deployment practices.

**Use this guide when**: You want to understand the complete methodology behind agent optimization, including theoretical foundations and advanced techniques.

**Quick start**: For practical, step-by-step instructions, see the [/improve-agent command documentation](../commands/improve-agent.md).

---

## Table of Contents

1. [Performance Analysis](#performance-analysis)
2. [Prompt Engineering](#prompt-engineering)
3. [Testing & Validation](#testing--validation)
4. [Deployment & Monitoring](#deployment--monitoring)
5. [Continuous Improvement](#continuous-improvement)

---

## Performance Analysis

### Principles

Agent optimization is **data-driven**, not intuition-driven. Effective optimization requires:
- **Quantitative baselines**: Measure before improving
- **Pattern identification**: Find systematic issues, not one-off failures
- **Root cause analysis**: Fix underlying problems, not symptoms
- **Statistical significance**: Ensure improvements aren't random noise

### Metrics Collection

**Primary Metrics**:
- **Task completion rate**: % of tasks completed successfully
- **Response accuracy**: Correctness of outputs (requires evaluation rubric)
- **Tool usage efficiency**: Correct tool selected / Total tool calls
- **User satisfaction**: Direct feedback or proxy metrics (corrections, retries)
- **Response latency**: Time from user request to final response (p50, p95, p99)
- **Token consumption**: Tokens used per task (cost proxy)

**Secondary Metrics**:
- Hallucination rate (factual errors per response)
- Consistency score (alignment with previous responses)
- Format compliance (matches specified output structure)
- Safety violations (constraint breaches)
- Context retention (performance degradation over long conversations)

**Data Collection Methods**:
1. **Logging**: Capture all agent interactions with metadata
2. **User feedback**: Explicit ratings, implicit signals (corrections, retries)
3. **Automated evaluation**: Run test suite, compare outputs to ground truth
4. **Production monitoring**: Real-time dashboards, alerting on anomalies

### Failure Mode Classification

**Taxonomy of Agent Failures**:

**1. Instruction Misunderstanding**
- **Symptom**: Agent performs wrong task despite clear instructions
- **Root causes**: Ambiguous role definition, unclear examples, conflicting constraints
- **Fix**: Refine role description, add negative examples ("don't do X")

**2. Output Format Errors**
- **Symptom**: Correct content but wrong structure (e.g., JSON when should be Markdown)
- **Root causes**: Inconsistent examples, vague format specification
- **Fix**: Add explicit format templates, validation examples

**3. Context Loss**
- **Symptom**: Performance degrades in long conversations, forgets earlier context
- **Root causes**: Prompt too long, relevant info buried, no memory management
- **Fix**: Implement summarization, priority-based context retention

**4. Tool Misuse**
- **Symptom**: Calls wrong tool, or correct tool with wrong parameters
- **Root causes**: Unclear tool descriptions, insufficient examples, no decision tree
- **Fix**: Tool selection flowchart, tool-specific few-shot examples

**5. Constraint Violations**
- **Symptom**: Violates safety rules, business logic, or user preferences
- **Root causes**: Constraints not prominent, conflicts between constraints
- **Fix**: Constitutional AI, explicit constraint checking

**6. Edge Case Handling**
- **Symptom**: Fails on unusual inputs (empty data, extreme values, unexpected types)
- **Root causes**: Training/examples only cover common cases
- **Fix**: Adversarial testing, edge case examples

### Baseline Performance Report

Generate a structured baseline to track improvements:

```json
{
  "agent_name": "customer-support",
  "version": "1.0.0",
  "evaluation_period": {
    "start": "2025-05-01",
    "end": "2025-05-31",
    "total_tasks": 1247
  },
  "metrics": {
    "success_rate": 0.87,
    "avg_corrections_per_task": 2.3,
    "tool_call_efficiency": 0.72,
    "user_satisfaction": 8.2,
    "response_latency_p95_ms": 450,
    "token_efficiency_ratio": 1.15
  },
  "top_failure_modes": [
    {
      "type": "instruction_misunderstanding",
      "frequency": 0.15,
      "example": "Misinterprets complex pricing queries"
    },
    {
      "type": "tool_misuse",
      "frequency": 0.08,
      "example": "Uses get_product_info instead of get_order_status"
    }
  ],
  "recommendations": [
    "Add few-shot examples for pricing scenarios",
    "Create tool decision tree in prompt"
  ]
}
```

---

## Prompt Engineering

### Chain-of-Thought (CoT) Enhancement

**Principle**: Make reasoning explicit to improve accuracy and debuggability.

**Before**:
```
Analyze the user's order status and provide an update.
```

**After (with CoT)**:
```
Analyze the user's order status step-by-step:

1. Extract order ID from user message
2. Call get_order_status(order_id) tool
3. Interpret the status code:
   - "shipped" → Provide tracking info
   - "processing" → Estimate delivery date
   - "cancelled" → Explain reason and next steps
4. Format response with empathy and clarity

Before responding, verify:
- Did I use the correct tool?
- Is the information accurate?
- Have I addressed the user's concern?
```

**Benefits**:
- Higher accuracy (agent self-corrects reasoning errors)
- Easier debugging (see where reasoning fails)
- Better generalization (explicit logic applies to new cases)

**When to use**: Complex tasks requiring multi-step reasoning, decisions with ambiguity, tasks prone to errors.

### Few-Shot Example Optimization

**Principle**: Show, don't just tell. High-quality examples teach patterns better than lengthy instructions.

**Example Selection Criteria**:
1. **Diversity**: Cover common use cases and important edge cases
2. **Quality**: Only use successful examples (unless showing what NOT to do)
3. **Clarity**: Annotate key decision points
4. **Relevance**: Match the task distribution (more examples for frequent tasks)
5. **Ordering**: Simple → Complex (scaffold learning)

**Example Structure**:
```
Example 1: Simple pricing query
User: "How much does the Pro plan cost?"
Agent Reasoning:
  - Task: Pricing information (common)
  - Tool: get_pricing_info(plan="Pro")
  - Format: Direct answer with value proposition
Agent Response: "The Pro plan costs $29/month. It includes unlimited projects,
priority support, and advanced analytics. Would you like to upgrade?"

Why this works: Concise, uses correct tool, adds value with benefits.

Example 2: Complex multi-item order status
[Detailed example with more complexity]
```

**Negative Examples** (what not to do):
```
❌ Bad Example:
User: "Where's my order?"
Agent: "Your order is processing."

Why this fails:
- Doesn't ask for order ID (missing information)
- No call to get_order_status tool
- Generic response without specifics
- No empathy or next steps

✅ Correct approach:
"I'd be happy to check your order status! Could you provide your order number?
It should be in your confirmation email and starts with 'ORD-'."
```

### Role Definition Refinement

**Principle**: Clear identity leads to consistent behavior.

**Components of a Strong Role**:
1. **Core purpose** (one sentence)
2. **Expertise domains** (what you know)
3. **Behavioral traits** (how you interact)
4. **Tool proficiency** (what actions you can take)
5. **Constraints** (what you must NOT do)
6. **Success criteria** (what good looks like)

**Example Role Definition**:
```
# Role: Customer Support Agent

## Core Purpose
Resolve customer inquiries efficiently and empathetically while maintaining high satisfaction.

## Expertise
- Product features and pricing (all tiers)
- Order processing and shipping policies
- Billing and account management
- Troubleshooting common technical issues

## Behavioral Traits
- Empathetic: Acknowledge user frustration, validate concerns
- Concise: Answer directly, avoid unnecessary details
- Proactive: Anticipate follow-up questions, offer related help
- Professional: Maintain friendly but business-appropriate tone

## Available Tools
- get_order_status(order_id) → Track orders
- get_pricing_info(plan) → Retrieve pricing
- search_docs(query) → Find help articles
- create_support_ticket(details) → Escalate complex issues

## Constraints
- Never promise features not yet released
- Don't provide refunds (escalate to human agent)
- Don't share other customers' information
- Always verify identity before discussing account details

## Success Criteria
- User's question fully answered in first response (>80%)
- No safety violations or constraint breaches (100%)
- User satisfaction rating >8/10
- Response time <30 seconds (p95)
```

### Constitutional AI Integration

**Principle**: Build self-correction directly into the agent's reasoning process.

**Constitutional Principles** (agent's internal checklist):
1. **Accuracy**: "Before responding, verify facts from reliable sources"
2. **Bias check**: "Review response for harmful stereotypes or unfair treatment"
3. **Completeness**: "Have I fully addressed the user's request?"
4. **Format compliance**: "Does my response match the required structure?"
5. **Consistency**: "Is this aligned with my previous responses to similar questions?"

**Implementation** (critique-and-revise loop):
```
Step 1: Generate initial response
Step 2: Self-critique against constitutional principles
Step 3: Identify violations
Step 4: Revise response to fix issues
Step 5: Validate and output
```

**Example**:
```
User: "What's the best way to lose weight fast?"

Initial Response: "Try the keto diet and intermittent fasting for rapid weight loss!"

Self-Critique:
- Principle 1 (Accuracy): ⚠️ "best" is subjective, "fast" weight loss can be unsafe
- Principle 2 (Bias): ✅ No harmful stereotypes detected
- Principle 3 (Completeness): ⚠️ Doesn't mention consulting doctor, individual differences
- Principle 5 (Consistency): ⚠️ Previous health advice emphasized gradual, sustainable changes

Revised Response: "Sustainable weight loss typically involves a balanced diet and
regular exercise, tailored to your individual needs. I'd recommend consulting a
healthcare provider or registered dietitian for personalized advice. Rapid weight
loss approaches may not be safe or effective long-term."
```

### Output Format Tuning

**Principle**: Structure enables consistency and downstream processing.

**Format Strategies**:
1. **Templates for common tasks**: Pre-defined structures reduce variability
2. **Dynamic formatting**: Adjust detail level based on query complexity
3. **Progressive disclosure**: Start simple, offer more detail on request
4. **Markdown optimization**: Use headings, lists, code blocks for readability
5. **JSON for structured data**: Enables programmatic consumption

**Example Templates**:
```markdown
## Order Status Response Template
**Order ID**: {order_id}
**Status**: {status}
**Expected Delivery**: {delivery_date}

**Next Steps**:
- {action_1}
- {action_2}

[Tracking Link]({tracking_url})
```

---

## Testing & Validation

### Test Suite Development

**Test Categories**:
1. **Golden path**: Common successful scenarios (smoke tests)
2. **Regression tests**: Previously failed tasks (ensure fixes stick)
3. **Edge cases**: Boundary conditions, unusual inputs
4. **Stress tests**: Complex multi-step tasks, long contexts
5. **Adversarial tests**: Inputs designed to break the agent
6. **Cross-domain tests**: Tasks combining multiple capabilities

**Example Test Suite Structure**:
```
tests/
├── golden_path/
│   ├── simple_pricing_query.json
│   ├── order_status_check.json
│   └── product_recommendation.json
├── edge_cases/
│   ├── empty_input.json
│   ├── extremely_long_query.json
│   └── ambiguous_request.json
├── regression/
│   ├── issue_123_tool_selection.json
│   └── issue_145_format_error.json
└── adversarial/
    ├── prompt_injection_attempt.json
    └── constraint_violation_test.json
```

### A/B Testing Framework

**Protocol**:
1. **Random assignment**: 50% traffic to version A, 50% to version B
2. **Minimum sample size**: ≥100 tasks per variant for statistical significance
3. **Duration**: Run until significance achieved or 7 days (whichever comes first)
4. **Blinding**: Evaluators don't know which version they're rating
5. **Statistical test**: Two-proportion z-test for success rate, t-test for continuous metrics

**Statistical Significance**:
```python
from scipy import stats

# Success rates: A=87%, B=94%, n=120 each
successes_a, successes_b = 104, 113
n = 120

# Two-proportion z-test
z_stat, p_value = stats.proportions_ztest(
    [successes_a, successes_b],
    [n, n]
)

if p_value < 0.05:
    print(f"Significant improvement (p={p_value:.4f})")

# Effect size (Cohen's h)
p_a, p_b = successes_a/n, successes_b/n
cohens_h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
print(f"Effect size: {cohens_h:.3f}")  # h=0.2 small, 0.5 medium, 0.8 large
```

### Human Evaluation Protocol

**Structured Rubric**:
```
Dimension 1: Correctness (0-5)
0 = Completely wrong
1 = Mostly wrong, minor correct elements
2 = Partially correct, significant errors
3 = Mostly correct, minor errors
4 = Correct with minor imperfections
5 = Completely correct

Dimension 2: Helpfulness (0-5)
[Similar scale]

Dimension 3: Format Compliance (0-5)
[Similar scale]

Overall Score = (Correctness × 0.5) + (Helpfulness × 0.3) + (Format × 0.2)
```

**Inter-Rater Reliability**:
- Use Cohen's Kappa or Krippendorff's Alpha
- Target: κ > 0.7 (substantial agreement)
- If low agreement: Refine rubric, add examples, retrain evaluators

---

## Deployment & Monitoring

### Version Management

**Semantic Versioning** (MAJOR.MINOR.PATCH):
- **MAJOR**: Significant capability changes (new tools, different behavior)
- **MINOR**: Prompt improvements, new examples (backward compatible)
- **PATCH**: Bug fixes, minor adjustments

**Version History**:
```
agents/customer-support/
├── v1.0.0.md  (baseline)
├── v1.1.0.md  (added pricing examples)
├── v1.1.1.md  (fixed tool selection bug)
├── v1.2.0.md  (added CoT reasoning)
└── v2.0.0.md  (added new escalation tool)
```

**Changelog**:
```markdown
# customer-support v1.2.0 (2025-06-01)

## Changes
- Added chain-of-thought reasoning for complex queries
- Improved few-shot examples (3 → 5 examples)
- Enhanced role definition with success criteria

## Metrics (vs v1.1.1)
- Success rate: 87% → 94% (+7pp, p<0.001)
- Avg corrections: 2.3 → 1.4 (-39%, p<0.01)
- User satisfaction: 8.2 → 8.9 (+0.7, p<0.05)

## Breaking Changes
None (backward compatible)
```

### Staged Rollout

**Progressive Deployment Strategy**:
1. **Alpha (5% traffic)**: Internal team testing, 24-hour monitoring
2. **Beta (20% traffic)**: Selected early-adopter users, 48-hour monitoring
3. **Canary (50% traffic)**: Half of production, 72-hour monitoring
4. **Full (100% traffic)**: Complete rollout after validation

**Rollout Decision Gates**:
```
Alpha → Beta:
- ✅ No critical errors
- ✅ Success rate ≥ baseline
- ✅ No safety violations

Beta → Canary:
- ✅ Success rate improves ≥10%
- ✅ User satisfaction ≥ baseline
- ✅ No performance regressions

Canary → Full:
- ✅ Statistical significance achieved (p<0.05)
- ✅ All metrics green for 72 hours
- ✅ User feedback positive
```

### Rollback Procedures

**Automatic Rollback Triggers**:
- Success rate drops >10% from baseline
- Critical errors increase >5%
- Safety violations detected
- Response latency increases >50%
- User complaints spike (≥3 in 1 hour)

**Rollback Process**:
1. **Detection**: Monitoring system triggers alert
2. **Immediate action**: Automatic traffic routing to previous version
3. **Investigation**: Analyze logs, user feedback, error patterns
4. **Root cause**: Identify what went wrong
5. **Fix and re-test**: Address issues in dev/staging
6. **Gradual re-rollout**: Start at alpha again

### Continuous Monitoring

**Real-Time Dashboards**:
- Success rate (rolling 1-hour, 24-hour, 7-day)
- Error rate and top error types
- Response latency (p50, p95, p99)
- Tool usage patterns
- User satisfaction (recent ratings)
- Token consumption and cost

**Alerting Thresholds**:
```yaml
alerts:
  critical:
    - success_rate < 0.75  # vs baseline 0.87
    - error_rate > 0.15
    - safety_violations > 0
  warning:
    - success_rate < 0.80
    - latency_p95 > 600ms  # vs baseline 450ms
    - cost_per_task > $0.10
  info:
    - new_user_feedback_available
    - weekly_performance_summary
```

---

## Continuous Improvement

### Improvement Cadence

**Weekly**:
- Review metrics dashboard
- Triage new failure cases
- Collect user feedback
- Prioritize improvements

**Monthly**:
- Analyze trends and patterns
- Plan optimization sprint
- Update test suite
- Benchmark vs competitors

**Quarterly**:
- Major version updates
- Capability additions
- Architecture reviews
- Team retrospectives

### Learning from Production

**Feedback Loop**:
```
Production Usage
    ↓
Logging & Monitoring
    ↓
Pattern Analysis
    ↓
Hypothesis Formation
    ↓
Prompt Improvements
    ↓
A/B Testing
    ↓
Deployment
    ↓
Production Usage (cycle continues)
```

**Pattern Mining**:
- Cluster similar failures (unsupervised learning)
- Identify common user intents not handled well
- Find emerging use cases
- Detect drift in user behavior

**Proactive Optimization**:
- Don't wait for failures, optimize good→great
- Benchmark against best-in-class agents
- Experiment with new techniques (retrieval augmentation, tool learning)
- Invest in long-term improvements (training data quality, model fine-tuning)

---

## Best Practices Summary

1. **Data-driven decisions**: Always measure, never guess
2. **Iterate quickly**: Small improvements compound
3. **Test rigorously**: Catch issues before production
4. **Deploy safely**: Gradual rollout with monitoring
5. **Learn continuously**: Production is your teacher
6. **Version everything**: Git for prompts, semantic versioning
7. **Document thoroughly**: Future you will thank you
8. **Prioritize impact**: Fix high-frequency issues first
9. **Balance speed and safety**: Move fast, but don't break things
10. **User-centric**: Optimize for user value, not vanity metrics

---

## Additional Resources

- [Phase 1: Performance Analysis](phase-1-analysis.md)
- [Phase 2: Prompt Engineering](phase-2-prompts.md)
- [Phase 3: Testing & Validation](phase-3-testing.md)
- [Phase 4: Deployment & Monitoring](phase-4-deployment.md)
- [Success Metrics Guide](success-metrics.md)
- [Prompt Engineering Techniques](prompt-techniques.md)

---

**Last Updated**: 2025-06-11
**Version**: 1.0.0
