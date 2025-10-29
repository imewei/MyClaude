# Cognitive Biases Checklist

## Overview

This checklist provides systematic detection criteria, severity assessment, and mitigation strategies for common cognitive biases in AI reasoning and decision-making.

---

## Bias 1: Availability Bias

### Definition
Over-relying on recent, memorable, or easily accessible examples when making judgments or decisions.

### Detection Criteria

**Red Flags:**
- ❌ Using only recent examples
- ❌ Citing the same example repeatedly
- ❌ Ignoring older but relevant cases
- ❌ Overemphasizing memorable but atypical instances

**Detection Questions:**
1. Are examples drawn from a representative time period?
2. Are memorable/dramatic cases given undue weight?
3. Are easily accessible examples preferred over harder-to-recall but relevant ones?
4. Is the sample biased toward recent experience?

### Example Instances

**Example 1: Technology Choice**
```
Biased: "We should use React because it worked great in the last project"
Issue: Last project may not be representative
Unbiased: "Comparing React, Vue, and Angular for our requirements..."
```

**Example 2: Bug Diagnosis**
```
Biased: "This looks like the memory leak we had last week"
Issue: Recent memory leak is most available, may not be same issue
Unbiased: "Checking multiple possible causes: memory leak, connection pool, cache..."
```

### Severity Assessment

| Severity | Impact | Characteristics |
|----------|--------|----------------|
| **Low** | Minor | Single instance, limited scope, easily corrected |
| **Medium** | Moderate | Multiple instances, affects key decisions, systematic pattern |
| **High** | Significant | Pervasive, affects critical decisions, leads to poor outcomes |

### Mitigation Strategies

1. **Systematic Search**: Explicitly search for diverse examples spanning different time periods
2. **Balanced Sampling**: Intentionally include older, less memorable, and harder-to-access examples
3. **Conscious Awareness**: Flag when using recent or memorable examples and justify why they're representative
4. **Structured Retrieval**: Use search tools to find representative examples, not just recalled ones

### Mitigation Template
```markdown
When making decision based on examples:
1. [ ] Identified multiple examples from different time periods?
2. [ ] Included non-recent examples?
3. [ ] Verified examples are representative, not just memorable?
4. [ ] Searched systematically rather than relying on recall?
```

---

## Bias 2: Anchoring Bias

### Definition
Over-relying on the first piece of information encountered (the "anchor") when making decisions.

### Detection Criteria

**Red Flags:**
- ❌ First solution considered dominates final choice
- ❌ Initial estimate heavily influences final estimate
- ❌ Alternative options evaluated relative to first option
- ❌ Insufficient exploration after initial finding

**Detection Questions:**
1. Was the first option/estimate given disproportionate weight?
2. Were alternatives evaluated independently or relative to the anchor?
3. Did the final decision closely match the initial suggestion?
4. Was sufficient effort made to explore beyond the initial option?

### Example Instances

**Example 1: Effort Estimation**
```
Biased: "Initial estimate was 2 weeks, adjusted to 2.5 weeks after analysis"
Issue: Final estimate anchored to initial guess rather than independent analysis
Unbiased: "Analyzed tasks independently: 5 days + 6 days + 4 days = 3 weeks"
```

**Example 2: Architecture Decision**
```
Biased: "First thought was microservices, confirmed after brief analysis"
Issue: Confirmation of initial thought, not independent evaluation
Unbiased: "Evaluated monolith, modular monolith, microservices independently..."
```

### Severity Assessment

| Severity | Impact | Characteristics |
|----------|--------|----------------|
| **Low** | Minor | Slight preference for first option, alternatives still considered |
| **Medium** | Moderate | Clear anchoring effect, alternatives underweighted |
| **High** | Significant | Decision predetermined by anchor, alternatives superficially considered |

### Mitigation Strategies

1. **Defer Judgment**: Avoid forming opinions before thorough analysis
2. **Independent Evaluation**: Evaluate each alternative independently before comparing
3. **Reverse Order**: Consider alternatives in different orders to detect anchoring
4. **Blind Analysis**: Analyze without initial estimates or suggestions when possible

### Mitigation Template
```markdown
When evaluating options:
1. [ ] Deferred judgment until all options analyzed?
2. [ ] Evaluated each option independently first?
3. [ ] Checked if first option has disproportionate influence?
4. [ ] Considered options in multiple orders?
```

---

## Bias 3: Confirmation Bias

### Definition
Seeking, interpreting, or recalling information in a way that confirms pre-existing beliefs or hypotheses.

### Detection Criteria

**Red Flags:**
- ❌ Seeking only supporting evidence
- ❌ Dismissing contradictory evidence
- ❌ Interpreting ambiguous evidence as confirming
- ❌ Stopping search after finding confirming evidence

**Detection Questions:**
1. Was disconfirming evidence actively sought?
2. Were contradictory findings given fair consideration?
3. Was ambiguous evidence interpreted neutrally?
4. Did the search continue after finding confirming evidence?

### Example Instances

**Example 1: Performance Hypothesis**
```
Biased: "I think it's the database. Let me check query times... yes, queries are slow!"
Issue: Only checked database, confirmed initial hypothesis
Unbiased: "Profiling all components: database, network, CPU, memory to find bottleneck"
```

**Example 2: Code Review**
```
Biased: "This looks like good code. Tests pass, structure is clean, meets requirements"
Issue: Only looked for positive signals
Unbiased: "Checking security issues, edge cases, performance, maintainability concerns..."
```

### Severity Assessment

| Severity | Impact | Characteristics |
|----------|--------|----------------|
| **Low** | Minor | Slight preference for confirming evidence, disconfirming evidence considered |
| **Medium** | Moderate | Systematic search for confirmation, resistance to contradictory evidence |
| **High** | Significant | Active avoidance of disconfirming evidence, dismissal of contradictions |

### Mitigation Strategies

1. **Seek Disconfirmation**: Actively look for evidence that contradicts hypothesis
2. **Steel Man Argument**: Present strongest possible case against own position
3. **Neutral Interpretation**: Interpret ambiguous evidence neutrally, not as confirming
4. **Balanced Search**: Allocate equal effort to finding confirming and disconfirming evidence

### Mitigation Template
```markdown
When testing hypothesis:
1. [ ] Actively sought disconfirming evidence?
2. [ ] Gave contradictory evidence fair weight?
3. [ ] Interpreted ambiguous evidence neutrally?
4. [ ] Presented strongest argument against own position?
5. [ ] Considered alternative explanations seriously?
```

---

## Bias 4: Recency Bias

### Definition
Giving disproportionate weight to recent events or information over historical patterns.

### Detection Criteria

**Red Flags:**
- ❌ Recent events dominate analysis
- ❌ Historical data underweighted or ignored
- ❌ Trends extrapolated from recent data alone
- ❌ Long-term patterns not considered

**Detection Questions:**
1. Is recent data given more weight than justified?
2. Are historical patterns and trends considered?
3. Are long-term statistics balanced with recent events?
4. Could the recent event be an outlier?

### Example Instances

**Example 1: Performance Trend**
```
Biased: "Performance dropped yesterday, we have a serious issue"
Issue: Single recent data point, may be normal variance
Unbiased: "Yesterday's drop fits within normal variance over past 6 months"
```

**Example 2: Team Productivity**
```
Biased: "Last sprint was slow, team productivity is declining"
Issue: Overweighting recent sprint
Unbiased: "Last sprint below average, but 6-month trend shows consistent velocity"
```

### Severity Assessment

| Severity | Impact | Characteristics |
|----------|--------|----------------|
| **Low** | Minor | Slight overemphasis on recent data, historical data still considered |
| **Medium** | Moderate | Recent events dominate, historical patterns underweighted |
| **High** | Significant | Decisions based almost entirely on recent events, history ignored |

### Mitigation Strategies

1. **Historical Context**: Always check historical data and long-term trends
2. **Outlier Detection**: Determine if recent event is outlier or part of pattern
3. **Weighted Analysis**: Weight recent and historical data proportionally
4. **Moving Averages**: Use rolling averages to smooth recency effects

### Mitigation Template
```markdown
When analyzing trends or patterns:
1. [ ] Checked historical data and trends?
2. [ ] Determined if recent event is outlier?
3. [ ] Weighted recent vs historical data appropriately?
4. [ ] Used statistical measures (moving averages, etc.)?
```

---

## Bias 5: Selection Bias

### Definition
Drawing conclusions from non-representative samples due to biased selection criteria.

### Detection Criteria

**Red Flags:**
- ❌ Non-random sample selection
- ❌ Convenience sampling (easiest cases)
- ❌ Survivor bias (only successful cases)
- ❌ Self-selection effects

**Detection Questions:**
1. Is the sample representative of the population?
2. Were cases selected randomly or by convenience?
3. Are there systematic exclusions (e.g., only successful cases)?
4. Could selection criteria bias results?

### Example Instances

**Example 1: User Feedback**
```
Biased: "Users love the feature! (Based on emails from enthusiastic users)"
Issue: Self-selection - only enthusiastic users sent email
Unbiased: "Survey of random user sample shows mixed feedback: 65% positive, 35% negative"
```

**Example 2: Performance Analysis**
```
Biased: "Our optimization worked! (Tested on the specific slow case we optimized for)"
Issue: Tested only on case known to benefit
Unbiased: "Optimization results on diverse workload: 50% faster (avg), 10% faster (worst case)"
```

### Severity Assessment

| Severity | Impact | Characteristics |
|----------|--------|----------------|
| **Low** | Minor | Slight selection effect, results roughly representative |
| **Medium** | Moderate | Systematic selection bias, results moderately skewed |
| **High** | Significant | Severe selection bias, conclusions invalid for population |

### Mitigation Strategies

1. **Random Sampling**: Use random selection when possible
2. **Representative Sampling**: Ensure sample represents population characteristics
3. **Stratified Sampling**: Sample across relevant subgroups
4. **Survivor Bias Check**: Explicitly look for excluded or missing cases

### Mitigation Template
```markdown
When selecting examples or test cases:
1. [ ] Used random or systematic sampling?
2. [ ] Verified sample is representative?
3. [ ] Checked for systematic exclusions?
4. [ ] Included both successful and unsuccessful cases?
5. [ ] Considered selection criteria bias?
```

---

## Bias 6: Overconfidence Bias

### Definition
Excessive confidence in one's own abilities, knowledge, or predictions.

### Detection Criteria

**Red Flags:**
- ❌ Certainty without sufficient evidence
- ❌ Underestimation of complexity or risk
- ❌ Dismissal of alternatives without analysis
- ❌ Lack of uncertainty quantification

**Detection Questions:**
1. Is confidence level justified by evidence?
2. Are risks and uncertainties acknowledged?
3. Are alternative outcomes considered?
4. Is uncertainty quantified explicitly?

### Example Instances

**Example 1: Effort Estimate**
```
Biased: "This will definitely take 2 weeks"
Issue: Certainty without accounting for uncertainty
Unbiased: "Estimate: 2 weeks ±3 days (80% confidence), accounting for integration complexity"
```

**Example 2: Technical Decision**
```
Biased: "This architecture will definitely scale to 1M users"
Issue: Overconfidence without testing or analysis
Unbiased: "This architecture should scale to 1M based on X and Y, but needs load testing to confirm"
```

### Severity Assessment

| Severity | Impact | Characteristics |
|----------|--------|----------------|
| **Low** | Minor | Slight overconfidence, uncertainty somewhat acknowledged |
| **Medium** | Moderate | Significant overconfidence, risks underestimated |
| **High** | Significant | Extreme overconfidence, uncertainty and risks ignored |

### Mitigation Strategies

1. **Explicit Uncertainty**: Quantify confidence levels and uncertainty ranges
2. **Risk Analysis**: Identify and assess potential risks systematically
3. **Humble Language**: Use hedging language appropriately ("likely", "probably", "appears to")
4. **Peer Review**: Seek external review to challenge assumptions

### Mitigation Template
```markdown
When making predictions or estimates:
1. [ ] Quantified uncertainty explicitly?
2. [ ] Identified potential risks?
3. [ ] Used appropriate hedging language?
4. [ ] Considered what could go wrong?
5. [ ] Provided confidence intervals or ranges?
```

---

## Bias 7: Sunk Cost Fallacy

### Definition
Continuing a course of action because of previously invested resources, despite evidence it should be abandoned.

### Detection Criteria

**Red Flags:**
- ❌ Justifying continuation based on past investment
- ❌ Reluctance to abandon failing approach
- ❌ Escalating commitment despite poor outcomes
- ❌ "Too far to turn back" reasoning

**Detection Questions:**
1. Is past investment influencing current decision?
2. Would you choose this option if starting fresh?
3. Are future costs and benefits evaluated independently?
4. Is there reluctance to abandon based on sunk costs?

### Example Instances

**Example 1: Technology Choice**
```
Biased: "We've already spent 6 months with this framework, we can't switch now"
Issue: Past investment driving future decision
Unbiased: "Framework isn't meeting needs. Switching costs $X, continuing costs $Y..."
```

**Example 2: Feature Development**
```
Biased: "We've already built 80% of this feature, let's finish it"
Issue: Completion percentage driving decision, not user value
Unbiased: "Re-evaluating: feature provides little value, better to pivot despite 80% completion"
```

### Severity Assessment

| Severity | Impact | Characteristics |
|----------|--------|----------------|
| **Low** | Minor | Slight consideration of sunk costs, future evaluation still dominant |
| **Medium** | Moderate | Sunk costs significantly influence decision, future costs underweighted |
| **High** | Significant | Decision driven by sunk costs, future costs/benefits ignored |

### Mitigation Strategies

1. **Fresh Start Analysis**: Evaluate as if starting from scratch
2. **Future Focus**: Consider only future costs and benefits, ignore sunk costs
3. **Explicit Sunk Cost**: Identify and acknowledge sunk costs to neutralize influence
4. **Pivot Threshold**: Set criteria for abandoning approach independent of investment

### Mitigation Template
```markdown
When deciding whether to continue approach:
1. [ ] Evaluated decision as if starting fresh?
2. [ ] Considered only future costs and benefits?
3. [ ] Identified sunk costs explicitly?
4. [ ] Set abandonment criteria independent of past investment?
```

---

## Comprehensive Bias Assessment Process

### Step 1: Systematic Review

Go through decision-making or reasoning chain chronologically:
- Identify key decisions and judgments
- Note evidence used for each decision
- Document reasoning process
- Flag potential bias instances

### Step 2: Bias Checklist Application

For each decision, check all seven biases:
1. **Availability**: Over-relying on recent/memorable examples?
2. **Anchoring**: First information dominating decision?
3. **Confirmation**: Seeking only confirming evidence?
4. **Recency**: Overweighting recent data?
5. **Selection**: Non-representative sampling?
6. **Overconfidence**: Unjustified certainty?
7. **Sunk Cost**: Past investment driving decision?

### Step 3: Severity Rating

For each identified bias:
- Assess impact: Low / Medium / High
- Consider frequency: One-time / Recurring / Systematic
- Evaluate consequences: Minor / Moderate / Significant

### Step 4: Mitigation Planning

For each significant bias:
1. Apply relevant mitigation strategy
2. Document corrected reasoning
3. Set up preventive measures
4. Plan for monitoring

### Step 5: Meta-Analysis

Reflect on bias patterns:
- Which biases occur most frequently?
- Are there systematic bias patterns?
- What situations trigger each bias?
- How effective are mitigation strategies?

---

## Bias Detection Report Template

```markdown
# Cognitive Bias Assessment

## Session Information
- **Date**: YYYY-MM-DD
- **Scope**: [Session / Decision / Project Phase]
- **Reviewer**: [Name / AI Agent]

## Biases Detected

### Bias 1: [Bias Name]
**Severity**: [Low / Medium / High]
**Frequency**: [One-time / Recurring / Systematic]

**Instance Description**:
[Specific example of bias]

**Impact Assessment**:
[How bias affected decision or reasoning]

**Mitigation Applied**:
[Strategy used to address bias]

**Corrected Approach**:
[How decision/reasoning should be revised]

---

### Bias 2: [Bias Name]
[Repeat structure for each bias]

---

## Summary Statistics

| Bias Type | Instances | Severity | Mitigation Status |
|-----------|-----------|----------|-------------------|
| Availability | 2 | Medium | ✅ Addressed |
| Anchoring | 1 | Low | ✅ Addressed |
| Confirmation | 3 | High | ⚠️ Partial |
| Recency | 0 | - | - |
| Selection | 1 | Medium | ✅ Addressed |
| Overconfidence | 2 | Low | ✅ Addressed |
| Sunk Cost | 0 | - | - |

**Total Biases Detected**: 9
**High Severity**: 1
**Medium Severity**: 3
**Low Severity**: 5

## Bias Patterns

**Most Frequent**: Confirmation bias (3 instances)
**Highest Impact**: Confirmation bias affecting key architecture decision
**Recurring Context**: Biases more frequent under time pressure

## Mitigation Effectiveness

**Successful Mitigations**:
- Availability: Systematic search for diverse examples
- Selection: Random sampling implemented
- Overconfidence: Uncertainty quantification added

**Partial Mitigations**:
- Confirmation: Sought disconfirming evidence, but need more thorough analysis

**Recommendations**:
1. **High Priority**: Implement stronger confirmation bias mitigation (seek disconfirmation actively)
2. **Medium Priority**: Add time pressure awareness (biases increase under pressure)
3. **Low Priority**: Regular bias audits (weekly/monthly)

## Action Items

- [ ] Revise architecture decision with disconfirming evidence
- [ ] Implement bias checklist in decision-making process
- [ ] Schedule regular bias reflection sessions
- [ ] Update decision-making template with bias mitigation steps
```

---

## Usage Guidelines

### When to Use This Checklist

**Proactive Use** (Before Decision):
- Before making significant decisions
- When starting analysis or problem-solving
- At project milestones
- During design reviews

**Reactive Use** (After Decision):
- During reflection sessions
- When reviewing decisions
- After unexpected outcomes
- In post-mortems

**Ongoing Use**:
- Weekly reflection sessions
- Monthly cognitive bias audits
- Quarterly pattern analysis
- Annual meta-analysis

### Integration with Meta-Cognitive Reflection

This checklist integrates with broader meta-cognitive reflection:
1. Use during reasoning pattern analysis
2. Apply in problem-solving evaluation
3. Include in communication assessment
4. Document in session reflection reports
5. Track patterns over time

### Continuous Improvement

- **Monitor**: Track bias frequency and severity over time
- **Learn**: Identify patterns and triggers
- **Adapt**: Refine mitigation strategies
- **Teach**: Share bias awareness and mitigation techniques
- **Evolve**: Update checklist based on experience
