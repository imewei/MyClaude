# Output Templates

**Version**: 1.0.3
**Purpose**: Executive summary and detailed report formats for ultra-think sessions

---

## Template 1: Executive Summary (Quick Mode)

**Use For**: Fast assessments (5-10 minutes, 5-8 thoughts)

```markdown
# Executive Summary: [Problem Title]

**Session**: [Date & Time]
**Duration**: [X minutes]
**Confidence**: [X%]

---

## Problem

[1-2 sentence problem statement]

---

## Top 3 Approaches

### Approach 1: [Name] (Recommended)
- **Confidence**: [X%]
- **Timeline**: [X days/weeks]
- **Risk**: [Low/Medium/High]
- **Rationale**: [1-2 sentences why this is best]

### Approach 2: [Name]
- **Confidence**: [X%]
- **Timeline**: [X days/weeks]
- **Risk**: [Low/Medium/High]
- **Rationale**: [1-2 sentences]

### Approach 3: [Name]
- **Confidence**: [X%]
- **Timeline**: [X days/weeks]
- **Risk**: [Low/Medium/High]
- **Rationale**: [1-2 sentences]

---

## Immediate Next Steps

1. [Action item 1]
2. [Action item 2]
3. [Action item 3]

---

## Key Uncertainties

- [Uncertainty 1]
- [Uncertainty 2]
```

### Example

```markdown
# Executive Summary: API Performance Optimization

**Session**: 2025-11-06, 14:30
**Duration**: 8 minutes
**Confidence**: 75%

---

## Problem

API p95 latency has increased from 200ms to 800ms over past month, 
affecting user experience for 10k daily active users.

---

## Top 3 Approaches

### Approach 1: Fix Cache Cleanup Job (Recommended)
- **Confidence**: 80%
- **Timeline**: 2 days
- **Risk**: Low
- **Rationale**: Evidence shows cache cleanup stopped running after 
  v2.3.1. Quick fix with high impact.

### Approach 2: Optimize Database Queries
- **Confidence**: 60%
- **Timeline**: 1 week
- **Risk**: Medium
- **Rationale**: Some queries are slow, but not the primary bottleneck. 
  Good long-term investment.

### Approach 3: Add Read Replicas
- **Confidence**: 50%
- **Timeline**: 2 weeks
- **Risk**: High
- **Rationale**: Scales database but doesn't address root cause. 
  Consider for future.

---

## Immediate Next Steps

1. Investigate cache cleanup job logs (1 hour)
2. Review v2.3.1 deployment changes (30 min)
3. Prototype cache cleanup fix in staging (2 hours)

---

## Key Uncertainties

- Whether cache is the only bottleneck (need further investigation)
- Impact of fix on other services using cache
```

---

## Template 2: Detailed Analysis Report (Standard Mode)

**Use For**: Comprehensive analysis (30-90 minutes, 20-40 thoughts)

```markdown
# Ultra-Think Analysis: [Problem Title]

**Session Metadata**
- Date: [YYYY-MM-DD]
- Duration: [X hours Y minutes]
- Framework: [Framework name]
- Depth: [shallow/deep/ultradeep]
- Thoughts: [Total number]
- Confidence: [Final percentage]

---

## Executive Summary

[2-3 paragraphs covering:
- Problem statement
- Approach used
- Key findings
- Recommended solution
- Confidence level]

---

## Phase 1: Problem Understanding

### Problem Statement
[Detailed problem description]

### Scope
**In Scope**:
- [Item 1]
- [Item 2]

**Out of Scope**:
- [Item 1]
- [Item 2]

### Constraints
- [Constraint 1]
- [Constraint 2]
- [Constraint 3]

### Success Criteria
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

---

## Phase 2: Approach Selection

### Problem Classification
[What type of problem is this?]

### Framework Selection
**Chosen Framework**: [Name]

**Rationale**: [Why this framework?]

**Alternative Considered**: [Other frameworks and why not chosen]

---

## Phase 3: Deep Analysis

[This section varies by framework. Include:]

### Key Findings
1. [Finding 1]
   - Evidence: [Supporting data]
   - Confidence: [X%]

2. [Finding 2]
   - Evidence: [Supporting data]
   - Confidence: [X%]

### Hypotheses Explored
[If applicable: hypotheses tested and results]

### Evidence Summary
[Key data points, metrics, observations]

---

## Phase 4: Synthesis

### Recommended Solution
[Detailed solution description]

### Implementation Plan
1. [Step 1]
   - Timeline: [X]
   - Resources: [Y]
   - Risk: [Low/Med/High]

2. [Step 2]
   - Timeline: [X]
   - Resources: [Y]
   - Risk: [Low/Med/High]

### Trade-offs
**Advantages**:
- [Pro 1]
- [Pro 2]

**Disadvantages**:
- [Con 1]
- [Con 2]

**Alternatives Not Chosen**:
- [Alternative 1]: [Why not]
- [Alternative 2]: [Why not]

---

## Phase 5: Validation

### Assumption Validation
| Assumption | Valid? | Evidence |
|------------|--------|----------|
| [Assumption 1] | ✅ Yes | [Evidence] |
| [Assumption 2] | ⚠️  Partial | [Evidence] |
| [Assumption 3] | ❌ No | [Evidence] |

### Risk Assessment
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [Risk 1] | [%] | [H/M/L] | [Mitigation plan] |
| [Risk 2] | [%] | [H/M/L] | [Mitigation plan] |

### Quality Checks
- [x] Solution addresses all requirements
- [x] Evidence supports conclusions
- [x] Assumptions validated
- [ ] Stakeholder approval obtained
- [x] Implementation is feasible

---

## Phase 6: Finalization

### Final Recommendation

[Clear, actionable recommendation]

**Confidence Level**: [X%]

**Basis for Confidence**:
- [Factor 1]
- [Factor 2]
- [Factor 3]

**What Would Increase Confidence**:
- [Action 1] → +[X%]
- [Action 2] → +[X%]

### Next Steps

**Immediate (Today)**:
- [ ] [Action 1]
- [ ] [Action 2]

**Short-term (This Week)**:
- [ ] [Action 1]
- [ ] [Action 2]

**Long-term (This Month)**:
- [ ] [Action 1]
- [ ] [Action 2]

### Success Metrics

How to measure if this solution worked:
- [Metric 1]: [Target]
- [Metric 2]: [Target]
- [Metric 3]: [Target]

---

## Appendices

### Appendix A: Thought Progression
[Optional: Summary of key thoughts]

### Appendix B: Evidence Collected
[Optional: Detailed data, logs, metrics]

### Appendix C: Alternative Approaches
[Optional: Detailed analysis of alternatives]
```

---

## Template 3: Decision Matrix (For Decision Analysis)

**Use For**: Technology selection, architectural decisions, vendor evaluation

```markdown
# Decision Analysis: [Decision Title]

**Date**: [YYYY-MM-DD]
**Decision Type**: [Technology/Architecture/Vendor/etc.]
**Confidence**: [X%]

---

## Decision Statement

[Clear statement of what's being decided]

**Timeline**: [When decision needed]
**Impact**: [Who/what is affected]

---

## Options Evaluated

1. [Option 1]
2. [Option 2]
3. [Option 3]
4. [Option 4]

---

## Evaluation Criteria

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| [Criterion 1] | [X%] | [Why this weight] |
| [Criterion 2] | [X%] | [Why this weight] |
| [Criterion 3] | [X%] | [Why this weight] |
| [Criterion 4] | [X%] | [Why this weight] |

Total: 100%

---

## Detailed Scoring

### Option 1: [Name]

| Criterion | Raw Score (1-10) | Weight | Weighted Score |
|-----------|------------------|--------|----------------|
| [Criterion 1] | [X] | [Y%] | [X×Y] |
| [Criterion 2] | [X] | [Y%] | [X×Y] |
| [Criterion 3] | [X] | [Y%] | [X×Y] |
| [Criterion 4] | [X] | [Y%] | [X×Y] |
| **Total** | | | **[Sum]** |

**Strengths**:
- [Strength 1]
- [Strength 2]

**Weaknesses**:
- [Weakness 1]
- [Weakness 2]

### [Repeat for each option]

---

## Scoring Summary

| Option | Total Score | Rank |
|--------|-------------|------|
| [Option 1] | [Score] | [1-4] |
| [Option 2] | [Score] | [1-4] |
| [Option 3] | [Score] | [1-4] |
| [Option 4] | [Score] | [1-4] |

---

## Sensitivity Analysis

### What if [Criterion 1] weight increases?

| Option | Original | New Score | Change |
|--------|----------|-----------|--------|
| [Option 1] | [X] | [Y] | [Δ] |
| [Option 2] | [X] | [Y] | [Δ] |

[Analysis of impact]

### What if [Score] is uncertain?

[Analysis of scoring uncertainty impact]

---

## Recommendation

**Selected Option**: [Option X]

**Rationale**:
- Highest weighted score ([X])
- Robust to sensitivity analysis
- [Additional reasons]

**Confidence**: [X%]

**Conditions**:
- [Condition 1]
- [Condition 2]

**Fallback**: If [selected option] fails, [fallback option] is next best

---

## Implementation

**Timeline**: [X weeks]

**Milestones**:
1. [Milestone 1] - [Date]
2. [Milestone 2] - [Date]
3. [Milestone 3] - [Date]

**Success Criteria**:
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]
```

---

## Template 4: Root Cause Analysis Report

**Use For**: Debugging, incident investigation, failure analysis

```markdown
# Root Cause Analysis: [Issue Title]

**Date**: [YYYY-MM-DD]
**Severity**: [Critical/High/Medium/Low]
**Status**: [Investigating/Root Cause Found/Resolved]
**Confidence**: [X%]

---

## Incident Summary

**What Happened**: [Brief description]

**Impact**:
- Users affected: [X]
- Duration: [Y hours/minutes]
- Business impact: [Description]

**Timeline**:
- [HH:MM] - [Event 1]
- [HH:MM] - [Event 2]
- [HH:MM] - [Event 3]

---

## Symptom Analysis

### Observed Symptoms
1. [Symptom 1]
   - First observed: [Time]
   - Frequency: [Always/Intermittent]
   - Conditions: [When does it occur]

2. [Symptom 2]
   - [Details]

### Evidence Collected

**Logs**:
```
[Relevant log entries]
```

**Metrics**:
- [Metric 1]: [Value] (normal: [X])
- [Metric 2]: [Value] (normal: [X])

**Environment**:
- Version: [X]
- Configuration: [Y]
- Load: [Z]

---

## Investigation Process

### Hypotheses Tested

#### Hypothesis 1: [Description]
- **Likelihood**: [X%]
- **Test**: [How tested]
- **Result**: ✅ Confirmed / ❌ Refuted
- **Evidence**: [Details]

#### Hypothesis 2: [Description]
- **Likelihood**: [X%]
- **Test**: [How tested]
- **Result**: ✅ Confirmed / ❌ Refuted
- **Evidence**: [Details]

---

## Root Cause

### Identified Root Cause

[Clear statement of root cause]

### 5 Whys Analysis

1. **Why did [symptom] happen?**
   → [Answer 1]

2. **Why did [answer 1]?**
   → [Answer 2]

3. **Why did [answer 2]?**
   → [Answer 3]

4. **Why did [answer 3]?**
   → [Answer 4]

5. **Why did [answer 4]?**
   → **[Root Cause]**

### Causal Chain

```
[Root Cause]
    ↓
[Intermediate Cause 1]
    ↓
[Intermediate Cause 2]
    ↓
[Symptom]
```

### Confidence Assessment

**Confidence in Root Cause**: [X%]

**Supporting Evidence**:
- [Evidence 1]
- [Evidence 2]
- [Evidence 3]

**Remaining Uncertainties**:
- [Uncertainty 1]
- [Uncertainty 2]

---

## Solution

### Immediate Fix

[Description of quick fix to restore service]

**Status**: ✅ Applied / ⏳ In Progress / ⏸️ Planned

### Permanent Fix

[Description of solution that addresses root cause]

**Implementation Plan**:
1. [Step 1] - [Timeline]
2. [Step 2] - [Timeline]
3. [Step 3] - [Timeline]

**Prevention Measures**:
- [Measure 1] - Prevents recurrence
- [Measure 2] - Earlier detection
- [Measure 3] - Faster recovery

---

## Lessons Learned

### What Went Well
- [Item 1]
- [Item 2]

### What Could Be Improved
- [Item 1]
- [Item 2]

### Action Items
- [ ] [Action 1] - Owner: [X], Due: [Date]
- [ ] [Action 2] - Owner: [Y], Due: [Date]
- [ ] [Action 3] - Owner: [Z], Due: [Date]

---

## Related Incidents

- [Incident 1] - [Date] - [Similarity]
- [Incident 2] - [Date] - [Similarity]

**Pattern**: [If applicable, describe pattern across incidents]
```

---

## Formatting Guidelines

### Use of Emphasis

- **Bold**: Key terms, recommendations, important findings
- *Italic*: Emphasis within sentences
- `Code`: Technical terms, commands, file names
- > Blockquotes: Important warnings or notes

### Visual Indicators

Use emoji or symbols sparingly for quick scanning:
- ✅ Confirmed, completed, validated
- ❌ Refuted, failed, invalid
- ⚠️  Warning, concern, partial
- ⏳ In progress
- ⏸️  Planned, not started
- 🔴 Critical, high priority
- 🟡 Medium priority
- 🟢 Low priority, good status

### Code Blocks

Use code blocks for:
- Log entries
- Command examples
- Data samples
- Pseudo-code

### Tables

Use tables for:
- Decision matrices
- Comparison of options
- Risk assessments
- Timeline tracking

---

*Part of the ai-reasoning plugin documentation*
