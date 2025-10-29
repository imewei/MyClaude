# Validation Checklist

Comprehensive checklist for validating reasoning integrity, logical consistency, and output quality.

---

## 1. Structural Validation

### Thought Structure ✅

- [ ] All thoughts have unique IDs following T[phase].[step].[branch] format
- [ ] Thought IDs are sequential and logical within phases
- [ ] All required fields present: id, stage, dependencies, content, confidence, status
- [ ] Optional fields used appropriately: evidence, assumptions, contradictions, tools_used
- [ ] Thought stages match phase (planning in Phase 1-2, analysis in Phase 3, etc.)

### Dependency Chain ✅

- [ ] All dependencies exist (no references to non-existent thoughts)
- [ ] Dependency chain is acyclic (no circular dependencies)
- [ ] Dependencies are logically ordered (earlier thoughts before later ones)
- [ ] Branch dependencies correctly reference parent thoughts
- [ ] Revision dependencies correctly reference original thoughts

### Phase Coverage ✅

- [ ] Phase 1 (Understanding): Problem captured, constraints identified, assumptions surfaced
- [ ] Phase 2 (Approach): Framework selected, strategy designed
- [ ] Phase 3 (Analysis): Framework executed, branches explored
- [ ] Phase 4 (Synthesis): Findings integrated, conclusions drawn
- [ ] Phase 5 (Validation): Contradictions checked, assumptions verified
- [ ] Phase 6 (Finalization): Summary created, actions defined

---

## 2. Logical Consistency

### Semantic Contradiction Check ✅

**Process**: Compare thought content for conflicting statements

**Examples of contradictions**:
- "System is read-heavy" vs "System needs write optimization"
- "Budget is $50K" vs "Recommended solution costs $75K"
- "Deadline is Q1" vs "Timeline extends to Q2"

**Check for**:
- [ ] No direct contradictions between thought content
- [ ] No contradictions between conclusions and evidence
- [ ] No contradictions between recommendations and constraints
- [ ] All identified contradictions have been resolved via revisions

### Constraint Violation Check ✅

**Process**: Verify all identified constraints remain satisfied throughout reasoning

**Common constraints**:
- Technical: Performance, scalability, compatibility
- Business: Budget, timeline, regulations
- Resource: Team size, skills, availability

**Check for**:
- [ ] All Phase 1 constraints acknowledged in later phases
- [ ] Recommendations don't violate stated constraints
- [ ] Tradeoffs explicitly acknowledge constraint flexibility
- [ ] Constraint violations are flagged and addressed

### Temporal Consistency Check ✅

**Process**: Verify cause-effect relationships are consistent

**Examples of inconsistencies**:
- "Optimization reduces latency" but later "Latency increased"
- "Added caching" but later "Cache not yet implemented"
- "Database migrated" but later "Migration pending"

**Check for**:
- [ ] Effects follow causes in proper sequence
- [ ] State changes are tracked consistently
- [ ] Timeline is internally consistent
- [ ] No retroactive changes without revisions

### Logical Fallacy Detection ✅

**Common fallacies to check**:

- [ ] **False Dichotomy**: Are there really only two options?
- [ ] **Slippery Slope**: Is the chain of consequences realistic?
- [ ] **Appeal to Authority**: Is evidence beyond "expert says so"?
- [ ] **Correlation ≠ Causation**: Is causal relationship validated?
- [ ] **Circular Reasoning**: Does conclusion assume what it tries to prove?
- [ ] **Straw Man**: Are counter-arguments fairly represented?
- [ ] **Ad Hominem**: Are arguments focused on ideas, not people?

---

## 3. Evidence Quality

### Evidence Sufficiency ✅

- [ ] Claims are supported by evidence (not just assertions)
- [ ] Evidence is specific and concrete (not vague generalities)
- [ ] Evidence is relevant to the claim it supports
- [ ] Sufficient evidence quantity for confidence level
- [ ] Evidence sources are identified

### Evidence Validity ✅

**Check for**:
- [ ] Evidence comes from reliable sources
- [ ] Evidence is current and applicable
- [ ] Evidence is correctly interpreted
- [ ] Evidence is not cherry-picked (alternative evidence considered)
- [ ] Statistical evidence is properly analyzed

### Evidence-Confidence Alignment ✅

**Confidence levels should match evidence**:

- High confidence (0.85-1.0): Strong, multiple sources of evidence
- Medium confidence (0.60-0.84): Moderate evidence, some gaps
- Low confidence (0.0-0.59): Weak or conflicting evidence

**Check for**:
- [ ] High confidence claims have strong evidence
- [ ] Low confidence acknowledged when evidence is weak
- [ ] Confidence levels are realistic, not optimistic
- [ ] Confidence changes are explained

---

## 4. Assumption Validation

### Assumption Documentation ✅

- [ ] All significant assumptions are explicitly stated
- [ ] Assumptions are listed in relevant thoughts
- [ ] Critical assumptions are identified (T1.3)
- [ ] Implicit assumptions have been surfaced

### Assumption Classification ✅

**Classify each assumption**:
- [ ] **Validated**: Confirmed with evidence
- [ ] **Reasonable**: Likely true, accepted risk
- [ ] **Unvalidated**: Unknown, requires investigation
- [ ] **Invalid**: Proven false, requires revision

### Assumption Impact Assessment ✅

**For each critical assumption**:
- [ ] Impact if false is understood
- [ ] Validation approach is defined
- [ ] Risk is assessed and accepted or mitigated
- [ ] Dependencies on assumptions are tracked

### Assumption Verification (T5.2) ✅

- [ ] Critical assumptions have been validated where possible
- [ ] Unvalidated assumptions are acknowledged as risks
- [ ] Invalid assumptions have triggered revisions
- [ ] Assumption validation is documented

---

## 5. Framework Application

### Framework Selection ✅

- [ ] Framework selection is justified (T2.3)
- [ ] Framework matches problem type
- [ ] Framework process is understood
- [ ] Alternative frameworks were considered

### Framework Execution ✅

**First Principles**:
- [ ] Assumptions listed and challenged
- [ ] Fundamentals identified
- [ ] Solution reconstructed from basics
- [ ] Reconstruction validated

**Systems Thinking**:
- [ ] System components mapped
- [ ] Relationships identified
- [ ] Feedback loops traced
- [ ] Leverage points identified

**Root Cause Analysis**:
- [ ] Problem defined precisely
- [ ] Evidence gathered systematically
- [ ] Hypotheses generated and tested
- [ ] Root cause validated

**Decision Analysis**:
- [ ] Criteria defined with weights
- [ ] Alternatives generated (3-5)
- [ ] Scoring completed
- [ ] Tradeoffs analyzed

**Design Thinking**:
- [ ] User needs understood
- [ ] Problem framed from user perspective
- [ ] Solutions ideated
- [ ] Prototyping and testing considered

**Scientific Method**:
- [ ] Hypothesis formulated
- [ ] Experiment designed
- [ ] Data analyzed
- [ ] Conclusion drawn

**OODA Loop**:
- [ ] Observation: Current state gathered
- [ ] Orientation: Context analyzed
- [ ] Decision: Action selected
- [ ] Act: Execution planned

### Framework Adaptation ✅

- [ ] Framework is adapted to problem context
- [ ] Unnecessary steps are justified if skipped
- [ ] Additional steps are justified if added
- [ ] Hybrid approaches are clearly explained

---

## 6. Branching Quality

### Branch Coverage ✅

- [ ] Multiple alternatives explored (at least 2-3 branches)
- [ ] Branches represent meaningfully different approaches
- [ ] Branches are explored to sufficient depth
- [ ] Edge cases considered in branches

### Branch Comparison ✅

- [ ] Branches are compared systematically (T4.1)
- [ ] Pros and cons identified for each branch
- [ ] Scoring or ranking provided where appropriate
- [ ] Best alternative selected with justification

### Branch Integration ✅

- [ ] Findings from all branches synthesized
- [ ] Convergent conclusions highlighted
- [ ] Divergent conclusions explained
- [ ] Complementary insights integrated

---

## 7. Revision Management

### Revision Necessity ✅

**Revisions should occur when**:
- [ ] New evidence contradicts previous thought
- [ ] Assumption found to be invalid
- [ ] Logical inconsistency detected
- [ ] Better approach identified

### Revision Quality ✅

**Each revision should**:
- [ ] Reference original thought ID
- [ ] Explain reason for revision
- [ ] State what changed
- [ ] Update confidence level
- [ ] Mark original as superseded

### Revision Impact ✅

- [ ] Downstream thoughts reviewed for impact
- [ ] Cascading revisions created if needed
- [ ] Synthesis updated to reflect revisions
- [ ] Conclusions validated against revisions

---

## 8. Confidence Tracking

### Confidence Calibration ✅

**Confidence should reflect**:
- [ ] Strength of evidence
- [ ] Number of assumptions
- [ ] Complexity of reasoning
- [ ] Validation status

### Confidence Distribution ✅

**Healthy distribution**:
- [ ] Not all thoughts at 1.0 (overconfidence)
- [ ] Not all thoughts at 0.5 (under-confidence)
- [ ] Higher confidence in validated conclusions
- [ ] Lower confidence in exploratory branches

### Confidence Trends ✅

- [ ] Confidence generally increases through phases
- [ ] Confidence decreases noted with revisions
- [ ] Confidence increases noted with validation
- [ ] Overall trajectory toward higher confidence

---

## 9. Output Quality

### Executive Summary (T6.1) ✅

- [ ] Problem is clearly stated
- [ ] Approach is summarized
- [ ] Key findings are listed (3-5 max)
- [ ] Top recommendations are clear
- [ ] Overall confidence is stated
- [ ] Next steps are actionable

### Action Plan (T6.2) ✅

- [ ] Actions are specific and concrete
- [ ] Actions have owners/responsibilities
- [ ] Actions have timelines
- [ ] Actions have success criteria
- [ ] Actions are prioritized
- [ ] Dependencies between actions are noted

### Decision Documentation (T6.3) ✅

- [ ] Key decisions are stated clearly
- [ ] Rationale for each decision is provided
- [ ] Alternatives considered are listed
- [ ] Trade-offs are explained
- [ ] Decision confidence is stated
- [ ] Context for future reference is preserved

---

## 10. Meta-Validation

### Completeness ✅

- [ ] All six phases completed
- [ ] No gaps in reasoning chain
- [ ] All questions from problem statement addressed
- [ ] All success criteria evaluated

### Depth Appropriateness ✅

**Shallow (5-15 thoughts)**:
- [ ] Appropriate for simple problem
- [ ] Core reasoning captured
- [ ] Quick analysis complete

**Deep (20-40 thoughts)**:
- [ ] Appropriate for complex problem
- [ ] Multiple branches explored
- [ ] Comprehensive analysis

**Ultra-Deep (50-100+ thoughts)**:
- [ ] Appropriate for critical/novel problem
- [ ] Exhaustive exploration
- [ ] Multiple frameworks applied

### Auditability ✅

- [ ] Reasoning path is clear
- [ ] Others could follow the logic
- [ ] Decisions are justified
- [ ] Sources are documented
- [ ] Process is transparent

### Actionability ✅

- [ ] Recommendations are specific
- [ ] Actions are implementable
- [ ] Timeline is realistic
- [ ] Resources are identified
- [ ] Success is measurable

---

## Validation Scoring

### Overall Score: __/100

| Category | Weight | Score | Weighted |
|----------|--------|-------|----------|
| Structural Validation | 10% | __/10 | __ |
| Logical Consistency | 20% | __/10 | __ |
| Evidence Quality | 15% | __/10 | __ |
| Assumption Validation | 10% | __/10 | __ |
| Framework Application | 15% | __/10 | __ |
| Branching Quality | 10% | __/10 | __ |
| Revision Management | 5% | __/10 | __ |
| Confidence Tracking | 5% | __/10 | __ |
| Output Quality | 5% | __/10 | __ |
| Meta-Validation | 5% | __/10 | __ |
| **Total** | **100%** | - | **__/10** |

### Score Interpretation

| Score | Quality Level | Action |
|-------|--------------|--------|
| 9-10 | Excellent | Publish, present |
| 7-8 | Very Good | Minor refinements |
| 5-6 | Good | Targeted improvements |
| 3-4 | Fair | Significant revision needed |
| 1-2 | Poor | Major rework required |

---

## Validation Process

### Step 1: Self-Check (During Reasoning)

Run quick validation after each phase:
- Phase 1: Check problem understanding completeness
- Phase 2: Verify framework selection
- Phase 3: Check for contradictions as you go
- Phase 4: Validate synthesis covers all branches
- Phase 5: Run full validation checklist
- Phase 6: Final quality check

### Step 2: Automated Checks (T5.1)

Run automated contradiction detection:
```yaml
T5.1 - Contradiction Detection:
  - Semantic scan: Check for conflicting statements
  - Constraint scan: Check for violated constraints
  - Temporal scan: Check for cause-effect inconsistencies
  - Report: List any contradictions found with thought IDs
```

### Step 3: Manual Review (T5.2-T5.4)

Manually review critical aspects:
```yaml
T5.2 - Assumption Verification:
  - Review all assumptions from T1.3
  - Classify as validated/unvalidated/invalid
  - Assess risk of unvalidated assumptions

T5.3 - Confidence Assessment:
  - Review confidence distribution
  - Check confidence-evidence alignment
  - Identify areas of uncertainty

T5.4 - Uncertainty Identification:
  - List known unknowns
  - Identify potential blind spots
  - Prioritize further investigation
```

### Step 4: Peer Review (Optional)

For high-stakes reasoning:
- [ ] Have another agent/person review reasoning chain
- [ ] Get feedback on logic and conclusions
- [ ] Validate assumptions with domain experts
- [ ] Test recommendations with stakeholders

### Step 5: Outcome Validation (After Implementation)

After actions are taken:
- [ ] Compare actual outcomes to predictions
- [ ] Assess confidence calibration (were high-confidence thoughts correct?)
- [ ] Identify what was missed or wrong
- [ ] Update reasoning process based on learnings

---

## Common Validation Issues

### Issue 1: Shallow Analysis
**Symptom**: Few thoughts, single branch, no revisions
**Fix**: Increase depth, explore alternatives, challenge assumptions

### Issue 2: Overconfidence
**Symptom**: All thoughts at 0.9-1.0 confidence
**Fix**: Be more realistic, acknowledge uncertainties

### Issue 3: Weak Evidence
**Symptom**: Claims without supporting evidence
**Fix**: Gather evidence, cite sources, validate claims

### Issue 4: Unvalidated Assumptions
**Symptom**: Critical assumptions not checked
**Fix**: Validate key assumptions, acknowledge risk

### Issue 5: Missing Synthesis
**Symptom**: Many branches but no integration
**Fix**: Complete Phase 4, integrate findings

### Issue 6: Vague Recommendations
**Symptom**: Generic or non-actionable advice
**Fix**: Be specific, add timelines and owners

### Issue 7: Incomplete Phases
**Symptom**: Skipped phases or minimal coverage
**Fix**: Complete all six phases systematically

---

## Validation Report Template

```markdown
# Validation Report

**Session**: [Session ID]
**Date**: [Date]
**Validator**: [Name/Agent]

## Validation Summary

**Overall Score**: __/10
**Quality Level**: [Excellent/Very Good/Good/Fair/Poor]

**Passed**: __ / __ checks
**Failed**: __ / __ checks
**Warnings**: __ issues

## Critical Issues 🔴

1. [Issue 1]: [Description and impact]
2. [Issue 2]: [Description and impact]

## Warnings ⚠️

1. [Warning 1]: [Description]
2. [Warning 2]: [Description]

## Recommendations

1. [Recommendation 1]: [How to improve]
2. [Recommendation 2]: [How to improve]

## Category Scores

| Category | Score | Status | Notes |
|----------|-------|--------|-------|
| Structural | __/10 | ✅/⚠️/🔴 | [Notes] |
| Logical | __/10 | ✅/⚠️/🔴 | [Notes] |
| Evidence | __/10 | ✅/⚠️/🔴 | [Notes] |
| Assumptions | __/10 | ✅/⚠️/🔴 | [Notes] |
| Framework | __/10 | ✅/⚠️/🔴 | [Notes] |
| Branching | __/10 | ✅/⚠️/🔴 | [Notes] |
| Revisions | __/10 | ✅/⚠️/🔴 | [Notes] |
| Confidence | __/10 | ✅/⚠️/🔴 | [Notes] |
| Outputs | __/10 | ✅/⚠️/🔴 | [Notes] |
| Meta | __/10 | ✅/⚠️/🔴 | [Notes] |

## Conclusion

[Overall assessment and recommendation for publishing/refining]
```
