# Thinking Session Structure

**Version**: 1.0.3
**Purpose**: Phase-by-phase templates for structured reasoning sessions

---

## Session Overview

A complete ultra-think session consists of 6 phases executed sequentially:

```
1. Problem Understanding (15-20% of thoughts)
2. Approach Selection (10-15% of thoughts)
3. Deep Analysis (40-50% of thoughts)
4. Synthesis (10-15% of thoughts)
5. Validation (10-15% of thoughts)
6. Finalization (5-10% of thoughts)
```

---

## Phase 1: Problem Understanding

**Goal**: Ensure complete comprehension before diving into solution

**Duration**: 4-8 thoughts (15-20% of session)

**Template**:
```
T1: Problem Statement
    [Restate problem in own words]
    [Identify what's being asked]
    
T2: Clarify Scope
    [What's in scope?]
    [What's out of scope?]
    [Boundary conditions]
    
T3: Identify Constraints
    [Technical constraints]
    [Resource constraints]
    [Time constraints]
    
T4: Define Success Criteria
    [What does a good solution look like?]
    [How will we measure success?]
    [What are dealbreakers?]
    
T5 (if needed): Stakeholder Analysis
    [Who is affected?]
    [What do they need?]
    [What are their priorities?]
```

**Example (API Performance Problem)**:
```
T1: Problem Statement
    "API response times have increased from 200ms to 800ms p95 over 
    the past month. Need to identify root cause and reduce to <300ms."
    
T2: Scope
    In scope: API server, database, caching layer
    Out of scope: Client-side performance, network latency
    
T3: Constraints
    - Cannot add new infrastructure (budget frozen)
    - Must maintain backward compatibility
    - Max 2 days for investigation + fix
    
T4: Success Criteria
    - p95 latency < 300ms
    - No regressions in other metrics
    - Solution is sustainable (not just temporary fix)
```

---

## Phase 2: Approach Selection

**Goal**: Choose the most effective reasoning framework

**Duration**: 3-5 thoughts (10-15% of session)

**Template**:
```
T[N]: Problem Classification
    [What type of problem is this?]
    [Novel? System? Debug? Decision? Design?]
    
T[N+1]: Framework Candidates
    [List 2-3 applicable frameworks]
    [Justify why each could work]
    
T[N+2]: Framework Selection
    [Choose primary framework]
    [Justification: why this one?]
    [How will we apply it?]
    
T[N+3] (optional): Hybrid Approach
    [Will we combine frameworks?]
    [Which phases use which framework?]
```

**Example**:
```
T6: Classification
    This is a performance regression problem requiring investigation.
    Need to find root cause and implement solution.
    
T7: Candidates
    1. Root Cause Analysis - Good for debugging regression
    2. Systems Thinking - Good for understanding performance holistically
    3. First Principles - Overkill, we likely have performance best practices
    
T8: Selection
    Start with Root Cause Analysis to find immediate issue.
    Then use Systems Thinking to ensure fix doesn't hurt other areas.
    
    Rationale: We need to move quickly (2-day constraint), so 
    Root Cause Analysis is most efficient. Systems Thinking will
    help us validate the fix won't cause new problems.
```

---

## Phase 3: Deep Analysis

**Goal**: Execute the chosen framework thoroughly

**Duration**: 15-25 thoughts (40-50% of session)

**Template varies by framework**. Key principles:

1. **Follow Framework Structure**: Use the phases defined for your chosen framework
2. **Be Thorough**: This is where most thinking happens
3. **Show Your Work**: Document reasoning at each step
4. **Branch When Uncertain**: Use branching for multiple paths
5. **Revise When New Info Emerges**: Don't hesitate to revisit earlier thoughts

**Example Structure (Root Cause Analysis)**:
```
T9-T12: Symptom Analysis
    T9: Precise symptom description
    T10: Timeline of changes
    T11: Evidence gathering (logs, metrics)
    T12: Pattern identification
    
T13-T18: Hypothesis Generation
    T13: Hypothesis 1 with likelihood estimate
    T14: Hypothesis 2 with likelihood estimate
    T15: Hypothesis 3 with likelihood estimate
    T16: Prioritize by likelihood × impact
    T17: Design tests for top 2 hypotheses
    T18: Plan investigation approach
    
T19-T25: Hypothesis Testing
    T19-T21: Test Hypothesis 1
        T19: Run test
        T20: Analyze results
        T21: Conclusion (confirmed/refuted)
    
    T22-T24: Test Hypothesis 2 (if H1 refuted)
        T22: Run test
        T23: Analyze results
        T24: Conclusion
    
    T25: Root cause identified
        [What is it?]
        [Why does it cause the symptom?]
        [How to fix?]
```

**Branching Example**:
```
T15: Two equally promising hypotheses
    Branch A: Database query regression
    Branch B: Cache invalidation bug
    
    [confidence_distribution: 50% A, 50% B]
    
T16 (Branch A): Investigate query performance
    [Check slow query log]
    [Result: No significant changes]
    [Branch A confidence: 50% → 20%]
    
T17 (Branch B): Investigate cache behavior
    [Check cache hit rate]
    [Result: Cache hits dropped from 85% to 30%]
    [Branch B confidence: 50% → 85%]
    
T18: Converge on Branch B
    Cache issue is clear winner. Focusing investigation there.
```

---

## Phase 4: Synthesis

**Goal**: Combine analysis into coherent solution/answer

**Duration**: 4-7 thoughts (10-15% of session)

**Template**:
```
T[N]: Synthesis Initiation
    [Summarize key findings from analysis]
    [What did we learn?]
    
T[N+1]: Solution Formulation
    [Describe solution/answer]
    [How does it address the problem?]
    [Why is this the right approach?]
    
T[N+2]: Implementation Outline
    [High-level steps to implement]
    [Resources needed]
    [Timeline estimate]
    
T[N+3]: Trade-offs
    [What are we giving up?]
    [Alternative approaches not chosen]
    [Why is this optimal?]
    
T[N+4]: Confidence Assessment
    [How confident are we?]
    [What could go wrong?]
    [What are remaining uncertainties?]
```

**Example**:
```
T26: Key Findings
    Root cause identified: Cache cleanup job stopped running after 
    deployment v2.3.1. Bug in job scheduler restart logic.
    
    Evidence:
    - Cache hit rate dropped from 85% → 30% at v2.3.1 deploy time
    - Old cache entries accumulating (128k stale entries)
    - Cache memory full, evicting active entries
    
T27: Solution
    1. Fix bug in scheduler restart logic (CacheScheduler.java:156)
    2. Manually purge stale cache entries (one-time cleanup)
    3. Add monitoring for cache job execution
    
T28: Implementation Steps
    1. Deploy scheduler fix (15 min)
    2. Run cache cleanup script (30 min)
    3. Add Datadog monitor for job failures (30 min)
    4. Verify latency returns to <300ms (monitoring)
    
    Total time: ~2 hours
    
T29: Trade-offs
    Considered but rejected:
    - Increasing cache size: Doesn't fix root cause
    - Disabling cache: Causes worse performance
    - Manual cache invalidation: Not sustainable
    
    Chosen approach fixes root cause and adds monitoring to 
    prevent recurrence. Small time investment, high confidence.
    
T30: Confidence
    90% confident this will solve the problem.
    
    Remaining risks:
    - 10% chance another issue contributes (but we've validated 
      cache is primary cause)
    - Fix might have side effects (mitigated by code review)
```

---

## Phase 5: Validation

**Goal**: Verify solution quality and identify gaps

**Duration**: 4-6 thoughts (10-15% of session)

**Template**:
```
T[N]: Solution Review
    [Does solution fully address problem?]
    [Are all requirements met?]
    [Any gaps or missing pieces?]
    
T[N+1]: Assumption Check
    [What assumptions did we make?]
    [Are they valid?]
    [How to validate uncertain assumptions?]
    
T[N+2]: Risk Assessment
    [What could go wrong?]
    [How likely are these risks?]
    [How to mitigate?]
    
T[N+3]: Alternative Validation
    [Did we miss any better approaches?]
    [Quick check of alternative paths]
    
T[N+4]: Stakeholder Check
    [Does this meet stakeholder needs?]
    [Any concerns or objections?]
```

**Example**:
```
T31: Solution Review
    ✅ Addresses root cause (scheduler bug)
    ✅ Restores cache hit rate to 85%
    ✅ Should bring latency to <300ms
    ✅ Includes monitoring to prevent recurrence
    
    Gap identified: No rollback plan if fix causes issues
    
T32: Assumptions
    1. Scheduler bug is the only cause (90% confident)
       → Validated with evidence
    2. Cleanup script won't impact production (85% confident)
       → Need to run in off-hours to be safe
    3. Fix is compatible with current config (95% confident)
       → Code review will confirm
    
T33: Risk Mitigation
    Risk 1: Fix causes new bug (20% probability, medium impact)
    → Mitigation: Thorough code review + staging test
    
    Risk 2: Cleanup script overloads database (10% probability, high impact)
    → Mitigation: Run during low-traffic hours (3am)
    
    Risk 3: Cache fill too slow (15% probability, medium impact)
    → Mitigation: Pre-warm cache with common queries
    
T34: Rollback Plan (from gap in T31)
    If issues arise:
    1. Revert scheduler fix deployment (5 min)
    2. Restore cache from backup (30 min)
    3. Monitor for stabilization
    
    This addresses the identified gap.
```

---

## Phase 6: Finalization

**Goal**: Produce clear, actionable output

**Duration**: 2-4 thoughts (5-10% of session)

**Template**:
```
T[N]: Executive Summary
    [Problem + Solution in 2-3 sentences]
    [Key metrics: confidence, timeline, effort]
    
T[N+1]: Detailed Recommendation
    [Specific actions to take]
    [Order of operations]
    [Success criteria]
    
T[N+2]: Next Steps
    [Immediate actions (next hour/day)]
    [Follow-up actions (next week)]
    [Long-term actions (next month)]
    
T[N+3]: Final Confidence
    [Overall confidence in recommendation]
    [What would increase confidence?]
```

**Example**:
```
T35: Executive Summary
    API latency regression from 200ms → 800ms caused by cache cleanup 
    job failure after v2.3.1 deployment. Fix scheduler bug, purge stale 
    cache, add monitoring. 90% confident, 2-hour implementation.
    
T36: Action Plan
    1. Code review scheduler fix (30 min)
    2. Deploy fix to production (15 min)
    3. Run cache cleanup script at 3am (30 min)
    4. Add Datadog monitor for job health (30 min)
    5. Verify latency <300ms over 24 hours
    
    Success: p95 latency <300ms, cache hit rate >80%
    
T37: Timeline
    Immediate (today):
    - Complete code review
    - Prepare deployment and scripts
    
    Tonight (3am):
    - Deploy scheduler fix
    - Run cache cleanup
    
    Tomorrow:
    - Monitor metrics
    - Add monitoring alerts
    - Document root cause for post-mortem
    
T38: Final Assessment
    Confidence: 90%
    
    Would increase to 95% with:
    - Staging environment validation
    - Load test after fix
    
    Recommend proceeding with plan. Risk is low and rollback 
    is straightforward if needed.
```

---

## Session Patterns

### Standard Session (30-40 thoughts)
```
Phase 1: Problem Understanding (6 thoughts, 15%)
Phase 2: Approach Selection (4 thoughts, 10%)
Phase 3: Deep Analysis (18 thoughts, 45%)
Phase 4: Synthesis (6 thoughts, 15%)
Phase 5: Validation (4 thoughts, 10%)
Phase 6: Finalization (2 thoughts, 5%)

Total: 40 thoughts, 60-90 minutes
```

### Quick Session (15-20 thoughts)
```
Phase 1: Problem Understanding (3 thoughts, 15%)
Phase 2: Approach Selection (2 thoughts, 10%)
Phase 3: Deep Analysis (8 thoughts, 40%)
Phase 4: Synthesis (3 thoughts, 15%)
Phase 5: Validation (3 thoughts, 15%)
Phase 6: Finalization (1 thought, 5%)

Total: 20 thoughts, 20-30 minutes
```

### Deep Session (50-70 thoughts)
```
Phase 1: Problem Understanding (10 thoughts, 15%)
Phase 2: Approach Selection (8 thoughts, 10%)
Phase 3: Deep Analysis (35 thoughts, 50%)
Phase 4: Synthesis (8 thoughts, 10%)
Phase 5: Validation (7 thoughts, 10%)
Phase 6: Finalization (2 thoughts, 5%)

Total: 70 thoughts, 90-120 minutes
```

---

## Quality Indicators

### Good Session Characteristics
- Clear phase boundaries
- Logical thought progression
- Explicit assumption tracking
- Evidence-based conclusions
- Confidence calibration
- Actionable recommendations

### Warning Signs
- Jumping to conclusions (skipping phases)
- Circular reasoning (revisiting same thoughts)
- Unsupported assertions
- Missing validation
- Vague recommendations
- Overconfidence without evidence

---

*Part of the ai-reasoning plugin documentation*
