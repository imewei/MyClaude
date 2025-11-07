# Thought Format Guide

**Version**: 1.0.3
**Purpose**: Best practices for structuring thoughts in ultra-think sessions

---

## Thought Numbering

### Hierarchical Format

Use the format `T[major].[minor].[detail]` for nested thoughts:

```
T1: Major topic
T1.1: Subtopic
T1.2: Another subtopic
T1.2.1: Detail of subtopic
T1.2.2: Another detail
T2: Next major topic
```

**When to nest**:
- Breaking down complex ideas
- Exploring alternatives under same topic
- Providing supporting evidence

**Example**:
```
T5: Evaluate database options
T5.1: PostgreSQL
T5.1.1: Strengths: ACID, mature ecosystem
T5.1.2: Weaknesses: Scaling horizontal is complex
T5.1.3: Score: 7.5/10
T5.2: MongoDB
T5.2.1: Strengths: Flexible schema, horizontal scaling
T5.2.2: Weaknesses: Less mature for transactions
T5.2.3: Score: 6.5/10
```

---

## Thought Content Structure

### Standard Thought Format

```
T[N]: [Topic/Question]
    
    [Main content - analysis, reasoning, observations]
    
    [Evidence/Examples - if applicable]
    
    [Conclusion/Decision - if applicable]
    
    [Next steps - if applicable]
```

### Examples by Thought Type

**1. Analytical Thought**:
```
T12: Analyze cache hit rate trend

    Cache hit rate dropped from 85% to 30% over past 2 weeks.
    
    Timeline correlation:
    - Week 1: 85% hit rate (normal)
    - Deploy v2.3.1: No immediate change
    - Week 2: Gradual decline to 75%
    - Week 3: Sharp drop to 30%
    
    This delayed degradation suggests:
    1. Not an immediate bug from v2.3.1
    2. More likely a gradual accumulation issue
    3. Possibly cache cleanup job not running
    
    Hypothesis: Cache cleanup stopped, stale entries accumulating
```

**2. Decision Thought**:
```
T18: Choose between Option A (refactor) and Option B (patch)

    Criteria comparison:
    
    Option A (Refactor):
    - Time: 2 weeks
    - Risk: Medium (larger change)
    - Long-term: Sustainable, prevents recurrence
    - Score: 7/10
    
    Option B (Patch):
    - Time: 2 days
    - Risk: Low (targeted fix)
    - Long-term: Might need more patches later
    - Score: 8/10
    
    Decision: Choose Option B (patch)
    
    Rationale: Given 2-day constraint, patch is more appropriate.
    Can schedule refactor for Q2 if issues recur.
```

**3. Branching Thought**:
```
T15: Two equally viable hypotheses - branching to explore both

    Branch A: Database query regression
    Likelihood: 50%
    Test: Check slow query log
    
    Branch B: Cache invalidation bug
    Likelihood: 50%
    Test: Monitor cache hit rate
    
    Will pursue both in parallel, then converge on stronger path.
```

**4. Revision Thought**:
```
T20: [REVISION of T12] New evidence changes cache analysis

    Previous thinking (T12): Assumed cache cleanup stopped gradually
    
    New evidence: Found cache cleanup job IS running, but config changed
    - Job runs every 6 hours (correct)
    - But TTL changed from 1 hour → 12 hours in v2.3.1
    - So cleanup runs, but doesn't remove entries (they're not expired)
    
    Revised conclusion: Root cause is TTL misconfiguration, not job failure
    
    This significantly changes our fix approach.
```

**5. Synthesis Thought**:
```
T30: Synthesize findings into solution

    Root Cause (from T20-T28):
    - TTL misconfigured: 1h → 12h in v2.3.1
    - Cache fills with 12h entries
    - Active entries get evicted to make room
    - Result: Hit rate plummets
    
    Solution:
    1. Revert TTL to 1 hour
    2. Purge current cache
    3. Add config validation in deployment pipeline
    
    Expected outcome: 85% hit rate restored within 1 hour
    Confidence: 95%
```

---

## Confidence Indicators

### When to Include Confidence

Include confidence levels when:
- Making predictions
- Drawing conclusions
- Choosing between options
- Finalizing recommendations

### Confidence Scale

```
95-100%: Nearly certain, strong evidence
85-94%:  High confidence, good evidence
70-84%:  Moderate confidence, reasonable evidence
50-69%:  Low confidence, limited evidence
<50%:    Uncertain, speculative
```

### Format

```
T25: Root cause identified
    [Analysis...]
    
    Confidence: 90%
    
    Basis for confidence:
    - Direct evidence from logs (strong)
    - Timing correlation (strong)
    - Mechanism explanation (strong)
    
    Uncertainty factors:
    - Haven't tested in staging (reduces by 5%)
    - Might be contributing factors (reduces by 5%)
```

---

## Evidence and Citations

### Reference Earlier Thoughts

```
T28: Confirm hypothesis from T15

    As hypothesized in T15, cache cleanup job is the issue.
    
    Evidence gathered since T15:
    - T18: Found job logs show zero executions
    - T22: Confirmed scheduler bug in code review
    - T25: Tested fix in local environment
    
    Hypothesis is now confirmed with 95% confidence.
```

### Link to External Evidence

```
T14: Database query performance analysis

    Query execution times from past 30 days:
    
    Evidence (from Datadog):
    - Week 1: avg 45ms, p95 120ms
    - Week 2: avg 48ms, p95 130ms (no significant change)
    - Week 3: avg 520ms, p95 1800ms (6x regression!)
    
    Clear performance regression started Week 3.
```

---

## Branching Guidelines

### When to Branch

Branch when:
- Multiple viable hypotheses with similar likelihood
- Exploring truly different approaches
- Uncertain which path is correct
- Want to avoid premature convergence

**Don't branch** when:
- One option is clearly better
- Just listing pros/cons (use single thought)
- Paths will rejoin immediately

### Branch Format

```
T15: Multiple paths forward - branching

    Branch A (path=aggressive): Full rewrite
    Branch B (path=conservative): Incremental fix
    
    [Confidence distribution: 40% A, 60% B]

T16 [Branch A]: Explore full rewrite approach
    [Analysis specific to Branch A...]
    
T17 [Branch B]: Explore incremental fix approach
    [Analysis specific to Branch B...]
    
T18 [Converge]: Merge insights from both branches
    
    Branch A insights: Rewrite would solve root cause permanently
    Branch B insights: Fix is faster and lower risk
    
    Given time constraints, proceeding with Branch B approach
    but noting Branch A for future consideration.
```

---

## Thought Progression Patterns

### Linear Progression (Standard)

```
T1 → T2 → T3 → T4 → T5
```

Each thought builds on previous. Use for straightforward analysis.

### Branching and Convergence

```
T1 → T2 → T3 → [Branch A: T4a → T5a]
               → [Branch B: T4b → T5b]
                            ↓
                         T6 (converge)
```

Use when exploring alternatives that need deep investigation.

### Iterative Refinement

```
T1 → T2 → T3 → T4 → T5 (conclusion)
                ↓
         T6 [REVISION of T3] (new evidence)
                ↓
         T7 (updated analysis)
                ↓
         T8 (revised conclusion)
```

Use when new evidence requires revisiting earlier thoughts.

### Parallel Investigation

```
T1 → T2 → [T3: Hypothesis A]
       → [T4: Hypothesis B]
       → [T5: Hypothesis C]
             ↓
       T6 (synthesize all three)
```

Use when testing multiple independent hypotheses.

---

## Common Mistakes

### 1. Jumping to Conclusions

**Bad**:
```
T1: API is slow
T2: Let's add caching
```

**Good**:
```
T1: API is slow (800ms p95)
T2: Investigate root cause
T2.1: Check database query times
T2.2: Check network latency
T2.3: Check processing time
T3: Analysis shows DB queries are slow (600ms)
T4: Consider solutions: caching, query optimization, indexing
T5: Choose query optimization (most direct fix)
```

### 2. Vague Reasoning

**Bad**:
```
T5: Option B is probably better because it's simpler
```

**Good**:
```
T5: Option B preferred over Option A

    Comparison:
    - Complexity: B is 2 weeks vs A is 6 weeks
    - Risk: B is medium, A is high
    - Maintainability: B is adequate, A is better
    
    Given 1-month deadline, Option B's faster timeline 
    outweighs Option A's maintainability advantage.
    
    Score: B=7.5, A=6.0
```

### 3. Missing Evidence

**Bad**:
```
T8: Cache hit rate has dropped significantly
```

**Good**:
```
T8: Cache hit rate has dropped significantly

    Evidence from monitoring (past 30 days):
    - Day 1-15: 85% hit rate (baseline)
    - Day 16-25: Gradual decline to 70%
    - Day 26-30: Sharp drop to 30%
    
    Source: Datadog metrics, redis-cache-* namespace
```

### 4. Overconfidence

**Bad**:
```
T15: This is definitely the root cause. 100% confident.
```

**Good**:
```
T15: Root cause identified with high confidence

    Likely root cause: Cache cleanup job failure
    
    Confidence: 85%
    
    Supporting evidence (strong):
    - Timing matches symptom onset
    - Logs show job not running
    - Mechanism explains symptoms
    
    Uncertainty factors:
    - Haven't ruled out other contributors (10%)
    - Need staging validation (5%)
```

---

## Advanced Techniques

### Meta-Thoughts

Occasionally step back to assess the thinking process:

```
T25: [META] Assess reasoning quality so far

    Progress check:
    - Problem well-defined (T1-T4) ✅
    - Multiple hypotheses explored (T8-T15) ✅
    - Evidence gathered systematically ✅
    - One hypothesis validated (T20) ✅
    
    Concern: Haven't considered edge cases enough
    
    Adjustment: Next 3 thoughts will explore edge cases before 
    finalizing solution.
```

### Confidence Evolution Tracking

Show how confidence changes with new evidence:

```
T10: Initial hypothesis (confidence: 60%)
T15: Supporting evidence found (confidence: 60% → 75%)
T20: Alternative ruled out (confidence: 75% → 85%)
T25: Staging test successful (confidence: 85% → 95%)
```

### Assumption Tracking

Explicitly track and validate assumptions:

```
T12: Key assumptions to validate

    Assumption 1: Database is bottleneck
    Status: Unvalidated
    Test: Check query times
    
    Assumption 2: Caching will help
    Status: Unvalidated
    Test: Prototype with cache
    
    Assumption 3: Budget allows cache infra
    Status: Validated (spoke with finance)
    
T15: [Update T12] Assumptions validated
    A1: Validated - DB queries are 600ms ✅
    A2: Validated - Cache prototype shows 50ms ✅
    A3: Still valid ✅
```

---

*Part of the ai-reasoning plugin documentation*
