# Best Practices Guide

**Version**: 1.0.3
**Purpose**: Maximize effectiveness of ultra-think and reflection commands

---

## Ultra-Think Best Practices

### 1. Start with Clear Problem Definition

**Good Example**:
```
Problem: API p95 latency increased from 200ms to 800ms over 2 weeks,
affecting 10k daily users. Must reduce to <300ms within 2 days.

Constraints: Cannot add infrastructure, must maintain compatibility
Success: p95 <300ms, no regressions, sustainable fix
```

**Bad Example**:
```
Problem: API is slow, need to make it faster
```

**Why It Matters**:
- Clear definition prevents scope creep
- Constraints guide solution space
- Success criteria enable validation

**Time Investment**: 5-10 minutes upfront saves 30+ minutes later

---

### 2. Choose Framework Intentionally

**Process**:
1. Classify problem type (2 min)
2. Review framework options (3 min)
3. Select best fit (1 min)
4. Commit for at least 30 minutes

**Common Mistake**: Starting without framework selection
**Impact**: 2x longer analysis, lower quality output
**Fix**: Always use Phase 2 (Approach Selection)

**Framework Selection Time**: 5-10 minutes
**Time Saved**: 30-60 minutes in analysis phase

---

### 3. Show Your Work (Document Reasoning)

**Good Thought**:
```
T12: Hypothesis - Cache cleanup job stopped

Evidence:
- Cache hit rate dropped from 85% → 30% at v2.3.1 deploy
- Heap dump shows 200k UserSession objects (expect ~500)
- Job logs show zero executions since deploy

Likelihood: 85%
Test: Review v2.3.1 code changes for cleanup job logic
```

**Bad Thought**:
```
T12: Probably a cache issue, will check the code
```

**Why It Matters**:
- Enables validation of logic
- Others can follow reasoning
- You can revisit later
- Builds confidence in conclusion

---

### 4. Use Evidence, Not Intuition

**Evidence-Based**:
```
T15: ClickHouse is 40x faster than PostgreSQL

Evidence:
- Benchmark with 100M production records
- PostgreSQL: p95 = 1800ms
- ClickHouse: p95 = 45ms
- Ratio: 1800/45 = 40x

Confidence: 95% (direct measurement)
```

**Intuition-Based**:
```
T15: ClickHouse is much faster than PostgreSQL
Confidence: 70% (heard it's faster)
```

**Impact of Evidence**:
- 95% vs 70% confidence
- Defensible recommendation
- Stakeholder buy-in

---

### 5. Calibrate Confidence Appropriately

**Confidence Scale**:
```
95-100%: Strong evidence, validated, nearly certain
85-94%:  Good evidence, high confidence
70-84%:  Reasonable evidence, moderate confidence
50-69%:  Limited evidence, low confidence
<50%:    Speculative, uncertain
```

**Confidence Factors**:

**Increase Confidence (+10-20% each)**:
- Direct evidence (logs, metrics, experiments)
- Multiple independent sources confirm
- Reproduced locally
- Expert validation
- Passed tests/experiments

**Decrease Confidence (-10-20% each)**:
- Based on assumptions
- Single source of evidence
- Hasn't been tested
- Complexity high
- Time pressure

**Example**:
```
Initial hypothesis: 60% (seems likely)
+ Found in logs: +20% → 80%
+ Reproduced locally: +15% → 95%
- Haven't tested fix: -5% → 90%

Final confidence: 90%
```

---

### 6. Branch at Uncertainty, Converge with Evidence

**When to Branch**:
- Multiple hypotheses with similar likelihood (40-60% each)
- Truly different approaches (not just variations)
- Uncertainty is significant
- Cost of exploring both is reasonable

**When NOT to Branch**:
- One hypothesis is clearly better (>70% likelihood)
- Branches will converge immediately
- Time-constrained (pick best path)

**Branching Pattern**:
```
T15: Two hypotheses with similar evidence

Branch A: Database is bottleneck (55% likely)
Branch B: Cache is bottleneck (45% likely)

T16 [Branch A]: Test DB query times
    Result: Queries are fast (avg 50ms)
    Update: Branch A likelihood 55% → 20%

T17 [Branch B]: Test cache hit rate
    Result: Hit rate dropped to 30% (was 85%)
    Update: Branch B likelihood 45% → 90%

T18 [Converge]: Focus on Branch B (cache)
```

---

### 7. Revise When New Evidence Emerges

**Don't Be Afraid to Backtrack**:
```
T12: Initial hypothesis - Database is slow

T15: Test results show database is fast
[REVISION of T12]
New evidence shows database is NOT the bottleneck.
Cache hit rate dropped - likely cache issue instead.
Revising hypothesis direction.

T16: New hypothesis - Cache cleanup stopped
```

**Why Revisions Are Good**:
- Shows you're responding to evidence
- Better than continuing down wrong path
- Builds more accurate mental model

**Red Flag**: 10+ thoughts without revision → might be ignoring evidence

---

### 8. Validate Before Finalizing

**Validation Checklist**:
- [ ] Does solution address all requirements?
- [ ] Are assumptions validated?
- [ ] What could go wrong? (risks)
- [ ] Did I miss better approaches?
- [ ] Do stakeholders agree?
- [ ] Can I implement this?

**Common Validation Gaps**:
1. **Assumption validation**: "We assume X" → Did you verify X?
2. **Risk assessment**: "This should work" → What if it doesn't?
3. **Alternative consideration**: "Best solution" → Did you check alternatives?

---

### 9. Produce Actionable Outputs

**Good Finalization**:
```
RECOMMENDATION: Deploy cache cleanup fix

Action Plan:
1. Code review (30 min)
2. Deploy to staging (15 min)
3. Run cache cleanup at 3am (30 min)
4. Add monitoring alerts (30 min)

Success Criteria:
- p95 latency <300ms (target: <100ms)
- Cache hit rate >80%
- Zero OOM crashes for 7 days

Rollback Plan:
- Revert commit if issues (5 min)
- Restart service as temporary fix (2 min)

Confidence: 95%
```

**Bad Finalization**:
```
We should fix the cache issue.
```

---

### 10. Time-Box Your Sessions

**Recommended Time Limits**:
- Quick mode: 10-15 minutes max
- Standard: 60-90 minutes max
- Deep: 120 minutes max

**Why Time-Box**:
- Prevents perfectionism
- Forces prioritization
- Maintains focus
- Diminishing returns after 2 hours

**If Approaching Limit**:
1. Assess progress (50% done? 80% done?)
2. Prioritize remaining work
3. Consider stopping at "good enough"
4. Document what's missing

---

## Reflection Best Practices

### 1. Reflect Proactively, Not Reactively

**When to Reflect**:
- ✅ After completing major features (every sprint)
- ✅ Before important decisions (architecture changes)
- ✅ After incidents (root cause review)
- ✅ End of research phase (publication readiness)

**Don't Wait For**:
- ❌ Problems to emerge
- ❌ User complaints
- ❌ Project post-mortems
- ❌ Annual reviews

**Proactive vs Reactive**:
```
Proactive: Reflection after feature completion
Result: Found 3 minor issues, fixed before production
Impact: Prevented 2 potential bugs

Reactive: Reflection after production incident
Result: Found root cause, but damage already done
Impact: 4 hours downtime, customer impact
```

**ROI of Proactive Reflection**: 5-10x (prevent issues vs fix them)

---

### 2. Use Quick Mode for Regular Check-ins

**Quick Check Mode** (2-5 minutes):
```bash
/reflection --mode=quick-check
```

**When to Use**:
- Daily/weekly check-ins
- Health assessment
- Before committing major changes
- During sprints

**What You Get**:
- Health scores (0-10)
- Top 3 observations
- Critical issues flagged
- 1-2 recommendations

**Time Investment**: 5 minutes
**Value**: Early warning system

---

### 3. Deep Reflection for Major Work

**When to Use Deep Mode**:
- End of major feature (multi-week work)
- Before publication (research)
- After significant refactoring
- Quarterly reviews

**What to Expect**:
- 30-45 minute analysis
- Multi-agent coordination
- Comprehensive assessment
- Strategic recommendations

**Preparation**:
- Ensure git history is clean
- Have metrics/data available
- Set aside dedicated time
- Clear your mind of biases

---

### 4. Act on Recommendations

**Common Mistake**: Reflect → Ignore findings → Repeat

**Better Approach**:
```
1. Reflect (30 min)
2. Triage recommendations (10 min)
   - Critical: Address this sprint
   - Important: Next sprint
   - Nice to have: Backlog

3. Create tasks (15 min)
4. Schedule work (5 min)
5. Execute (ongoing)
6. Verify improvements (next reflection)
```

**Tracking Impact**:
```
Reflection 1 (Week 1):
- Code quality: 6.5/10
- Recommendations: Reduce duplication, add tests

Actions taken:
- Refactored data processing (2 days)
- Added 15 unit tests (1 day)

Reflection 2 (Week 4):
- Code quality: 7.8/10 ✅ +1.3 improvement
- Validation: Recommendations were effective
```

---

### 5. Multi-Agent for Complex Projects

**When to Use `--agents=all`**:
- Research projects (methodology + code + publication)
- Large refactorings (code + architecture + tests)
- Strategic decisions (technical + business + team)

**What It Provides**:
- Multiple perspectives (session, code, research)
- Cross-agent pattern detection
- Contradictions identified
- Blind spots highlighted

**Example**:
```bash
/reflection research --agents=all

Result:
- Research agent: Methodology strong (8.5/10)
- Code agent: Reproducibility gaps (6.0/10)
- Session agent: Good reasoning (8.0/10)

Cross-agent insight:
All agents flagged: "Missing synthetic dataset"
Priority: HIGH (affects both reproducibility and publication)
```

---

### 6. Honest Self-Assessment

**Be Honest About**:
- Weaknesses and gaps
- Technical debt
- Shortcuts taken
- Known issues
- Assumptions made

**Example**:
```
Good (Honest):
"Testing coverage is 45%, below our 80% target.
Critical paths are tested, but edge cases are missing.
This is technical debt from rushed sprint."

Bad (Defensive):
"Testing is good, we covered the main cases."
```

**Why Honesty Matters**:
- Reflection is for improvement, not judgment
- Hiding issues prevents addressing them
- Honest assessment → better recommendations
- Build culture of continuous improvement

---

### 7. Compare Against Baselines

**Track Over Time**:
```
Project: Analytics Platform

Reflection 1 (Month 1):
- Code quality: 6.0/10
- Tech debt: 7.5/10 (moderate)
- Testing: 45% coverage

Actions: Refactoring sprint

Reflection 2 (Month 3):
- Code quality: 7.5/10 (+1.5) ✅
- Tech debt: 6.0/10 (-1.5) ⚠️ Still concerning
- Testing: 68% coverage (+23pp) ✅

Actions: Dedicate 20% time to debt reduction

Reflection 3 (Month 6):
- Code quality: 8.0/10 (+0.5) ✅
- Tech debt: 7.5/10 (+1.5) ✅ Improving
- Testing: 82% coverage (+14pp) ✅ Target met
```

**Benefits**:
- Shows improvement trajectory
- Validates that actions work
- Motivates team (visible progress)
- Identifies stagnant areas

---

## Common Anti-Patterns

### Anti-Pattern 1: Analysis Paralysis

**Symptom**: 40+ thoughts, no conclusion

**Cause**:
- Perfectionism
- Fear of wrong answer
- Exploring every possible path
- Not time-boxing

**Fix**:
- Set thought budget (e.g., 30 thoughts max)
- Time-box to 90 minutes
- Accept "good enough" (80% confidence)
- Make decision with best available info

---

### Anti-Pattern 2: Jumping to Conclusions

**Symptom**: T1: Problem, T2: Solution

**Cause**:
- Overconfidence
- Time pressure
- Anchoring on first idea
- Skipping analysis

**Fix**:
- Force yourself through framework phases
- Generate 2-3 alternatives
- Gather evidence before deciding
- Minimum 20 thoughts for non-trivial problems

---

### Anti-Pattern 3: Confirmation Bias

**Symptom**: Only seeking evidence that supports initial hypothesis

**Example**:
```
T5: Hypothesis - Database is slow
T6: Found one slow query (confirming)
T7: Query times mostly good (ignoring)
T8: Conclusion - Database is problem
```

**Fix**:
- Actively seek contradicting evidence
- Challenge your own hypotheses
- Consider alternatives seriously
- Use branching for competing hypotheses

---

### Anti-Pattern 4: Vague Reasoning

**Symptom**: "Probably", "might", "seems like", "should work"

**Fix**:
```
Vague: "Option A is probably better"
Specific: "Option A scores 8.5 vs Option B scores 7.2 
          due to 2x performance advantage (weighted 35%)"

Vague: "This should fix the issue"
Specific: "This fixes the root cause (95% confident)
          based on local reproduction and code analysis"
```

---

### Anti-Pattern 5: Ignoring Constraints

**Example**:
```
Constraint: 2-day timeline
Solution: Complete rewrite (2 weeks)

Result: Rejected by stakeholders, wasted time
```

**Fix**:
- Document constraints in T1-T3
- Validate solutions against constraints
- Consider "ideal" vs "realistic" solutions
- Propose phased approach if needed

---

## Productivity Tips

### 1. Use Templates

Keep templates for common patterns:
- Root cause analysis
- Decision matrix
- Reflection report

**Time Saved**: 10-15 minutes per session

---

### 2. Maintain a Knowledge Base

Document learnings:
- Framework selection lessons
- Common patterns in your domain
- Successful solution archetypes

**Example**:
```
Pattern: API Performance Issues

Common Root Causes (in order):
1. Database queries (40% of cases)
2. Cache misses (30%)
3. Network latency (15%)
4. Code inefficiency (15%)

Framework: Root Cause Analysis
Typical time: 30-45 minutes
Success rate: 85%
```

---

### 3. Parallel Thinking Sessions

For complex problems, run sessions in parallel:
```
Session 1: Technical analysis (Engineer A)
Session 2: Business impact (Engineer B)

Sync after 30 minutes:
- Share findings
- Identify gaps
- Converge on solution
```

**When to Use**: High-stakes, time-sensitive decisions

---

### 4. Iterate on Reflection

Don't expect perfection first time:
```
Reflection 1: Individual (15 min)
Reflection 2: Team review (30 min)
Reflection 3: With stakeholders (45 min)

Each iteration adds perspective and depth
```

---

## Measuring Success

### Ultra-Think Metrics

**Process Metrics**:
- Time to decision (aim: <90 min for standard)
- Confidence level (aim: >85% for important decisions)
- Framework adherence (complete all phases)

**Outcome Metrics**:
- Decision quality (revisit after 30 days)
- Stakeholder satisfaction
- Implementation success rate

**Target**: 85%+ success rate (decision was correct in retrospect)

---

### Reflection Metrics

**Process Metrics**:
- Reflection frequency (aim: every sprint)
- Recommendation completion rate (aim: >75%)
- Time to action (aim: <1 week for critical items)

**Outcome Metrics**:
- Code quality trend (improving over time)
- Bug rate trend (decreasing over time)
- Team velocity (stable or improving)

**Target**: Visible improvement every 2-3 months

---

*Part of the ai-reasoning plugin documentation*
