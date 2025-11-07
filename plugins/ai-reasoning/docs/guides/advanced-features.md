# Advanced Features

**Version**: 1.0.3
**Purpose**: Advanced patterns for power users

---

## Session Management

### Resuming Ultra-Think Sessions

**Problem**: Long analysis interrupted, need to continue later

**Solution**: Save state and resume

```bash
# Start session
/ultra-think "Complex architecture decision" --depth=deep

# ... work for 45 minutes ...
# Save progress (happens automatically in session state)

# Later: Resume from last checkpoint
# Context is preserved across session boundaries
```

**What's Preserved**:
- All thoughts up to interruption
- Framework selection
- Evidence gathered
- Confidence levels
- Branch states

**Best Practices**:
- Note where you left off (T27, Branch B)
- Review last 5-10 thoughts before continuing
- Verify assumptions still valid (things change!)

---

### Multi-Session Workflows

**Pattern**: Break large analysis into multiple sessions

**Example - Research Project Assessment**:
```
Session 1 (30 min): Methodology assessment
→ /reflection research --focus=methodology

Session 2 (30 min): Reproducibility audit
→ /reflection research --focus=reproducibility

Session 3 (45 min): Publication readiness
→ /reflection research --focus=publication

Session 4 (20 min): Synthesis
→ /reflection research --agents=all
```

**Benefits**:
- Focused analysis per dimension
- Can be done by different people
- Avoids fatigue
- Easier to schedule

**Coordination**:
- Share findings between sessions
- Final session synthesizes all
- Document cross-session dependencies

---

## Multi-Agent Patterns

### Pattern 1: Parallel Specialist Analysis

**Use Case**: Complex project needs multiple perspectives

```bash
/reflection code research workflow --agents=all --depth=deep
```

**What Happens**:
1. Code agent analyzes code quality
2. Research agent assesses methodology
3. Workflow agent evaluates practices
4. Meta-orchestrator synthesizes findings

**Output**:
- Individual agent reports
- Cross-agent pattern detection
- Synthesized recommendations
- Priority ranking

**When to Use**:
- Research projects (code + methodology)
- Major refactorings (code + architecture + practices)
- End-of-quarter reviews

---

### Pattern 2: Sequential Expert Consultation

**Use Case**: Each agent builds on previous insights

```bash
# Step 1: Code analysis
/reflection code --depth=deep

# Step 2: Use findings to guide research reflection
/reflection research --context="Code review identified reproducibility gaps"

# Step 3: Workflow improvements based on both
/reflection workflow --context="Address gaps from code and research reviews"
```

**Benefits**:
- Each analysis informs next
- Progressive refinement
- Targeted improvements

---

### Pattern 3: Contradiction Detection

**Use Case**: Identify disagreements between agents

**Example**:
```
Session agent: "Reasoning quality: 8.5/10"
Code agent: "Implementation quality: 6.0/10"

Contradiction detected:
"High reasoning quality but low implementation quality
suggests gap between design and execution."

Recommendation: Focus on improving implementation
to match design quality.
```

**Value**:
- Identifies misalignments
- Highlights blind spots
- Prompts deeper investigation

---

## Advanced Branching Techniques

### Weighted Branching

**Problem**: Multiple paths, different likelihoods

**Technique**: Allocate effort proportional to likelihood

```
T15: Three hypotheses

Branch A (Database): 60% likely → Allocate 6 thoughts
Branch B (Cache): 30% likely → Allocate 3 thoughts  
Branch C (Network): 10% likely → Allocate 1 thought

Budget: 10 thoughts total for all branches
```

**Benefits**:
- Efficient resource allocation
- Still explore all paths
- Prioritize likely causes

---

### Cascading Branches

**Pattern**: Branches that spawn sub-branches

```
T10: Main branch point

Branch A: Database optimization
  T11: Sub-branch A1: Query optimization
  T12: Sub-branch A2: Index optimization
  
Branch B: Caching strategy
  T13: Sub-branch B1: Redis cache
  T14: Sub-branch B2: In-memory cache

[Explore each, converge based on evidence]
```

**When to Use**:
- Complex problems with multiple layers
- Need to explore design space
- Trade-off analysis

**Caution**: Can get unwieldy >3 levels deep

---

### Parallel Branch Validation

**Pattern**: Test multiple branches simultaneously

```
T15: Two promising hypotheses

[Fork validation]
T16a [Branch A]: Run experiment A (async)
T16b [Branch B]: Run experiment B (async)

[Wait for results]
T17: [Converge] Both experiments completed
     Branch A: 85% validated
     Branch B: 40% validated
     → Focus on Branch A
```

**When to Use**:
- Can run tests in parallel
- Both paths expensive to explore
- Want to reduce time-to-decision

---

## Confidence Engineering

### Confidence Tracking Over Time

**Pattern**: Document how confidence evolves

```
T10: Initial hypothesis (confidence: 60%)
     Basis: Intuition, limited evidence

T15: Gathered evidence (confidence: 60% → 75%)
     + Found supporting logs (+15%)
     
T20: Alternative ruled out (confidence: 75% → 85%)
     + Competing hypothesis refuted (+10%)
     
T25: Local reproduction (confidence: 85% → 95%)
     + Successfully reproduced (+10%)
     
T30: Staging validation (confidence: 95% → 98%)
     + Passed staging tests (+3%)
     
Final confidence: 98%
```

**Benefits**:
- Shows evidence accumulation
- Validates reasoning process
- Helps calibration over time

---

### Confidence Decomposition

**Pattern**: Break confidence into components

```
T25: Root cause confidence: 90%

Decomposition:
  Symptom explanation: 95% (very clear)
  Mechanism understanding: 90% (well understood)
  Code analysis: 95% (found exact bug)
  Local reproduction: 85% (reproduced, minor variance)
  Staging validation: pending (not yet tested)

Weighted confidence:
= (0.95 × 0.2) + (0.90 × 0.2) + (0.95 × 0.2) + (0.85 × 0.2) + (0.0 × 0.2)
= 0.73 → 73% with current evidence
= Will increase to 90% after staging validation
```

**Benefits**:
- Identifies confidence bottlenecks
- Shows what evidence would help
- More precise than single number

---

### Calibration Feedback Loop

**Pattern**: Track actual outcomes vs predicted confidence

```
Decision Log:

Decision 1: Choose ClickHouse (confidence: 85%)
Outcome: Successful, exceeded expectations
Calibration: Well-calibrated (85% → 100% success)

Decision 2: Fix will resolve issue (confidence: 95%)
Outcome: Resolved, but had minor side effect
Calibration: Slightly overconfident (95% → 90% in retrospect)

Decision 3: API optimization (confidence: 70%)
Outcome: Successful, met targets
Calibration: Underconfident (70% → 95% in retrospect)

Trend: Slightly overconfident on bug fixes, 
       underconfident on performance work
       
Adjustment: Reduce bug fix confidence by 5%,
            increase perf confidence by 10%
```

**Benefits**:
- Improves calibration over time
- Identifies systematic biases
- Builds confidence in decision-making

---

## Integration Patterns

### Pattern 1: Ultra-Think + Reflection

```bash
# 1. Deep analysis of problem
/ultra-think "How to optimize ML pipeline?" --depth=deep

# 2. Implement solution
[... implementation work ...]

# 3. Reflect on quality of implementation
/reflection code --depth=shallow

# 4. Reflect on quality of reasoning
/reflection session
```

**Value**:
- Validates reasoning quality
- Catches implementation gaps
- Continuous improvement loop

---

### Pattern 2: Reflection + Ultra-Think

```bash
# 1. Assess current state
/reflection code research workflow --agents=all

# 2. Identify critical issues
# Output: "Technical debt critical (3.5/10), refactoring needed"

# 3. Deep dive on how to address
/ultra-think "How to reduce technical debt to 7/10?" --framework=decision-analysis

# 4. Implement recommendations
[... work ...]

# 5. Validate improvement
/reflection code --mode=quick-check
```

**Value**:
- Reflection finds problems
- Ultra-think solves them
- Validation closes loop

---

### Pattern 3: Chain of Reasoning

```bash
# Session 1: What's wrong?
/ultra-think "Why is system slow?" --framework=root-cause-analysis

# Session 2: How to fix long-term?
/ultra-think "How to prevent performance issues?" --framework=first-principles

# Session 3: Which solution to implement?
/ultra-think "Caching vs scaling vs optimization?" --framework=decision-analysis

# Session 4: Validate approach quality
/reflection session --depth=deep
```

**Value**:
- Each session builds on previous
- Progressive refinement
- Comprehensive solution

---

## Custom Frameworks

### Creating Framework Variants

**Example**: Root Cause Analysis with Performance Focus

```
Custom Framework: "Performance RCA"

Phase 1: Symptom Analysis (T1-T8)
  + Include: Metrics, profiling, benchmarks
  + Focus: Latency, throughput, resource usage
  
Phase 2: Hypothesis Generation (T9-T15)
  + Categories: Algorithm, I/O, Network, Memory
  + Prioritize: By performance impact
  
Phase 3: Validation (T16-T25)
  + Method: Profiling, A/B testing, benchmarks
  + Quantify: Performance improvements
  
[Standard RCA phases continue...]
```

**When to Create**:
- Recurring problem type in your domain
- Standard framework needs adaptation
- Team has specific needs

---

### Framework Composition

**Pattern**: Combine framework phases

**Example**: "Design-Driven Decision"

```
Phases 1-2: Design Thinking (empathize, ideate)
→ Generate user-centric alternatives

Phase 3: Decision Analysis (evaluate)
→ Score options against criteria

Phase 4: Systems Thinking (validate)
→ Check for system-wide impacts

Phase 5: Scientific Method (test)
→ A/B test with users
```

**Use Case**: Product decisions needing creativity + rigor

---

## Automation and Tooling

### Automated Reflection Triggers

**Pattern**: Trigger reflections automatically

**Git Hook Example**:
```bash
# .git/hooks/pre-push

# If >1000 lines changed, suggest reflection
LINES_CHANGED=$(git diff --stat main | tail -1 | awk '{print $4}')

if [ "$LINES_CHANGED" -gt 1000 ]; then
    echo "⚠️  Large change detected ($LINES_CHANGED lines)"
    echo "Consider running: /reflection code --mode=quick-check"
fi
```

**CI/CD Integration**:
```yaml
# .github/workflows/reflection.yml

name: Weekly Reflection Reminder

on:
  schedule:
    - cron: '0 9 * * MON'  # Every Monday 9am

jobs:
  remind:
    runs-on: ubuntu-latest
    steps:
      - name: Post reminder
        run: |
          echo "Weekly reflection reminder!"
          echo "Run: /reflection code research workflow"
```

---

### Metrics Tracking

**Pattern**: Track ultra-think and reflection metrics

**Metrics to Track**:
```
Ultra-Think:
- Session count per week
- Average session duration
- Confidence levels (distribution)
- Framework usage (which frameworks used most)
- Success rate (revisit after 30 days)

Reflection:
- Reflection frequency
- Dimensional scores over time
- Recommendation completion rate
- Time to action
- Quality improvement trajectory
```

**Dashboard Example**:
```
📊 AI Reasoning Metrics (Q1 2025)

Ultra-Think:
- Sessions: 24
- Avg duration: 47 min
- Avg confidence: 87%
- Success rate: 92% (22/24 decisions validated)

Reflection:
- Code quality: 6.5 → 8.0 (+1.5) ✅
- Tech debt: 7.0 → 5.5 (-1.5) ✅  
- Test coverage: 45% → 78% (+33pp) ✅

ROI: ~15 hours/week saved from better decisions
```

---

## Advanced Output Formats

### Structured Decision Records

**Pattern**: Generate architecture decision records (ADRs)

**Template**:
```markdown
# ADR-027: Database Selection for Analytics Platform

**Date**: 2025-09-20
**Status**: Accepted
**Confidence**: 85%

## Context

[From ultra-think T1-T5]

## Decision

[From ultra-think T31]

## Rationale

[From ultra-think scoring matrix + sensitivity analysis]

## Consequences

[From ultra-think validation phase]

## Alternatives Considered

[From ultra-think evaluation phase]
```

---

### Reflection Report Dashboards

**Pattern**: Generate visual reflection reports

**Components**:
- Radar chart of dimensional scores
- Trend lines over time
- Priority matrix (impact vs effort)
- Recommendation tracking

**Tools**: Generate with matplotlib, plotly, or export to BI tools

---

## Expert Tips

### 1. Meta-Cognition

**Technique**: Think about your thinking

**Example**:
```
T25: [META] Assess reasoning quality

Progress so far:
- Problem well-defined ✅
- Framework appropriate ✅
- Evidence gathered systematically ✅
- One hypothesis validated ✅

Concern: Haven't considered edge cases enough

Adjustment: Next 3 thoughts will explore edge cases
```

**Frequency**: Every 10-15 thoughts for complex analysis

---

### 2. Thought Budgeting

**Technique**: Allocate thoughts to phases

**Example**:
```
Total thought budget: 35 thoughts

Phase 1 (Problem): 5 thoughts (14%)
Phase 2 (Approach): 3 thoughts (9%)
Phase 3 (Analysis): 18 thoughts (51%)
Phase 4 (Synthesis): 5 thoughts (14%)
Phase 5 (Validation): 3 thoughts (9%)
Phase 6 (Finalize): 1 thought (3%)

Monitor: If Phase 3 exceeds 18, reduce other phases
```

---

### 3. Evidence Hierarchy

**Technique**: Weight evidence by strength

```
Strongest (1.0x):
- Direct measurement/experiment
- Reproduced multiple times
- Independent validation

Strong (0.8x):
- Single reproduction
- Expert confirmation
- Multiple indirect sources

Moderate (0.6x):
- Logical inference
- Single source
- Historical precedent

Weak (0.3x):
- Intuition
- Anecdotal evidence
- Assumptions

Use to calibrate confidence:
Confidence = Base × Evidence_Weight
```

---

### 4. Hypothesis Tracking Table

**Technique**: Maintain table of all hypotheses

```
| ID | Hypothesis | Likelihood | Evidence | Status |
|----|-----------|------------|----------|---------|
| H1 | DB slow | 60% → 20% | Query times fast | Refuted |
| H2 | Cache miss | 40% → 90% | Hit rate 30% | Confirmed |
| H3 | Network | 10% → 5% | Latency normal | Unlikely |
```

**Update after each test**

---

### 5. Solution Design Pattern Library

**Technique**: Catalog successful solution patterns

**Example**:
```
Pattern: "Cache-Aside with Lazy Loading"

Context: Performance improvement needed, read-heavy workload

Solution:
1. Check cache first
2. On miss, load from DB
3. Populate cache
4. Return data

When used: 12 times
Success rate: 11/12 (92%)
Avg improvement: 10x latency reduction

Anti-pattern: When writes are frequent (cache invalidation complex)
```

**Build library over time**

---

*Part of the ai-reasoning plugin documentation*
