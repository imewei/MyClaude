# Framework Selection Guide

**Version**: 1.0.3
**Purpose**: Choose the right reasoning framework for your problem

---

## Quick Selection Flow

```
Start → What type of problem?

├─ Novel/revolutionary problem → First Principles
├─ Complex system optimization → Systems Thinking  
├─ Debugging/incident response → Root Cause Analysis
├─ Technology/architecture choice → Decision Analysis
├─ Product/UX design → Design Thinking
├─ Research/validation question → Scientific Method
└─ Time-critical/competitive → OODA Loop
```

---

## Framework Comparison Matrix

| Framework | Problem Type | Time | Complexity | Output |
|-----------|--------------|------|------------|--------|
| **First Principles** | Novel, paradigm-shifting | 60-120m | High | Revolutionary solution |
| **Systems Thinking** | Complex systems, optimization | 45-90m | High | System-wide improvements |
| **Root Cause Analysis** | Debugging, incidents | 30-60m | Medium | Root cause + fix |
| **Decision Analysis** | Choices, trade-offs | 40-80m | Medium | Quantified recommendation |
| **Design Thinking** | User-centric design | 50-100m | Medium | User-validated design |
| **Scientific Method** | Validation, experiments | 60-120m | High | Validated hypothesis |
| **OODA Loop** | Time-critical decisions | 20-40m | Low | Rapid action |

---

## Detailed Selection Criteria

### 1. First Principles

**Use When**:
- ✅ Solving truly novel problems with no precedent
- ✅ Questioning fundamental assumptions in your domain
- ✅ Seeking revolutionary (not evolutionary) solutions
- ✅ Breaking free from "how we've always done it"

**Don't Use When**:
- ❌ Time-constrained (need answer quickly)
- ❌ Established best practices exist
- ❌ Incremental improvement is sufficient
- ❌ Stakes are low (over-engineering)

**Examples**:
- "How can we make rocket launches 100x cheaper?" (SpaceX)
- "Can we eliminate servers entirely?" (serverless computing)
- "How to make  database queries 30x faster?" (rethink storage layer)

**Key Questions**:
1. Can I challenge every assumption?
2. Do I need a revolutionary solution?
3. Is the conventional approach fundamentally limited?
4. Do I have time for deep deconstruction?

**Confidence**: Use when 80%+ confident you need to rethink fundamentals

---

### 2. Systems Thinking

**Use When**:
- ✅ Optimizing complex systems with multiple components
- ✅ Understanding feedback loops and emergent behaviors
- ✅ Changes affect multiple parts of system
- ✅ Unintended consequences are likely

**Don't Use When**:
- ❌ Problem is isolated to single component
- ❌ Simple cause-effect relationship
- ❌ No interdependencies
- ❌ Linear problem (no feedback loops)

**Examples**:
- "How to scale API without breaking other services?"
- "Why does fixing one bottleneck create another?"
- "How to optimize entire data pipeline (not just one stage)?"

**Key Questions**:
1. Are there multiple interacting components?
2. Will my change ripple through the system?
3. Are there feedback loops (positive or negative)?
4. Do I need to optimize holistically?

**Confidence**: Use when system has 3+ interacting components

---

### 3. Root Cause Analysis

**Use When**:
- ✅ Debugging production issues
- ✅ Investigating test failures
- ✅ Understanding why something broke
- ✅ Recent change caused regression

**Don't Use When**:
- ❌ Designing new features (use Design Thinking)
- ❌ Choosing between options (use Decision Analysis)
- ❌ No clear symptom to investigate
- ❌ Symptom is understood, need solution design

**Examples**:
- "Why is API throwing 500 errors?"
- "Why did deployment cause memory leak?"
- "Why do tests fail intermittently?"

**Key Questions**:
1. Is there a clear symptom or failure?
2. Do I need to find the cause?
3. Was it working before (regression)?
4. Do I have evidence/logs to analyze?

**Confidence**: Use when problem has clear symptom + timeline

---

### 4. Decision Analysis

**Use When**:
- ✅ Choosing between 3+ viable options
- ✅ Multiple competing criteria
- ✅ Stakeholder buy-in is important
- ✅ Need objective, defensible choice

**Don't Use When**:
- ❌ Only 1-2 options (just compare directly)
- ❌ Decision is obvious
- ❌ Must decide immediately (<1 day)
- ❌ Low stakes (not worth formal analysis)

**Examples**:
- "Which database for analytics platform?"
- "React vs Vue vs Svelte for new project?"
- "Build vs buy vs partner for authentication?"

**Key Questions**:
1. Do I have 3+ viable alternatives?
2. Are there complex trade-offs?
3. Do I need stakeholder alignment?
4. Is this a high-stakes decision?

**Confidence**: Use when decision affects multiple people/teams

---

### 5. Design Thinking

**Use When**:
- ✅ Designing user-facing features
- ✅ User needs are unclear
- ✅ Need creative, user-centric solution
- ✅ Can iterate with user feedback

**Don't Use When**:
- ❌ Technical problem (no user aspect)
- ❌ User needs are well-defined
- ❌ Can't get user feedback
- ❌ Backend/infrastructure problem

**Examples**:
- "How to improve onboarding experience?"
- "Design dashboard for non-technical users"
- "Make complex workflow intuitive"

**Key Questions**:
1. Is this a user-centric problem?
2. Do I understand user needs deeply?
3. Can I prototype and test with users?
4. Is creativity/innovation important?

**Confidence**: Use when user experience is primary concern

---

### 6. Scientific Method

**Use When**:
- ✅ Testing hypotheses empirically
- ✅ Validating assumptions with data
- ✅ Research questions
- ✅ A/B testing scenarios

**Don't Use When**:
- ❌ Can't run experiments
- ❌ Need answer immediately
- ❌ No measurable outcomes
- ❌ Hypothesis already validated

**Examples**:
- "Does caching improve performance?"
- "Will feature X increase engagement?"
- "Is algorithm A better than algorithm B?"

**Key Questions**:
1. Can I formulate a testable hypothesis?
2. Can I design an experiment?
3. Can I measure outcomes objectively?
4. Do I need empirical validation?

**Confidence**: Use when empirical validation is possible and needed

---

### 7. OODA Loop

**Use When**:
- ✅ Time-critical situations
- ✅ Rapidly changing environment
- ✅ Competitive scenarios
- ✅ Incident response

**Don't Use When**:
- ❌ Have time for deeper analysis
- ❌ Situation is stable
- ❌ One-time decision (not ongoing)
- ❌ Need comprehensive solution

**Examples**:
- "Production outage - what do we do NOW?"
- "Competitor launched feature - how to respond?"
- "Security incident - immediate action needed"

**Key Questions**:
1. Do I need to act within hours/minutes?
2. Is the situation rapidly evolving?
3. Is speed more important than perfection?
4. Will I iterate multiple times?

**Confidence**: Use when speed is critical (hours, not days)

---

## Framework Combinations

### Common Pairings

**Systems Thinking → Decision Analysis**
```
Use Case: System optimization requiring technology choice
Example: "How to scale our architecture?" 

Step 1: Map system holistically (Systems Thinking)
Step 2: Identify bottlenecks and leverage points
Step 3: Evaluate scaling options (Decision Analysis)
Step 4: Choose optimal approach
```

**Root Cause Analysis → First Principles**
```
Use Case: Recurring problem requiring fundamental fix
Example: "Memory leaks keep happening"

Step 1: Debug current leak (Root Cause Analysis)
Step 2: Identify pattern across incidents
Step 3: Redesign memory management (First Principles)
Step 4: Prevent entire class of bugs
```

**Design Thinking → Scientific Method**
```
Use Case: User-centric design with validation
Example: "New onboarding flow"

Step 1: Empathize with users (Design Thinking)
Step 2: Generate design alternatives
Step 3: A/B test designs (Scientific Method)
Step 4: Validate with user metrics
```

**Root Cause Analysis → Systems Thinking**
```
Use Case: Bug fix with system-wide implications
Example: "Fix causes performance regression elsewhere"

Step 1: Find root cause (Root Cause Analysis)
Step 2: Map system impacts (Systems Thinking)
Step 3: Design fix considering ripple effects
Step 4: Implement with monitoring
```

---

## Decision Trees

### For Technical Problems

```
Q1: Is this a bug/failure?
    Yes → Root Cause Analysis
    No → Go to Q2

Q2: Is this a design/UX problem?
    Yes → Design Thinking
    No → Go to Q3

Q3: Are you choosing between options?
    Yes → Decision Analysis
    No → Go to Q4

Q4: Is it a complex system?
    Yes → Systems Thinking
    No → Go to Q5

Q5: Is it a novel/revolutionary problem?
    Yes → First Principles
    No → Go to Q6

Q6: Need to validate empirically?
    Yes → Scientific Method
    No → Go to Q7

Q7: Is it time-critical?
    Yes → OODA Loop
    No → Consider if you need ultra-think at all
```

### For Strategic Problems

```
Q1: Is this competitive/time-sensitive?
    Yes → OODA Loop
    No → Go to Q2

Q2: Are you making a high-stakes choice?
    Yes → Decision Analysis
    No → Go to Q3

Q3: Is it a novel business model/approach?
    Yes → First Principles
    No → Go to Q4

Q4: Does it involve complex organizational systems?
    Yes → Systems Thinking
    No → Go to Q5

Q5: Is it customer/market facing?
    Yes → Design Thinking
    No → Go to Q6

Q6: Need to validate assumptions?
    Yes → Scientific Method
    No → Consider simpler analysis
```

---

## Case Studies: Framework Selection

### Case 1: API Performance Problem

**Situation**: API latency increased from 200ms to 800ms

**Initial Instinct**: Use Systems Thinking (it's a system)

**Better Choice**: Root Cause Analysis

**Why**:
- Clear symptom (latency increase)
- Recent change likely cause (regression)
- Need to find specific bug
- Systems Thinking would be overkill

**Outcome**: Found bug in 47 minutes, fixed immediately

---

### Case 2: Database Selection

**Situation**: Choose database for new analytics platform

**Initial Instinct**: First Principles (rethink data storage)

**Better Choice**: Decision Analysis

**Why**:
- Not a novel problem (many precedents)
- 4 viable options exist
- Complex trade-offs (performance, cost, ops)
- Need stakeholder buy-in
- First Principles would waste time

**Outcome**: Objective choice, team aligned, successful deployment

---

### Case 3: Recurring Memory Leaks

**Situation**: Third memory leak in 6 months, different causes each time

**Initial Instinct**: Root Cause Analysis (find this leak)

**Better Choice**: Root Cause → First Principles

**Why**:
- Pattern indicates deeper issue
- Need to redesign memory management
- Just fixing symptoms not sustainable
- First Principles will prevent recurrence

**Outcome**: Redesigned lifecycle management, zero leaks since

---

## Common Mistakes

### Mistake 1: Using First Principles for Everything

**Problem**: "Let's rethink everything from scratch!"

**Why It's Wrong**:
- Most problems have established solutions
- Reinventing the wheel is expensive
- Time constraints don't allow it
- Risk of worse solution than existing

**When to Avoid**: If established best practices exist and work

**Fix**: Use First Principles only for truly novel problems

---

### Mistake 2: Skipping Framework Selection

**Problem**: "I'll just start thinking and figure it out"

**Why It's Wrong**:
- Leads to unstructured analysis
- Easy to go in circles
- Miss important considerations
- Waste time on wrong approach

**Impact**: 2-3x longer analysis, lower quality

**Fix**: Spend 5 minutes selecting framework upfront

---

### Mistake 3: Framework Hopping

**Problem**: "This framework isn't working, let me try another"

**Why It's Wrong**:
- Frameworks need time to work
- Switching wastes initial effort
- Might just be at hard part
- Analysis becomes fragmented

**When It's OK**: After 30+ min if truly not working

**Fix**: Commit to chosen framework for at least 30 minutes

---

### Mistake 4: Overcomplicating Simple Problems

**Problem**: Using Decision Analysis for 2-option choice

**Why It's Wrong**:
- Formal framework is overkill
- Simple comparison would suffice
- Wastes time on process
- Over-engineering decision

**Fix**: Use frameworks when complexity justifies it

---

## Framework Selection Checklist

Before starting ultra-think, answer these:

### Problem Characteristics

- [ ] What type of problem is this? (bug, design, choice, optimization, novel)
- [ ] How complex is it? (simple, moderate, complex)
- [ ] How urgent is it? (minutes, hours, days, weeks)
- [ ] What are the stakes? (low, medium, high, critical)

### Framework Fit

- [ ] Which framework matches problem type?
- [ ] Do I have enough time for this framework?
- [ ] Do I have the information needed (data, evidence, options)?
- [ ] Is this framework worth the effort for this problem?

### Confidence Check

- [ ] Am I 70%+ confident this is the right framework?
- [ ] Would a different framework be clearly better?
- [ ] Do I need to combine multiple frameworks?

### Commitment

- [ ] Can I commit to this framework for 30+ minutes?
- [ ] Do I understand the framework phases?
- [ ] Am I ready to follow the structure?

---

*Part of the ai-reasoning plugin documentation*
