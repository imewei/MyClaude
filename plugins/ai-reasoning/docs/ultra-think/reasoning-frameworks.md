# Reasoning Frameworks

**Version**: 1.0.3
**Purpose**: Detailed guides for all 7 structured reasoning frameworks

---

## Framework Selection Quick Reference

| Framework | Best For | Time | Thoughts |
|-----------|----------|------|----------|
| First Principles | Novel problems, paradigm shifts | 60-120 min | 30-50 |
| Systems Thinking | Complex systems, optimization | 45-90 min | 25-40 |
| Root Cause Analysis | Debugging, incident response | 30-60 min | 20-35 |
| Decision Analysis | Technology choices, architecture | 40-80 min | 25-40 |
| Design Thinking | Product design, UX problems | 50-100 min | 30-45 |
| Scientific Method | Research questions, validation | 60-120 min | 35-50 |
| OODA Loop | Time-critical, competitive situations | 20-40 min | 15-25 |

---

## 1. First Principles

### When to Use
- Solving novel problems with no established solutions
- Questioning fundamental assumptions in your domain
- Designing revolutionary (not evolutionary) solutions
- Breaking free from conventional thinking patterns

### Framework Structure

**Phase 1: Deconstruction (T1-T10)**
```
Goal: Break problem down to fundamental truths

T1: State the problem clearly
T2: List all current assumptions
T3-T5: Question each assumption
    "Why do we assume X?"
    "What if X weren't true?"
    "What evidence supports X?"
T6-T8: Identify fundamental truths
    What MUST be true?
    What are the physical/logical constraints?
T9-T10: Document false constraints
    What are we assuming that isn't necessary?
```

**Phase 2: Reconstruction (T11-T25)**
```
Goal: Build solution from fundamental truths

T11-T15: Generate novel approaches
    Start from fundamentals only
    Ignore "how it's always been done"
T16-T20: Evaluate each approach
    Feasibility given fundamental truths
    Trade-offs and constraints
T21-T25: Synthesize optimal solution
    Combine best elements
    Verify against fundamentals
```

**Phase 3: Validation (T26-T35)**
```
Goal: Ensure solution is sound

T26-T30: Challenge the solution
    What assumptions did we sneak back in?
    Does it violate any fundamental truths?
T31-T35: Implementation path
    Practical steps
    Risk mitigation
```

### Example Application

**Problem**: "How can we make database queries 10x faster?"

**First Principles Analysis**:
```
T1: Current queries take 500ms average
T2: Assumptions to question:
    - Must use SQL database
    - Must query on every request
    - Must store all data
    - Must return complete results

T3-T5: Questioning assumptions
    "Why SQL?" → Historical choice, not fundamental requirement
    "Why query on request?" → Could precompute common queries
    "Why store all data?" → Could keep only recent/relevant data
    "Why complete results?" → Users often need only top N results

T6-T8: Fundamental truths
    - Data must be accessible in <50ms for good UX
    - Most queries are for recent data (80/20 rule)
    - Users rarely scroll past first page

T11-T15: Novel approaches
    1. Hybrid: Keep recent data in Redis, old data in SQL
    2. Precomputation: Generate common query results offline
    3. Streaming: Return first page immediately, rest async

T21: Optimal solution
    Combine all three:
    - Redis cache for recent data (20x faster)
    - Precomputed results for common queries
    - Streaming for long result sets
    
Result: 50ms queries (10x improvement) ✅
```

---

## 2. Systems Thinking

### When to Use
- Analyzing complex systems with multiple interacting components
- Optimizing performance across entire system (not just one part)
- Understanding feedback loops and emergent behaviors
- Predicting unintended consequences of changes

### Framework Structure

**Phase 1: System Mapping (T1-T12)**
```
T1-T3: Identify system boundaries
    What's inside the system?
    What's the environment?
    What are inputs/outputs?

T4-T7: Map components
    List all major components
    Identify component functions
    Document component states

T8-T12: Map relationships
    How do components interact?
    What are the dependencies?
    Identify feedback loops
```

**Phase 2: Dynamic Analysis (T13-T25)**
```
T13-T17: Identify feedback loops
    Positive feedback (amplifying)
    Negative feedback (stabilizing)
    Delayed feedback

T18-T22: Find leverage points
    Where can small changes create big effects?
    What are system bottlenecks?

T23-T25: Predict emergent behaviors
    What happens when components interact?
    Unintended consequences?
```

**Phase 3: Intervention Design (T26-T35)**
```
T26-T30: Design interventions
    Target leverage points
    Consider ripple effects
    Plan for feedback

T31-T35: Simulate outcomes
    Best case
    Worst case
    Most likely case
```

### Example: API Performance Optimization

```
T1-T3: System boundaries
    Components: API server, database, cache, message queue
    Environment: Load balancer, CDN, user requests
    
T4-T7: Components
    - API: Process requests, business logic
    - DB: Store persistent data
    - Cache: Store frequently accessed data
    - Queue: Handle async jobs

T8-T12: Relationships
    API → DB (read/write)
    API → Cache (read/write)
    API → Queue (write)
    Queue → DB (write)

T13-T17: Feedback loops
    POSITIVE: More requests → DB slower → cache miss → more DB load
    NEGATIVE: Cache hit → reduce DB load → faster responses
    DELAYED: Queue full → job delays → user retry → more requests

T18-T22: Leverage points
    1. Cache hit rate (high leverage)
    2. DB query efficiency (medium)
    3. Queue processing rate (medium)

T26: Intervention: Increase cache TTL from 5m to 30m
    Effect: +20% cache hits → -15% DB load → +25% throughput
    
Result: System-wide optimization, not just DB tuning ✅
```

---

## 3. Root Cause Analysis

### When to Use
- Debugging production incidents
- Investigating test failures
- Understanding why a solution isn't working
- Tracing errors to their source

### Framework Structure

**Phase 1: Symptom Analysis (T1-T8)**
```
T1-T2: Describe symptoms precisely
    What is happening?
    When did it start?
    
T3-T5: Gather evidence
    Logs, metrics, traces
    Reproduction steps
    Environment details
    
T6-T8: Identify patterns
    Frequency, timing
    Correlation with changes
```

**Phase 2: Hypothesis Generation (T9-T18)**
```
T9-T12: Generate hypotheses
    Use "5 Whys" technique
    Consider all system layers
    
T13-T15: Prioritize hypotheses
    By likelihood
    By testability
    By impact
    
T16-T18: Design tests
    How to validate each hypothesis?
    What evidence would confirm/refute?
```

**Phase 3: Validation (T19-T30)**
```
T19-T25: Test hypotheses
    Run experiments
    Gather results
    Eliminate false paths
    
T26-T30: Identify root cause
    Verify with 5 Whys
    Distinguish cause from symptom
    Confirm with fix
```

### Example: Memory Leak Investigation

```
T1-T2: Symptom
    Memory usage grows from 500MB to 8GB over 6 hours
    Eventually causes OOM crash
    
T3-T5: Evidence
    - Heap dump shows 200k instances of UserSession
    - Started after deployment v2.3.1
    - Garbage collection logs show old gen filling up
    
T9-T12: Hypotheses
    H1: Sessions not being cleaned up (most likely)
    H2: Memory leak in new caching code
    H3: Database connection pool leak
    H4: Logging framework issue
    
T13-T15: Prioritization
    1. H1 (sessions) - 70% likely, easy to test
    2. H2 (cache) - 20% likely, medium difficulty
    3. H3 (DB) - 5% likely, hard to test
    4. H4 (logging) - 5% likely, hard to test
    
T19-T22: Testing H1
    - Check session cleanup code in v2.3.1
    - Found: cleanup timer not restarted after config reload
    - Confirms: sessions accumulate indefinitely
    
T26: Root cause identified
    Bug in session cleanup introduced in v2.3.1:156
    Config reload logic doesn't restart cleanup timer
    
Result: Bug fixed, memory stable at 500MB ✅
```

---

## 4. Decision Analysis

### When to Use
- Choosing between multiple technical options
- Architectural decisions with long-term impact
- Vendor/technology selection
- Resource allocation decisions

### Framework Structure

**Phase 1: Frame the Decision (T1-T8)**
```
T1-T3: Define decision clearly
    What are we choosing?
    What are the constraints?
    What does success look like?
    
T4-T6: Identify stakeholders
    Who is affected?
    What do they care about?
    
T7-T8: List all viable options
    Brainstorm broadly
    Include status quo
```

**Phase 2: Define Criteria (T9-T15)**
```
T9-T12: Identify criteria
    Performance, cost, complexity, etc.
    
T13-T15: Weight criteria
    What matters most?
    Assign weights (sum to 100)
```

**Phase 3: Evaluate Options (T16-T30)**
```
T16-T25: Score each option
    Rate on scale (1-10) per criterion
    Gather evidence for scores
    
T26-T30: Calculate weighted scores
    Score × Weight for each criterion
    Sum to get total score
```

**Phase 4: Sensitivity Analysis (T31-T40)**
```
T31-T35: Test assumptions
    What if weights change?
    What if scores uncertain?
    
T36-T40: Recommend decision
    Highest score
    Confidence level
    Risk factors
```

### Example: Database Selection

```
T1-T3: Decision
    Choose database for new analytics platform
    Constraints: <$50k/year, <100ms p95 latency
    Success: Handles 10M records, 1000 QPS
    
T7-T8: Options
    1. PostgreSQL (relational)
    2. MongoDB (document)
    3. Cassandra (columnar)
    4. ClickHouse (OLAP)
    
T9-T15: Criteria & Weights
    - Query performance: 30
    - Scalability: 25
    - Operational complexity: 20
    - Cost: 15
    - Ecosystem: 10
    
T16-T25: Scoring (1-10 scale)
    
    PostgreSQL: 7×30 + 6×25 + 9×20 + 9×15 + 10×10 = 740
    MongoDB:    6×30 + 7×25 + 7×20 + 7×15 + 8×10  = 635
    Cassandra:  8×30 + 9×25 + 5×20 + 6×15 + 6×10  = 695
    ClickHouse: 10×30 + 8×25 + 6×20 + 8×15 + 5×10 = 760 ⭐
    
T31-T35: Sensitivity
    If query performance weight drops to 20:
    ClickHouse: 690, PostgreSQL: 670 (still wins)
    
    If operational complexity critical (weight 35):
    PostgreSQL: 795 (wins), ClickHouse: 730
    
Recommendation: ClickHouse (score: 760)
Confidence: 75%
Caveat: Only if team can handle moderate ops complexity
```

---

## 5. Design Thinking

### When to Use
- Product design and UX problems
- User-centric solutions
- Innovation challenges
- Service design

### Framework Structure

**Phase 1: Empathize (T1-T10)**
```
T1-T5: Understand users
    Who are they?
    What are their goals?
    What are their pain points?
    
T6-T10: Gather insights
    User research findings
    Behavioral patterns
    Context of use
```

**Phase 2: Define (T11-T18)**
```
T11-T15: Synthesize findings
    Identify patterns
    Core problems
    
T16-T18: Frame problem
    User-centric problem statement
    "How might we...?"
```

**Phase 3: Ideate (T19-T30)**
```
T19-T25: Generate ideas
    Brainstorm widely
    Build on others' ideas
    Defer judgment
    
T26-T30: Converge
    Group similar ideas
    Select promising concepts
```

**Phase 4: Prototype & Test (T31-T40)**
```
T31-T35: Quick prototypes
    Low-fidelity mockups
    Paper prototypes
    
T36-T40: Test with users
    Gather feedback
    Iterate rapidly
```

---

## 6. Scientific Method

### When to Use
- Research questions requiring validation
- Testing hypotheses about system behavior
- Experimental design
- Validating assumptions empirically

### Framework Structure

**Phase 1: Observation & Question (T1-T8)**
```
T1-T3: Observe phenomenon
    What did you notice?
    What's surprising or unexplained?
    
T4-T8: Formulate question
    Clear, testable question
    Define variables
```

**Phase 2: Hypothesis (T9-T15)**
```
T9-T12: Develop hypothesis
    Proposed explanation
    Testable prediction
    
T13-T15: Identify variables
    Independent variable
    Dependent variable
    Control variables
```

**Phase 3: Experiment Design (T16-T25)**
```
T16-T20: Design experiment
    How to test hypothesis?
    Controls needed
    Sample size
    
T21-T25: Plan measurements
    What to measure?
    How to measure?
    Statistical tests
```

**Phase 4: Analysis (T26-T40)**
```
T26-T32: Conduct experiment
    Execute protocol
    Collect data
    Document observations
    
T33-T40: Analyze results
    Statistical analysis
    Support or refute hypothesis?
    Draw conclusions
```

---

## 7. OODA Loop

### When to Use
- Time-critical decisions
- Competitive situations
- Rapidly changing environments
- Incident response

### Framework Structure

**Observe (T1-T5)**: Gather current situation data
**Orient (T6-T10)**: Analyze context, update mental model
**Decide (T11-T15)**: Choose action based on analysis
**Act (T16-T20)**: Execute decision, monitor results
**Loop**: Repeat faster than competition

### Example: Production Incident

```
Cycle 1 (5 min):
  T1-T2 OBSERVE: Error rate spike, 5% → 25%
  T3-T4 ORIENT: Recent deployment, possibly related
  T5 DECIDE: Rollback deployment
  T6 ACT: Initiate rollback
  
Cycle 2 (3 min):
  T7 OBSERVE: Error rate still 20%, not improving
  T8 ORIENT: Rollback not sufficient, deeper issue
  T9 DECIDE: Check database metrics
  T10 ACT: Query DB monitoring
  
Cycle 3 (2 min):
  T11 OBSERVE: DB connection pool exhausted
  T12 ORIENT: Connection leak likely
  T13 DECIDE: Restart API servers with conn limit
  T14 ACT: Rolling restart
  T15 OBSERVE: Error rate dropping, 20% → 5%
  
Result: Incident resolved in 10 minutes ✅
```

---

## Framework Combination Patterns

### Sequence: System Thinking → Decision Analysis
```
Use Case: Optimizing complex system
1. Systems Thinking: Understand system holistically
2. Decision Analysis: Choose optimal intervention

Example: "How to scale our API?"
→ Map system (Systems Thinking)
→ Identify bottlenecks
→ Evaluate scaling options (Decision Analysis)
```

### Sequence: Root Cause → First Principles
```
Use Case: Solving recurring problem permanently
1. Root Cause: Find immediate cause
2. First Principles: Redesign to prevent recurrence

Example: "Memory leaks keep happening"
→ Debug current leak (Root Cause)
→ Redesign memory management (First Principles)
```

### Parallel: Design Thinking + Scientific Method
```
Use Case: Product development with validation
1. Design Thinking: Generate user-centric solution
2. Scientific Method: Validate with users

Example: "New feature design"
→ Empathize with users (Design Thinking)
→ A/B test designs (Scientific Method)
```

---

*Part of the ai-reasoning plugin documentation*
