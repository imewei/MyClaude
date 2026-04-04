---
name: reasoning-frameworks
description: Structured reasoning frameworks for complex problem-solving. Covers First Principles, Root Cause Analysis, Decision Analysis, Systems Thinking, and OODA Loop with branching exploration, confidence scoring, and meta-cognitive reflection. Use when applying systematic analysis to debugging, architectural decisions, strategic planning, or high-stakes decisions.
---

# Reasoning Frameworks

## Expert Agent

For structured reasoning, cognitive frameworks, and meta-cognitive analysis, delegate to:

- **`reasoning-engine`**: Applies First Principles, Root Cause Analysis, Tree-of-Thought, and systematic problem decomposition with confidence scoring.
  - *Location*: `plugins/agent-core/agents/reasoning-engine.md`

Unified guide for maximizing cognitive performance through systematic reasoning, structured thought processing, meta-cognitive reflection, and self-correction.

---

## From: Advanced Reasoning

Expert guide for maximizing LLM cognitive performance through systematic reasoning and self-correction.

### 1. Reasoning Frameworks

- **First Principles**: Deconstruct problems into fundamental truths and rebuild solutions from the ground up.
- **Systems Thinking**: Analyze complex interdependencies and feedback loops within a system.
- **Root Cause Analysis (RCA)**: Use the "Five Whys" and Fishbone diagrams to identify underlying issues.
- **OODA Loop**: Rapidly observe, orient, decide, and act in dynamic environments.

### 2. Thought Structuring

- **Explicit Chains**: Document the step-by-step logic used to reach a conclusion.
- **Branching**: Explore multiple potential solutions concurrently before committing to a path.
- **Evidence-Based**: Support every major claim with data, logs, or established facts.
- **Confidence Scoring**: Assign honest confidence levels to conclusions to highlight uncertainty.

### 3. Meta-Cognitive Reflection

- **Self-Correction**: Periodically review the reasoning chain for contradictions or invalid assumptions.
- **Bias Detection**: Actively search for and mitigate common cognitive biases (e.g., confirmation bias).
- **Session Reflection**: Analyze the entire problem-solving session to improve future reasoning strategies.

### 4. Reasoning Checklist

- [ ] **Assumptions**: Are all assumptions made explicit and validated?
- [ ] **Branches**: Have alternative interpretations or solutions been considered?
- [ ] **Contradictions**: Has the reasoning chain been checked for internal consistency?
- [ ] **Auditability**: Can a human follow the logic from the problem statement to the final decision?

---

## From: Structured Reasoning

Systematic problem-solving with explicit reasoning chains and cognitive frameworks.

### Framework Selection

| Framework | Use Case | Example |
|-----------|----------|---------|
| First Principles | Novel problems, paradigm shifts | Design new architecture |
| Root Cause Analysis | Debugging, incidents, quality issues | Production outage |
| Decision Analysis | Technology choices, strategic planning | Framework selection |
| Systems Thinking | Complex interdependencies | Distributed systems |
| OODA Loop | Rapid adaptive decisions | Incident response |

---

### Six-Phase Process

| Phase | Goal | Key Outputs |
|-------|------|-------------|
| 1. Understanding | Comprehend problem space | Constraints, assumptions, success criteria |
| 2. Approach | Select framework | Framework choice, reasoning strategy |
| 3. Analysis | Execute framework | Multiple solution branches, evidence |
| 4. Synthesis | Integrate findings | Key insights, conclusions |
| 5. Validation | Verify integrity | Contradiction check, confidence levels |
| 6. Finalization | Generate outputs | Summary, action plan, decisions |

---

### Thought Structure

```yaml
thought:
  id: "T3.2.1"  # Phase.Step.Branch
  stage: "analysis"  # planning|analysis|synthesis|revision|validation
  dependencies: ["T3.1", "T3.2"]
  content: "[Detailed reasoning]"
  evidence: "[Supporting facts]"
  assumptions: "[Stated assumptions]"
  confidence: 0.85  # 0-1 scale
  status: "active"  # active|revised|superseded|validated
```

---

### First Principles Framework

**When**: Novel problems, deep understanding needed

```
T3.1: List all assumptions
T3.2: Challenge each - "Is this necessarily true?"
T3.3: Identify fundamental truths (physics, math, logic)
T3.4: Reconstruct solution from fundamentals
T3.5: Validate reconstruction
```

---

### Root Cause Analysis

**When**: Debugging, incident response, quality issues

```
T3.1: Define problem precisely (symptoms, scope, timeline)
T3.2: Gather evidence (logs, metrics, observations)
T3.3: Generate hypotheses
  T3.3.1: Test hypothesis 1
  T3.3.2: Test hypothesis 2
T3.4: Validate root cause
T3.5: Propose solutions
```

**Techniques**: 5 Whys, Fishbone diagram, Fault Tree Analysis

---

### Decision Analysis

**When**: Technology choices, architectural decisions

```
T3.1: Define criteria with weights
T3.2: Generate 3-5 alternatives
T3.3-T3.5: Score each alternative
T3.6: Compare tradeoffs
T3.7: Make recommendation with confidence
```

---

### Branching & Revision

#### When to Branch
- Alternative solutions: T3.1.1, T3.1.2, T3.1.3
- Hypothesis testing: T3.2.v (validation)
- Different frameworks: T3.1.f (framework switch)

#### When to Revise
- New evidence contradicts previous thought
- Assumption found invalid
- Better approach identified

```
T3.5.1 [Revision of T3.5]
Original: [Previous reasoning]
Reason: [Why update needed]
Updated: [Corrected analysis]
Confidence: +0.10 (0.75 -> 0.85)
```

---

### Confidence Levels

| Level | Range | Criteria |
|-------|-------|----------|
| High | 0.85-1.0 | Strong evidence, validated, low uncertainty |
| Medium | 0.60-0.84 | Moderate evidence, some assumptions unvalidated |
| Low | 0.0-0.59 | Weak evidence, high uncertainty, needs investigation |

---

### Contradiction Detection

| Level | Type | Example |
|-------|------|---------|
| Semantic | Conflicting statements | "Read-heavy" vs "Needs write optimization" |
| Constraint | Requirement violation | Budget $50K vs Cost $75K |
| Temporal | Cause-effect inconsistency | "Reduces latency" vs "Increases latency" |

**Resolution**: Flag -> Analyze root cause -> Create revision branch -> Update chain -> Validate downstream

---

### Depth Modes

| Mode | Thoughts | Duration | Branches | Use Case |
|------|----------|----------|----------|----------|
| Shallow | 5-15 | 5-10 min | 1-2 | Quick analysis, initial exploration |
| Deep | 20-40 | 30-60 min | 3-5 | Important decisions, complex problems |
| Ultra-Deep | 50-100+ | 2-4 hrs | 10+ | Strategic decisions, high-stakes |

---

### Best Practices

| Practice | Implementation |
|----------|----------------|
| Explicit dependencies | List which thoughts each builds upon |
| Evidence-based | Support with concrete facts/data |
| Assumption tracking | Make all assumptions explicit |
| Confidence honesty | Low confidence acceptable if uncertain |
| Branch early | Explore alternatives before committing |
| Revise freely | Course-correct with new information |
| Validate thoroughly | Apply full checklist before finalizing |

---

### Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Skipping phases | Miss critical constraints or assumptions |
| Single path | No alternative exploration |
| Implicit assumptions | Unvalidated beliefs affect conclusions |
| Overconfidence | High confidence without evidence |
| No revision | Failing to update with new information |

### Parallel Reasoning

| Pattern | Implementation | Use Case |
|---------|----------------|----------|
| **Branching Exploration** | Explore T3.1.1, T3.1.2 concurrently | Rapid hypothesis testing |
| **Multi-Framework** | Run RCA and Systems Thinking in parallel | Complex incidents |
| **Team Reasoning** | Assign branches to different experts | Large-scale strategy |
| **Background Validation** | Async fact-checking of assumptions | Real-time confidence updates |

### Comprehensive Checklist

- [ ] Problem fully understood with constraints
- [ ] Framework selected appropriate to problem type
- [ ] Multiple branches explored
- [ ] Evidence gathered for key claims
- [ ] Assumptions made explicit and validated
- [ ] Confidence levels assigned honestly
- [ ] Contradictions checked and resolved
- [ ] Conclusions synthesized with rationale

## Related Skills

- `reflection-framework` -- Meta-cognitive reflection and bias detection that extends reasoning analysis
- `llm-application-patterns` -- Prompt engineering patterns (CoT, few-shot) that apply reasoning frameworks
