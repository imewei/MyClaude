---
description: Advanced structured reasoning engine with step-by-step thought processing,
  branching logic, and dynamic adaptation for complex problem-solving
triggers:
- /ultra-think
- advanced structured reasoning engine
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<question-or-problem> [--mode=quick|standard] [--framework=...] [--depth=shallow|deep|ultradeep]`
The agent should parse these arguments from the user's request.

# Ultra-Think: Advanced Structured Reasoning

Systematic problem-solving through structured thought processing, branching exploration, and multi-agent coordination.

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Problem

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Thoughts | Use Case |
|------|----------|----------|----------|
| `--mode=quick` | 5-10 min | 5-8 | Initial exploration, time-constrained |
| standard (default) | 30-90 min | 20-40 | Important decisions, complex problems |
| `--depth=ultradeep` | 2-4 hours | 50-100+ | Strategic, high-stakes, novel problems |

---

## Framework Selection

| Problem Type | Framework | Core Method |
|--------------|-----------|-------------|
| Novel problems, paradigm shifts | **First Principles** | Break to fundamentals, rebuild |
| Complex systems, optimization | **Systems Thinking** | Map relationships, feedback loops |
| Debugging, incidents | **Root Cause Analysis** | 5 Whys, Fishbone, fault trees |
| Technology/architecture choices | **Decision Analysis** | Weighted criteria matrix |
| Product/UX design | **Design Thinking** | Empathize→Define→Ideate→Prototype→Test |
| Research, validation | **Scientific Method** | Hypothesis→Experiment→Analyze |
| Time-critical, competitive | **OODA Loop** | Observe→Orient→Decide→Act→Loop |

---

## Thought Structure

Each thought uses hierarchical ID: `T[phase].[step].[branch]`

**Format:**
```markdown
### T1.2 - [Stage]: [Title]
**Dependencies**: [T1.1]
**Reasoning**: [Analysis]
**Confidence**: [High/Medium/Low] (0.XX)
**Next**: [Implications]
```

**Stages:** Planning → Analysis → Synthesis → Validation → Revision

---

## Execution Phases

### Phase 1: Problem Understanding (Sequential)
- T1.1: Capture problem statement
- T1.2: Identify constraints/requirements
- T1.3: List assumptions (explicit + implicit)
- T1.4: Define success criteria
- T1.5: Frame core question

### Phase 2: Approach Selection (Sequential)
- T2.1: Evaluate applicable frameworks
- T2.2: Select primary framework
- T2.3: Design reasoning strategy

### Phase 3: Deep Analysis (Parallel Execution)

> **Orchestration Note**: Execute independent analysis branches and multi-agent roles concurrently.

- **Execute framework steps**
- **Branch explorations**: T3.1.1, T3.1.2, etc. (Run concurrently)
- **Multi-Agent Roles**:
    - **Researcher**: Gather evidence/context
    - **Analyst**: Extract patterns
    - **Critic**: Check for bias/consistency
- **Synthesize patterns** (Merge point)

### Phase 4: Synthesis (Sequential)
- Integrate findings
- Draw conclusions
- Formulate recommendations

### Phase 5: Validation (Parallel Execution)
- Check for contradictions
- Verify assumptions held
- Assess confidence levels

### Phase 6: Finalization (Sequential)
- Generate summary
- Create action plan
- Save session

---

## Branching & Revision

**Branch when:**
- Multiple valid approaches exist
- Need to test assumption
- Exploring alternatives

**Revision format:**
```
T3.3: Original (confidence: 0.70)
T3.3.1 [Revision]: Corrected (confidence: 0.90)
  - Reason: Found contradictory evidence
  - Status: T3.3 → Superseded
```

---

## Contradiction Detection

When detected:
1. Flag conflicting thoughts (e.g., T2.3 vs T4.1)
2. Analyze root cause (assumption? data? logic?)
3. Create revision branch from earliest affected
4. Update confidence levels downstream

---

## Multi-Agent Roles

| Agent | Function |
|-------|----------|
| **Planner** | Problem decomposition, framework selection |
| **Researcher** | Evidence gathering, context building |
| **Analyst** | Pattern analysis, insight extraction |
| **Critic** | Consistency checking, bias detection |
| **Synthesizer** | Integration, recommendations |

---

## Quick Mode Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ultra-Think Quick Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem Type: [Detected type]
Framework: [Selected]
Confidence: 0.XX

Top 3 Approaches:
1. [Approach] (Impact: High, Effort: Medium, Confidence: 0.XX)
2. [Approach] (Impact: Medium, Effort: Low, Confidence: 0.XX)
3. [Approach] (Impact: High, Effort: High, Confidence: 0.XX)

Recommended: Start with #X - [Rationale]

Next: /ultra-think "..." --depth=deep for comprehensive analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Standard/Deep Output

**Executive Summary (1-page):**
- Problem statement (2-3 sentences)
- Root cause/key insight + confidence
- Recommended solution with rationale
- Top 3 insights
- Immediate next steps
- Risks & mitigation

**Detailed Report (if needed):**
- Full thought progression
- Branches explored
- Solution comparison matrix
- Implementation roadmap
- Confidence assessment

---

## Session Management

```bash
# Resume session
/ultra-think --resume=ultra-think-20250427-143022

# Sessions saved to: .ultra-think/sessions/<session-id>/
```

---

## Best Practices

1. **Clear framing**: "Reduce API P95 from 500ms to <100ms" not "make API faster"
2. **Track confidence**: Update at each thought, be honest about uncertainty
3. **Embrace branching**: Explore alternatives early
4. **Welcome revisions**: Course-correct when new info emerges
5. **Validate before finalizing**: Challenge assumptions, stress-test conclusions

---

**Execute structured reasoning with thought tracking, branching exploration, and multi-agent coordination.**
