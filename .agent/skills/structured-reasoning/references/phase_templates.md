# Phase Templates

Detailed templates for each of the six reasoning phases with structured thought formats, key questions, and quality criteria.

---

## Phase 1: Problem Understanding

### Goal
Fully comprehend the problem space before attempting solutions.

### Thought Template

**T1.1 - Capture Problem**
```yaml
id: T1.1
stage: planning
dependencies: []
content: |
  Core question: [Restate the problem in your own words]
  Stakeholders: [Who is affected or involved?]
  Context: [Relevant background information]
  Constraints: [Known limitations]
evidence: [Source of problem statement, user input, etc.]
assumptions: [Any initial assumptions being made]
confidence: 0.7-0.9
status: active
```

**T1.2 - Identify Constraints**
```yaml
id: T1.2
stage: planning
dependencies: [T1.1]
content: |
  Technical constraints: [Technology limitations, compatibility, performance]
  Business constraints: [Budget, timeline, resources, regulations]
  Resource constraints: [Team size, skills, availability]
  Timeline constraints: [Deadlines, milestones]
evidence: [Requirements docs, stakeholder input, technical specs]
assumptions: [Assumptions about constraints]
confidence: 0.6-0.8
status: active
```

**T1.3 - Surface Assumptions**
```yaml
id: T1.3
stage: planning
dependencies: [T1.1, T1.2]
content: |
  Explicit assumptions: [Stated assumptions]
  Implicit assumptions: [Unstated but present assumptions]
  Critical assumptions: [Assumptions that would break solution if false]
  Assumptions to validate: [List requiring verification]
evidence: [Problem statement analysis, domain knowledge]
assumptions: [Meta-assumptions about the assumptions]
confidence: 0.5-0.7
status: active
```

**T1.4 - Define Success**
```yaml
id: T1.4
stage: planning
dependencies: [T1.1, T1.2, T1.3]
content: |
  Must-have criteria: [Non-negotiable requirements]
  Should-have criteria: [Important but flexible]
  Nice-to-have criteria: [Bonus features]
  Success metrics: [How to measure success quantitatively]
evidence: [Requirements, stakeholder expectations]
assumptions: [Assumptions about what matters most]
confidence: 0.7-0.9
status: active
```

**T1.5 - Refine Framing**
```yaml
id: T1.5
stage: planning
dependencies: [T1.1, T1.2, T1.3, T1.4]
content: |
  Refined problem statement: [Clear, precise problem description]
  Scope: [What's in scope, what's out of scope]
  Key challenges: [Primary difficulties anticipated]
  Success definition: [Clear statement of desired outcome]
evidence: [Synthesis of T1.1-T1.4]
assumptions: [Final validated assumptions]
confidence: 0.8-0.95
status: active
```

### Key Questions

1. What exactly is the problem we're trying to solve?
2. Who are the stakeholders and what do they need?
3. What are the constraints we must work within?
4. What assumptions are we making?
5. How will we know when we've succeeded?

### Quality Criteria

- ✅ Problem is clearly articulated in specific terms
- ✅ All major stakeholders identified
- ✅ Constraints are explicit and validated
- ✅ Assumptions are surfaced and documented
- ✅ Success criteria are measurable and specific

---

## Phase 2: Approach Selection

### Goal
Choose the optimal reasoning framework for this problem.

### Thought Template

**T2.1 - Identify Applicable Frameworks**
```yaml
id: T2.1
stage: planning
dependencies: [T1.5]
content: |
  Framework candidates:
  1. [Framework name]: [Initial fit score 0-10]
  2. [Framework name]: [Initial fit score 0-10]
  3. [Framework name]: [Initial fit score 0-10]

  Rationale for each score: [Why each framework might or might not fit]
evidence: [Problem characteristics from Phase 1]
assumptions: [Assumptions about framework applicability]
confidence: 0.6-0.8
status: active
```

**T2.2 - Evaluate Framework Fit**
```yaml
id: T2.2
stage: planning
dependencies: [T2.1]
content: |
  Problem characteristics:
  - Novel/unknown: [Yes/No]
  - Complex system: [Yes/No]
  - Bug/issue: [Yes/No]
  - Choice between options: [Yes/No]
  - User-focused: [Yes/No]
  - Needs validation: [Yes/No]
  - Time-critical: [Yes/No]

  Framework evaluation:
  [Framework 1]: [Detailed fit analysis]
  [Framework 2]: [Detailed fit analysis]
evidence: [Framework selection guide, problem analysis]
assumptions: [Assumptions about problem type]
confidence: 0.7-0.85
status: active
```

**T2.3 - Select Framework**
```yaml
id: T2.3
stage: planning
dependencies: [T2.2]
content: |
  Primary framework: [Selected framework]
  Rationale: [Why this framework is optimal]
  Secondary frameworks: [Complementary approaches if needed]
  Framework process: [High-level process steps]
evidence: [Framework evaluation from T2.2]
assumptions: [Assumptions about framework effectiveness]
confidence: 0.75-0.9
status: active
```

**T2.4 - Design Strategy**
```yaml
id: T2.4
stage: planning
dependencies: [T2.3]
content: |
  Reasoning sequence: [Planned thought progression]
  Branching points: [Where to explore alternatives]
  Key milestones: [Checkpoints in reasoning process]
  Expected thought count: [Estimated number of thoughts]
  Validation points: [Where to check consistency]
evidence: [Framework process, problem complexity]
assumptions: [Assumptions about reasoning path]
confidence: 0.7-0.85
status: active
```

### Key Questions

1. What type of problem is this fundamentally?
2. Which frameworks are applicable to this problem type?
3. What are the strengths and limitations of each framework?
4. Which framework best matches our constraints and goals?
5. How will we apply the selected framework?

### Quality Criteria

- ✅ Multiple frameworks considered
- ✅ Framework selection is justified
- ✅ Framework process is understood
- ✅ Strategy includes branching and validation
- ✅ Approach is appropriate for problem complexity

---

## Phase 3: Deep Analysis

### Goal
Execute framework and explore solution space thoroughly.

### Thought Template

**Main Analysis Thought**
```yaml
id: T3.X
stage: analysis
dependencies: [T2.4, ...]
content: |
  [Framework-specific analysis]

  Key findings: [What was discovered]
  Evidence: [Supporting data, observations]
  Implications: [What this means for the solution]
evidence: [Concrete facts, data, observations]
assumptions: [Assumptions made in analysis]
confidence: 0.6-0.9 (varies by evidence strength)
status: active
tools_used: [Any tools employed]
```

**Branch Thought**
```yaml
id: T3.X.1
stage: analysis
dependencies: [T3.X]
content: |
  Alternative: [Name of alternative approach]

  Approach: [How this alternative works]
  Pros: [Advantages of this approach]
  Cons: [Disadvantages of this approach]
  Risk assessment: [Potential risks]
  Confidence: [How likely is success]
evidence: [Supporting evidence for this alternative]
assumptions: [Assumptions specific to this branch]
confidence: 0.5-0.8
status: active
```

**Revision Thought**
```yaml
id: T3.X.r
stage: revision
dependencies: [T3.X]
content: |
  Original thought: [Summary of T3.X]
  Reason for revision: [Why update is needed]
  New evidence: [What changed]
  Updated reasoning: [Corrected analysis]
  Impact: [How this affects downstream thoughts]
evidence: [New evidence that prompted revision]
assumptions: [Updated assumptions]
confidence: [New confidence level] (delta: +/- X.XX)
status: active
contradictions: [T3.X marked as superseded]
```

### Key Questions (Framework-Dependent)

**For First Principles**:
1. What are the fundamental truths here?
2. Which assumptions can we challenge?
3. How would we rebuild this from basics?

**For Root Cause Analysis**:
1. What are the symptoms vs. root causes?
2. What hypotheses can we generate?
3. How can we test each hypothesis?

**For Decision Analysis**:
1. What are our evaluation criteria?
2. What alternatives should we consider?
3. How does each alternative score?

### Quality Criteria

- ✅ Framework process followed systematically
- ✅ Multiple alternatives explored through branches
- ✅ Evidence supports each conclusion
- ✅ Assumptions are explicit and justified
- ✅ Revisions made when new evidence emerges
- ✅ Confidence levels reflect evidence strength

---

## Phase 4: Synthesis

### Goal
Integrate findings into coherent understanding.

### Thought Template

**T4.1 - Integrate Findings**
```yaml
id: T4.1
stage: synthesis
dependencies: [T3.1, T3.2, T3.3, ...]
content: |
  Branch comparison:
  - Branch T3.1.1: [Summary and key insights]
  - Branch T3.1.2: [Summary and key insights]
  - Branch T3.1.3: [Summary and key insights]

  Convergence: [Where branches agree]
  Divergence: [Where branches disagree]
  Complementarity: [How branches complement each other]
evidence: [Synthesis of branch findings]
assumptions: [Assumptions in integration]
confidence: 0.7-0.9
status: active
```

**T4.2 - Identify Key Insights**
```yaml
id: T4.2
stage: synthesis
dependencies: [T4.1]
content: |
  Primary insights:
  1. [Most important discovery]
  2. [Second most important discovery]
  3. [Third most important discovery]

  Supporting insights:
  - [Additional insight 1]
  - [Additional insight 2]

  Novel insights: [Unexpected discoveries]
evidence: [Analysis findings]
assumptions: [Assumptions about insight importance]
confidence: 0.75-0.95
status: active
```

**T4.3 - Draw Conclusions**
```yaml
id: T4.3
stage: synthesis
dependencies: [T4.1, T4.2]
content: |
  Primary conclusions:
  1. [Main conclusion with confidence]
  2. [Secondary conclusion with confidence]

  Implications: [What these conclusions mean]
  Limitations: [What remains uncertain]
  Alternative interpretations: [Other ways to read the evidence]
evidence: [Integrated findings]
assumptions: [Assumptions in conclusions]
confidence: 0.7-0.9
status: active
```

**T4.4 - Formulate Recommendations**
```yaml
id: T4.4
stage: synthesis
dependencies: [T4.3]
content: |
  Recommendations:
  1. [Action 1]: [Rationale and expected outcome]
  2. [Action 2]: [Rationale and expected outcome]
  3. [Action 3]: [Rationale and expected outcome]

  Priority order: [Why this sequence]
  Dependencies: [Which actions depend on others]
  Risk mitigation: [How to reduce risk]
evidence: [Conclusions from T4.3]
assumptions: [Assumptions about recommendations]
confidence: 0.75-0.9
status: active
```

### Key Questions

1. What patterns emerge across all branches?
2. What are the most important insights?
3. What can we conclude with confidence?
4. What remains uncertain?
5. What should we do based on these findings?

### Quality Criteria

- ✅ All branch findings integrated
- ✅ Insights are specific and actionable
- ✅ Conclusions are evidence-based
- ✅ Limitations are acknowledged
- ✅ Recommendations are prioritized and justified

---

## Phase 5: Validation

### Goal
Verify reasoning integrity and consistency.

### Thought Template

**T5.1 - Contradiction Detection**
```yaml
id: T5.1
stage: validation
dependencies: [All previous thoughts]
content: |
  Contradiction scan results:

  Semantic contradictions: [Any conflicting statements]
  Constraint violations: [Any violated constraints]
  Temporal inconsistencies: [Any cause-effect conflicts]

  Contradictions found: [List with thought IDs]
  Resolution needed: [Which thoughts need revision]
evidence: [Automated and manual contradiction checks]
assumptions: [Assumptions about contradictions]
confidence: 0.8-0.95
status: active
contradictions: [List of conflicting thought pairs]
```

**T5.2 - Assumption Verification**
```yaml
id: T5.2
stage: validation
dependencies: [All thoughts with assumptions]
content: |
  Critical assumptions review:
  1. [Assumption 1]: [Validated/Unvalidated] - [Evidence]
  2. [Assumption 2]: [Validated/Unvalidated] - [Evidence]
  3. [Assumption 3]: [Validated/Unvalidated] - [Evidence]

  Validated: [Count and confidence increase]
  Unvalidated: [Count and risk assessment]
  Invalid: [Count and revision needed]
evidence: [Verification evidence]
assumptions: [Meta-assumptions]
confidence: 0.7-0.9
status: active
```

**T5.3 - Confidence Assessment**
```yaml
id: T5.3
stage: validation
dependencies: [All previous thoughts]
content: |
  Confidence distribution:
  - High confidence (0.85-1.0): [Count] thoughts
  - Medium confidence (0.60-0.84): [Count] thoughts
  - Low confidence (0.0-0.59): [Count] thoughts

  Areas of high confidence: [What we're sure about]
  Areas of uncertainty: [What remains unclear]
  Confidence trend: [Increasing/Stable/Decreasing]
evidence: [Confidence levels from all thoughts]
assumptions: [Assumptions about confidence assessment]
confidence: 0.8-0.95
status: active
```

**T5.4 - Uncertainty Identification**
```yaml
id: T5.4
stage: validation
dependencies: [T5.1, T5.2, T5.3]
content: |
  Known unknowns: [What we know we don't know]
  Unknown unknowns: [Potential blind spots]

  Critical uncertainties: [Uncertainties that matter most]
  Manageable uncertainties: [Uncertainties we can accept]

  Further investigation needed: [What to explore next]
evidence: [Uncertainty analysis]
assumptions: [Assumptions about unknowns]
confidence: 0.6-0.8
status: active
```

### Key Questions

1. Are there any logical contradictions?
2. Have critical assumptions been validated?
3. Where is confidence high vs. low?
4. What important uncertainties remain?
5. Is the reasoning chain sound?

### Quality Criteria

- ✅ No unresolved contradictions
- ✅ Critical assumptions validated or acknowledged
- ✅ Confidence levels are realistic
- ✅ Uncertainties are identified and assessed
- ✅ Reasoning chain is logically consistent

---

## Phase 6: Finalization

### Goal
Generate actionable outputs and preserve work.

### Thought Template

**T6.1 - Executive Summary**
```yaml
id: T6.1
stage: validation
dependencies: [T4.4, T5.4]
content: |
  Problem: [One-sentence problem statement]

  Approach: [Framework and method used]

  Key findings:
  1. [Finding 1]
  2. [Finding 2]
  3. [Finding 3]

  Recommendations:
  1. [Top recommendation]
  2. [Second recommendation]
  3. [Third recommendation]

  Confidence: [Overall confidence level]
  Next steps: [Immediate actions]
evidence: [Synthesis from Phase 4 and 5]
assumptions: [Key assumptions to note]
confidence: 0.8-0.95
status: validated
```

**T6.2 - Action Plan**
```yaml
id: T6.2
stage: validation
dependencies: [T6.1]
content: |
  Immediate actions (Today/This week):
  1. [Action 1]: [Owner, Timeline, Success criteria]
  2. [Action 2]: [Owner, Timeline, Success criteria]

  Short-term actions (This month):
  3. [Action 3]: [Owner, Timeline, Success criteria]
  4. [Action 4]: [Owner, Timeline, Success criteria]

  Medium-term actions (This quarter):
  5. [Action 5]: [Owner, Timeline, Success criteria]

  Success metrics: [How to measure progress]
  Review cadence: [When to check in]
evidence: [Recommendations from T4.4]
assumptions: [Assumptions about execution]
confidence: 0.75-0.9
status: validated
```

**T6.3 - Document Decisions**
```yaml
id: T6.3
stage: validation
dependencies: [All phases]
content: |
  Key decisions made:

  Decision 1: [Decision statement]
  - Rationale: [Why this decision]
  - Alternatives considered: [What else was considered]
  - Trade-offs: [What was gained/lost]
  - Confidence: [X.XX]

  Decision 2: [Decision statement]
  - Rationale: [Why this decision]
  - Alternatives considered: [What else was considered]
  - Trade-offs: [What was gained/lost]
  - Confidence: [X.XX]

  Context for future: [Important context to preserve]
evidence: [Decision points throughout reasoning]
assumptions: [Key assumptions in decisions]
confidence: 0.8-0.95
status: validated
```

**T6.4 - Session Save**
```yaml
id: T6.4
stage: validation
dependencies: [T6.1, T6.2, T6.3]
content: |
  Session metadata:
  - Duration: [Time spent]
  - Depth: [shallow/deep/ultradeep]
  - Framework: [Framework used]
  - Thought count: [Total thoughts]
  - Branch count: [Total branches]
  - Revision count: [Total revisions]

  Save location: [File path]
  Format: [JSON + Markdown]

  Session quality: [Self-assessment]
evidence: [Complete session history]
assumptions: [N/A]
confidence: 1.0
status: validated
```

### Key Questions

1. What are the most important takeaways?
2. What actions should be taken?
3. What decisions were made and why?
4. How should we preserve this work?
5. What should future readers know?

### Quality Criteria

- ✅ Executive summary is clear and concise
- ✅ Action plan is specific and time-bound
- ✅ Decisions are documented with rationale
- ✅ Session is saved with proper metadata
- ✅ Outputs are ready for stakeholder consumption

---

## Cross-Phase Best Practices

### Thought Quality

**Content**:
- Be specific and concrete
- Provide evidence
- State assumptions explicitly
- Show reasoning, not just conclusions

**Dependencies**:
- List all thoughts this builds upon
- Ensure dependency chain is clear
- Validate dependencies exist

**Confidence**:
- Be realistic about certainty
- Adjust confidence based on evidence
- Track confidence changes

**Status**:
- Update status as work progresses
- Mark superseded thoughts
- Validate final thoughts

### Progressive Refinement

1. **Draft thoughts quickly** - Get ideas out
2. **Revise with evidence** - Add supporting data
3. **Validate consistency** - Check for contradictions
4. **Finalize with confidence** - Reach stable state

### Documentation

- Preserve reasoning path for auditability
- Make implicit thinking explicit
- Enable others to follow your logic
- Support future reference and learning
