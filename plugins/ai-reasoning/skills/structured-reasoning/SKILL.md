---
name: structured-reasoning
description: Advanced structured reasoning methodology with step-by-step thought processing, branching exploration, and multi-framework analysis. This skill should be used when executing complex problem-solving tasks requiring systematic thinking, exploring multiple solution paths, tracking reasoning chains with dependencies, or applying cognitive frameworks like First Principles, Systems Thinking, Root Cause Analysis, or Decision Analysis. Use for debugging complex issues, architectural decisions, strategic planning, or any problem requiring deep, auditable reasoning.
---

# Structured Reasoning

## Overview

Apply advanced structured reasoning methodologies to complex problems through systematic thought processing, branching exploration, and multi-framework analysis. This skill provides comprehensive frameworks, templates, and workflows for executing ultra-think style reasoning with full auditability and course correction capabilities.

## Core Methodology

### Thought Structure

Every reasoning step follows a structured format with hierarchical tracking:

**Thought ID Format**: `T[phase].[step].[branch]`
- Phase: 1-6 (Understanding, Approach, Analysis, Synthesis, Validation, Finalization)
- Step: Sequential number within phase
- Branch: Optional sub-exploration (1, 2, 3, etc.)

**Example**: T3.2.1 = Phase 3, Step 2, Branch 1

**Thought Components**:
```yaml
thought:
  id: "T3.2.1"
  stage: "analysis"  # planning|analysis|synthesis|revision|validation
  dependencies: ["T3.1", "T3.2"]
  content: "[Detailed reasoning]"
  evidence: "[Supporting facts]"
  assumptions: "[Stated assumptions]"
  confidence: 0.85  # 0-1 scale
  status: "active"  # active|revised|superseded|validated
  contradictions: []
  next_steps: "[Implications]"
```

---

## Six-Phase Reasoning Process

### Phase 1: Problem Understanding (T1.x)

**Goal**: Fully comprehend the problem space

**Key Thoughts**:
1. **T1.1 - Capture Problem**: Parse and frame the core question
2. **T1.2 - Identify Constraints**: Technical, business, resource, timeline
3. **T1.3 - Surface Assumptions**: Explicit and implicit assumptions
4. **T1.4 - Define Success**: Must-have, should-have, nice-to-have criteria
5. **T1.5 - Refine Framing**: Precise problem statement with scope

**Template**: Use `references/phase_templates.md` for detailed structure

---

### Phase 2: Approach Selection (T2.x)

**Goal**: Choose optimal reasoning framework

**Framework Options**:
- **First Principles**: Break down to fundamentals, rebuild
- **Systems Thinking**: Analyze interconnections and feedback loops
- **Root Cause Analysis**: 5 Whys, Fishbone, systematic investigation
- **Decision Analysis**: Weighted criteria, option scoring
- **Design Thinking**: User-centered, iterative
- **Scientific Method**: Hypothesis-driven
- **OODA Loop**: Rapid adaptive decision-making

**Key Thoughts**:
1. **T2.1 - Identify Applicable Frameworks**: Score each for fit
2. **T2.2 - Evaluate Framework**: Match to problem characteristics
3. **T2.3 - Select Framework**: Choose primary approach
4. **T2.4 - Design Strategy**: Plan thought sequence and branches

**Reference**: `references/framework_guide.md` for detailed framework descriptions

---

### Phase 3: Deep Analysis (T3.x)

**Goal**: Execute framework and explore solution space

**Branching Strategy**:
```
T3.1 [Analysis]: Main investigation
  ├─ T3.1.1 [Branch]: Alternative A
  ├─ T3.1.2 [Branch]: Alternative B
  └─ T3.1.3 [Branch]: Alternative C
T3.2 [Synthesis]: Compare branches
T3.3 [Revision]: Update based on findings
```

**Key Activities**:
- Execute framework steps systematically
- Generate multiple solution branches
- Create revisions when new evidence emerges
- Apply contradiction detection
- Gather and validate evidence

**Pattern**: Follow framework-specific patterns from `references/framework_guide.md`

---

### Phase 4: Synthesis (T4.x)

**Goal**: Integrate findings into coherent understanding

**Key Thoughts**:
1. **T4.1 - Integrate Findings**: Combine insights from branches
2. **T4.2 - Identify Key Insights**: Extract most important discoveries
3. **T4.3 - Draw Conclusions**: Primary and secondary conclusions
4. **T4.4 - Formulate Recommendations**: Actionable next steps

**Focus**: Pattern recognition, relationship mapping, insight extraction

---

### Phase 5: Validation (T5.x)

**Goal**: Verify reasoning integrity and consistency

**Validation Checks**:
1. **T5.1 - Contradiction Detection**: Scan for logical conflicts
2. **T5.2 - Assumption Verification**: Validate critical assumptions
3. **T5.3 - Confidence Assessment**: Map high/medium/low confidence areas
4. **T5.4 - Uncertainty Identification**: What remains unknown

**Reference**: `references/validation_checklist.md` for comprehensive checks

---

### Phase 6: Finalization (T6.x)

**Goal**: Generate actionable outputs

**Key Outputs**:
1. **T6.1 - Executive Summary**: 1-page high-level overview
2. **T6.2 - Action Plan**: Immediate, short-term, medium-term actions
3. **T6.3 - Document Decisions**: Key decisions with rationale
4. **T6.4 - Session Save**: Archive for future reference

**Template**: Use `assets/ultra_think_report_template.md`

---

## Branching and Revision

### When to Branch

**Branch to explore**:
- Alternative solution approaches (T3.1.1, T3.1.2, T3.1.3)
- Hypothesis testing (T3.2.v for validation)
- Different framework applications
- Edge cases or special scenarios

**Branch Naming**:
- `.1, .2, .3` for alternatives
- `.v` for validation branches
- `.r` for refinement branches
- `.c` for correction branches

### When to Revise

**Create revision when**:
- New evidence contradicts previous thought
- Assumption found to be invalid
- Logical inconsistency detected
- Better approach identified

**Revision Format**:
```
T3.5.1 [Revision of T3.5]
Original: [Previous reasoning]
Reason for revision: [Why update needed]
Updated reasoning: [Corrected analysis]
Confidence delta: +0.10 (improved from 0.75 to 0.85)
```

---

## Framework Application Guides

### Framework 1: First Principles

**When to use**: Novel problems, paradigm shifts, deep understanding needed

**Process**:
1. List all assumptions about the problem
2. Challenge each: "Is this necessarily true?"
3. Identify fundamental truths (physics, math, logic)
4. Reconstruct solution from fundamentals
5. Validate reconstruction

**Example Structure**:
```
T3.1: List assumptions
T3.2: Challenge assumption 1
T3.3: Challenge assumption 2
T3.4: Identify fundamental constraints
T3.5: Reconstruct from basics
T3.6: Validate new approach
```

**Full guide**: `references/framework_guide.md#first-principles`

---

### Framework 2: Root Cause Analysis

**When to use**: Debugging, incident response, quality issues

**Process**:
1. Define problem precisely (symptoms, scope, timeline)
2. Gather evidence (logs, metrics, observations)
3. Generate hypotheses (5 Whys, Fishbone)
4. Test hypotheses systematically
5. Validate root cause
6. Propose solutions

**Example Structure**:
```
T3.1: Define problem precisely
T3.2: Gather evidence
T3.3: Generate hypotheses
  T3.3.1: Test hypothesis 1
  T3.3.2: Test hypothesis 2
  T3.3.3: Test hypothesis 3
T3.4: Validate root cause
T3.5: Propose solutions
```

**Techniques**: 5 Whys, Fishbone diagram, Fault Tree Analysis

---

### Framework 3: Decision Analysis

**When to use**: Technology choices, architectural decisions, strategic planning

**Process**:
1. Define decision criteria with weights
2. Generate 3-5 alternatives
3. Score each against criteria
4. Quantify uncertainties and risks
5. Analyze tradeoffs
6. Make recommendation with confidence level

**Example Structure**:
```
T3.1: Define criteria and weights
T3.2: Generate alternatives
T3.3: Score alternative 1
T3.4: Score alternative 2
T3.5: Score alternative 3
T3.6: Compare and analyze tradeoffs
T3.7: Make recommendation
```

**Tool**: Decision matrix with weighted scoring

---

## Contradiction Detection

### Automated Checks

**Three levels of detection**:

**Level 1: Semantic Contradiction**
- Compare thought content for conflicting statements
- Example: "Database is read-heavy" vs "Needs write optimization"

**Level 2: Constraint Violation**
- Verify all constraints remain satisfied
- Example: Budget $50K vs Recommendation costs $75K

**Level 3: Temporal Inconsistency**
- Check cause-effect consistency
- Example: "Reduces latency" vs "Increases latency"

### Resolution Process

When contradiction detected:
1. Flag thoughts in conflict with IDs
2. Analyze root cause (assumption, data, logic error)
3. Create revision branch from earliest affected thought
4. Update reasoning chain
5. Validate downstream thoughts
6. Update confidence levels

---

## Confidence Tracking

### Confidence Levels

**High (0.85-1.0)**:
- Strong evidence supporting conclusion
- Multiple validation points
- Low uncertainty
- High agreement across analyses

**Medium (0.60-0.84)**:
- Moderate evidence
- Some assumptions unvalidated
- Moderate uncertainty
- Reasonable confidence

**Low (0.0-0.59)**:
- Weak or conflicting evidence
- Critical assumptions unvalidated
- High uncertainty
- Requires further investigation

### Confidence Updates

Track confidence changes:
- Initial assessment: 0.70
- After validation: 0.85 (+0.15)
- After contradiction: 0.55 (-0.15)
- After revision: 0.80 (+0.25)

---

## Depth Modes

### Shallow (5-15 thoughts)
**Duration**: 5-10 minutes
**Branching**: 1-2 branches
**Output**: Quick analysis, top recommendations

**Use for**: Time-constrained, straightforward problems, initial exploration

---

### Deep (20-40 thoughts)
**Duration**: 30-60 minutes
**Branching**: 3-5 branches
**Output**: Comprehensive analysis, detailed recommendations, implementation plan

**Use for**: Important decisions, complex problems, stakeholder buy-in needed

---

### Ultra-Deep (50-100+ thoughts)
**Duration**: 2-4 hours
**Branching**: 10+ branches
**Output**: Exhaustive analysis, multiple scenarios, research agenda, executive + technical reports

**Use for**: Strategic decisions, novel problems, high-stakes situations

---

## Workflow: Executing Structured Reasoning

### Step 1: Initialize Session

Parse problem statement and set parameters:
- Extract core question
- Detect depth mode (default: deep)
- Select or auto-detect framework
- Generate session ID
- Set thought budget estimate

### Step 2: Execute Phase 1 (Understanding)

Use phase template from `references/phase_templates.md`:
- T1.1: Capture problem
- T1.2: Identify constraints
- T1.3: Surface assumptions
- T1.4: Define success criteria
- T1.5: Refine framing

Document each thought with full structure (dependencies, evidence, confidence, etc.)

### Step 3: Execute Phase 2 (Approach)

Select optimal framework:
- Review framework options in `references/framework_guide.md`
- Score applicability of each
- Select primary framework
- Design reasoning strategy

### Step 4: Execute Phase 3 (Analysis)

Apply selected framework:
- Follow framework-specific process
- Generate multiple branches for alternatives
- Create revisions as needed
- Gather and validate evidence
- Check for contradictions

### Step 5: Execute Phase 4 (Synthesis)

Integrate findings:
- Combine insights from branches
- Identify key discoveries
- Draw conclusions
- Formulate recommendations

### Step 6: Execute Phase 5 (Validation)

Verify integrity using `references/validation_checklist.md`:
- Check for contradictions
- Verify assumptions
- Assess confidence levels
- Identify uncertainties

### Step 7: Execute Phase 6 (Finalization)

Generate outputs using `assets/ultra_think_report_template.md`:
- Create executive summary
- Build action plan
- Document key decisions
- Save session

---

## Resources

### references/

**framework_guide.md**
Comprehensive guide to all seven cognitive frameworks with detailed processes, examples, and application patterns. Read when selecting or applying frameworks.

**phase_templates.md**
Detailed templates for each of the six reasoning phases with thought structures, key questions, and quality criteria.

**validation_checklist.md**
Complete validation checklist covering contradiction detection, assumption verification, confidence assessment, and quality checks.

### assets/

**ultra_think_report_template.md**
Professional markdown template for ultra-think analysis reports. Includes executive summary, detailed analysis, thought progression, recommendations, and meta-analysis sections.

---

## Best Practices

### 1. Explicit Dependencies
Always list which thoughts each new thought builds upon.

### 2. Evidence-Based
Support reasoning with concrete facts, data, or observations.

### 3. Assumption Tracking
Make all assumptions explicit and validate critical ones.

### 4. Confidence Honesty
Be realistic about confidence levels. Low confidence is acceptable if uncertainty is high.

### 5. Branch Early
Explore alternatives before committing to a single path.

### 6. Revise Freely
Course-correct when new information emerges. Mark original as superseded.

### 7. Validate Thoroughly
Apply full validation checklist before finalizing.

### 8. Document Journey
Preserve reasoning path for auditability and learning.

### 9. Synthesize Regularly
Integrate findings every 10-15 thoughts to maintain coherence.

### 10. Quality Over Speed
Depth and correctness matter more than rapid completion.

---

## Example Usage

**Example 1: Debugging production issue**
```
Problem: "Memory leak causing OOM after 6 hours"
Framework: Root Cause Analysis
Depth: Deep

Process:
1. Phase 1: Understand problem (symptoms, timeline, scope)
2. Phase 2: Select root cause framework
3. Phase 3: Generate hypotheses, test systematically
   - Branch 1: Background job hypothesis
   - Branch 2: Cache growth hypothesis
   - Branch 3: Connection leak hypothesis
4. Phase 4: Identify root cause, validate
5. Phase 5: Check logical consistency
6. Phase 6: Recommend fix with confidence

Output: Root cause identified with 95% confidence, immediate fix proposed
```

**Example 2: Architectural decision**
```
Problem: "Should we migrate monolith to microservices?"
Framework: Decision Analysis
Depth: Ultra-Deep

Process:
1. Phase 1: Understand constraints, stakeholders, goals
2. Phase 2: Select decision analysis framework
3. Phase 3: Define criteria, generate alternatives, score
   - Branch 1: Keep monolith, optimize
   - Branch 2: Microservices migration
   - Branch 3: Modular monolith hybrid
   - Branch 4: Serverless approach
4. Phase 4: Integrate findings, analyze tradeoffs
5. Phase 5: Validate assumptions, check biases
6. Phase 6: Recommend approach with implementation plan

Output: Comprehensive analysis with weighted recommendation
```

**Example 3: Novel problem**
```
Problem: "Design authentication for multi-tenant SaaS"
Framework: First Principles
Depth: Deep

Process:
1. Phase 1: Understand security requirements, user types
2. Phase 2: Select first principles framework
3. Phase 3: Break down to fundamentals
   - What is authentication fundamentally?
   - What are immutable security constraints?
   - Reconstruct from basics
4. Phase 4: Synthesize modern approach
5. Phase 5: Validate against requirements
6. Phase 6: Document architecture with rationale

Output: Novel authentication design with strong foundation
```

---

## Integration with Ultra-Think Command

This skill directly supports the `/ultra-think` command by providing:

1. **Structured Thought Format**: Template for all reasoning steps
2. **Framework Guides**: Detailed processes for each cognitive framework
3. **Phase Templates**: Structure for all six reasoning phases
4. **Validation Checklists**: Comprehensive quality checks
5. **Report Templates**: Professional output formatting

Use this skill whenever executing ultra-think reasoning to ensure systematic, auditable, high-quality analysis.
