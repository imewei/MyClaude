---
name: meta-cognitive-reflection
description: Analyze AI reasoning patterns, cognitive biases, and conversation quality through systematic meta-cognitive reflection. This skill should be used when analyzing AI decision-making processes, evaluating reasoning effectiveness, identifying cognitive biases, or assessing conversation patterns and problem-solving quality. Use for session reflections, reasoning audits, and improving AI interaction effectiveness.
---

# Meta-Cognitive Reflection

## Overview

Analyze and reflect on AI reasoning patterns, cognitive processes, and conversation quality to improve decision-making effectiveness and identify areas for improvement. This skill provides frameworks for systematic meta-cognitive analysis including reasoning pattern identification, bias detection, problem-solving evaluation, and communication effectiveness assessment.

## Core Capabilities

### 1. Reasoning Pattern Analysis

Identify and evaluate different types of reasoning used in AI responses:

**Reasoning Types:**
- **Deductive Reasoning**: Drawing specific conclusions from general principles
- **Inductive Reasoning**: Drawing general conclusions from specific observations
- **Abductive Reasoning**: Inferring best explanations for observations
- **Analogical Reasoning**: Drawing parallels from similar domains
- **Causal Reasoning**: Identifying cause-effect relationships

For detailed pattern taxonomy and examples, read `references/reasoning_patterns_taxonomy.md`.

**Analysis Process:**
1. Review conversation history or specific responses
2. Identify instances of each reasoning type
3. Evaluate effectiveness and appropriateness
4. Assess logical validity and evidence strength
5. Generate insights and recommendations

### 2. Cognitive Bias Detection

Identify and mitigate cognitive biases in AI reasoning:

**Common Biases:**
- **Availability Bias**: Over-relying on recent or readily available examples
- **Anchoring Bias**: Over-relying on first information encountered
- **Confirmation Bias**: Seeking evidence that supports initial hypotheses
- **Recency Bias**: Overweighting recent information
- **Selection Bias**: Non-representative sampling or example selection

For complete bias checklist and mitigation strategies, read `references/cognitive_biases_checklist.md`.

**Detection Process:**
1. Review decision-making patterns
2. Check for bias indicators
3. Assess severity and impact
4. Recommend mitigation strategies
5. Document for learning

### 3. Problem-Solving Evaluation

Assess problem-solving approach and effectiveness:

**Evaluation Stages:**
1. **Problem Understanding**
   - Clarity of problem definition
   - Requirement coverage
   - Assumption identification

2. **Strategy Formulation**
   - Approaches considered
   - Selection rationale
   - Anticipated challenges

3. **Implementation**
   - Execution quality
   - Error handling
   - Efficiency

4. **Validation**
   - Testing thoroughness
   - Edge case coverage
   - Verification methods

5. **Iteration**
   - Adaptation to feedback
   - Refinement effectiveness
   - Learning incorporation

### 4. Communication Effectiveness

Evaluate communication quality and clarity:

**Assessment Dimensions:**
- **Clarity**: Appropriate jargon level, explanation depth, structure
- **Accuracy**: Factual correctness, technical precision, citation quality
- **Pedagogy**: Example quality, progressive disclosure, concept scaffolding
- **Relevance**: On-topic focus, priority alignment, tangent handling
- **Engagement**: Interactivity, encouragement, collaboration

### 5. Session Reflection Report

Generate comprehensive reflection reports using the template:

```bash
# Use the report template
cp assets/session_reflection_template.md ./session_reflection_YYYYMMDD.md
```

**Report Sections:**
- Executive Summary: Overall assessment and key findings
- Reasoning Analysis: Pattern identification and effectiveness
- Cognitive Biases: Detected biases and mitigation
- Problem-Solving: Approach evaluation and insights
- Communication: Effectiveness assessment
- Meta-Reflection: Reflection on reflection quality
- Recommendations: Actionable improvements

## Workflow: Conducting Meta-Cognitive Reflection

### Step 1: Define Reflection Scope

Determine what to reflect on:
- Single conversation or session
- Specific decision or reasoning chain
- Problem-solving approach for a task
- Communication patterns over time

### Step 2: Gather Context

Collect relevant information:
- Conversation history
- Decision points
- Reasoning chains
- Outcomes and feedback

### Step 3: Analyze Reasoning Patterns

Use the reasoning patterns taxonomy (`references/reasoning_patterns_taxonomy.md`):
1. Identify each reasoning type used
2. Count instances and assess frequency
3. Evaluate appropriateness and effectiveness
4. Note strengths and weaknesses

### Step 4: Detect Cognitive Biases

Use the bias checklist (`references/cognitive_biases_checklist.md`):
1. Review decision-making process
2. Check for bias indicators
3. Assess severity and impact
4. Document mitigation strategies

### Step 5: Evaluate Problem-Solving

Assess each stage of problem-solving:
1. Problem understanding clarity
2. Strategy formulation quality
3. Implementation effectiveness
4. Validation thoroughness
5. Iteration and learning

### Step 6: Assess Communication

Evaluate communication across dimensions:
1. Clarity and structure
2. Technical accuracy
3. Pedagogical effectiveness
4. Relevance to goals
5. Engagement level

### Step 7: Generate Meta-Insights

Synthesize findings:
- Cross-cutting patterns
- Strengths to leverage
- Weaknesses to address
- Opportunities for improvement
- Strategic recommendations

### Step 8: Create Reflection Report

Use the template (`assets/session_reflection_template.md`):
1. Complete executive summary
2. Fill in detailed analysis sections
3. Document insights and recommendations
4. Include metrics and scores
5. Add action items with priorities

## Resources

### references/

**reasoning_patterns_taxonomy.md**
Comprehensive guide to reasoning pattern types with examples, effectiveness metrics, and usage guidelines. Read this to understand how to identify and evaluate different reasoning approaches.

**cognitive_biases_checklist.md**
Complete checklist of common cognitive biases with detection criteria, severity assessment, and mitigation strategies. Read this when analyzing decision-making for potential biases.

### assets/

**session_reflection_template.md**
Professional markdown template for session reflection reports. Copy and customize this template to create structured reflection documents with consistent formatting and comprehensive sections.

## Example Usage

**Example 1: Analyzing a problem-solving session**
```
User: "Reflect on how I approached debugging that memory leak"

Process:
1. Review conversation about memory leak debugging
2. Identify reasoning patterns used (abductive, causal)
3. Check for biases (availability, anchoring)
4. Evaluate problem-solving stages
5. Assess communication clarity
6. Generate insights and recommendations
7. Create reflection report
```

**Example 2: Evaluating reasoning quality**
```
User: "Analyze the reasoning I used for choosing the database architecture"

Process:
1. Review architecture decision conversation
2. Identify reasoning types (analogical, deductive, causal)
3. Evaluate evidence strength and logical validity
4. Check for confirmation bias in option evaluation
5. Assess alternative consideration
6. Provide meta-cognitive insights
```

**Example 3: Session-wide reflection**
```
User: "Reflect on today's coding session"

Process:
1. Review entire session conversation
2. Analyze all reasoning patterns used
3. Detect any recurring biases
4. Evaluate problem-solving effectiveness across tasks
5. Assess communication quality
6. Generate comprehensive session reflection report
7. Provide actionable recommendations
```

## Best Practices

1. **Be Objective**: Base assessments on evidence, not assumptions
2. **Quantify When Possible**: Use metrics and counts for patterns and biases
3. **Context Matters**: Consider task complexity when evaluating effectiveness
4. **Balance Strengths and Weaknesses**: Acknowledge both what worked well and what needs improvement
5. **Actionable Recommendations**: Provide specific, implementable improvements
6. **Meta-Reflect**: Periodically reflect on the reflection process itself
7. **Learn and Iterate**: Use insights to improve future reasoning and communication
