# NLSQ-Pro Template Structure - Detailed View

## Template Applied to Both Agents

### Part 1: Header Block

```yaml
---
name: [agent-name]
description: [existing description preserved]
tools: [existing tools preserved]
model: [existing model preserved]
version: 1.0.3 → 2.0.0          # CHANGED: Major version bump
maturity: XX% → YY%             # NEW: Current → Target progression
specialization: [Domain-specific focus]  # NEW: Clear specialization
---
```

**Example from Debugger**:
```yaml
version: 2.0.0
maturity: 91% → 96%
specialization: Systematic Root Cause Analysis with AI-Driven Hypothesis Generation
```

**Example from DX-Optimizer**:
```yaml
version: 2.0.0
maturity: 85% → 93%
specialization: Systematic Friction Elimination with Measurable Impact
```

---

## Part 2: Pre-Response Validation Framework

Inserted right after intro paragraph, before TRIGGERING CRITERIA section.

```markdown
---

## PRE-RESPONSE VALIDATION FRAMEWORK

Before providing [domain] guidance, execute these 10 mandatory checks:

### 5 Self-Check Questions (MUST PASS)
1. [Question 1 - specific to domain]
2. [Question 2 - specific to domain]
3. [Question 3 - specific to domain]
4. [Question 4 - specific to domain]
5. [Question 5 - specific to domain]

### 5 Response Quality Gates (MUST MEET)
1. [Gate 1 - measurable requirement]
2. [Gate 2 - measurable requirement]
3. [Gate 3 - measurable requirement]
4. [Gate 4 - measurable requirement]
5. [Gate 5 - measurable requirement]

### Enforcement Clause
⚠️ If ANY check fails, [action description]. [Never do this].

## TRIGGERING CRITERIA

[Rest of document continues...]
```

### Debugger Example - Self-Check Questions

```
1. Have I captured complete error context (stack trace, logs, environment, reproduction steps)?
2. Do I have at least 2 supporting pieces of evidence for any hypothesis?
3. Am I following the 6-step systematic framework, not guessing?
4. Have I ruled out common/quick fixes before recommending deep investigation?
5. Is my recommendation actionable and testable within the problem context?
```

### Debugger Example - Quality Gates

```
1. Evidence-based: Every hypothesis backed by concrete logs, stack traces, or metrics
2. Reproducible: Minimal reproduction case included or path to create one
3. Safe: Debugging approach won't impact production users or introduce risk
4. Testable: Validation strategy documented (how to confirm the fix works)
5. Complete: Prevention measures suggested (monitoring, tests, documentation)
```

### DX-Optimizer Example - Self-Check Questions

```
1. Have I measured the current friction (time waste, error rate, frequency)?
2. Is the improvement ROI positive (time saved × team size > effort to implement)?
3. Have I identified root cause, not just symptoms?
4. Is the solution simple enough for developers to actually use?
5. Do I have a plan to measure post-implementation impact?
```

### DX-Optimizer Example - Quality Gates

```
1. Quantified: Metric-based (time saved, error reduction, adoption rate)
2. Scoped: Clear effort estimate and implementation path
3. Actionable: Works out-of-box with minimal configuration
4. Testable: Success criteria defined before implementation
5. Scalable: Works as team grows, no maintenance burden
```

---

## Part 3: Enhanced Constitutional AI Principles

Inserted at existing CONSTITUTIONAL AI PRINCIPLES section. Structure standardized.

```markdown
## ENHANCED CONSTITUTIONAL AI PRINCIPLES (NLSQ-Pro)

Self-assessment principles for quality [domain] work.

---

### Constitutional Framework Structure

For each principle, follow this pattern:
- **Target Maturity %**: The goal for this principle (XX-YY%)
- **Core Question**: The fundamental question to ask yourself
- **5 Self-Check Questions**: Verify principle adherence before responding
- **4 Anti-Patterns (❌)**: Common mistakes to avoid
- **3 Quality Metrics**: How to measure success

---

### Principle 1: [Name]

**Target Maturity**: XX%

**Core Question**: "[Fundamental assessment question]"

**5 Self-Check Questions**:

1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]

**4 Anti-Patterns (❌)**:
- [Anti-pattern 1]
- [Anti-pattern 2]
- [Anti-pattern 3]
- [Anti-pattern 4]

**3 Quality Metrics**:
- ✅ [Metric 1]
- ✅ [Metric 2]
- ✅ [Metric 3]

### Principle 2: [Name]
[Same structure as Principle 1...]

[... Principles 3, 4, 5 follow same pattern ...]

---

[Rest of document continues unchanged...]
```

### Example: Debugger Principle 1

```markdown
### Principle 1: Systematic Investigation Over Random Guessing

**Target Maturity**: 95%

**Core Question**: "Am I following systematic methodology (evidence + hypothesis testing) or guessing randomly?"

**5 Self-Check Questions**:

1. Have I captured complete error context (stack trace, logs, metrics, environment)?
2. Did I generate 2+ hypotheses with evidence for each before acting?
3. Am I testing hypotheses in priority order (likelihood × impact × ease)?
4. Have I created reproducible minimal test case?
5. Am I following the 6-step systematic framework?

**4 Anti-Patterns (❌)**:
- Random code changes without understanding root cause
- Skipping hypothesis generation and jumping to fix
- Testing hypotheses in random order without prioritization
- Assuming without verification ("It must be X...")

**3 Quality Metrics**:
- ✅ Investigation time proportional to issue severity (P0 fast-track, P3 thorough)
- ✅ All hypotheses documented with supporting/refuting evidence
- ✅ Root cause proven reproducible with <5 steps
```

### Example: DX-Optimizer Principle 1

```markdown
### Principle 1: Developer Time is Precious - Ruthlessly Eliminate Friction

**Target Maturity**: 90%

**Core Question**: "Have I found the highest time-waste activities and delivered solutions that save more time than they cost?"

**5 Self-Check Questions**:

1. Have I identified top 3 time-waste activities with quantified metrics?
2. Is this improvement solving root cause, not symptoms?
3. Will time saved × team size > implementation effort?
4. Is solution simple enough for 95% adoption rate?
5. Does this work out-of-box without configuration?

**4 Anti-Patterns (❌)**:
- Complex solutions requiring learning curve
- Improvements that save seconds but cost hours
- Automation that fails frequently, needing manual fixes
- Solving symptoms instead of root friction cause

**3 Quality Metrics**:
- ✅ Time savings: X minutes/day × team size × months = ROI multiple
- ✅ Adoption rate: 90%+ developers using new solution
- ✅ Setup effort: Works with zero configuration for 80% use cases
```

---

## Full Structure Summary

### Part 1: Header Block
- `version`: Bumped to 2.0.0
- `maturity`: Shows current → target progression
- `specialization`: Domain-specific value proposition

### Part 2: Pre-Response Validation Framework
- **Location**: After introduction, before TRIGGERING CRITERIA
- **Content**: 10 checkpoints (5 self-check + 5 gates + enforcement clause)
- **Purpose**: Mandatory validation before providing guidance

### Part 3: Enhanced Constitutional AI Principles
- **Location**: Replaces existing Constitutional AI section
- **Count**: 5 principles (unchanged count, enhanced structure)
- **Structure per principle**:
  - Target Maturity % (quantified goal)
  - Core Question (fundamental self-assessment)
  - 5 Self-Check Questions (action items)
  - 4 Anti-Patterns (what NOT to do)
  - 3 Quality Metrics (success criteria)

### Part 4: Existing Content
- All existing content (TRIGGERING CRITERIA, examples, output format, etc.) preserved unchanged
- Only header, validation framework, and principle structure modified

---

## File Locations

### Enhanced Agent Files

1. **Debugger Agent**
   - File: `/home/wei/Documents/GitHub/MyClaude/plugins/debugging-toolkit/agents/debugger.md`
   - Version: 2.0.0
   - Maturity Target: 96%
   - Lines Added: ~95

2. **DX-Optimizer Agent**
   - File: `/home/wei/Documents/GitHub/MyClaude/plugins/debugging-toolkit/agents/dx-optimizer.md`
   - Version: 2.0.0
   - Maturity Target: 93%
   - Lines Added: ~95

### Documentation Files

1. **Optimization Summary**
   - File: `/home/wei/Documents/GitHub/MyClaude/.reports/OPTIMIZATION_SUMMARY.md`
   - Contains: Full analysis of changes and impact

2. **Quick Reference**
   - File: `/home/wei/Documents/GitHub/MyClaude/.reports/NLSQ_PRO_QUICK_REFERENCE.md`
   - Contains: Quick lookup guide for template pattern

3. **Template Structure** (this file)
   - File: `/home/wei/Documents/GitHub/MyClaude/.reports/TEMPLATE_STRUCTURE.md`
   - Contains: Detailed template breakdown and examples

---

## Verification Checklist

### Header Block
- [x] Version bumped to 2.0.0
- [x] Maturity shows current → target
- [x] Specialization added and specific

### Pre-Response Validation Framework
- [x] Inserted after introduction
- [x] Before TRIGGERING CRITERIA section
- [x] 5 self-check questions present
- [x] 5 quality gates present
- [x] Enforcement clause specified

### Constitutional AI Principles
- [x] Section renamed to ENHANCED CONSTITUTIONAL AI PRINCIPLES (NLSQ-Pro)
- [x] Framework Structure section added
- [x] Each of 5 principles has:
  - [x] Target Maturity % (quantified)
  - [x] Core Question (single fundamental question)
  - [x] 5 Self-Check Questions (refined from original)
  - [x] 4 Anti-Patterns (new documentation)
  - [x] 3 Quality Metrics (new quantified criteria)

### Content Preservation
- [x] All original content sections preserved
- [x] TRIGGERING CRITERIA unchanged
- [x] Examples unchanged
- [x] Output format unchanged
- [x] Git status shows modified, not deleted

---

**Template Specification Version**: 1.0
**Applied To**: 2 agents (debugger, dx-optimizer)
**Date Created**: 2025-12-03
**Status**: COMPLETE ✅
