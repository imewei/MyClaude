---
description: Advanced reflection engine for AI reasoning, session analysis, and research
  optimization
triggers:
- /reflection
- workflow for reflection
version: 1.0.7
allowed-tools: Bash(find:*), Bash(grep:*), Bash(git:*), Read, Grep, Task
argument-hint: '[session|code|research|workflow] [--mode=quick-check|standard] [--depth=shallow|deep]'
color: purple
execution-modes:
  quick-check:
    description: Fast health assessment
    time: 2-5 minutes
  standard:
    description: Comprehensive reflection
    time: 15-45 minutes
agents:
  primary:
  - research-intelligence
  conditional:
  - agent: systems-architect
    trigger: pattern "architecture|design|system" OR argument "code"
  - agent: code-quality
    trigger: pattern "quality|test|lint" OR argument "workflow"
  orchestrated: true
---


# Advanced Reflection Engine

Deep reflection on AI reasoning, session effectiveness, research quality, and development practices.

## Context

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| `--mode=quick-check` | 2-5 min | Health scores + top 3 observations |
| standard (default) | 15-45 min | Detailed report + recommendations |

---

## Quick Check Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reflection Quick Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Session Activity:
- Commits: X (today), Files modified: Y
- Lines changed: +A, -B

Code Quality:
- Tech debt: N TODOs, M FIXMEs
- Test coverage: X% (estimated)

Top 3 Observations:
1. [Observation with emoji indicator]
2. [Observation]
3. [Observation]

Recommendations:
- [Action 1]
- [Action 2]

Next: /reflection session --depth=deep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Reflection Dimensions

| Dimension | Aspects |
|-----------|---------|
| **Meta-Cognitive** | Reasoning patterns, decision-making, bias detection, confidence calibration |
| **Technical** | Code quality, architecture, performance, technical debt, testing |
| **Research** | Methodology, experimental design, reproducibility, publication readiness |
| **Collaborative** | Communication, documentation, knowledge transfer |
| **Strategic** | Goal alignment, priority assessment, risk identification |

---

## Reflection Types

### Session Reflection

**Scoring (each /10):**
- Reasoning Patterns (40%): Logic coherence, assumption tracking, evidence usage
- Problem-Solving (30%): Approach selection, solution creativity, trade-off analysis
- Communication (20%): Clarity, technical depth, stakeholder adaptation
- Effectiveness (10%): Goal achievement, efficiency

**Grades:** 9-10 Exceptional, 7-8 Strong, 5-6 Adequate, 3-4 Weak, 1-2 Poor

---

### Research Reflection

**Assessment Areas (each /10):**

| Area | Focus |
|------|-------|
| Methodology | Hypothesis clarity, method selection, controls |
| Reproducibility | Code availability, environment specs, data access |
| Experimental Design | Statistical power, sample size, ablations |
| Data Quality | Accuracy, protocol consistency, bias handling |
| Analysis Rigor | Statistical tests, effect sizes, confidence intervals |
| Publication Readiness | Manuscript, figures, limitations |

**Thresholds:**
- <5.0: Major revisions required
- 5.0-6.5: Significant improvements needed
- 6.5-8.0: Minor revisions for publication
- 8.0-9.0: Strong candidate
- >9.0: High-impact potential

---

### Code Reflection

**Metrics:**

| Metric | Target |
|--------|--------|
| Test Coverage | ≥80% |
| Code Complexity | <10 avg cyclomatic |
| Tech Debt | Minimize TODOs/FIXMEs |
| Documentation | ≥75% coverage |

**Scoring:**
- 8-10: Excellent, sustainable
- 6-7: Good, minor improvements
- 4-5: Adequate, concerning patterns
- 2-3: Poor, debt accumulating
- 0-1: Critical, immediate intervention

---

## Report Templates

### Session Report

```markdown
## Executive Summary
- Overall Effectiveness: X/10
- Key Strengths: [bullets]
- Primary Improvements: [bullets]

## Scores
- Reasoning Quality: X/10
- Problem-Solving: X/10
- Communication: X/10

## Actionable Recommendations
1. [High priority]
2. [Medium priority]
```

### Research Report

```markdown
## Executive Summary
- Overall Score: X/10
- Publication Readiness: Tier X
- Timeline: N weeks

## Critical Findings
- ✅ Strengths: [top 3]
- 🔴 Critical: [must-fix]
- ⚠️ Improvements: [should-fix]

## Priority Actions
[Ordered by urgency]
```

### Code Report

```markdown
## Health Score: X/10

## Metrics
- Test Coverage: X%
- Complexity: X avg
- Tech Debt: N items

## Trends
[Improving/Declining indicators]

## Priority Actions
1. 🔴 [Critical]
2. ⚠️ [Important]
```

---

## Best Practices

**When to reflect:**
- Session: After major features, before PRs, weekly
- Research: Before submission, after experiments, monthly
- Code: Before merging, after refactoring, quarterly

**Maximize value:**
1. Be specific - vague reflections → vague insights
2. Track over time - identify trends
3. Act on insights - reflection without action is waste
4. Share learnings - compound knowledge

**Avoid:**
- Confirmation bias → seek contradictory evidence
- Recency bias → review entire timeline
- Perfection paralysis → focus on meaningful improvements
