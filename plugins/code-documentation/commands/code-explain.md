---
version: "1.0.6"
category: code-documentation
command: /code-explain
description: Detailed code explanation with visual aids and domain-specific expertise
allowed-tools: Bash(find:*), Bash(grep:*), Bash(git:*)
argument-hint: <code-path-or-snippet>
color: cyan
execution_modes:
  quick: "5-10 minutes"
  standard: "15-25 minutes"
  comprehensive: "30-45 minutes"
agents:
  primary:
    - research-intelligence
  conditional:
    - agent: systems-architect
      trigger: complexity > 10 OR pattern "architecture|design.*pattern"
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|scientific"
    - agent: neural-architecture-engineer
      trigger: pattern "torch|pytorch|tensorflow|deep.*learning"
    - agent: jax-pro
      trigger: pattern "jax|flax|@jit|@vmap|grad\\("
    - agent: simulation-expert
      trigger: pattern "lammps|gromacs|molecular.*dynamics"
  orchestrated: false
---

# Code Explanation and Analysis

Transform complex code into clear explanations with progressive disclosure and visual aids.

## Requirements

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Phases |
|------|----------|--------|
| Quick | 5-10 min | Overview + Key Concepts |
| Standard (default) | 15-25 min | + Diagrams + Examples + Pitfalls |
| Comprehensive | 30-45 min | + Learning Resources + Practice |

---

## External Documentation

| Topic | Reference | Lines |
|-------|-----------|-------|
| Code Analysis | [code-analysis-framework.md](../docs/code-documentation/code-analysis-framework.md) | ~350 |
| Visualizations | [visualization-techniques.md](../docs/code-documentation/visualization-techniques.md) | ~400 |
| Learning Resources | [learning-resources.md](../docs/code-documentation/learning-resources.md) | ~450 |
| Scientific Computing | [scientific-code-explanation.md](../docs/code-documentation/scientific-code-explanation.md) | ~400 |

---

## Phase 1: Code Comprehension Analysis

| Analysis | Output |
|----------|--------|
| Structure parsing | Classes, functions, imports |
| Complexity metrics | Cyclomatic complexity, nesting depth |
| Concept detection | Async, decorators, generators |
| Pattern detection | Singleton, Observer, Factory |
| Difficulty assessment | Beginner/intermediate/advanced/expert |

---

## Phase 2: Visual Explanation (Standard+)

| Diagram Type | Use Case |
|--------------|----------|
| Flowcharts | Control flow (if/else, loops) |
| Class diagrams | OOP structure, relationships |
| Sequence diagrams | Object interactions |
| Algorithm visualization | Step-by-step execution |
| Architecture diagrams | System components |

Use Mermaid for inline diagrams.

---

## Phase 3: Step-by-Step Explanation

### Progressive Disclosure Structure

1. **High-level overview** - What does this code do?
2. **Function breakdown** - How does each part work?
3. **Concept explanations** - Deep dive into patterns used
4. **Design pattern analysis** - Why these patterns?
5. **Performance considerations** - Efficiency notes

---

## Phase 4: Interactive Examples (Standard+)

| Example Type | Purpose |
|--------------|---------|
| Basic usage | Simple cases |
| Edge cases | Error scenarios |
| Try-it-yourself | Hands-on practice |
| Comparison | Alternative approaches |

---

## Phase 5: Pitfalls & Best Practices

### Common Issues

| Category | Examples |
|----------|----------|
| Error handling | Bare except clauses |
| State management | Global variables, mutable defaults |
| Security | eval(), SQL injection |
| Performance | Memory inefficiencies |

---

## Phase 6: Scientific Computing Support

| Domain | Explanation Focus |
|--------|-------------------|
| NumPy/SciPy | Broadcasting, vectorization, memory layout |
| JAX | @jit, grad, vmap, pmap transformations |
| Pandas | Method chaining, GroupBy, memory efficiency |
| Julia | Type stability, multiple dispatch |
| MD Simulations | Integrators, force calculations, neighbor lists |
| ML Training | Forward/backward pass, optimizer patterns |

---

## Phase 7: Learning Resources (Comprehensive)

| Component | Content |
|-----------|---------|
| Knowledge gaps | From complexity analysis |
| Topic recommendations | Deeper understanding |
| Curated resources | Tutorials, books, docs |
| Practice projects | By skill level |
| Learning plan | Week-by-week |

---

## Output Format

```markdown
# Code Explanation: [Name]

## Complexity Analysis
- Difficulty Level: [level]
- Key Concepts: [list]
- Design Patterns: [patterns]

## What This Code Does
[Clear explanation]

## Visual Representation
[Mermaid diagrams]

## Step-by-Step Breakdown
[Detailed walkthrough]

## Common Pitfalls
[Issues to avoid]

## [If scientific]
## Scientific Computing Context
[Domain-specific explanations]

## [Comprehensive only]
## Your Learning Path
[Personalized resources]
```

---

## Success Criteria

- [ ] Complexity accurately assessed
- [ ] All concepts identified and explained
- [ ] Diagrams generated where helpful
- [ ] Step-by-step breakdown provided
- [ ] Pitfalls highlighted
- [ ] Examples runnable and clear
- [ ] Matches user skill level
- [ ] Scientific code includes domain context
