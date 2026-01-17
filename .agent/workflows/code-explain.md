---
description: Detailed code explanation with visual aids and domain expertise
triggers:
- /code-explain
- detailed code explanation with
allowed-tools: [Read, Glob, Grep, Task]
version: 1.0.0
---



## User Input
Input arguments pattern: `<code-path-or-snippet>`
The agent should parse these arguments from the user's request.

# Code Explanation

$ARGUMENTS

## Modes

| Mode | Time | Phases |
|------|------|--------|
| Quick | 5-10min | Overview + Key Concepts |
| Standard | 15-25min | + Diagrams + Examples + Pitfalls |
| Comprehensive | 30-45min | + Learning Resources + Practice |

## External Docs

- `code-analysis-framework.md` (~350 lines)
- `visualization-techniques.md` (~400 lines)
- `learning-resources.md` (~450 lines)
- `scientific-code-explanation.md` (~400 lines)

## Process

1. **Comprehension Analysis**:
   | Analysis | Output |
   |----------|--------|
   | Structure | Classes, functions, imports |
   | Complexity | Cyclomatic complexity, nesting |
   | Concepts | Async, decorators, generators |
   | Patterns | Singleton, Observer, Factory |
   | Difficulty | Beginner/intermediate/advanced/expert |

2. **Visual Explanation** (Standard+):
   | Diagram | Use |
   |---------|-----|
   | Flowcharts | Control flow (if/else, loops) |
   | Class diagrams | OOP structure, relationships |
   | Sequence | Object interactions |
   | Algorithm viz | Step-by-step execution |
   | Architecture | System components |

   Use Mermaid for inline diagrams

3. **Progressive Disclosure**:
   - High-level: What does this code do?
   - Function breakdown: How does each part work?
   - Concepts: Deep dive into patterns
   - Design patterns: Why these patterns?
   - Performance: Efficiency notes

4. **Interactive Examples** (Standard+):
   - Basic usage: Simple cases
   - Edge cases: Error scenarios
   - Try-it-yourself: Hands-on practice
   - Comparison: Alternative approaches

5. **Pitfalls & Best Practices**:
   | Category | Examples |
   |----------|----------|
   | Error handling | Bare except clauses |
   | State | Global vars, mutable defaults |
   | Security | eval(), SQL injection |
   | Performance | Memory inefficiencies |

6. **Scientific Computing**:
   | Domain | Focus |
   |--------|-------|
   | NumPy/SciPy | Broadcasting, vectorization, memory layout |
   | JAX | @jit, grad, vmap, pmap transforms |
   | Pandas | Method chaining, GroupBy, memory |
   | Julia | Type stability, multiple dispatch |
   | MD Sims | Integrators, forces, neighbor lists |
   | ML Training | Forward/backward, optimizers |

7. **Learning Resources** (Comprehensive):
   - Knowledge gaps from complexity analysis
   - Topic recommendations
   - Curated resources (tutorials, books, docs)
   - Practice projects by skill level
   - Learning plan (week-by-week)

## Output

```markdown
# Code Explanation: [Name]

## Complexity
- Difficulty: [level]
- Concepts: [list]
- Patterns: [patterns]

## What This Does
[Clear explanation]

## Visual Representation
[Mermaid diagrams]

## Step-by-Step Breakdown
[Detailed walkthrough]

## Common Pitfalls
[Issues to avoid]

## [If scientific]
## Scientific Context
[Domain-specific explanations]

## [Comprehensive]
## Your Learning Path
[Personalized resources]
```

## Success

- [ ] Complexity assessed
- [ ] Concepts explained
- [ ] Diagrams where helpful
- [ ] Step-by-step breakdown
- [ ] Pitfalls highlighted
- [ ] Examples runnable
- [ ] Matches user skill level
- [ ] Scientific code has domain context
