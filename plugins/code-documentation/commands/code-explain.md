---
version: "1.0.3"
category: "code-documentation"
command: "/code-explain"
description: Detailed explanation of code structure, functionality, and design patterns with scientific computing support
allowed-tools: Bash(find:*), Bash(grep:*), Bash(git:*)
argument-hint: <code-path-or-snippet>
color: cyan
execution_modes:
  quick: "5-10 minutes - Basic code walkthrough with key concepts"
  standard: "15-25 minutes - Comprehensive explanation with diagrams and examples"
  comprehensive: "30-45 minutes - Full analysis with learning resources and practice exercises"
agents:
  primary:
    - research-intelligence
  conditional:
    - agent: systems-architect
      trigger: complexity > 10 OR pattern "architecture|design.*pattern|system.*design|scalability"
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|matplotlib|scientific.*computing|numerical|simulation"
    - agent: neural-architecture-engineer
      trigger: pattern "torch|pytorch|tensorflow|keras|neural.*network|deep.*learning"
    - agent: jax-pro
      trigger: pattern "jax|flax|@jit|@vmap|@pmap|grad\\(|optax"
    - agent: correlation-function-expert
      trigger: pattern "correlation|fft|spectral.*analysis|statistical.*physics"
    - agent: simulation-expert
      trigger: pattern "lammps|gromacs|molecular.*dynamics|md.*simulation|ase"
  orchestrated: false
---

# Code Explanation and Analysis

Transform complex code into clear, understandable explanations through progressive disclosure, visual aids, and domain-specific expertise.

## Execution Modes

| Mode | Time | Use Case | Phases Included |
|------|------|----------|-----------------|
| **quick** | 5-10 min | Simple code walkthrough, urgent explanations | Overview + Key Concepts |
| **standard** (default) | 15-25 min | Typical code explanation needs | Overview + Diagrams + Examples |
| **comprehensive** | 30-45 min | Complex code, learning-focused, teaching | All phases + Learning Resources |

## Quick Reference Documentation

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| Code Analysis | [code-analysis-framework.md](../docs/code-documentation/code-analysis-framework.md) | ~350 |
| Visualizations | [visualization-techniques.md](../docs/code-documentation/visualization-techniques.md) | ~400 |
| Learning Resources | [learning-resources.md](../docs/code-documentation/learning-resources.md) | ~450 |
| Scientific Computing | [scientific-code-explanation.md](../docs/code-documentation/scientific-code-explanation.md) | ~400 |

**Total External Documentation**: ~1,600 lines of detailed patterns and examples

## Requirements

$ARGUMENTS

## Core Workflow

### Phase 1: Code Comprehension Analysis

**Analyze code complexity and structure** using AST parsing:

1. **Parse code structure** (classes, functions, imports)
2. **Calculate complexity metrics** (cyclomatic complexity, nesting depth)
3. **Identify programming concepts** (async, decorators, generators, etc.)
4. **Detect design patterns** (Singleton, Observer, Factory, etc.)
5. **Assess difficulty level** (beginner/intermediate/advanced/expert)

**Implementation**: See [code-analysis-framework.md](../docs/code-documentation/code-analysis-framework.md) for:
- `CodeAnalyzer` class with full AST analysis
- Multi-language analyzers (Python, JavaScript, Go)
- Complexity calculation algorithms
- Pattern detection logic
- Best practices checker

### Phase 2: Visual Explanation Generation

**Create visual representations** of code flow and structure:

1. **Generate flowcharts** for logic flow (if/else, loops, control flow)
2. **Create class diagrams** for OOP structure and relationships
3. **Build sequence diagrams** for object interactions
4. **Visualize algorithms** step-by-step (sorting, recursion, etc.)
5. **Draw architecture diagrams** for system components

**Implementation**: See [visualization-techniques.md](../docs/code-documentation/visualization-techniques.md) for:
- Mermaid diagram generation (flowcharts, class diagrams, sequence diagrams)
- Algorithm step-by-step visualization
- Recursion tree rendering
- Data structure visualization
- Architecture diagram patterns

### Phase 3: Step-by-Step Explanation

**Progressive disclosure** from simple to complex:

1. **High-level overview** - What does this code do?
2. **Function-by-function breakdown** - How does each part work?
3. **Concept explanations** - Deep dive into programming concepts used
4. **Design pattern analysis** - Why these patterns were chosen
5. **Performance considerations** - Efficiency and optimization notes

**Output Structure**:
```markdown
## What This Code Does
[2-3 sentence summary]

## Key Concepts
[List of concepts with links to explanations]

## Step-by-Step Breakdown
### Step 1: [Function Name]
- Purpose: [What it does]
- How it works: [Logic breakdown]
- [Diagram if complex]

## Deep Dive: [Concept]
[Detailed explanation with examples]
```

### Phase 4: Interactive Examples (Standard & Comprehensive modes)

**Provide runnable examples** for experimentation:

1. **Basic usage examples** - Simple cases
2. **Edge case handling** - Error scenarios
3. **Try-it-yourself exercises** - Hands-on practice
4. **Comparison examples** - Alternative approaches

**Example patterns**: See [learning-resources.md](../docs/code-documentation/learning-resources.md) for:
- Interactive code examples (error handling, async, decorators)
- Try-it-yourself templates
- Comparison of approaches
- Exercise suggestions

### Phase 5: Common Pitfalls and Best Practices

**Highlight potential issues** and improvements:

1. **Identify anti-patterns** in the code
2. **Explain pitfalls** with clear examples
3. **Suggest improvements** with refactored code
4. **Provide best practices** for similar scenarios

**Pitfall categories**: See [learning-resources.md](../docs/code-documentation/learning-resources.md#common-pitfalls-explained) for:
- Bare except clauses
- Global variable usage
- Mutable default arguments
- Security issues (eval, SQL injection)
- Memory inefficiencies

### Phase 6: Scientific Computing Support (when applicable)

**Domain-specific explanations** for scientific code:

**NumPy/SciPy**: Array broadcasting, vectorization, memory layout
**JAX**: Functional transformations (@jit, grad, vmap, pmap)
**Pandas**: Method chaining, GroupBy patterns, memory efficiency
**Julia**: Type stability, multiple dispatch, performance annotations
**Molecular Dynamics**: Integrators, force calculations, neighbor lists
**ML Training**: Forward/backward pass, optimizer patterns, numerical stability

**Implementation**: See [scientific-code-explanation.md](../docs/code-documentation/scientific-code-explanation.md) for:
- Complete NumPy broadcasting explanations
- JAX transformation patterns
- Pandas optimization techniques
- Julia performance patterns
- MD simulation structures
- ML training loop architectures

### Phase 7: Learning Resources (Comprehensive mode only)

**Personalized learning path** based on code analysis:

1. **Identified knowledge gaps** from complexity analysis
2. **Recommended topics** for deeper understanding
3. **Curated resources** (tutorials, books, documentation)
4. **Practice projects** (beginner/intermediate/advanced)
5. **Structured learning plan** (week-by-week)

**Resources**: See [learning-resources.md](../docs/code-documentation/learning-resources.md) for:
- Design pattern library
- Programming concept tutorials
- Curated resource links
- Project suggestions by level

## Mode-Specific Execution

### Quick Mode (5-10 minutes)
**Phases**: 1, 3 (abbreviated)
**Output**: High-level overview, key concepts, basic breakdown
**Skip**: Diagrams, exercises, learning paths

### Standard Mode (15-25 minutes) - DEFAULT
**Phases**: 1, 2, 3, 4, 5
**Output**: Complete explanation with diagrams, examples, pitfalls
**Skip**: Learning paths

### Comprehensive Mode (30-45 minutes)
**Phases**: All 7 phases
**Output**: Full analysis, diagrams, examples, pitfalls, learning resources
**Includes**: Personalized learning path and practice exercises

## Output Format

```markdown
# Code Explanation: [Code Name/Function]

## Complexity Analysis
- **Difficulty Level**: [beginner/intermediate/advanced/expert]
- **Key Concepts**: [List of concepts]
- **Design Patterns**: [Patterns found]
- **Complexity Score**: [Metrics]

## What This Code Does
[Clear, concise explanation]

## Visual Representation
[Mermaid diagrams]

## Step-by-Step Breakdown
[Detailed walkthrough]

## Interactive Examples
[Runnable code examples]

## Common Pitfalls
[Issues to avoid]

## Best Practices
[Recommended improvements]

[If scientific code]
## Scientific Computing Context
[Domain-specific explanations]

[Comprehensive mode only]
## Your Learning Path
[Personalized resources and plan]
```

## Agent Integration

- **research-intelligence**: Primary agent for code analysis
- **systems-architect**: Triggered for complex architecture (>10 complexity or architecture patterns)
- **hpc-numerical-coordinator**: Triggered for scientific computing code
- **neural-architecture-engineer**: Triggered for deep learning code
- **jax-pro**: Triggered for JAX-specific code
- **correlation-function-expert**: Triggered for spectral analysis code
- **simulation-expert**: Triggered for molecular dynamics code

## Success Criteria

✅ Code complexity accurately assessed
✅ All programming concepts identified and explained
✅ Visual diagrams generated where helpful
✅ Step-by-step breakdown provided
✅ Common pitfalls highlighted
✅ Examples are runnable and clear
✅ Explanation matches user's skill level
✅ Scientific code includes domain-specific context

Focus on **clarity**, **progressive disclosure**, and **practical understanding** to transform difficult code into accessible knowledge.
