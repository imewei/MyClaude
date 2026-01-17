---
description: Optimize prompts for better LLM performance through CoT, few-shot learning,
  and constitutional AI
triggers:
- /prompt-optimize
- optimize prompts for better
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<prompt_to_optimize>`
The agent should parse these arguments from the user's request.

# Prompt Optimization

Transform basic instructions into production-ready prompts. Can improve accuracy by 40%, reduce hallucinations by 30%, and cut costs by 50-80%.

## Prompt to Optimize

$ARGUMENTS

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| `--quick` | 5-10 min | Analysis + one technique, 3 test cases |
| standard (default) | 15-25 min | Full optimization (CoT + Few-Shot + Constitutional), 10 test cases |
| `--comprehensive` | 30-45 min | All + meta-prompt generation, A/B testing strategy, 20+ test cases |

## Phase 1: Analyze Current Prompt

### Assessment Dimensions

| Dimension | Evaluate |
|-----------|----------|
| Clarity (1-10) | Ambiguity, explicit vs implicit expectations |
| Structure | Logical flow, section boundaries |
| Model Alignment | Capability utilization, token efficiency |
| Performance | Success rate, failure modes, edge cases |

### Decomposition
- Core objective and constraints
- Output format requirements
- Context dependencies
- Variable elements

## Phase 2: Apply Chain-of-Thought

### CoT Patterns

| Pattern | When to Use |
|---------|-------------|
| Zero-Shot CoT | Add "Let's think step-by-step" |
| Few-Shot CoT | Provide examples with reasoning |
| Tree-of-Thoughts | Explore multiple solution paths |

**Structure:** Break reasoning into numbered steps, each building on previous.

## Phase 3: Add Few-Shot Learning

### Example Selection

| Type | Purpose |
|------|---------|
| Simple case | Demonstrates basic pattern |
| Edge case | Shows complexity handling |
| Counter-example | What NOT to do |

**Format:** Input → Output with clear labels

## Phase 4: Apply Constitutional AI

### Self-Critique Pattern

1. Generate initial response
2. Review against principles:
   - ACCURACY: Verify claims, flag uncertainties
   - SAFETY: Check for harm, bias, ethics
   - QUALITY: Clarity, consistency, completeness
3. Produce refined response

**Benefits:** -40% harmful outputs, +25% factual accuracy

## Phase 5: Model-Specific Optimization

### GPT-4 Style
```
##CONTEXT##
##OBJECTIVE##
##INSTRUCTIONS## (numbered)
##OUTPUT FORMAT## (JSON/structured)
```

### Claude Style
```xml
<context>background</context>
<task>objective</task>
<thinking>step-by-step</thinking>
<output_format>structure</output_format>
```

## Phase 6: Evaluate and Test

### Test Protocol

| Category | Count | Purpose |
|----------|-------|---------|
| Typical | 10 | Standard inputs |
| Edge | 5 | Boundary conditions |
| Adversarial | 3 | Stress testing |
| Out-of-scope | 2 | Rejection behavior |

### LLM-as-Judge Criteria
1. Task completion (fully addressed?)
2. Accuracy (factually correct?)
3. Reasoning (logical and structured?)
4. Format (matches requirements?)
5. Safety (unbiased and safe?)

## Common Patterns by Task Type

| Task | Apply |
|------|-------|
| Reasoning | CoT + verification step |
| Classification | Few-shot with each class + structured output |
| Generation | Clear constraints + quality criteria |
| RAG | Citation requirements + gap handling |

## Output Format

```yaml
analysis:
  clarity: X/10
  token_count: N
  estimated_success: X%

improvements:
  - technique: "Name"
    impact: "description"

performance_projection:
  success_rate: before% → after%
  quality_score: before → after

deployment:
  model: recommended
  temperature: value
  testing: A/B strategy
```

## Best Practices

1. Start simple, add complexity as needed
2. Test early with real examples
3. Measure impact (success rate, quality, cost)
4. Version control prompts
5. A/B test before committing
6. Document design decisions

## External Documentation

- `prompt-patterns.md` - Complete technique library (~500 lines)
- `prompt-examples.md` - Production examples (~400 lines)
- `prompt-evaluation.md` - Testing and monitoring (~300 lines)
