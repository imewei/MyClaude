name: prompt-engineering-patterns
version: "2.1.0"
description: Master advanced prompt engineering with chain-of-thought, few-shot learning, and production templates. Use when designing prompts for AI applications, implementing structured reasoning, optimizing for consistency, or building reusable prompt systems.

# Prompt Engineering Patterns

Advanced techniques for maximizing LLM performance and reliability.

## Core Techniques

| Technique | Use Case | Implementation |
|-----------|----------|----------------|
| Zero-shot CoT | Complex reasoning | "Let's think step by step" |
| Few-shot | Task demonstration | 2-5 input-output examples |
| Self-consistency | Reliability | Sample multiple paths, vote |
| Tree-of-thought | Complex planning | Branch and evaluate paths |
| Self-verification | Accuracy | Ask model→check its answer |

## Prompt Structure

```
[System Context] → [Task Instruction] → [Examples] → [Input] → [Output Format]
```

### Example Template

```python
template = """You are an expert SQL developer.

Examples:
Q: Find all users registered last week
A: SELECT * FROM users WHERE created_at > NOW() - INTERVAL '7 days'

Q: {query}
A: """
```

## Progressive Disclosure

| Level | Example |
|-------|---------|
| 1. Direct | "Summarize this article" |
| 2. Constrained | "Summarize in 3 bullet points, focus on findings" |
| 3. Reasoning | "Read, identify main findings, then summarize" |
| 4. Few-shot | Include 2-3 example summaries |

## Few-Shot Selection

| Strategy | When to Use |
|----------|-------------|
| Semantic similarity | Examples matching input domain |
| Diversity sampling | Cover edge cases |
| Difficulty-based | Match input complexity |
| Random (baseline) | When unsure |

## Performance Optimization

| Goal | Technique |
|------|-----------|
| Reduce tokens | Remove redundant words, use abbreviations |
| Lower latency | Shorter prompts, streaming output |
| Improve consistency | Add output format constraints |
| Handle failures | Include fallback instructions |

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Be specific | Vague prompts → inconsistent results |
| Show, don't tell | Examples > descriptions |
| Test extensively | Diverse, representative inputs |
| Iterate rapidly | Small changes → large impacts |
| Version control | Treat prompts as code |
| Monitor production | Track accuracy, latency, costs |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Over-engineering | Start simple, add complexity as needed |
| Example pollution | Use examples matching target task |
| Context overflow | Limit examples to fit token budget |
| Ambiguous instructions | Eliminate multiple interpretations |
| No edge case testing | Test unusual and boundary inputs |

## Parallel Prompt Optimization

| Strategy | Implementation | Benefit |
|----------|----------------|---------|
| **Batch Testing** | Test N variations concurrently | Rapid iteration cycle |
| **Grid Search** | Cartesian product of components | Exhaustive parameter tuning |
| **Self-Consistency** | Generate k paths in parallel | Majority vote for reliability |
| **Ensemble** | Query multiple models | Diverse perspectives |

## Integration Patterns

### With RAG

```python
prompt = f"""Context: {retrieved_context}

Question: {user_question}

Answer based solely on the context. If insufficient, state what's missing."""
```

### With Validation

```python
prompt = f"""{main_task}

After responding, verify:
1. Answers the question directly
2. Uses only provided context
3. Acknowledges uncertainty"""
```

## Checklist

- [ ] Clear task instruction defined
- [ ] Output format specified
- [ ] Examples selected (if few-shot)
- [ ] Token budget considered
- [ ] Edge cases tested
- [ ] Validation step included
- [ ] Version tracked
