---
name: prompt-engineering-patterns
description: Master advanced prompt engineering with chain-of-thought, few-shot learning, prompt versioning, and production prompt templates. Use when designing prompts for LLM applications, improving agent performance, or crafting system prompts.
---

# Prompt Engineering Patterns

## Expert Agent

For prompt design, optimization, and assessment, delegate to:

- **`reasoning-engine`**: Applies structured reasoning to prompt decomposition and iterative refinement.
  - *Location*: `plugins/agent-core/agents/reasoning-engine.md`

Comprehensive guide for designing, testing, and maintaining production-grade prompts across LLM applications.

---

## 1. Chain-of-Thought (CoT) Patterns

### Zero-Shot CoT

Append a reasoning trigger to elicit step-by-step logic without examples.

```text
Analyze the following error log and determine the root cause.
Think step by step before providing your answer.
```

### Few-Shot CoT

Provide worked examples that demonstrate the desired reasoning chain.

```text
Q: A server receives 1000 req/s with 50ms avg latency. Will it handle 2000 req/s?
A: Step 1: Current throughput = 1000 req/s at 50ms = 50 concurrent requests.
   Step 2: At 2000 req/s and 50ms latency = 100 concurrent requests.
   Step 3: Check thread pool / connection limits against 100 concurrent.
   Conclusion: Depends on thread pool size. If pool >= 100, yes.

Q: A database query takes 200ms on 1M rows. Will it scale to 10M rows?
A: [Model completes the reasoning chain following the pattern]
```

### Self-Consistency

Run the same prompt N times, then aggregate answers by majority vote or confidence weighting.

| Strategy | When to Use | Cost |
|----------|-------------|------|
| Majority Vote | Discrete answers (classification) | N x base cost |
| Weighted Average | Numeric outputs | N x base cost |
| Best-of-N | Creative generation | N x base cost |

---

## 2. Prompt Template Design

### Structured Output Prompting

```text
You are a code review assistant. Analyze the provided diff and respond
in the following JSON format:

{
  "severity": "critical|high|medium|low",
  "category": "security|performance|correctness|style",
  "line_range": [start, end],
  "issue": "description of the problem",
  "suggestion": "concrete fix recommendation"
}

Rules:
- Return ONLY valid JSON, no markdown wrapping.
- If no issues found, return an empty array [].
```

### Role-Task-Format (RTF) Pattern

| Component | Purpose | Example |
|-----------|---------|---------|
| **Role** | Set expertise context | "You are a senior security engineer" |
| **Task** | Define the specific action | "Review this code for SQL injection" |
| **Format** | Constrain the output | "Return findings as a numbered list" |

### System Prompt Best Practices

- Place critical instructions at the **beginning** and **end** of the system prompt (primacy/recency effect).
- Use XML tags or markdown headers to create clear sections.
- Define explicit refusal behavior for out-of-scope requests.
- Include 1-2 grounding examples for ambiguous tasks.

---

## 3. Prompt Versioning Strategies

### Version Control Schema

```yaml
prompt:
  id: "code-review-v3.2"
  version: "3.2.0"  # semver: major.minor.patch
  model: "claude-sonnet-4-20250514"
  temperature: 0.2
  created: "2026-03-15"
  changelog: "Added security category, tightened JSON schema"
  template: |
    [prompt text here]
  test_cases:
    - input: "example diff with SQL injection"
      expected_contains: ["security", "critical"]
```

### Versioning Rules

| Change Type | Version Bump | Example |
|-------------|-------------|---------|
| Output format change | Major | JSON -> YAML output |
| New capability added | Minor | Add "security" category |
| Wording refinement | Patch | Clarify ambiguous instruction |

---

## 4. Prompt Testing and Assessment

### Assessment Dimensions

| Metric | Measurement | Target |
|--------|-------------|--------|
| **Accuracy** | Correct outputs / total | >= 90% |
| **Format Compliance** | Valid schema outputs / total | >= 98% |
| **Latency** | Time to first token | < 2s |
| **Consistency** | Agreement across N runs | >= 85% |
| **Refusal Rate** | Appropriate refusals / total | Context-dependent |

---

## 5. Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Vague instructions | Inconsistent outputs | Be specific about format and constraints |
| Prompt stuffing | Dilutes focus, increases cost | Split into focused sub-prompts |
| No examples | Ambiguous interpretation | Add 1-2 grounding examples |
| Hardcoded context | Breaks on new inputs | Use template variables |
| No assessment | Silent quality regression | Maintain test suites |
| Wildcard delegation | Loss of control | Define explicit scope boundaries |

---

## 6. Production Prompt Checklist

- [ ] Role and expertise level defined
- [ ] Task described with concrete success criteria
- [ ] Output format specified with schema or example
- [ ] Edge cases handled (empty input, malformed data, out-of-scope)
- [ ] Temperature and model version pinned
- [ ] At least 3 test cases with expected outputs
- [ ] Version tracked in source control
- [ ] Latency and cost measured under realistic load
- [ ] Refusal behavior defined for adversarial inputs

---

## Related Skills

- `thinkfirst` -- Upstream interview workflow for turning vague ideas or brain dumps into the structured requirements this skill then refines
- `reasoning-frameworks` -- Structured reasoning patterns (CoT, branching) that prompts can invoke
- `llm-application-patterns` -- Application-level patterns for RAG, agents, and tool use that consume prompts
