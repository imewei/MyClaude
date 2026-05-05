---
name: llm-application-dev
description: Production implementation of LLM applications using LangChain/LangGraph with code examples for RAG pipelines, agentic workflows, cost optimization, and deployment. Use when writing LLM application code, implementing retry logic, or deploying with monitoring. For architectural design patterns, see agent-core llm-application-patterns.
---

# LLM Application Development

Building robust applications with Large Language Models.

## Expert Agents

For LLM application development, delegate to:

- **`ai-engineer`**: RAG systems, agentic workflows, and application architecture.
  - *Location*: `plugins/science-suite/agents/ai-engineer.md`
- **`prompt-engineer`**: Prompt optimization, safety, and evaluation.
  - *Location*: `plugins/science-suite/agents/prompt-engineer.md`

> For RAG, LangChain, and evaluation routing, use the parent hub: `science-suite:llm-and-ai`.

## Application Patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| Chat | Stateful conversation with memory | Customer support, assistants |
| RAG | Retrieval-augmented generation | Knowledge bases, document Q&A |
| Extraction | Structured output from text | Data parsing, form filling |
| Agent | Autonomous tool-using LLM | Research, multi-step tasks |
| Pipeline | Chained LLM transformations | Content processing, translation |

## Prompt Management

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Version prompts as templates, not inline strings
SYSTEM_PROMPT = """Role: {role}
Instructions: {instructions}
Output format: {format}"""

# Few-shot with examples stored externally
examples = [{"input": "...", "output": "..."}]
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"), ("ai", "{output}")
    ]),
    examples=examples,
)
```

## Error Handling and Retries

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(prompt: str) -> str:
    """Wrap LLM calls with retry logic for transient failures."""
    response = llm.invoke(prompt)
    if not response.content:
        raise ValueError("Empty LLM response")
    return response.content
```

## Cost Optimization

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Response caching | 50-90% | Stale answers for dynamic queries |
| Prompt compression | 20-40% | Slight quality reduction |
| Model tiering (haiku first) | 60-80% | Route complex queries to larger model |
| Batch API calls | 50% | Higher latency (24h window) |
| Max token limits | Variable | Truncated outputs |

## Deployment Checklist

- [ ] Prompts versioned and parameterized (no hardcoded strings)
- [ ] Retry logic with exponential backoff on all LLM calls
- [ ] Input validation and output parsing with Pydantic models
- [ ] Cost monitoring with token usage tracking per request
- [ ] Rate limiting to prevent runaway API spend
- [ ] Guardrails for PII filtering and content safety
- [ ] Structured logging of prompt/response pairs for debugging
- [ ] Evaluation suite measuring accuracy, latency, and cost per query
