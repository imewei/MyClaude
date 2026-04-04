---
name: llm-application-patterns
description: Architectural patterns for LLM applications including prompt engineering principles (CoT, few-shot), RAG design, and evaluation strategies. Use when designing LLM system architecture, choosing RAG vs fine-tuning approaches, or establishing evaluation frameworks. For production implementation with LangChain/LangGraph code, see science-suite llm-application-dev.
---

# LLM Application Patterns

## Expert Agent

For advanced prompt engineering, RAG design, and LLM evaluation, delegate to:

- **`reasoning-engine`**: Masters Chain-of-Thought, few-shot prompting, constitutional AI principles, and prompt optimization.
  - *Location*: `plugins/agent-core/agents/reasoning-engine.md`
- **`context-specialist`**: Architects RAG retrieval pipelines, vector database queries, and context injection strategies.
  - *Location*: `plugins/agent-core/agents/context-specialist.md`

Expert guide for building reliable, performant, and scalable applications on top of Large Language Models.

## 1. Advanced Prompt Engineering

- **Chain-of-Thought (CoT)**: Encourage explicit reasoning steps to improve performance on complex tasks.
- **Few-Shot Learning**: Provide 2-5 high-quality examples to demonstrate the desired output format and style.
- **Output Constraints**: Use structured formats (JSON, Markdown) and schemas (Zod, Pydantic) for reliable parsing.
- **Self-Verification**: Ask the model to review and correct its own responses before final delivery.

## 2. RAG & Data Augmentation

- **Retrieval-Augmented Generation**: Combine LLMs with external knowledge bases for up-to-date and domain-specific context.
- **Context Injection**: Carefully curate and rank retrieved snippets to fit within token budgets.
- **Citation Mastery**: Ensure the model cites its sources directly within the generated text.

## 3. Evaluation & Optimization

- **LLM-as-a-Judge**: Use powerful models to evaluate the quality and accuracy of other model outputs.
- **Prompt Versioning**: Treat prompts as code; use version control and systematic A/B testing.
- **Latency Optimization**: Use streaming, shorter prompts, and faster models where appropriate.

## 4. Implementation Checklist

- [ ] **Instruction Clarity**: Is the task described without ambiguity?
- [ ] **Few-Shot Quality**: Are the provided examples representative and diverse?
- [ ] **Safety & Alignment**: Are there guardrails to prevent harmful or off-topic outputs?
- [ ] **Observability**: Are latency, tokens, and success rates being monitored?
- [ ] **Validation**: Is the output being validated against a schema or secondary model?

## Related Skills

- `reasoning-frameworks` -- Structured reasoning methods (CoT, First Principles) used in advanced prompting
- `mcp-integration` -- Context7 and tool integration for RAG retrieval pipelines
- `reflection-framework` -- Meta-cognitive evaluation of LLM output quality and reasoning
