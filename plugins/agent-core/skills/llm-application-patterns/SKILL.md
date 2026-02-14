---
name: llm-application-patterns
version: "2.2.1"
description: Design and build robust LLM-powered applications. Covers advanced prompt engineering (CoT, few-shot), RAG implementation, and LLM evaluation.
---

# LLM Application Patterns

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
