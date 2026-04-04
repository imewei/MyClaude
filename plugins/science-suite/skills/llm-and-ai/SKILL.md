---
name: llm-and-ai
description: Meta-orchestrator for LLM applications and AI engineering. Routes to LLM app development, evaluation, LangChain, RAG, and NLP skills. Use when building LLM-powered applications, evaluating model outputs, implementing RAG pipelines, or processing text with NLP. For prompt engineering patterns, see agent-core `llm-engineering` hub (`plugins/agent-core/skills/llm-engineering/SKILL.md`).
---

# LLM and AI

Orchestrator for LLM application development and AI engineering. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`ai-engineer`**: Specialist for LLM applications, RAG systems, and AI product development.
  - *Location*: `plugins/science-suite/agents/ai-engineer.md`
  - *Capabilities*: LLM APIs, LangChain, RAG pipelines, prompt engineering, evaluation frameworks, and NLP.

## Core Skills

### [LLM Application Dev](../llm-application-dev/SKILL.md)
Building LLM-powered apps: API integration, streaming, tool use, and agent patterns.

### [LLM Evaluation](../llm-evaluation/SKILL.md)
Evaluation frameworks: benchmarks, LLM-as-judge, human evaluation, and output quality metrics.

### [LangChain Architecture](../langchain-architecture/SKILL.md)
LangChain / LangGraph: chains, agents, memory, tools, and multi-step workflows.

### [RAG Implementation](../rag-implementation/SKILL.md)
Retrieval-Augmented Generation: vector stores, chunking, re-ranking, and hybrid retrieval.

### [NLP Fundamentals](../nlp-fundamentals/SKILL.md)
NLP foundations: tokenization, embeddings, text classification, NER, and sequence modeling.

## Routing Decision Tree

```
What is the LLM / AI task?
|
+-- Build an LLM-powered application?
|   --> llm-application-dev
|
+-- Evaluate LLM output quality?
|   --> llm-evaluation
|
+-- Use LangChain / LangGraph agents?
|   --> langchain-architecture
|
+-- Build a RAG system?
|   --> rag-implementation
|
+-- NLP text processing / embeddings?
    --> nlp-fundamentals
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| API integration, streaming, tool use | `llm-application-dev` |
| Benchmarks, LLM-as-judge | `llm-evaluation` |
| Chains, agents, memory | `langchain-architecture` |
| Vector stores, re-ranking | `rag-implementation` |
| Tokenization, embeddings, NER | `nlp-fundamentals` |

## Checklist

- [ ] Use routing tree to select the most specific sub-skill
- [ ] Define evaluation metrics before building the application
- [ ] Prototype prompts with `prompt-engineering-patterns` before full implementation
- [ ] Benchmark RAG retrieval recall before optimizing generation
- [ ] Validate LLM outputs against ground truth with `llm-evaluation`
- [ ] Use structured outputs (JSON mode / function calling) to reduce parsing failures
- [ ] Monitor token costs and latency in production LLM applications
- [ ] Store prompt versions alongside model versions for reproducibility
