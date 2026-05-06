---
name: agent-core
description: Top-level router for agent orchestration, reasoning, and LLM engineering. Use for: multi-agent workflow design/agent coordination/tool chains/task handoff/delegation patterns/agent evaluation/performance optimization; reasoning pipelines/reflection/self-improvement loops/DSPy/TextGrad/knowledge graphs/memory-augmented agents; LLM feature implementation/prompt design patterns/RAG systems/MCP tool integration/safety guardrails; writing or optimizing prompts — "help me write a prompt", "optimize this prompt", "make this prompt better", brain dumps for LLM goals, "I want to build an LLM to do X", non-English prompt requests, or /thinkfirst.
---

# Agent Core

## Expert Agents

- **`orchestrator`**: Multi-agent workflows, team assembly, inter-agent coordination.
- **`context-specialist`**: Context management, memory retrieval, information scoping.
- **`reasoning-engine`**: Structured reasoning, chain-of-thought, reflection loops.

## Hub Skills

- [**agent-systems**](../agent-systems/SKILL.md) — Multi-agent coordination, agent evaluation, tool use patterns.
- [**reasoning-and-memory**](../reasoning-and-memory/SKILL.md) — Reasoning frameworks, reflection, self-improving agents, memory systems.
- [**llm-engineering**](../llm-engineering/SKILL.md) — LLM app dev, prompt systems, RAG, tool use, eval, safety.
- [**thinkfirst**](../thinkfirst/SKILL.md) — Prompt writing and optimization; use when the user wants to write or improve a prompt.

## Routing Decision Tree

```
What is the primary task?
|
+-- Write, improve, or optimize a prompt?
|   --> thinkfirst
|
+-- Build an LLM app, RAG pipeline, eval system, or safety layer?
|   --> llm-engineering
|
+-- Design reasoning chains, reflection loops, or memory systems?
|   --> reasoning-and-memory
|
+-- Coordinate multiple agents, evaluate agent output, or design tool chains?
    --> agent-systems
```

## Routing Table

| Trigger | Hub skill |
|---|---|
| Multi-agent design, tool chaining, agent evaluation | `agent-systems` |
| Reasoning frameworks, memory, self-improvement | `reasoning-and-memory` |
| LLM apps, RAG, evals, safety, prompt systems | `llm-engineering` |
| Writing or optimizing a prompt | `thinkfirst` |

## Checklist

- [ ] Identify the primary concern using the routing decision tree before selecting a hub
- [ ] For prompt tasks, always route to `thinkfirst` even if other agent topics are mentioned
- [ ] For multi-agent + reasoning overlap, prefer `agent-systems` (reasoning is a sub-concern)
- [ ] Confirm the selected hub skill is invoked — do not answer from the meta-router alone
- [ ] Escalate to an expert agent for deep orchestration, context, or reasoning problems
- [ ] Validate that the chosen hub covers the full scope before starting implementation
