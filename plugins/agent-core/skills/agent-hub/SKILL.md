---
name: agent-hub
description: >-
  Top-level router for agent orchestration, reasoning, and LLM engineering. Use for: multi-agent workflow design/agent coordination/tool chains/task handoff/delegation patterns/agent evaluation/performance optimization; reasoning pipelines/reflection/self-improvement loops/DSPy/TextGrad/knowledge graphs/memory-augmented agents; LLM feature implementation/prompt design patterns/RAG systems/MCP tool integration/safety guardrails; writing or optimizing prompts — "help me write a prompt", "optimize this prompt", "make this prompt better", brain dumps for LLM goals, "I want to build an LLM to do X", non-English prompt requests, or /thinkfirst.
---

# Agent Core (agent-hub)

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
|   --> agent-core:thinkfirst
|
+-- Build an LLM app, RAG pipeline, eval system, safety layer, production prompts, or integrate MCP tools?
|   --> agent-core:llm-engineering
|
+-- Design reasoning chains, reflection loops, memory systems, or knowledge graphs?
|   --> agent-core:reasoning-and-memory
|
+-- Coordinate multiple agents, evaluate agent output, design tool chains, or optimize agent performance / latency?
|   --> agent-core:agent-systems
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to orchestrator for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger | Hub skill |
|---|---|
| Multi-agent design, tool chaining, agent evaluation, agent performance optimization, latency, caching | `agent-core:agent-systems` |
| Reasoning frameworks, memory, self-improvement, DSPy, TextGrad, knowledge graphs, entity resolution, vector stores | `agent-core:reasoning-and-memory` |
| LLM apps, RAG, evals, safety, prompt systems, MCP integration, production prompt engineering | `agent-core:llm-engineering` |
| Writing or optimizing a prompt, /thinkfirst, brain dump, "help me write a prompt" | `agent-core:thinkfirst` |

## Checklist

- [ ] Identify the primary concern using the routing decision tree before selecting a hub
- [ ] For prompt tasks, always route to `agent-core:thinkfirst` even if other agent topics are mentioned
- [ ] For multi-agent + reasoning overlap, prefer `agent-core:agent-systems` (reasoning is a sub-concern)
- [ ] Confirm the selected hub skill is invoked — do not answer from the meta-router alone
- [ ] Escalate to an expert agent for deep orchestration, context, or reasoning problems
- [ ] Validate that the chosen hub covers the full scope before starting implementation
