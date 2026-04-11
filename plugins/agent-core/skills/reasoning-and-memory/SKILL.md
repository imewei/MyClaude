---
name: reasoning-and-memory
description: Meta-orchestrator for reasoning frameworks, reflection, self-improving agents, and memory systems. Routes to specialized skills for structured reasoning, reflection, self-improvement loops, knowledge graphs, and persistent memory. Use when designing reasoning pipelines, implementing reflection or self-improvement loops, optimizing prompts with DSPy / TextGrad, or building memory-augmented agents.
---

# Reasoning and Memory

Orchestrator for reasoning, meta-cognition, self-improvement, and memory system design. Routes problems to the appropriate specialized skill based on whether the task involves structured inference, meta-cognitive reflection, closed-loop self-improvement, knowledge representation, or persistent memory retrieval.

## Expert Agent

For complex reasoning, self-improvement loop architecture, and memory problems requiring deep analytical expertise, delegate to the expert agent:

- **`reasoning-engine`**: Specialist for structured reasoning, reflection pipelines, self-improving agent design, and memory-augmented agent design.
  - *Location*: `plugins/agent-core/agents/reasoning-engine.md`
  - *Capabilities*: Chain-of-Thought design, bias detection, closed-loop self-improvement, knowledge graph reasoning, and vector memory integration.

## Core Skills

### [Reasoning Frameworks](../reasoning-frameworks/SKILL.md)
Chain-of-Thought, First Principles decomposition, and structured multi-step analysis. Use when designing how an agent should think through a problem systematically.

### [Reflection Framework](../reflection-framework/SKILL.md)
Meta-cognitive analysis, bias detection, and session retrospectives. Use when an agent needs to evaluate the quality of its own reasoning or outputs.

### [Self-Improving Agents](../self-improving-agents/SKILL.md)
Closed-loop self-improvement: reflection-refine-validate, self-consistency ensembles, DSPy and TextGrad automatic prompt optimization, evolutionary prompt search, and constitutional self-critique. Use when the agent must persist an improved prompt, policy, or reasoning chain across runs — not just critique a single turn.

### [Knowledge Graph Patterns](../knowledge-graph-patterns/SKILL.md)
Entity resolution, semantic reasoning, and graph traversal strategies. Use when structured relational knowledge must be stored, queried, or updated.

### [Memory System Patterns](../memory-system-patterns/SKILL.md)
Vector stores, conversation history management, and context window optimization. Use when agents need to retrieve or persist information across turns or sessions.

## Routing Decision Tree

```
What is the primary need?
|
+-- Agent needs to reason through a problem step-by-step?
|   --> reasoning-frameworks (CoT, First Principles, structured analysis)
|
+-- Agent needs to evaluate its own reasoning (one-shot critique)?
|   --> reflection-framework (meta-cognition, bias detection)
|
+-- Agent needs to improve its own prompt / policy / chain (persistent)?
|   --> self-improving-agents (DSPy, TextGrad, self-consistency, constitutional)
|
+-- Knowledge must be stored as structured relationships?
|   --> knowledge-graph-patterns (entity resolution, graph traversal)
|
+-- Information must persist or be retrieved across turns?
    --> memory-system-patterns (vector stores, conversation history)
```

## Checklist

- [ ] Identify the primary need using the routing decision tree before selecting a sub-skill
- [ ] Confirm reasoning frameworks are appropriate for the problem complexity (avoid over-structured CoT for simple tasks)
- [ ] Verify reflection is triggered on high-stakes outputs, not every turn
- [ ] For self-improvement loops: defined a concrete metric and held out a separate eval set before compiling
- [ ] Ensure knowledge graph schemas define entity types and relation cardinality before population
- [ ] Validate memory retrieval includes relevance scoring and staleness handling
- [ ] Document reasoning trace format so downstream agents can parse intermediate steps
