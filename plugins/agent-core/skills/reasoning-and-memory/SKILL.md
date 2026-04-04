---
name: reasoning-and-memory
description: Meta-orchestrator for reasoning frameworks and memory systems. Routes to specialized skills for structured reasoning, reflection, knowledge graphs, and persistent memory. Use when designing reasoning pipelines, implementing reflection, or building memory-augmented agents.
---

# Reasoning and Memory

Orchestrator for reasoning and memory system design. Routes problems to the appropriate specialized skill based on whether the task involves structured inference, meta-cognitive reflection, knowledge representation, or persistent memory retrieval.

## Expert Agent

For complex reasoning and memory problems requiring deep analytical expertise, delegate to the expert agent:

- **`reasoning-engine`**: Specialist for structured reasoning, reflection pipelines, and memory-augmented agent design.
  - *Location*: `plugins/agent-core/agents/reasoning-engine.md`
  - *Capabilities*: Chain-of-Thought design, bias detection, knowledge graph reasoning, and vector memory integration.

## Core Skills

### [Reasoning Frameworks](../reasoning-frameworks/SKILL.md)
Chain-of-Thought, First Principles decomposition, and structured multi-step analysis. Use when designing how an agent should think through a problem systematically.

### [Reflection Framework](../reflection-framework/SKILL.md)
Meta-cognitive analysis, bias detection, and session retrospectives. Use when an agent needs to evaluate the quality of its own reasoning or outputs.

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
+-- Agent needs to evaluate or improve its own reasoning?
|   --> reflection-framework (meta-cognition, bias detection)
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
- [ ] Ensure knowledge graph schemas define entity types and relation cardinality before population
- [ ] Validate memory retrieval includes relevance scoring and staleness handling
- [ ] Document reasoning trace format so downstream agents can parse intermediate steps
