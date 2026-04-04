---
name: agent-systems
description: Meta-orchestrator for multi-agent systems. Routes to specialized skills for agent coordination, performance optimization, evaluation, and tool use patterns. Use when designing multi-agent workflows, optimizing agent performance, or implementing tool chains.
---

# Agent Systems

Orchestrator for multi-agent system design and operation. Routes problems to the appropriate specialized skill based on whether the task involves coordination architecture, performance tuning, quality evaluation, or tool chain design.

## Expert Agent

For complex multi-agent problems requiring deep orchestration expertise, delegate to the expert agent:

- **`orchestrator`**: Specialist for multi-agent workflows, team assembly, and inter-agent coordination.
  - *Location*: `plugins/agent-core/agents/orchestrator.md`
  - *Capabilities*: DAG workflow design, agent team assembly, task decomposition, and coordination protocol design.

## Core Skills

### [Multi-Agent Coordination](../multi-agent-coordination/SKILL.md)
DAG workflows, team assembly, inter-agent messaging, and task decomposition. Use when designing the structure and communication patterns of a multi-agent system.

### [Agent Performance Optimization](../agent-performance-optimization/SKILL.md)
Metrics collection, response caching, load balancing, and latency reduction. Use when an existing agent system is too slow or resource-intensive.

### [Agent Evaluation](../agent-evaluation/SKILL.md)
Benchmarking, quality assessment, output scoring, and regression tracking. Use when measuring or improving the reliability and accuracy of agent outputs.

### [Tool Use Patterns](../tool-use-patterns/SKILL.md)
Tool selection heuristics, chaining strategies, error handling, and retry logic. Use when designing how agents discover, invoke, and recover from tool failures.

## Routing Decision Tree

```
What is the primary concern?
|
+-- Designing how agents collaborate or communicate?
|   --> multi-agent-coordination (DAG workflows, team assembly)
|
+-- System is too slow or uses too many resources?
|   --> agent-performance-optimization (caching, load balancing)
|
+-- Measuring or improving output quality?
|   --> agent-evaluation (benchmarks, scoring, regression)
|
+-- Designing how an agent selects or chains tools?
    --> tool-use-patterns (selection, chaining, error handling)
```

## Checklist

- [ ] Identify the primary concern using the routing decision tree before selecting a sub-skill
- [ ] Confirm the agent team composition matches the task complexity (avoid over-engineering)
- [ ] Verify inter-agent message schemas are typed and validated at boundaries
- [ ] Ensure all tool invocations include error handling and retry budgets
- [ ] Validate performance metrics are collected before and after optimization changes
- [ ] Document evaluation criteria and acceptance thresholds before running benchmarks
