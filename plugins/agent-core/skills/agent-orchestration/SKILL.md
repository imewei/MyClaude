---
name: agent-orchestration
description: Coordinate multiple AI agents through workflow orchestration, task allocation, and inter-agent communication protocols. Masters DAG-based workflows and team assembly. Use when designing multi-agent workflows, coordinating agent teams, or implementing DAG-based task execution.
---

# Agent Orchestration

## Expert Agent

For multi-agent workflow design and team assembly, delegate to:

- **`orchestrator`**: Coordinates complex workflows, assembles agent teams, manages DAG-based execution and inter-agent handoffs.
  - *Location*: `plugins/agent-core/agents/orchestrator.md`

Expert guide for designing and managing multi-agent systems and complex AI workflows.

## 1. Coordination Patterns

- **Sequential**: Linear chains where one agent's output is another's input.
- **Parallel**: Independent tasks executed concurrently using `asyncio` or worker pools.
- **DAG-based**: Complex workflows with explicit task dependencies and data flow.
- **Broadcast**: One-to-many communication for status updates or global state changes.

## 2. Team Assembly & Task Allocation

- **Capability Matching**: Dynamically select agents based on their defined skills and specializations.
- **Workload Management**: Balance tasks across available agents to prevent bottlenecks and maximize throughput.
- **Fault Tolerance**: Implement retries with exponential backoff and fallback agents for critical tasks.

## 3. Communication Protocols

- **Request/Response**: Synchronous inter-agent queries with unique correlation IDs.
- **Message Broker**: Use a centralized broker for loose coupling and reliable message delivery.
- **Context Management**: Ensure relevant state and history are propagated across agent boundaries.

## 4. Orchestration Checklist

- [ ] **Dependencies**: Is the task graph (DAG) correctly defined?
- [ ] **Isolation**: Are agents truly decoupled with clear interfaces?
- [ ] **Concurrency**: Are independent tasks fanned out to maximize performance?
- [ ] **Recovery**: Is there a plan for handling individual agent failures or timeouts?
- [ ] **Monitoring**: Are agent performance and workload metrics being tracked?

## Related Skills

- `multi-agent-coordination` -- Production implementation patterns for DAG workflows, team assembly, and inter-agent messaging
- `agent-performance-optimization` -- Monitoring, caching, and load balancing for orchestrated agent systems
- `mcp-integration` -- Tool coordination across MCP servers used in multi-agent workflows
