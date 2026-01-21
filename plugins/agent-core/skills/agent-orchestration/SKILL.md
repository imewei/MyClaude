---
name: agent-orchestration
version: "2.1.0"
description: Coordinate multiple AI agents through workflow orchestration, task allocation, and inter-agent communication protocols. Masters DAG-based workflows and team assembly.
---

# Agent Orchestration

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
