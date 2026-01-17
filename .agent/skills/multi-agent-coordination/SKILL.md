---
name: multi-agent-coordination
version: "1.0.7"
maturity: "5-Expert"
specialization: Multi-Agent Workflow Orchestration
description: Coordinate multiple AI agents through workflow orchestration, task allocation, and distributed systems. Use when designing DAG-based workflows with task dependencies, building agent team assembly with capability matching, implementing inter-agent communication protocols, orchestrating parallel execution with synchronization, designing fault-tolerant systems with retry logic, or managing distributed agent teams.
---

# Multi-Agent Coordination

Production-ready patterns for multi-agent orchestration and workflow management.

---

## Coordination Patterns

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| Sequential | Linear task chains | Low |
| Parallel | Independent tasks | Medium |
| DAG-based | Complex dependencies | High |
| Broadcast | Status updates | Low |
| Request/Response | Inter-agent queries | Medium |

---

## Agent Team Assembly

```python
from dataclasses import dataclass
from enum import Enum
from typing import Set, List, Dict

class AgentCapability(Enum):
    BACKEND_DEV = "backend-development"
    FRONTEND_DEV = "frontend-development"
    ML_ENGINEERING = "ml-engineering"
    DEVOPS = "devops"
    TESTING = "testing"

@dataclass
class Agent:
    name: str
    capabilities: Set[AgentCapability]
    specializations: List[str]
    availability: float  # 0.0 to 1.0
    performance_score: float
    max_concurrent_tasks: int

@dataclass
class Task:
    id: str
    required_capabilities: Set[AgentCapability]
    required_specializations: List[str]
    priority: int
    dependencies: List[str]

class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def find_capable_agents(
        self,
        required_capabilities: Set[AgentCapability],
        required_specializations: List[str] = None
    ) -> List[Agent]:
        capable = [
            agent for agent in self.agents.values()
            if required_capabilities.issubset(agent.capabilities)
            and (not required_specializations or
                 all(s in agent.specializations for s in required_specializations))
        ]
        return sorted(capable, key=lambda a: a.performance_score * a.availability, reverse=True)

class TeamBuilder:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def build_team(self, tasks: List[Task]) -> Dict[str, List[str]]:
        assignments: Dict[str, List[str]] = {}
        workload: Dict[str, int] = {}

        for task in sorted(tasks, key=lambda t: (len(t.dependencies), -t.priority)):
            capable = self.registry.find_capable_agents(
                task.required_capabilities, task.required_specializations
            )
            for agent in capable:
                if workload.get(agent.name, 0) < agent.max_concurrent_tasks:
                    assignments[task.id] = [agent.name]
                    workload[agent.name] = workload.get(agent.name, 0) + 1
                    break

        return assignments
```

---

## DAG Workflow Engine

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Set, Any, Callable, Optional
import asyncio
from datetime import datetime

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowTask:
    id: str
    name: str
    agent: str
    action: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None

class WorkflowEngine:
    def __init__(self):
        self.tasks: Dict[str, WorkflowTask] = {}
        self.results: Dict[str, Any] = {}

    def add_task(self, task: WorkflowTask) -> None:
        self.tasks[task.id] = task

    async def execute_task(self, task: WorkflowTask) -> None:
        for attempt in range(task.retry_count):
            try:
                task.status = TaskStatus.RUNNING
                task.result = await asyncio.wait_for(task.action(), timeout=task.timeout)
                task.status = TaskStatus.COMPLETED
                self.results[task.id] = task.result
                return
            except Exception as e:
                if attempt < task.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        task.status = TaskStatus.FAILED

    async def execute_workflow(self) -> Dict[str, Any]:
        # Build dependency graph
        pending = set(self.tasks.keys())
        ready = [tid for tid, t in self.tasks.items() if not t.dependencies]
        pending -= set(ready)

        while ready or any(t.status == TaskStatus.RUNNING for t in self.tasks.values()):
            if ready:
                await asyncio.gather(*[self.execute_task(self.tasks[tid]) for tid in ready])
                ready.clear()

            # Find newly ready tasks
            for tid in list(pending):
                task = self.tasks[tid]
                if all(self.tasks[d].status == TaskStatus.COMPLETED for d in task.dependencies):
                    ready.append(tid)
                    pending.remove(tid)
                elif any(self.tasks[d].status == TaskStatus.FAILED for d in task.dependencies):
                    task.status = TaskStatus.SKIPPED
                    pending.remove(tid)

        return self.results
```

---

## Inter-Agent Messaging

```python
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import asyncio

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    BROADCAST = "broadcast"

@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid4()))
    type: MessageType = MessageType.TASK_REQUEST
    sender: str = ""
    receiver: Optional[str] = None  # None for broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

class MessageBroker:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}

    def subscribe(self, agent_name: str, handler: Callable) -> None:
        self.subscribers.setdefault(agent_name, []).append(handler)

    async def publish(self, message: Message) -> None:
        if message.receiver:
            for handler in self.subscribers.get(message.receiver, []):
                await handler(message)
        else:  # Broadcast
            for name, handlers in self.subscribers.items():
                if name != message.sender:
                    for handler in handlers:
                        await handler(message)

        # Handle response pairing
        if message.type == MessageType.TASK_RESPONSE and message.correlation_id:
            if message.correlation_id in self.pending_responses:
                self.pending_responses[message.correlation_id].set_result(message)

    async def request(self, sender: str, receiver: str, payload: Dict, timeout: float = 30) -> Message:
        request_msg = Message(type=MessageType.TASK_REQUEST, sender=sender, receiver=receiver, payload=payload)
        response_future = asyncio.Future()
        self.pending_responses[request_msg.id] = response_future

        await self.publish(request_msg)

        try:
            return await asyncio.wait_for(response_future, timeout=timeout)
        finally:
            self.pending_responses.pop(request_msg.id, None)
```

---

## Agent Implementation

```python
class Agent:
    def __init__(self, name: str, broker: MessageBroker):
        self.name = name
        self.broker = broker
        broker.subscribe(name, self.handle_message)

    async def handle_message(self, message: Message) -> None:
        if message.type == MessageType.TASK_REQUEST:
            result = await self.process_task(message.payload)
            response = Message(
                type=MessageType.TASK_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                payload={'result': result},
                correlation_id=message.id
            )
            await self.broker.publish(response)

    async def process_task(self, task_data: Dict) -> Any:
        # Agent-specific task processing
        return {'status': 'completed', 'data': task_data}

    async def request_help(self, target: str, task: Dict) -> Message:
        return await self.broker.request(self.name, target, task)
```

---

## Best Practices

| Area | Practice |
|------|----------|
| Team Assembly | Capability-based agent selection |
| Workflow | DAG for dependency management |
| Execution | Parallel where possible |
| Failures | Retry with exponential backoff |
| Communication | Message broker for loose coupling |
| Monitoring | Track workload and availability |

---

## Checklist

- [ ] Define agent capabilities and specializations
- [ ] Build dependency graph (DAG) for tasks
- [ ] Implement retry logic with backoff
- [ ] Use message broker for inter-agent communication
- [ ] Handle agent failures gracefully
- [ ] Monitor agent workload and performance
- [ ] Implement request/response correlation
- [ ] Support broadcast for status updates

---

**Version**: 1.0.5
