---
description: Coordinate multiple AI agents through sophisticated workflow orchestration, intelligent task allocation, and distributed system management. Use this skill when designing multi-agent workflows with complex task dependencies (DAG-based execution), building agent team assembly systems with capability matching, implementing inter-agent communication protocols (message brokers, request/response patterns, broadcast messaging), orchestrating parallel agent execution with synchronization points, designing fault-tolerant multi-agent systems with retry logic and graceful degradation, managing distributed agent teams for large-scale projects, implementing agent selection strategies based on capabilities and workload, handling agent coordination for microservices or multi-domain architectures, or building production-ready agent orchestration systems. This skill is essential when coordinating specialized agents across different technical domains (ML + infrastructure + frontend + backend), when implementing workflow engines for multi-stage agent pipelines, when building distributed agent architectures that require inter-agent messaging, when optimizing task allocation strategies to balance agent workload, or when scaling agent teams dynamically based on project complexity.
---

# Multi-Agent Coordination and Workflow Management

## When to use this skill

- When coordinating multiple specialized AI agents across different technical domains (backend, frontend, ML, DevOps, testing)
- When designing DAG-based workflow orchestration systems with task dependencies and execution ordering
- When implementing agent team assembly logic with capability matching and skill-based selection
- When building inter-agent communication protocols using message brokers and event-driven architecture
- When orchestrating parallel agent execution with synchronization points and dependency management
- When implementing fault-tolerant multi-agent systems with automatic retries and error recovery
- When managing distributed workflows that span multiple agents and technical areas
- When building agent selection strategies based on agent capabilities, performance history, and workload
- When implementing request/response messaging patterns between agents
- When designing broadcast communication for multi-agent coordination and status updates
- When creating workflow engines that execute multi-stage pipelines with agent collaboration
- When handling complex task decomposition and distributing subtasks to specialized agents
- When implementing graceful degradation when agent instances fail or become unavailable
- When tracking agent workload and availability for optimal task distribution
- When building production multi-agent orchestration systems for enterprise applications
- When writing Python code for multi-agent coordination, workflow engines, or distributed agent systems
- When working on projects requiring 5+ specialized agents coordinating across domains
- When designing microservices architectures where agents represent different services

## Overview

This skill provides production-ready patterns for multi-agent coordination, including task decomposition, agent selection strategies, communication protocols, and fault-tolerant execution.

## Core Topics

### 1. Agent Team Assembly and Selection

#### Agent Capability Matching

```python
# agent_orchestration/team_builder.py
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import json

class AgentCapability(Enum):
    """Agent capability categories."""
    BACKEND_DEV = "backend-development"
    FRONTEND_DEV = "frontend-development"
    ML_ENGINEERING = "ml-engineering"
    DEVOPS = "devops"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    DATABASE = "database"

@dataclass
class Agent:
    """Agent definition."""
    name: str
    capabilities: Set[AgentCapability]
    specializations: List[str]
    availability: float  # 0.0 to 1.0
    performance_score: float  # Historical performance
    max_concurrent_tasks: int

@dataclass
class Task:
    """Task definition."""
    id: str
    required_capabilities: Set[AgentCapability]
    required_specializations: List[str]
    estimated_effort: int  # Story points
    priority: int  # 1-5
    dependencies: List[str]  # Task IDs

class AgentRegistry:
    """Registry of available agents."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self._load_agents()

    def _load_agents(self):
        """Load agent definitions."""
        # Example agents
        self.agents = {
            "ml-engineer": Agent(
                name="ml-engineer",
                capabilities={
                    AgentCapability.ML_ENGINEERING,
                    AgentCapability.BACKEND_DEV
                },
                specializations=["pytorch", "tensorflow", "model-deployment"],
                availability=0.8,
                performance_score=0.92,
                max_concurrent_tasks=3
            ),
            "fullstack-developer": Agent(
                name="fullstack-developer",
                capabilities={
                    AgentCapability.FRONTEND_DEV,
                    AgentCapability.BACKEND_DEV,
                    AgentCapability.DATABASE
                },
                specializations=["react", "fastapi", "postgresql"],
                availability=0.9,
                performance_score=0.88,
                max_concurrent_tasks=4
            ),
            "devops-engineer": Agent(
                name="devops-engineer",
                capabilities={
                    AgentCapability.DEVOPS,
                    AgentCapability.SECURITY
                },
                specializations=["kubernetes", "terraform", "ci-cd"],
                availability=0.7,
                performance_score=0.95,
                max_concurrent_tasks=2
            )
        }

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        return self.agents.get(name)

    def find_capable_agents(
        self,
        required_capabilities: Set[AgentCapability],
        required_specializations: List[str] = None
    ) -> List[Agent]:
        """Find agents matching requirements."""
        capable = []

        for agent in self.agents.values():
            # Check capabilities
            if not required_capabilities.issubset(agent.capabilities):
                continue

            # Check specializations
            if required_specializations:
                if not all(s in agent.specializations for s in required_specializations):
                    continue

            capable.append(agent)

        # Sort by performance and availability
        capable.sort(
            key=lambda a: (a.performance_score * a.availability),
            reverse=True
        )

        return capable

class TeamBuilder:
    """Build optimal agent teams for tasks."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def build_team(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """
        Build optimal team for tasks.

        Returns:
            Dict mapping task IDs to assigned agent names
        """
        assignments: Dict[str, List[str]] = {}
        agent_workload: Dict[str, int] = {}

        # Sort tasks by priority and dependencies
        sorted_tasks = self._topological_sort(tasks)

        for task in sorted_tasks:
            # Find capable agents
            capable_agents = self.registry.find_capable_agents(
                task.required_capabilities,
                task.required_specializations
            )

            if not capable_agents:
                print(f"Warning: No agents found for task {task.id}")
                continue

            # Select best available agent
            selected = None
            for agent in capable_agents:
                current_load = agent_workload.get(agent.name, 0)
                if current_load < agent.max_concurrent_tasks:
                    selected = agent
                    break

            if selected:
                assignments[task.id] = [selected.name]
                agent_workload[selected.name] = agent_workload.get(selected.name, 0) + 1
            else:
                print(f"Warning: No available agents for task {task.id}")

        return assignments

    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by dependencies and priority."""
        # Simple implementation - can be enhanced with proper DAG sorting
        return sorted(tasks, key=lambda t: (len(t.dependencies), -t.priority))

# Usage example
def main():
    registry = AgentRegistry()
    builder = TeamBuilder(registry)

    tasks = [
        Task(
            id="task-1",
            required_capabilities={AgentCapability.ML_ENGINEERING},
            required_specializations=["pytorch"],
            estimated_effort=5,
            priority=5,
            dependencies=[]
        ),
        Task(
            id="task-2",
            required_capabilities={
                AgentCapability.FRONTEND_DEV,
                AgentCapability.BACKEND_DEV
            },
            required_specializations=["react", "fastapi"],
            estimated_effort=8,
            priority=4,
            dependencies=["task-1"]
        ),
        Task(
            id="task-3",
            required_capabilities={AgentCapability.DEVOPS},
            required_specializations=["kubernetes"],
            estimated_effort=3,
            priority=3,
            dependencies=["task-2"]
        )
    ]

    assignments = builder.build_team(tasks)

    print("Task Assignments:")
    for task_id, agents in assignments.items():
        print(f"  {task_id}: {', '.join(agents)}")

if __name__ == "__main__":
    main()
```

### 2. Workflow Orchestration

#### DAG-Based Workflow Engine

```python
# agent_orchestration/workflow.py
from typing import Dict, List, Set, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowTask:
    """Workflow task definition."""
    id: str
    name: str
    agent: str
    action: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300  # seconds
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class WorkflowEngine:
    """Async workflow orchestration engine."""

    def __init__(self):
        self.tasks: Dict[str, WorkflowTask] = {}
        self.results: Dict[str, Any] = {}
        self.running_tasks: Set[str] = set()

    def add_task(self, task: WorkflowTask) -> None:
        """Add task to workflow."""
        self.tasks[task.id] = task

    def build_dag(self) -> Dict[str, Set[str]]:
        """Build dependency DAG."""
        dag: Dict[str, Set[str]] = {task_id: set() for task_id in self.tasks}

        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"Unknown dependency: {dep_id}")
                dag[task_id].add(dep_id)

        # Check for cycles
        if self._has_cycle(dag):
            raise ValueError("Workflow contains circular dependencies")

        return dag

    def _has_cycle(self, dag: Dict[str, Set[str]]) -> bool:
        """Detect cycles in DAG."""
        visited = set()
        rec_stack = set()

        def visit(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in dag.get(node, set()):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in dag:
            if node not in visited:
                if visit(node):
                    return True

        return False

    async def execute_task(self, task: WorkflowTask) -> None:
        """Execute single task with retries."""
        for attempt in range(task.retry_count):
            try:
                task.status = TaskStatus.RUNNING
                task.start_time = datetime.now()
                self.running_tasks.add(task.id)

                print(f"[{task.agent}] Executing task: {task.name} (attempt {attempt + 1})")

                # Execute with timeout
                task.result = await asyncio.wait_for(
                    task.action(),
                    timeout=task.timeout
                )

                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()
                self.results[task.id] = task.result

                print(f"[{task.agent}] ✓ Completed: {task.name}")
                return

            except asyncio.TimeoutError:
                task.error = f"Timeout after {task.timeout}s"
                print(f"[{task.agent}] ✗ Timeout: {task.name}")

            except Exception as e:
                task.error = str(e)
                print(f"[{task.agent}] ✗ Error: {task.name} - {e}")

            finally:
                self.running_tasks.discard(task.id)

            if attempt < task.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        task.status = TaskStatus.FAILED
        task.end_time = datetime.now()

    async def execute_workflow(self) -> Dict[str, Any]:
        """Execute entire workflow."""
        dag = self.build_dag()

        # Find tasks with no dependencies
        ready_tasks = [
            task_id for task_id, deps in dag.items()
            if not deps
        ]

        pending_tasks = set(self.tasks.keys()) - set(ready_tasks)

        # Execute tasks
        while ready_tasks or self.running_tasks:
            if ready_tasks:
                # Start ready tasks
                tasks_to_run = [self.tasks[task_id] for task_id in ready_tasks]
                await asyncio.gather(
                    *[self.execute_task(task) for task in tasks_to_run]
                )
                ready_tasks.clear()

            # Check for newly ready tasks
            for task_id in list(pending_tasks):
                task = self.tasks[task_id]

                # Check if all dependencies completed
                deps_completed = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )

                if deps_completed:
                    ready_tasks.append(task_id)
                    pending_tasks.remove(task_id)
                else:
                    # Check if any dependency failed
                    deps_failed = any(
                        self.tasks[dep_id].status == TaskStatus.FAILED
                        for dep_id in task.dependencies
                    )

                    if deps_failed:
                        task.status = TaskStatus.SKIPPED
                        task.error = "Dependency failed"
                        pending_tasks.remove(task_id)
                        print(f"[{task.agent}] ○ Skipped: {task.name}")

        return self.results

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary."""
        total_tasks = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)

        total_duration = sum(
            (t.end_time - t.start_time).total_seconds()
            for t in self.tasks.values()
            if t.start_time and t.end_time
        )

        return {
            'total_tasks': total_tasks,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': completed / total_tasks if total_tasks > 0 else 0,
            'total_duration': total_duration,
            'failed_tasks': [
                {'id': t.id, 'name': t.name, 'error': t.error}
                for t in self.tasks.values()
                if t.status == TaskStatus.FAILED
            ]
        }

# Usage example
async def main():
    workflow = WorkflowEngine()

    # Define tasks
    async def task1_action():
        await asyncio.sleep(1)
        return {"data": "task1 result"}

    async def task2_action():
        await asyncio.sleep(1)
        return {"data": "task2 result"}

    async def task3_action():
        await asyncio.sleep(1)
        return {"data": "task3 result"}

    workflow.add_task(WorkflowTask(
        id="task-1",
        name="Data preprocessing",
        agent="data-engineer",
        action=task1_action,
        dependencies=[]
    ))

    workflow.add_task(WorkflowTask(
        id="task-2",
        name="Model training",
        agent="ml-engineer",
        action=task2_action,
        dependencies=["task-1"]
    ))

    workflow.add_task(WorkflowTask(
        id="task-3",
        name="Model deployment",
        agent="devops-engineer",
        action=task3_action,
        dependencies=["task-2"]
    ))

    # Execute workflow
    results = await workflow.execute_workflow()

    # Print summary
    summary = workflow.get_execution_summary()
    print(f"\nWorkflow Summary:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total duration: {summary['total_duration']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Inter-Agent Communication

#### Message-Based Communication Protocol

```python
# agent_orchestration/messaging.py
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import json
from uuid import uuid4

class MessageType(Enum):
    """Message types."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    QUERY = "query"
    BROADCAST = "broadcast"

@dataclass
class Message:
    """Inter-agent message."""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: MessageType = MessageType.TASK_REQUEST
    sender: str = ""
    receiver: Optional[str] = None  # None for broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For request/response pairing

class MessageBroker:
    """Message broker for inter-agent communication."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.pending_responses: Dict[str, asyncio.Future] = {}

    def subscribe(self, agent_name: str, handler: Callable) -> None:
        """Subscribe agent to messages."""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(handler)

    async def publish(self, message: Message) -> None:
        """Publish message to subscribers."""
        self.message_history.append(message)

        # Targeted message
        if message.receiver:
            if message.receiver in self.subscribers:
                for handler in self.subscribers[message.receiver]:
                    await handler(message)
        # Broadcast
        else:
            for agent_name, handlers in self.subscribers.items():
                if agent_name != message.sender:  # Don't send to self
                    for handler in handlers:
                        await handler(message)

        # Handle response pairing
        if message.type == MessageType.TASK_RESPONSE and message.correlation_id:
            if message.correlation_id in self.pending_responses:
                self.pending_responses[message.correlation_id].set_result(message)

    async def request(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Message:
        """Send request and wait for response."""
        request_msg = Message(
            type=MessageType.TASK_REQUEST,
            sender=sender,
            receiver=receiver,
            payload=payload
        )

        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[request_msg.id] = response_future

        # Send request
        await self.publish(request_msg)

        try:
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f"No response from {receiver} within {timeout}s")
        finally:
            self.pending_responses.pop(request_msg.id, None)

    async def respond(
        self,
        request_msg: Message,
        sender: str,
        payload: Dict[str, Any]
    ) -> None:
        """Send response to request."""
        response_msg = Message(
            type=MessageType.TASK_RESPONSE,
            sender=sender,
            receiver=request_msg.sender,
            payload=payload,
            correlation_id=request_msg.id
        )

        await self.publish(response_msg)

    def get_message_history(
        self,
        agent_name: Optional[str] = None,
        message_type: Optional[MessageType] = None
    ) -> List[Message]:
        """Get filtered message history."""
        messages = self.message_history

        if agent_name:
            messages = [
                m for m in messages
                if m.sender == agent_name or m.receiver == agent_name
            ]

        if message_type:
            messages = [m for m in messages if m.type == message_type]

        return messages

# Example agent implementation
class Agent:
    """Example agent with messaging."""

    def __init__(self, name: str, broker: MessageBroker):
        self.name = name
        self.broker = broker
        self.broker.subscribe(name, self.handle_message)

    async def handle_message(self, message: Message) -> None:
        """Handle incoming message."""
        print(f"[{self.name}] Received {message.type.value} from {message.sender}")

        if message.type == MessageType.TASK_REQUEST:
            # Process request
            result = await self.process_task(message.payload)

            # Send response
            await self.broker.respond(
                message,
                self.name,
                {'result': result}
            )

        elif message.type == MessageType.BROADCAST:
            # Handle broadcast
            print(f"[{self.name}] Broadcast: {message.payload}")

    async def process_task(self, task_data: Dict[str, Any]) -> Any:
        """Process task."""
        await asyncio.sleep(1)  # Simulate work
        return {'status': 'completed', 'data': task_data}

    async def request_help(self, target_agent: str, task: Dict[str, Any]) -> Message:
        """Request help from another agent."""
        print(f"[{self.name}] Requesting help from {target_agent}")
        response = await self.broker.request(
            self.name,
            target_agent,
            task
        )
        return response

    async def broadcast_status(self, status: Dict[str, Any]) -> None:
        """Broadcast status update."""
        message = Message(
            type=MessageType.BROADCAST,
            sender=self.name,
            payload=status
        )
        await self.broker.publish(message)

# Usage example
async def main():
    broker = MessageBroker()

    # Create agents
    ml_agent = Agent("ml-engineer", broker)
    devops_agent = Agent("devops-engineer", broker)
    data_agent = Agent("data-engineer", broker)

    # ML agent requests help from data agent
    response = await ml_agent.request_help(
        "data-engineer",
        {'task': 'preprocess_data', 'dataset': 'train.csv'}
    )

    print(f"\n[ml-engineer] Got response: {response.payload}")

    # Broadcast status
    await ml_agent.broadcast_status({'status': 'training_complete', 'accuracy': 0.95})

    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### Agent Coordination
1. Use capability-based agent selection
2. Implement proper dependency management
3. Handle agent failures gracefully
4. Monitor agent workload and availability
5. Implement retry logic with exponential backoff

### Workflow Design
1. Create DAGs to visualize dependencies
2. Enable parallel execution where possible
3. Implement checkpointing for long workflows
4. Use timeouts to prevent hanging
5. Provide clear error messages

### Communication
1. Use message brokers for loose coupling
2. Implement request/response patterns
3. Support broadcast for coordination
4. Track message history for debugging
5. Use correlation IDs for request tracking

## Quick Reference

```python
# Team assembly
registry = AgentRegistry()
builder = TeamBuilder(registry)
assignments = builder.build_team(tasks)

# Workflow execution
workflow = WorkflowEngine()
workflow.add_task(task)
results = await workflow.execute_workflow()

# Messaging
broker = MessageBroker()
agent = Agent("agent-name", broker)
response = await agent.request_help("other-agent", task)
```
